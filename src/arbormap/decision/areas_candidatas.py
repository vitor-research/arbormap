"""
Áreas candidatas para o mapa interativo, com o efeito individual de arborizar cada uma

Base: composição da estação seca (city_layers) e modelo do fold em que a cidade é validação.
Plantio parcial: em cada pixel, só a fração plantável da classe vira copa; as entradas do
modelo (NDVI, NDBI, NDWI) são misturadas com a assinatura de vegetação densa observada na cidade
na proporção dessa fração. Nenhum canal de temperatura é editado.

Tipos de área:
  praca    : manchas de praças/parques (OpenStreetMap) ainda pouco arborizadas
  calcada  : corredores de calçada/canteiro, agrupados pelo escore multicritério
  livre    : lotes e outros espaços livres (em geral privados; "a verificar")

Métricas por área (resfriamento local relativo, copas formadas):
  resfriamento_c       média de ΔT nos pixels da área, ponderada pela fração plantada
  area_05c_ha          área (plantio + vizinhança) com ΔT ≥ 0,5 °C
  resfriamento_int     soma de ΔT positivo × área (°C·ha): "quanto frescor" a área produz
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from arbormap.decision.city_layers import load_model
from arbormap.decision.simulate_urbanheatnet import (
    MODEL_CH, PIXEL_HA, advected_cooling, build_corridors, cooling_potential, multicriteria_score,
)
from arbormap.decision.export_map import _mask_to_polygons
from arbormap.model.hybrid_pipeline import CH_DEM, CH_LST, CH_NDBI, CH_NDVI, CH_NDWI, LST_CLIP_MAX, LST_CLIP_MIN

NDVI_MAX_PLANTIO = 0.3     # acima disso o pixel já é vegetado
NDVI_VEG_DENSA = 0.5


def input_from_layers(L, stats):
    mean, std = np.asarray(stats[0], np.float32), np.asarray(stats[1], np.float32)
    raw = np.stack([L["ndvi"], L["ndwi"], L["ndbi"], L["dem"], L["lst_obs"]]).astype(np.float32)
    med = np.nanmedian(raw.reshape(5, -1), axis=1)
    raw = np.where(np.isfinite(raw), raw, med[:, None, None])
    raw[CH_LST] = np.clip(raw[CH_LST], LST_CLIP_MIN, LST_CLIP_MAX)
    raw[CH_NDVI] = np.clip(raw[CH_NDVI], -1, 1)
    raw[CH_NDWI] = np.clip(raw[CH_NDWI], -1, 1)
    norm = (raw - mean[:, None, None]) / (std[:, None, None] + 1e-6)
    x = np.stack([norm[CH_NDVI], norm[CH_NDBI], norm[CH_DEM], norm[CH_NDWI]])
    return torch.from_numpy(x).unsqueeze(0).float()


class Simulador:
    def __init__(self, L, meta, ckpt, device="cpu"):
        self.L = L
        self.model, st = load_model(ckpt, device)
        self.stats = (np.asarray(st["mean"]), np.asarray(st["std"]))
        self.mu, self.sd = meta["lst_media_composicao_c"], meta["lst_dp_composicao_c"]
        self.x = input_from_layers(L, self.stats)
        self.valid = np.isfinite(L["lst_pred"])
        self.base = self.predict(self.x)
        veg = self.valid & (L["ndvi"] >= NDVI_VEG_DENSA)
        self.sig = {c: float(np.nanmedian(L[k][veg])) for c, k in ((CH_NDVI, "ndvi"), (CH_NDBI, "ndbi"), (CH_NDWI, "ndwi"))}
        self.n_veg = int(veg.sum())

    def predict(self, x):
        with torch.no_grad():
            return (self.model(x)[0, 0].numpy() * self.sd + self.mu).astype(np.float32)

    def plant(self, frac):
        """Mistura as entradas com a assinatura de vegetação na proporção `frac` (0–1) por pixel."""
        x = self.x.clone()
        f = torch.from_numpy(np.clip(frac, 0, 1).astype(np.float32))
        mean, std = self.stats
        for c, v in self.sig.items():
            k = MODEL_CH[c]
            tgt = float((v - mean[c]) / (std[c] + 1e-6))
            x[0, k] = x[0, k] + f * (tgt - x[0, k])
        return x

    def effect(self, frac):
        after = self.predict(self.plant(frac))
        cool = advected_cooling(self.base, after)
        m = frac > 0
        w = frac[m]
        return cool, {
            "resfriamento_c": float((cool[m] * w).sum() / max(w.sum(), 1e-9)),
            "area_05c_ha": float(((cool >= 0.5) & self.valid).sum() * PIXEL_HA),
            "resfriamento_int": float(np.clip(cool[self.valid], 0, None).sum() * PIXEL_HA),
        }


def _components(mask, min_px=3):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return [lab == i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= min_px]


def _equipamentos(osm_json):
    els = json.loads(Path(osm_json).read_text(encoding="utf-8"))["elements"]
    out = []
    for e in els:
        t = e.get("tags", {})
        lat = e.get("lat", e.get("center", {}).get("lat"))
        lon = e.get("lon", e.get("center", {}).get("lon"))
        if lat is None:
            continue
        a, hc = t.get("amenity", ""), t.get("healthcare", "")
        if a == "kindergarten":
            cat = "creche"
        elif a in ("school", "college", "university"):
            cat = "escola"
        elif a in ("clinic", "hospital") or hc in ("centre", "clinic", "hospital"):
            cat = "saude"   # farmácias, consultórios e laboratórios ficam fora
        else:
            continue
        out.append({"categoria": cat, "nome": t.get("name") or "(sem nome no OSM)", "lat": lat, "lon": lon})
    return out


def _pix_to_lonlat(ys, xs, bounds, shape):
    w, s, e, n = bounds
    h, wd = shape
    return w + (xs + 0.5) * (e - w) / wd, n - (ys + 0.5) * (n - s) / h


def gerar(L, meta, ckpt, fr, bairro_ids, bairros_nomes, bounds, osm_dir, max_por_tipo=None, log=print):
    max_por_tipo = max_por_tipo or {"praca": 200, "calcada": 160, "livre": 100}
    sim = Simulador(L, meta, ckpt)
    log(f"assinatura de vegetação (n={sim.n_veg}): {sim.sig}")
    valid = sim.valid & (L["artefato_fixo"] < 0.5)
    ndvi = L["ndvi"]
    pot = cooling_potential(sim.model, sim.x, torch.from_numpy(valid[None, None].astype(np.float32)), sim.sig,
                            sim.stats, sim.sd, sim.mu)
    candidatos = []

    # praças/parques
    m_praca = valid & (fr["praca"] > 0.05) & (ndvi < NDVI_MAX_PLANTIO + 0.1)
    for comp in sorted(_components(m_praca, 2), key=lambda c: -fr["praca"][c].sum())[:max_por_tipo["praca"]]:
        candidatos.append(("praca", comp, fr["praca"] * comp))
    # calçadas e lotes: corredores pelo escore multicritério
    for tipo, chave, alvo in (("calcada", "calcada", 2500), ("livre", "outros", 1500)):
        plant = valid & (fr[chave] > 0.08) & (ndvi < NDVI_MAX_PLANTIO)
        score = multicriteria_score(pot, sim.base, plant, {"peso_potencial": 0.6, "peso_urgencia": 0.4})
        cors, _ = build_corridors(score, plant, {"area_alvo_candidatos_px": alvo, "min_px_corredor": 4,
                                                 "area_candidata_min_px": alvo // 2})
        for c in cors[:max_por_tipo[tipo]]:
            candidatos.append((tipo, c["mask"], fr[chave] * c["mask"]))
    log(f"{len(candidatos)} áreas candidatas; simulando uma a uma...")

    eq = _equipamentos(Path(osm_dir) / "equipamentos.json")
    feats = []
    for i, (tipo, mask, frac) in enumerate(candidatos):
        _, ef = sim.effect(frac)
        ys, xs = np.where(mask)
        lon, lat = _pix_to_lonlat(ys.mean(), xs.mean(), bounds, mask.shape)
        ids = bairro_ids[mask]
        ids = ids[ids > 0]
        bairro = bairros_nomes.get(int(np.bincount(ids).argmax())) if ids.size else None
        props = {"id": i + 1, "tipo": tipo, "bairro": bairro, "centro": [round(lon, 6), round(lat, 6)],
                 "area_plantavel_ha": round(float(frac.sum() * PIXEL_HA), 3),
                 "n_pixels": int(mask.sum()),
                 "lst_media_c": round(float(sim.base[mask].mean()), 2),
                 "incerteza_c": round(float(L["incerteza"][mask].mean()), 2),
                 "potencial": round(float(pot[mask].mean()), 4),
                 **{k: round(v, 3) for k, v in ef.items()}}
        polys = _mask_to_polygons(mask, bounds)
        geom = {"type": "Polygon", "coordinates": polys[0]} if len(polys) == 1 else {"type": "MultiPolygon", "coordinates": polys}
        feats.append({"type": "Feature", "geometry": geom, "properties": props})
        if (i + 1) % 25 == 0:
            log(f"   {i + 1}/{len(candidatos)}")
    vincular_equipamentos(feats, eq)
    return sim, candidatos, feats, eq


def vincular_equipamentos(feats, eq):
    """Grava em cada área o equipamento (escola, creche, saúde) mais próximo do seu centro."""
    if not eq:
        return
    xy = np.array([[e["lon"], e["lat"]] for e in eq])
    for f in feats:
        p = f["properties"]
        lon, lat = p["centro"]
        d = np.hypot((xy[:, 0] - lon) * np.cos(np.deg2rad(lat)), xy[:, 1] - lat) * 111_320
        j = int(d.argmin())
        p["equipamento_proximo"] = {"nome": eq[j]["nome"], "categoria": eq[j]["categoria"], "dist_m": round(float(d[j]))}


def teste_aditividade(sim, candidatos, feats, n=20):
    """Compara o efeito de n áreas plantadas juntas com a soma dos efeitos individuais."""
    order = sorted(range(len(feats)), key=lambda i: -feats[i]["properties"]["resfriamento_int"])[:n]
    frac = np.zeros_like(candidatos[0][2])
    for i in order:
        frac = np.maximum(frac, candidatos[i][2])
    _, ef = sim.effect(frac)
    soma = sum(feats[i]["properties"]["resfriamento_int"] for i in order)
    return {"n_areas": n, "integral_conjunto": round(ef["resfriamento_int"], 2), "soma_individuais": round(soma, 2),
            "razao_conjunto_soma": round(ef["resfriamento_int"] / max(soma, 1e-9), 3)}
