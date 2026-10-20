"""
Validações do modelo com o mundo real (Teresina), sem treino novo

5.1 Experimento natural: entre duas composições secas (anos iniciais × finais), pixels que ganharam
    ou perderam vegetação. A mudança observada da anomalia de LST é comparada com a mudança que o
    modelo prevê só a partir das mudanças em NDVI, NDBI, NDWI e DEM.
5.2 Parques existentes: diferença de temperatura entre o entorno (90 a 240 m) e o interior de
    parques/praças já arborizados, observada × prevista.
5.5 Ablação: cada canal de entrada é substituído pela sua média na cidade (perde a informação
    espacial) e mede-se a piora do erro e da correlação.

Em todos os testes o modelo é o do fold em que Teresina é validação (não a viu no treino).
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from arbormap.decision.city_layers import CROP, _center_crop, dry_season_composite, load_model, model_input
from arbormap.model.hybrid_pipeline import CH_LST, CH_NDBI, CH_NDVI, CH_NDWI

NOMES_CANAIS = ["NDVI", "NDBI", "DEM", "NDWI"]   # ordem da entrada do modelo


def _crop_band(comp, c):
    y0, x0, hh, ww = _center_crop(comp)
    out = np.full((CROP, CROP), np.nan, np.float32)
    out[:hh, :ww] = comp[c, y0:y0 + hh, x0:x0 + ww]
    return out


def prever_composicao(comp, model, st):
    """Anomalia prevista (°C, na escala da LST observada da composição) e a observada."""
    obs = _crop_band(comp, CH_LST)
    mu, sd = float(np.nanmean(obs)), float(np.nanstd(obs))
    x, valid, _, _ = model_input(comp, (st["mean"], st["std"]))
    with torch.no_grad():
        z = model(x)[0, 0].numpy()
    return np.where(valid, z * sd, np.nan), obs - mu, x, (mu, sd)


def _ols(x, y):
    a, b = np.polyfit(x, y, 1)
    return float(a), float(b)


def experimento_natural(city_dir, model, st, anos_a, anos_b, limiar=0.15):
    ca, _, _, reg_a = dry_season_composite(city_dir, years=anos_a)
    cb, static, _, reg_b = dry_season_composite(city_dir, years=anos_b)
    pa, oa, _, _ = prever_composicao(ca, model, st)
    pb, ob, _, _ = prever_composicao(cb, model, st)
    dndvi = _crop_band(cb, CH_NDVI) - _crop_band(ca, CH_NDVI)
    terra = (_crop_band(ca, CH_NDWI) < 0.1) & (_crop_band(cb, CH_NDWI) < 0.1)
    ok = terra & np.isfinite(oa) & np.isfinite(ob) & np.isfinite(pa) & np.isfinite(pb) & np.isfinite(dndvi)
    d_obs, d_pred = (ob - oa)[ok], (pb - pa)[ok]
    dv = dndvi[ok]
    grupos = {}
    for nome, m in (("ganhou_vegetacao", dv > limiar), ("perdeu_vegetacao", dv < -limiar), ("estavel", np.abs(dv) < 0.05)):
        grupos[nome] = {"n_pixels": int(m.sum()), "dT_observado_c": round(float(d_obs[m].mean()), 2),
                        "dT_previsto_c": round(float(d_pred[m].mean()), 2)}
    mud = np.abs(dv) > limiar
    bins = np.arange(-0.4, 0.401, 0.05)
    curva = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (dv >= lo) & (dv < hi)
        if m.sum() >= 30:
            curva.append({"dndvi": round(float((lo + hi) / 2), 3), "obs": round(float(d_obs[m].mean()), 3),
                          "pred": round(float(d_pred[m].mean()), 3), "n": int(m.sum())})
    so, _ = _ols(dv, d_obs)
    sp, _ = _ols(dv, d_pred)
    return {
        "anos_a": sorted(anos_a), "anos_b": sorted(anos_b),
        "cenas_a": [u["arquivo"] for u in reg_a["usadas"]], "cenas_b": [u["arquivo"] for u in reg_b["usadas"]],
        "limiar_dndvi": limiar, "grupos": grupos,
        "pearson_pixels_que_mudaram": round(float(np.corrcoef(d_obs[mud], d_pred[mud])[0, 1]), 3),
        "mae_pixels_que_mudaram_c": round(float(np.abs(d_obs[mud] - d_pred[mud]).mean()), 2),
        "inclinacao_obs_c_por_0p1_ndvi": round(so / 10, 3), "inclinacao_pred_c_por_0p1_ndvi": round(sp / 10, 3),
        "curva": curva,
    }


def parques(anom_obs, anom_pred, ndvi, ndwi, park_mask_list, nomes, min_px=10, ndvi_min=0.35):
    """Ilha fresca dos parques: entorno (3 a 8 pixels fora) menos interior."""
    todos = np.zeros_like(ndvi, bool)
    for m in park_mask_list:
        todos |= m
    agua = ndwi >= 0.1
    k = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    linhas = []
    for m, nome in zip(park_mask_list, nomes):
        dentro = m & np.isfinite(anom_obs) & np.isfinite(anom_pred) & ~agua
        if dentro.sum() < min_px or np.nanmean(ndvi[dentro]) < ndvi_min:
            continue
        mu8 = m.astype(np.uint8)
        anel = (cv2.dilate(mu8, k(8)) > 0) & ~(cv2.dilate(mu8, k(3)) > 0) & ~todos & ~agua
        anel &= np.isfinite(anom_obs) & np.isfinite(anom_pred)
        if anel.sum() < min_px:
            continue
        linhas.append({"nome": nome, "area_ha": round(float(dentro.sum() * 0.09), 1),
                       "ndvi": round(float(np.nanmean(ndvi[dentro])), 2),
                       "obs_c": round(float(anom_obs[anel].mean() - anom_obs[dentro].mean()), 2),
                       "pred_c": round(float(anom_pred[anel].mean() - anom_pred[dentro].mean()), 2)})
    o = np.array([l["obs_c"] for l in linhas])
    p = np.array([l["pred_c"] for l in linhas])
    return {"n_parques": len(linhas), "obs_media_c": round(float(o.mean()), 2), "pred_media_c": round(float(p.mean()), 2),
            "pearson": round(float(np.corrcoef(o, p)[0, 1]), 3) if len(o) > 2 else None,
            "mae_c": round(float(np.abs(o - p).mean()), 2),
            "parques": sorted(linhas, key=lambda l: -l["area_ha"])}


def ablacao(x, obs_anom, model, sd):
    ok = np.isfinite(obs_anom)
    def avalia(xx):
        with torch.no_grad():
            p = model(xx)[0, 0].numpy() * sd
        return float(np.abs(p - obs_anom)[ok].mean()), float(np.corrcoef(p[ok], obs_anom[ok])[0, 1])
    base_mae, base_r = avalia(x)
    out = {"completo": {"mae_c": round(base_mae, 2), "pearson": round(base_r, 3)}}
    for k, nome in enumerate(NOMES_CANAIS):
        xx = x.clone()
        xx[0, k] = x[0, k].mean()
        mae, r = avalia(xx)
        out[nome] = {"mae_c": round(mae, 2), "pearson": round(r, 3), "piora_mae_c": round(mae - base_mae, 2)}
    return out


def run(city_dir, ckpt, layers_npz, park_masks, park_names, anos_a=(2018, 2019, 2020, 2021), anos_b=(2023, 2024), log=print):
    model, st = load_model(ckpt)
    log("5.1 experimento natural...")
    nat = experimento_natural(city_dir, model, st, set(anos_a), set(anos_b))
    log(json.dumps({k: v for k, v in nat.items() if k != "curva"}, ensure_ascii=False))
    L = dict(np.load(layers_npz))
    comp, _, _, _ = dry_season_composite(city_dir)
    pred, obs, x, (mu, sd) = prever_composicao(comp, model, st)
    log("5.2 parques...")
    par = parques(obs, pred, L["ndvi"], L["ndwi"], park_masks, park_names)
    log(json.dumps({k: v for k, v in par.items() if k != "parques"}, ensure_ascii=False))
    log("5.5 ablação...")
    abl = ablacao(x, obs, model, sd)
    log(json.dumps(abl, ensure_ascii=False))
    return {"experimento_natural": nat, "parques": par, "ablacao": abl, "modelo": Path(ckpt).name}
