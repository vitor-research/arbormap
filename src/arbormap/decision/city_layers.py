"""
Camadas da cidade para o mapa interativo: composição da estação seca, previsão e incerteza

- Pixels zerados em quase todas as cenas (≥ 90%) são um artefato fixo do download (blocos
  retangulares, ~2,4% de Teresina) e viram "sem dado" permanente, não nuvem.
- Nuvem numa cena = pixel zerado fora desse artefato, ou mancha fria em terra firme
  (LST < mediana − 3 × DP robusto, fora da água).
- Composição: mediana, pixel a pixel, das cenas limpas da estação seca (jun–out), com os
  pixels de nuvem de cada cena excluídos. Entra no modelo como um "período seco típico".
- Previsão: modelo do fold em que a cidade é validação (nunca a viu no treino). Incerteza:
  desvio-padrão entre os 5 modelos dos folds, em °C.
"""

import json
from pathlib import Path

import numpy as np
import torch

from arbormap.model.hybrid_pipeline import (
    CH_DEM, CH_LST, CH_NDBI, CH_NDVI, CH_NDWI, LST_CLIP_MAX, LST_CLIP_MIN, HybridUrbanHeatNet,
    ensure_channels_first,
)

MESES_SECOS = (6, 7, 8, 9, 10)
CROP = 400


def _load(path):
    return ensure_channels_first(np.load(path).astype(np.float32))


def static_nodata_mask(files, freq_min=0.9):
    """Pixels com LST zerada em pelo menos `freq_min` das cenas (artefato fixo)."""
    z = np.stack([~(_load(f)[CH_LST] > 1.0) for f in files])   # zero ou NaN
    return z.mean(0) >= freq_min


def cloud_mask(img, static):
    """Pixels de nuvem numa cena: zerados (ou NaN) fora do artefato fixo ou manchas frias em terra firme."""
    lst, ndwi = img[CH_LST], img[CH_NDWI]
    zeros = ~(lst > 1.0) & ~static
    land = (ndwi < 0.1) & (lst > 1.0) & ~static
    vals = lst[land]
    med = float(np.median(vals)) if vals.size else 0.0
    mad = float(np.median(np.abs(vals - med))) * 1.4826 if vals.size else 0.0
    cold = land & (lst < med - 3 * mad)
    return zeros | cold


def dry_season_composite(city_dir, months=MESES_SECOS, max_cloud=0.03, years=None, buraco_so_lst=False):
    """Mediana das cenas limpas da estação seca (opcionalmente só dos anos `years`).
    Devolve as bandas compostas e o registro.

    buraco_so_lst: para os dados do Planetary Computer. Onde falta só a banda térmica (o produto de
    temperatura da superfície do Landsat C2 tem lacunas onde as demais bandas existem), descarta-se
    só a LST, mantendo NDVI, NDBI, NDWI e DEM, que são as entradas do modelo."""
    files = sorted(Path(city_dir).glob("*.npy"))
    static = static_nodata_mask(files)
    usable = ~static
    used, rejected, stack = [], [], []
    for f in files:
        mes = int(f.stem.split("_")[-1])
        if mes not in months or (years is not None and int(f.stem.split("_")[0]) not in years):
            continue
        img = _load(f)
        cm = cloud_mask(img, static)
        frac = float(cm[usable].mean())
        if frac > max_cloud:
            rejected.append({"arquivo": f.name, "nuvem": round(frac, 4)})
            continue
        img = img.copy()
        if buraco_so_lst:
            gap = ~(img[CH_LST] > 1.0) & np.isfinite(img[CH_NDVI])
            img[:, cm & ~gap] = np.nan
            img[CH_LST, gap | static] = np.nan
        else:
            img[:, cm | static] = np.nan
        stack.append(img)
        used.append({"arquivo": f.name, "nuvem": round(frac, 4)})
    if not stack:
        raise RuntimeError("Nenhuma cena limpa na estação seca.")
    comp = np.nanmedian(np.stack(stack), axis=0)
    count = np.isfinite(np.stack([s[CH_LST] for s in stack])).sum(0)
    return comp, static, count, {"usadas": used, "rejeitadas": rejected, "meses": list(months),
                                 "max_nuvem": max_cloud, "pixels_artefato_fixo": int(static.sum())}


def _center_crop(a, crop=CROP):
    h, w = a.shape[-2:]
    y0, x0 = max(0, (h - crop) // 2), max(0, (w - crop) // 2)
    return y0, x0, min(crop, h - y0), min(crop, w - x0)


def model_input(comp, stats, crop=CROP):
    """Replica UrbanHeatDataset.__getitem__ (validação) para a composição: recorte central,
    clip, normalização com as estatísticas do fold e ordem [NDVI, NDBI, DEM, NDWI]."""
    mean, std = np.asarray(stats[0], dtype=np.float32), np.asarray(stats[1], dtype=np.float32)
    y0, x0, hh, ww = _center_crop(comp, crop)
    raw = np.zeros((5, crop, crop), dtype=np.float32)
    part = comp[:, y0:y0 + hh, x0:x0 + ww]
    # bandas de entrada ausentes (artefato fixo) recebem a mediana da banda, para não virarem 0
    filled = np.where(np.isfinite(part), part, np.nanmedian(part.reshape(5, -1), axis=1)[:, None, None])
    raw[:, :hh, :ww] = filled
    raw[CH_LST] = np.clip(raw[CH_LST], LST_CLIP_MIN, LST_CLIP_MAX)
    raw[CH_NDVI] = np.clip(raw[CH_NDVI], -1, 1)
    raw[CH_NDWI] = np.clip(raw[CH_NDWI], -1, 1)
    valid = np.zeros((crop, crop), dtype=np.float32)
    valid[:hh, :ww] = 1.0
    norm = (raw - mean[:, None, None]) / (std[:, None, None] + 1e-6) * valid
    x = np.stack([norm[CH_NDVI], norm[CH_NDBI], norm[CH_DEM], norm[CH_NDWI]])
    return torch.from_numpy(x).unsqueeze(0).float(), valid.astype(bool), (y0, x0, hh, ww), raw


def load_model(ckpt, device="cpu"):
    st = torch.load(ckpt, map_location=device, weights_only=False)
    m = HybridUrbanHeatNet(in_channels=4, base_c=st.get("base_c", 64)).to(device).eval()
    m.load_state_dict(st["state_dict"])
    return m, st


def predict_city(city_dir, ckpt_dir, city, device="cpu", months=MESES_SECOS, max_cloud=0.03):
    """Composição + previsão do modelo que não viu a cidade + incerteza entre os 5 modelos."""
    comp, static, count, reg = dry_season_composite(city_dir, months, max_cloud)
    y0, x0, hh, ww = _center_crop(comp)
    obs = np.full((CROP, CROP), np.nan, dtype=np.float32)
    obs[:hh, :ww] = comp[CH_LST, y0:y0 + hh, x0:x0 + ww]
    ok = np.isfinite(obs)
    mu, sd = float(np.nanmean(obs)), float(np.nanstd(obs))

    preds, held_out = [], None
    for ck in sorted(Path(ckpt_dir).glob("best_hybrid_fold_*.pt")):
        model, st = load_model(ck, device)
        x, valid, _, raw = model_input(comp, (st["mean"], st["std"]))
        with torch.no_grad():
            z = model(x.to(device))[0, 0].cpu().numpy()
        c = np.where(valid, z * sd + mu, np.nan)
        preds.append(c)
        if city in st["val_cities"]:
            held_out = {"pred": c, "ckpt": ck.name, "fold": st["fold"] + 1, "val_cities": st["val_cities"],
                        "raw_input": raw, "x": x, "stats": (st["mean"], st["std"]), "lst_min_valida": st.get("lst_min_valida")}
    if held_out is None:
        raise RuntimeError(f"Nenhum fold tem {city} na validação.")
    P = np.stack(preds)
    pred = held_out["pred"]
    err = np.abs(pred - obs)
    meta = {"cidade": city, "composicao": reg, "lst_media_composicao_c": mu, "lst_dp_composicao_c": sd,
            "modelo_principal": held_out["ckpt"], "fold_principal": held_out["fold"],
            "val_cities_fold_principal": held_out["val_cities"],
            "mae_composicao_c": float(np.nanmean(err[ok])),
            "pearson_composicao": float(np.corrcoef(pred[ok], obs[ok])[0, 1]),
            "incerteza_media_c": float(np.nanmean(P.std(0))),
            "shape_original": list(comp.shape[1:]), "recorte_y0_x0": [y0, x0], "recorte_h_w": [hh, ww]}
    band = lambda c: np.pad(comp[c, y0:y0 + hh, x0:x0 + ww], ((0, CROP - hh), (0, CROP - ww)), constant_values=np.nan)
    layers = {"lst_pred": pred, "lst_obs": obs, "incerteza": P.std(0), "anomalia": pred - mu,
              "ndvi": band(CH_NDVI), "ndbi": band(CH_NDBI), "ndwi": band(CH_NDWI), "dem": band(CH_DEM),
              "n_cenas": np.pad(count[y0:y0 + hh, x0:x0 + ww], ((0, CROP - hh), (0, CROP - ww))),
              "artefato_fixo": np.pad(static[y0:y0 + hh, x0:x0 + ww], ((0, CROP - hh), (0, CROP - ww)))}
    return layers, meta, held_out


def _janelas(n, win, stride):
    pos = list(range(0, max(1, n - win) + 1, stride))
    if pos[-1] != max(0, n - win):
        pos.append(max(0, n - win))
    return pos


def _peso(win):
    t = np.minimum(np.arange(win) + 1, np.arange(win)[::-1] + 1).astype(np.float32)
    t = np.minimum(t, win / 4) / (win / 4)
    return np.outer(t, t)


def windowed_input(comp, stats):
    """Entrada normalizada da grade inteira (4, H, W), com as bandas ausentes preenchidas pela mediana."""
    mean, std = np.asarray(stats[0], np.float32), np.asarray(stats[1], np.float32)
    raw = np.where(np.isfinite(comp), comp, np.nanmedian(comp.reshape(5, -1), axis=1)[:, None, None]).astype(np.float32)
    raw[CH_LST] = np.clip(raw[CH_LST], LST_CLIP_MIN, LST_CLIP_MAX)
    raw[CH_NDVI] = np.clip(raw[CH_NDVI], -1, 1)
    raw[CH_NDWI] = np.clip(raw[CH_NDWI], -1, 1)
    norm = (raw - mean[:, None, None]) / (std[:, None, None] + 1e-6)
    return np.stack([norm[CH_NDVI], norm[CH_NDBI], norm[CH_DEM], norm[CH_NDWI]])


def predict_windowed(model, x, obs, win=CROP, stride=CROP // 2, device="cpu"):
    """Previsão numa grade maior que o recorte de treino: janelas de `win` px com sobreposição.
    Em cada janela, a anomalia padronizada vira °C com a média e o DP da LST observada da própria
    janela (a mesma conversão da validação); as janelas são combinadas com peso maior no centro."""
    _, H, W = x.shape
    acc = np.zeros((H, W), np.float32); wsum = np.zeros((H, W), np.float32)
    pw = _peso(win)
    for y0 in _janelas(H, win, stride):
        for x0 in _janelas(W, win, stride):
            o = obs[y0:y0 + win, x0:x0 + win]
            if np.isfinite(o).mean() < 0.2:      # precisa de LST observada para converter em °C
                continue
            mu, sd = float(np.nanmean(o)), float(np.nanstd(o))
            xx = torch.from_numpy(x[:, y0:y0 + win, x0:x0 + win][None]).float().to(device)
            with torch.no_grad():
                z = model(xx)[0, 0].cpu().numpy()
            acc[y0:y0 + win, x0:x0 + win] += (z * sd + mu) * pw
            wsum[y0:y0 + win, x0:x0 + win] += pw
    return np.where(wsum > 0, acc / np.maximum(wsum, 1e-6), np.nan)


def predict_city_large(city_dir, ckpt_dir, city, device="cpu", months=MESES_SECOS, max_cloud=1.0, log=print):
    """Como predict_city, para uma grade maior que 400 px (sem recorte), por janelas."""
    comp, static, count, reg = dry_season_composite(city_dir, months, max_cloud, buraco_so_lst=True)
    reg["pixels_sem_lst_observada"] = int((~np.isfinite(comp[CH_LST])).sum())
    static = ~np.isfinite(comp[CH_NDVI])          # sem dado só onde faltam as entradas do modelo
    reg["pixels_artefato_fixo"] = int(static.sum())
    obs = comp[CH_LST].copy()
    ok = np.isfinite(obs)
    mu = float(np.nanmean(obs))
    preds, held = [], None
    for ck in sorted(Path(ckpt_dir).glob("best_hybrid_fold_*.pt")):
        model, st = load_model(ck, device)
        x = windowed_input(comp, (st["mean"], st["std"]))
        p = predict_windowed(model, x, obs, device=device)
        preds.append(p)
        log(f"   {ck.name}: MAE {np.nanmean(np.abs(p - obs)[ok]):.2f} °C")
        if city in st["val_cities"]:
            held = {"pred": p, "ckpt": ck.name, "fold": st["fold"] + 1, "val_cities": st["val_cities"], "stats": (st["mean"], st["std"])}
    P = np.stack(preds)
    pred = held["pred"]
    v = ok & np.isfinite(pred)
    meta = {"cidade": city, "composicao": reg, "lst_media_composicao_c": mu, "lst_dp_composicao_c": float(np.nanstd(obs)),
            "modelo_principal": held["ckpt"], "fold_principal": held["fold"], "val_cities_fold_principal": held["val_cities"],
            "mae_composicao_c": float(np.abs(pred - obs)[v].mean()), "pearson_composicao": float(np.corrcoef(pred[v], obs[v])[0, 1]),
            "incerteza_media_c": float(np.nanmean(P.std(0))), "shape_original": list(comp.shape[1:]),
            "recorte_y0_x0": [0, 0], "recorte_h_w": list(comp.shape[1:]), "janela_px": CROP, "passo_px": CROP // 2}
    layers = {"lst_pred": pred, "lst_obs": obs, "incerteza": P.std(0), "anomalia": pred - mu,
              "ndvi": comp[CH_NDVI], "ndbi": comp[CH_NDBI], "ndwi": comp[CH_NDWI], "dem": comp[CH_DEM],
              "n_cenas": count, "artefato_fixo": static}
    return layers, meta, held


def save_layers(layers, meta, out_dir, prefix):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{prefix}_camadas.npz", **{k: np.asarray(v, dtype=np.float32) for k, v in layers.items()})
    (out_dir / f"{prefix}_camadas.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
