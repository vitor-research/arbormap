"""
Motor de decisão sobre a UrbanHeatNet (4 canais): seleção multicritério, corredores e baselines

Substitui, para o modelo final, a simulação de intervention_simulation.py (célula 7), que foi
escrita para o UNetConvLSTM de 8 canais. Mantém do original o critério híbrido (60% potencial
de resfriamento estimado pela rede + 40% urgência térmica), a suavização gaussiana do
potencial, o espalhamento do resfriamento (0,7 local 5×5 + 0,3 brisa 15×15) e a métrica
(redução média de LST nos pixels plantados). Diferenças:
  - nenhum canal de temperatura é editado: a rede só vê NDVI, NDBI, DEM e NDWI;
  - o pixel plantado recebe a assinatura espectral mediana da vegetação densa observada
    na própria cena (NDVI >= 0,5), em vez de valores fixos (o original usava NDWI = 0,10,
    valor típico de água);
  - áreas inviáveis: corpos d'água (NDWI >= 0,1) e declividade > 15° calculada do DEM
    (o original comparava o DEM com a média da cena);
  - corredores formados por morfologia (fechamento + remoção de componentes pequenos),
    ordenados por prioridade, com o efeito individual de cada corredor;
  - baseline aleatória com média e desvio-padrão de várias sorteios.
"""

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from arbormap.model.hybrid_pipeline import (
    CH_DEM,
    CH_LST,
    CH_NDBI,
    CH_NDVI,
    CH_NDWI,
    LST_CLIP_MAX,
    LST_CLIP_MIN,
    HybridUrbanHeatNet,
    UrbanHeatDataset,
    ensure_channels_first,
)

PIXEL_M = 30.0
PIXEL_HA = PIXEL_M * PIXEL_M / 10_000.0          # 0,09 ha
CROP = 400                                       # mesmo recorte central do dataset de validação

# Posição de cada canal bruto no tensor de entrada da rede: [NDVI, NDBI, DEM, NDWI]
MODEL_CH = {CH_NDVI: 0, CH_NDBI: 1, CH_DEM: 2, CH_NDWI: 3}

PARAMS_PADRAO = {
    "ndvi_plantavel_max": 0.2,     # só se planta onde ainda não há vegetação (critério do original)
    "ndwi_agua_min": 0.1,          # NDWI a partir do qual o pixel é tratado como água (original)
    "declividade_max_graus": 15.0,
    "ndvi_vegetacao_densa": 0.5,
    "peso_potencial": 0.6,
    "peso_urgencia": 0.4,
    "area_alvo_candidatos_px": 400,  # ~36 ha de candidatos: cobre o controle de 0 a 15 ha com folga
    "min_px_corredor": 4,
    "area_candidata_min_px": 200,    # mínimo após a morfologia (~18 ha)
    "n_sorteios_aleatoria": 20,
    "seed": 42,
}


# ============================================================
# MODELO E CENA
# ============================================================
def load_fold_model(ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = HybridUrbanHeatNet(in_channels=4, base_c=state.get("base_c", 64)).to(device)
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model, state


def fold_of_city(fold_results, city):
    for r in fold_results:
        if city in r["val_cities"]:
            return r["fold"] - 1
    raise ValueError(f"{city} não está na validação de nenhum fold.")


def load_scene(data_dir, city, valid_files_set, stats, pair_index=-1, lst_min_valida=None):
    """Carrega um par (cena base, cena alvo) exatamente como o dataset de validação do treino.

    Devolve o tensor de entrada normalizado, o alvo, a máscara válida, as bandas brutas
    recortadas da cena base, a LST observada da cena alvo e os metadados do recorte.
    lst_min_valida deve ser o mesmo do treino (gravado no checkpoint).
    """
    from arbormap.model.train_urbanheatnet import UrbanHeatDatasetMasked
    ds = UrbanHeatDatasetMasked(data_dir, allowed_cities=[city], valid_files_set=valid_files_set,
                                normalize=True, stats=stats, is_train=False, lst_min_valida=lst_min_valida)
    if len(ds) == 0:
        raise RuntimeError(f"Nenhum par válido para {city}.")
    idx = pair_index % len(ds)
    x, tgt_z, valid, std_anom = ds[idx]
    base_file, target_file = ds.samples[idx]

    base = ensure_channels_first(np.load(base_file).astype(np.float32))
    tgt = ensure_channels_first(np.load(target_file).astype(np.float32))
    h, w = min(base.shape[1], tgt.shape[1]), min(base.shape[2], tgt.shape[2])
    y0, x0 = max(0, (h - CROP) // 2), max(0, (w - CROP) // 2)
    hh, ww = min(CROP, h - y0), min(CROP, w - x0)

    raw = np.zeros((5, CROP, CROP), dtype=np.float32)
    raw[:, :hh, :ww] = base[:, y0:y0 + hh, x0:x0 + ww]
    raw[CH_NDVI] = np.clip(raw[CH_NDVI], -1, 1)
    raw[CH_NDWI] = np.clip(raw[CH_NDWI], -1, 1)
    lst_obs = np.full((CROP, CROP), np.nan, dtype=np.float32)
    lst_obs[:hh, :ww] = np.clip(tgt[CH_LST, y0:y0 + hh, x0:x0 + ww], LST_CLIP_MIN, LST_CLIP_MAX)
    # Pixels com LST <= 1 °C são lacunas de nuvem gravadas como 0 no download, não temperatura
    lst_obs[lst_obs <= 1.0] = np.nan

    valid_np = valid[0].numpy().astype(bool) & np.isfinite(lst_obs)
    macro = float(np.nanmean(lst_obs[valid_np]))
    # Para predição, plantio e mapa vale a área com entrada válida (todo o recorte),
    # inclusive onde a LST observada foi perdida para nuvem.
    valid = torch.zeros_like(valid)
    valid[0, :hh, :ww] = 1.0
    meta = {"cidade": city, "arquivo_base": Path(base_file).name, "arquivo_alvo": Path(target_file).name,
            "par_indice": idx, "n_pares": len(ds), "shape_original": [int(h), int(w)],
            "recorte_y0_x0": [int(y0), int(x0)], "recorte_h_w": [int(hh), int(ww)],
            "lst_media_cena_c": macro, "lst_dp_cena_c": float(std_anom)}
    return x.unsqueeze(0), tgt_z.unsqueeze(0), valid.unsqueeze(0), raw, lst_obs, float(std_anom), macro, meta


def select_clear_dry_pair(data_dir, city, valid_files_set, dry_months=(6, 7, 8, 9, 10)):
    """Índice do par (base -> alvo) mais recente da estação seca com céu limpo.

    Entre os pares cujo mês-alvo está na estação seca, mantêm-se os de menor fração de
    pixels de LST abaixo de 15 °C (até 0,1 ponto percentual acima do mínimo) e com desvio-
    padrão da LST até a mediana desses pares; escolhe-se o mais recente. Devolve também a
    tabela usada, para registro.
    """
    ds = UrbanHeatDataset(data_dir, allowed_cities=[city], valid_files_set=valid_files_set, normalize=False)
    rows = []
    for i, (_, tgt_file) in enumerate(ds.samples):
        month = int(Path(tgt_file).stem.split("_")[-1])
        lst = ensure_channels_first(np.load(tgt_file).astype(np.float32))[CH_LST]
        rows.append({"indice": i, "alvo": Path(tgt_file).name, "mes": month,
                     "fracao_abaixo_15c": float((lst < 15).mean()), "dp_c": float(lst.std())})
    dry = [r for r in rows if r["mes"] in dry_months] or rows
    fmin = min(r["fracao_abaixo_15c"] for r in dry)
    clear = [r for r in dry if r["fracao_abaixo_15c"] <= fmin + 0.001]
    med = float(np.median([r["dp_c"] for r in clear]))
    chosen = max((r for r in clear if r["dp_c"] <= med), key=lambda r: r["indice"])
    return chosen["indice"], rows


def cloud_score(path):
    """Fração da cena com sinal de nuvem: LST zerada (lacuna) ou mancha fria em terra firme.

    Mancha fria: pixel fora da água (NDWI < 0,1) com LST abaixo de mediana − 3 × DP robusto
    (1,4826 × desvio absoluto mediano) da terra firme da própria cena.
    """
    img = ensure_channels_first(np.load(path).astype(np.float32))
    lst, ndwi = img[CH_LST], img[CH_NDWI]
    zeros = lst <= 1.0
    land = (ndwi < 0.1) & ~zeros
    vals = lst[land]
    med = float(np.median(vals)) if vals.size else 0.0
    mad = float(np.median(np.abs(vals - med))) * 1.4826 if vals.size else 0.0
    cold = land & (lst < med - 3 * mad)
    return {"nuvem": float((zeros | cold).mean()), "zerados": float(zeros.mean()), "frios": float(cold.mean())}


def select_recent_clear_pair(data_dir, city, valid_files_set, quantil=0.3):
    """Par mais recente cujas duas cenas estão entre as mais limpas da cidade.

    Percorre os pares do mais recente para o mais antigo e escolhe o primeiro em que o escore
    de nuvem da cena-base e da cena-alvo é no máximo o quantil `quantil` dos escores da cidade.
    Devolve o índice e a tabela de todos os pares (para registro e para a prancha).
    """
    ds = UrbanHeatDataset(data_dir, allowed_cities=[city], valid_files_set=valid_files_set, normalize=False)
    cache = {}
    def score(p):
        if p not in cache:
            cache[p] = cloud_score(p)
        return cache[p]
    rows = []
    for i, (b, t) in enumerate(ds.samples):
        sb, st = score(b), score(t)
        rows.append({"indice": i, "base": Path(b).name, "alvo": Path(t).name,
                     "nuvem_base": sb["nuvem"], "nuvem_alvo": st["nuvem"],
                     "zerados_alvo": st["zerados"], "frios_alvo": st["frios"]})
    thr = float(np.quantile([v["nuvem"] for v in cache.values()], quantil))
    chosen = next((r for r in reversed(rows) if r["nuvem_base"] <= thr and r["nuvem_alvo"] <= thr), None)
    if chosen is None:   # nenhum par abaixo do limiar: o de menor escore de nuvem
        chosen = min(rows, key=lambda r: max(r["nuvem_base"], r["nuvem_alvo"]))
    for r in rows:
        r["limiar"] = thr
        r["escolhido"] = r["indice"] == chosen["indice"]
    return chosen["indice"], rows


def figure_scene_sheet(data_dir, city, rows, save_path, n=12):
    """Prancha com a LST-alvo dos `n` pares mais recentes, escore de nuvem e o par escolhido."""
    sel = rows[-n:]
    cols = 4
    lines = int(np.ceil(len(sel) / cols))
    fig, axes = plt.subplots(lines, cols, figsize=(4 * cols, 4 * lines), squeeze=False)
    for ax in axes.flat:
        ax.set_axis_off()
    for ax, r in zip(axes.flat, reversed(sel)):
        lst = ensure_channels_first(np.load(Path(data_dir) / city / r["alvo"]).astype(np.float32))[CH_LST]
        show = np.where(lst <= 1.0, np.nan, lst)
        lo, hi = np.nanpercentile(show, [2, 98])
        ax.imshow(show, cmap="inferno", vmin=lo, vmax=hi)
        ok = r["nuvem_base"] <= r["limiar"] and r["nuvem_alvo"] <= r["limiar"]
        cor = "green" if r["escolhido"] else ("black" if ok else "red")
        ax.set_title(f"{r['base'][:7]} → {r['alvo'][:7]}\nnuvem base {r['nuvem_base']:.1%} | alvo {r['nuvem_alvo']:.1%}"
                     + ("\nESCOLHIDO" if r["escolhido"] else ""), fontsize=9, color=cor)
    fig.suptitle(f"{city}: pares mais recentes (vermelho = acima do limiar de nuvem {sel[0]['limiar']:.1%})", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def predict_celsius(model, x, std_ref, mean_ref):
    with torch.no_grad():
        z = model(x)
    return (z * std_ref + mean_ref).squeeze().cpu().numpy()


# ============================================================
# CRITÉRIOS FÍSICOS
# ============================================================
def slope_degrees(dem):
    gy, gx = np.gradient(dem.astype(np.float64), PIXEL_M)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def physical_masks(raw, valid, p):
    water = raw[CH_NDWI] >= p["ndwi_agua_min"]
    steep = slope_degrees(raw[CH_DEM]) > p["declividade_max_graus"]
    inviavel = (water | steep) & valid
    plantavel = valid & ~inviavel & (raw[CH_NDVI] < p["ndvi_plantavel_max"])
    return inviavel, plantavel, water, steep


def vegetation_signature(raw, valid, p):
    """Mediana de NDVI, NDBI e NDWI dos pixels de vegetação densa já existentes na cena."""
    ndvi = raw[CH_NDVI]
    sel = valid & (ndvi >= p["ndvi_vegetacao_densa"])
    criterio = f"NDVI >= {p['ndvi_vegetacao_densa']}"
    if sel.sum() < 20:
        thr = np.quantile(ndvi[valid], 0.98)
        sel = valid & (ndvi >= thr)
        criterio = f"2% de maior NDVI (NDVI >= {thr:.3f})"
    sig = {c: float(np.median(raw[c][sel])) for c in (CH_NDVI, CH_NDBI, CH_NDWI)}
    return sig, {"criterio": criterio, "n_pixels": int(sel.sum()),
                 "ndvi": sig[CH_NDVI], "ndbi": sig[CH_NDBI], "ndwi": sig[CH_NDWI]}


def apply_planting(x, mask, signature, stats):
    """Troca NDVI, NDBI e NDWI dos pixels plantados pela assinatura de vegetação (normalizada)."""
    mean, std = stats
    out = x.clone()
    m = torch.from_numpy(mask).to(x.device)
    for c, v in signature.items():
        out[0, MODEL_CH[c]][m] = float((v - mean[c]) / (std[c] + 1e-6))
    return out


# ============================================================
# POTENCIAL DE RESFRIAMENTO E CORREDORES
# ============================================================
def cooling_potential(model, x, valid, signature, stats, std_ref, mean_ref):
    """Derivada direcional do calor nos hotspots ao converter cada pixel em vegetação.

    H = soma da LST prevista nos pixels acima do percentil 90. Para cada pixel, estima-se
    a variação de H ao mover NDVI, NDBI e NDWI até a assinatura de vegetação (aproximação
    de primeira ordem); o potencial é a redução estimada (valores positivos).
    """
    xg = x.clone().detach().requires_grad_(True)
    lst = model(xg) * std_ref + mean_ref
    vmask = valid.to(x.device) == 1.0
    q90 = torch.quantile(lst[vmask], 0.9)
    heat = (lst * ((lst >= q90) & vmask).float()).sum()
    model.zero_grad()
    heat.backward()
    grad = xg.grad[0].detach().cpu().numpy()
    x_np = x[0].detach().cpu().numpy()
    mean, std = stats
    dheat = np.zeros(x_np.shape[1:], dtype=np.float64)
    for c, v in signature.items():
        k = MODEL_CH[c]
        target_norm = (v - mean[c]) / (std[c] + 1e-6)
        dheat += grad[k] * (target_norm - x_np[k])
    return np.maximum(0.0, -dheat).astype(np.float32)


def _norm01(a, mask):
    vals = a[mask]
    lo, hi = (vals.min(), vals.max()) if vals.size else (0.0, 1.0)
    return (a - lo) / (hi - lo + 1e-8)


def multicriteria_score(potential, lst_base, plantavel, p):
    smoothed = cv2.GaussianBlur(potential, (5, 5), 1.0)
    score = p["peso_potencial"] * _norm01(smoothed, plantavel) + p["peso_urgencia"] * _norm01(lst_base, plantavel)
    return np.where(plantavel, score, 0.0).astype(np.float32)


def build_corridors(score, plantavel, p):
    """Candidatos de maior escore -> fechamento morfológico -> componentes conexos -> corredores.

    O limiar é o quantil que deixa cerca de area_alvo_candidatos_px pixels candidatos; se,
    depois da morfologia e da remoção de componentes pequenos, sobrar menos que
    area_candidata_min_px, a área-alvo é ampliada.
    """
    vals = score[plantavel]
    alvo = p["area_alvo_candidatos_px"]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    while True:
        q = max(0.5, 1.0 - alvo / max(vals.size, 1))
        cand = plantavel & (score >= np.quantile(vals, q))
        closed = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool) & plantavel
        n, labels, st, _ = cv2.connectedComponentsWithStats(closed.astype(np.uint8), connectivity=8)
        keep = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= p["min_px_corredor"]]
        area = sum(st[i, cv2.CC_STAT_AREA] for i in keep)
        if area >= p["area_candidata_min_px"] or q <= 0.5:
            break
        alvo = int(alvo * 1.5)
    corridors = []
    for i in keep:
        m = labels == i
        corridors.append({"mask": m, "n_px": int(m.sum()), "score_medio": float(score[m].mean())})
    corridors.sort(key=lambda c: c["score_medio"], reverse=True)
    for rank, c in enumerate(corridors, 1):
        c["prioridade"] = rank
    return corridors, q


def advected_cooling(lst_base, lst_after):
    raw = (lst_base - lst_after).astype(np.float32)
    return 0.7 * cv2.GaussianBlur(raw, (5, 5), 1.0) + 0.3 * cv2.GaussianBlur(raw, (15, 15), 3.0)


def evaluate(model, x, mask, signature, stats, std_ref, mean_ref, lst_base):
    after = predict_celsius(model, apply_planting(x, mask, signature, stats), std_ref, mean_ref)
    cooling = advected_cooling(lst_base, after)
    return cooling, (float(cooling[mask].mean()) if mask.any() else 0.0), after


def select_multi(corridors, score, budget_px):
    """Acumula corredores por prioridade; o último entra parcialmente pelos pixels de maior escore."""
    mask = np.zeros_like(score, dtype=bool)
    left = budget_px
    for c in corridors:
        if left <= 0:
            break
        if c["n_px"] <= left:
            mask |= c["mask"]
            left -= c["n_px"]
        else:
            ys, xs = np.where(c["mask"])
            order = np.argsort(score[ys, xs])[::-1][:left]
            mask[ys[order], xs[order]] = True
            left = 0
    return mask


def select_greedy(lst_base, plantavel, budget_px):
    ys, xs = np.where(plantavel)
    order = np.argsort(lst_base[ys, xs])[::-1][:budget_px]
    mask = np.zeros_like(plantavel)
    mask[ys[order], xs[order]] = True
    return mask


def select_random(plantavel, budget_px, rng):
    ys, xs = np.where(plantavel)
    pick = rng.choice(len(ys), size=min(budget_px, len(ys)), replace=False)
    mask = np.zeros_like(plantavel)
    mask[ys[pick], xs[pick]] = True
    return mask


# ============================================================
# EXECUÇÃO COMPLETA PARA UMA CIDADE
# ============================================================
def run_city(model, ckpt_state, data_dir, valid_files_set, city="Teresina", pair_index=-1,
             budgets=(10, 20, 30, 40, 45, 50, 75, 100, 150), budget_figura=45, params=None,
             hotspot_mae_ref=None, device=None):
    p = {**PARAMS_PADRAO, **(params or {})}
    device = device or next(model.parameters()).device
    stats = (np.asarray(ckpt_state["mean"]), np.asarray(ckpt_state["std"]))

    x, tgt_z, valid_t, raw, lst_obs, std_ref, mean_ref, meta = load_scene(
        data_dir, city, valid_files_set, stats, pair_index, lst_min_valida=ckpt_state.get("lst_min_valida"))
    x = x.to(device)
    valid = valid_t[0, 0].numpy().astype(bool)
    lst_base = predict_celsius(model, x, std_ref, mean_ref)

    inviavel, plantavel, water, steep = physical_masks(raw, valid, p)
    signature, sig_meta = vegetation_signature(raw, valid, p)
    potential = cooling_potential(model, x, valid_t, signature, stats, std_ref, mean_ref)
    score = multicriteria_score(potential, lst_base, plantavel, p)
    corridors, q_used = build_corridors(score, plantavel, p)

    for c in corridors:
        _, c["resfriamento_c"], after = evaluate(model, x, c["mask"], signature, stats, std_ref, mean_ref, lst_base)
        c["lst_media_antes"] = float(lst_base[c["mask"]].mean())
        c["lst_media_depois"] = float(after[c["mask"]].mean())
        c["area_ha"] = c["n_px"] * PIXEL_HA

    rng = np.random.default_rng(p["seed"])
    curve = []
    figure_data = None
    for b in budgets:
        if b > plantavel.sum():
            continue
        m_multi = select_multi(corridors, score, b)
        m_greedy = select_greedy(lst_base, plantavel, b)
        cool_multi, avg_multi, _ = evaluate(model, x, m_multi, signature, stats, std_ref, mean_ref, lst_base)
        cool_greedy, avg_greedy, _ = evaluate(model, x, m_greedy, signature, stats, std_ref, mean_ref, lst_base)
        rand_vals, cool_rand, m_rand = [], None, None
        for _ in range(p["n_sorteios_aleatoria"]):
            m = select_random(plantavel, b, rng)
            cool, avg, _ = evaluate(model, x, m, signature, stats, std_ref, mean_ref, lst_base)
            rand_vals.append(avg)
            if cool_rand is None:
                cool_rand, m_rand = cool, m
        benef = lambda cool: float(((cool > 0.1) & valid).sum() * PIXEL_HA)
        row = {"budget_px": int(b), "area_ha": round(b * PIXEL_HA, 2),
               "multi_c": avg_multi, "multi_px_selecionados": int(m_multi.sum()),
               "gulosa_c": avg_greedy,
               "aleatoria_c_media": float(np.mean(rand_vals)), "aleatoria_c_dp": float(np.std(rand_vals, ddof=1)),
               # área com resfriamento > 0,1 °C (plantio + vizinhança), em hectares
               "multi_area_beneficiada_ha": benef(cool_multi), "gulosa_area_beneficiada_ha": benef(cool_greedy),
               "aleatoria_area_beneficiada_ha": benef(cool_rand)}
        curve.append(row)
        print(f"   Orçamento {b:3d} px ({row['area_ha']:.2f} ha): Multi {avg_multi:.2f} °C | "
              f"Gulosa {avg_greedy:.2f} °C | Aleatória {row['aleatoria_c_media']:.2f} ± {row['aleatoria_c_dp']:.2f} °C")
        if b == budget_figura:
            figure_data = {"budget": b, "multi": (m_multi, cool_multi, avg_multi),
                           "greedy": (m_greedy, cool_greedy, avg_greedy),
                           "random": (m_rand, cool_rand, float(np.mean(rand_vals)))}

    return {
        "meta": {**meta, "assinatura_vegetacao": sig_meta, "quantil_candidatos_usado": q_used,
                 "parametros": p, "n_corredores": len(corridors),
                 "area_candidata_ha": round(sum(c["n_px"] for c in corridors) * PIXEL_HA, 2),
                 "pixels_inviaveis": int(inviavel.sum()), "pixels_agua": int((water & valid).sum()),
                 "pixels_declividade_maior_15": int((steep & valid).sum()),
                 "pixels_plantaveis": int(plantavel.sum()), "hotspot_mae_ref_c": hotspot_mae_ref},
        "valid": valid, "raw": raw, "lst_obs": lst_obs, "lst_base": lst_base,
        "inviavel": inviavel, "plantavel": plantavel, "score": score,
        "corridors": corridors, "curve": curve, "figure": figure_data,
    }


# ============================================================
# FIGURAS
# ============================================================
def figure_obs_pred_error(res, save_path):
    """Figura 4: LST observada × prevista × erro absoluto, em °C, na cena da cidade."""
    v = res["valid"]
    obs = np.where(v, res["lst_obs"], np.nan)
    pred = np.where(v, res["lst_base"], np.nan)
    err = np.abs(obs - pred)
    lo, hi = np.nanpercentile(np.concatenate([obs[v], pred[v]]), [1, 99])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, img, title, cmap, vmin, vmax in [
        (axes[0], obs, "LST observada (°C)", "inferno", lo, hi),
        (axes[1], pred, "LST prevista (°C)", "inferno", lo, hi),
        (axes[2], err, f"Erro absoluto (°C) | MAE = {np.nanmean(err):.2f} °C", "Reds", 0, np.nanpercentile(err, 99)),
    ]:
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    return float(np.nanmean(err))


def figure_strategies(res, save_path, hotspot_label="Hotspot MAE"):
    """Figura 5: alocação e resfriamento das três estratégias no orçamento da figura."""
    fd = res["figure"]
    if fd is None:
        return
    base = np.where(res["valid"], res["lst_base"], np.nan)
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.5), gridspec_kw={"wspace": 0.05, "hspace": 0.12})
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    im_base = axes[0, 0].imshow(base, cmap="coolwarm")
    axes[0, 0].set_title("LST prevista (°C)", fontsize=10, fontweight="bold")
    vmax = max(fd["multi"][1].max(), 1e-6)
    for col, (title, key) in enumerate([("Multicritério", "multi"), ("Gulosa", "greedy"), ("Aleatória", "random")], 1):
        mask, cool, _ = fd[key]
        axes[0, col].imshow(base, cmap="gray", alpha=0.6)
        overlay = np.zeros((*mask.shape, 4))
        overlay[mask] = [0, 0.8, 0, 1]
        axes[0, col].imshow(overlay)
        axes[0, col].set_title(f"Alocação: {title}", fontsize=10)
        im_cool = axes[1, col].imshow(cool, cmap="Blues", vmin=0, vmax=vmax)
        axes[1, col].set_title("ΔT resfriamento", fontsize=10)
    axes[1, 0].axis("off")
    hs = res["meta"]["hotspot_mae_ref_c"]
    text = (f"Orçamento: {fd['budget']} px ({fd['budget'] * PIXEL_HA:.2f} ha)\n"
            f"Redução média (ΔT):\n"
            f"• Multicritério: {fd['multi'][2]:.2f} °C\n"
            f"• Gulosa: {fd['greedy'][2]:.2f} °C\n"
            f"• Aleatória: {fd['random'][2]:.2f} °C\n"
            + (f"\n{hotspot_label}: {hs:.2f} °C" if hs is not None else ""))
    axes[1, 0].text(0.5, 0.5, text, ha="center", va="center", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray"))
    fig.colorbar(im_base, cax=fig.add_axes([0.13, 0.52, 0.01, 0.35]))
    fig.colorbar(im_cool, cax=fig.add_axes([0.91, 0.12, 0.01, 0.35]), label="ΔT (°C)")
    plt.subplots_adjust(left=0.18, right=0.9, top=0.9, bottom=0.1)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def figure_budget_curve(res, save_path):
    """Figura 6: redução média por orçamento, com faixa ±1 DP da alocação aleatória."""
    c = res["curve"]
    if not c:
        return
    b = [r["budget_px"] for r in c]
    rm = np.array([r["aleatoria_c_media"] for r in c])
    rs = np.array([r["aleatoria_c_dp"] for r in c])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(b, [r["multi_c"] for r in c], marker="o", color="#10b981", linewidth=2.5, label="Multicritério (proposto)")
    ax1.plot(b, [r["gulosa_c"] for r in c], marker="s", linestyle="--", color="#ef4444", linewidth=2, label="Gulosa")
    ax1.plot(b, rm, marker="^", linestyle=":", color="#6b7280", linewidth=2, label="Aleatória (média)")
    ax1.fill_between(b, rm - rs, rm + rs, color="#6b7280", alpha=0.15, label="Aleatória (±1 DP)")
    ax1.set_title("(a) Redução média de LST nos pixels plantados", fontweight="bold")
    ax1.set_xlabel("Orçamento (pixels de 900 m²)")
    ax1.set_ylabel("ΔT (°C)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(frameon=True)
    if "multi_area_beneficiada_ha" in c[0]:
        ax2.plot(b, [r["multi_area_beneficiada_ha"] for r in c], marker="o", color="#10b981", linewidth=2.5, label="Multicritério (proposto)")
        ax2.plot(b, [r["gulosa_area_beneficiada_ha"] for r in c], marker="s", linestyle="--", color="#ef4444", linewidth=2, label="Gulosa")
        ax2.plot(b, [r["aleatoria_area_beneficiada_ha"] for r in c], marker="^", linestyle=":", color="#6b7280", linewidth=2, label="Aleatória (1 sorteio)")
        ax2.set_title("(b) Área com resfriamento > 0,1 °C (plantio + vizinhança)", fontweight="bold")
        ax2.set_xlabel("Orçamento (pixels de 900 m²)")
        ax2.set_ylabel("Área beneficiada (ha)")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(frameon=True)
    else:
        ax2.set_axis_off()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_summary(res, out_dir, prefix="teresina"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "meta": res["meta"],
        "curva_orcamento": res["curve"],
        "corredores": [{k: v for k, v in c.items() if k != "mask"} for c in res["corridors"]],
    }
    (out_dir / f"{prefix}_simulacao.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
