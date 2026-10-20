"""
Verificação independente dos resultados da UrbanHeatNet (v1 e v2)

Responde, com a mesma régua para todos os modelos:
  1. Quais canais entram na rede (e que a LST não está entre eles).
  2. Se o "MAE em °C" é mesmo em °C: compara a conversão usada no treino (média e desvio-
     padrão da própria cena-alvo, desconhecidos no momento da previsão) com uma conversão
     realista (média e desvio-padrão da cena-base, a cena de entrada).
  3. Se o modelo supera referências simples: prever a média da cena; regressão linear
     sobre os mesmos 4 índices; e persistência (LST da cena anterior, só como referência,
     porque usa LST na entrada).
Todas as métricas são calculadas nos mesmos pixels: dentro do recorte e com LST observada
> 1 °C nas duas cenas (sem as lacunas de nuvem gravadas como 0).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from arbormap.model.hybrid_pipeline import (
    CH_LST,
    CHANNELS_TO_KEEP,
    LST_CLIP_MAX,
    LST_CLIP_MIN,
    HybridUrbanHeatNet,
    ensure_channels_first,
)
from arbormap.model.train_urbanheatnet import UrbanHeatDatasetMasked

CROP = 400
NOMES_ENTRADA = ["NDVI", "NDBI", "DEM", "NDWI"]   # ordem montada em UrbanHeatDataset.__getitem__


def _load_model(ckpt, device):
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model = HybridUrbanHeatNet(in_channels=4, base_c=state.get("base_c", 64)).to(device).eval()
    model.load_state_dict(state["state_dict"])
    return model, state


def _raw_lst_crop(path, keep_zeros=False):
    """LST bruta com o mesmo recorte central do dataset de validação (NaN fora do recorte).

    keep_zeros=False também marca como NaN as lacunas de nuvem gravadas como 0.
    """
    img = ensure_channels_first(np.load(path).astype(np.float32))[CHANNELS_TO_KEEP]
    h, w = img.shape[1:]
    y0, x0 = max(0, (h - CROP) // 2), max(0, (w - CROP) // 2)
    out = np.full((CROP, CROP), np.nan, dtype=np.float32)
    c = np.clip(img[CH_LST, y0:y0 + CROP, x0:x0 + CROP], LST_CLIP_MIN, LST_CLIP_MAX)
    out[:c.shape[0], :c.shape[1]] = c
    if not keep_zeros:
        out[out <= 1.0] = np.nan
    return out


def _scene_stats(path, lmin):
    """Média e desvio-padrão da cena na convenção do treino (v1 incluía os zeros; v2 não)."""
    a = _raw_lst_crop(path, keep_zeros=lmin is None)
    return float(np.nanmean(a)), float(np.nanstd(a))


def _fit_linear(train_ds, max_pairs=200, px_per_pair=2000, seed=0):
    """Regressão linear pixel a pixel: anomalia padronizada ~ 4 índices (mínimos quadrados)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(train_ds), size=min(max_pairs, len(train_ds)), replace=False)
    X, y = [], []
    for i in idx:
        x, t, v, _ = train_ds[int(i)]
        m = (v[0] > 0).numpy()
        ys, xs = np.where(m)
        if ys.size == 0:
            continue
        pick = rng.choice(ys.size, size=min(px_per_pair, ys.size), replace=False)
        X.append(x[:, ys[pick], xs[pick]].numpy().T)
        y.append(t[0, ys[pick], xs[pick]].numpy())
    X = np.concatenate(X)
    X = np.hstack([X, np.ones((X.shape[0], 1), dtype=X.dtype)])
    coef, *_ = np.linalg.lstsq(X, np.concatenate(y), rcond=None)
    return coef


def evaluate_fold(ckpt, data_dir, valid_files_set, device, linear=True):
    model, state = _load_model(ckpt, device)
    stats = (np.asarray(state["mean"]), np.asarray(state["std"]))
    lmin = state.get("lst_min_valida")
    val_ds = UrbanHeatDatasetMasked(data_dir, allowed_cities=state["val_cities"], valid_files_set=valid_files_set,
                                    normalize=True, stats=stats, is_train=False, lst_min_valida=lmin)
    coef = None
    if linear:
        train_ds = UrbanHeatDatasetMasked(data_dir, allowed_cities=state["train_cities"],
                                          valid_files_set=valid_files_set, normalize=True, stats=stats,
                                          is_train=False, lst_min_valida=1.0)
        coef = _fit_linear(train_ds)

    acc = {k: [] for k in ["oraculo", "realista", "media_cena", "linear_oraculo", "persistencia"]}
    corr, n_px = [], 0
    for i in range(len(val_ds)):
        x, _, _, _ = val_ds[i]
        base_f, tgt_f = val_ds.samples[i]
        obs, prev = _raw_lst_crop(tgt_f), _raw_lst_crop(base_f)
        m = np.isfinite(obs) & np.isfinite(prev)
        if m.sum() < 100:
            continue
        with torch.no_grad():
            z = model(x.unsqueeze(0).to(device))[0, 0].cpu().numpy()
        # Cada modelo é convertido para °C na sua própria convenção de treino:
        #   oráculo  = média e DP da cena-alvo (o que o MAE do treino usa; desconhecidos na previsão)
        #   realista = média e DP da cena-base (disponíveis no momento da previsão)
        mu_t, sd_t = _scene_stats(tgt_f, lmin)
        mu_b, sd_b = _scene_stats(base_f, lmin)
        pred_or = z * sd_t + mu_t
        pred_re = z * sd_b + mu_b
        mu_b, sd_b = float(np.nanmean(prev[m])), float(np.nanstd(prev[m]))
        acc["oraculo"].append(np.abs(pred_or - obs)[m].mean())
        acc["realista"].append(np.abs(pred_re - obs)[m].mean())
        acc["media_cena"].append(np.abs(np.nanmean(obs[m]) - obs)[m].mean())
        acc["persistencia"].append(np.abs((prev - mu_b) + np.nanmean(obs[m]) - obs)[m].mean())
        if coef is not None:
            xl = x.numpy().reshape(4, -1).T
            zl = (np.hstack([xl, np.ones((xl.shape[0], 1), dtype=xl.dtype)]) @ coef).reshape(CROP, CROP)
            acc["linear_oraculo"].append(np.abs(zl * np.nanstd(obs[m]) + np.nanmean(obs[m]) - obs)[m].mean())
        obs_z = (obs - np.nanmean(obs[m])) / (np.nanstd(obs[m]) + 1e-6)
        corr.append(float(np.corrcoef(z[m], obs_z[m])[0, 1]))
        n_px += int(m.sum())
    out = {k: float(np.mean(v)) for k, v in acc.items() if v}
    out.update({"pearson_anomalia": float(np.mean(corr)), "n_pares": len(corr), "n_pixels": n_px,
                "fold": state["fold"] + 1, "val_cities": state["val_cities"], "lst_min_valida": lmin,
                "canais_entrada": NOMES_ENTRADA, "shape_entrada": list(val_ds[0][0].shape)})
    return out


def check_inputs(data_dir, valid_files_set, city):
    """Confere em execução que nenhum canal de entrada é a LST da cena-base ou da cena-alvo."""
    ds = UrbanHeatDatasetMasked(data_dir, allowed_cities=[city], valid_files_set=valid_files_set,
                                normalize=False, is_train=False, lst_min_valida=1.0)
    x, _, _, _ = ds[0]
    base_f, tgt_f = ds.samples[0]
    lst_b, lst_t = _raw_lst_crop(base_f), _raw_lst_crop(tgt_f)
    m = np.isfinite(lst_b) & np.isfinite(lst_t)
    rows = []
    for k, nome in enumerate(NOMES_ENTRADA):
        ch = x[k].numpy()
        rows.append({"canal": nome, "corr_com_LST_base": float(np.corrcoef(ch[m], lst_b[m])[0, 1]),
                     "corr_com_LST_alvo": float(np.corrcoef(ch[m], lst_t[m])[0, 1]),
                     "igual_LST_base": bool(np.allclose(ch[m], lst_b[m]))})
    return {"shape_entrada": list(x.shape), "canais": rows,
            "arquivo_base": Path(base_f).name, "arquivo_alvo": Path(tgt_f).name}


def run_verification(out_dirs, data_dir, save_dir, device=None, city_check="Teresina"):
    """Roda check_inputs, evaluate_fold em todos os folds de cada versão e plot_all_folds.

    out_dirs: {"v1": pasta, "v2": pasta}. Grava verificacao.json e curvas_todos_folds.png.
    """
    from arbormap.model.hybrid_pipeline import filter_cloudy_files_runtime
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    vf = filter_cloudy_files_runtime(list(Path(data_dir).rglob("*.npy")))
    entradas = check_inputs(data_dir, vf, city_check)
    print("ENTRADAS", json.dumps(entradas, ensure_ascii=False))
    keys = ["oraculo", "realista", "media_cena", "linear_oraculo", "persistencia", "pearson_anomalia"]
    rows = []
    for lab, d in out_dirs.items():
        for f in range(5):
            ck = Path(d) / f"best_hybrid_fold_{f}.pt"
            if not ck.exists():
                continue
            r = evaluate_fold(ck, data_dir, vf, device)
            r["versao"] = lab
            rows.append(r)
            print(lab, "Fold", f + 1, {k: round(r[k], 3) for k in keys if k in r})
    resumo = {}
    for lab in out_dirs:
        rs = [r for r in rows if r["versao"] == lab]
        if rs:
            resumo[lab] = {k: [round(float(np.mean([r[k] for r in rs])), 3),
                               round(float(np.std([r[k] for r in rs], ddof=1)) if len(rs) > 1 else 0.0, 3)]
                           for k in keys if k in rs[0]}
            print("MEDIA", lab, resumo[lab])
    (save_dir / "verificacao.json").write_text(json.dumps({"entradas": entradas, "folds": rows, "resumo": resumo},
                                                          indent=1, ensure_ascii=False), encoding="utf-8")
    plot_all_folds(out_dirs, save_dir / "curvas_todos_folds.png")
    return resumo


def plot_all_folds(out_dirs, save_path):
    """Curvas de treino e validação por época dos 5 folds, uma linha por versão."""
    labels = list(out_dirs)
    fig, axes = plt.subplots(len(labels), 5, figsize=(22, 4.2 * len(labels)), squeeze=False)
    for r, lab in enumerate(labels):
        for f in range(5):
            p = Path(out_dirs[lab]) / f"fold_{f}.json"
            ax = axes[r, f]
            if not p.exists():
                ax.set_axis_off()
                continue
            fr = json.loads(p.read_text(encoding="utf-8"))
            h = fr["history"]
            ep = [e["epoch"] for e in h]
            ax.plot(ep, [e["train_loss"] for e in h], label="treino")
            ax.plot(ep, [e["val_loss"] for e in h], label="validação")
            ax.axvline(fr["best_epoch"], color="gray", ls=":")
            ax2 = ax.twinx()
            ax2.plot(ep, [e["mae_c"] for e in h], color="#2ca02c", alpha=0.6, label="MAE (°C)")
            ax.set_title(f"{lab} | Fold {f + 1} | melhor época {fr['best_epoch']}", fontsize=9)
            ax.set_xlabel("Época")
            if f == 0:
                ax.set_ylabel("Perda")
                ax.legend(loc="upper right", fontsize=7)
            if f == 4:
                ax2.set_ylabel("MAE (°C)", color="#2ca02c")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
