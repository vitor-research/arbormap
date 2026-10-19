"""
Figuras 2 e 3 a partir do histórico salvo por train_urbanheatnet.py (fold_<k>.json)

Substitui training_curves.py, que tinha os valores copiados à mão de um log anterior.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_fold(out_dir, fold_index):
    return json.loads((Path(out_dir) / f"fold_{fold_index}.json").read_text(encoding="utf-8"))


def figure_training_curves(fold_result, save_path):
    """Figura 2: (a) perda de treino e validação; (b) MAE, Hotspot MAE e IoU na validação."""
    h = fold_result["history"]
    ep = [r["epoch"] for r in h]
    best = fold_result["best_epoch"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(ep, [r["train_loss"] for r in h], marker="o", ms=3, label="Treino")
    ax1.plot(ep, [r["val_loss"] for r in h], marker="s", ms=3, label="Validação")
    ax1.axvline(best, color="gray", linestyle=":", label=f"Melhor época ({best})")
    ax1.set_title("(a) Perda de treino vs. validação", fontweight="bold")
    ax1.set_xlabel("Época"); ax1.set_ylabel("Perda"); ax1.grid(True, linestyle="--", alpha=0.5); ax1.legend()
    ax2.plot(ep, [r["mae_c"] for r in h], marker="o", ms=3, color="#2ca02c", label="MAE (°C)")
    ax2.plot(ep, [r["hotspot_mae_c"] for r in h], marker="^", ms=3, color="#9467bd", label="Hotspot MAE (°C)")
    ax2.plot(ep, [r["iou"] for r in h], marker="d", ms=3, color="#d62728", label="IoU")
    ax2.axvline(best, color="gray", linestyle=":")
    ax2.set_title("(b) Métricas na validação", fontweight="bold")
    ax2.set_xlabel("Época"); ax2.set_ylabel("Valor"); ax2.grid(True, linestyle="--", alpha=0.5); ax2.legend()
    fig.suptitle(f"Fold {fold_result['fold']} | validação: {', '.join(fold_result['val_cities'])}", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def figure_std(fold_result, save_path):
    """Figura 3: desvio-padrão da anomalia prevista vs. do alvo, por época."""
    h = fold_result["history"]
    ep = [r["epoch"] for r in h]
    plt.figure(figsize=(8, 5))
    plt.plot(ep, [r["std_pred"] for r in h], marker="o", color="#d62728", label="DP (predição)")
    plt.plot(ep, [r["std_target"] for r in h], linestyle="--", color="black", linewidth=2, label="DP (alvo)")
    plt.xlabel("Época"); plt.ylabel("Desvio-padrão (anomalia padronizada)")
    plt.title(f"Fold {fold_result['fold']}", fontsize=10)
    plt.grid(True, alpha=0.5); plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
