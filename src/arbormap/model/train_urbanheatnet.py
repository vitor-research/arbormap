"""
Treino da UrbanHeatNet nos cinco folds, com registro completo e retomada automática

Reaproveita, sem alterações, o dataset, a arquitetura (HybridUrbanHeatNet), a perda
(HybridPhysicsLoss: SSIM + L1 + hotspot) e as métricas de hybrid_pipeline.py (célula 32).
Diferenças em relação ao laço da célula 32:
  - treina os cinco folds (sem o `break`);
  - folds definidos explicitamente pela Tabela 1 do artigo, em vez de KFold sobre a ordem
    de Path.iterdir(), que depende do sistema de arquivos;
  - estatísticas de normalização calculadas só com as cidades de treino de cada fold;
  - histórico por época salvo em JSON (inclui o desvio-padrão da predição e do alvo),
    checkpoint do melhor modelo com as estatísticas de normalização e retomada após
    queda da sessão do Colab;
  - Tabela 2 (melhor época por fold, média ± DP amostral) gerada a partir do próprio log.
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from arbormap.model.hybrid_pipeline import (
    HybridPhysicsLoss,
    HybridUrbanHeatNet,
    UrbanHeatDataset,
    compute_metrics,
    filter_cloudy_files_runtime,
)

# ============================================================
# FOLDS (Tabela 1 do artigo: IDs 01-04, 05-08, 09-12, 13-16, 17-20)
# ============================================================
FOLDS_ARTIGO = [
    ["Iguatu_CE", "Parnaiba", "Pedro_II", "Teresina"],
    ["Juazeiro_do_Norte_CE", "Floriano", "Sao_Raimundo_Nonato", "Picos"],
    ["Barreiras_BA", "Imperatriz_MA", "Patos_PB", "Crato_CE"],
    ["Mossoro_RN", "Bom_Jesus", "Urucui", "Araguaina_TO"],
    ["Sobral_CE", "Corrente", "Vitoria_da_Conquista_BA", "Paulo_Afonso_BA"],
]

# Hiperparâmetros da célula 32 (hybrid_pipeline.py)
CONFIG_PADRAO = {
    "epochs": 30,
    "lr": 5e-4,
    "weight_decay": 1e-4,
    "batch_size": 2,
    "patience": 5,
    "base_c": 64,
    "num_workers": 2,
    "seed": 42,
    "hotspot_w": 0.1,
    "temp_threshold": 15.0,
    "min_cloud_pct": 30.0,
}


class _Tee:
    """Duplica o que é impresso para um arquivo de log."""

    def __init__(self, path):
        self.file = open(path, "a", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def _seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _std_pred_target(pred, target, valid):
    """Desvio-padrão (anomalia padronizada) da predição e do alvo nos pixels válidos."""
    m = valid == 1.0
    return pred[m].std().item(), target[m].std().item()


def _train_fold(fold, val_cities, train_cities, data_dir, valid_files_set, out_dir, cfg, device):
    ckpt_best = out_dir / f"best_hybrid_fold_{fold}.pt"
    ckpt_last = out_dir / f"last_fold_{fold}.pt"

    _seed_all(cfg["seed"] + fold)
    train_ds = UrbanHeatDataset(data_dir, allowed_cities=train_cities, valid_files_set=valid_files_set,
                                normalize=True, stats=None, is_train=True)
    stats = (train_ds.mean, train_ds.std)
    val_ds = UrbanHeatDataset(data_dir, allowed_cities=val_cities, valid_files_set=valid_files_set,
                              normalize=True, stats=stats, is_train=False)
    print(f"   pares de treino: {len(train_ds)} | pares de validação: {len(val_ds)}")
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(f"Fold {fold + 1} sem dados de treino ou validação.")

    loader_kw = {"batch_size": cfg["batch_size"], "num_workers": cfg["num_workers"],
                 "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)

    model = HybridUrbanHeatNet(in_channels=4, base_c=cfg["base_c"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                                           patience=cfg["patience"])
    loss_fn = HybridPhysicsLoss(hotspot_w=cfg["hotspot_w"])

    history, best_loss, start_epoch = [], float("inf"), 0
    if ckpt_last.exists():
        state = torch.load(ckpt_last, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history, best_loss, start_epoch = state["history"], state["best_loss"], state["epoch"] + 1
        print(f"   retomando do fim da época {start_epoch} (checkpoint {ckpt_last.name})")

    for epoch in range(start_epoch, cfg["epochs"]):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        for seq, target, valid, std_anomaly in train_loader:
            seq, target, valid = seq.to(device), target.to(device), valid.to(device)
            optimizer.zero_grad()
            pred = model(seq)
            loss = loss_fn(pred, target, seq[:, 0:1, :, :], valid)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = val_mae = val_hs = val_iou = std_p = std_t = 0.0
        with torch.no_grad():
            for seq, target, valid, std_anomaly in val_loader:
                seq, target, valid = seq.to(device), target.to(device), valid.to(device)
                std_anomaly = std_anomaly.to(device)
                pred = model(seq)
                val_loss += loss_fn(pred, target, seq[:, 0:1, :, :], valid).item()
                mae_c, hs_c, iou = compute_metrics(pred, target, valid, std_anomaly)
                val_mae += mae_c
                val_hs += hs_c
                val_iou += iou
                sp, st = _std_pred_target(pred, target, valid)
                std_p += sp
                std_t += st

        n_t, n_v = max(len(train_loader), 1), max(len(val_loader), 1)
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss / n_t,
            "val_loss": val_loss / n_v,
            "mae_c": val_mae / n_v,
            "hotspot_mae_c": val_hs / n_v,
            "iou": val_iou / n_v,
            "std_pred": std_p / n_v,
            "std_target": std_t / n_v,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": round(time.time() - t0, 1),
        }
        history.append(row)
        scheduler.step(row["val_loss"])

        print(f"Fold {fold + 1} | Epoch {row['epoch']:02d} | T.Loss: {row['train_loss']:.4f} | "
              f"V.Loss: {row['val_loss']:.4f} | MAE (°C): {row['mae_c']:.4f} | "
              f"HS MAE (°C): {row['hotspot_mae_c']:.4f} | IoU: {row['iou']:.4f} | "
              f"Std(Pred): {row['std_pred']:.4f} vs Std(Target): {row['std_target']:.4f} | {row['seconds']}s")

        if row["val_loss"] < best_loss:
            best_loss = row["val_loss"]
            torch.save({"state_dict": model.state_dict(), "mean": stats[0], "std": stats[1],
                        "fold": fold, "epoch": row["epoch"], "val_cities": val_cities,
                        "train_cities": train_cities, "base_c": cfg["base_c"]}, ckpt_best)

        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "history": history,
                    "best_loss": best_loss, "epoch": epoch}, ckpt_last)

    best = min(history, key=lambda r: r["val_loss"])
    result = {"fold": fold + 1, "val_cities": val_cities, "train_cities": train_cities,
              "n_train_pairs": len(train_ds), "n_val_pairs": len(val_ds),
              "best_epoch": best["epoch"], "best": best, "history": history,
              "norm_mean": [float(v) for v in stats[0]], "norm_std": [float(v) for v in stats[1]]}
    (out_dir / f"fold_{fold}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if ckpt_last.exists():
        ckpt_last.unlink()
    return result


def summarize(results, out_dir):
    """Tabela 2: melhor época (menor V.Loss) de cada fold, média ± desvio-padrão amostral."""
    keys = [("val_loss", "Val. Loss"), ("mae_c", "MAE (°C)"), ("hotspot_mae_c", "Hotspot MAE (°C)"), ("iou", "IoU")]
    lines = ["Fold;Cidades de validação;Melhor época;" + ";".join(k[1] for k in keys)]
    for r in results:
        lines.append(f"{r['fold']};{', '.join(r['val_cities'])};{r['best_epoch']};"
                     + ";".join(f"{r['best'][k]:.4f}" for k, _ in keys))
    summary = {}
    for k, label in keys:
        vals = np.array([r["best"][k] for r in results], dtype=float)
        sd = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        summary[k] = {"media": float(vals.mean()), "dp_amostral": sd}
    lines.append("Média ± DP;todas;-;" + ";".join(
        f"{summary[k]['media']:.3f} ± {summary[k]['dp_amostral']:.3f}" for k, _ in keys))
    (out_dir / "tabela2.csv").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    (out_dir / "resultados_kfold.json").write_text(
        json.dumps({"folds": [{k: v for k, v in r.items() if k != "history"} for r in results],
                    "resumo": summary}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + "=" * 70 + "\nTABELA 2 (melhor época por fold)\n" + "=" * 70)
    for line in lines:
        print(line.replace(";", " | "))
    return summary


def run_kfold(data_dir, out_dir, folds=None, config=None, device=None, only_folds=None):
    """Treina os folds pendentes e devolve a lista de resultados de todos os folds concluídos.

    data_dir: pasta com uma subpasta por cidade, cada uma com os .npy mensais.
    out_dir: pasta de saída (use uma pasta do Drive para sobreviver a quedas da sessão).
    only_folds: lista opcional de índices (0 a 4) para treinar só alguns folds.
    """
    cfg = {**CONFIG_PADRAO, **(config or {})}
    folds = folds or FOLDS_ARTIGO
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tee = _Tee(out_dir / "treino_log.txt")
    sys.stdout = tee
    try:
        print(f"\n### Início: {time.strftime('%Y-%m-%d %H:%M:%S')} | dispositivo: {device}")
        print("Configuração:", json.dumps(cfg, ensure_ascii=False))

        existing = sorted(d.name for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith("."))
        expected = sorted(c for f in folds for c in f)
        missing = sorted(set(expected) - set(existing))
        extra = sorted(set(existing) - set(expected))
        if missing:
            raise FileNotFoundError(f"Pastas de cidade ausentes em {data_dir}: {missing}")
        if extra:
            print(f"Aviso: pastas fora dos folds (ignoradas): {extra}")

        all_files = [f for c in expected for f in (data_dir / c).glob("*.npy")]
        valid_files_set = filter_cloudy_files_runtime(all_files, temp_threshold=cfg["temp_threshold"],
                                                      min_cloud_pct=cfg["min_cloud_pct"])
        (out_dir / "config.json").write_text(json.dumps(
            {"config": cfg, "folds": folds, "n_arquivos": len(all_files),
             "n_validos": len(valid_files_set)}, indent=2, ensure_ascii=False), encoding="utf-8")

        results = []
        for fold, val_cities in enumerate(folds):
            done = out_dir / f"fold_{fold}.json"
            if done.exists():
                print(f"\n--- FOLD {fold + 1}: já concluído ({done.name}), pulando ---")
                results.append(json.loads(done.read_text(encoding="utf-8")))
                continue
            if only_folds is not None and fold not in only_folds:
                continue
            train_cities = [c for f in folds if f is not val_cities for c in f]
            print(f"\n--- Iniciando FOLD {fold + 1} | validação: {', '.join(val_cities)} ---")
            results.append(_train_fold(fold, val_cities, train_cities, data_dir, valid_files_set,
                                       out_dir, cfg, device))

        if len(results) == len(folds):
            summarize(results, out_dir)
        else:
            print(f"\n{len(results)} de {len(folds)} folds concluídos. Rode de novo para continuar.")
        return results
    finally:
        sys.stdout = tee.stdout
        tee.file.close()
