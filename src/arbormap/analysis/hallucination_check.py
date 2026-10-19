"""
Análise de alucinação e correlação física

Origem: notebook arbormap.ipynb, célula 26 (Google Colab).
Verifica coerência entre predições e índices físicos no fold de validação.
Código preservado como no notebook; caminhos do Colab/Drive devem ser ajustados nas CONFIGURAÇÕES.
"""

import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

# Certifique-se de importar suas classes e variáveis do script principal!
# from seu_script_principal import UrbanHeatNet, UrbanHeatDataset, K_FOLDS, SEED

def get_fold_data(root, fold_idx, all_cities, global_stats):
    """Reconstrói apenas o dataset de validação necessário para o fold específico."""
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

    splits = list(kf.split(all_cities))
    _, val_idx = splits[fold_idx]

    val_cities = [all_cities[i] for i in val_idx]
    # CORREÇÃO: Adicionado is_train=False para desligar o Augmentation na avaliação
    return UrbanHeatDataset(root, allowed_cities=val_cities, stats=global_stats, is_train=False)

def analisar_alucinacao(model_path, fold_idx, data_root):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)

    model = UrbanHeatNet().to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_cities = [d.name for d in Path(data_root).iterdir() if d.is_dir() and not d.name.startswith('.')]
    global_stats = (checkpoint['mean'], checkpoint['std'])

    # Extrai de forma segura o escalar da temperatura para estimar Celsius
    # O LST é o índice 0 na array original de estatísticas
    std_lst = float(checkpoint['std'][0].flatten()[0])

    val_ds = get_fold_data(data_root, fold_idx, all_cities, global_stats)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=True)

    print(f"✅ Analisando modelo do Fold {fold_idx}...")

    seq, _ = next(iter(val_loader))
    seq = seq.to(DEVICE)

    with torch.no_grad():
        # AQUI ESTÁ A MUDANÇA: Cortamos o canal 5 (NDWI) e passamos apenas 0 a 4
        # model_input = seq[:, :CH_NDWI, :, :]
        model_input = seq

    # with torch.no_grad():
        pred_base = model(model_input)

        # CORREÇÃO: Como o LST foi expulso da entrada, não há risco de vazar ruído
        # de temperatura na entrada. O tensor 'seq' tem 6 canais espaciais.
        noise = torch.randn_like(model_input) * 0.01
        pred_noisy = model(model_input + noise)

        diff_celsius = torch.abs(pred_base - pred_noisy).mean().item() * std_lst

    print(f"📊 Estabilidade: {diff_celsius:.6f} °C (variação média na predição sob ruído leve)")

    plot_physical_correlation(model, val_loader, DEVICE, std_lst)

def plot_physical_correlation(model, loader, device, std_lst):
    all_ndvi, all_preds = [], []

    with torch.no_grad():
        for seq, _ in loader:
            model_input = seq[:, :5, :, :]
            model_input = model_input.to(device)
            pred = model(model_input)

            # Desnormaliza a Anomalia prevista para escala aproximada (°C)
            pred_celsius = pred * std_lst

            # CORREÇÃO: A arquitetura 2D não tem dimensão de tempo.
            # O tensor 'seq' é (Batch, 6, H, W) onde NDVI agora é o canal 0.
            ndvi = seq[:, 0, :, :].cpu().numpy().flatten()
            pred_flat = pred_celsius.cpu().numpy().flatten()

            all_ndvi.extend(ndvi)
            all_preds.extend(pred_flat)
            if len(all_ndvi) > 5000:
                break

    plt.figure(figsize=(8, 5))
    plt.scatter(all_ndvi, all_preds, alpha=0.1, s=1, color='red')
    plt.xlabel("NDVI (Vegetação, normalizado Z-Score)")
    plt.ylabel("Anomalia de Temperatura Prevista (°C)")
    plt.title("Relação Física Universal: NDVI x Anomalia Térmica")
    plt.grid(True)
    plt.show()

# --- EXECUÇÃO ---
if __name__ == "__main__":
    analisar_alucinacao(
        model_path="best_model_fold_0.pt",
        fold_idx=0,
        data_root=os.environ.get("ARBORMAP_DATA_DIR", "./data")
    )
