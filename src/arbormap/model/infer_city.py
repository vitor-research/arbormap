"""
Inferência por cidade com o modelo de um fold

Origem: notebook arbormap.ipynb, célula 17 (Google Colab).
Lista as cidades de validação do fold e gera predições.
Código preservado como no notebook; caminhos do Colab/Drive devem ser ajustados nas CONFIGURAÇÕES.
"""

import torch
import os
import numpy as np
from pathlib import Path

# ============================================================
# CONFIGURAÇÕES
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "best_model_fold_0.pt"     # altere para o fold desejado
DATA_DIR = os.environ.get("ARBORMAP_DATA_DIR", "./data")

# As mesmas cidades usadas como validação no fold
VAL_CITIES = [
    # coloque aqui as cidades daquele fold
    "Teresina",
    "Parnaiba",
    "Pedro_II",
    "Juazeiro_do_Norte_CE",
    "Imperatriz_MA"
]

# ============================================================
# IMPORTA O MODELO
# ============================================================

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

model = UrbanHeatNet().to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Modelo carregado!")

# ============================================================
# CALCULA O DESVIO PADRÃO DAS ANOMALIAS
# ============================================================

stds = []

for city in VAL_CITIES:

    folder = Path(DATA_DIR) / city

    files = sorted(folder.glob("*.npy"))

    for file in files:

        img = np.load(file).astype(np.float32)

        img = ensure_channels_first(img)[CHANNELS_TO_KEEP]

        lst = img[CH_LST]

        lst = np.clip(lst, LST_CLIP_MIN, LST_CLIP_MAX)

        macro = lst.mean()

        anomaly = lst - macro

        std = anomaly.std()

        stds.append(std)

stds = np.array(stds)

print(f"Número de imagens : {len(stds)}")
print(f"Desvio médio       : {stds.mean():.4f} °C")
print(f"Desvio mediano     : {np.median(stds):.4f} °C")
print(f"Desvio mínimo      : {stds.min():.4f} °C")
print(f"Desvio máximo      : {stds.max():.4f} °C")

# ============================================================
# CONVERSÃO
# ============================================================

MAE_NORMALIZADO = 0.44

mae_real = MAE_NORMALIZADO * stds.mean()

print()
print("="*40)
print(f"MAE normalizado : {MAE_NORMALIZADO:.4f}")
print(f"MAE em LST      : {mae_real:.4f} °C")
print("="*40)
