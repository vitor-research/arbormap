"""
Resíduos espaciais por cidade

Origem: notebook arbormap.ipynb, célula 28 (Google Colab).
Mapas de erro absoluto (base da Figura 4).
Código preservado como no notebook; caminhos do Colab/Drive devem ser ajustados nas CONFIGURAÇÕES.
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

def analisar_residuos_espaciais(model_path, data_root, city_name):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Carregar Modelo e Stats
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model = UrbanHeatNet().to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    std_lst = float(checkpoint['std'][0].flatten()[0])

    # 2. Carregar a cidade escolhida
    dataset = UrbanHeatDataset(data_root, allowed_cities=[city_name],
                               stats=(checkpoint['mean'], checkpoint['std']), is_train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    seq, target = next(iter(loader))
    seq, target = seq.to(DEVICE), target.to(DEVICE)

    with torch.no_grad():
        # AQUI ESTÁ A MUDANÇA: Cortamos o canal 5 (NDWI) e passamos apenas 0 a 4
        model_input = seq[:, :CH_NDWI, :, :]
        pred = model(model_input).squeeze()

    # 3. Conversão para Celsius
    real_celsius = target.squeeze().cpu().numpy() * std_lst
    pred_celsius = pred.cpu().numpy() * std_lst
    residuos = real_celsius - pred_celsius

    # Preparar a entrada (seq) para visualização:
    # O seq tem shape (1, 6, 400, 400). Transformamos para (400, 400, 6)
    img_input = seq.squeeze().cpu().permute(1, 2, 0).numpy()

    # --- CÁLCULO DE PERCENTIL PARA ESCALA JUSTA ---
    v_min = np.percentile(real_celsius, 2)
    v_max = np.percentile(real_celsius, 98)

    # 4. Plotagem em 5 painéis (Adicionado o Painel de Entrada)
    fig, axes = plt.subplots(1, 5, figsize=(30, 6))

    # Painel 1: Entrada (Selecionando apenas o canal 0 para visualização)
    axes[0].imshow(img_input[:, :, 0], cmap='gray')
    axes[0].set_title("Entrada (Canal 0)")
    axes[0].axis('off')

    # Painel 2: Real
    im1 = axes[1].imshow(real_celsius, cmap='coolwarm_r', vmin=v_min, vmax=v_max)
    axes[1].set_title("Real (Ground Truth) °C")
    plt.colorbar(im1, ax=axes[1])

    # Painel 3: Predição
    im2 = axes[2].imshow(pred_celsius, cmap='coolwarm_r', vmin=v_min, vmax=v_max)
    axes[2].set_title("Predição do Modelo (°C)")
    plt.colorbar(im2, ax=axes[2])

    # Painel 4: Erro (Residual)
    im3 = axes[3].imshow(residuos, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    axes[3].set_title("Erro Espacial (Real - Predito)")
    plt.colorbar(im3, ax=axes[3])

    # Painel 5: Histograma
    axes[4].hist(residuos.flatten(), bins=60, color='darkblue', alpha=0.7)
    axes[4].set_title("Distribuição do Erro")
    axes[4].set_xlim(-1.0, 1.0)

    plt.tight_layout()
    plt.show()

    # Estatísticas Críticas
    mae = np.mean(np.abs(residuos))
    bias = np.mean(residuos)
    print(f"📊 Relatório de Erro para {city_name}:")
    print(f"MAE (Erro Médio Absoluto): {mae:.4f} °C")
    print(f"BIAS (Viés Médio): {bias:.4f} °C")

if __name__ == "__main__":
    analisar_residuos_espaciais(
        model_path="best_model_fold_0.pt",
        data_root=os.environ.get("ARBORMAP_DATA_DIR", "./data"),
        city_name="Jose_de_Freitas"
    )
