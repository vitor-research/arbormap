"""
Resíduos e plantio com o modelo híbrido

Origem: notebook arbormap.ipynb, célula 30 (Google Colab).
Análise integrada por cidade.
Código preservado como no notebook; caminhos do Colab/Drive devem ser ajustados nas CONFIGURAÇÕES.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

def analisar_residuos_e_plantio_hibrido(model_path, data_root, city_name):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Aplicar o Filtro de Nuvens específico para a cidade escolhida
    city_path = Path(data_root) / city_name
    all_city_files = list(city_path.glob("*.npy"))

    print(f"\n🔍 Analisando {city_name}...")
    valid_files_set = filter_cloudy_files_runtime(all_city_files, temp_threshold=15.0, min_cloud_pct=30.0)

    if len(valid_files_set) < 2:
        print(f"❌ Imagens válidas insuficientes para {city_name} após o filtro de nuvens.")
        return

    # 2. Computar estatísticas locais
    # Como o novo torch.save guarda apenas os pesos, recalculamos as médias para a cidade
    print("📊 Calculando estatísticas de normalização locais...")
    mean_city, std_city = compute_dataset_statistics(valid_files_set)
    global_stats = (mean_city, std_city)

    # 3. Carregar o Modelo Híbrido
    model = HybridUrbanHeatNet().to(DEVICE)
    # O novo script salva apenas o state_dict diretamente
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()

    # 4. Carregar o Dataset
    dataset = UrbanHeatDataset(
        data_root,
        allowed_cities=[city_name],
        valid_files_set=valid_files_set,
        stats=global_stats,
        is_train=False
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # O novo dataset retorna 3 elementos
    seq, target, valid_mask = next(iter(loader))
    seq, target, valid_mask = seq.to(DEVICE), target.to(DEVICE), valid_mask.to(DEVICE)

    # 5. Inferência Base
    with torch.no_grad():
        model_input = seq[:, :5, :, :]
        pred_base = model(model_input).squeeze()

    # 6. Otimização com Algoritmo Genético
    print(f"🌲 Iniciando simulação de plantio com Algoritmo Genético...")
    simulator = GeneticPlantingSimulator(model, DEVICE, global_stats)

    opt_y, opt_x, best_fitness = simulator.optimize_genetic(
        seq, num_trees=50, pop_size=30, generations=50
    )

    # 7. Inferência com Intervenção (Cenário Pós-Plantio)
    with torch.no_grad():
        y_tensor = torch.tensor(opt_y, device=DEVICE)
        x_tensor = torch.tensor(opt_x, device=DEVICE)
        seq_interv = simulator.apply_intervention(seq, y_tensor, x_tensor)
        pred_interv = model(seq_interv[:, :5, :, :]).squeeze()

    # 8. Conversão (O modelo prediz a Anomalia Espacial Normalizada Z-Score)
    real_anomaly = target.squeeze().cpu().numpy()
    pred_anomaly = pred_base.cpu().numpy()
    pred_interv_anomaly = pred_interv.cpu().numpy()

    residuos = real_anomaly - pred_anomaly
    cooling_map = pred_anomaly - pred_interv_anomaly # Valores positivos = resfriamento

    # Preparar a entrada para visualização (foco no NDVI - Canal 0)
    img_input = seq.squeeze().cpu().permute(1, 2, 0).numpy()
    ndvi_map = img_input[:, :, 0]

    # Limites para escala de cores justas ignorando outliers
    v_min = np.percentile(real_anomaly, 2)
    v_max = np.percentile(real_anomaly, 98)

    # 9. Plotagem em Grade 2x3
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    axes = axes.flatten()

    # Painel 1: Entrada (NDVI)
    axes[0].imshow(ndvi_map, cmap='Greens')
    axes[0].set_title("NDVI Original (Vegetação Atual)")
    axes[0].axis('off')

    # Painel 2: Real
    im1 = axes[1].imshow(real_anomaly, cmap='coolwarm_r', vmin=v_min, vmax=v_max)
    axes[1].set_title("Real (Anomalia Z-Score)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].axis('off')

    # Painel 3: Predição Base
    im2 = axes[2].imshow(pred_anomaly, cmap='coolwarm_r', vmin=v_min, vmax=v_max)
    axes[2].set_title("Predição da HybridNet")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    axes[2].axis('off')

    # Painel 4: Erro (Residual)
    im3 = axes[3].imshow(residuos, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    axes[3].set_title("Erro Espacial (Real - Predito)")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    axes[3].axis('off')

    # Painel 5: Locais de Plantio Otimizados
    axes[4].imshow(ndvi_map, cmap='Greens')
    axes[4].scatter(opt_x, opt_y, c='red', s=40, marker='X', label='Árvores Sugeridas')
    axes[4].set_title(f"Locais de Plantio ({len(opt_x)} Árvores)")
    axes[4].legend(loc="upper right")
    axes[4].axis('off')

    # Painel 6: Mapa de Resfriamento
    v_cool_max = max(np.max(cooling_map), 0.01)
    im5 = axes[5].imshow(cooling_map, cmap='Blues', vmin=0, vmax=v_cool_max)
    axes[5].set_title("Impacto do Plantio (Redução da Anomalia)")
    plt.colorbar(im5, ax=axes[5], fraction=0.046, pad=0.04)
    axes[5].axis('off')

    plt.tight_layout()
    plt.show()

    # Estatísticas Críticas considerando a valid_mask
    valid_mask_np = valid_mask.squeeze().cpu().numpy()
    mae = np.sum(np.abs(residuos) * valid_mask_np) / (np.sum(valid_mask_np) + 1e-8)
    bias = np.sum(residuos * valid_mask_np) / (np.sum(valid_mask_np) + 1e-8)
    resfriamento_total = cooling_map[cooling_map > 0].sum()

    print(f"\n📊 Relatório de Desempenho para {city_name}:")
    print(f"MAE (Erro Médio Absoluto em área válida): {mae:.4f}")
    print(f"BIAS (Viés Médio em área válida): {bias:.4f}")
    print(f"🌡️ Resfriamento Total Estimado (Z-Score): {resfriamento_total:.2f}")

if __name__ == "__main__":
    analisar_residuos_e_plantio_hibrido(
        model_path="best_hybrid_fold_0.pt",
        data_root="/content/drive/MyDrive/heat_dataset_v3",
        city_name="Teresina" # Teste com outra cidade se quiser
    )
