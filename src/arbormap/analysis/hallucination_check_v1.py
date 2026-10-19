"""
Análise de alucinação (versão anterior)

Origem: notebook arbormap.ipynb, célula 13 (Google Colab).
Mantida para rastreabilidade.
Código preservado como no notebook; caminhos do Colab/Drive devem ser ajustados nas CONFIGURAÇÕES.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

def analisar_alucinacao(model_path, fold_idx, data_root, all_cities, valid_files_set, global_stats):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Extrai média e desvio padrão reais da Banda LST (índice 4)
    mean_lst = global_stats[0][4].item()
    std_lst = global_stats[1][4].item()

    # 1. Recuperar o split exato do fold usando a mesma semente do treino (SEED=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(kf.split(all_cities))
    _, val_idx = splits[fold_idx]

    val_cities = [all_cities[i] for i in val_idx]

    # 2. Criação do Dataset e Loader de Validação
    val_ds = UrbanHeatDataset(
        data_root,
        allowed_cities=val_cities,
        valid_files_set=valid_files_set,
        stats=global_stats,
        is_train=False
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=True)

    # 3. Carregar o Modelo Híbrido (agora com 4 canais fixos)
    model = HybridUrbanHeatNet(in_channels=4).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()

    print(f"\n✅ Analisando modelo do Fold {fold_idx} ({model_path})...")

    # 4. Teste de Estabilidade (Alucinação via Ruído)
    seq, target, valid = next(iter(val_loader))
    seq, valid = seq.to(DEVICE), valid.to(DEVICE)

    with torch.no_grad():
        pred_base = model(seq)

        # Injeta ruído leve (1%) nos canais de entrada
        noise = torch.randn_like(seq) * 0.01
        pred_noisy = model(seq + noise)

        # Calcula a diferença absoluta ignorando o padding (valid_mask)
        valid_count = valid.sum() + 1e-8
        diff_abs_norm = (torch.abs(pred_base - pred_noisy) * valid).sum() / valid_count

        # Multiplica pelo desvio padrão global para obter o erro real em °C
        diff_celsius = diff_abs_norm.item() * std_lst

    print(f"📊 Estabilidade: Variação média de {diff_celsius:.6f} °C sob ruído leve")

    # 5. Plotagem da Física Aprendida
    plot_physical_correlation(model, val_loader, DEVICE, mean_lst, std_lst)

def plot_physical_correlation(model, loader, device, mean_lst, std_lst):
    all_ndvi, all_preds = [], []

    with torch.no_grad():
        for seq, target, valid in loader:
            seq, valid = seq.to(device), valid.to(device)
            pred = model(seq)

            # Extrai o NDVI (Canal 0 na construção do out_seq)
            ndvi = seq[:, 0, :, :].squeeze()

            # Desfaz a normalização da predição para graus Celsius reais!
            pred_celsius = (pred.squeeze() * std_lst) + mean_lst
            valid_mask = valid.squeeze().bool()

            # Pega APENAS os pixels válidos, ignorando os zeros do padding
            ndvi_valid = ndvi[valid_mask].cpu().numpy()
            pred_valid = pred_celsius[valid_mask].cpu().numpy()

            # Sub-amostragem: Pega 1 pixel a cada 50 para evitar sobrecarga de memória
            all_ndvi.extend(ndvi_valid[::50])
            all_preds.extend(pred_valid[::50])

            # Limita a 10.000 pontos totais para o gráfico renderizar rápido
            if len(all_ndvi) > 10000:
                break

    plt.figure(figsize=(8, 5))
    plt.scatter(all_ndvi, all_preds, alpha=0.15, s=4, color='red')
    plt.xlabel("NDVI (Padronizado via Z-score)")
    plt.ylabel("Temperatura Prevista (°C)")
    plt.title("Relação Física: NDVI vs LST Prevista")

    # Adiciona linha de tendência
    z = np.polyfit(all_ndvi, all_preds, 1)
    p = np.poly1d(z)
    plt.plot(all_ndvi, p(all_ndvi), "k--", alpha=0.8, label=f"Tendência: y={z[0]:.2f}x + {z[1]:.2f}")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- COMO CHAMAR ---
# Cole isso no final do seu script principal, fora do laço do K-Fold
if __name__ == "__main__":
    # Garante que as variáveis já existam no ambiente de execução
    try:
        analisar_alucinacao(
            model_path="best_hybrid_fold_0.pt", # Nome do arquivo do modelo salvo
            fold_idx=0,
            data_root=DATA_DIR,
            all_cities=all_cities,
            valid_files_set=valid_files_set,
            global_stats=global_stats
        )
    except NameError:
        print("⚠️ Variáveis globais não encontradas. Execute o treino/carregamento primeiro.")
