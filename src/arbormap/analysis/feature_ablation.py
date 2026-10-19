"""
Ablação de canais de entrada (Fold 0)

Origem: notebook arbormap.ipynb, célula 24 (Google Colab).
Mede a contribuição de cada canal físico.
Código preservado como no notebook; caminhos do Colab/Drive devem ser ajustados nas CONFIGURAÇÕES.
"""

import torch
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

# ============================================================
# FUNÇÃO DE FEATURE ABLATION
# ============================================================
def perform_feature_ablation(model, val_loader, device):
    """
    Realiza a ablação de características zerando um canal por vez e
    medindo a degradação na performance (MAE e R2).
    """
    print("\n" + "="*60)
    print("🔬 INICIANDO FEATURE ABLATION (FOLD 0)")
    print("="*60)

    model.eval()

    # Nomes dos 6 canais na ordem exata em que entram no modelo
    feature_names = ['NDVI', 'NDBI', 'NDWI', 'DEM', 'Mes_Sin', 'Mes_Cos']
    num_features = len(feature_names)

    def evaluate_loader(loader, ablate_idx=None):
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for seq, target in loader:
                seq = seq.to(device)
                target = target.to(device)

                # Se ablate_idx for passado, zeramos o canal correspondente
                # Como os dados numéricos estão em Z-score, a média é 0.
                # Para seno e cosseno, a média anual também é 0.
                if ablate_idx is not None:
                    seq[:, ablate_idx, :, :] = 0.0

                with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                    pred = model(seq)

                all_preds.append(pred.detach().cpu())
                all_targets.append(target.detach().cpu())

        all_preds_t = torch.cat(all_preds)
        all_targets_t = torch.cat(all_targets)

        # Reaproveita as métricas já definidas no seu código
        mae = compute_mae(all_preds_t, all_targets_t).item()
        r2 = compute_r2(all_preds_t, all_targets_t).item()
        return mae, r2

    # 1. Avalia o Baseline (Todos os canais ativos)
    print("Calculando Baseline (Todos os features)...")
    baseline_mae, baseline_r2 = evaluate_loader(val_loader, ablate_idx=None)
    print(f"Baseline -> MAE: {baseline_mae:.4f} | R2: {baseline_r2:.4f}\n")

    # 2. Avalia a ablação para cada canal
    results = []
    for idx, feature in enumerate(feature_names):
        print(f"Ablacionando canal {idx}: {feature}...")
        mae, r2 = evaluate_loader(val_loader, ablate_idx=idx)

        # Calcula a degradação (quanto piorou o modelo sem essa feature)
        # Quedas maiores no R2 ou aumentos no MAE indicam maior importância
        r2_drop = baseline_r2 - r2
        mae_increase = mae - baseline_mae

        results.append({
            'feature': feature,
            'mae': mae,
            'r2': r2,
            'r2_drop': r2_drop,
            'mae_increase': mae_increase
        })

    # 3. Exibe o Ranking de Importância
    print("\n" + "-"*60)
    print("🏆 RANKING DE IMPORTÂNCIA DOS RECURSOS (Por Queda de R2)")
    print("-"*60)

    # Ordena os resultados pela maior queda no R2 (do mais importante para o menos)
    results_sorted = sorted(results, key=lambda x: x['r2_drop'], reverse=True)

    for rank, res in enumerate(results_sorted, 1):
        print(f"{rank}. {res['feature'].ljust(10)} | "
              f"R2 Após Ablação: {res['r2']:.4f} (Queda de {res['r2_drop']:.4f}) | "
              f"MAE Aumento: +{res['mae_increase']:.4f}")

# ============================================================
# SCRIPT DE CARREGAMENTO (EXECUÇÃO)
# ============================================================
if __name__ == "__main__":
    # Garantir a mesma seed para obter o mesmo Fold 0 do treinamento
    SEED = 42
    np.random.seed(SEED)

    all_cities = [d.name for d in Path(DATA_DIR).iterdir() if d.is_dir() and not d.name.startswith('.')]

    if len(all_cities) < K_FOLDS:
        print("Cidades insuficientes no diretório de dados.")
        exit()

    # Recriar as partições exatas do Fold 0
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    train_idx, val_idx = next(kf.split(all_cities)) # Pega apenas a primeira divisão (Fold 0)

    train_cities = [all_cities[i] for i in train_idx]
    val_cities = [all_cities[i] for i in val_idx]

    # Carregar o checkpoint do modelo do Fold 0
    model_path = "./best_model_fold_0.pt"
    if not Path(model_path).exists():
        print(f"Erro: Arquivo '{model_path}' não encontrado. Certifique-se de que o fold 0 concluiu o treinamento.")
        exit()

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)

    # Extrair as estatísticas de normalização que foram salvas no checkpoint
    saved_stats = (checkpoint['mean'], checkpoint['std'])

    # Instanciar o Dataset de Validação com as cidades e stats corretos
    val_ds = UrbanHeatDataset(
        DATA_DIR,
        allowed_cities=val_cities,
        stats=saved_stats,
        is_train=False
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Inicializar o Modelo e carregar os pesos
    model = UrbanHeatNet(base_channels=64).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Executar a análise
    perform_feature_ablation(model, val_loader, DEVICE)
