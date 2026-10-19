"""
Driver do K-Fold (GroupKFold / KFold por cidade)

Origem: notebook arbormap.ipynb, célula 14 (Google Colab).
Laço de treino por fold com Grad-CAM do melhor modelo.
Código preservado como no notebook; caminhos do Colab/Drive devem ser ajustados nas CONFIGURAÇÕES.
"""

import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold

# Certifique-se de importar suas classes e funções atualizadas aqui!
# from seu_script import AttentionUNetLST, RegressionGradCAM, UrbanHeatDataset, filter_cloudy_files_runtime, compute_min_max, plot_grad_cam, set_seed

all_cities = [d.name for d in Path(DATA_DIR).iterdir() if d.is_dir() and not d.name.startswith('.')]
if len(all_cities) < K_FOLDS: exit("Cidades insuficientes para K-Fold.")

# 1. Aplicando Filtro Global de Nuvens
all_files_raw = list(Path(DATA_DIR).rglob("*.npy"))
valid_files_set = filter_cloudy_files_runtime(all_files_raw, temp_threshold=15.0, min_cloud_pct=30.0)

if not valid_files_set: exit("Nenhum arquivo válido após o filtro de nuvens!")

print("1. Extraindo Estatísticas Globais (Apenas Arquivos Válidos)...")
base_ds = UrbanHeatDataset(DATA_DIR, allowed_cities=all_cities, valid_files_set=valid_files_set, normalize=True, is_train=False)
global_stats = (base_ds.mean, base_ds.std)

kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

print("\n2. Iniciando K-Fold Cross Validation Espacial...")
for fold, (train_idx, val_idx) in enumerate(kf.split(all_cities)):
    train_cities = [all_cities[i] for i in train_idx]
    val_cities = [all_cities[i] for i in val_idx]

    train_ds = UrbanHeatDataset(DATA_DIR, allowed_cities=train_cities, valid_files_set=valid_files_set, stats=global_stats, is_train=True)
    val_ds = UrbanHeatDataset(DATA_DIR, allowed_cities=val_cities, valid_files_set=valid_files_set, stats=global_stats, is_train=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = HybridUrbanHeatNet().to(DEVICE)

    # GERAR GRAD-CAM DO MELHOR MODELO NO FOLD (Com Correção de Dimensões)
    model.load_state_dict(torch.load(f"best_hybrid_fold_{fold}.pt", weights_only=True))
    cam_generator = RegressionGradCAM(model, target_layer=model.up1.conv.conv2)
    sample_seq, sample_tgt, _ = next(iter(val_loader))
    # CORREÇÃO APLICADA: Pega o antepenúltimo e o penúltimo do lote, mantendo os 5 canais
    cam_map, pred_np = cam_generator.generate(sample_seq[0:1, :5, :, :].to(DEVICE))
    plot_grad_cam(sample_seq[0:1], cam_map, pred_np, sample_tgt[0, 0].numpy(), save_path=f"gradcam_fold_{fold}.png")
    print(f"-> Imagem do Grad-CAM salva como gradcam_fold_{fold}.png")

    break # Testando apenas o fold 1. Remova o break para rodar todos.
