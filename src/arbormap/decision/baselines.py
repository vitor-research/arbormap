"""
Baselines de alocação: aleatória, gulosa estática e gulosa dinâmica

Origem: notebook arbormap.ipynb, célula 19 (Google Colab).
Avaliação comparativa das estratégias sob orçamento fixo.
Código preservado como no notebook; caminhos do Colab/Drive devem ser ajustados nas CONFIGURAÇÕES.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# 0. CONFIGURAÇÃO INICIAL E CARREGAMENTO
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "/content/drive/MyDrive/heat_dataset_v3"

# Carregamento do Modelo
checkpoint = torch.load("best_model_fold_0.pt", map_location=device, weights_only=False)

in_channels = checkpoint.get('input_channels', 5)
model = UrbanHeatNet(base_channels=64).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Extração de Estatísticas
CH_LST = 4
stats_mean = checkpoint['mean']
stats_std = checkpoint['std']

MACRO_MEAN_FOLD_1 = stats_mean[CH_LST].item() if torch.is_tensor(stats_mean) else stats_mean[CH_LST]
STD_TRAIN_FOLD_1 = stats_std[CH_LST].item() if torch.is_tensor(stats_std) else stats_std[CH_LST]

print(f"Modelo carregado! Canais: {in_channels} | LST Mean: {MACRO_MEAN_FOLD_1:.2f} | LST Std: {STD_TRAIN_FOLD_1:.4f}")

# ==========================================
# 1. AS TRÊS ESTRATÉGIAS DE ARBORIZAÇÃO
# ==========================================

def baseline_aleatorio(valid_indices, N_celulas):
    """Nível 0: Plantio randômico (Controle)."""
    if len(valid_indices) == 0:
        return []
    indices = np.random.choice(len(valid_indices), size=min(N_celulas, len(valid_indices)), replace=False)
    return valid_indices[indices].tolist()

def baseline_guloso_estatico(valid_indices, pred_anomalia_base, N_celulas):
    """Nível 1: LST Bruta. Escolhe os N pixels mais quentes ignorando sobreposição."""
    valid_mask = np.zeros_like(pred_anomalia_base)
    for y, x in valid_indices:
        valid_mask[y, x] = 1

    masked_heatmap = np.copy(pred_anomalia_base)
    masked_heatmap[valid_mask == 0] = -np.inf

    flat_indices = np.argsort(masked_heatmap.ravel())[-N_celulas:]
    y_coords, x_coords = np.unravel_index(flat_indices, masked_heatmap.shape)

    return list(zip(y_coords.tolist(), x_coords.tolist()))

def baseline_guloso_dinamico(model, seq_tensor, simulador, N_celulas):
    """
    Nível 2: UrbanHeatNet com Consciência Espacial (O Diferencial do Artigo).
    Usa predição dinâmica da U-Net combinada com um kernel de área de influência.
    """
    print(f"Executando Guloso Dinâmico (UrbanHeatNet) para {N_celulas} passos...")
    model.eval()

    H, W = seq_tensor.shape[-2:]
    valid_indices = simulador.get_valid_planting_mask(seq_tensor)
    valid_mask = torch.zeros((H, W), dtype=torch.bool, device=simulador.device)
    valid_mask[valid_indices[:, 0], valid_indices[:, 1]] = True

    ocupado = torch.zeros((H, W), dtype=torch.bool, device=simulador.device)
    pontos_plantados = []
    seq_atual = seq_tensor.clone()

    # Previsão inicial (Estado 0) para ancoragem térmica
    with torch.no_grad():
        anomalia_base = model(seq_atual[:, :5, :, :]).squeeze()

    # Kernel de Suavização (Simula a área de influência de uma árvore: 3x3 ou 5x5)
    # Isso faz a rede buscar o "centro" do calor, não um pico isolado
    kernel_size = 3
    pad = kernel_size // 2
    avg_kernel = (torch.ones((1, 1, kernel_size, kernel_size), device=simulador.device) / (kernel_size**2))

    for step in range(N_celulas):
        with torch.no_grad():
            anomalia_atual = model(seq_atual[:, :5, :, :]).squeeze()

        # 1. HÍBRIDO TÉRMICO: 30% foco no problema original, 70% foco na situação atual
        mapa_hibrido = (0.3 * anomalia_base) + (0.7 * anomalia_atual)

        # 2. CONSCIÊNCIA ESPACIAL: Suaviza o mapa para encontrar "bolsões" de calor
        mapa_unsqueeze = mapa_hibrido.unsqueeze(0).unsqueeze(0)
        mapa_suavizado = F.conv2d(mapa_unsqueeze, avg_kernel, padding=pad).squeeze()

        # 3. MÁSCARA: Remove locais inválidos e já ocupados
        busca_mask = valid_mask & (~ocupado)
        mapa_busca = mapa_suavizado.clone()
        mapa_busca[~busca_mask] = -float('inf')

        # 4. DECISÃO: Escolhe o centro da maior massa térmica viável
        flat_idx = torch.argmax(mapa_busca)
        best_y = (flat_idx // W).item()
        best_x = (flat_idx % W).item()

        pontos_plantados.append((best_y, best_x))
        # 4. DECISÃO: Escolhe o centro da maior massa térmica viável
        flat_idx = torch.argmax(mapa_busca)
        best_y = (flat_idx // W).item()
        best_x = (flat_idx % W).item()

        pontos_plantados.append((best_y, best_x))

        # --- NOVO: RAIO DE EXCLUSÃO (Ex: bloco de 5x5 pixels) ---
        raio_bloqueio = 2
        y_min = max(0, best_y - raio_bloqueio)
        y_max = min(H, best_y + raio_bloqueio + 1)
        x_min = max(0, best_x - raio_bloqueio)
        x_max = min(W, best_x + raio_bloqueio + 1)

        # Bloqueia a área vizinha para forçar o algoritmo a procurar o PRÓXIMO bolsão de calor
        ocupado[y_min:y_max, x_min:x_max] = True

        # 5. ATUALIZAÇÃO FÍSICA
        y_tensor = torch.tensor([best_y], device=simulador.device)
        x_tensor = torch.tensor([best_x], device=simulador.device)
        seq_atual = simulador.apply_intervention(seq_atual, y_tensor, x_tensor)

    return pontos_plantados

# ==========================================
# 2. AVALIAÇÃO E PLOTAGEM
# ==========================================

def avaliar_estrategia(model, seq_tensor, pontos_intervencao, simulador, anomalia_base):
    if not pontos_intervencao:
        return 0.0, 0.0, np.zeros_like(anomalia_base)

    y_coords = torch.tensor([p[0] for p in pontos_intervencao], device=simulador.device)
    x_coords = torch.tensor([p[1] for p in pontos_intervencao], device=simulador.device)

    seq_interv = simulador.apply_intervention(seq_tensor, y_coords, x_coords)

    with torch.no_grad():
        anomalia_interv = model(seq_interv[:, :5, :, :]).cpu().numpy()[0, 0]

    mapa_resfriamento = anomalia_base - anomalia_interv

    # --- NOVA MÉTRICA: RESFRIAMENTO ÚTIL ---
    # Define o limiar de Ilha de Calor (acima da média + 0.5 desvio padrão da cena)
    limiar_critico = np.mean(anomalia_base) + (np.std(anomalia_base) * 0.5)

    # Cria uma máscara booleana apontando apenas para as áreas críticas
    mascara_ilhas_calor = anomalia_base > limiar_critico

    # Soma o resfriamento APENAS nas áreas que eram ilhas de calor
    resfriamento_util = np.sum(mapa_resfriamento[mascara_ilhas_calor & (mapa_resfriamento > 0)])

    # A eficiência prévia antes da escala
    eficiencia_util = resfriamento_util / max(len(pontos_intervencao), 1)

    return resfriamento_util, eficiencia_util, mapa_resfriamento

def executar_analise_plantio(model, seq_tensor, simulador, std_train, N_celulas=50):
    with torch.no_grad():
        anomalia_base = model(seq_tensor[:, :5, :, :]).cpu().numpy()[0, 0]

    valid_indices = simulador.get_valid_planting_mask(seq_tensor)

    # Execução das 3 Estratégias
    pts_aleat = baseline_aleatorio(valid_indices, N_celulas)
    resf_aleat, ef_aleat, _ = avaliar_estrategia(model, seq_tensor, pts_aleat, simulador, anomalia_base)

    pts_estatico = baseline_guloso_estatico(valid_indices.cpu().numpy(), anomalia_base, N_celulas)
    resf_estatico, ef_estatico, _ = avaliar_estrategia(model, seq_tensor, pts_estatico, simulador, anomalia_base)

    pts_dinamico = baseline_guloso_dinamico(model, seq_tensor, simulador, N_celulas)
    resf_dinamico, ef_dinamico, mapa_resf_dinamico = avaliar_estrategia(model, seq_tensor, pts_dinamico, simulador, anomalia_base)

    # Escala para Graus Celsius baseada no desvio físico real do dataset de treino
    resf_aleat *= std_train
    resf_estatico *= std_train
    resf_dinamico *= std_train

    # A eficiência é calculada sobre o resfriamento escalado corrigido
    ef_aleat = resf_aleat / N_celulas
    ef_estatico = resf_estatico / N_celulas
    ef_dinamico = resf_dinamico / N_celulas

    # Impressão Acadêmica
    print("\n" + "="*85)
    print("TABELA 1: DESEMPENHO DAS ESTRATÉGIAS DE MITIGAÇÃO TÉRMICA (RESFRIAMENTO ÚTIL)")
    print("="*85)
    print(f"{'Estratégia (Política de Plantio)':<35} | {'Resfriamento Útil (°C)':<22} | {'Eficiência Útil (°C/célula)':<20}")
    print("-" * 85)
    print(f"{'1. Controle (Aleatório)':<35} | {resf_aleat:<22.2f} | {ef_aleat:<20.4f}")
    print(f"{'2. Guloso Estático (LST Bruta)':<35} | {resf_estatico:<22.2f} | {ef_estatico:<20.4f}")
    print(f"{'3. Guloso Dinâmico (UrbanHeatNet)':<35} | {resf_dinamico:<22.2f} | {ef_dinamico:<20.4f}")
    print("="*85)

    # --- Plotagem Espacial (O Gráfico do Artigo) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    H, W = seq_tensor.shape[-2:]
    valid_mask_2d = np.zeros((H, W))
    for y, x in valid_indices:
        valid_mask_2d[y, x] = 1

    # FUNDO DE ALTO CONTRASTE
    # 1.0 (Branco puro) para áreas válidas
    # 0.15 (Cinza muito escuro/quase preto) para áreas inválidas (rios/construções densas)
    fundo = np.where(valid_mask_2d == 1, 1.0, 0.15)

    # ---------------------------------------------------------
    # Plot 1: Estratégia Estática
    # ---------------------------------------------------------
    axes[0].imshow(fundo, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title("Estratégia Estática: Foco Apenas na LST Bruta", pad=10)

    if pts_estatico:
        y_lst, x_lst = zip(*pts_estatico)
        # Marcador vermelho com linha mais grossa para destacar no fundo branco
        axes[0].scatter(x_lst, y_lst, c='red', marker='x', s=50, linewidths=2.5, label='Decisão (LST Bruta)')

    axes[0].legend(loc='upper right', framealpha=0.95)
    axes[0].axis('off') # Remove os eixos numéricos para visual mais limpo

    # ---------------------------------------------------------
    # Plot 2: Mapa Térmico e UrbanHeatNet
    # ---------------------------------------------------------
    axes[1].imshow(fundo, cmap='gray', vmin=0, vmax=1)

    # Prepara o mapa de resfriamento
    mapa_resfriamento_plot = np.where(mapa_resf_dinamico > 0, mapa_resf_dinamico * std_train, np.nan)
    # Garante que só será colorido onde a máscara é válida
    mapa_resf_visual = np.where(valid_mask_2d == 1, mapa_resfriamento_plot, np.nan)

    v_min = np.nanmin(mapa_resf_visual)
    v_max = np.nanmax(mapa_resf_visual)

    # Plota o resfriamento com alpha maior (0.85) para cores mais vivas
    im1 = axes[1].imshow(mapa_resf_visual, cmap='coolwarm_r', vmin=v_min, vmax=v_max, alpha=0.85)

    axes[1].set_title("Estratégia Dinâmica: UrbanHeatNet (Foco no Espalhamento)", pad=10)

    if pts_dinamico:
        y_unet, x_unet = zip(*pts_dinamico)
        # Marcador verde limão com borda preta forte para máximo destaque
        axes[1].scatter(x_unet, y_unet, c='#00FF00', marker='o', s=45, edgecolors='black', linewidth=1.2, label='Decisão (UrbanHeatNet)')

    axes[1].legend(loc='upper right', framealpha=0.95)
    axes[1].axis('off')

    # Ajuste da barra de cores
    cbar = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Resfriamento Projetado (°C)", rotation=270, labelpad=15)

    plt.tight_layout()
    plt.savefig("figura_estrategias_comparativas.png", dpi=300, bbox_inches='tight')
    plt.show()

# ==========================================
# EXECUÇÃO PRINCIPAL
# ==========================================
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    CIDADE_ALVO = "Teresinar"
    BATCH_SIZE = 4

    cidade_path = Path(DATA_DIR) / CIDADE_ALVO
    if not cidade_path.exists():
        print(f"Erro: A pasta {cidade_path} não foi encontrada!")
        exit()

    fold_stats = (stats_mean, stats_std)

    val_ds = UrbanHeatDataset(
        root=DATA_DIR,
        allowed_cities=[CIDADE_ALVO],
        stats=fold_stats,
        is_train=False,
        normalize=True
    )

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    if len(val_ds) > 0:
        seq_batch, _ = next(iter(val_loader))
        seq_tensor = seq_batch[0:1].to(device)

        simulador = GeneticPlantingSimulator(model, device, dataset_stats=fold_stats)

        print("\nIniciando Benchmark de Estratégias para o Artigo...")
        executar_analise_plantio(model, seq_tensor, simulador, STD_TRAIN_FOLD_1, N_celulas=50)
