"""
Motor de decisão: simulação de intervenção, filtros e sensibilidade de orçamento

Origem: notebook arbormap.ipynb, célula 7 (Google Colab).
Seleção multicritério, morfologia de corredores (OpenCV), espalhamento do efeito de resfriamento, curva de sensibilidade (Figuras 5 e 6 do artigo).
Código preservado como no notebook; caminhos do Colab/Drive devem ser ajustados nas CONFIGURAÇÕES.
"""

import os
import random
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn.functional as F

# ============================================================
# 1. SIMULAÇÃO DE INTERVENÇÃO COM ORÇAMENTO PARAMETRIZADO
# ============================================================
def run_intervention_simulation(model, input_tensor, valid_mask, city_mean, scale_factor=5.0, budget=30, known_hs_mae=1.5):
    """
    input_tensor: shape (1, Time, Channels, H, W)
    valid_mask: shape (1, 1, H, W)
    city_mean: float ou tensor escalar (temperatura média da cidade)
    """
    model.eval()
    CH_NDVI, CH_NDWI, CH_NDBI, CH_DEM = 0, 1, 2, 3

    # --- CENÁRIO BASE ---
    input_tensor = input_tensor.clone().detach()
    input_tensor.requires_grad_(True)

    # Forward Pass
    base_pred_norm = model(input_tensor)

    # Reconstrução Térmica Corrigida (usando city_mean)
    base_celsius = (base_pred_norm * scale_factor) + city_mean
    base_celsius_np = base_celsius.squeeze().cpu().detach().numpy()

    # Extrai o estado atual (último instante de tempo: index -1)
    current_ndvi = input_tensor[0, -1, CH_NDVI, :, :].detach()
    current_ndwi = input_tensor[0, -1, CH_NDWI, :, :].detach()
    current_dem  = input_tensor[0, -1, CH_DEM, :, :].detach()

    # --- Filtros Restritivos (Solo/Concreto, Sem Água, Sem Relevo Íngreme) ---
    is_concreto = (current_ndvi < 0.2).float()
    not_water = (current_ndwi < 0.1).float()
    dem_mean = current_dem[valid_mask[0, 0] == 1.0].mean()
    is_flat = (torch.abs(current_dem - dem_mean) < 1.5).float()

    valid_intervention_base = is_concreto * not_water * is_flat * valid_mask[0, 0].to(input_tensor.device)
    valid_base_np = valid_intervention_base.cpu().numpy()

    valid_y, valid_x = np.where(valid_base_np == 1)
    if len(valid_y) < budget:
        print(f"⚠️ Área válida insuficiente para o budget solicitado ({budget} pixels).")
        return None

    # --- CÁLCULO DO GRADIENTE TÉRMICO ---
    valid_pixels = base_celsius[valid_mask == 1.0]
    q90_temp = torch.quantile(valid_pixels, 0.9)
    hotspot_mask = (base_celsius >= q90_temp).float() * valid_mask
    total_hotspot_heat = (base_celsius * hotspot_mask).sum()

    model.zero_grad()
    total_hotspot_heat.backward()

    # 1. Gradiente Composto (Sensibilidade a TODAS as variáveis alteradas)
    # Pegamos os gradientes do último instante temporal (-1)
    grad_ndvi = -input_tensor.grad[0, -1, CH_NDVI, :, :].cpu().numpy()
    grad_ndwi = -input_tensor.grad[0, -1, CH_NDWI, :, :].cpu().numpy()
    grad_ndbi = input_tensor.grad[0, -1, CH_NDBI, :, :].cpu().numpy() # Invertido pois queremos reduzir o NDBI
    grad_lst_past = input_tensor.grad[0, -1, 4, :, :].cpu().numpy()   # Canal 4 (Inércia térmica)

    # 2. Soma Ponderada do Potencial de Mitigação
    # Focamos apenas nas direções que ajudam a esfriar (ReLU)
    composite_gradient = (
        np.maximum(0, grad_ndvi) * 0.3 +
        np.maximum(0, grad_ndwi) * 0.2 +
        np.maximum(0, grad_ndbi) * 0.2 +
        np.maximum(0, grad_lst_past) * 1.0  # A inércia tem peso maior
    )

    # 3. Suavização Espacial (Filtro de Ruído)
    # Remove "artefatos matemáticos" de 1 pixel e força a busca por clusters férteis
    smoothed_gradient = cv2.GaussianBlur(composite_gradient, (5, 5), 1.0)

    # ================= ESTRATÉGIAS DE ALOCAÇÃO =================

    # A. MULTI-CRITÉRIO HÍBRIDO (Gradiente + Urgência Térmica)
    # Normalizamos o gradiente e a temperatura para somá-los de forma justa
    grad_norm = (smoothed_gradient - smoothed_gradient.min()) / (smoothed_gradient.max() - smoothed_gradient.min() + 1e-8)
    temp_norm = (base_celsius_np - base_celsius_np.min()) / (base_celsius_np.max() - base_celsius_np.min() + 1e-8)

    # Score final: 60% potencial de esfriamento (Rede Neural) + 40% urgência (Física)
    hybrid_score = (grad_norm * 0.6) + (temp_norm * 0.4)
    planting_potential = hybrid_score * valid_base_np

    gradient_vals = planting_potential[valid_y, valid_x]
    top_multi_idx = np.argsort(gradient_vals)[::-1][:budget]
    mask_multi = np.zeros_like(base_celsius_np, dtype=bool)
    mask_multi[valid_y[top_multi_idx], valid_x[top_multi_idx]] = True

    # B. GULOSA
    valid_temps = base_celsius_np[valid_y, valid_x]
    top_greedy_idx = np.argsort(valid_temps)[::-1][:budget]
    mask_greedy = np.zeros_like(base_celsius_np, dtype=bool)
    mask_greedy[valid_y[top_greedy_idx], valid_x[top_greedy_idx]] = True

    # C. ALEATÓRIA
    random_idx = np.random.choice(len(valid_y), size=budget, replace=False)
    mask_random = np.zeros_like(base_celsius_np, dtype=bool)
    mask_random[valid_y[random_idx], valid_x[random_idx]] = True

    # ================= FUNÇÃO AVALIADORA TERMODINÂMICA =================
    def evaluate_intervention(mask):
        simulated_input = input_tensor.clone().detach()

        # 1. Modifica a cobertura do solo em toda a sequência de tempo
        simulated_input[0, :, CH_NDVI, mask] = 0.55
        simulated_input[0, :, CH_NDBI, mask] = -0.15
        simulated_input[0, :, CH_NDWI, mask] = 0.10

        # 2. CORREÇÃO DA INÉRCIA TÉRMICA (Canal 4 = LST_past)
        # Definimos como -0.2 (representando que a área já era 1.0°C mais fria que a média da cidade)
        simulated_input[0, :, 4, mask] = -0.20

        with torch.no_grad():
            future_celsius = (model(simulated_input) * scale_factor) + city_mean

        raw_cooling = (base_celsius - future_celsius).squeeze().cpu().detach().numpy()

        # Advecção Multi-escala (Simula dispersão do vento)
        local_cooling = cv2.GaussianBlur(raw_cooling, (5, 5), 1.0)
        breeze_cooling = cv2.GaussianBlur(raw_cooling, (15, 15), 3.0)
        smoothed_cooling = (0.7 * local_cooling) + (0.3 * breeze_cooling)

        avg_cooling = smoothed_cooling[mask].mean() if mask.sum() > 0 else 0
        return smoothed_cooling, avg_cooling

    cool_multi, avg_multi = evaluate_intervention(mask_multi)
    cool_greedy, avg_greedy = evaluate_intervention(mask_greedy)
    cool_rand, avg_rand = evaluate_intervention(mask_random)

    return {
        'base_celsius': base_celsius_np,
        'error_margin': known_hs_mae,
        'budget': budget,
        'multi': {'mask': mask_multi, 'cooling': cool_multi, 'avg_cooling': avg_multi},
        'greedy': {'mask': mask_greedy, 'cooling': cool_greedy, 'avg_cooling': avg_greedy},
        'random': {'mask': mask_random, 'cooling': cool_rand, 'avg_cooling': avg_rand}
    }


# ============================================================
# 2. PLOTAGEM ESPACIAL COMPACTA (PARA O ARTIGO)
# ============================================================
def plot_intervention_compact(results, save_path="figura_artigo_espacial.png"):
    if results is None: return

    fig, axes = plt.subplots(2, 4, figsize=(12, 5.5), gridspec_kw={'wspace': 0.05, 'hspace': 0.1})

    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])

    vmin, vmax = results['base_celsius'].min(), results['base_celsius'].max()

    # Base e Título
    im_base = axes[0, 0].imshow(results['base_celsius'], cmap='coolwarm', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title('LST Base (°C)', fontsize=10, fontweight='bold')

    strategies = [
        ('Multi-Critério', results['multi'], axes[0, 1], axes[1, 1]),
        ('Gulosa', results['greedy'], axes[0, 2], axes[1, 2]),
        ('Aleatória', results['random'], axes[0, 3], axes[1, 3])
    ]

    for title, data, ax_mask, ax_cool in strategies:
        # Fundo cinza + pixels de intervenção
        ax_mask.imshow(results['base_celsius'], cmap='gray', alpha=0.6)
        overlay = np.zeros((*data['mask'].shape, 4))
        overlay[data['mask']] = [0, 1, 0, 1.0] # Verde forte para contraste
        ax_mask.imshow(overlay)
        ax_mask.set_title(f'Alocação: {title}', fontsize=10)

        # Mapas de Resfriamento
        im_cool = ax_cool.imshow(data['cooling'], cmap='Blues', vmin=0, vmax=results['multi']['cooling'].max())
        ax_cool.set_title(f'ΔT Resfriamento', fontsize=10)

    # Bloco de texto com métricas
    axes[1, 0].axis('off')
    stats_text = (
        f"Métricas (Budget={results['budget']} px)\n"
        f"------------------------\n"
        f"Redução Média (ΔT):\n"
        f"• Multi: -{results['multi']['avg_cooling']:.2f} °C\n"
        f"• Gulosa: -{results['greedy']['avg_cooling']:.2f} °C\n"
        f"• Aleat.: -{results['random']['avg_cooling']:.2f} °C\n\n"
        f"MAE Modelo: ±{results['error_margin']} °C"
    )
    axes[1, 0].text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray'))

    # Colorbars ajustados
    cbar_ax_base = fig.add_axes([0.13, 0.52, 0.01, 0.35])
    fig.colorbar(im_base, cax=cbar_ax_base)

    cbar_ax_cool = fig.add_axes([0.91, 0.12, 0.01, 0.35])
    fig.colorbar(im_cool, cax=cbar_ax_cool, label='ΔT (°C)')

    plt.subplots_adjust(left=0.18, right=0.9, top=0.9, bottom=0.1)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================
# 3. PLOTAGEM DA CURVA DE SENSIBILIDADE DO BUDGET
# ============================================================
def plot_budget_sensitivity_curve(model, input_tensor, valid_mask, city_mean, scale_factor, budgets, save_path="figura_artigo_sensibilidade.png"):
    print("\n📈 Executando análise paramétrica (Sensibilidade de Orçamento)...")
    res_multi, res_greedy, res_random = [], [], []
    valid_budgets = []

    for b in budgets:
        res = run_intervention_simulation(model, input_tensor, valid_mask, city_mean, scale_factor, budget=b)
        if res is not None:
            res_multi.append(res['multi']['avg_cooling'])
            res_greedy.append(res['greedy']['avg_cooling'])
            res_random.append(res['random']['avg_cooling'])
            valid_budgets.append(b)
            print(f"   - Budget {b:3d}: Multi (-{res['multi']['avg_cooling']:.2f}°C) | Gulosa (-{res['greedy']['avg_cooling']:.2f}°C)")

    if not valid_budgets:
        print("❌ Nenhum budget válido para plotar.")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(valid_budgets, res_multi, marker='o', linestyle='-', color='#10b981', linewidth=2.5, label='Multi-Critério (Proposto)')
    plt.plot(valid_budgets, res_greedy, marker='s', linestyle='--', color='#ef4444', linewidth=2, label='Baseline Gulosa')
    plt.plot(valid_budgets, res_random, marker='^', linestyle=':', color='#6b7280', linewidth=2, label='Baseline Aleatória')
    plt.title('Eficiência de Resfriamento vs. Orçamento de Intervenção', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Orçamento de Intervenção (Número de Pixels, ~900m² cada)', fontsize=10)
    plt.ylabel('Redução Média de LST Absoluta (|ΔT| °C)', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(frameon=True, facecolor='white', edgecolor='gray')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================
if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SCALE_FACTOR = 5.0
    INDICE_IMAGEM = 0
    BUDGET_ESPACIAL = 45
    BUDGETS_CURVA = [10, 20, 30, 40, 50, 75, 100, 150]
    DATA_DIR = os.environ.get("ARBORMAP_DATA_DIR", "./data")

    # 1. Carrega o modelo novo (ConvLSTM)
    # OBS: Certifique-se de que a classe UNetConvLSTM já foi executada na célula anterior do seu Jupyter Notebook
    model = UNetConvLSTM(in_channels=8).to(DEVICE)
    modelo_path = "best_convlstm_fold_0.pt"

    if os.path.exists(modelo_path):
        model.load_state_dict(torch.load(modelo_path, map_location=DEVICE, weights_only=True))
    else:
        print(f"⚠️ AVISO: Pesos não encontrados ({modelo_path}). Usando pesos aleatórios.")
    model.eval()

    # 2. Carrega os dados com o Dataset Temporal
    all_files_raw = list(Path(DATA_DIR).rglob("*.npy"))
    valid_files_set = filter_cloudy_files_runtime(all_files_raw)

    teresina_ds = UrbanHeatTemporalDataset(root=DATA_DIR, allowed_cities=["Teresina"], valid_files_set=valid_files_set, is_train=False)

    if len(teresina_ds) == 0:
        print("❌ Nenhum dado encontrado para Teresina.")
    else:
        # Acesso Direto pelo Índice
        sample_seq, sample_tgt, sample_valid, city_mean = teresina_ds[INDICE_IMAGEM]

        # Adicionando a dimensão de batch
        input_tensor = sample_seq.unsqueeze(0).to(DEVICE)
        valid_mask = sample_valid.unsqueeze(0).to(DEVICE)

        # Formata o city_mean para escalar Python (ou tensor no mesmo device)
        if isinstance(city_mean, torch.Tensor):
            city_mean = city_mean.to(DEVICE).item()

        print(f"\n🌳 INICIANDO SIMULAÇÃO DE INTERVENÇÃO (Imagem Índice: {INDICE_IMAGEM})")
        print("="*70)

        res_espacial = run_intervention_simulation(
            model=model,
            input_tensor=input_tensor,
            valid_mask=valid_mask,
            city_mean=city_mean,
            scale_factor=SCALE_FACTOR,
            budget=BUDGET_ESPACIAL,
            known_hs_mae=1.5
        )

        if res_espacial:
            plot_intervention_compact(res_espacial, "figura_artigo_espacial.png")
            print(f"✅ Mapa espacial gerado (Budget {BUDGET_ESPACIAL}): 'figura_artigo_espacial.png'")

        plot_budget_sensitivity_curve(
            model=model,
            input_tensor=input_tensor,
            valid_mask=valid_mask,
            city_mean=city_mean,
            scale_factor=SCALE_FACTOR,
            budgets=BUDGETS_CURVA,
            save_path="figura_artigo_sensibilidade.png"
        )

        print("\n🎉 Todas as simulações foram concluídas com sucesso!")
