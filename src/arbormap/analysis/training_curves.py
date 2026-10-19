"""
Curvas de treinamento (Figura 2)

Origem: notebook arbormap.ipynb, célula 21 (Google Colab).
Dados extraídos do log de treino.
Código preservado como no notebook; caminhos do Colab/Drive devem ser ajustados nas CONFIGURAÇÕES.
"""

import matplotlib.pyplot as plt

# Dados extraídos do seu log de treinamento
epochs = list(range(1, 21))

t_loss = [0.3805, 0.3130, 0.2999, 0.2928, 0.2885, 0.2834, 0.2814, 0.2804, 0.2792, 0.2778,
          0.2757, 0.2756, 0.2761, 0.2769, 0.2756, 0.2761, 0.2735, 0.2725, 0.2702, 0.2694]

v_loss = [0.2944, 0.2721, 0.2685, 0.2641, 0.2587, 0.2591, 0.2563, 0.2635, 0.2556, 0.2551,
          0.2569, 0.2546, 0.2596, 0.2586, 0.2675, 0.2586, 0.2756, 0.2520, 0.2571, 0.2645]

r2 = [0.4406, 0.4845, 0.4830, 0.4924, 0.5049, 0.5066, 0.5062, 0.4930, 0.5085, 0.5056,
      0.5015, 0.5065, 0.4924, 0.5030, 0.4763, 0.5053, 0.4591, 0.5101, 0.4980, 0.4806]

pearson = [0.6744, 0.6964, 0.7015, 0.7042, 0.7131, 0.7145, 0.7121, 0.7074, 0.7134, 0.7138,
           0.7098, 0.7128, 0.7049, 0.7124, 0.6906, 0.7160, 0.6854, 0.7148, 0.7057, 0.6981]

std_pred = [0.7915, 0.7114, 0.7896, 0.7251, 0.6694, 0.6553, 0.6963, 0.6196, 0.6899, 0.7007,
            0.6812, 0.7188, 0.7700, 0.6440, 0.6636, 0.6313, 0.5811, 0.7422, 0.7095, 0.6260]

# O std do target é constante em todas as épocas
std_target = [0.9979] * 20

# Configurações globais de estilo
plt.style.use('seaborn-v0_8-whitegrid')

# ==============================================================================
# FIGURA 1: Curvas de treinamento (Loss, R² e Pearson)
# ==============================================================================
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# (a) Perda de Treino vs Validação
ax1.plot(epochs, t_loss, label='Treino (T.Loss)', marker='o', linestyle='-', color='tab:blue')
ax1.plot(epochs, v_loss, label='Validação (V.Loss)', marker='s', linestyle='-', color='tab:orange')
ax1.set_title('(a) Perda de Treino vs. Validação')
ax1.set_xlabel('Época')
ax1.set_ylabel('Loss (MSE)')
ax1.set_xticks(epochs[::2]) # Mostrar ticks a cada 2 épocas
ax1.legend()

# (b) R² e Correlação de Pearson
ax2.plot(epochs, r2, label='R² (Validação)', marker='^', linestyle='-', color='tab:green')
ax2.plot(epochs, pearson, label='Pearson (Validação)', marker='d', linestyle='-', color='tab:purple')
ax2.set_title('(b) Métricas de Avaliação na Validação')
ax2.set_xlabel('Época')
ax2.set_ylabel('Valor da Métrica')
ax2.set_xticks(epochs[::2])
ax2.legend()

# fig1.suptitle('FIGURA 1 – Curvas de treinamento do Fold 1', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ==============================================================================
# FIGURA 2: Desvio-padrão da Predição vs Alvo
# ==============================================================================
fig2, ax3 = plt.subplots(figsize=(8, 5))

ax3.plot(epochs, std_pred, label='Std (Predição)', marker='o', linestyle='-', color='tab:red')
ax3.plot(epochs, std_target, label='Std (Alvo)', linestyle='--', linewidth=2.5, color='black')
# ax3.set_title('FIGURA 2 – Desvio-padrão: Predição vs. Alvo ao longo do treinamento', fontsize=12, fontweight='bold')
ax3.set_xlabel('Época')
ax3.set_ylabel('Desvio Padrão')
ax3.set_xticks(epochs[::2])
ax3.legend()

plt.tight_layout()
plt.show()
