"""
Simulador genético de plantio (experimental)

Origem: notebook arbormap.ipynb, célula 27 (Google Colab).
Alternativa evolutiva ao motor de decisão; não usada nos resultados do artigo.
Código preservado como no notebook; caminhos do Colab/Drive devem ser ajustados nas CONFIGURAÇÕES.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib.colors as mcolors

# ====================================================================
# 1. IMPORTAÇÕES DO SEU SCRIPT PRINCIPAL
# ====================================================================
# Substitua 'seu_script_treino' pelo nome do arquivo onde estão as classes.
# Se estiver no Colab no mesmo Notebook, não precisa importar, apenas rode a célula!
# try:
    # from seu_script_treino import UrbanHeatNet, UrbanHeatDataset, CH_LST, CH_NDVI, CH_NDBI, CH_NDWI
# except ImportError:
#     print("Aviso: Certifique-se de que UrbanHeatNet e UrbanHeatDataset estão na memória.")
#     # Índices da nova arquitetura 2D (apenas para garantir)
#     CH_NDVI, CH_NDBI, CH_NDWI = 0, 1, 2


# ====================================================================
# 2. SIMULADOR GENÉTICO (OTIMIZADO PARA ARQUITETURA 2D)
# ====================================================================
class GeneticPlantingSimulator:
    def __init__(self, model, device, std_lst):
        self.model = model
        self.device = device
        self.std_lst = std_lst  # Fator de conversão para °C

        # Kernel de espalhamento (árvores resfriam os vizinhos diretos)
        self.spillover_kernel = torch.tensor([
            [0.10, 0.25, 0.10],
            [0.25, 1.00, 0.25],
            [0.10, 0.25, 0.10]
        ], device=device).view(1, 1, 3, 3)

    def get_valid_planting_mask(self, seq):
        """Bloqueia o plantio na água (NDWI alto) e foca em áreas não-vegetadas (NDVI baixo)."""
        ndvi = seq[0, CH_NDVI, :, :]
        ndwi = seq[0, CH_NDWI, :, :]

        # O limiar depende da sua normalização Z-score.
        # ndwi < 0.0 geralmente significa solo seco/asfalto.
        valid_mask = ndvi < 0.5
        return torch.nonzero(valid_mask)

    def apply_intervention(self, seq, coords_y, coords_x):
        """Planta árvores virtuais e espalha o efeito de vegetação localmente."""
        seq_interv = seq.clone()
        _, _, H, W = seq_interv.shape
        plant_mask = torch.zeros((1, 1, H, W), device=self.device)

        for y, x in zip(coords_y, coords_x):
            if 0 <= y < H and 0 <= x < W:
                plant_mask[0, 0, y, x] = 1.0

        # Espalha o efeito da copa das árvores para pixels vizinhos
        impact_map = F.conv2d(plant_mask, self.spillover_kernel, padding=1).squeeze(0)

        # Aumenta vigorosamente o NDVI (Canal 0) e diminui Prédios/Asfalto (Canal 1)
        # Limitamos o NDVI máximo para não ultrapassar a realidade física
        current_ndvi = seq_interv[0, CH_NDVI, :, :]
        room_to_grow = torch.clamp(2.0 - current_ndvi, min=0.0)
        actual_boost = torch.min(impact_map[0] * 1.5, room_to_grow)

        seq_interv[0, CH_NDVI, :, :] += actual_boost
        seq_interv[0, CH_NDBI, :, :] -= actual_boost * 0.5

        return seq_interv

    @torch.no_grad()
    def evaluate_fitness(self, seq, population):
        self.model.eval()
        # Predição base antes de plantar qualquer árvore
        base_pred = self.model(seq).squeeze()

        scores = []
        for coords in population:
            y_coords, x_coords = coords[:, 0], coords[:, 1]

            # Predição com as novas árvores
            seq_interv = self.apply_intervention(seq, y_coords, x_coords)
            interv_pred = self.model(seq_interv).squeeze()

            # Resfriamento em Graus Celsius = (Base - Intervenção) * Desvio Padrão Real
            # Valores Positivos = Temperatura Caiu (Sucesso)
            cooling_map = (base_pred - interv_pred) * self.std_lst

            # O fitness é o somatório do resfriamento gerado (penaliza se esquentar)
            score = cooling_map[cooling_map > 0.01].sum().item()
            scores.append(score)

        return np.array(scores)

    def optimize(self, seq, num_trees=50, pop_size=20, generations=15):
        valid_indices = self.get_valid_planting_mask(seq)

        if len(valid_indices) < num_trees:
            print("⚠️ Poucos espaços válidos! Reduzindo número de árvores.")
            num_trees = len(valid_indices)

        # 1. População Inicial Aleatória
        population = []
        for _ in range(pop_size):
            idx = torch.randperm(len(valid_indices))[:num_trees]
            population.append(valid_indices[idx])

        best_solution = None
        best_fitness = -float('inf')

        print(f"🌱 Iniciando Otimização Genética ({generations} Gerações)...")
        for gen in range(generations):
            scores = self.evaluate_fitness(seq, population)
            max_idx = np.argmax(scores)

            if scores[max_idx] > best_fitness:
                best_fitness = scores[max_idx]
                best_solution = population[max_idx].clone()

            print(f"   ➜ Geração {gen+1:02d} | Resfriamento Máx Projetado: {best_fitness:.2f} °C Acumulados")

            # 2. Crossover e Mutação
            new_pop = []
            for _ in range(pop_size):
                # Torneio: Escolhe o melhor entre 2 indivíduos aleatórios
                i1, i2 = random.sample(range(pop_size), 2)
                parent = population[i1] if scores[i1] > scores[i2] else population[i2]
                child = parent.clone()

                # Mutação: Troca algumas árvores de lugar
                if random.random() < 0.6:
                    mut_count = random.randint(1, max(1, num_trees // 4))
                    mut_indices = random.sample(range(num_trees), mut_count)
                    new_spots = valid_indices[torch.randperm(len(valid_indices))[:mut_count]]
                    child[mut_indices] = new_spots

                new_pop.append(child)
            population = new_pop

        return best_solution, best_fitness


# ====================================================================
# 3. FUNÇÃO DE VISUALIZAÇÃO E EXECUÇÃO PRINCIPAL
# ====================================================================
def executar_simulacao(model_path, data_root, num_trees=100):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nCarregando Modelo...")

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model = UrbanHeatNet().to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # O LST estava no índice 0 no array global de estatísticas
    std_lst = float(checkpoint['std'][0].flatten()[0])

    print("Carregando Dados de Validação...")
    all_cities = [d.name for d in Path(data_root).iterdir() if d.is_dir() and not d.name.startswith('.')]

    # Pega qualquer cidade de validação (ex: a primeira da lista)
    val_ds = UrbanHeatDataset(data_root, allowed_cities=[all_cities[0]],
                              stats=(checkpoint['mean'], checkpoint['std']), is_train=False)

    val_loader = DataLoader(val_ds, batch_size=1, shuffle=True)
    seq, target = next(iter(val_loader))
    seq = seq.to(DEVICE)

    # Inicia o Simulador
    simulator = GeneticPlantingSimulator(model, DEVICE, std_lst)

    # Roda a otimização
    best_coords, total_cooling = simulator.optimize(seq, num_trees=num_trees, pop_size=20, generations=20)

    # ================== RESULTADOS VISUAIS ==================
    with torch.no_grad():
        base_pred = model(seq).squeeze()
        seq_interv = simulator.apply_intervention(seq, best_coords[:, 0], best_coords[:, 1])
        interv_pred = model(seq_interv).squeeze()

        # Diferença Térmica em °C
        cooling_map = (base_pred - interv_pred).cpu().numpy() * std_lst

    # Prepara dados originais para plot (desnormaliza NDVI aproximado)
    ndvi_base = seq[0, CH_NDVI, :, :].cpu().numpy()

    plt.figure(figsize=(18, 5))

    # 1. Mapa de NDVI Base
    plt.subplot(1, 3, 1)
    plt.imshow(ndvi_base, cmap='YlGn', vmin=-1.5, vmax=2.0)
    plt.title("NDVI Original (Áreas Verdes)")
    plt.axis('off')

    # 2. Locais de Plantio Otimizados
    plt.subplot(1, 3, 2)
    plt.imshow(ndvi_base, cmap='gray', alpha=0.5)
    plt.scatter(best_coords[:, 1].cpu(), best_coords[:, 0].cpu(), c='lime', s=10, label='Novas Árvores')
    plt.title(f"Plano de Plantio ({num_trees} árvores)")
    plt.legend()
    plt.axis('off')

    # 3. Mapa de Impacto Térmico (Resfriamento)
    plt.subplot(1, 3, 3)
    # Criamos um colormap focado em resfriamento (Azul)
    div_norm = mcolors.TwoSlopeNorm(vmin=-0.1, vcenter=0.0, vmax=np.max(cooling_map))
    im = plt.imshow(cooling_map, cmap='coolwarm_r', norm=div_norm)
    plt.title(f"Resfriamento Estimado\nImpacto Total: {total_cooling:.2f} °C")
    plt.colorbar(im, label="Queda de Temperatura (°C)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    executar_simulacao(
        model_path="best_model_fold_0.pt",
        data_root="/content/drive/MyDrive/heat_dataset_v3",
        num_trees=500  # Quantas árvores a prefeitura quer plantar?
    )
