"""
Versão intermediária: UrbanHeatNet com módulo ASPP e PhysicsInformedLoss

Origem: notebook arbormap.ipynb, célula 18 (Google Colab).
Substituída pela arquitetura híbrida com atenção.
Código preservado como no notebook; caminhos do Colab/Drive devem ser ajustados nas CONFIGURAÇÕES.
"""

import os

import random

import numpy as np

from pathlib import Path

from sklearn.model_selection import KFold

import math



import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader



# ============================================================

# CONFIGURAÇÃO GERAL

# ============================================================

SEED = 42

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(SEED)



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "/content/drive/MyDrive/heat_dataset_v3"



INPUT_MONTHS = 6

TOTAL_RAW_CHANNELS = 5  # LST, NDVI, NDBI, NDWI, DEM no .npy bruto



# Mantemos todos os 5 canais (água é carregada, mas escondida da rede)

CHANNELS_TO_KEEP = [0, 1, 2, 3, 4]

INPUT_CHANNELS = len(CHANNELS_TO_KEEP)



CH_NDVI = 0

CH_NDWI = 1

CH_NDBI = 2

CH_DEM = 3

CH_LST = 4



BATCH_SIZE = 2

EPOCHS = 20

LR = 1e-4

WEIGHT_DECAY = 0.05

BASE_CHANNELS = 64

K_FOLDS = 5



# Limites físicos

LST_CLIP_MIN, LST_CLIP_MAX = -20.0, 60.0





# ============================================================

# UTILITÁRIO DE FORMATO

# ============================================================

def ensure_channels_first(img):

    if img.ndim != 3:

        raise ValueError(f"Esperado array 3D, recebido shape {img.shape}")

    if img.shape[0] == TOTAL_RAW_CHANNELS:

        return img

    elif img.shape[-1] == TOTAL_RAW_CHANNELS:

        return np.transpose(img, (2, 0, 1))

    else:

        raise ValueError(

            f"Não foi possível identificar o eixo de canais no shape {img.shape}. "

            f"Esperava um eixo de tamanho {TOTAL_RAW_CHANNELS}."

        )





# ============================================================

# ESTATÍSTICAS

# ============================================================

def compute_dataset_statistics(files):

    count = 0

    sum_ = np.zeros(INPUT_CHANNELS, dtype=np.float64)

    sum_sq = np.zeros(INPUT_CHANNELS, dtype=np.float64)



    for f in files:

        img = np.load(f).astype(np.float32)

        img = ensure_channels_first(img)[CHANNELS_TO_KEEP, :, :]



        # Computa sobre a imagem inteira válida

        img[CH_LST] = np.clip(img[CH_LST], LST_CLIP_MIN, LST_CLIP_MAX)

        img[CH_NDVI] = np.clip(img[CH_NDVI], -1.0, 1.0)

        img[CH_NDWI] = np.clip(img[CH_NDWI], -1.0, 1.0)



        flat = img.reshape(INPUT_CHANNELS, -1).astype(np.float64)

        sum_ += flat.sum(axis=1)

        sum_sq += (flat ** 2).sum(axis=1)

        count += flat.shape[1]



    mean = sum_ / count

    var = np.clip(sum_sq / count - mean ** 2, a_min=0.0, a_max=None)

    std = np.sqrt(var)



    return mean.astype(np.float32), std.astype(np.float32)





def compute_delta_statistics(root, allowed_cities):

    root = Path(root)

    deltas = []



    for city in allowed_cities:

        folder = root / city

        if not folder.exists(): continue

        city_files = sorted(folder.glob("*.npy"))

        if len(city_files) < 2: continue



        for i in range(len(city_files) - 1):

            img_a = ensure_channels_first(np.load(city_files[i]).astype(np.float32))[CH_LST]

            img_b = ensure_channels_first(np.load(city_files[i + 1]).astype(np.float32))[CH_LST]



            h = min(img_a.shape[0], img_b.shape[0])

            w = min(img_a.shape[1], img_b.shape[1])

            img_a = np.clip(img_a[:h, :w], LST_CLIP_MIN, LST_CLIP_MAX)

            img_b = np.clip(img_b[:h, :w], LST_CLIP_MIN, LST_CLIP_MAX)



            deltas.append((img_b - img_a).flatten())



    if not deltas:

        raise ValueError("Nenhum par consecutivo encontrado.")



    deltas = np.concatenate(deltas)

    return float(np.mean(deltas)), float(np.std(deltas))





class Normalize:

    def __init__(self, mean, std):

        self.mean = torch.tensor(mean, dtype=torch.float32).view(INPUT_CHANNELS, 1, 1)

        self.std = torch.tensor(std, dtype=torch.float32).view(INPUT_CHANNELS, 1, 1) + 1e-6



    def __call__(self, img):

        if isinstance(img, np.ndarray):

            img = torch.from_numpy(img).float()

        return (img - self.mean) / self.std





# ============================================================

# DATASET COM RANDOM CROPPING E AUGMENTATION

# ============================================================

class UrbanHeatDataset(Dataset):

    def __init__(self, root, allowed_cities, normalize=True, stats=None, delta_stats=None, is_train=False):

        self.root = Path(root)

        self.normalize = normalize

        self.is_train = is_train

        self.samples = []

        all_files = []



        for city in allowed_cities:

            folder = self.root / city

            if not folder.exists(): continue

            files = sorted(folder.glob("*.npy"))

            all_files.extend(files)

            # Precisamos de apenas 2 imagens: a atual (geometria) e a futura (alvo)

            if len(files) < 2: continue

            for i in range(len(files) - 1):

                self.samples.append((files[i], files[i + 1]))



        if normalize and len(all_files) > 0:

            if stats is not None:

                self.mean, self.std = stats

            else:

                stats_file = self.root / "dataset_stats_all_channels.npz"

                if stats_file.exists():

                    saved = np.load(stats_file)

                    self.mean, self.std = saved["mean"], saved["std"]

                else:

                    self.mean, self.std = compute_dataset_statistics(all_files)

                    np.savez(stats_file, mean=self.mean, std=self.std)

            self.normalizer = Normalize(self.mean, self.std)



    def __len__(self):

        return len(self.samples)



    def __getitem__(self, idx):

        base_file, target_file = self.samples[idx]



        # 1. Carrega as imagens

        base_img = np.load(base_file).astype(np.float32)

        base_img = ensure_channels_first(base_img)[CHANNELS_TO_KEEP]

        tgt_img = np.load(target_file).astype(np.float32)

        tgt_img = ensure_channels_first(tgt_img)[CHANNELS_TO_KEEP]



        max_h = min(base_img.shape[1], tgt_img.shape[1])

        max_w = min(base_img.shape[2], tgt_img.shape[2])



        # 2. Crop

        crop_size = 400

        if self.is_train and max_h > crop_size and max_w > crop_size:

            y = random.randint(0, max_h - crop_size)

            x = random.randint(0, max_w - crop_size)

        else:

            y = (max_h - crop_size) // 2 if max_h > crop_size else 0

            x = (max_w - crop_size) // 2 if max_w > crop_size else 0



        # Aplica crop e normalização na base

        cropped_base = base_img[:, y:y+crop_size, x:x+crop_size]

        padded_base = np.zeros((INPUT_CHANNELS, crop_size, crop_size), dtype=np.float32)

        padded_base[:, :cropped_base.shape[1], :cropped_base.shape[2]] = cropped_base



        padded_base[CH_LST] = np.clip(padded_base[CH_LST], LST_CLIP_MIN, LST_CLIP_MAX)

        padded_base[CH_NDVI] = np.clip(padded_base[CH_NDVI], -1.0, 1.0)

        padded_base[CH_NDWI] = np.clip(padded_base[CH_NDWI], -1.0, 1.0)



        if self.normalize:

            padded_base = self.normalizer(padded_base)



        if isinstance(padded_base, np.ndarray):

            padded_base = torch.from_numpy(padded_base).float()



        # 3. EXTRAÇÃO DE FEATURES (O NDWI vai para o final do batch como máscara)

        model_features = padded_base[[CH_NDVI, CH_NDBI, CH_DEM], :, :] # 3 canais pra rede

        water_mask = padded_base[[CH_NDWI], :, :] # 1 canal pro Algoritmo Genético



        # 4. Sazonalidade

        try:

            nome_arquivo = target_file.stem

            mes_str = "".join(filter(str.isdigit, nome_arquivo.split('_')[-1]))

            mes_alvo = int(mes_str) if mes_str else (idx % 12) + 1

            if not (1 <= mes_alvo <= 12): mes_alvo = (idx % 12) + 1

        except Exception:

            mes_alvo = (idx % 12) + 1



        seno_mes = math.sin(2.0 * math.pi * mes_alvo / 12.0)

        cos_mes = math.cos(2.0 * math.pi * mes_alvo / 12.0)



        mapa_seno = torch.full((1, crop_size, crop_size), seno_mes, dtype=torch.float32)

        mapa_cos = torch.full((1, crop_size, crop_size), cos_mes, dtype=torch.float32)



        # Concatena: [NDVI(0), NDBI(1), DEM(2), Seno(3), Cosseno(4), NDWI(5)]

        out_seq = torch.cat([model_features, mapa_seno, mapa_cos, water_mask], dim=0)



        # 5. O ALVO: ANOMALIA ESPACIAL (Microclima)

        cropped_tgt = tgt_img[CH_LST, y:y+crop_size, x:x+crop_size]

        padded_tgt_lst = np.zeros((crop_size, crop_size), dtype=np.float32)

        padded_tgt_lst[:cropped_tgt.shape[0], :cropped_tgt.shape[1]] = cropped_tgt

        padded_tgt_lst = np.clip(padded_tgt_lst, LST_CLIP_MIN, LST_CLIP_MAX)



        tgt_tensor = torch.from_numpy(padded_tgt_lst)



        # Subtrai a média da cidade para isolar a Ilha de Calor

        macro_climate = tgt_tensor.mean()

        spatial_anomaly = tgt_tensor - macro_climate



        # Normaliza o desvio para a rede neural

        std_anomaly = spatial_anomaly.std() + 1e-6

        out_tgt = (spatial_anomaly / std_anomaly).unsqueeze(0).float()



        # Augmentation

        if self.is_train:

            if random.random() > 0.5:

                out_seq = torch.flip(out_seq, (-1,))

                out_tgt = torch.flip(out_tgt, (-1,))

            if random.random() > 0.5:

                out_seq = torch.flip(out_seq, (-2,))

                out_tgt = torch.flip(out_tgt, (-2,))

            k = random.randint(0, 3)

            if k > 0:

                out_seq = torch.rot90(out_seq, k, [-1, -2])

                out_tgt = torch.rot90(out_tgt, k, [-1, -2])



        return out_seq, out_tgt



# ============================================================

# MÉTRICAS E FUNÇÃO DE PERDA ESTÁVEL

# ============================================================

def compute_mae(pred, target):

    return torch.mean(torch.abs(pred.float() - target.float()))



def compute_r2(pred, target):

    p, t = pred.float(), target.float()

    target_mean = torch.mean(t)

    ss_tot = torch.sum((t - target_mean) ** 2)

    ss_res = torch.sum((t - p) ** 2)

    if ss_tot == 0: return torch.tensor(0.0, device=p.device)

    return 1 - (ss_res / ss_tot)



def compute_pearson(pred, target):

    p, t = pred.float(), target.float()

    pred_mean = torch.mean(p)

    target_mean = torch.mean(t)

    pred_centered = p - pred_mean

    target_centered = t - target_mean



    cov = torch.sum(pred_centered * target_centered)

    std_pred = torch.sqrt(torch.sum(pred_centered ** 2))

    std_target = torch.sqrt(torch.sum(target_centered ** 2))



    denominator = std_pred * std_target

    if denominator < 1e-6: return torch.tensor(0.0, device=p.device)

    return cov / denominator



def spatial_gradient_loss(pred, target):

    dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]

    dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]



    dx_tgt = target[:, :, :, 1:] - target[:, :, :, :-1]

    dy_tgt = target[:, :, 1:, :] - target[:, :, :-1, :]



    loss_x = torch.mean(torch.abs(dx_pred - dx_tgt))

    loss_y = torch.mean(torch.abs(dy_pred - dy_tgt))



    return loss_x + loss_y



class PhysicsInformedLoss(nn.Module):

    def __init__(self, phys_weight=0.01, struct_weight=0.2):

        super().__init__()

        self.data_loss_fn = nn.HuberLoss(reduction='mean', delta=1.0)

        self.phys_weight = phys_weight

        self.struct_weight = struct_weight



        self.laplacian_kernel = torch.tensor([

            [0.0, 1.0, 0.0],

            [1.0, -4.0, 1.0],

            [0.0, 1.0, 0.0]

        ]).view(1, 1, 3, 3)



    def laplacian(self, x):

        kernel = self.laplacian_kernel.to(x.device)

        return F.conv2d(x, kernel, padding=1)



    def forward(self, pred, target, ndvi_map):

        data_loss = self.data_loss_fn(pred, target)

        struct_loss = spatial_gradient_loss(pred, target)

        lap_pred = self.laplacian(pred)

        phys_loss = torch.mean(torch.abs(lap_pred) * torch.relu(ndvi_map))



        total_loss = data_loss + (self.struct_weight * struct_loss) + (self.phys_weight * phys_loss)

        return total_loss, data_loss, phys_loss



class ConvBNAct(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1, padding=1, dilation=1, kernel_size=3):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation=dilation, bias=False),

            nn.GroupNorm(num_groups=8, num_channels=out_ch),

            nn.GELU()

        )

    def forward(self, x):

        return self.block(x)



class ResidualBlock(nn.Module):

    def __init__(self, ch):

        super().__init__()

        self.conv1 = ConvBNAct(ch, ch)

        self.drop = nn.Dropout2d(0.1)

        self.conv2 = nn.Sequential(

            nn.Conv2d(ch, ch, 3, padding=1, bias=False),

            nn.GroupNorm(num_groups=8, num_channels=ch)

        )

        self.act = nn.GELU()

    def forward(self, x):

        return self.act(self.conv2(self.drop(self.conv1(x))) + x)





# ============================================================

# ASPP (Atrous Spatial Pyramid Pooling)

# ============================================================

class ASPP(nn.Module):

    def __init__(self, in_ch, out_ch):

        super().__init__()

        self.c1 = nn.Conv2d(in_ch, out_ch, 1, bias=False)

        self.c2 = ConvBNAct(in_ch, out_ch, padding=6, dilation=6)

        self.c3 = ConvBNAct(in_ch, out_ch, padding=12, dilation=12)

        self.c4 = ConvBNAct(in_ch, out_ch, padding=18, dilation=18)

        self.project = nn.Sequential(

            nn.Conv2d(out_ch * 4, out_ch, 1, bias=False),

            nn.GroupNorm(num_groups=8, num_channels=out_ch),

            nn.GELU(),

            nn.Dropout2d(0.2)

        )

    def forward(self, x):

        return self.project(torch.cat([self.c1(x), self.c2(x), self.c3(x), self.c4(x)], dim=1))



# ============================================================

# BLOCOS U-NET (ENCODER / DECODER)

# ============================================================

class DownBlock(nn.Module):

    def __init__(self, in_ch, out_ch):

        super().__init__()

        self.block = nn.Sequential(

            ConvBNAct(in_ch, out_ch, stride=2),

            ResidualBlock(out_ch)

        )

    def forward(self, x):

        return self.block(x)



class UpBlock(nn.Module):

    def __init__(self, in_ch, skip_ch, out_ch):

        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.conv = nn.Sequential(

            ConvBNAct(in_ch + skip_ch, out_ch),

            ResidualBlock(out_ch)

        )

    def forward(self, x, skip):

        x_up = self.up(x)

        return self.conv(torch.cat([x_up, skip], dim=1))



# ============================================================

# ARQUITETURA PRINCIPAL FUNDIDA

# ============================================================

class SpatialStem(nn.Module):

    def __init__(self, in_ch=5, out_ch=64): # Rede recebe 5 canais (NDWI omitido)

        super().__init__()

        self.net = nn.Sequential(

            ConvBNAct(in_ch, out_ch, kernel_size=3, padding=1),

            ResidualBlock(out_ch),

            ConvBNAct(out_ch, out_ch, kernel_size=3, padding=1)

        )



    def forward(self, x):

        return self.net(x)



class UrbanHeatNet(nn.Module):

    def __init__(self, base_channels=64):

        super().__init__()

        # Entrada tem 5 canais: [NDVI, NDBI, DEM, Seno, Cosseno]

        self.stem = SpatialStem(in_ch=5, out_ch=base_channels)



        self.down1 = DownBlock(base_channels, base_channels * 2)

        self.down2 = DownBlock(base_channels * 2, base_channels * 4)



        self.aspp = ASPP(base_channels * 4, base_channels * 4)



        self.up2 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)

        self.up1 = UpBlock(base_channels * 2, base_channels, base_channels)



        self.head = nn.Sequential(

            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1),

            nn.GELU(),

            nn.Dropout2d(0.1),

            nn.Conv2d(base_channels // 2, 1, kernel_size=1, bias=True)

        )



    def forward(self, x):

        s1 = self.stem(x)

        s2 = self.down1(s1)

        deep = self.down2(s2)



        deep = self.aspp(deep)



        x_up = self.up2(deep, s2)

        x_up = self.up1(x_up, s1)



        return self.head(x_up)



# ============================================================

# ALGORITMO GENÉTICO COM MÁSCARA DE VIABILIDADE

# ============================================================

class GeneticPlantingSimulator:

    def __init__(self, model, device, dataset_stats, delta_stats=None):

        self.model = model

        self.device = device



        # Mapeamento com base nos índices do CHANNELS_TO_KEEP original

        self.mean_ndvi = dataset_stats[0][0]

        self.std_ndvi  = dataset_stats[1][0]



        self.mean_ndwi = dataset_stats[0][1]

        self.std_ndwi  = dataset_stats[1][1]



        self.mean_ndbi = dataset_stats[0][2]

        self.std_ndbi  = dataset_stats[1][2]



        # Define o kernel explicitamente como float32 para evitar conflitos no autocast

        self.spillover_kernel = torch.tensor([

            [0.15, 0.30, 0.15],

            [0.30, 1.00, 0.30],

            [0.15, 0.30, 0.15]

        ], dtype=torch.float32, device=device).view(1, 1, 3, 3)



        self.max_ndvi = 0.70  # Teto realista de NDVI urbano denso na região

        self.ndvi_impact = 0.35

        self.ndbi_reduction = 0.40



    def get_valid_planting_mask(self, sequence):

        recent_ndbi_z = sequence[0, 1, :, :].squeeze()

        recent_ndwi_z = sequence[0, 5, :, :].squeeze()



        # 1. Desnormaliza para valores físicos reais

        recent_ndbi_raw = (recent_ndbi_z * self.std_ndbi) + self.mean_ndbi

        recent_ndwi_raw = (recent_ndwi_z * self.std_ndwi) + self.mean_ndwi



        # 2. Aplica a restrição física do ARBORMAP

        valid_mask = (recent_ndwi_raw < 0.0) & (recent_ndbi_raw < 1.0)

        return torch.nonzero(valid_mask)



    def apply_intervention(self, sequence, coords_y, coords_x):

        seq_interv = sequence.clone()

        _, _, H, W = seq_interv.shape



        plant_mask = torch.zeros((1, 1, H, W), dtype=torch.float32, device=self.device)

        coords_y = coords_y.long()

        coords_x = coords_x.long()

        plant_mask[0, 0, coords_y, coords_x] = 1.0



        impact_map = F.conv2d(plant_mask, self.spillover_kernel, padding=1)

        impact_flat = impact_map.squeeze()



        current_ndvi_z = seq_interv[0, 0, :, :]



        # 1. Recupera o valor físico do NDVI

        current_ndvi_raw = (current_ndvi_z * self.std_ndvi) + self.mean_ndvi



        # 2. Calcula o crescimento no espaço físico

        room_to_grow = torch.clamp(self.max_ndvi - current_ndvi_raw, min=0.0)

        actual_boost_raw = torch.min(impact_flat * self.ndvi_impact, room_to_grow)



        # 3. Converte o impacto físico para a proporção Z-score

        boost_z_ndvi = actual_boost_raw / self.std_ndvi

        reduction_z_ndbi = (actual_boost_raw * self.ndbi_reduction) / self.std_ndbi



        # 4. Soma a proporção normalizada ao tensor

        seq_interv[0, 0, :, :] += boost_z_ndvi

        seq_interv[0, 1, :, :] -= reduction_z_ndbi



        return seq_interv



    @torch.no_grad()

    def evaluate_fitness(self, sequence, individuals):

        self.model.eval()

        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'



        with torch.amp.autocast(device_type):

            base_delta = self.model(sequence[:, :5, :, :]).squeeze()



        fitness_scores = np.zeros(len(individuals), dtype=np.float32)



        for i, coords in enumerate(individuals):

            # --- CORREÇÃO AQUI ---

            # Garante que coords seja um tensor no device correto

            if not isinstance(coords, torch.Tensor):

                coords = torch.tensor(coords, device=self.device)



            y_coords, x_coords = coords[:, 0], coords[:, 1]

            seq_interv = self.apply_intervention(sequence, y_coords, x_coords)

            # ---------------------



            with torch.amp.autocast(device_type):

                interv_delta = self.model(seq_interv[:, :5, :, :]).squeeze()



            impact = (base_delta - interv_delta) * self.std_delta

            fitness_scores[i] = impact.sum().item()



        return fitness_scores



    def optimize_genetic(self, sequence, num_trees, pop_size=50, generations=100, seed=42):

        # 1. Garantia de Determinismo

        if seed is not None:

            torch.manual_seed(seed)

            if torch.cuda.is_available():

                torch.cuda.manual_seed_all(seed)

            np.random.seed(seed)

            random.seed(seed)



        # 2. Definição da máscara de viabilidade (Corrigindo o erro)

        H, W = sequence.shape[-2:]

        valid_indices = self.get_valid_planting_mask(sequence)

        num_valid = len(valid_indices)



        # Cria a máscara booleana para consulta O(1)

        valid_mask_bool = torch.zeros((H, W), dtype=torch.bool, device=self.device)

        valid_mask_bool[valid_indices[:, 0], valid_indices[:, 1]] = True



        # Restante da inicialização da população...

        population = []

        for _ in range(pop_size):

            if num_valid >= num_trees:

                idx = torch.randperm(num_valid, device=self.device)[:num_trees]

                pts = valid_indices[idx]

            else:

                H, W = sequence.shape[-2:]

                pts = torch.stack([

                    torch.randint(0, H, (num_trees,), device=self.device),

                    torch.randint(0, W, (num_trees,), device=self.device)

                ], dim=1)

            population.append(pts)



        best_solution, best_fitness = None, -float('inf')



        print(f"Evoluindo {pop_size} indivíduos por {generations} gerações (Seed: {seed})...")

        for gen in range(generations):

            scores = self.evaluate_fitness(sequence, population)



            max_idx = int(np.argmax(scores))

            if scores[max_idx] > best_fitness:

                best_fitness = scores[max_idx]

                best_solution = population[max_idx].clone()



            # OTIMIZAÇÃO: Elitismo e reconstrução da população

            new_population = [best_solution.clone()]



            while len(new_population) < pop_size:

                # 1. SELEÇÃO (Torneio para escolher 2 pais diferentes)

                i1, i2 = random.sample(range(pop_size), 2)

                parent1 = population[i1] if scores[i1] > scores[i2] else population[i2]



                i3, i4 = random.sample(range(pop_size), 2)

                parent2 = population[i3] if scores[i3] > scores[i4] else population[i4]



                # 2. CRUZAMENTO / CROSSOVER (Uniforme)

                # Mistura as árvores do pai 1 com as do pai 2 (~50% de chance para cada)

                child = parent1.clone()

                cross_mask = torch.rand(num_trees, device=self.device) > 0.5

                child[cross_mask] = parent2[cross_mask]



                # 3. MUTAÇÃO (Por gene/árvore)

                # Damos a CADA árvore ~5% de chance de mudar de lugar

                mutation_rate = 0.05

                for mut_idx in range(num_trees):

                    if random.random() < mutation_rate:

                        if random.random() < 0.5:

                            # Mutação Local (Explotação): Move de 1 a 3 pixels

                            dy, dx = random.randint(-3, 3), random.randint(-3, 3)

                            new_y = max(0, min(H-1, child[mut_idx, 0].item() + dy))

                            new_x = max(0, min(W-1, child[mut_idx, 1].item() + dx))



                            if valid_mask_bool[new_y, new_x]:

                                child[mut_idx] = torch.tensor([new_y, new_x], device=self.device)

                            else:

                                child[mut_idx] = valid_indices[random.randint(0, num_valid - 1)]

                        else:

                            # Mutação Global (Exploração): Salta para um lugar aleatório

                            child[mut_idx] = valid_indices[random.randint(0, num_valid - 1)]



                # 4. PREVENÇÃO DE COLISÃO (Impede empilhamento no mesmo pixel)

                # Converte coordenadas (y, x) em índices 1D para achar os valores únicos

                flat_coords = child[:, 0] * W + child[:, 1]

                unique_coords = torch.unique(flat_coords)



                if len(unique_coords) < num_trees:

                    # Descobre quantas árvores "sumiram" por sobreposição

                    num_missing = num_trees - len(unique_coords)



                    # Reconstrói a criança apenas com coordenadas únicas

                    new_child = torch.zeros_like(child)

                    new_child[:len(unique_coords), 0] = unique_coords // W

                    new_child[:len(unique_coords), 1] = unique_coords % W



                    # Sorteia novas posições aleatórias para preencher a cota

                    idx = torch.randperm(num_valid, device=self.device)[:num_missing]

                    new_child[len(unique_coords):] = valid_indices[idx]

                    child = new_child



                new_population.append(child)



            population = new_population # Agora população tem tamanho pop_size

            print(f"Gen {gen}: Best Fitness = {best_fitness:.4f}")



        return best_solution[:, 0].tolist(), best_solution[:, 1].tolist(), best_fitness



# ============================================================

# TREINAMENTO E VALIDAÇÃO

# ============================================================

def fit_fold(model, train_loader, val_loader, epochs, fold_idx, global_stats):

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)



    loss_fn = PhysicsInformedLoss(phys_weight=0.0)

    scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    best_loss = float('inf')



    print(f"\n--- Iniciando FOLD {fold_idx + 1} ---")

    for epoch in range(epochs):

        model.train()

        train_loss = 0

        n_batches_t = 0



        for seq, target in train_loader:

            seq, target = seq.to(DEVICE), target.to(DEVICE)



            # O NDVI é o índice 0 da nova sequência

            ndvi_map = seq[:, 0, :, :].unsqueeze(1)



            # Cortamos a água (canal 5) e passamos apenas os 5 canais para a rede

            # print(f"Valor mínimo do NDWI na imagem: {seq[:, 5, :, :].min().item():.4f}")

            # print(f"Valor máximo do NDWI na imagem: {seq[:, 5, :, :].max().item():.4f}")

            model_input = seq[:, :5, :, :]



            optimizer.zero_grad()



            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):

                pred = model(model_input)

                total_loss, _, _ = loss_fn(pred, target, ndvi_map)



            if not torch.isfinite(total_loss): continue



            scaler.scale(total_loss).backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)

            scaler.update()



            train_loss += total_loss.item()

            n_batches_t += 1



        train_loss /= max(n_batches_t, 1)



        # --- VALIDAÇÃO ---

        model.eval()

        v_loss, v_mae = 0, 0

        n_batches_v = 0



        all_preds = []

        all_targets = []



        for seq, target in val_loader:

            seq, target = seq.to(DEVICE), target.to(DEVICE)

            ndvi_map = seq[:, 0, :, :].unsqueeze(1)



            # Cortamos a água para validação também

            model_input = seq[:, :5, :, :]



            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):

                pred = model(model_input)

                total_loss, _, _ = loss_fn(pred, target, ndvi_map)



            if not torch.isfinite(total_loss): continue



            v_loss += total_loss.item()

            v_mae += compute_mae(pred, target).item()

            n_batches_v += 1



            all_preds.append(pred.detach().cpu())

            all_targets.append(target.detach().cpu())



        n_v = max(n_batches_v, 1)

        v_loss /= n_v

        v_mae /= n_v



        all_preds_t = torch.cat(all_preds)

        all_targets_t = torch.cat(all_targets)



        v_r2 = compute_r2(all_preds_t, all_targets_t).item()

        v_pearson = compute_pearson(all_preds_t, all_targets_t).item()

        v_std_pred = torch.std(all_preds_t).item()

        v_std_target = torch.std(all_targets_t).item()



        scheduler.step(v_loss)



        print(f"Fold {fold_idx+1} | Epoch {epoch+1:02d} | T.Loss: {train_loss:.4f} | V.Loss: {v_loss:.4f} | MAE: {v_mae:.4f}")

        print(f"       -> R2: {v_r2:.4f} | Pearson: {v_pearson:.4f} | Std(Pred): {v_std_pred:.4f} vs Std(Target): {v_std_target:.4f}")



        if v_loss < best_loss:

            best_loss = v_loss

            checkpoint = {

                'model_state_dict': model.state_dict(),

                'mean': global_stats[0],

                'std': global_stats[1],

                'input_channels': 5 # Atualizado para 5

            }

            torch.save(checkpoint, f"best_model_fold_{fold_idx}.pt")



    return best_loss



def test_single_batch_overfit(model, train_loader, device, epochs=150):

    print("\n" + "="*60)

    print("🚀 INICIANDO TESTE DE OVERFIT EM UM ÚNICO BATCH")

    print("="*60)



    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    loss_fn = torch.nn.MSELoss()



    seq_batch, target_batch = next(iter(train_loader))

    seq_batch = seq_batch.to(device)

    target_batch = target_batch.to(device)



    target_std = torch.std(target_batch).item()

    print(f"Tamanho do Batch de Entrada: {seq_batch.shape}")

    print(f"Desvio Padrão do Alvo: {target_std:.4f}\n")



    model.eval()



    for epoch in range(epochs):

        optimizer.zero_grad()



        # O NDVI é o canal 0

        ndvi_map = seq_batch[:, 0, :, :].unsqueeze(1)



        # Corta o NDWI

        model_input = seq_batch[:, :5, :, :]



        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):

            pred = model(model_input)

            total_loss = loss_fn(pred, target_batch)



        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()



        if epoch % 10 == 0 or epoch == epochs - 1:

            with torch.no_grad():

                mae = compute_mae(pred, target_batch).item()

                r2 = compute_r2(pred, target_batch).item()

                pearson = compute_pearson(pred, target_batch).item()

                pred_std = torch.std(pred).item()



            print(f"Epoch {epoch:03d} | Loss: {total_loss.item():.4f} | MAE: {mae:.4f} | R2: {r2:.4f} | Pearson: {pearson:.4f} | Std(Pred): {pred_std:.4f}")



    print("="*60 + "\n")



# ============================================================

# EXECUÇÃO PRINCIPAL

# ============================================================

if __name__ == "__main__":

    all_cities = [d.name for d in Path(DATA_DIR).iterdir() if d.is_dir() and not d.name.startswith('.')]



    if not all_cities:

        print("Nenhuma cidade encontrada no diretório!")

        exit()



    print("\n1. Extraindo Estatísticas Globais...")



    if len(all_cities) < K_FOLDS:

        print(f"Poucas cidades. O K-Fold exige pelo menos {K_FOLDS}.")

        exit()



    base_ds = UrbanHeatDataset(DATA_DIR, allowed_cities=all_cities, normalize=True, is_train=False)

    global_stats = (base_ds.mean, base_ds.std)



    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

    fold_results = []



    print("\n2. Iniciando K-Fold Cross Validation Espacial...")

    # all_cities_array = np.array(all_cities) # Converte para numpy array para facilitar o indexamento



    # for fold_idx, (train_idx, val_idx) in enumerate(kf.split(all_cities_array)):

    #     # Seleciona os nomes usando os índices retornados pelo KFold

    #     train_cities = all_cities_array[train_idx]

    #     val_cities = all_cities_array[val_idx]



    #     print(f"\nFold {fold_idx + 1}:")

    #     print(f"  Treino ({len(train_cities)} cidades): {', '.join(train_cities)}")

    #     print(f"  Validação ({len(val_cities)} cidades): {', '.join(val_cities)}")

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_cities)):

        # Remoção do break "if fold < 4", agora treina todos.

        # if fold < 1:

        #   continue



        train_cities = [all_cities[i] for i in train_idx]

        val_cities = [all_cities[i] for i in val_idx]



        train_files = []

        for city in train_cities:

            folder = Path(DATA_DIR) / city

            if folder.exists():

                train_files.extend(sorted(folder.glob("*.npy")))



        print(f"Calculando estatísticas de normalização ({len(train_files)} arquivos de treino)...")

        fold_mean, fold_std = compute_dataset_statistics(train_files)



        fold_stats = (fold_mean, fold_std)



        train_ds = UrbanHeatDataset(

            DATA_DIR, allowed_cities=train_cities,

            stats=fold_stats, is_train=True

        )

        val_ds = UrbanHeatDataset(

            DATA_DIR, allowed_cities=val_cities,

            stats=fold_stats, is_train=False

        )



        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

        model = UrbanHeatNet().to(DEVICE)



        for m in model.modules():

            if isinstance(m, (nn.Conv2d, nn.Conv3d)):

                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

                if m.bias is not None: torch.nn.init.zeros_(m.bias)



        best_val = fit_fold(model, train_loader, val_loader, EPOCHS, fold, global_stats)

        fold_results.append(best_val)



        if fold == K_FOLDS - 1:

            print("\n3. Executando Algoritmo Genético no Fold de Teste...")

            model.load_state_dict(torch.load(f"best_model_fold_{fold}.pt", weights_only=False)['model_state_dict'])

            seq, _ = next(iter(val_loader))

            seq = seq[0:1].to(DEVICE)



            simulator = GeneticPlantingSimulator(model, DEVICE, global_stats)

            opt_y, opt_x, opt_cool_sum = simulator.optimize_genetic(seq, num_trees=50, pop_size=50, generations=100)



            with torch.no_grad():

                base_delta = simulator.model(seq[:, :5, :, :]).squeeze()

                seq_interv = simulator.apply_intervention(seq, opt_y, opt_x)

                interv_delta = simulator.model(seq_interv[:, :5, :, :]).squeeze()



                mapa_resfriamento = (base_delta - interv_delta) * simulator.std_delta

                resfriamento_total_celsius = mapa_resfriamento[mapa_resfriamento > 0.01].sum().item()



                print(f"\n--- Otimização Final  com Restrição Geográfica ---")

                print(f"Resfriamento Total do Projeto: {resfriamento_total_celsius:.2f} °C acumulados")

                print(f"Eficiência Média por Célula: {resfriamento_total_celsius/50:.3f} °C")



    print(f"\nValidação Cruzada Concluída. Média Val Loss: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")
