"""
Versão anterior do pipeline híbrido (com dataset temporal)

Origem: notebook arbormap.ipynb, célula 29 (Google Colab).
Mantida para rastreabilidade das decisões de projeto.
Código preservado como no notebook; caminhos do Colab/Drive devem ser ajustados nas CONFIGURAÇÕES.
"""

import os
import random
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import cv2
import functools
from concurrent.futures import ProcessPoolExecutor

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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "/content/drive/MyDrive/heat_dataset_v3"

SEQ_LEN = 2 # Quantidade de instantes de tempo no histórico
INPUT_CHANNELS = 8 # NDVI, NDWI, NDBI, DEM, LST_past, Coord_X, Coord_Y, Global_Mean_LST

CH_NDVI, CH_NDWI, CH_NDBI, CH_DEM, CH_LST = 0, 1, 2, 3, 4

BATCH_SIZE = 8 # Reduzido devido à adição da dimensão temporal
ACCUM_STEPS = 2
EPOCHS = 40
LR = 3e-4
WEIGHT_DECAY = 1e-4
K_FOLDS = 5
CROP_SIZE = 256

LST_CLIP_MIN, LST_CLIP_MAX = -20.0, 60.0

def ensure_channels_first(img):
    if img.ndim != 3: raise ValueError(f"Esperado array 3D, recebido {img.shape}")
    if img.shape[0] == 5: return img
    elif img.shape[-1] == 5: return np.transpose(img, (2, 0, 1))
    else: raise ValueError("Não foi possível identificar eixo de canais.")

# ============================================================
# FILTRO DE NUVENS
# ============================================================
def _check_cloud_fraction(filepath, temp_threshold, min_cloud_fraction):
    try:
        data = np.load(filepath)
        lst_band = ensure_channels_first(data)[CH_LST]
        cloud_fraction = np.sum(lst_band < temp_threshold) / lst_band.size
        return Path(filepath), cloud_fraction < min_cloud_fraction
    except Exception:
        return Path(filepath), False

def filter_cloudy_files_runtime(filepaths, temp_threshold=15.0, min_cloud_pct=30.0):
    valid_files = set()
    print(f"\n☁️ Filtrando {len(filepaths)} imagens...")
    check_func = functools.partial(_check_cloud_fraction, temp_threshold=temp_threshold, min_cloud_fraction=min_cloud_pct/100.0)
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(check_func, filepaths))
    for filepath, is_valid in results:
        if is_valid: valid_files.add(filepath)
    print(f" -> Imagens válidas mantidas: {len(valid_files)}\n")
    return valid_files

# ============================================================
# DATASET TEMPORAL COM CORREÇÃO FÍSICA E COORDCONV
# ============================================================
class UrbanHeatTemporalDataset(Dataset):
    def __init__(self, root, allowed_cities, valid_files_set, is_train=False):
        self.root = Path(root)
        self.is_train = is_train
        self.samples = []

        for city in allowed_cities:
            folder = self.root / city
            if not folder.exists(): continue
            files = sorted([f for f in folder.glob("*.npy") if f in valid_files_set])

            # Necessário SEQ_LEN imagens para entrada + 1 para o alvo
            if len(files) < SEQ_LEN + 1: continue

            for i in range(len(files) - SEQ_LEN):
                # (Lista de entradas, Arquivo Alvo)
                seq_files = files[i : i + SEQ_LEN]
                target_file = files[i + SEQ_LEN]
                self.samples.append((seq_files, target_file))

        # Criação dos grids de coordenadas para o CoordConv
        y_grid, x_grid = np.mgrid[-1:1:complex(0, CROP_SIZE), -1:1:complex(0, CROP_SIZE)]
        self.coord_y = y_grid.astype(np.float32)
        self.coord_x = x_grid.astype(np.float32)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        seq_files, target_file = self.samples[idx]

        # 1. Carrega as imagens completas
        seq_imgs = [ensure_channels_first(np.load(f).astype(np.float32)) for f in seq_files]
        tgt_img = ensure_channels_first(np.load(target_file).astype(np.float32))

        # 2. Definição do Crop Central ou Aleatório
        max_h, max_w = tgt_img.shape[1], tgt_img.shape[2]
        if self.is_train and max_h > CROP_SIZE and max_w > CROP_SIZE:
            y, x = random.randint(0, max_h - CROP_SIZE), random.randint(0, max_w - CROP_SIZE)
        else:
            y, x = max(0, (max_h - CROP_SIZE) // 2), max(0, (max_w - CROP_SIZE) // 2)

        # 3. Tratamento do Alvo (Anomalia baseada na CIDADE INTEIRA)
        is_nodata_tgt = (tgt_img[CH_NDVI] == 0.0) & (tgt_img[CH_NDBI] == 0.0)
        valid_mask_tgt_full = ~is_nodata_tgt

        valid_pixels_tgt = tgt_img[CH_LST][valid_mask_tgt_full]
        city_mean_target = valid_pixels_tgt.mean() if len(valid_pixels_tgt) > 0 else 0.0

        # Anomalia Global calculada antes do crop
        tgt_anomaly_full = (tgt_img[CH_LST] - city_mean_target) * valid_mask_tgt_full.astype(np.float32)

        cropped_tgt_anomaly = tgt_anomaly_full[y:y+CROP_SIZE, x:x+CROP_SIZE]
        crop_valid_mask = valid_mask_tgt_full[y:y+CROP_SIZE, x:x+CROP_SIZE].astype(np.float32)

        padded_tgt_anomaly = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float32)
        valid_mask = np.zeros((1, CROP_SIZE, CROP_SIZE), dtype=np.float32)
        padded_tgt_anomaly[:cropped_tgt_anomaly.shape[0], :cropped_tgt_anomaly.shape[1]] = cropped_tgt_anomaly
        valid_mask[0, :crop_valid_mask.shape[0], :crop_valid_mask.shape[1]] = crop_valid_mask

        target_norm = (np.clip(padded_tgt_anomaly, -20.0, 20.0) / 5.0) * valid_mask[0]

        # 4. Construção da Sequência (Histórico)
        seq_tensors = []
        for img_t in seq_imgs:
            # Calcula a média da cidade no instante t
            mask_t = ~((img_t[CH_NDVI] == 0.0) & (img_t[CH_NDBI] == 0.0))
            city_mean_t = img_t[CH_LST][mask_t].mean() if mask_t.sum() > 0 else 0.0
            anomaly_t = (img_t[CH_LST] - city_mean_t) * mask_t.astype(np.float32)

            cropped_base = img_t[:, y:y+CROP_SIZE, x:x+CROP_SIZE]
            cropped_anomaly_t = anomaly_t[y:y+CROP_SIZE, x:x+CROP_SIZE]

            padded_base = np.zeros((5, CROP_SIZE, CROP_SIZE), dtype=np.float32)
            padded_base[:, :cropped_base.shape[1], :cropped_base.shape[2]] = cropped_base

            padded_base[CH_NDVI] = np.clip(padded_base[CH_NDVI], -1.0, 1.0)
            padded_base[CH_NDWI] = np.clip(padded_base[CH_NDWI], -1.0, 1.0)
            padded_base[CH_NDBI] = np.clip(padded_base[CH_NDBI], -1.0, 1.0)

            if valid_mask.sum() > 0:
                local_dem_mean = padded_base[CH_DEM][valid_mask[0] == 1.0].mean()
                padded_base[CH_DEM] = (padded_base[CH_DEM] - local_dem_mean) / 50.0

            # Anomalia LST do passado normalizada
            lst_past_norm = np.clip(cropped_anomaly_t, -20.0, 20.0) / 5.0
            lst_past_padded = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float32)
            lst_past_padded[:lst_past_norm.shape[0], :lst_past_norm.shape[1]] = lst_past_norm

            # Global Mean (Proxy Meteorológico) expandido espacialmente
            global_mean_layer = np.full((CROP_SIZE, CROP_SIZE), city_mean_t / 40.0, dtype=np.float32)

            # Empilha: [NDVI, NDWI, NDBI, DEM, LST_past_anomaly, CoordY, CoordX, GlobalMean]
            step_features = np.stack([
                padded_base[CH_NDVI] * valid_mask[0],
                padded_base[CH_NDWI] * valid_mask[0],
                padded_base[CH_NDBI] * valid_mask[0],
                padded_base[CH_DEM] * valid_mask[0],
                lst_past_padded * valid_mask[0],
                self.coord_y * valid_mask[0],
                self.coord_x * valid_mask[0],
                global_mean_layer * valid_mask[0]
            ], axis=0)

            seq_tensors.append(torch.from_numpy(step_features).float())

        # seq_out shape: (Time, Channels, H, W)
        seq_out = torch.stack(seq_tensors, dim=0)
        out_tgt = torch.from_numpy(target_norm).unsqueeze(0).float()
        out_valid = torch.from_numpy(valid_mask).float()

        # Augmentation (aplicado consistentemente em todo o volume de tempo)
        if self.is_train:
            if random.random() > 0.5:
                seq_out = torch.flip(seq_out, dims=[-1])
                out_tgt, out_valid = torch.flip(out_tgt, dims=[-1]), torch.flip(out_valid, dims=[-1])
            if random.random() > 0.5:
                seq_out = torch.flip(seq_out, dims=[-2])
                out_tgt, out_valid = torch.flip(out_tgt, dims=[-2]), torch.flip(out_valid, dims=[-2])
            k = random.randint(0, 3)
            if k > 0:
                seq_out = torch.rot90(seq_out, k, dims=[-2, -1])
                out_tgt, out_valid = torch.rot90(out_tgt, k, dims=[-2, -1]), torch.rot90(out_valid, k, dims=[-2, -1])

        return seq_out, out_tgt, out_valid, city_mean_target

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBNAct(in_channels, out_channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels)
        )
        self.act = nn.GELU()

        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.GroupNorm(num_groups=8, num_channels=out_channels)
            )

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        return self.act(out + self.shortcut(x))

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=False),
            nn.GroupNorm(8, F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=False),
            nn.GroupNorm(8, F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.act = nn.GELU()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)

        psi_out = self.psi(self.act(g1 + x1))
        return x * psi_out

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = ResidualBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x_up = self.up(x)
        return self.conv(torch.cat([x_up, skip], dim=1))

class HybridUrbanHeatNet(nn.Module):
    def __init__(self, in_channels=4, base_c=64):
        super().__init__()
        self.stem = ResidualBlock(in_channels, base_c)

        # Encoder
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(base_c, base_c*2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(base_c*2, base_c*4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(base_c*4, base_c*8))

        # Bottleneck estruturado exatamente como o checkpoint espera:
        # - conv1: ConvBNAct (gera chaves bottle.conv1.block.*)
        # - conv2: Sequential direto (gera chaves bottle.conv2.0.* e bottle.conv2.1.*)
        self.bottle = nn.ModuleDict({
            'conv1': ConvBNAct(base_c*8, base_c*8),
            'conv2': nn.Sequential(
                nn.Conv2d(base_c*8, base_c*8, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(num_groups=8, num_channels=base_c*8)
            )
        })

        # Decoder + Attention Gates
        self.att3 = AttentionBlock(base_c*8, base_c*4, base_c*4)
        self.up3 = UpBlock(base_c*8, base_c*4, base_c*4)

        self.att2 = AttentionBlock(base_c*4, base_c*2, base_c*2)
        self.up2 = UpBlock(base_c*4, base_c*2, base_c*2)

        self.att1 = AttentionBlock(base_c*2, base_c, base_c)
        self.up1 = UpBlock(base_c*2, base_c, base_c)

        # Head
        self.head = nn.Sequential(
            nn.Conv2d(base_c, base_c//2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_c//2, 1, 1)
        )

    def forward(self, x):
        s1 = self.stem(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)

        # Bottleneck forward
        d3 = self.down3(s3)
        b_feat = self.bottle['conv1'](d3)
        bottle = self.bottle['conv2'](b_feat)

        u3 = self.up3(bottle, self.att3(bottle, s3))
        u2 = self.up2(u3, self.att2(u3, s2))
        u1 = self.up1(u2, self.att1(u2, s1))

        return self.head(u1)

class BaselineLoss(nn.Module):
    def __init__(self, hotspot_w=2.0, top_q=0.9):
        super().__init__()
        self.hotspot_w = hotspot_w
        self.top_q = top_q

    def forward(self, pred, target, valid_mask):
        pred = pred * valid_mask
        target = target * valid_mask

        l1_diff = torch.abs(pred - target)
        valid_count = valid_mask.sum() + 1e-8
        base_loss = l1_diff.sum() / valid_count

        hotspot_mask = torch.zeros_like(target)
        for i in range(target.size(0)):
            valid_pixels = target[i][valid_mask[i] == 1.0]
            if valid_pixels.numel() > 0:
                q90 = torch.quantile(valid_pixels, self.top_q)
                hotspot_mask[i] = (target[i] >= q90).float() * valid_mask[i]

        hs_count = hotspot_mask.sum() + 1e-8
        hotspot_loss = (l1_diff * hotspot_mask).sum() / hs_count if hotspot_mask.sum() > 0 else torch.tensor(0.0, device=target.device)

        return base_loss + (self.hotspot_w * hotspot_loss)

# ============================================================
# INTERPRETABILIDADE: GRAD-CAM PARA CONV-LSTM
# ============================================================
class RegressionGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate(self, input_tensor):
        self.model.eval()
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output.sum()
        score.backward()

        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().detach().numpy()

        if cam.ndim == 3: cam = cam[0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, output.squeeze().cpu().detach().numpy()

def plot_grad_cam(input_tensor, cam_map, pred_norm, real_norm, city_mean, save_path="grad_cam.png"):
    # Pegamos o NDVI do último instante de tempo do histórico (t-1)
    ndvi_input = input_tensor[0, -1, CH_NDVI].cpu().numpy()

    # Reconstrução LST Absoluta usando a média correta do Batch/Crop
    pred_celsius = (pred_norm * 5.0) + city_mean
    real_celsius = (real_norm * 5.0) + city_mean

    if pred_celsius.ndim == 3: pred_celsius = pred_celsius[0]
    if real_celsius.ndim == 3: real_celsius = real_celsius[0]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(ndvi_input, cmap='gray')
    axes[0].set_title('NDVI (Instante Atual)')

    heatmap = cv2.applyColorMap(np.uint8(255 * cam_map), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    axes[1].imshow(ndvi_input, cmap='gray', alpha=0.5)
    axes[1].imshow(heatmap, alpha=0.6)
    axes[1].set_title('Atenção Espaço-Temporal')

    im2 = axes[2].imshow(pred_celsius, cmap='inferno')
    axes[2].set_title('Temp. Prevista (°C)')
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    im3 = axes[3].imshow(real_celsius, cmap='inferno')
    axes[3].set_title('Temp. Real (°C)')
    fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================
if __name__ == "__main__":
    all_cities = [d.name for d in Path(DATA_DIR).iterdir() if d.is_dir() and not d.name.startswith('.')]
    all_files_raw = list(Path(DATA_DIR).rglob("*.npy"))
    valid_files_set = filter_cloudy_files_runtime(all_files_raw)

    print("\n2. Iniciando K-Fold Cross Validation Espaço-Temporal...")
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    folds_list = list(kf.split(all_cities))

    for fold, (train_idx, val_idx) in enumerate(folds_list):
        train_cities = [all_cities[i] for i in train_idx]
        val_cities = [all_cities[i] for i in val_idx]

        train_ds = UrbanHeatTemporalDataset(DATA_DIR, allowed_cities=train_cities, valid_files_set=valid_files_set, is_train=True)
        val_ds = UrbanHeatTemporalDataset(DATA_DIR, allowed_cities=val_cities, valid_files_set=valid_files_set, is_train=False)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

        model = HybridUrbanHeatNet(in_channels=INPUT_CHANNELS).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6)

        loss_fn = BaselineLoss()
        best_loss = float('inf')

        print(f"\n--- Iniciando FOLD {fold + 1} ---")
        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0
            optimizer.zero_grad()

            for step, (seq, target, valid, city_mean) in enumerate(train_loader):
                seq, target, valid = seq.to(DEVICE), target.to(DEVICE), valid.to(DEVICE)

                pred = model(seq)
                loss = loss_fn(pred, target, valid)
                (loss / ACCUM_STEPS).backward()
                train_loss += loss.item()

                if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            model.eval()
            val_loss, val_mae_c, val_hs_mae_c, val_iou = 0, 0, 0, 0

            with torch.no_grad():
                for seq, target, valid, city_mean in val_loader:
                    seq, target, valid = seq.to(DEVICE), target.to(DEVICE), valid.to(DEVICE)
                    pred = model(seq)

                    val_loss += loss_fn(pred, target, valid).item()

                    # Passando o city_mean do batch para reconstrução térmica correta
                    mae_c, hs_mae_c, iou = compute_metrics(pred, target, valid, city_mean.to(DEVICE), top_q=0.9)
                    val_mae_c += mae_c
                    val_hs_mae_c += hs_mae_c
                    val_iou += iou

            n_t, n_v = max(len(train_loader), 1), max(len(val_loader), 1)
            train_loss /= n_t
            val_loss /= n_v
            scheduler.step(val_loss)

            print(f"Epoch {epoch+1:02d} | T.Loss: {train_loss:.4f} | V.Loss: {val_loss:.4f} | MAE (°C): {val_mae_c/n_v:.4f} | HS MAE: {val_hs_mae_c/n_v:.4f} | IoU: {val_iou/n_v:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), f"best_convlstm_fold_{fold}.pt")

        # =======================================================
        # GERAÇÃO DO GRAD-CAM NO FIM DO FOLD
        # =======================================================
        print(f"\nGerando Grad-CAM Espaço-Temporal para o Fold {fold+1}...")

        model.load_state_dict(torch.load(f"best_convlstm_fold_{fold}.pt", weights_only=True))
        cam_generator = RegressionGradCAM(model, target_layer=model.ups[-1])

        sample_seq, sample_tgt, sample_valid, sample_city_mean = next(iter(val_loader))

        cam_map, pred_np = cam_generator.generate(sample_seq[0:1].to(DEVICE))
        real_np = sample_tgt[0:1].cpu().numpy()
        city_mean_val = sample_city_mean[0].item()

        plot_grad_cam(
            sample_seq[0:1],
            cam_map,
            pred_np,
            real_np,
            city_mean_val,
            save_path=f"gradcam_convlstm_fold_{fold+1}.png"
        )
        print(f"-> Imagem do Grad-CAM salva como 'gradcam_convlstm_fold_{fold+1}.png'\n")
