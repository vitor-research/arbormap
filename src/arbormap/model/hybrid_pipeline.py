"""
Pipeline principal: UrbanHeatNet (Attention U-Net híbrida), dataset, perdas, treino K-Fold por cidade

Origem: notebook arbormap.ipynb, célula 32 (Google Colab).
Versão final usada no artigo: blocos residuais + portões de atenção, Group Normalization, HybridPhysicsLoss (erro + gradiente espacial + SSIM), validação cruzada espacial k=5, Grad-CAM.
Código preservado como no notebook; caminhos do Colab/Drive devem ser ajustados nas CONFIGURAÇÕES.
"""

import os
import random
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
import math
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

TOTAL_RAW_CHANNELS = 5
CHANNELS_TO_KEEP = [0, 1, 2, 3, 4]
INPUT_CHANNELS = len(CHANNELS_TO_KEEP)

CH_NDVI, CH_NDWI, CH_NDBI, CH_DEM, CH_LST = 0, 1, 2, 3, 4

BATCH_SIZE = 2
EPOCHS = 30
LR = 5e-4
WEIGHT_DECAY = 1e-4
K_FOLDS = 5

LST_CLIP_MIN, LST_CLIP_MAX = -20.0, 60.0

def ensure_channels_first(img):
    if img.ndim != 3: raise ValueError(f"Esperado array 3D, recebido {img.shape}")
    if img.shape[0] == TOTAL_RAW_CHANNELS: return img
    elif img.shape[-1] == TOTAL_RAW_CHANNELS: return np.transpose(img, (2, 0, 1))
    else: raise ValueError("Não foi possível identificar eixo de canais.")

# ============================================================
# FILTRO DE NUVENS EM PARALELO
# ============================================================
def _check_cloud_fraction(filepath, temp_threshold, min_cloud_fraction):
    try:
        data = np.load(filepath)
        lst_band = ensure_channels_first(data)[CH_LST]
        cloud_fraction = np.sum(lst_band < temp_threshold) / lst_band.size
        is_valid = cloud_fraction < min_cloud_fraction
        return Path(filepath), is_valid
    except Exception:
        return Path(filepath), False

def filter_cloudy_files_runtime(filepaths, temp_threshold=15.0, min_cloud_pct=30.0):
    min_cloud_fraction = min_cloud_pct / 100.0
    valid_files = set()
    discarded_count = 0

    print(f"\n☁️ Filtrando {len(filepaths)} imagens (Limite de Nuvem: >{min_cloud_pct}%) em paralelo...")

    check_func = functools.partial(_check_cloud_fraction,
                                   temp_threshold=temp_threshold,
                                   min_cloud_fraction=min_cloud_fraction)

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(check_func, filepaths))

    for filepath, is_valid in results:
        if is_valid:
            valid_files.add(filepath)
        else:
            discarded_count += 1

    print(f" -> Imagens descartadas (Muito nubladas): {discarded_count}")
    print(f" -> Imagens válidas mantidas: {len(valid_files)}\n")

    return valid_files

# ============================================================
# ESTATÍSTICAS E NORMALIZAÇÃO
# ============================================================
def compute_dataset_statistics(files):
    count = 0
    sum_ = np.zeros(INPUT_CHANNELS, dtype=np.float64)
    sum_sq = np.zeros(INPUT_CHANNELS, dtype=np.float64)
    for f in files:
        img = ensure_channels_first(np.load(f).astype(np.float32))[CHANNELS_TO_KEEP]
        img[CH_LST] = np.clip(img[CH_LST], LST_CLIP_MIN, LST_CLIP_MAX)
        img[CH_NDVI] = np.clip(img[CH_NDVI], -1.0, 1.0)
        img[CH_NDWI] = np.clip(img[CH_NDWI], -1.0, 1.0)
        flat = img.reshape(INPUT_CHANNELS, -1).astype(np.float64)
        sum_ += flat.sum(axis=1)
        sum_sq += (flat ** 2).sum(axis=1)
        count += flat.shape[1]
    mean = sum_ / count
    var = np.clip(sum_sq / count - mean ** 2, a_min=0.0, a_max=None)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)

class Normalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(INPUT_CHANNELS, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(INPUT_CHANNELS, 1, 1) + 1e-6
    def __call__(self, img):
        if isinstance(img, np.ndarray): img = torch.from_numpy(img).float()
        return (img - self.mean) / self.std

# ============================================================
# DATASET
# ============================================================
class UrbanHeatDataset(Dataset):
    def __init__(self, root, allowed_cities, valid_files_set, normalize=True, stats=None, is_train=False):
        self.root = Path(root)
        self.normalize = normalize
        self.is_train = is_train
        self.samples = []
        all_files = []

        for city in allowed_cities:
            folder = self.root / city
            if not folder.exists(): continue

            files = sorted([f for f in folder.glob("*.npy") if f in valid_files_set])
            all_files.extend(files)

            if len(files) < 2: continue
            for i in range(len(files) - 1):
                self.samples.append((files[i], files[i + 1]))

        if normalize and len(all_files) > 0:
            if stats is not None:
                self.mean, self.std = stats
            else:
                self.mean, self.std = compute_dataset_statistics(all_files)
            self.normalizer = Normalize(self.mean, self.std)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        base_file, target_file = self.samples[idx]
        base_img = ensure_channels_first(np.load(base_file).astype(np.float32))[CHANNELS_TO_KEEP]
        tgt_img = ensure_channels_first(np.load(target_file).astype(np.float32))[CHANNELS_TO_KEEP]

        max_h = min(base_img.shape[1], tgt_img.shape[1])
        max_w = min(base_img.shape[2], tgt_img.shape[2])
        crop_size = 400

        if self.is_train and max_h > crop_size and max_w > crop_size:
            y = random.randint(0, max_h - crop_size)
            x = random.randint(0, max_w - crop_size)
        else:
            y, x = max(0, (max_h - crop_size) // 2), max(0, (max_w - crop_size) // 2)

        cropped_base = base_img[:, y:y+crop_size, x:x+crop_size]
        padded_base = np.zeros((INPUT_CHANNELS, crop_size, crop_size), dtype=np.float32)
        padded_base[:, :cropped_base.shape[1], :cropped_base.shape[2]] = cropped_base

        valid_mask = np.zeros((1, crop_size, crop_size), dtype=np.float32)
        valid_mask[0, :cropped_base.shape[1], :cropped_base.shape[2]] = 1.0

        padded_base[CH_LST] = np.clip(padded_base[CH_LST], LST_CLIP_MIN, LST_CLIP_MAX)
        padded_base[CH_NDVI] = np.clip(padded_base[CH_NDVI], -1.0, 1.0)
        padded_base[CH_NDWI] = np.clip(padded_base[CH_NDWI], -1.0, 1.0)

        if self.normalize:
            padded_base = self.normalizer(padded_base)
            if isinstance(padded_base, torch.Tensor):
                padded_base = padded_base.numpy()

        padded_base = padded_base * valid_mask

        if isinstance(padded_base, np.ndarray):
            padded_base = torch.from_numpy(padded_base).float()

        model_features = padded_base[[CH_NDVI, CH_NDBI, CH_DEM], :, :]
        water_mask = padded_base[[CH_NDWI], :, :]
        out_seq = torch.cat([model_features, water_mask], dim=0)

        cropped_tgt = tgt_img[CH_LST, y:y+crop_size, x:x+crop_size]
        padded_tgt_lst = np.zeros((crop_size, crop_size), dtype=np.float32)
        padded_tgt_lst[:cropped_tgt.shape[0], :cropped_tgt.shape[1]] = cropped_tgt
        padded_tgt_lst = np.clip(padded_tgt_lst, LST_CLIP_MIN, LST_CLIP_MAX)

        tgt_tensor = torch.from_numpy(padded_tgt_lst)
        valid_mask_tensor = torch.from_numpy(valid_mask[0])

        valid_tgt = tgt_tensor[valid_mask_tensor == 1.0]
        macro_climate = valid_tgt.mean() if valid_tgt.numel() > 0 else tgt_tensor.mean()

        if valid_tgt.numel() > 1:
            std_anomaly = valid_tgt.std() + 1e-6
        else:
            std_anomaly = torch.tensor(1.0)

        spatial_anomaly = (tgt_tensor - macro_climate) * valid_mask_tensor
        out_tgt = (spatial_anomaly / std_anomaly).unsqueeze(0).float()
        out_valid = torch.tensor(valid_mask).float()

        # ========================================================
        # DATA AUGMENTATION (Apenas no treino para generalização)
        # ========================================================
        if self.is_train:
            # Flip Horizontal
            if random.random() > 0.5:
                out_seq = torch.flip(out_seq, dims=[-1])
                out_tgt = torch.flip(out_tgt, dims=[-1])
                out_valid = torch.flip(out_valid, dims=[-1])

            # Flip Vertical
            if random.random() > 0.5:
                out_seq = torch.flip(out_seq, dims=[-2])
                out_tgt = torch.flip(out_tgt, dims=[-2])
                out_valid = torch.flip(out_valid, dims=[-2])

            # Rotação 90/180/270
            k = random.randint(0, 3)
            if k > 0:
                out_seq = torch.rot90(out_seq, k, dims=[-2, -1])
                out_tgt = torch.rot90(out_tgt, k, dims=[-2, -1])
                out_valid = torch.rot90(out_valid, k, dims=[-2, -1])

        return out_seq, out_tgt, out_valid, std_anomaly.clone().detach().float()

# ============================================================
# BLOCOS CONSTRUTIVOS
# ============================================================
class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, padding=1, dilation=1, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation=dilation, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_ch),
            nn.GELU()
        )
    def forward(self, x): return self.block(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBNAct(in_channels, out_channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels)
        )
        self.act = nn.GELU()
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.GroupNorm(num_groups=8, num_channels=out_channels)
            )
    def forward(self, x):
        return self.act(self.conv2(self.conv1(x)) + self.shortcut(x))

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1, bias=False), nn.GroupNorm(8, F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1, bias=False), nn.GroupNorm(8, F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1, bias=False), nn.BatchNorm2d(1), nn.Sigmoid())
        self.act = nn.GELU()

    def forward(self, g, x):
        g1, x1 = self.W_g(g), self.W_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        psi = self.act(g1 + x1)
        return x * self.psi(psi)

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = ResidualBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        return self.conv(torch.cat([self.up(x), skip], dim=1))

class HybridUrbanHeatNet(nn.Module):
    def __init__(self, in_channels=4, base_c=64):
        super().__init__()
        self.stem = ResidualBlock(in_channels, base_c)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(base_c, base_c*2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(base_c*2, base_c*4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(base_c*4, base_c*8))

        # Gargalo baseado em ResidualBlock puro (sem ASPP)
        self.bottle = ResidualBlock(base_c*8, base_c*8)

        self.att3 = AttentionBlock(F_g=base_c*8, F_l=base_c*4, F_int=base_c*4)
        self.up3 = UpBlock(base_c*8, base_c*4, base_c*4)

        self.att2 = AttentionBlock(F_g=base_c*4, F_l=base_c*2, F_int=base_c*2)
        self.up2 = UpBlock(base_c*4, base_c*2, base_c*2)

        self.att1 = AttentionBlock(F_g=base_c*2, F_l=base_c, F_int=base_c)
        self.up1 = UpBlock(base_c*2, base_c, base_c)

        self.head = nn.Sequential(
            nn.Conv2d(base_c, base_c//2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_c//2, 1, 1)
        )

    def forward(self, x):
        s1 = self.stem(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        d_deep = self.down3(s3)

        bottle = self.bottle(d_deep)

        x3 = self.att3(g=bottle, x=s3)
        u3 = self.up3(bottle, x3)

        x2 = self.att2(g=u3, x=s2)
        u2 = self.up2(u3, x2)

        x1 = self.att1(g=u2, x=s1)
        u1 = self.up1(u2, x1)

        return self.head(u1)

# ============================================================
# MÉTRICAS E LOSS
# ============================================================
def compute_metrics(pred, target, valid_mask, std_anomaly, top_q=0.9):
    std_expanded = std_anomaly.view(-1, 1, 1, 1)
    pred_c = pred * std_expanded
    target_c = target * std_expanded

    valid_count = valid_mask.sum() + 1e-8
    mae_celsius = (torch.abs(pred_c - target_c) * valid_mask).sum() / valid_count

    batch_size = target.size(0)
    intersection = 0.0
    union = 0.0
    hotspot_mae = 0.0
    hs_count = 0.0

    for i in range(batch_size):
        valid_pixels = target_c[i][valid_mask[i] == 1.0]
        if valid_pixels.numel() > 0:
            q90_real = torch.quantile(valid_pixels, top_q)

            hs_real_mask = (target_c[i] >= q90_real) & (valid_mask[i] == 1.0)
            hs_pred_mask = (pred_c[i] >= q90_real) & (valid_mask[i] == 1.0)

            intersection += (hs_pred_mask & hs_real_mask).float().sum()
            union += (hs_pred_mask | hs_real_mask).float().sum()

            if hs_real_mask.sum() > 0:
                hotspot_mae += (torch.abs(pred_c[i] - target_c[i]) * hs_real_mask.float()).sum()
                hs_count += hs_real_mask.float().sum()

    iou = (intersection + 1e-6) / (union + 1e-6)
    hotspot_mae_celsius = hotspot_mae / (hs_count + 1e-8)

    return mae_celsius.item(), hotspot_mae_celsius.item(), iou.item()

def create_gaussian_window(window_size, sigma=1.5):
    # Create a 1D Gaussian
    coords = torch.arange(window_size, dtype=torch.float32)
    coords -= window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    # Create a 2D Gaussian by outer product
    g2d = g.unsqueeze(1) * g.unsqueeze(0)
    # Reshape to [out_channels, in_channels/groups, H, W]
    window = g2d.unsqueeze(0).unsqueeze(0)
    return window

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, data_range=6.0):
        super().__init__()
        self.window_size = window_size
        self.data_range = data_range
        self.register_buffer('window', create_gaussian_window(window_size))

    def forward(self, pred, target, valid_mask):
        # Ensure window is on the same device as the inputs
        window = self.window.to(pred.device)

        # Calculate local means
        mu1 = F.conv2d(pred, window, padding=self.window_size//2)
        mu2 = F.conv2d(target, window, padding=self.window_size//2)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        # Calculate local variances and covariances
        sigma1_sq = F.conv2d(pred * pred, window, padding=self.window_size//2) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=self.window_size//2) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=self.window_size//2) - mu1_mu2

        # SSIM constants based on data range
        C1 = (0.01 * self.data_range) ** 2
        C2 = (0.03 * self.data_range) ** 2

        # SSIM formula
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        # SSIM is 1 for perfect match. Loss is (1 - SSIM). Apply mask to ignore padding.
        loss_map = (1.0 - ssim_map) / 2.0

        valid_count = valid_mask.sum() + 1e-8
        return (loss_map * valid_mask).sum() / valid_count

class HybridPhysicsLoss(nn.Module):
    def __init__(self, ssim_w=0.8, l1_w=0.2, hotspot_w=0.1, top_q=0.9):
        super().__init__()
        self.l1_loss = nn.L1Loss(reduction='none')
        self.ssim_loss = SSIMLoss(window_size=11, data_range=6.0)

        self.ssim_w = ssim_w
        self.l1_w = l1_w
        self.hotspot_w = hotspot_w
        self.top_q = top_q

    def forward(self, pred, target, ndvi_map, valid_mask):
        valid_count = valid_mask.sum() + 1e-8

        # 1. L1 Base Loss (Anchors absolute values)
        l1_map = self.l1_loss(pred, target) * valid_mask
        l1_loss_val = l1_map.sum() / valid_count

        # 2. SSIM Loss (Forces structural patterns)
        ssim_loss_val = self.ssim_loss(pred, target, valid_mask)

        # 3. Hotspot Loss
        hotspot_mask = torch.zeros_like(target)
        for i in range(target.size(0)):
            valid_pixels = target[i][valid_mask[i] == 1.0]
            if valid_pixels.numel() > 0:
                q90 = torch.quantile(valid_pixels, self.top_q)
                hotspot_mask[i] = (target[i] >= q90).float() * valid_mask[i]

        if hotspot_mask.sum() > 0:
            hotspot_loss_val = (l1_map * hotspot_mask).sum() / hotspot_mask.sum()
        else:
            hotspot_loss_val = torch.tensor(0.0, device=target.device)

        # Final Combined Loss
        total_loss = (self.ssim_w * ssim_loss_val) + (self.l1_w * l1_loss_val) + (self.hotspot_w * hotspot_loss_val)

        return total_loss

# ============================================================
# GRAD-CAM & OVERFIT
# ============================================================
class RegressionGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output): self.activations = output
    def save_gradient(self, module, grad_in, grad_out): self.gradients = grad_out[0]

    def generate(self, input_tensor):
        self.model.eval()
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output.sum()
        score.backward()

        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().detach().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, output.squeeze().cpu().detach().numpy()

def plot_grad_cam(input_tensor, cam_map, pred_anomaly, real_anomaly, save_path="grad_cam.png"):
    ndvi_input = input_tensor[0, 0].cpu().numpy()
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(ndvi_input, cmap='gray')
    axes[0].set_title('NDVI Input (Features)')

    heatmap = cv2.applyColorMap(np.uint8(255 * cam_map), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    axes[1].imshow(ndvi_input, cmap='gray', alpha=0.5)
    axes[1].imshow(heatmap, alpha=0.6)
    axes[1].set_title('Atenção da Rede (Grad-CAM)')

    im2 = axes[2].imshow(pred_anomaly, cmap='inferno')
    axes[2].set_title('Anomalia Prevista (Microclima)')
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    im3 = axes[3].imshow(real_anomaly, cmap='inferno')
    axes[3].set_title('Anomalia Real (Microclima)')
    fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def test_single_batch_overfit(model, train_loader, device, epochs=50):
    print("\n" + "="*70 + "\n🚀 TESTE DE SANIDADE: OVERFIT EM 1 BATCH\n" + "="*70)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)
    loss_fn = HybridPhysicsLoss(hotspot_w=0.1)

    seq, target, valid, std_anomaly = next(iter(train_loader))

    seq = seq.to(device)
    target = target.to(device)
    valid = valid.to(device)
    std_anomaly = std_anomaly.to(device)

    model_input = seq
    ndvi = seq[:, 0:1, :, :]

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        pred = model(model_input)
        loss = loss_fn(pred, target, ndvi, valid)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                mae_c, hs_mae_c, iou = compute_metrics(pred, target, valid, std_anomaly)
            print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f} | MAE (°C): {mae_c:.4f} | HS MAE (°C): {hs_mae_c:.4f} | HS IoU: {iou:.4f}")

    print("="*70)

# ============================================================
# LOOP PRINCIPAL
# ============================================================
if __name__ == "__main__":
    all_cities = [d.name for d in Path(DATA_DIR).iterdir() if d.is_dir() and not d.name.startswith('.')]
    if len(all_cities) < K_FOLDS: exit("Cidades insuficientes para K-Fold.")

    all_files_raw = list(Path(DATA_DIR).rglob("*.npy"))
    valid_files_set = filter_cloudy_files_runtime(all_files_raw, temp_threshold=15.0, min_cloud_pct=30.0)

    if not valid_files_set: exit("Nenhum arquivo válido após o filtro de nuvens!")

    print("1. Extraindo Estatísticas Globais (Apenas Arquivos Válidos)...")
    base_ds = UrbanHeatDataset(DATA_DIR, allowed_cities=all_cities, valid_files_set=valid_files_set, normalize=True, is_train=False)
    global_stats = (base_ds.mean, base_ds.std)

    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

    temp_train = UrbanHeatDataset(DATA_DIR, allowed_cities=all_cities, valid_files_set=valid_files_set, stats=global_stats, is_train=True)
    temp_loader = DataLoader(temp_train, batch_size=BATCH_SIZE, shuffle=True)
    temp_model = HybridUrbanHeatNet(in_channels=4).to(DEVICE)
    test_single_batch_overfit(temp_model, temp_loader, DEVICE, epochs=50)
    del temp_model, temp_loader, temp_train

    print("\n2. Iniciando K-Fold Cross Validation Espacial...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(all_cities)):
        train_cities = [all_cities[i] for i in train_idx]
        val_cities = [all_cities[i] for i in val_idx]

        train_ds = UrbanHeatDataset(DATA_DIR, allowed_cities=train_cities, valid_files_set=valid_files_set, stats=global_stats, is_train=True)
        val_ds = UrbanHeatDataset(DATA_DIR, allowed_cities=val_cities, valid_files_set=valid_files_set, stats=global_stats, is_train=False)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

        model = HybridUrbanHeatNet(in_channels=4).to(DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        loss_fn = HybridPhysicsLoss(hotspot_w=0.1)
        best_loss = float('inf')

        print(f"\n--- Iniciando FOLD {fold + 1} ---")
        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0

            for seq, target, valid, std_anomaly in train_loader:
                seq, target, valid, std_anomaly = seq.to(DEVICE), target.to(DEVICE), valid.to(DEVICE), std_anomaly.to(DEVICE)
                optimizer.zero_grad()

                pred = model(seq)

                loss = loss_fn(pred, target, seq[:, 0:1, :, :], valid)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            val_loss, val_mae_c, val_hs_mae_c, val_iou = 0, 0, 0, 0

            with torch.no_grad():
                for seq, target, valid, std_anomaly in val_loader:
                    seq, target, valid, std_anomaly = seq.to(DEVICE), target.to(DEVICE), valid.to(DEVICE), std_anomaly.to(DEVICE)

                    pred = model(seq)
                    val_loss += loss_fn(pred, target, seq[:, 0:1, :, :], valid).item()

                    mae_c, hs_mae_c, iou = compute_metrics(pred, target, valid, std_anomaly)
                    val_mae_c += mae_c
                    val_hs_mae_c += hs_mae_c
                    val_iou += iou

            n_t, n_v = max(len(train_loader), 1), max(len(val_loader), 1)
            train_loss /= n_t
            val_loss /= n_v

            scheduler.step(val_loss)

            print(f"Epoch {epoch+1:02d} | T.Loss: {train_loss:.4f} | V.Loss: {val_loss:.4f} | MAE (°C): {val_mae_c/n_v:.4f} | HS MAE (°C): {val_hs_mae_c/n_v:.4f} | IoU: {val_iou/n_v:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), f"best_hybrid_fold_{fold}.pt")

        # GERAR GRAD-CAM DO MELHOR MODELO NO FOLD
        model.load_state_dict(torch.load(f"best_hybrid_fold_{fold}.pt", weights_only=True))
        cam_generator = RegressionGradCAM(model, target_layer=model.up1.conv.conv2)

        # Correção 1: Desempacotar 4 variáveis retornadas pelo dataloader
        sample_seq, sample_tgt, sample_valid, sample_std = next(iter(val_loader))

        # Correção 2: Slice de 4 canais ao invés de 5 para alinhar com o input da rede
        cam_map, pred_np = cam_generator.generate(sample_seq[0:1, :4, :, :].to(DEVICE))

        plot_grad_cam(sample_seq[0:1], cam_map, pred_np, sample_tgt[0, 0].numpy(), save_path=f"gradcam_fold_{fold}.png")
        print(f"-> Imagem do Grad-CAM salva como gradcam_fold_{fold}.png")

        break # Remova o break para rodar todos os folds
