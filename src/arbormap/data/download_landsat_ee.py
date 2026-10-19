"""
Download das composições Landsat 8/9 via Google Earth Engine

Origem: notebook arbormap.ipynb, célula 1 (Google Colab).
Constrói composições mensais por mediana (LST, NDVI, NDBI, NDWI, DEM), mascara nuvens e salva .npy por cidade.
Código preservado como no notebook; caminhos do Colab/Drive devem ser ajustados nas CONFIGURAÇÕES.
"""

import ee
import geemap
import numpy as np
import os
import json
import time
from datetime import datetime

# ============================================================
# CONFIG
# ============================================================
IS_FREE_COLAB = True
FILES_DIR = os.environ.get("ARBORMAP_DATA_DIR", "./data")

if IS_FREE_COLAB:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
    except ImportError:
        print("Aviso: google.colab indisponível; executando fora do Colab com FILES_DIR =", FILES_DIR)

ee.Authenticate()
ee.Initialize(project="script-39a43")

# ============================================================
# CIDADES
#
# Critérios de seleção:
#   - Variabilidade de porte (5k–900k hab)
#   - Variabilidade de altitude (20 m–1 000 m)
#   - Diferentes biomas do Nordeste (Caatinga, Cerrado, Litoral)
#   - Cidades do Piauí originais mantidas
#   - Adicionadas cidades de outros estados nordestinos para
#     melhorar generalização sem sair do clima semiárido/tropical
# ============================================================
CIDADES = {
    # ── Piauí (originais) ──────────────────────────────────────
    # "Teresina":               (-42.790, -5.080),   # ~900k hab, vale do Parnaíba
    # "Parnaiba":               (-41.770, -2.930),   # litoral, brisa marítima
    # "Picos":                  (-41.460, -7.080),   # sertão, altitude ~200 m
    # "Floriano":               (-43.010, -6.780),   # sertão sul
    # "Bom_Jesus":              (-44.375, -9.075),   # cerrado piauiense
    # "Sao_Raimundo_Nonato":    (-42.705, -8.975),   # caatinga, ~30k hab
    # "Corrente":               (-45.125,-10.435),   # extremo sul, cerrado
    # "Urucui":                 (-44.450, -7.200),   # agronegócio, chapada
    # "Pedro_II":               (-41.425, -4.415),   # altitude ~970 m — outlier térmico valioso

    # # ── Nordeste — diversidade climática e urbana ───────────────
    # "Vitoria_da_Conquista_BA":(-40.844, -14.866),  # ~340k hab, altitude 923 m — mais frio
    # "Juazeiro_do_Norte_CE":   (-39.315, -7.213),   # ~270k hab, caatinga seca
    # "Mossoro_RN":             (-37.344, -5.188),   # ~300k hab, salineiras, muito árido
    # "Imperatriz_MA":          (-47.492, -5.527),   # ~260k hab, transição cerrado-amazônia
    # "Barreiras_BA":           (-44.993, -12.144),  # ~140k hab, MATOPIBA, cerrado intenso
    # "Paulo_Afonso_BA":        (-38.215, -9.400),   # ~110k hab, caatinga + Rio São Francisco
    # "Sobral_CE":              (-40.348, -3.686),   # ~210k hab, sertão cearense

    # Cidades adicionadas
    "Araguaina_TO":             (-48.206, -7.191),   # ~180k hab, transição cerrado-amazônia (Tocantins)
    "Patos_PB":                 (-37.281, -7.025),   # ~100k hab, depressão sertaneja (Paraíba)
    "Crato_CE":                 (-39.407, -7.233),   # ~130k hab, região metropolitana do Cariri (Ceará)
    "Iguatu_CE":                (-39.299, -6.356),   # ~100k hab, centro-sul cearense (Ceará)

    # Cidades removidas por excesso de nuvens
    # "Feira_de_Santana_BA":    (-38.966, -12.252),  # ~620k hab, entroncamento rodoviário
    # "Campina_Grande_PB":      (-35.880, -7.230),   # ~420k hab, altitude 550 m, polo industrial
    # "Altos":                  (-42.400, -5.000),   # cinturão verde de Teresina
    # "Jose_de_Freitas":        (-42.535, -4.735),   # periurbano de Teresina
    # "Caruaru_PE":             (-36.010, -8.280),   # ~360k hab, altitude 554 m, agreste






}

# ── validação rápida ──────────────────────────────────────────
assert all(len(v) == 2 for v in CIDADES.values()), "Coordenadas inválidas"

# ============================================================
# PARÂMETROS DE DOWNLOAD
# ============================================================
BUFFER_M    = 6_000    # raio do ROI em metros (→ ~12×12 km)
SCALE_M     = 30       # resolução Landsat em metros
MAX_NAN_PCT = 0.30     # descarta frame se >30 % dos pixels forem NaN
MAX_RETRIES = 4        # tentativas por mês antes de desistir
RETRY_WAIT  = 8        # segundos entre tentativas

# ============================================================
# CLOUD MASK
# ============================================================
def mask_landsat_clouds(image):
    qa     = image.select("QA_PIXEL")
    dilated = 1 << 1
    cloud   = 1 << 3
    shadow  = 1 << 4
    # inclui bit de "dilatated cloud" para margens de nuvem
    mask = (qa.bitwiseAnd(dilated).eq(0)
              .And(qa.bitwiseAnd(cloud).eq(0))
              .And(qa.bitwiseAnd(shadow).eq(0)))
    return image.updateMask(mask)

# ============================================================
# ÍNDICES E LST
# ============================================================
dem = ee.Image("USGS/SRTMGL1_003").select("elevation").rename("DEM")

def scale_sr(img, band):
    return img.select(band).multiply(0.0000275).add(-0.2)

def build_month_image(collection, roi):
    med = collection.median()

    b3  = scale_sr(med, "SR_B3")
    b4  = scale_sr(med, "SR_B4")
    b5  = scale_sr(med, "SR_B5")
    b6  = scale_sr(med, "SR_B6")

    ndvi = b5.subtract(b4).divide(b5.add(b4).abs().add(1e-6)).rename("NDVI")
    ndwi = b3.subtract(b5).divide(b3.add(b5).abs().add(1e-6)).rename("NDWI")
    ndbi = b6.subtract(b5).divide(b6.add(b5).abs().add(1e-6)).rename("NDBI")

    # LST em °C — média em vez de mediana para temperatura superficial
    lst = (collection.select("ST_B10")
                     .mean()
                     .multiply(0.00341802)
                     .add(149.0)
                     .subtract(273.15)
                     .rename("LST"))

    # DEM clipado ao mesmo ROI
    dem_clip = dem.clip(roi)

    return ee.Image.cat([ndvi, ndwi, ndbi, dem_clip, lst]).clip(roi)

# ============================================================
# NaN HANDLING
#
# Estratégia: NaN genuíno (nuvem/sombra residual) é preenchido
# por interpolação espacial local via focal_mean do GEE antes
# do download. Se após isso ainda restar >MAX_NAN_PCT, o frame
# é descartado e registrado como gap.
# ============================================================
def fill_nan_spatial(array: np.ndarray) -> np.ndarray:
    """
    Preenche NaN por interpolação dos vizinhos mais próximos
    canal a canal. Mais honesto que substituir por 0.
    Usa scipy se disponível, senão fallback para a média do canal.
    """
    try:
        from scipy.ndimage import generic_filter

        def _fill(channel):
            nan_mask = np.isnan(channel)
            if not nan_mask.any():
                return channel
            filled = channel.copy()
            # média 5×5 ignorando NaN
            def _mean_no_nan(vals):
                v = vals[~np.isnan(vals)]
                return v.mean() if len(v) else np.nan
            filled = generic_filter(filled, _mean_no_nan, size=5, mode="nearest")
            # pixels que ainda são NaN (borda/ilha grande) → média global do canal
            still_nan = np.isnan(filled)
            if still_nan.any():
                ch_mean = np.nanmean(channel)
                filled[still_nan] = ch_mean if not np.isnan(ch_mean) else 0.0
            return filled

        out = np.stack([_fill(array[c]) for c in range(array.shape[0])], axis=0)
        return out

    except ImportError:
        # fallback simples: substituir pela média do canal (melhor que 0)
        out = array.copy()
        for c in range(out.shape[0]):
            ch      = out[c]
            ch_mean = np.nanmean(ch)
            ch[np.isnan(ch)] = ch_mean if not np.isnan(ch_mean) else 0.0
            out[c] = ch
        return out

# ============================================================
# DOWNLOAD COM RETRY
# ============================================================
def _download_with_retry(image, roi, scale, retries=MAX_RETRIES):
    for attempt in range(retries):
        try:
            arr = geemap.ee_to_numpy(image, region=roi, scale=scale)
            return arr
        except Exception as e:
            if attempt < retries - 1:
                wait = RETRY_WAIT * (2 ** attempt)   # backoff exponencial
                print(f"    Tentativa {attempt+1} falhou ({e}). Aguardando {wait}s…")
                time.sleep(wait)
            else:
                raise

# ============================================================
# DOWNLOAD PRINCIPAL
# ============================================================
def download_dataset(start_year: int, end_year: int, folder: str):
    os.makedirs(folder, exist_ok=True)

    landsat8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
    landsat9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
    base_col = (landsat8.merge(landsat9)
                        .filter(ee.Filter.lt("CLOUD_COVER", 20))
                        .map(mask_landsat_clouds))

    log_ok   = []   # frames salvos
    log_skip = []   # frames descartados (gap)
    log_err  = []   # erros irrecuperáveis

    for city, (lon, lat) in CIDADES.items():
        city_folder = os.path.join(folder, city)
        os.makedirs(city_folder, exist_ok=True)

        roi = ee.Geometry.Point([lon, lat]).buffer(BUFFER_M).bounds()

        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                start = f"{year}-{month:02d}-01"
                end   = (f"{year+1}-01-01" if month == 12
                         else f"{year}-{month+1:02d}-01")

                col = base_col.filterBounds(roi).filterDate(start, end)

                try:
                    n_images = col.size().getInfo()
                except Exception as e:
                    print(f"  [{city} {year}-{month:02d}] Erro ao contar imagens: {e}")
                    log_err.append({"city": city, "year": year, "month": month, "error": str(e)})
                    continue

                if n_images == 0:
                    log_skip.append({"city": city, "year": year, "month": month,
                                     "reason": "no_images"})
                    continue

                fname = os.path.join(city_folder, f"{year}_{month:02d}.npy")
                if os.path.exists(fname):
                    print(f"  [{city} {year}-{month:02d}] já existe, pulando.")
                    continue

                try:
                    image = build_month_image(col, roi)
                    array = _download_with_retry(image, roi, SCALE_M)

                    if array is None or array.size == 0:
                        log_skip.append({"city": city, "year": year, "month": month,
                                         "reason": "empty_array"})
                        continue

                    # [H, W, C] → [C, H, W]
                    array = np.transpose(array, (2, 0, 1)).astype(np.float32)

                    # ── NaN check ──────────────────────────────────────
                    nan_pct = np.isnan(array).mean()
                    if nan_pct > MAX_NAN_PCT:
                        print(f"  [{city} {year}-{month:02d}] {nan_pct:.0%} NaN — descartado")
                        log_skip.append({"city": city, "year": year, "month": month,
                                         "reason": f"nan_pct={nan_pct:.2f}"})
                        continue

                    # preenchimento espacial dos NaN restantes
                    if np.isnan(array).any():
                        array = fill_nan_spatial(array)

                    np.save(fname, array)
                    log_ok.append({"city": city, "year": year, "month": month,
                                   "n_images": n_images, "shape": list(array.shape)})
                    print(f"  [{city} {year}-{month:02d}] OK  shape={array.shape}  "
                          f"n_imgs={n_images}  nan_orig={nan_pct:.1%}")

                except Exception as e:
                    print(f"  [{city} {year}-{month:02d}] ERRO irrecuperável: {e}")
                    log_err.append({"city": city, "year": year, "month": month, "error": str(e)})

    # ── salva logs ────────────────────────────────────────────────────────
    for name, data in [("log_ok", log_ok), ("log_skip", log_skip), ("log_err", log_err)]:
        with open(os.path.join(folder, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nConcluído: {len(log_ok)} salvos | "
          f"{len(log_skip)} pulados | {len(log_err)} erros")
    print(f"Logs em: {folder}/log_*.json")


# ============================================================
# RESUMO DO DATASET (utilitário pós-download)
# ============================================================
def print_dataset_summary(folder: str):
    """Imprime quantos frames cada cidade tem e aponta gaps."""
    print(f"\n{'Cidade':<30} {'Frames':>6}  {'Gaps (esperados)':>18}")
    print("-" * 60)
    expected = (2024 - 2018 + 1) * 12   # meses entre 2018 e 2024

    for city in sorted(CIDADES):
        city_folder = os.path.join(folder, city)
        if not os.path.isdir(city_folder):
            print(f"  {city:<28} {'—':>6}")
            continue
        frames = len(glob.glob(os.path.join(city_folder, "*.npy")))
        gaps   = expected - frames
        print(f"  {city:<28} {frames:>6}  {gaps:>18}")


if __name__ == "__main__":
    import glob

    FOLDER = f"{FILES_DIR}/heat_dataset_v3"
    download_dataset(start_year=2018, end_year=2024, folder=FOLDER)
    print_dataset_summary(FOLDER)
