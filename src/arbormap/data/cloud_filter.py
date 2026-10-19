"""
Filtro de qualidade das composições

Origem: notebook arbormap.ipynb, célula 11 (Google Colab).
Descarta composições com >=30% de pixels abaixo de 15 °C (nuvens/pixels corrompidos) e gera ranking por cidade.
Código preservado como no notebook; caminhos do Colab/Drive devem ser ajustados nas CONFIGURAÇÕES.
"""

from collections import defaultdict
from pathlib import Path
import numpy as np

TOTAL_RAW_CHANNELS = 5

def ensure_channels_first(img):
    if img.ndim != 3: raise ValueError(f"Esperado array 3D, recebido {img.shape}")
    if img.shape[0] == TOTAL_RAW_CHANNELS: return img
    elif img.shape[-1] == TOTAL_RAW_CHANNELS: return np.transpose(img, (2, 0, 1))
    else: raise ValueError("Não foi possível identificar eixo de canais.")


def analyze_cloudy_cities(data_dir, temp_threshold=15.0, min_cloud_pct=30.0):
    """
    Analisa o dataset, identifica as fotos nubladas por cidade
    e exibe um ranking ordenado da cidade com mais fotos nubladas para a com menos.
    """
    root_path = Path(data_dir)
    min_cloud_fraction = min_cloud_pct / 100.0

    # Dicionário para armazenar as contagens por cidade
    cloudy_stats = defaultdict(lambda: {"total": 0, "cloudy": 0})

    print(f"\n🔍 Analisando nuvens por cidade (Limite: >{min_cloud_pct}%)...")

    # Percorre todas as pastas de cidades
    for city_folder in root_path.iterdir():
        if not city_folder.is_dir() or city_folder.name.startswith('.'):
            continue

        city_name = city_folder.name
        npy_files = list(city_folder.glob("*.npy"))

        cloudy_stats[city_name]["total"] = len(npy_files)

        for filepath in npy_files:
            try:
                data = np.load(filepath)
                # Utiliza a mesma função de extração de canais já definida no seu script
                lst_band = ensure_channels_first(data)[CH_LST]

                # Calcula a fração de pixels abaixo do limiar de temperatura (nuvens)
                cloud_fraction = np.sum(lst_band < temp_threshold) / lst_band.size

                if cloud_fraction >= min_cloud_fraction:
                    cloudy_stats[city_name]["cloudy"] += 1
            except Exception as e:
                # Caso ocorra algum erro na leitura do arquivo específico
                continue

    # Ordena o ranking decrescente pelo número de fotos nubladas
    ranking = sorted(
        cloudy_stats.items(),
        key=lambda item: item[1]["cloudy"] / item[1]["total"],
        reverse=True
    )

    # Exibe o resultado formatado em tabela
    print("\n" + "="*65)
    print(f"{'RANKING DE CIDADES COM MAIS FOTOS NUBLADAS':^65}")
    print("="*65)
    print(f"{'Pos.':<6} | {'Cidade':<25} | {'Nubladas':<10} | {'Total':<10} | {'% Nublado':<10}")
    print("-" * 65)

    for idx, (city, stats) in enumerate(ranking, start=1):
        total = stats["total"]
        cloudy = stats["cloudy"]
        pct = (cloudy / total * 100) if total > 0 else 0
        print(f"{idx:<6} | {city:<25} | {cloudy:<10} | {total:<10} | {pct:.1f}%")

    print("="*65)
    return ranking

# ============================================================
# COMO EXECUTAR NO SEU SCRIPT:
# ============================================================
if __name__ == "__main__":
    # Certifique-se de que a variável DATA_DIR está definida
    ranking_cidades = analyze_cloudy_cities("/content/drive/MyDrive/heat_dataset_v3", temp_threshold=15.0, min_cloud_pct=30.0)
