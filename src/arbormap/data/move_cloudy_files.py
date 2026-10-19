"""
Utilitário: mover composições nubladas para pasta separada

Origem: notebook arbormap.ipynb, célula 5 (Google Colab).
Classificação e movimentação em paralelo.
Código preservado como no notebook; caminhos do Colab/Drive devem ser ajustados nas CONFIGURAÇÕES.
"""

import os
import shutil
import random
import numpy as np
from pathlib import Path
import functools
from concurrent.futures import ProcessPoolExecutor

# ============================================================
# CONFIGURAÇÃO GERAL
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_DIR = Path(os.environ.get("ARBORMAP_DATA_DIR", "./data"))
# Pasta de destino para onde as imagens poluídas serão movidas
QUARANTINE_DIR = DATA_DIR.parent / (DATA_DIR.name + "_poluidas")

CH_LST = 4

def ensure_channels_first(img):
    if img.ndim != 3: raise ValueError(f"Esperado array 3D, recebido {img.shape}")
    if img.shape[0] == 5: return img
    elif img.shape[-1] == 5: return np.transpose(img, (2, 0, 1))
    else: raise ValueError("Não foi possível identificar eixo de canais.")

# ============================================================
# FILTRO E MOVIMENTAÇÃO DE NUVENS
# ============================================================
def _check_and_classify_file(filepath, temp_threshold, min_cloud_fraction):
    """
    Verifica a fração de nuvens e retorna o caminho do ficheiro juntamente com
    um booleano indicando se é válido (True) ou se deve ser movido (False).
    """
    try:
        data = np.load(filepath)
        lst_band = ensure_channels_first(data)[CH_LST]
        cloud_fraction = np.sum(lst_band < temp_threshold) / lst_band.size
        is_valid = cloud_fraction < min_cloud_fraction
        return Path(filepath), is_valid
    except Exception:
        # Se houver erro ao ler o ficheiro, considera inválido/corrompido por precaução
        return Path(filepath), False

def move_cloudy_files_recursive(temp_threshold=15.0, min_cloud_pct=30.0):
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)

    # Procura recursivamente por todos os ficheiros .npy em todas as subpastas (cidades)
    all_files = list(DATA_DIR.rglob("*.npy"))

    if not all_files:
        print(f"Nenhum ficheiro .npy encontrado em {DATA_DIR} ou nas suas subpastas.")
        return

    valid_count = 0
    moved_count = 0

    print(f"\n☁️ A analisar {len(all_files)} imagens distribuídas pelas pastas de cidades...")

    check_func = functools.partial(
        _check_and_classify_file,
        temp_threshold=temp_threshold,
        min_cloud_fraction=min_cloud_pct/100.0
    )

    # Processamento paralelo para agilizar a verificação
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(check_func, all_files))

    for filepath, is_valid in results:
        if is_valid:
            valid_count += 1
        else:
            try:
                # Opcional: Se quiser manter o nome da cidade no nome do ficheiro ao mover
                # (ex: teresina_imagem123.npy) para evitar conflitos caso cidades diferentes
                # tenham ficheiros com exatamente o mesmo nome:
                relative_path = filepath.relative_to(DATA_DIR)
                dest_filename = f"{str(relative_path).replace(os.sep, '_')}"
                dest_path = QUARANTINE_DIR / dest_filename

                shutil.move(str(filepath), str(dest_path))
                moved_count += 1
            except Exception as e:
                print(f"Erro ao mover o ficheiro {filepath.name}: {e}")

    print(f" -> Imagens válidas mantidas nas pastas de origem: {valid_count}")
    print(f" -> Imagens poluídas movidas para '{QUARANTINE_DIR}': {moved_count}\n")

# ============================================================
# EXECUÇÃO
# ============================================================
if __name__ == "__main__":
    move_cloudy_files_recursive(
        temp_threshold=15.0,
        min_cloud_pct=30.0
    )
