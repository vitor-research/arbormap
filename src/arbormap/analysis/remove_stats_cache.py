"""
Remove cache de estatísticas de normalização

Origem: notebook arbormap.ipynb, célula 25 (Google Colab).
Utilitário.
Código preservado como no notebook; caminhos do Colab/Drive devem ser ajustados nas CONFIGURAÇÕES.
"""

import os
stats_path = os.path.join(DATA_DIR, "dataset_stats_zscore_without_ndwi.npz")
if os.path.exists(stats_path):
    os.remove(stats_path)
