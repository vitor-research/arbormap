"""
Download do checkpoint no Colab

Origem: notebook arbormap.ipynb, célula 31 (Google Colab).
Utilitário.
Código preservado como no notebook; caminhos do Colab/Drive devem ser ajustados nas CONFIGURAÇÕES.
"""

from google.colab import files

# Caminho do arquivo
arquivo = "best_hybrid_fold_0.pt"

# Faz o download
files.download(arquivo)
