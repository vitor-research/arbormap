"""
Download do checkpoint no Colab

Origem: notebook arbormap.ipynb, célula 31 (Google Colab).
Utilitário.
Código preservado como no notebook; caminhos do Colab/Drive devem ser ajustados nas CONFIGURAÇÕES.
"""

try:
    from google.colab import files
except ImportError:
    files = None
    print("Aviso: google.colab indisponível; este utilitário só faz o download dentro do Colab.")

# Caminho do arquivo
arquivo = "best_hybrid_fold_0.pt"

# Faz o download
if files is not None:
    files.download(arquivo)
