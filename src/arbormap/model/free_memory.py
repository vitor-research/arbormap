"""
Liberação de memória GPU no Colab

Origem: notebook arbormap.ipynb, célula 12 (Google Colab).
Utilitário.
Código preservado como no notebook; caminhos do Colab/Drive devem ser ajustados nas CONFIGURAÇÕES.
"""

import gc
import torch

# 1. Remova a referência ao objeto
del model
# 2. Forçe o Python a liberar a memória (o PyTorch detecta isso)
gc.collect()

# 3. (Opcional) Limpe o cache residual na GPU
torch.cuda.empty_cache()
