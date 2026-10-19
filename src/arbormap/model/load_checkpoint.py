"""
Carregamento de checkpoint do modelo híbrido

Origem: notebook arbormap.ipynb, célula 33 (Google Colab).
Exemplo mínimo de inferência.
Código preservado como no notebook; caminhos do Colab/Drive devem ser ajustados nas CONFIGURAÇÕES.
"""

import torch

# Substitua 'seu_modelo' pela classe que define sua rede
model = HybridUrbanHeatNet(in_channels=4)
checkpoint = torch.load("best_hybrid_fold_1.pt", map_location="cpu")

# Se o checkpoint for um dicionário contendo o state_dict:
model.load_state_dict(checkpoint)

# Para ver a arquitetura completa com as camadas e formatos:
print(model)
