# ARBORMAP

**Previsão de ilhas de calor e indicação de áreas prioritárias para arborização urbana com inteligência artificial.**

Sistema de apoio à decisão que estima a Temperatura da Superfície Terrestre (LST) a partir de imagens Landsat 8/9 (30 m) e indica, sob restrições físicas do terreno, onde a arborização produz o maior resfriamento por hectare. Desenvolvido como projeto de iniciação científica no IFPI – Campus Teresina Central e submetido à FEBRACE 2027 (categoria Engenharia).

- Autor: Vítor Emanuel da Silva Rodrigues
- Orientador: Abraão Lima Sousa · Coorientador: Adriano Cleyton da Silva Pereira
- Instituição: Instituto Federal de Educação, Ciência e Tecnologia do Piauí (IFPI), Campus Teresina Central

## Como funciona

1. **Dados** – composições mensais por mediana de Landsat 8/9 (2018–2024) para 20 municípios do Nordeste e da transição Cerrado-Amazônia, obtidas pela API do Google Earth Engine, com os canais NDVI, NDBI, NDWI e modelo digital de elevação (DEM). Composições com 30% ou mais de pixels abaixo de 15 °C (nuvens/pixels corrompidos) são descartadas.
2. **Modelo** – *UrbanHeatNet*: U-Net com blocos residuais, conexões de atalho e portões de atenção espacial (Attention U-Net), Group Normalization, treinada em PyTorch para prever a anomalia térmica padronizada, com função de perda que combina erro preditivo, penalização do gradiente espacial e SSIM.
3. **Validação** – validação cruzada espacial por cidade (K-Fold, k = 5): cada município é avaliado apenas quando ausente do treinamento. Métricas: MAE, Hotspot MAE (percentil 90) e Boundary IoU.
4. **Motor de decisão** – seleção multicritério que exclui corpos d'água (NDWI) e encostas com declividade superior a 15° (DEM), seleção iterativa sob orçamento fixo, morfologia matemática para formar corredores contínuos e simulação do espalhamento do efeito de resfriamento (OpenCV).
5. **Avaliação** – em Teresina, sob orçamento de 4,05 ha, comparação com alocação gulosa e aleatória e análise de sensibilidade por orçamento.

## Estrutura do repositório

```
src/arbormap/
  data/        download no Earth Engine e filtro de qualidade das composições
  model/       UrbanHeatNet híbrida, dataset, perdas, treino K-Fold, inferência
  decision/    motor de decisão, baselines e simulador genético (experimental)
  analysis/    ablação de canais, resíduos espaciais, correlação física, figuras
notebooks/     arbormap.ipynb (notebook de trabalho, limpo)
logs/          logs de treinamento e de simulação registrados durante o projeto
figures/       figuras do artigo, em resolução original
legacy/        v1 (ConvLSTM, abril/2026) e versão intermediária com ASPP
```

Os arquivos em `src/` preservam o código tal como executado no Google Colab, com um cabeçalho indicando a célula de origem. As constantes de caminho (`DATA_DIR`, `FILES_DIR`) apontam para o Google Drive do autor e devem ser ajustadas para o ambiente local.

## Reprodução

```bash
pip install -r requirements.txt
# 1. autenticar no Earth Engine e baixar o dataset
python src/arbormap/data/download_landsat_ee.py
python src/arbormap/data/cloud_filter.py
# 2. treinar com validação cruzada por cidade (GPU recomendada; ~45 min/fold em T4)
python src/arbormap/model/hybrid_pipeline.py
# 3. simular intervenções e comparar com baselines
python src/arbormap/decision/intervention_simulation.py
python src/arbormap/decision/baselines.py
```

Todo o treinamento do modelo final foi executado em GPU T4 gratuita do Google Colab.

## Resultados principais

| Métrica (validação cruzada por cidade, k = 5) | Valor |
|---|---|
| MAE médio em cidades inéditas | 3,64 ± 0,48 °C |
| Hotspot MAE (percentil 90) | 2,75 ± 0,58 °C |
| Redução média de LST em Teresina (4,05 ha) – ARBORMAP | 2,86 °C |
| Idem – alocação gulosa | 1,51 °C |
| Idem – alocação aleatória | 0,15 °C |

## Histórico de versões

- **v1 (abril/2026)** – U-Net ConvLSTM com sequências de 6 composições. Abandonada: revisita de 16 dias + descarte de nuvens gera séries irregulares; o calor urbano é governado por fatores quase estáticos; a abordagem agravava a regressão à média e sextuplicava memória e tempo de treino.
- **v2 (intermediária)** – UrbanHeatNet com módulo ASPP e perda informada pela física.
- **v3 (final)** – UrbanHeatNet híbrida com blocos residuais e portões de atenção, Group Normalization e perda híbrida. Resultados do artigo.

## Próximos passos (até fevereiro de 2027)

- Validação das áreas recomendadas em Teresina com dados de temperatura do ar do INMET
- Mapa interativo para gestores municipais e versão para cidadãos
- Experimentação de perdas adversariais para reduzir a subestimação de extremos térmicos

## Citação

Ver `CITATION.cff`.

## Licença

MIT – ver `LICENSE`.
