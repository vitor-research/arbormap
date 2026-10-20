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

Os arquivos em `src/` preservam o código tal como executado no Google Colab, com um cabeçalho indicando a célula de origem. As constantes de caminho (`DATA_DIR`, `FILES_DIR`, `data_root`) são lidas da variável de ambiente `ARBORMAP_DATA_DIR`; quando ela não está definida, usa-se `./data`. No Google Colab, defina a variável antes de executar qualquer script, apontando para a pasta do dataset no Drive:

```python
import os
os.environ["ARBORMAP_DATA_DIR"] = "/content/drive/MyDrive/heat_dataset_v3"
```

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

### Refazer os resultados com a UrbanHeatNet (Colab)

O notebook `notebooks/refazer_urbanheatnet.ipynb` executa, em sequência e com retomada automática se a sessão cair:

1. `model/train_urbanheatnet.py`: treino da UrbanHeatNet (célula 32) nos cinco folds da Tabela 1, com estatísticas de normalização por fold, log completo e `tabela2.csv` gerada do próprio histórico;
2. `analysis/figures_urbanheatnet.py`: Figuras 2 e 3 a partir do histórico salvo;
3. `decision/simulate_urbanheatnet.py`: simulação em Teresina com o modelo do fold em que Teresina é validação (seleção multicritério, corredores por morfologia, baselines gulosa e aleatória, Figuras 4 a 6);
4. `decision/export_map.py`: arquivos georreferenciados do mapa interativo (`*_lst_pred`, `*_mask_inviavel`, `*_corredores.geojson`, `bounds.json`).

## Mapa interativo (página do projeto)

Página estática em `docs/`, publicada pelo GitHub Pages: **https://abraaolm-maker.github.io/arbormap/**

- **Gestor:** mapa de calor dos 123 bairros de Teresina, escolha de praças e calçadas por número de árvores ou clicando no mapa, plano completo numa nova aba (mapa do trecho, áreas com coordenadas, bairros atendidos, espécies nativas), ranking de bairros por calor × vulnerabilidade e aba de confiabilidade.
- **Cidadão** (`?modo=cidadao`): busca por bairro, faixa de resfriamento se o bairro for arborizado e passo a passo para cobrar a Prefeitura.
- Os dados da página são gerados por `decision/export_pagina.py`; não há servidor.

Para publicar num repositório próprio: copie o repositório, ative o GitHub Pages em *Settings → Pages → Deploy from a branch → main, pasta /docs*.

## Resultados principais (modelo final v1.0.0, 20 épocas fixas)

| Métrica (validação cruzada por cidade, k = 5, época fixa) | Valor |
|---|---|
| MAE médio em cidades inéditas | 2,04 ± 0,20 °C |
| Hotspot MAE (percentil 90) | 2,62 ± 0,44 °C |
| Boundary IoU | 0,230 ± 0,011 |
| MAE em Teresina (composição seca, área ampliada, 123 bairros) | 1,29 °C (correlação 0,89) |
| Comparação, mesma régua: regressão linear / média da cena | 2,36 / 2,68 °C |
| Experimento natural (onde a vegetação aumentou, 2018–21 → 2023–24) | −1,00 °C medido × −1,06 °C previsto |
| Parques existentes (38): frescor medido × previsto | 0,76 × 1,12 °C (correlação 0,94) |
| Superfície × ar (INMET A312, 80 dias; seca) | superfície 8,2 °C acima do ar; r = 0,62 |

Detalhes e histórico das correções: `ACHADOS` e o documento técnico enviados junto com o artigo (versão 4).

## Módulos novos (outubro/2026)

```
src/arbormap/
  model/train_urbanheatnet.py      treino final (CONFIG_FINAL: 20 épocas, máscara de LST ≤ 1 °C)
  analysis/verify_urbanheatnet.py  verificação independente e referências simples
  analysis/validacao_real.py       experimento natural, parques existentes e ablação
  analysis/validacao_inmet.py      superfície (Landsat) × ar (INMET A312)
  analysis/figuras_artigo_v4.py    figuras do artigo, versão 4
  data/landsat_pc.py               composições Landsat do Planetary Computer (área ampliada)
  decision/city_layers.py          composição seca, previsão por janelas e incerteza
  decision/bairros.py              métricas por bairro (IBGE, Censo 2022)
  decision/plantable.py            fração plantável por pixel (Overture + OpenStreetMap)
  decision/areas_candidatas.py     praças e calçadas candidatas, efeito simulado de cada uma
  decision/especies.py             espécies nativas (Manual de Fortaleza + Flora e Funga do Brasil)
  decision/export_pagina.py        dados da página (docs/data)
docs/                              página interativa (GitHub Pages)
```

Tags: `v1.0.0` (modelo final), `v1.1-completa` (página com orçamento, cenários e desenho livre, antes da simplificação).

## Histórico de versões

- **ConvLSTM (abril/2026)** – U-Net ConvLSTM com 8 canais, incluindo a LST da cena anterior. Gerou a Tabela 2 e as Figuras 5 e 6 das versões 1 a 3 do artigo; abandonada pelos motivos descritos no artigo.
- **ASPP (intermediária)** – UrbanHeatNet com módulo ASPP.
- **UrbanHeatNet final (v1.0.0)** – blocos residuais, portões de atenção, Group Normalization, 4 canais (NDVI, NDBI, DEM, NDWI), sem LST de entrada; retreinada em outubro/2026 com os pixels de LST zerada fora do alvo e época fixa.

## Próximos passos

- Medição de campo em pontos escolhidos no mapa
- Conferência das espécies com Machado et al. (2006) e com o viveiro municipal
- Experimentação de perdas adversariais para reduzir a subestimação de extremos térmicos
## Citação

Ver `CITATION.cff`.

## Licença

MIT – ver `LICENSE`.
