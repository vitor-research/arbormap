# ARBORMAP: o que foi feito e como atualizar o artigo (versão 4)

Projeto FEBRACE 2027 · Vítor Emanuel da Silva Rodrigues (autor) · Abraão Lima Sousa (orientador) · Adriano Cleyton da Silva Pereira (coorientador) · IFPI Campus Teresina Central

Documento gerado em 21/10/2026. Base: tudo o que foi feito nas sessões de trabalho com o Claude Code, de 19 a 21/10/2026, conferido nos arquivos do projeto.

- Repositório: **https://github.com/abraaolm-maker/arbormap** (ramo `main`)
- Página: **https://abraaolm-maker.github.io/arbormap/**

---

## Parte 1. Resumo do que foi feito

1. **Auditoria do artigo v3 contra o código.**
   - A Tabela 2 (MAE 3,64 °C) e as Figuras 5 e 6 (2,86 °C) da v3 não vieram da UrbanHeatNet, e sim de um modelo ConvLSTM. Esse modelo recebia a temperatura (LST) da cena anterior como entrada.
   - A simulação de plantio desse modelo escrevia diretamente no canal de temperatura. Parte do resfriamento era, portanto, imposta, e não aprendida.
   - A Figura 3 (desvio-padrão) vinha de outra rodada, com outro modelo.
   - A Figura 1 tinha Uruçuí no lugar errado: o mapa usava o código IBGE de União.
2. **Organização do código.**
   - Repositório público limpo, com o histórico pessoal fora.
   - Caminhos configuráveis pela variável `ARBORMAP_DATA_DIR`.
   - Teste automático e tag `v1.0.0`.
3. **Retreino da UrbanHeatNet** (o modelo descrito no artigo, célula 32 do notebook, conferido contra o checkpoint enviado pelo Vitor).
   - Feito em três rodadas no Colab Pro, GPU A100.
   - Correções aplicadas:
     - os pixels com LST = 0 °C, que são falhas fixas do download, saíram do alvo;
     - foram usados os hiperparâmetros que o próprio artigo descreve;
     - a época passou a ser fixa (20), sem escolher a melhor pela validação.
   - Resultado: MAE de 3,64 → **2,04 ± 0,20 °C**, com um modelo que **não usa a temperatura como entrada**.
4. **Verificação independente.**
   - O modelo supera a regressão linear (2,36 °C) e a previsão pela média da cena (2,68 °C) em todos os folds.
   - Ficou documentado que o MAE em °C é o erro do **padrão espacial** (anomalia), não da temperatura absoluta do mês.
5. **Validações com o mundo real, sem treino novo.**
   - **Experimento natural:** onde a vegetação aumentou entre 2018–21 e 2023–24, o satélite mediu −1,00 °C e o modelo previu −1,06 °C.
   - **Parques existentes:** correlação de 0,94 entre o frescor medido e o previsto, mas o modelo superestima: 1,12 contra 0,76 °C.
   - **Superfície × ar (estação INMET A312):** na estação seca, a superfície fica 8,2 °C acima do ar.
   - **Ablação:** a vegetação (NDVI) é a informação mais usada pelo modelo.
6. **Mapa de Teresina inteira.**
   - Composição da estação seca com 13 meses de 2018 a 2024.
   - A área foi ampliada de 89 para os **123 bairros**, com o mesmo produto Landsat lido de uma fonte aberta. Na área comum, ele concorda com o dado original (r = 0,989).
   - A previsão é feita por janelas de 400 px, o tamanho do treino.
7. **Do pixel ao plano.**
   - **Fração plantável por pixel:** 435 mil prédios (Overture) mais ruas, praças e água (OpenStreetMap).
   - **Áreas candidatas:** 737 áreas (praças, calçadas e áreas livres), cada uma com o efeito simulado pelo modelo.
   - **Bairros:** métricas por bairro (calor, população vulnerável) com o Censo 2022.
   - **Espécies:** espécies nativas da Caatinga e do Cerrado com fonte oficial.
8. **Página interativa** (GitHub Pages, sem servidor).
   - **Modo Gestor:** escolher áreas, ranking de bairros, plano completo e aba de confiabilidade.
   - **Modo Cidadão:** faixa de resfriamento do bairro e passo a passo para cobrar a Prefeitura.
   - Uma versão mais completa (orçamento, cenários, desenho livre com o modelo rodando no navegador) foi construída e depois **retirada da página para focar no essencial**. Ela fica guardada na tag `v1.1-completa`.

---

## Parte 2. Atualização do artigo, seção por seção

Para cada seção: **o que sai**, **o que entra** e **onde entra**. Números em negrito são os novos. O esboço da V4 (`ARBORMAP_artigo_FEBRACE_v4_esboco.docx`) já aplica tudo isto.

### Título e resumo

- **Sai:**
  - "redução média de 2,86 °C";
  - "Hotspot MAE de ± 2,82 °C";
  - a margem de 0,04 °C;
  - "perda focada no erro preditivo e na penalização do gradiente espacial";
  - "recomendação autônoma".
- **Entra:**
  - MAE de **2,04 ± 0,20 °C** em cidades inéditas, contra 2,36 °C da regressão linear;
  - o experimento natural (**−1,00 °C medido × −1,06 °C previsto**);
  - Teresina inteira (**123 bairros, MAE 1,29 °C**);
  - a página com os modos Gestor e Cidadão;
  - o resultado do plano: **164 ha** com pelo menos 0,5 °C a menos com 7.595 árvores, contra 79 ha na escolha aleatória;
  - a leitura honesta: resfriamento **local**, na **superfície**, tratado como **limite superior**.
- **Título:** pode ficar. Sugestão opcional: "ARBORMAP: previsão de ilhas de calor e mapa interativo de áreas prioritárias para arborização com inteligência artificial".

### I. Introdução

- **Fica:** o contexto de ilhas de calor, Teresina, sensoriamento remoto e U-Net.
- **Muda (parágrafo da proposta):** o objetivo final é um **mapa interativo** para gestores e cidadãos, e não uma "estratégia de escolha guiada por simulação oracular".
- **Sai:** a subseção C ("Seleção iterativa baseada em simulação").
- **Entra no lugar dela:** "C. Do mapa de calor ao plano de arborização". Ela explica:
  - a anomalia espacial;
  - a fração plantável;
  - a simulação de cada área;
  - o uso por gestores e cidadãos.
- **Opcional:** uma frase sobre a diferença entre temperatura da superfície e do ar. Hoje é apenas uma pista, e a validação com o INMET a confirma.

### II. Objetivo e questão problema

- **Fica:** a questão problema (passar do diagnóstico à prescrição).
- **Objetivos específicos, novos:**
  - (i) base de 20 municípios: fica;
  - (ii) UrbanHeatNet só com índices físicos, **sem LST de entrada**;
  - (iii) validação cruzada por cidade **com época fixa** e comparação com referências simples;
  - (iv) **validação com o mundo real** (experimento natural, parques, INMET);
  - (v) **mapa de Teresina inteira** com áreas candidatas (praças e calçadas) e prioridade por bairro;
  - (vi) **página interativa** para gestores e cidadãos.
- **Sai:** o objetivo (v) antigo (multicritério × gulosa × aleatória sob orçamento fixo). Agora é uma comparação de estratégias pela área beneficiada (Figura 9).
- **Critérios de projeto, ajustar:**
  - meta de MAE: 1,5 °C, com resultado de 2,04 °C (ver Resultados);
  - "superar as estratégias gulosa e aleatória" vira "superar a escolha aleatória e a dos pontos mais quentes em área beneficiada";
  - o critério dos corredores contínuos sai: a página trabalha com praças e trechos de calçada;
  - entra o critério "comunicar a incerteza".

### III. Materiais e métodos

| Subseção | O que muda |
|---|---|
| A. Área e dados | Fica: Tabela 1 (os IDs conferem com os folds do treino) e Figura 1 **refeita** (Uruçuí, código 2211209). Entra: os pares são consecutivos (as entradas do mês *t* preveem a anomalia do mês *t+1*); são 847 composições e 827 pares. |
| B. UrbanHeatNet | Corrigir a perda: **0,8 × SSIM + 0,2 × L1 + 0,1 × L1 nos hotspots (percentil 90)**, sem termo de gradiente. Entra: os pixels com LST ≤ 1 °C ficam fora do alvo e das métricas. A rede tem 13,2 milhões de parâmetros. |
| C. Filtros e corredores | **É substituída** por "C. Composição da estação seca, fração plantável e simulação": mediana de 13 meses secos; previsão por janelas de 400 px; fração plantável por pixel (Overture + OSM); áreas candidatas; plantio = aproximar NDVI, NDBI e NDWI da vegetação densa de Teresina na proporção da parte plantável, sem editar a temperatura; espalhamento (média ponderada de vizinhanças de 150 e 450 m). |
| D. Protocolo | Corrigir: **20 épocas fixas**, modelo da última época, taxa de aprendizado ajustada pela perda de treino (ReduceLROnPlateau, fator 0,5); recortes de 400 px com espelhamento e rotação; corte de gradiente 1,0. **Sem precisão mista.** Entra: as referências simples (média da cena, regressão linear, persistência). |
| E. Validação com o mundo real | **Nova:** experimento natural (5 meses secos em cada período); 38 parques do OSM (entorno de 90 a 240 m menos o interior); INMET A312 (80 passagens; LST < 15 °C descartada, o mesmo critério do dataset); ablação. |
| F. Implementação | Corrigir: GPU **A100** no Colab Pro (cerca de 29 s por época, cerca de 10 min por fold). Entra: a página estática (Leaflet), os dados abertos (IBGE Censo 2022, Overture, OSM, INMET, Planetary Computer) e o endereço do repositório. O histórico das versões (ConvLSTM abandonado) **fica**, acrescido de: "os números das versões anteriores deste artigo vinham do ConvLSTM e foram substituídos". |

### IV. Resultados e discussão

| Subseção | O que muda |
|---|---|
| A. Convergência | Nova Figura 2 (perda do Fold 2 e MAE de validação dos 5 folds). Sai: "melhor época" e "divergência a partir da época 22". Entra: a perda de validação fica quase estável, há sobreajuste leve e por isso a época é fixa. |
| B. Variância | Corrigir: o desvio-padrão das previsões fica entre **0,86 e 1,00** (0,93 a 0,97 na época 20), e não entre 0,60 e 0,80. A regressão à média é pequena. Nova Figura 3 (5 folds). |
| C. Desempenho entre folds | **Nova Tabela 2** (época 20) e **Figura 4** (referências). MAE de 2,04 ± 0,20 °C; Hotspot MAE de 2,62 ± 0,44 °C; IoU de 0,230 ± 0,011. Supera a linear e a média da cena em todos os folds. A meta de 1,5 °C não foi atingida. |
| D. Mapa de Teresina | Nova **Figura 5** (observado × previsto × erro, 123 bairros). MAE de 1,29 °C e correlação de 0,89, com o modelo que nunca viu Teresina. |
| E. Validação com o mundo real | **Nova:** Figura 6 (experimento natural e parques) e Figura 7 (INMET). |
| F. Prioridade por bairro | **Nova:** Figura 8. Os 10 bairros prioritários: Itararé, Renascença, Parque Alvorada, Promorar, Parque Ideal, N. S. das Graças, Parque do Piauí, Vila Operária, Macaúba e Piçarra. |
| G. Do mapa ao plano | **Substitui** as antigas E e F (2,86 °C, Figuras 5 e 6). Entram a Figura 9 e o plano padrão: 7.595 árvores, 103 áreas, 19,0 ha, 0,77 °C de resfriamento local médio, 161 ha com pelo menos 0,5 °C a menos, 41 bairros e 80 áreas perto de escolas, creches e unidades de saúde. |
| H. Página interativa | **Nova:** Figuras 10 e 11 (prints dos modos Gestor e Cidadão). A faixa do Cidadão: Itararé de 0,8 a 1,2 °C, Mocambinho de 1,5 a 2,2 °C. |
| I. Incerteza e limitações | **Substitui** as antigas G e H. Sai: a régua "2,86 > 2,82". Entram cinco pontos: (1) é temperatura da superfície; no ar, ~0,2 °C por grau na seca. (2) O resfriamento é local, porque o alvo é a anomalia: a soma líquida na cidade é ≈ 0 por construção. (3) O efeito vale para copas adultas e é um limite superior (parques: 68% do previsto). (4) Calçadas estreitas têm efeito pequeno. (5) A fração plantável vem de mapas abertos e pede conferência em campo. |
| J. Implicações práticas | Fica a ideia do texto, sem "simulação oracular". |

### V. Conclusões

- **Sai:**
  - "MAE de 3,64 °C";
  - "alta eficácia";
  - "prova de conceito fundamental";
  - os próximos passos já cumpridos (código aberto, INMET, mapa interativo).
- **Entra:**
  - o modelo final (2,04 °C), melhor que as referências;
  - o mecanismo confirmado pelo experimento natural;
  - Teresina inteira;
  - a página com os dois modos.
- **Próximos passos novos:**
  - medição de campo;
  - perdas que recuperem os extremos;
  - conferir as espécies com Machado et al. (2006) e o viveiro municipal;
  - outras cidades na página.

### Referências (acrescentar)

- Prefeitura de Fortaleza (2020). *Manual de Arborização Urbana de Fortaleza*. SEUMA.
- Flora e Funga do Brasil. Jardim Botânico do Rio de Janeiro.
- INMET. Banco de dados meteorológicos, estação A312.
- IBGE. Censo Demográfico 2022.
- Overture Maps Foundation (2026), buildings.
- OpenStreetMap.
- Microsoft Planetary Computer, Landsat Collection 2 Level-2.
- **Corrigir:** "COOK et al., 2014" no texto contra "2016" na lista.
- **Corrigir:** o link do código passa a ser o repositório público.

---

## Parte 3. Relatório técnico (para o Vitor)

### 3.1 Auditoria do artigo v3 (19/10/2026)

| Elemento da v3 | Origem real | Situação |
|---|---|---|
| Tabela 2 (MAE 3,64) | log da célula 4 (`UNetConvLSTM`, 8 canais com LST anterior, BatchNorm, L1 + hotspot, LR 3e-4, recorte de 256) | substituída pelo retreino |
| Figuras 5 e 6 (2,86 °C) | célula 7, ConvLSTM; escreve `-0.20` no canal de LST dos pixels plantados | substituídas |
| Figura 3 (DP de 0,6 a 0,8) | `logs/06`, família ASPP, outro fold | substituída |
| Figura 1 | `fold_map.py` com o código IBGE 2211100 (União) no lugar de Uruçuí (2211209) | refeita |
| Célula 29 | quebra com entrada 5D (`Expected 3D or 4D input`) | não gerou nada publicado |
| `best_hybrid_fold_1.pt` do Vitor | carrega com `strict=True` na `HybridUrbanHeatNet` da célula 32 (109 tensores, 13.218.829 parâmetros) | arquitetura confirmada |

O código original de treino da rede híbrida se perdeu (informação do Vitor). A célula 32 é a reconstrução dele e foi a base do retreino.

### 3.2 Dados

- **Treino:**
  - 20 municípios e 847 composições mensais Landsat 8/9 (2018 a 2024) baixadas do Earth Engine pelo Vitor;
  - receita: mediana da reflectância, média da LST, máscara do QA_PIXEL (bits 1, 3 e 4);
  - 5 canais: NDVI, NDWI, NDBI, DEM (SRTM) e LST;
  - pares consecutivos: 827.
- **Problema encontrado:** cerca de 2,4% dos pixels têm LST = 0 °C em blocos retangulares fixos, um artefato do `sampleRectangle` com `defaultValue = 0`.
  - Na v1 do retreino, eles entravam no alvo e na média e no desvio da cena, o que inflava o erro: 4,33 °C.
  - Na v2 e na final, os pixels com LST ≤ 1 °C ficam fora do alvo, da perda e das métricas (`UrbanHeatDatasetMasked`).
- **Teresina ampliada** (`data/landsat_pc.py`):
  - mesmo produto e mesma receita, lidos do Microsoft Planetary Computer, sem conta;
  - grade de 1056 × 725 px (32 × 22 km), alinhada à grade antiga;
  - DEM NASADEM;
  - os mesmos 13 meses secos.
- **Conferência na área comum:** LST com r = 0,989 (+0,13 °C) e NDVI com r = 0,957, sem deslocamento.
- **Lacuna do produto Landsat no nordeste da área:** 11% dos pixels não têm LST, com as demais bandas presentes. A composição descarta só a LST ali (`buraco_so_lst=True`).
- **Demais fontes:**
  - **IBGE Censo 2022:** bairros, setores e demografia, com as variáveis V01031 (0 a 4 anos) e V01040 + V01041 (60 anos ou mais);
  - **Overture Maps** (release 2026-09-23.1): 435.209 prédios;
  - **OpenStreetMap** via Overpass: vias, áreas verdes, equipamentos e água;
  - **INMET:** estação A312, 2018 a 2024, extraída dos zips anuais por leitura parcial.

### 3.3 Modelo e treino final (tag `v1.0.0`, commit `a6f04c6`)

- **Arquitetura:** `HybridUrbanHeatNet`, uma U-Net com:
  - blocos residuais;
  - portões de atenção;
  - Group Normalization;
  - `base_c = 64`;
  - 13,2 milhões de parâmetros.
- **Entradas:** NDVI, NDBI, DEM e NDWI, normalizados pelas estatísticas do fold.
- **Alvo:** anomalia padronizada da LST da cena seguinte, `(LST − média) / DP`, em pixels válidos.
- **Perda:** 0,8 × SSIM (janela 11) + 0,2 × L1 + 0,1 × L1 nos pixels do percentil 90.
- **`CONFIG_FINAL`:**
  - 20 épocas, LR 1e-4, weight decay 0,05, lote 2;
  - recorte de 400 px com espelhamento e rotação de 90°;
  - corte de gradiente 1,0;
  - ReduceLROnPlateau sobre a perda de treino (fator 0,5, paciência 5);
  - modelo da última época;
  - `lst_min_valida = 1,0`.
- **GPU:** A100-SXM4-40GB no Colab Pro, cerca de 28 s por época.
- **Folds:** os da Tabela 1 (IDs 01–04, 05–08, 09–12, 13–16, 17–20).

| Fold | Validação | MAE (°C) | Hotspot MAE (°C) | IoU |
|---|---|---|---|---|
| 1 | Iguatu, Parnaíba, Pedro II, Teresina | 2,34 | 3,34 | 0,226 |
| 2 | Juazeiro do Norte, Floriano, São Raimundo Nonato, Picos | 1,88 | 2,26 | 0,227 |
| 3 | Barreiras, Imperatriz, Patos, Crato | 2,01 | 2,68 | 0,215 |
| 4 | Mossoró, Bom Jesus, Uruçuí, Araguaína | 1,85 | 2,27 | 0,240 |
| 5 | Sobral, Corrente, Vitória da Conquista, Paulo Afonso | 2,09 | 2,52 | 0,241 |
| Média ± DP | | **2,04 ± 0,20** | **2,62 ± 0,44** | **0,230 ± 0,011** |

**Histórico do retreino:**

| Versão | Principal diferença | MAE (°C) |
|---|---|---|
| v1 | célula 32 sem alterações (30 épocas, LR 5e-4) | 4,33 ± 1,29 |
| v2 | máscara de LST zerada + hiperparâmetros do artigo, melhor época | 1,96 ± 0,23 |
| Final | v2 com época 20 fixa | 2,04 ± 0,20 |

O viés por escolher a melhor época pela validação era de cerca de 0,1 °C.

**Desvio-padrão das previsões** (alvo = 1), na época 20 e entre todas as épocas:

| Fold | Época 20 | Faixa |
|---|---|---|
| 1 | 0,963 | 0,90 a 0,99 |
| 2 | 0,925 | 0,87 a 0,93 |
| 3 | 0,949 | 0,92 a 1,00 |
| 4 | 0,967 | 0,86 a 0,99 |
| 5 | 0,954 | 0,92 a 1,00 |

### 3.4 Verificação independente (`analysis/verify_urbanheatnet.py`)

Mesma régua para todos: pixels com LST > 1 °C nas duas cenas; conversão para °C com a média e o DP da cena-alvo, como no treino.

| Método | MAE (°C) |
|---|---|
| UrbanHeatNet final | 2,03 ± 0,19 |
| Regressão linear (4 índices) | 2,36 ± 0,24 |
| Média da cena (anomalia zero) | 2,68 ± 0,34 |
| Persistência (LST do mês anterior; usa LST, só referência) | 1,59 ± 0,09 |

- Correlação espacial com a anomalia observada: 0,675 ± 0,022.
- Com conversão "realista" (média e DP da cena anterior), o MAE é de 4,86 °C, porque o modelo não prevê o nível médio do mês seguinte. **O que ele prevê é o padrão espacial.**
- O NDBI tem correlação de 0,85 com a LST da cena-base. É uma relação física (infravermelho de ondas curtas), não vazamento da banda térmica.

### 3.5 Teresina: composição seca e previsão

- **Composição:** mediana de 13 meses secos (2018_06, 2018_08, 2020_07, 2020_08, 2021_07, 2022_06, 2022_07, 2022_08, 2023_06, 2023_07, 2023_09, 2024_06 e 2024_07), com cada mês filtrado por nuvem.
- **Previsão:** modelo do Fold 1 (Teresina na validação).
  - Na caixa original de 12 km: MAE de 1,01 °C e r = 0,914.
  - Na área ampliada, por janelas de 400 px (passo de 200, peso maior no centro, conversão por janela): MAE de 1,29 °C e r = 0,887.
- **Incerteza:** desvio entre os 5 modelos, de 0,31 °C.
- **Anomalia:** relativa à média da área mapeada (36,0 °C), que inclui bordas rurais e rios.

### 3.6 Validação com o mundo real (`analysis/validacao_real.py`, `analysis/validacao_inmet.py`)

**Experimento natural** (seca de 2018–21 × 2023–24):

| Pixels | n | Medido | Previsto |
|---|---|---|---|
| ganharam vegetação (ΔNDVI > 0,15) | 2.786 | −1,00 °C | −1,06 °C |
| perderam vegetação (ΔNDVI < −0,15) | 3.850 | +1,06 °C | +1,20 °C |
| estáveis | 105.191 | +0,05 °C | +0,03 °C |

- Correlação de 0,75 nos pixels que mudaram.
- Inclinação: −0,41 °C (medido) × −0,45 °C (previsto) por +0,1 de NDVI.

**Parques existentes** (38 polígonos do OSM com NDVI ≥ 0,35; entorno de 90 a 240 m menos o interior):

- medido 0,76 °C e previsto 1,12 °C em média;
- r = 0,935; MAE de 0,59 °C;
- o modelo **superestima** (fator 0,68 = medido ÷ previsto).

**Ablação** (canal trocado pela média; MAE de 1,01 °C com todos): sem NDVI, +1,40; sem NDWI, +0,40; sem NDBI, +0,26; sem DEM, −0,01.

**INMET A312 × Landsat:**

- Passagens com nuvem < 20% (≈ 12h59 UTC), contra a temperatura do ar às 13h UTC.
- 82 dias; 2 descartados com LST < 15 °C (12,3 °C e −123 °C: nuvem ou pixel corrompido).
- Com 80 dias: r = 0,58 (p = 1,5 × 10⁻⁸).
- Na seca (60 dias): superfície 8,2 °C acima do ar (39,3 × 31,1 °C), r = 0,62, e 0,19 °C de ar por °C de superfície.
- Na chuvosa: +8,6 °C.
- Chuva média na estação: dezembro a abril, 170 a 372 mm por mês; junho a outubro, 1 a 27 mm.

> Correção registrada: uma versão anterior dizia "na chuvosa, superfície e ar ficam próximos" e r = 0,56. Isso era efeito do ponto de −123 °C.

### 3.7 Do mapa ao plano

- **Fração plantável** (`decision/plantable.py`):
  - método: rasterização a 3 m (fator 10) de prédios, pistas (largura por tipo de via), faixas de calçada de 2,5 m, áreas verdes e água;
  - classes: praça, calçada, livre (outros), construído e pista.
  - Totais na área ampliada: 558 ha de praças e 1.967 ha de faixas de calçada.
- **Plantio simulado:**
  - as entradas NDVI, NDBI e NDWI de cada pixel são misturadas com a assinatura da vegetação densa de Teresina (NDVI ≥ 0,5: NDVI 0,77, NDBI −0,24, NDWI −0,72), na proporção da fração plantável;
  - a temperatura não é editada;
  - resfriamento = 0,7 × desfoque gaussiano (5 px, σ 1) + 0,3 × desfoque (15 px, σ 3) da diferença.
- **Áreas candidatas** (`decision/areas_candidatas.py`):
  - **737 áreas:** 300 praças, 300 trechos de calçada e 137 áreas livres;
  - cada uma é simulada na janela de 400 px centrada nela;
  - métricas: ΔT local (ponderado pela fração), área com ΔT ≥ 0,5 °C e integral em °C·ha;
  - **aditividade:** 20 áreas juntas dão 0,727 da soma dos efeitos individuais;
  - a página usa só praças e calçadas com ΔT local ≥ 0,05 °C: **525 áreas**.
- **Limitação de método:** a soma líquida de ΔT na cidade é ≈ 0, porque o alvo é a anomalia com média zero.
  - O frescor local aparece "compensado" por um aquecimento difuso que não é físico.
  - Por isso as métricas usam só o benefício local.
- **Estratégias** (Figura 9), com 7.595 árvores:

| Estratégia | Área com pelo menos 0,5 °C a menos |
|---|---|
| maior resfriamento por árvore (proposta) | 164 ha |
| aleatória (média de 50) | 79 ha |
| mais quentes primeiro | 32 ha |

- **Por que "mais quentes primeiro" perde:** os pontos mais quentes são construídos e têm pouca fração plantável.
- **Número de árvores:** espaçamento do Manual de Fortaleza: médio porte (8 m) nas praças e pequeno porte (5 m) ao longo das calçadas.

### 3.8 Bairros e espécies

- **Bairros** (`decision/bairros.py`):
  - métricas: temperatura média, anomalia, % em ilha de calor (pixels acima do percentil 90, 40,6 °C) e % de vegetação densa;
  - população e vulneráveis: crianças de 0 a 4 anos + pessoas com 60 anos ou mais;
  - índice = 60% × posição em calor + 40% × posição em vulneráveis por km².
- **Espécies** (`decision/especies.py`):
  - as 102 espécies das tabelas 12 a 16 do Manual de Fortaleza (p. 120 a 123), conferidas na Flora e Funga do Brasil via API do GBIF;
  - ficam **59 árvores e 3 palmeiras nativas da Caatinga ou do Cerrado que ocorrem no PI**: 22 de pequeno porte, 27 de médio e 10 de grande;
  - Machado et al. (2006), específico de Teresina, não foi conferido (site da revista fora do ar).

### 3.9 Página (`docs/`)

- **Tecnologia:** HTML, CSS e JavaScript estáticos, com Leaflet e Chart.js, sem servidor. Os dados são exportados por `decision/export_pagina.py`.
- **Modo Gestor:**
  - quantas árvores (ou hectares);
  - estratégia (maior resfriamento por árvore ou seleção manual);
  - filtros (praça ou calçada, bairro, até X m de escola, creche ou saúde);
  - resultado (árvores, ha, ΔT local, área com pelo menos 0,5 °C a menos, bairros, áreas perto de equipamentos);
  - "Gerar plano completo" (`plano.html`): mapa do trecho, áreas com coordenadas e link para o Google Maps, bairros, espécies, confiabilidade e fontes;
  - aba de bairros (índice com peso ajustável e ficha do bairro), aba de confiabilidade e aba Sobre.
- **Modo Cidadão:**
  - busca do bairro;
  - faixa "se o bairro for arborizado": o teto é o ΔT local da melhor área do bairro e o piso é o teto × 0,68 (correção dos parques);
  - estimativa no ar (× 0,19);
  - passo a passo para cobrar a Prefeitura: Ouvidoria, e-SIC (Lei nº 12.527/2011), vereador, audiência do orçamento (LRF) e acompanhamento.
- **Retirado para simplificar** (tag `v1.1-completa`):
  - custos SINAPI-PI (98510: R$ 115,24; 98511: R$ 209,23);
  - cenários;
  - linha do tempo do crescimento (premissa sem fonte local);
  - desenho livre (ONNX fp16, 26,5 MB, WebGPU ≈ 3,5 s, conferido contra o Python);
  - áreas livres privadas.

### 3.10 Como reproduzir

```bash
git clone https://github.com/abraaolm-maker/arbormap.git
pip install -r requirements.txt
# treino (Colab): notebooks/refazer_urbanheatnet.ipynb (CONFIG_FINAL)
# validações, camadas, áreas e página: módulos em src/arbormap/analysis e src/arbormap/decision
# figuras do artigo v4:
python -c "from arbormap.analysis.figuras_artigo_v4 import main; main('DADOS', 'SAIDA')"
```

- Para publicar no repositório do Vitor: copiar o repositório e ativar o GitHub Pages (*Settings → Pages → main, /docs*).
- Os dados grandes (`.npy`, checkpoints `.pt`, camadas derivadas) ficam fora do Git e estão na pasta `_dados` da máquina do Abraão e no Drive (`arbormap_final/`).

### 3.11 Pendências

- Medição de campo (10 a 15 pontos escolhidos no mapa).
- Conferir as espécies com Machado et al. (2006) e com o viveiro municipal.
- DOI no Zenodo.
- Revisão final do texto pelo autor e pelos orientadores.

---

## Parte 4. Figuras e tabelas da V4

| Nº | Arquivo | Conteúdo | Origem |
|---|---|---|---|
| 1 | `Figura01_municipios_por_fold.png` | 20 municípios por fold (Uruçuí corrigido) | IBGE (malhas), gerada |
| 2 | `Figura02_curvas_treinamento.png` | perda do Fold 2 e MAE de validação dos 5 folds | histórico do treino final (Drive) |
| 3 | `Figura03_desvio_padrao.png` | DP das previsões × alvo, 5 folds | histórico do treino final |
| 4 | `Figura04_comparacao_referencias.png` | MAE do modelo × referências | `verificacao.json` (Drive) |
| 5 | `Figura05_teresina_observado_previsto_erro.png` | Teresina inteira: observado, previsto, erro | composição seca, área ampliada |
| 6 | `Figura06_validacao_mundo_real.png` | experimento natural e parques | `validacao_real.json` |
| 7 | `Figura07_superficie_x_ar_inmet.png` | LST × ar (INMET) | `validacao_inmet.json` |
| 8 | `Figura08_bairros_calor_prioridade.png` | calor por bairro e prioridade | bairros + Censo 2022 |
| 9 | `Figura09_area_beneficiada_x_arvores.png` | estratégias: área com pelo menos 0,5 °C a menos × árvores | áreas candidatas |
| 10 | `Figura10_pagina_gestor.png` | página: modo Gestor (plano) | print da página online (Claude in Chrome) |
| 11 | `Figura11_pagina_cidadao.png` | página: modo Cidadão (faixa e passo a passo) | print da página online |
| extra | `Extra_pagina_bairros.png`, `Extra_pagina_confiabilidade.png`, `Extra_plano_completo.png` | outras telas | prints |

**Tabela 1:** municípios (sem mudança). **Tabela 2:** métricas por fold do modelo final (época 20).

**Diagramação:** as Figuras 5, 10 e 11 são largas e ficam pequenas na coluna de 8,8 cm. Na versão final, vale colocá-las ocupando as duas colunas (seção de uma coluna só para a figura) ou usar as versões separadas em `figuras/`.

**A conferir na revisão final:** a referência COOK et al. (ano e título, porque o texto da v3 citava 2014 e a lista trazia 2016) e o ano da publicação do IBGE usada (Censo 2022, agregados por setores).
