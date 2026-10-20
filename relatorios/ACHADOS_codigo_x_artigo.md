# ARBORMAP: conferência código × artigo (19/10/2026, revisão 2)

Fontes conferidas:

- `arbormap.ipynb` (34 células) e `code.ipynb`;
- `ARBORMAP_artigo_FEBRACE_final_v3.docx`;
- logs e figuras do repositório.

Foram executados só testes de arquitetura (PyTorch CPU, com entradas sintéticas). Nenhum treino foi rodado e nenhum dado do Drive foi acessado.

## Resumo

| Elemento do artigo | De onde vem, de fato | Confere com o texto? |
|---|---|---|
| Tabela 2 (MAE 3,64 ± 0,48; HS MAE 2,75 ± 0,58) | log da célula 6 = `logs/01`, modelo **UNetConvLSTM** (célula 4), 8 canais com LST passada | números sim; modelo não |
| Figura 2 (curvas, Fold 2, 40 épocas) | mesma rodada da Tabela 2 | sim (mas `training_curves.py` do repositório é versão anterior) |
| Figura 3 (desvio-padrão) e Seção IV-B (0,60 a 0,80) | `logs/06`, Fold 1, 20 épocas, rodada com 21 cidades (família ASPP, célula 18) | não: outra rodada, outro modelo, outro fold |
| Figura 4 (observado × previsto × erro) | código não encontrado nos arquivos; escala de −1 a 2 rotulada "°C" parece anomalia normalizada | indeterminado |
| Figuras 5 e 6, redução de 2,86 °C | célula 7 = `intervention_simulation.py`, **UNetConvLSTM** com `best_convlstm_fold_0.pt` | modelo não |
| Tabela 1 e Figura 1 (cidades por fold) | lista escrita à mão em `fold_map.py`, diferente da divisão real | não (só Teresina no Fold 1 confere) |

## 1. Tabela 2: qual código gerou o log

Conclusão: **célula 4 (UNetConvLSTM)**. Três evidências independentes:

1. **Teste executado.** Extraí as classes das células 4, 29 e 32 e passei pelos modelos o tensor que o `UrbanHeatTemporalDataset` entrega, no formato (lote, tempo, canal, H, W) = (2, 2, 8, 64, 64).
   - Célula 4, `UNetConvLSTM`: processa normalmente, saída (2, 1, 64, 64).
   - Célula 29, `HybridUrbanHeatNet` com 8 canais: falha no primeiro lote com `RuntimeError: Expected 3D or 4D input to conv2d, but got input of size [2, 2, 8, 64, 64]`. Do jeito que está, ela não chega a treinar uma época, muito menos 5 folds × 40 épocas. Além disso, a célula 29 usa `ConvBNAct` sem definir a classe; ela dependia de outra célula executada antes.
2. **Ordem do notebook.** Os logs colados aparecem logo depois do código que os gerou: célula 8 (log da simulação) depois da 7; células 22 e 23 depois da família da 18. O log da Tabela 2 (célula 6) vem logo depois da célula 4. A célula 5 é só um utilitário de nuvens.
3. **Hiperparâmetros.** 40 épocas, filtro 1036 → 847 e formato de print idêntico ao da célula 4.

O que diverge do artigo (III-B, III-D, III-E):

- o modelo é o ConvLSTM, que o artigo diz ter sido abandonado;
- usa 8 canais, incluindo a LST da cena anterior, e não 4;
- usa BatchNorm, não GroupNorm;
- a perda é L1 + hotspot (peso 2,0), sem gradiente espacial e sem SSIM;
- LR 3e-4 e weight decay 1e-4 (o artigo diz 1e-4 e 0,05);
- recorte de 256 px (o artigo diz 400).

Os hiperparâmetros citados no artigo (LR 1e-4, weight decay 0,05, GroupNorm, perda com gradiente) são os da célula 18 (ASPP), que não gerou a Tabela 2.

O `break` de `hybrid_pipeline.py` (célula 32) não afeta o artigo: essa célula não gerou nenhum resultado publicado.

## 2. Simulação (2,86 °C) e um problema de método

- A Figura 5 ("Métricas (Budget=45 px)", "MAE Modelo: ±2.3 °C") só é produzida pela célula 7. Ela carrega `UNetConvLSTM(in_channels=8)` com `best_convlstm_fold_0.pt`.
- O `logs/02` (saída dessa célula, sem aviso de pesos aleatórios) é coerente com a Figura 5 e com a Figura 6: −2,82 °C com 40 px, −3,13 °C com 50 px, −4,41 °C com 150 px.
- **Ponto de método:** ao simular o plantio, a célula 7 escreve diretamente no canal de LST passada dos pixels plantados (`simulated_input[0, :, 4, mask] = -0.20`, comentado como "a área já era 1,0 °C mais fria"). Parte do resfriamento previsto é, portanto, imposta como entrada, e não aprendida a partir da vegetação. Com a UrbanHeatNet de 4 canais (sem LST de entrada), esse atalho deixa de existir.
- A única tentativa registrada de rodar a simulação com a rede híbrida (célula 30, `best_hybrid_fold_1.pt`) falhou com erro de `state_dict`. A suposição do handoff de que os mapas de Teresina vêm de `best_hybrid_fold_1.pt` não se confirma.

## 3. Cidades por fold (reconstrução, sem acesso ao Drive)

Todas as células dividem com `KFold(5, shuffle=True, random_state=42)` sobre `Path.iterdir()`. A divisão, então, depende só da ordem em que o Drive lista as pastas.

- **Regra confirmada numa rodada anterior.** A divisão de 21 cidades registrada no notebook (imagem salva na célula 20 e `VAL_CITIES` da célula 17) é reproduzida em **5 de 5 folds** quando as pastas estão na ordem de criação: Piauí, depois Altos e José de Freitas, depois Campina Grande, Caruaru e Feira de Santana, depois o bloco do Nordeste na ordem do dicionário `CIDADES`. As contagens de imagens fecham com o `logs/06`: o Fold 1 soma 242 imagens = 1058 − 816.
- **Divisão provável da rodada da Tabela 2 (20 cidades),** com a mesma regra (5 pastas removidas, 4 criadas no fim):

| Fold | Cidades de validação |
|---|---|
| 1 | Teresina, Parnaíba, Sobral, Patos |
| 2 | Floriano, São Raimundo Nonato, Pedro II, Mossoró |
| 3 | Picos, Barreiras, Araguaína, Crato |
| 4 | Bom Jesus, Vitória da Conquista, Imperatriz, Iguatu |
| 5 | Corrente, Uruçuí, Juazeiro do Norte, Paulo Afonso |

- Teresina no Fold 1 confere com o artigo. As demais cidades da Tabela 1 e da Figura 1 não conferem.
- O artigo não tem como responder isso sozinho, porque a Tabela 1 foi montada à mão.
- Confirmação definitiva (Colab, segundos, sem retreino):

```python
from pathlib import Path
from sklearn.model_selection import KFold
all_cities = [d.name for d in Path("/content/drive/MyDrive/heat_dataset_v3").iterdir() if d.is_dir() and not d.name.startswith('.')]
for f, (_, v) in enumerate(KFold(5, shuffle=True, random_state=42).split(all_cities), 1):
    print(f, [all_cities[i] for i in v])
```

- Detalhe da Figura 1: no `fold_map.py`, o código IBGE `2211100` aparece como Uruçuí, mas o polígono desenhado (rótulo 15) fica ao norte de Teresina, posição compatível com União (PI). Conferir o código de Uruçuí.

## 4. Inconsistências do handoff, revistas

- **SSIM:** não deve entrar no texto. Só a célula 32 usa SSIM, e ela não gerou nenhum resultado publicado.
- **ASPP:** é a célula 18 (`legacy/v2_aspp`). Os logs 04 a 07 vêm dessa família, inclusive os 06/07, que o README dos logs chama de "versão híbrida final" (rótulo a corrigir).
- **`best_hybrid_fold_1.pt`:** não foi usado em nenhum resultado publicado.

## 5. Informações do Vitor (19/10/2026) e conferência do checkpoint

- O Vitor informou que **apagou sem querer o código original** usado para treinar o modelo híbrido. A célula 32 ("o último código grande") é a reconstrução feita por ele, compatível com o modelo treinado; ele mesmo avisa que "talvez o resultado ao rodar seja diferente".
- Ele enviou `best_hybrid_fold_1.pt` (Fold 2, 50 MB). Conferência: o arquivo é um `state_dict` que carrega com `strict=True` no `HybridUrbanHeatNet` da célula 32 (109 tensores, 13.218.829 valores, entrada de 4 canais, gargalo residual sem ASPP). **A arquitetura do modelo treinado é idêntica à da célula 32.**
- Os hiperparâmetros, a perda e a divisão de folds usados naquele treino não estão registrados, porque o código se perdeu.
- Isso reforça a conclusão do item 1: o log que gerou a Tabela 2 tem o formato das células 4/29, e não há registro de treino com a arquitetura híbrida. O retreino com a célula 32 (`train_urbanheatnet.py`) é o que vai produzir números rastreáveis.
- Retreino iniciado em 19/10/2026 no Colab Pro: primeiro na T4 (cerca de 250 s/época, 2 épocas descartadas), depois recomeçado do zero numa NVIDIA A100-SXM4-40GB (cerca de 29 s/época, cerca de 75 min no total), para que os cinco folds rodem na mesma GPU. O log parcial da T4 ficou como `treino_log_T4_interrompido.txt`. Fold 1, épocas 1 a 5 na A100: MAE entre 4,79 e 5,79 °C (ainda no início do treino). A GPU deve ser declarada no artigo (Seção III-E diz "GPU T4").

## 6. Resultados do retreino com a UrbanHeatNet (A100, 19/10/2026)

Configuração: célula 32 sem alterações de modelo, perda ou hiperparâmetros (30 épocas, LR 5e-4, weight decay 1e-4, lote 2, recorte 400 px); folds da Tabela 1; normalização por fold. Arquivos no Drive, em `arbormap_urbanheatnet/`.

**Tabela 2 nova (melhor época por fold):**

| Fold | Validação | Melhor época | MAE (°C) | Hotspot MAE (°C) | IoU |
|---|---|---|---|---|---|
| 1 | Iguatu, Parnaíba, Pedro II, Teresina | 4 | 4,79 | 3,76 | 0,167 |
| 2 | Juazeiro, Floriano, S. R. Nonato, Picos | 3 | 4,51 | 3,15 | 0,143 |
| 3 | Barreiras, Imperatriz, Patos, Crato | 1 | 3,13 | 2,68 | 0,127 |
| 4 | Mossoró, Bom Jesus, Uruçuí, Araguaína | 2 | 3,07 | 2,85 | 0,158 |
| 5 | Sobral, Corrente, V. da Conquista, P. Afonso | 2 | 6,15 | 3,33 | 0,156 |
| Média ± DP | | | **4,33 ± 1,29** | **3,16 ± 0,42** | 0,150 ± 0,015 |

Comparação com o artigo atual (gerado pelo ConvLSTM): MAE 3,64 ± 0,48 e Hotspot MAE 2,75 ± 0,58. A UrbanHeatNet ficou pior nas duas métricas.

**Problemas encontrados:**

1. **Sobreajuste precoce.** A melhor época vai de 1 a 4 em todos os folds; depois disso a perda de treino cai e a de validação sobe (Figura 2, Fold 2). As 26 a 29 épocas restantes não ajudam.
2. **Pixels de LST = 0 no dataset.** Cerca de 2,4% dos pixels de cada cena de Teresina têm LST exatamente 0 °C, espalhados pela imagem (lacunas de nuvem gravadas como 0 pelo download; o `sampleRectangle` usa `defaultValue = 0`). No treino eles entram como alvo e na média da cena (`macro_climate`), funcionando como "ilhas frias" falsas. Corrigido só na avaliação e no mapa; o treino ainda os usa.
3. **Escolha da cena.** O par mais recente (nov → dez/2024, chuvoso) tinha desvio-padrão de LST de 15,7 °C e MAE de 9,9 °C, o que inflava o resfriamento para 5 a 8 °C. Passou-se a usar o par mais recente da estação seca com céu limpo: jun → jul/2024 (desvio-padrão de 6,5 °C). Nessa cena, sem os pixels zerados, **MAE = 2,04 °C**.

**Simulação em Teresina (modelo do Fold 1, cena jun → jul/2024):**

| Orçamento | Multicritério | Gulosa | Aleatória (média ± DP, 20 sorteios) |
|---|---|---|---|
| 10 px (0,9 ha) | 2,18 °C | 2,00 °C | 0,37 ± 0,02 °C |
| 30 px (2,7 ha) | 2,23 °C | 2,38 °C | 0,37 ± 0,02 °C |
| 45 px (4,05 ha) | **2,68 °C** | **3,31 °C** | 0,37 ± 0,02 °C |
| 100 px (9 ha) | 2,70 °C | 3,44 °C | 0,37 ± 0,01 °C |
| 150 px (13,5 ha) | 2,74 °C | 3,36 °C | 0,38 ± 0,01 °C |

**A estratégia gulosa supera a multicritério a partir de 30 px.** O artigo afirma o contrário (2,86 contra 1,51 °C), mas aquele resultado vinha do ConvLSTM, que editava o canal de temperatura da entrada. As duas superam a aleatória por larga margem. A margem sobre a incerteza também mudou: no Fold 1, o Hotspot MAE é de 3,76 °C, acima dos 2,68 °C da multicritério.

**Mapa:** arquivos em `arbormap_urbanheatnet/mapa/`, com 22 corredores (35,5 ha) e legenda de 27,6 a 42,9 °C. Os limites geográficos são aproximados (`USAR_EE = False`).

**Por que a UrbanHeatNet (v1) ficou pior que a Tabela 2 do artigo:**

1. **Tarefa mais difícil.** O modelo da Tabela 2 atual (ConvLSTM, célula 4) recebia a LST da cena anterior entre os 8 canais, ou seja, prevê em boa parte por persistência do mapa térmico. A UrbanHeatNet usa só NDVI, NDBI, DEM e NDWI e precisa inferir a temperatura da cobertura do solo. É a formulação coerente com a simulação de plantio, mas é mais difícil.
2. **Sobreajuste.** São 13,2 milhões de parâmetros para cerca de 650 pares correlacionados, com LR 5e-4 e weight decay 1e-4.
3. **Pixels de LST = 0 no alvo** (cerca de 2,4% por cena): o modelo é punido por não prever 0 °C em pontos aleatórios, e esses zeros distorcem a média e o desvio-padrão de cada cena.
4. **Vitória da Conquista** (923 m, fora da faixa das demais cidades) eleva o MAE do Fold 5.

## 7. Retreino v2 (iniciado em 20/10/2026, 00:31, A100)

`CONFIG_V2`: hiperparâmetros que o artigo descreve (40 épocas, LR 1e-4, weight decay 0,05) e pixels de LST ≤ 1 °C excluídos do alvo, da perda e das métricas (`UrbanHeatDatasetMasked`, testada contra a classe original). Resultados em `arbormap_urbanheatnet_v2/`; a v1 fica preservada em `arbormap_urbanheatnet/`.

- Primeira época do Fold 1: MAE de 2,50 °C, contra 5,20 °C na v1.
- Ritmo de 29 s/época, cerca de 100 min no total.

### Resultados v2 (concluído em 20/10/2026)

**Tabela 2 (melhor época por fold):**

| Fold | Validação | Melhor época | MAE (°C) | Hotspot MAE (°C) | IoU |
|---|---|---|---|---|---|
| 1 | Iguatu, Parnaíba, Pedro II, Teresina | 17 | 2,29 | 3,31 | 0,226 |
| 2 | Juazeiro, Floriano, S. R. Nonato, Picos | 18 | 1,80 | 2,26 | 0,237 |
| 3 | Barreiras, Imperatriz, Patos, Crato | 16 | 1,86 | 2,21 | 0,259 |
| 4 | Mossoró, Bom Jesus, Uruçuí, Araguaína | 2 | 1,74 | 1,96 | 0,271 |
| 5 | Sobral, Corrente, V. da Conquista, P. Afonso | 7 | 2,09 | 2,28 | 0,272 |
| Média ± DP | | | **1,96 ± 0,23** | **2,40 ± 0,52** | **0,253 ± 0,021** |

**Comparação entre as três versões:**

| Versão | MAE (°C) | Hotspot MAE (°C) | IoU |
|---|---|---|---|
| Artigo atual (ConvLSTM) | 3,64 ± 0,48 | 2,75 ± 0,58 | 0,189 ± 0,064 |
| UrbanHeatNet v1 | 4,33 ± 1,29 | 3,16 ± 0,42 | 0,150 ± 0,015 |
| UrbanHeatNet v2 | **1,96 ± 0,23** | **2,40 ± 0,52** | **0,253 ± 0,021** |

A v2 é melhor nas três métricas e fica perto da meta de projeto de 1,5 °C. A melhor época passou a cair entre 16 e 18 na maioria dos folds; no Fold 4 é a 2 e no Fold 5 é a 7.

**Simulação v2 (Teresina, modelo do Fold 1, cena escolhida automaticamente: jul → ago/2024, desvio-padrão de 3,13 °C, MAE na cena de 1,21 °C):**

| Orçamento | Multicritério | Gulosa | Aleatória |
|---|---|---|---|
| 10 px (0,9 ha) | 2,12 °C | 2,36 °C | 0,65 ± 0,08 °C |
| 20 px (1,8 ha) | 3,72 °C | 3,38 °C | 0,62 ± 0,04 °C |
| 30 px (2,7 ha) | 3,74 °C | 3,74 °C | 0,61 ± 0,04 °C |
| 45 px (4,05 ha) | 3,20 °C | 4,90 °C | 0,64 ± 0,03 °C |
| 100 px (9 ha) | 3,58 °C | 6,20 °C | 0,64 ± 0,03 °C |
| 150 px (13,5 ha) | 3,56 °C | 5,93 °C | 0,65 ± 0,02 °C |

Leitura da simulação, com 45 px:

- **Redução média nos pixels plantados:** a gulosa vence (4,90 contra 3,20 °C), porque planta nos pixels mais quentes, onde há mais o que reduzir.
- **Área beneficiada** (pixels com ΔT acima de 0,1 °C): a multicritério vence, com 1.013 pixels contra 502 da gulosa e 4.698 da aleatória. Na aleatória o efeito é espalhado, mas fraco (0,64 °C nos pixels plantados).
- **Limitação de método:** a soma de ΔT na cidade é aproximadamente zero, ligeiramente negativa nas três estratégias. O alvo do modelo é a anomalia relativa à média da cena, então a simulação estima resfriamento local relativo, não redução líquida da temperatura da cidade. Isso precisa ser declarado no artigo (Seções III-C, IV-E e IV-G).
- **Margem sobre a incerteza:** o Hotspot MAE do Fold 1 é de 3,31 °C. A gulosa (4,90 °C) fica acima dele e a multicritério (3,20 °C) fica um pouco abaixo.

**Arquivos (Drive, `arbormap_urbanheatnet_v2/`):** `tabela2.csv`, `resultados_kfold.json`, `config.json`, `treino_log.txt`, `figuras/` (Figuras 2 a 6), `teresina_simulacao.json` e `mapa/` (27 corredores, 31,95 ha, legenda de 31,7 a 45,0 °C, limites geográficos aproximados).

## 8. Verificação independente da v2 (pedida pelo Vitor, 20/10/2026)

Código: `src/arbormap/analysis/verify_urbanheatnet.py`. Resultados: `arbormap_urbanheatnet_v2/verificacao/` no Drive (`verificacao.json` e `curvas_todos_folds.png`). Todas as métricas foram medidas nos mesmos pixels (recorte com LST > 1 °C nas duas cenas), com v1 e v2 avaliadas pela mesma régua.

1. **Arquitetura:** a mesma `HybridUrbanHeatNet` da célula 32 (e do checkpoint do Vitor) nas duas versões. Mudaram só os dados do alvo e os hiperparâmetros.
2. **Entradas:** tensor 4 × 400 × 400 = NDVI, NDBI, DEM, NDWI da cena-base. Nenhum canal é a LST (`igual_LST = False` em todos). O NDBI tem correlação de 0,85 com a LST da cena-base, uma relação física conhecida: o índice usa a refletância do infravermelho de ondas curtas, não a banda térmica. A LST não entra como entrada nem como Δ.
3. **"Está em °C?"** Sim, mas é o erro da **anomalia espacial**. A conversão usa a média e o desvio-padrão da própria cena-alvo, que não se conhecem no momento da previsão. Por isso:
   - na régua do treino, v2 = **1,94 ± 0,24 °C**, valor que bate com o do treino (1,96);
   - com conversão "realista" (média e desvio-padrão da cena anterior), v2 = 4,82 ± 0,33 °C, porque erra o nível médio do mês seguinte, que a rede não prevê.

   O artigo deve dizer que o modelo prevê o **padrão espacial** (onde é mais quente ou mais frio) e não a temperatura absoluta do mês.
4. **Comparação com referências simples (mesma régua):**

| Método | MAE (°C) |
|---|---|
| Prever a média da cena (anomalia zero) | 2,68 ± 0,34 |
| Regressão linear com os 4 índices | 2,36 ± 0,24 |
| **UrbanHeatNet v2** | **1,94 ± 0,24** |
| UrbanHeatNet v1 | 3,02 ± 0,59 (pior que prever a média) |
| Persistência: LST do mês anterior (usa LST, só referência) | 1,59 ± 0,09 |

   A v2 supera a média e a regressão linear em todos os folds. A persistência é melhor, mas usa a temperatura como entrada, que é justamente o que o ConvLSTM do artigo atual fazia.
5. **A melhora da v1 para a v2 é sobretudo de escala, não de padrão.** A correlação espacial entre a anomalia prevista e a observada é quase igual: 0,694 ± 0,033 na v1 e 0,706 ± 0,033 na v2. A queda do MAE (3,02 → 1,94) vem principalmente de retirar do alvo os pixels zerados, que inflavam o desvio-padrão usado para converter em °C. A Tabela 2 da v1 estava distorcida por esse problema dos dados, não por um modelo pior.
6. **Sobreajuste ou subajuste** (curvas dos 5 folds): a perda de treino cai sempre; a de validação fica quase plana nos Folds 1 e 5 e sobe depois de ~20, ~16 e 2 épocas nos Folds 2, 3 e 4. Há sobreajuste leve: o modelo aprende o que generaliza nas primeiras épocas e depois passa a decorar as cidades de treino. Não há subajuste.
7. **Viés da escolha da melhor época:** a melhor época é escolhida pelas próprias cidades de validação, o que é otimista. Com época fixa, igual em todos os folds:

| Seleção | MAE (°C) | Hotspot MAE (°C) | IoU |
|---|---|---|---|
| Época 20 | 2,05 ± 0,20 | 2,61 | 0,230 |
| Época 40 | 2,12 ± 0,16 | 2,78 | 0,214 |
| Melhor época por fold | 1,96 ± 0,23 | 2,40 | 0,253 |

   Recomenda-se reportar a época fixa (por exemplo, a 20), ou declarar o viés de cerca de 0,1 a 0,15 °C.

## 9. Versão oficial final (v1.0.0, 20/10/2026)

`CONFIG_FINAL`: a v2 com **20 épocas fixas**, modelo da última época e taxa de aprendizado ajustada pela perda de treino. As cidades de validação não influenciam nem a escolha do modelo nem o treino. GPU: NVIDIA A100-SXM4-40GB. Código: tag `v1.0.0` (commit `a6f04c6`). Resultados no Drive, em `arbormap_final/`.

**Tabela 2 final (época 20):**

| Fold | Validação | MAE (°C) | Hotspot MAE (°C) | IoU |
|---|---|---|---|---|
| 1 | Iguatu, Parnaíba, Pedro II, Teresina | 2,34 | 3,34 | 0,226 |
| 2 | Juazeiro, Floriano, S. R. Nonato, Picos | 1,88 | 2,26 | 0,227 |
| 3 | Barreiras, Imperatriz, Patos, Crato | 2,01 | 2,68 | 0,215 |
| 4 | Mossoró, Bom Jesus, Uruçuí, Araguaína | 1,85 | 2,27 | 0,240 |
| 5 | Sobral, Corrente, V. da Conquista, P. Afonso | 2,09 | 2,52 | 0,241 |
| Média ± DP | | **2,04 ± 0,20** | **2,62 ± 0,44** | **0,230 ± 0,011** |

**Verificação independente (mesma régua, pixels limpos):**

| Método | MAE (°C) |
|---|---|
| Modelo final | 2,03 ± 0,19 |
| Regressão linear com os 4 índices | 2,36 ± 0,24 |
| Prever a média da cena | 2,68 ± 0,34 |
| Persistência (usa LST; só referência) | 1,59 ± 0,09 |

- O modelo supera a regressão linear e a média da cena em todos os 5 folds.
- Correlação espacial com a anomalia observada: 0,675 ± 0,022.
- Entradas conferidas: só NDVI, NDBI, DEM e NDWI.

**Simulação em Teresina** (modelo do Fold 1, que nunca viu Teresina; checkpoint conferido por `val_cities`):

- **Cena:** escolhida do par mais recente para trás até achar um sem nuvem. Os pares de nov → dez, out → nov, set → out, ago → set e jul → ago/2024 foram rejeitados; o escolhido foi **jun → jul/2024** (índice 39), com escore de nuvem de 2,4% e 3,5%. Na cena: desvio-padrão de 3,06 °C e **MAE de 1,20 °C**.
- **Áreas:** 32 corredores (32,9 ha); 27.305 pixels plantáveis.

| Orçamento | Multicritério | Gulosa | Aleatória | Área > 0,1 °C (ha): multi / gulosa / aleatória |
|---|---|---|---|---|
| 10 px (0,9 ha) | 2,32 °C | 1,41 °C | 0,66 °C | 16,9 / 41,9 / 88,7 |
| 20 px (1,8 ha) | 2,27 °C | 2,08 °C | 0,67 °C | 54,9 / 51,1 / 190,8 |
| 45 px (4,05 ha) | 2,52 °C | 4,21 °C | 0,66 °C | 107,7 / 76,8 / 462,1 |
| 100 px (9 ha) | 3,35 °C | 5,32 °C | 0,66 °C | 203,2 / 135,1 / 879,9 |
| 150 px (13,5 ha) | 3,72 °C | 4,95 °C | 0,67 °C | 236,0 / 224,6 / 1.096,8 |

**Leitura:**

- A multicritério vence a gulosa em orçamentos pequenos (10 a 20 px). A partir de 30 px, a gulosa concentra mais resfriamento nos pixels plantados.
- **A métrica "área > 0,1 °C" não discrimina bem:** a aleatória espalha pixels isolados e soma muita área de efeito fraco (0,66 °C nos pixels plantados). Ela deve ser substituída por um limiar mais alto (0,5 ou 1 °C) ou pela integral do resfriamento (°C·ha). A fazer na etapa 2 do plano.
- **Dados:** as lacunas na LST observada (cerca de 2,4% em todas as cenas) aparecem como **blocos retangulares** na Figura 4, o que sugere artefato de preenchimento do download, não nuvem. Isso reforça a correção dos dados na origem.

## 10. Recomendação para a simulação e o mapa

Refazer com a UrbanHeatNet (célula 32, 4 canais), mantendo como plano B a correção do texto:

1. Treinar a célula 32 nos cinco folds, sem o `break`. O artigo estima cerca de 45 min por fold em T4.
2. Recalcular a Tabela 2 e as Figuras 2 a 4 a partir desse treino, gravando o log inteiro.
3. Adaptar a simulação ao modelo de 4 canais, sem escrever temperatura na entrada, e gerar as Figuras 5 e 6.
4. Exportar dessa mesma rodada os arquivos do mapa (passo 2).

Se o prazo não comportar esse caminho, a alternativa honesta é reescrever as Seções III-B, III-D e III-E e a conclusão para descrever o ConvLSTM de 8 canais que de fato gerou os números, e declarar a limitação da simulação.

## 11. Validação com o mundo real e simulação no navegador (20/10/2026)

Modelo: `best_hybrid_fold_0.pt` (Teresina na validação). Código: `src/arbormap/analysis/validacao_real.py`; resultados em `_dados/derivados/teresina_validacao_real.json` e na aba "Confiabilidade" da página.

**5.1 Experimento natural** (estação seca 2018–21 × 2023–24, 5 cenas em cada período, sem treino novo):

| Pixels | n | ΔT observado | ΔT previsto |
|---|---|---|---|
| ganharam vegetação (ΔNDVI > 0,15) | 2.786 | −1,00 °C | −1,06 °C |
| perderam vegetação (ΔNDVI < −0,15) | 3.850 | +1,06 °C | +1,20 °C |
| estáveis (\|ΔNDVI\| < 0,05) | 105.191 | +0,05 °C | +0,03 °C |

Correlação pixel a pixel nos que mudaram: 0,75 (MAE 1,03 °C). Inclinação: −0,41 °C (observado) × −0,45 °C (previsto) por +0,1 de NDVI. O mecanismo usado no plantio simulado (mais NDVI → menos calor relativo) se confirma, com leve superestimativa (~9%).

**5.2 Parques existentes** (38 polígonos do OSM com NDVI ≥ 0,35 e ≥ 10 px; entorno de 90 a 240 m menos interior): observado 0,76 °C, previsto 1,12 °C em média; correlação 0,935; MAE 0,59 °C. O modelo ordena bem os parques, mas **superestima** o frescor em ~0,36 °C. A página trata os resfriamentos simulados como limite superior.

**5.5 Ablação** (canal trocado pela média da cidade; composição de Teresina, MAE 1,01 °C com tudo): sem NDVI +1,40 °C; sem NDWI +0,40; sem NDBI +0,26; sem DEM −0,01 (não usa o relevo).

**Integral de resfriamento só positiva × líquida** (amostra de 40 áreas candidatas): todas têm ΔT local positivo (praças 0,40 °C, calçadas 0,19, livres 0,70), mas a soma líquida na cidade é ≈ 0 (razão líquido/positivo −0,03 na mediana). O alvo é anomalia padronizada por cena (média zero), então o frescor local volta como aquecimento difuso no resto da cidade, efeito do desenho do alvo e não físico. A métrica °C·ha mede só o benefício local; isso está dito na página. Plantios muito esparsos (fração plantável média ~11% por pixel, calçadas no Centro) podem não ter efeito detectável (exemplo testado: −0,11 °C, ruído).

**2.3 Modelo no navegador:** ONNX fp32 53 MB; fp16 nos pesos 26,5 MB (diferença máx. 0,004 °C); int8 13,4 MB, mas até 0,27 °C de diferença (descartado). onnxruntime-web 1.20.1: WebGPU ≈ 3,5 s por previsão, WASM ≈ 30 s (sem multithread no GitHub Pages). Janelas menores que a cidade erram ~20% (a rede usa o contexto da cena inteira), então a previsão é sempre da cidade toda. Conferência JS × Python no mesmo caso: 1,4326 × 1,4331 °C local; 30,49 × 30,50 °C·ha; área ≥ 0,5 °C idêntica. Modo "desenhar área livre" liberado.

**Custos e espécies:** referência SINAPI 98510 (plantio de árvore ornamental, muda ≤ 2 m), R$ 95,15, média nacional 09/2026, não desonerado (via orcamentor.com); falta o valor do Piauí (planilha SINAPI-PI) ou a ata da SEMAM. Espécies: Machado, Meunier, Silva e Castro (2006), "Árvores nativas para a arborização de Teresina, Piauí", REVSBAU v. 1, n. 1, p. 10–18, DOI 10.5380/revsbau.v1i1.66226; o site da revista estava em manutenção e a lista não foi conferida no texto completo, por isso ainda não está na página.
**Atualização do custo (20/10/2026):** a referência nacional de terceiros (R$ 95,15, orcamentor.com) foi substituída pela planilha oficial `SINAPI-2026-09-formato-xlsx.zip` (Caixa, 15,9 MB, em `_dados/sinapi_2026_09/`), aba CSD, coluna PI (Teresina), sem desoneração: **98510 = R$ 115,24** (muda de até 2 m; conferido: 0,2117 h × R$ 24,37 + 1,0584 h × R$ 23,66 + muda 358 R$ 85,05 = R$ 115,25) e **98511 = R$ 209,23** (muda de 2 a 4 m). A média nacional calculada pela própria planilha é R$ 128,15, então o valor do site de terceiros não batia. As composições não incluem adubação, tutor, protetor nem manutenção. A ata da SEMAM, se houver, prevalece.
## 12. Ar × superfície, área ampliada, espécies e plano para a prefeitura (20 a 21/10/2026)

**5.3 INMET A312 × Landsat** (`src/arbormap/analysis/validacao_inmet.py`): o CSV da estação foi extraído dos zips anuais do INMET por leitura parcial (cerca de 2,3 MB baixados em vez de ~750 MB). A LST foi lida no pixel da estação (3 × 3, máscara de nuvem igual à do download) em cada passagem do Landsat 8/9 com nuvem < 20% (≈ 12h59 UTC), contra a temperatura do ar às 13h UTC do mesmo dia. Foram 82 dias, de 2018 a 2024; 2 foram descartados com o mesmo critério de qualidade do dataset (LST abaixo de 15 °C: 12,3 °C em 20/12/2018 e −123 °C em 23/12/2022, nuvem ou pixel corrompido), ficando 80. Correlação 0,58 (p = 1,5 × 10⁻⁸). Na seca (60 dias), a superfície fica 8,2 °C acima do ar (39,3 × 31,1 °C), r = 0,62, e cada 1 °C de superfície veio com 0,19 °C de ar. Na chuvosa a superfície também fica acima do ar (+8,6 °C). **Correção (21/10/2026):** a versão anterior desta nota dizia "na chuvosa, as duas ficam próximas (+0,3 °C)" e r = 0,56; isso era efeito do ponto inválido de −123 °C. Chuva média mensal na estação (meses completos, 2018 a 2024): dezembro a abril com 170 a 372 mm; junho a outubro com 1 a 27 mm.

**Área ampliada (8):** os dados originais vêm de uma caixa de 6 km de raio no Earth Engine, que cobre 89 bairros. A nova grade tem 1056 × 725 px (32 × 22 km) e cobre os 123 bairros. Usa o mesmo produto (Landsat C2 L2) e a mesma receita mensal, lidos do Microsoft Planetary Computer, sem conta (`src/arbormap/data/landsat_pc.py`), com os mesmos 13 meses secos e DEM NASADEM.
- Na caixa antiga, as duas fontes concordam: LST observada r = 0,989 (+0,13 °C), NDVI r = 0,957, sem deslocamento de posição (a grade antiga estava certa).
- O produto de temperatura do Landsat não tem dado no nordeste da área: 11% dos pixels, com as demais bandas presentes. A composição passa a descartar só a LST onde falta só ela (`buraco_so_lst`). As entradas do modelo ficam, e o mapa mostra a previsão.
- A previsão é feita por janelas de 400 px (o tamanho do treino), com passo de 200 e pesos maiores no centro. Cada janela é convertida em °C com a média e o DP da LST observada da própria janela. MAE na área toda: 1,29 °C (r = 0,887; incerteza 0,31 °C). Na caixa antiga: 1,01 °C.
- Bairros: 123 de 123 com métricas (786 mil de 789 mil moradores).
- A anomalia agora é relativa à média da área toda, que inclui bordas rurais e rios; os valores ficam ~1,9 °C maiores que os da caixa antiga.
- Fração plantável por faixas (resultado idêntico ao da versão anterior na caixa antiga): 558 ha de praças e 1.967 ha de faixas de calçada.
- As áreas candidatas e a simulação no navegador também usam janelas de 400 px centradas na área. A conferência JS × Python em dois casos bateu (5,630 × 5,630 °C; 3,878 × 3,878 °C).

**Espécies (3.4):** a lista e o porte vêm do Manual de Arborização Urbana de Fortaleza (Prefeitura de Fortaleza/SEUMA, 2020, tabelas 12 a 16, p. 120 a 123): 102 espécies. O bioma e a ocorrência no Piauí foram conferidos na Flora e Funga do Brasil (JBRJ), via API do GBIF, para as 102 espécies. Ficam 59 árvores e 3 palmeiras nativas da Caatinga ou do Cerrado que ocorrem no PI (22 de pequeno porte, 27 de médio e 10 de grande). Regras do mesmo manual:
- espaçamento de 5, 8 e 12 m (p. 68);
- largura de calçada por porte (p. 67 e 68);
- só pequeno porte sob fiação (p. 68);
- distâncias (p. 69 a 73);
- muda com fuste ≥ 1,80 m e DAP ≥ 3 cm (p. 14);
- rega de 10 a 20 L, 3 vezes por semana, por 3 meses (p. 93).

Machado et al. (2006) continua sem conferência (site da revista em manutenção).

**Plano completo (`docs/plano.html`):** abre em nova aba com mapa do trecho (OSM + anomalia + áreas numeradas), resumo, custos (SINAPI 98510/98511, água da rega), efeito ao longo do tempo, cenários (menor custo, padrão técnico, sombra mais rápida, menor manutenção), espécies por tipo de local, regras, áreas com coordenadas, bairros atendidos, confiabilidade e fontes. O número de árvores passou a vir do espaçamento do manual, e não mais de 36 m² por árvore.
- **Crescimento (premissa, sem fonte local):** copa adulta em 5, 8 e 12 anos (pequeno, médio e grande porte), com efeito proporcional a (t/T)² até a copa adulta. A premissa é editável na página.
- **Leitura do resfriamento (pergunta do Abraão):** ~0,5 °C é a média dos planos com plantio parcial do pixel (calçadas com 10 a 20% do pixel). A conversão total de um pixel construído em vegetação densa dá ~2,3 a 2,5 °C, o que é coerente com o experimento natural (−0,41 °C por +0,1 de NDVI).