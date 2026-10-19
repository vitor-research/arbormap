# Como rodar o ARBORMAP no Google Colab (passo a passo)

Tempo total (medido, 30 épocas por fold, 5 folds): cerca de 10 a 11 horas na GPU T4 (cerca de 250 s por época) ou cerca de 75 minutos na A100 (cerca de 29 s por época). Com Colab Pro, a A100 consome cerca de 5,3 unidades de computação por hora (o treino inteiro, cerca de 7 unidades); escolha-a no passo 3 se estiver disponível. Não precisa ficar olhando; deixe a aba aberta e o computador sem entrar em suspensão. Se a sessão cair, o treino continua de onde parou (passo 7).

## Antes de começar

1. Confira onde está a pasta do dataset no seu Google Drive: `heat_dataset_v3`, com uma subpasta por cidade (20 pastas, por exemplo `Teresina`, `Parnaiba`, `Iguatu_CE`...).
   - O notebook procura em `Meu Drive/heat_dataset_v3`. O script de download salvava em `Meu Drive/IA/Pibic/Vulnerability/heat_dataset_v3`; se for esse o caso, você vai ajustar o caminho no passo 6.
   - Se outra pessoa for rodar com a conta dela: compartilhe a pasta `heat_dataset_v3` com ela; ela abre "Compartilhados comigo", clica com o botão direito na pasta → **Organizar → Adicionar atalho** → **Meu Drive**. Assim o caminho `Meu Drive/heat_dataset_v3` funciona sem mudar nada.
2. Use o Google Chrome, logado na mesma conta Google do Drive.

## Passo a passo

1. Abra este link:
   https://colab.research.google.com/github/abraaolm-maker/arbormap/blob/main/notebooks/refazer_urbanheatnet.ipynb
2. Clique em **Arquivo → Salvar uma cópia no Drive**. A cópia abre numa aba nova; trabalhe nela.
3. Ative a GPU: **Ambiente de execução → Alterar o tipo de ambiente de execução → T4 GPU → Salvar**.
4. Rode a **primeira célula de código** (botão ▶ à esquerda, ou Ctrl+Enter).
   - Vai aparecer um pedido para conectar ao Google Drive: clique em **Conectar ao Google Drive**, escolha sua conta e clique em **Permitir**.
   - No fim, a célula deve mostrar `GPU: Tesla T4`. Se mostrar `NENHUMA`, volte ao passo 3.
5. Se aparecer o aviso "Este notebook não foi criado pelo Google", clique em **Executar assim mesmo**.
6. Na célula **"1. Caminhos"**, confira a linha `DATA_DRIVE = Path('/content/drive/MyDrive/heat_dataset_v3')`. Se o dataset estiver em outra pasta, troque o caminho. Rode a célula.
   - Deve aparecer algo como `1036 arquivos .npy em /content/heat_dataset_v3`. A cópia leva alguns minutos.
7. Rode a célula **"2. Treino nos cinco folds"**. É a parte demorada (cerca de 2 h por fold).
   - A cada época aparece uma linha `Fold N | Epoch NN | ... MAE (°C) ...`.
   - **Se a sessão cair** (acontece no Colab gratuito): reconecte, ative a GPU de novo e rode as células 1, 2 e 3 outra vez. O treino continua de onde parou, porque tudo fica salvo em `Meu Drive/arbormap_urbanheatnet`.
8. Quando aparecer **"TABELA 2 (melhor época por fold)"**, o treino terminou. Rode, em ordem, as células:
   - **"3. Tabela 2 e Figuras 2 e 3"**;
   - **"4. Simulação em Teresina"** (alguns minutos);
   - **"5. Arquivos do mapa interativo"** (no fim, o navegador baixa `arbormap_mapa.zip`).
9. Pronto. Os resultados ficam em `Meu Drive/arbormap_urbanheatnet`:
   - `tabela2.csv`: a Tabela 2 nova do artigo;
   - `figuras/`: Figuras 2 a 6;
   - `teresina_simulacao.json`: os resultados de resfriamento por estratégia;
   - `treino_log.txt`: o registro completo (serve para o diário de bordo);
   - `arbormap_mapa.zip` (na pasta de downloads): os arquivos do mapa interativo.

## Se der erro

- `FileNotFoundError: Pastas de cidade ausentes`: o caminho do passo 6 está errado ou falta alguma cidade. A mensagem lista quais.
- `CUDA out of memory`: Ambiente de execução → Reiniciar sessão, depois rode tudo de novo (o treino continua de onde parou).
- Qualquer outro erro: copie a mensagem inteira (a parte vermelha) e mande para o orientador.
