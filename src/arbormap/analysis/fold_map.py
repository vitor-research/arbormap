"""
Mapa dos municípios por fold (Figura 1)

Origem: notebook arbormap.ipynb, célula 20 (Google Colab).
Usa malha IBGE via geopandas.
Código preservado como no notebook; caminhos do Colab/Drive devem ser ajustados nas CONFIGURAÇÕES.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import geopandas as gpd
import requests
import io

# 1. Mapeamento e Numeração
folds_ibge = {
    # Fold 1: IDs 01 a 04 (Iguatu - CE, Parnaíba - PI, Pedro II - PI, Juazeiro do Norte - CE)
    1: ['2305506', '2207702', '2207900', '2307304'],

    # Fold 2: IDs 05 a 08 (Floriano - PI, Teresina - PI, São Raimundo Nonato - PI, Picos - PI)
    2: ['2203909', '2211001', '2210607', '2208007'],

    # Fold 3: IDs 09 a 12 (Barreiras - BA, Imperatriz - MA, Patos - PB, Crato - CE)
    3: ['2903201', '2105302', '2510808', '2304202'],

    # Fold 4: IDs 13 a 16 (Mossoró - RN, Bom Jesus - PI, Uruçuí - PI, Araguaína - TO)
    4: ['2402001', '2201903', '2211100', '1702109'],

    # Fold 5: IDs 17 a 20 (Sobral - CE, Corrente - PI, Vitória da Conquista - BA, Paulo Afonso - BA)
    5: ['2312908', '2202901', '2933307', '2924009']
}

# Criando lista plana de todos os códigos e o dicionário de números
all_codes = [cod for sublist in folds_ibge.values() for cod in sublist]
mapa_numero_cidade = {cod: i+1 for i, cod in enumerate(all_codes)}

# 2. Downloads (IBGE)
headers = {'User-Agent': 'Mozilla/5.0'}
url_estados = "https://servicodados.ibge.gov.br/api/v3/malhas/paises/BR?formato=application/vnd.geo+json&intrarregiao=UF"
url_mun = "https://servicodados.ibge.gov.br/api/v3/malhas/paises/BR?formato=application/vnd.geo+json&intrarregiao=municipio"

br_estados = gpd.read_file(io.StringIO(requests.get(url_estados, headers=headers).text))
br_mun = gpd.read_file(io.StringIO(requests.get(url_mun, headers=headers).text))

# 3. Plotagem
fig, ax = plt.subplots(figsize=(12, 12))
br_estados.plot(ax=ax, color='#f0f0f0', edgecolor='#cccccc', linewidth=0.5)

colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

for fold_num, codigos in folds_ibge.items():
    mun_fold = br_mun[br_mun['codarea'].isin(codigos)]
    mun_fold.plot(ax=ax, color=colors[fold_num-1], edgecolor='black', linewidth=0.5, zorder=4)

    # Adicionando os números (labels)
    for _, row in mun_fold.iterrows():
        centroid = row.geometry.centroid
        ax.text(centroid.x, centroid.y, str(mapa_numero_cidade[row['codarea']]),
          fontsize=8, ha='center', va='center', color='white',
          fontweight='bold', zorder=10,
          path_effects=[pe.withStroke(linewidth=2, foreground='black')])

# Ajustes Finais
ax.set_xlim(-49, -34); ax.set_ylim(-16, -1)
legend_patches = [mpatches.Patch(color=colors[i-1], label=f'Fold {i}') for i in folds_ibge.keys()]
plt.legend(handles=legend_patches, title="Validação Cruzada", loc='lower right')
plt.title("Distribuição Espacial da Validação Cruzada (Índice Numérico)")
plt.show()
