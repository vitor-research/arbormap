"""
Exporta os dados do mapa interativo (docs/data/) a partir das camadas derivadas

Saídas (todas leves, lidas pela página estática):
  {cidade}_anomalia.png      anomalia de LST prevista (°C em relação à média da cidade), divergente
  {cidade}_incerteza.png     desvio-padrão entre os 5 modelos (°C)
  {cidade}_plantavel.png     fração livre para plantio por pixel de 30 m
  {cidade}_inviavel.png      água, declividade > 15° e lacunas fixas do dado (hachurado)
  {cidade}_bairros.geojson   bairros com métricas
  {cidade}_areas.geojson     áreas candidatas com o efeito individual
  {cidade}_equipamentos.geojson
  meta.json                  bounds, legendas, métricas do modelo, verificação, premissas e fontes
"""

import json
import shutil
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from arbormap.decision.bairros import grid_bounds
from arbormap.decision.simulate_urbanheatnet import slope_degrees

PREMISSAS = {
    "largura_pista_m": "por tipo de via do OpenStreetMap (residencial 7 m, terciária 10 m, secundária 12 m, primária 14 m)",
    "faixa_calcada_m": 2.5,
    "ndvi_ja_vegetado": 0.3,
    "agua_ndwi_min": 0.1,
    "declividade_max_graus": 15,
    "resolucao_m": 30,
}


# Referências de preço (SINAPI, Teresina) enquanto não houver a ata de registro de preços da SEMAM.
# A página só as usa se o gestor clicar; os campos de custo começam vazios.
# Conferência da 98510: 0,2117 h × 24,37 (jardineiro) + 1,0584 h × 23,66 (servente) + 85,05 (muda 358) = 115,25.
CUSTOS_REFERENCIA = {
    "fonte": "SINAPI (Caixa), planilha SINAPI_Referência_2026_09.xlsx, aba CSD, coluna PI (Teresina)",
    "base": "09/2026, sem desoneração, emitida em 09/10/2026",
    "observacao": "As composições não incluem adubação, tutor, protetor nem manutenção.",
    "opcoes": [
        {"codigo": "98510", "valor": 115.24,
         "descricao": "plantio de árvore ornamental com muda de até 2,00 m (muda de 1 m: R$ 85,05; jardineiro 0,2117 h; servente 1,0584 h)"},
        {"codigo": "98511", "valor": 209.23,
         "descricao": "plantio de árvore ornamental com muda de 2,00 a 4,00 m (mais usual na arborização de ruas)"},
    ],
}


def _rgba(values, valid, cmap, vmin, vmax):
    norm = np.clip((values - vmin) / (vmax - vmin + 1e-9), 0, 1)
    rgba = matplotlib.colormaps[cmap](np.nan_to_num(norm))
    rgba[..., 3] = np.where(valid & np.isfinite(values), 1.0, 0.0)
    return rgba


def _hatch(mask, spacing=4):
    yy, xx = np.indices(mask.shape)
    rgba = np.zeros((*mask.shape, 4))
    rgba[mask] = [0.2, 0.2, 0.2, 0.35]
    rgba[mask & (((xx + yy) % spacing) == 0)] = [0.05, 0.05, 0.05, 0.85]
    return rgba


def _save(path, rgba, upscale=2):
    if upscale > 1:
        rgba = np.kron(rgba, np.ones((upscale, upscale, 1)))
    plt.imsave(str(path), np.clip(rgba, 0, 1))


def exportar(derivados, drive_final, out_dir, city="Teresina"):
    D, out = Path(derivados), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = city.lower()
    L = dict(np.load(D / f"{p}_camadas.npz"))
    meta = json.loads((D / f"{p}_camadas.json").read_text(encoding="utf-8"))
    fr = dict(np.load(D / f"{p}_plantavel.npz"))
    bounds = grid_bounds(meta, city)
    w, s, e, n = bounds

    art = L["artefato_fixo"] > 0.5
    valid = np.isfinite(L["lst_pred"]) & ~art
    anom_lim = float(np.ceil(np.nanpercentile(np.abs(L["anomalia"][valid]), 99)))
    _save(out / f"{p}_anomalia.png", _rgba(L["anomalia"], valid, "RdYlBu_r", -anom_lim, anom_lim))
    inc_max = round(float(np.nanpercentile(L["incerteza"][valid], 99)), 1)
    _save(out / f"{p}_incerteza.png", _rgba(L["incerteza"], valid, "magma_r", 0, inc_max))
    _save(out / f"{p}_plantavel.png", _rgba(fr["livre"], valid & (L["ndvi"] < 0.3), "YlGn", 0, 1))
    dem = np.where(np.isfinite(L["dem"]), L["dem"], np.nanmedian(L["dem"]))
    inviavel = (L["ndwi"] >= PREMISSAS["agua_ndwi_min"]) | (slope_degrees(dem) > PREMISSAS["declividade_max_graus"]) | art
    _save(out / f"{p}_inviavel.png", _hatch(inviavel), upscale=2)

    for nome in ("bairros", "areas", "equipamentos"):
        src = D / f"{p}_{nome}.geojson"
        if src.exists():
            shutil.copy(src, out / f"{p}_{nome}.geojson")

    F = Path(drive_final)
    kfold = json.loads((F / "resultados_kfold.json").read_text(encoding="utf-8"))
    verif = json.loads((F / "verificacao" / "verificacao.json").read_text(encoding="utf-8"))
    binfo = json.loads((D / f"{p}_bairros_info.json").read_text(encoding="utf-8"))
    ainfo_p = D / f"{p}_areas_info.json"
    ainfo = json.loads(ainfo_p.read_text(encoding="utf-8")) if ainfo_p.exists() else {}
    vreal_p = D / f"{p}_validacao_real.json"
    vreal = json.loads(vreal_p.read_text(encoding="utf-8")) if vreal_p.exists() else None
    vinmet_p = D / f"{p}_validacao_inmet.json"
    vinmet = json.loads(vinmet_p.read_text(encoding="utf-8")) if vinmet_p.exists() else None
    if (D / "especies.json").exists():
        shutil.copy(D / "especies.json", out / "especies.json")
    m = {
        "cidade": city,
        "bounds_leaflet": [[s, w], [n, e]],
        "camadas": {
            "anomalia": {"arquivo": f"{p}_anomalia.png", "min": -anom_lim, "max": anom_lim, "cmap": "RdYlBu_r", "unidade": "°C"},
            "incerteza": {"arquivo": f"{p}_incerteza.png", "min": 0, "max": inc_max, "cmap": "magma_r", "unidade": "°C"},
            "plantavel": {"arquivo": f"{p}_plantavel.png", "min": 0, "max": 1, "cmap": "YlGn", "unidade": "fração"},
            "inviavel": {"arquivo": f"{p}_inviavel.png"},
        },
        "lst_media_cidade_c": round(meta["lst_media_composicao_c"], 2),
        "composicao": {"cenas": [c["arquivo"] for c in meta["composicao"]["usadas"]], "meses": meta["composicao"]["meses"],
                       "mae_c": round(meta["mae_composicao_c"], 2), "pearson": round(meta["pearson_composicao"], 3),
                       "incerteza_media_c": round(meta["incerteza_media_c"], 2), "modelo": meta["modelo_principal"],
                       "fold": meta["fold_principal"], "val_cities": meta["val_cities_fold_principal"]},
        "modelo": {"versao": "v1.0.0", "folds": [{k: f[k] for k in ("fold", "val_cities")} | {
                       "mae_c": round(f["best"]["mae_c"], 2), "hotspot_mae_c": round(f["best"]["hotspot_mae_c"], 2),
                       "iou": round(f["best"]["iou"], 3)} for f in kfold["folds"]],
                   "resumo": {k: {"media": round(v["media"], 3), "dp": round(v["dp_amostral"], 3)} for k, v in kfold["resumo"].items()}},
        "verificacao": verif["resumo"],
        "bairros": binfo,
        "areas": ainfo,
        "validacao_real": vreal,
        "validacao_inmet": vinmet,
        "custos_referencia": CUSTOS_REFERENCIA,
        "premissas": PREMISSAS,
        "fontes": [
            "Landsat 8/9 Collection 2 Nível 2 (USGS): LST, NDVI, NDBI, NDWI. Treino do modelo com dados via Google Earth Engine (DEM SRTM); "
            "mapa de Teresina ampliado com os mesmos produtos via Microsoft Planetary Computer (DEM NASADEM); na área em comum, "
            "LST observada com correlação de 0,989 e NDVI com 0,957 entre as duas fontes",
            "Modelo: UrbanHeatNet (ARBORMAP v1.0.0), validação cruzada por cidade",
            "IBGE, Censo 2022: bairros, setores e demografia",
            "Overture Maps (Google Open Buildings, OpenStreetMap, Microsoft): contornos de prédios",
            "OpenStreetMap: vias, praças e parques, escolas, creches e unidades de saúde",
        ],
    }
    (out / "meta.json").write_text(json.dumps(m, indent=1, ensure_ascii=False), encoding="utf-8")
    return m
