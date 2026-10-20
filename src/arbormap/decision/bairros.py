"""
Métricas por bairro para o mapa interativo (calor, vegetação, incerteza e população vulnerável)

Fontes: malha de bairros e de setores do Censo 2022 (IBGE, com atributos) e agregados de
demografia por setor (IBGE). Vulnerável = crianças de 0 a 4 anos (V01031) + pessoas de 60 anos
ou mais (V01040 + V01041). Renda por setor ainda não foi divulgada para 2022.

O recorte de satélite (12 × 12 km) não cobre a cidade inteira: cada bairro recebe a fração
da sua área dentro do recorte, e bairros pouco cobertos não recebem métricas de calor.
"""

import json
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio import features
from rasterio.transform import from_bounds

from arbormap.decision.export_map import CIDADES_LONLAT, approx_bounds, crop_bounds

PIXEL_KM2 = 0.03 * 0.03
COBERTURA_MIN = 0.3   # abaixo disso, o bairro aparece sem métricas de calor


def grid_bounds(meta, city):
    if meta.get("bounds"):          # grade com georreferenciamento exato (landsat_pc)
        return tuple(meta["bounds"])
    full = approx_bounds(*CIDADES_LONLAT[city])
    return crop_bounds(full, meta["shape_original"], meta["recorte_y0_x0"], meta["recorte_h_w"])


def load_bairros(gpkg, cd_mun):
    g = gpd.read_file(gpkg)
    g = g[g["CD_MUN"].astype(str) == str(cd_mun)].to_crs(4326).reset_index(drop=True)
    return g


def vulnerable_population(setores_gpkg, demografia_zip, cd_mun):
    """População vulnerável por bairro, agregando os setores."""
    st = gpd.read_file(setores_gpkg, columns=["CD_SETOR", "CD_MUN", "CD_BAIRRO"], ignore_geometry=True)
    st = st[st["CD_MUN"].astype(str) == str(cd_mun)]
    with zipfile.ZipFile(demografia_zip) as z:
        name = [n for n in z.namelist() if n.endswith(".csv")][0]
        cols = ["CD_SETOR", "V01006", "V01031", "V01040", "V01041"]
        df = pd.read_csv(z.open(name), sep=";", usecols=lambda c: c.upper() in cols, dtype=str,
                         encoding="latin-1", na_values=["X", "."])
    df.columns = [c.upper() for c in df.columns]
    digits = lambda s: s.astype(str).str.replace(r"\D", "", regex=True)
    df["CD_SETOR"] = digits(df["CD_SETOR"])
    st = st.assign(CD_SETOR=digits(st["CD_SETOR"]), CD_BAIRRO=st["CD_BAIRRO"].astype(str))
    df = df[df["CD_SETOR"].isin(st["CD_SETOR"])]
    df = df.merge(st[["CD_SETOR", "CD_BAIRRO"]], on="CD_SETOR")
    for c in ["V01006", "V01031", "V01040", "V01041"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    g = df.groupby("CD_BAIRRO")[["V01006", "V01031", "V01040", "V01041"]].sum()
    return pd.DataFrame({"moradores": g["V01006"], "criancas_0_4": g["V01031"],
                         "idosos_60mais": g["V01040"] + g["V01041"]}).reset_index()


def bairro_stats(layers, bairros, bounds):
    """Rasteriza os bairros na grade do mapa e calcula as métricas de cada um."""
    h, w = layers["lst_pred"].shape
    tr = from_bounds(*bounds, w, h)
    ids = features.rasterize(((geom, i + 1) for i, geom in enumerate(bairros.geometry)),
                             out_shape=(h, w), transform=tr, fill=0, all_touched=False, dtype="int32")
    pred, anom, inc = layers["lst_pred"], layers["anomalia"], layers["incerteza"]
    ndvi, ndwi = layers["ndvi"], layers["ndwi"]
    ok = np.isfinite(pred) & (layers["artefato_fixo"] < 0.5)
    p90 = float(np.nanpercentile(pred[ok], 90))
    rows = []
    for i, r in bairros.iterrows():
        m = (ids == i + 1)
        n_in = int(m.sum())
        cobertura = min(1.0, n_in * PIXEL_KM2 / float(r["AREA_KM2"])) if r["AREA_KM2"] else 0.0
        mm = m & ok
        row = {"cd_bairro": str(r["CD_BAIRRO"]), "bairro": r["NM_BAIRRO"], "regiao": r["NM_SUBDIST"],
               "area_km2": round(float(r["AREA_KM2"]), 3), "cobertura": round(cobertura, 3),
               "populacao": int(r["v0001"]) if pd.notna(r["v0001"]) else None}
        if cobertura >= COBERTURA_MIN and mm.sum() > 0:
            land = mm & (ndwi < 0.1)
            row.update({
                "lst_media_c": round(float(np.nanmean(pred[mm])), 2),
                "anomalia_c": round(float(np.nanmean(anom[mm])), 2),
                "hotspot_pct": round(100 * float((pred[mm] >= p90).mean()), 1),
                "ndvi_medio": round(float(np.nanmean(ndvi[mm])), 3),
                "vegetacao_densa_pct": round(100 * float((ndvi[land] >= 0.5).mean()), 1) if land.any() else 0.0,
                "incerteza_c": round(float(np.nanmean(inc[mm])), 2),
            })
        rows.append(row)
    return pd.DataFrame(rows), ids, p90


def build(layers, meta, city, cd_mun, bairros_gpkg, setores_gpkg, demografia_zip, out_dir, simplify_m=8):
    bounds = grid_bounds(meta, city)
    bairros = load_bairros(bairros_gpkg, cd_mun)
    stats, ids, p90 = bairro_stats(layers, bairros, bounds)
    vul = vulnerable_population(setores_gpkg, demografia_zip, cd_mun)
    stats = stats.merge(vul, how="left", left_on="cd_bairro", right_on="CD_BAIRRO").drop(columns="CD_BAIRRO")
    stats["vulneraveis"] = (stats["criancas_0_4"].fillna(0) + stats["idosos_60mais"].fillna(0)).astype(int)
    stats["vulneraveis_pct"] = (100 * stats["vulneraveis"] / stats["moradores"].replace(0, np.nan)).round(1)
    stats["densidade_hab_km2"] = (stats["populacao"] / stats["area_km2"].replace(0, np.nan)).round(0)
    gdf = bairros[["CD_BAIRRO", "geometry"]].rename(columns={"CD_BAIRRO": "cd_bairro"})
    gdf["cd_bairro"] = gdf["cd_bairro"].astype(str)
    gdf = gdf.merge(stats, on="cd_bairro")
    gdf["geometry"] = gdf.to_crs(31983).simplify(simplify_m).to_crs(4326)   # SIRGAS 2000 / UTM 23S
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_dir / f"{city.lower()}_bairros.geojson", driver="GeoJSON")
    np.save(out_dir / f"{city.lower()}_bairro_ids.npy", ids)
    info = {"lst_p90_c": round(p90, 2), "n_bairros": len(gdf),
            "n_com_metricas": int(gdf["lst_media_c"].notna().sum()),
            "cobertura_minima": COBERTURA_MIN, "bounds": list(bounds),
            "fontes": ["IBGE, Censo 2022: malha de bairros e setores com atributos",
                       "IBGE, Censo 2022: agregados por setor, demografia (V01031, V01040, V01041)"]}
    (out_dir / f"{city.lower()}_bairros_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return gdf, info
