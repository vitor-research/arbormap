"""
Fração plantável por pixel de 30 m e tipo de área (praça/parque, calçada/canteiro, outros livres)

Rasteriza em subcélulas de ~3 m (fator 10) o que não permite plantio — prédios (Overture Maps:
Google Open Buildings, OpenStreetMap, Microsoft), pista das vias (OpenStreetMap, largura
estimada pelo tipo) e água (OpenStreetMap) — e classifica o restante:
  praça/parque     : dentro de áreas verdes do OpenStreetMap (leisure=park/garden/..., landuse=grass/...)
  calçada/canteiro : faixa de FAIXA_CALCADA_M ao lado da pista das vias urbanas
  outros livres    : o resto (lotes, terrenos; em geral privados ou a verificar)
A fração de cada classe por pixel de 30 m é a média das subcélulas.

Larguras de pista são estimativas por tipo de via (premissas declaradas na página).
"""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
from rasterio import features
from rasterio.transform import from_bounds
from shapely import wkb
from shapely.geometry import LineString, Point, Polygon, shape

# Largura total da pista (m) por tipo de via do OpenStreetMap
LARGURA_PISTA_M = {"motorway": 20, "trunk": 18, "primary": 14, "secondary": 12, "tertiary": 10,
                   "unclassified": 7, "residential": 7, "living_street": 6, "service": 5, "track": 4,
                   "motorway_link": 8, "trunk_link": 8, "primary_link": 8, "secondary_link": 7, "tertiary_link": 7}
VIAS_COM_CALCADA = {"primary", "secondary", "tertiary", "unclassified", "residential", "living_street",
                    "trunk", "primary_link", "secondary_link", "tertiary_link"}
FAIXA_CALCADA_M = 2.5
FATOR = 10
UTM = 31983   # SIRGAS 2000 / UTM zona 23S


def _ways(osm_json, keep=None):
    els = json.loads(Path(osm_json).read_text(encoding="utf-8"))["elements"]
    feats = []
    for el in els:
        tags = el.get("tags", {})
        if keep and not keep(tags):
            continue
        if el["type"] == "way" and "geometry" in el:
            pts = [(p["lon"], p["lat"]) for p in el["geometry"]]
            closed = len(pts) >= 4 and pts[0] == pts[-1]
            geom = Polygon(pts) if closed and tags.get("highway") is None and tags.get("waterway") is None else LineString(pts)
            feats.append({"geometry": geom, **{k: tags.get(k) for k in ("highway", "leisure", "landuse", "name", "natural", "waterway", "place", "area")}})
        elif el["type"] == "relation" and "members" in el:
            for mbr in el["members"]:
                if mbr.get("role") == "outer" and "geometry" in mbr:
                    pts = [(p["lon"], p["lat"]) for p in mbr["geometry"]]
                    if len(pts) >= 4 and pts[0] == pts[-1]:
                        feats.append({"geometry": Polygon(pts), "leisure": tags.get("leisure"), "landuse": tags.get("landuse"),
                                      "name": tags.get("name"), "natural": tags.get("natural")})
    return gpd.GeoDataFrame(feats, crs=4326) if feats else gpd.GeoDataFrame(geometry=[], crs=4326)


def load_layers(osm_dir, buildings_parquet, bounds):
    w, s, e, n = bounds
    import duckdb
    b = duckdb.query(f"SELECT * FROM read_parquet('{Path(buildings_parquet).as_posix()}')").df()
    b = gpd.GeoDataFrame(b.drop(columns="geometry"), geometry=[wkb.loads(bytes(g)) for g in b["geometry"]], crs=4326)
    b = b.cx[w:e, s:n]
    vias = _ways(Path(osm_dir) / "vias.json", keep=lambda t: t.get("highway") in LARGURA_PISTA_M)
    verdes = _ways(Path(osm_dir) / "areas_verdes.json")
    verdes = verdes[verdes.geometry.geom_type == "Polygon"]
    agua = _ways(Path(osm_dir) / "agua.json")
    return b, vias, verdes, agua


def _buffer_m(gdf, dist):
    return gdf.to_crs(UTM).buffer(dist).to_crs(4326)


def plantable_fractions(bounds, shape_hw, buildings, vias, verdes, agua):
    """Frações (0–1) por pixel de 30 m: livre total e por classe; também a fração construída."""
    h, w = shape_hw
    H, W = h * FATOR, w * FATOR
    tr = from_bounds(*bounds, W, H)
    burn = lambda geoms: features.rasterize(((g, 1) for g in geoms if g is not None and not g.is_empty),
                                            out_shape=(H, W), transform=tr, fill=0, dtype="uint8").astype(bool)
    larg = vias["highway"].map(LARGURA_PISTA_M).fillna(6).to_numpy()
    pista = burn(vias.to_crs(UTM).buffer(larg / 2, cap_style=2).to_crs(4326))
    com_calc = vias[vias["highway"].isin(VIAS_COM_CALCADA)]
    larg_c = com_calc["highway"].map(LARGURA_PISTA_M).to_numpy()
    faixa = burn(com_calc.to_crs(UTM).buffer(larg_c / 2 + FAIXA_CALCADA_M, cap_style=2).to_crs(4326))
    predio = burn(buildings.geometry)
    agua_poly = agua[agua.geometry.geom_type == "Polygon"].geometry
    agua_lin = agua[agua.geometry.geom_type == "LineString"]
    rios = _buffer_m(agua_lin, 10) if len(agua_lin) else []
    agua_m = burn(list(agua_poly) + list(rios))
    verde = burn(verdes.geometry)

    ocupado = predio | pista | agua_m
    livre = ~ocupado
    praca = livre & verde
    calcada = livre & faixa & ~verde
    outros = livre & ~verde & ~faixa
    block = lambda a: a.reshape(h, FATOR, w, FATOR).mean(axis=(1, 3)).astype(np.float32)
    return {"livre": block(livre), "praca": block(praca), "calcada": block(calcada), "outros": block(outros),
            "construido": block(predio), "pista": block(pista)}
