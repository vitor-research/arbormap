"""
Composições mensais Landsat 8/9 para uma área maior, sem Google Earth Engine

Reproduz a receita de download_landsat_ee.py (mesmo produto: Landsat Collection 2 Nível 2 da USGS):
  - cenas com cobertura de nuvens < 20%, Landsat 8 e 9;
  - máscara do QA_PIXEL: nuvem dilatada (bit 1), nuvem (bit 3) e sombra (bit 4);
  - reflectância: mediana do mês; NDVI, NDWI e NDBI com as mesmas fórmulas;
  - LST: média do mês de ST_B10 (× 0,00341802 + 149 − 273,15);
  - DEM: NASADEM (reprocessamento do SRTM; o original usava SRTMGL1_003).
Os dados vêm do Microsoft Planetary Computer (acesso aberto, sem conta). A grade é EPSG:4326
com o mesmo tamanho de pixel da grade original, para a composição caber no modelo sem reamostrar.

Diferenças em relação ao arquivo original do Earth Engine: os NaN (nuvem) ficam como NaN, em vez de
serem preenchidos pela média 5 × 5; e o georreferenciamento é exato (o original foi aproximado).
"""

import numpy as np

BANDAS = {"green": "SR_B3", "red": "SR_B4", "nir08": "SR_B5", "swir16": "SR_B6"}
ORDEM = ("NDVI", "NDWI", "NDBI", "DEM", "LST")   # mesma ordem de canais dos .npy originais


def grade(bounds, dx, dy):
    """bounds (w, s, e, n) ajustados a múltiplos do pixel; devolve transform, (h, w) e bounds."""
    from rasterio.transform import from_origin
    w, s, e, n = bounds
    W = int(np.ceil((e - w) / dx)); H = int(np.ceil((n - s) / dy))
    return from_origin(w, n, dx, dy), (H, W), (w, n - H * dy, w + W * dx, n)


def _ler(href, tr, shape, resampling):
    import rasterio
    from rasterio.vrt import WarpedVRT
    with rasterio.open(href) as src, WarpedVRT(src, crs="EPSG:4326", transform=tr, width=shape[1], height=shape[0],
                                               resampling=resampling, nodata=src.nodata) as v:
        return v.read(1)


def cenas_do_mes(bounds, ano, mes, nuvem_max=20):
    from pystac_client import Client
    import planetary_computer as pc
    fim = f"{ano + 1}-01-01" if mes == 12 else f"{ano}-{mes + 1:02d}-01"
    cat = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    s = cat.search(collections=["landsat-c2-l2"], bbox=list(bounds), datetime=f"{ano}-{mes:02d}-01/{fim}",
                   query={"eo:cloud_cover": {"lt": nuvem_max}, "platform": {"in": ["landsat-8", "landsat-9"]}})
    # o filtro de data do STAC é inclusivo no fim; tira cenas do dia 1 do mês seguinte
    return [i for i in s.items() if (i.datetime.year, i.datetime.month) == (ano, mes)]


def composicao_mensal(bounds, tr, shape, ano, mes, log=print):
    from rasterio.enums import Resampling
    itens = cenas_do_mes(bounds, ano, mes)
    sr = {b: [] for b in BANDAS}
    lst = []
    for it in itens:
        qa = _ler(it.assets["qa_pixel"].href, tr, shape, Resampling.nearest)
        ok = ((qa & 2) == 0) & ((qa & 8) == 0) & ((qa & 16) == 0) & (qa != 1) & (qa != 0)
        if ok.mean() < 0.01:
            continue
        for b in BANDAS:
            dn = _ler(it.assets[b].href, tr, shape, Resampling.nearest).astype(np.float32)
            sr[b].append(np.where(ok & (dn > 0), dn * 0.0000275 - 0.2, np.nan))
        dn = _ler(it.assets["lwir11"].href, tr, shape, Resampling.nearest).astype(np.float32)
        lst.append(np.where(ok & (dn > 0), dn * 0.00341802 + 149.0 - 273.15, np.nan))
        log(f"   {it.id}: {ok.mean():.0%} limpo")
    if not lst:
        return None, itens
    med = {b: np.nanmedian(np.stack(v), 0) for b, v in sr.items()}
    b3, b4, b5, b6 = med["green"], med["red"], med["nir08"], med["swir16"]
    ndvi = (b5 - b4) / (np.abs(b5 + b4) + 1e-6)
    ndwi = (b3 - b5) / (np.abs(b3 + b5) + 1e-6)
    ndbi = (b6 - b5) / (np.abs(b6 + b5) + 1e-6)
    return {"NDVI": ndvi, "NDWI": ndwi, "NDBI": ndbi, "LST": np.nanmean(np.stack(lst), 0)}, itens


def dem(bounds, tr, shape):
    from pystac_client import Client
    import planetary_computer as pc
    from rasterio.enums import Resampling
    cat = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    itens = list(cat.search(collections=["nasadem"], bbox=list(bounds)).items())
    out = np.full(shape, np.nan, np.float32)
    for it in itens:
        a = _ler(it.assets["elevation"].href, tr, shape, Resampling.bilinear).astype(np.float32)
        out = np.where(np.isfinite(out), out, np.where(a > -1000, a, np.nan))
    return out, [i.id for i in itens]


def baixar(bounds, dx, dy, meses, pasta, log=print):
    """Salva {ano}_{mes}.npy (5, H, W) na ordem ORDEM e grade.json com o georreferenciamento."""
    import json
    from pathlib import Path
    P = Path(pasta); P.mkdir(parents=True, exist_ok=True)
    tr, shape, b = grade(bounds, dx, dy)
    elev, dem_ids = dem(b, tr, shape)
    reg = {"bounds": b, "dx": dx, "dy": dy, "shape": list(shape), "dem": dem_ids, "meses": {}}
    for ano, mes in meses:
        f = P / f"{ano}_{mes:02d}.npy"
        log(f"{ano}-{mes:02d}")
        if f.exists():
            continue
        c, itens = composicao_mensal(b, tr, shape, ano, mes, log)
        if c is None:
            log("   sem cena utilizável"); continue
        c["DEM"] = elev
        np.save(f, np.stack([c[k] for k in ORDEM]).astype(np.float32))
        reg["meses"][f.stem] = [i.id for i in itens]
    (P / "grade.json").write_text(json.dumps(reg, indent=1), encoding="utf-8")
    return reg
