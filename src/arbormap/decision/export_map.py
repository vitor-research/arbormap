"""
Exportação georreferenciada (EPSG:4326) dos resultados da simulação para o mapa interativo

Gera, para uma cidade:
  <prefixo>_lst_pred.png / .tif     LST prevista (°C)
  <prefixo>_lst_obs.png / .tif      LST observada (°C), para a vista comparativa
  <prefixo>_mask_inviavel.png / .tif  1 = inviável (água ou declividade > 15°), 0 = viável
  <prefixo>_corredores.geojson      polígonos com prioridade, area_ha, resfriamento_c, lst_media_antes
  bounds.json                       [[sul, oeste], [norte, leste]] para L.imageOverlay, e escalas
Os .tif só são gravados se o pacote rasterio estiver instalado; os .png + bounds.json bastam
para a página.

Georreferência: download_landsat_ee.py exporta o retângulo envolvente de um buffer de
6 km em torno do ponto da cidade (Point.buffer(6000).bounds()), a 30 m. As bordas exatas
podem ser obtidas do Earth Engine com exact_bounds_ee(); sem ele, usa-se a aproximação
geodésica de approx_bounds(), com erro da ordem de um pixel.
"""

import json
import math
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Coordenadas (lon, lat) do dicionário CIDADES de download_landsat_ee.py
CIDADES_LONLAT = {"Teresina": (-42.790, -5.080)}
BUFFER_M = 6_000


def approx_bounds(lon, lat, buffer_m=BUFFER_M):
    """(oeste, sul, leste, norte) do retângulo envolvente de um círculo de raio buffer_m (WGS84)."""
    phi = math.radians(lat)
    m_per_deg_lat = 111132.92 - 559.82 * math.cos(2 * phi) + 1.175 * math.cos(4 * phi)
    m_per_deg_lon = 111412.84 * math.cos(phi) - 93.5 * math.cos(3 * phi)
    dlat, dlon = buffer_m / m_per_deg_lat, buffer_m / m_per_deg_lon
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def exact_bounds_ee(lon, lat, buffer_m=BUFFER_M):
    """Mesmo retângulo usado no download, calculado pelo Earth Engine (requer ee autenticado)."""
    import ee
    coords = ee.Geometry.Point([lon, lat]).buffer(buffer_m).bounds().coordinates().getInfo()[0]
    xs, ys = [c[0] for c in coords], [c[1] for c in coords]
    return min(xs), min(ys), max(xs), max(ys)


def crop_bounds(full_bounds, shape_original, y0x0, hw):
    """Bordas do recorte central (CROP×CROP) dentro da cena completa."""
    west, south, east, north = full_bounds
    H, W = shape_original
    dx, dy = (east - west) / W, (north - south) / H
    y0, x0 = y0x0
    h, w = hw
    return west + x0 * dx, north - (y0 + h) * dy, west + (x0 + w) * dx, north - y0 * dy


def _write_png_rgba(path, rgba):
    # matplotlib em vez de cv2.imwrite, que falha sem aviso em caminhos com acento no Windows
    plt.imsave(str(path), np.clip(rgba, 0, 1))


def _colorize(values, valid, vmin, vmax, cmap="inferno"):
    norm = np.clip((values - vmin) / (vmax - vmin + 1e-9), 0, 1)
    rgba = matplotlib.colormaps[cmap](np.nan_to_num(norm))
    rgba[..., 3] = np.where(valid & np.isfinite(values), 1.0, 0.0)
    return rgba


def _hatch(mask, spacing=4):
    yy, xx = np.indices(mask.shape)
    line = ((xx + yy) % spacing) == 0
    rgba = np.zeros((*mask.shape, 4))
    rgba[mask] = [0.25, 0.25, 0.25, 0.35]
    rgba[mask & line] = [0.1, 0.1, 0.1, 0.85]
    return rgba


def _write_tif(path, array, bounds, nodata=-9999.0):
    try:
        import rasterio
        from rasterio.transform import from_bounds
    except ImportError:
        return False
    h, w = array.shape
    data = np.where(np.isfinite(array), array, nodata).astype(np.float32)
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=1, dtype="float32",
                       crs="EPSG:4326", transform=from_bounds(*bounds, w, h), nodata=nodata) as dst:
        dst.write(data, 1)
    return True


def _mask_to_polygons(mask, bounds, upscale=4):
    """Contornos externos da máscara, em [lon, lat], seguindo as bordas dos pixels."""
    west, south, east, north = bounds
    h, w = mask.shape
    dx, dy = (east - west) / w, (north - south) / h
    big = np.kron(mask.astype(np.uint8), np.ones((upscale, upscale), dtype=np.uint8))
    contours, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for cnt in contours:
        pts = cnt[:, 0, :].astype(np.float64)
        if len(pts) < 3:
            continue
        # centro do subpixel ampliado -> coordenada contínua em pixels originais
        px = (pts[:, 0] + 0.5) / upscale
        py = (pts[:, 1] + 0.5) / upscale
        ring = [[round(west + x * dx, 7), round(north - y * dy, 7)] for x, y in zip(px, py)]
        ring.append(ring[0])
        polys.append([ring])
    return polys


def export_city(res, out_dir, prefix="teresina", full_bounds=None, legend_percentiles=(1, 99)):
    """Grava os arquivos do mapa a partir do resultado de simulate_urbanheatnet.run_city()."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = res["meta"]
    city = meta["cidade"]
    if full_bounds is None:
        full_bounds = approx_bounds(*CIDADES_LONLAT[city])
        bounds_fonte = "aproximação geodésica do buffer de 6 km (approx_bounds)"
    else:
        bounds_fonte = "Earth Engine: Point.buffer(6000).bounds()"

    hh, ww = meta["recorte_h_w"]
    b = crop_bounds(full_bounds, meta["shape_original"], meta["recorte_y0_x0"], (hh, ww))
    valid = res["valid"][:hh, :ww]
    pred = np.where(valid, res["lst_base"][:hh, :ww], np.nan)
    obs = np.where(valid, res["lst_obs"][:hh, :ww], np.nan)
    obs[obs <= 1.0] = np.nan   # lacunas de nuvem gravadas como 0 no download
    inviavel = res["inviavel"][:hh, :ww]

    vals = np.concatenate([pred[valid], obs[valid & np.isfinite(obs)]])
    vmin, vmax = (float(v) for v in np.percentile(vals, legend_percentiles))

    _write_png_rgba(out_dir / f"{prefix}_lst_pred.png", _colorize(pred, valid, vmin, vmax))
    _write_png_rgba(out_dir / f"{prefix}_lst_obs.png", _colorize(obs, valid, vmin, vmax))
    _write_png_rgba(out_dir / f"{prefix}_mask_inviavel.png", _hatch(inviavel))
    tif = [_write_tif(out_dir / f"{prefix}_lst_pred.tif", pred, b),
           _write_tif(out_dir / f"{prefix}_lst_obs.tif", obs, b),
           _write_tif(out_dir / f"{prefix}_mask_inviavel.tif", np.where(valid, inviavel.astype(np.float32), np.nan), b)]

    features = []
    for c in res["corridors"]:
        polys = _mask_to_polygons(c["mask"][:hh, :ww], b)
        if not polys:
            continue
        geometry = {"type": "Polygon", "coordinates": polys[0]} if len(polys) == 1 else \
                   {"type": "MultiPolygon", "coordinates": polys}
        features.append({"type": "Feature", "geometry": geometry, "properties": {
            "prioridade": c["prioridade"],
            "area_ha": round(c["area_ha"], 2),
            "resfriamento_c": round(c["resfriamento_c"], 3),
            "lst_media_antes": round(c["lst_media_antes"], 2),
            "lst_media_depois": round(c["lst_media_depois"], 2),
            "n_pixels": c["n_px"],
            "score_medio": round(c["score_medio"], 4),
        }})
    features.sort(key=lambda f: f["properties"]["prioridade"])
    (out_dir / f"{prefix}_corredores.geojson").write_text(json.dumps(
        {"type": "FeatureCollection", "features": features}, ensure_ascii=False), encoding="utf-8")

    west, south, east, north = b
    info = {
        "cidade": city,
        "bounds_leaflet": [[south, west], [north, east]],
        "bbox_oeste_sul_leste_norte": [west, south, east, north],
        "fonte_bounds": bounds_fonte,
        "crs": "EPSG:4326",
        "pixel_m": 30,
        "legenda_lst_c": {"min": round(vmin, 2), "max": round(vmax, 2), "cmap": "inferno",
                          "percentis": list(legend_percentiles)},
        "lst_media_cidade_prevista_c": round(float(np.nanmean(pred)), 3),
        "lst_media_cidade_observada_c": round(float(np.nanmean(obs)), 3),
        "arquivos_tif_gravados": all(tif),
        "cena": {k: meta[k] for k in ("arquivo_base", "arquivo_alvo", "par_indice", "n_pares")},
        "curva_orcamento": res["curve"],
        "hotspot_mae_ref_c": meta["hotspot_mae_ref_c"],
    }
    (out_dir / "bounds.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return info
