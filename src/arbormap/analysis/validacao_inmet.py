"""
5.3 Temperatura da superfície (Landsat) × temperatura do ar (INMET A312, Teresina)

Para cada passagem do Landsat 8/9 (≈ 12h59 UTC, 9h59 no horário local) com cobertura de nuvens
da cena < 20% (o mesmo filtro do download original), lê a LST no pixel da estação (média 3 × 3,
com a máscara de nuvem do QA_PIXEL igual à do download) e a média da caixa de 12 km usada pelo
modelo; compara com a temperatura do ar registrada pela estação às 13h UTC do mesmo dia.

Dados: Landsat Collection 2 Nível 2 (USGS), lido do Microsoft Planetary Computer (acesso aberto);
INMET, dados históricos anuais (portal.inmet.gov.br/dadoshistoricos), estação A312.
"""

import glob
import io
from pathlib import Path

import numpy as np
import pandas as pd

ESTACAO = {"codigo": "A312", "lat": -5.03472221, "lon": -42.80138888}
CENTRO_CAIXA = (-42.790, -5.080)
RAIO_CAIXA_M = 6000


def ler_inmet(pasta):
    partes = []
    for f in sorted(glob.glob(str(Path(pasta) / "INMET_*A312*.CSV"))):
        txt = Path(f).read_bytes().decode("latin-1")
        linhas = txt.splitlines()
        i = next(k for k, l in enumerate(linhas) if l.startswith("Data;") or l.startswith("DATA (YYYY-MM-DD);"))
        df = pd.read_csv(io.StringIO("\n".join(linhas[i:])), sep=";", decimal=",", dtype=str)
        col_t = next(c for c in df.columns if c.startswith("TEMPERATURA DO AR - BULBO SECO"))
        col_d, col_h = df.columns[0], df.columns[1]
        out = pd.DataFrame({
            "data": pd.to_datetime(df[col_d].str.replace("-", "/"), format="%Y/%m/%d"),
            "hora_utc": df[col_h].str.extract(r"(\d{2})")[0].astype(int),
            "tar": pd.to_numeric(df[col_t].str.replace(",", "."), errors="coerce"),
        })
        partes.append(out)
    d = pd.concat(partes, ignore_index=True)
    d.loc[d["tar"] <= -9000, "tar"] = np.nan
    return d


def _mascara_nuvem(qa):
    return ((qa & (1 << 1)) == 0) & ((qa & (1 << 3)) == 0) & ((qa & (1 << 4)) == 0) & (qa != 1)


def ler_cena(item, lon, lat, meia_janela=1, raio_caixa_m=RAIO_CAIXA_M, centro=CENTRO_CAIXA):
    """LST (°C) no pixel da estação (janela (2k+1)²) e média na caixa do modelo."""
    import rasterio
    from rasterio.warp import transform
    from rasterio.windows import Window
    import planetary_computer as pc

    it = pc.sign(item)
    with rasterio.open(it.assets["lwir11"].href) as st, rasterio.open(it.assets["qa_pixel"].href) as qa:
        def janela(x, y, k):
            xs, ys = transform("EPSG:4326", st.crs, [x], [y])
            r, c = st.index(xs[0], ys[0])
            return Window(c - k, r - k, 2 * k + 1, 2 * k + 1)

        def lst(win):
            dn = st.read(1, window=win, boundless=True, fill_value=0).astype(np.float32)
            q = qa.read(1, window=win, boundless=True, fill_value=1)
            ok = _mascara_nuvem(q) & (dn > 0)
            return np.where(ok, dn * 0.00341802 + 149.0 - 273.15, np.nan)

        p = lst(janela(lon, lat, meia_janela))
        k = int(raio_caixa_m / 30)
        cx = lst(janela(*centro, k))
    return {"lst_estacao": float(np.nanmean(p)) if np.isfinite(p).sum() >= 5 else np.nan,
            "lst_caixa": float(np.nanmean(cx)) if np.isfinite(cx).mean() > 0.7 else np.nan,
            "frac_limpa_caixa": float(np.isfinite(cx).mean())}


def buscar_cenas(lon, lat, inicio="2018-01-01", fim="2024-12-31", nuvem_max=20):
    from pystac_client import Client
    cat = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    s = cat.search(collections=["landsat-c2-l2"], intersects={"type": "Point", "coordinates": [lon, lat]},
                   datetime=f"{inicio}/{fim}",
                   query={"eo:cloud_cover": {"lt": nuvem_max}, "platform": {"in": ["landsat-8", "landsat-9"]}})
    return list(s.items())


def comparar(pasta_inmet, log=print):
    ar = ler_inmet(pasta_inmet)
    ar13 = ar[ar["hora_utc"] == 13].set_index("data")["tar"]
    tmax = ar.groupby("data")["tar"].max()
    itens = buscar_cenas(ESTACAO["lon"], ESTACAO["lat"])
    linhas, vistos = [], set()
    for it in sorted(itens, key=lambda i: i.datetime):
        dia = pd.Timestamp(it.datetime.date())
        if dia in vistos:
            continue
        try:
            r = ler_cena(it, ESTACAO["lon"], ESTACAO["lat"])
        except Exception as e:  # cena sem a estação dentro, ou leitura falhou
            log(f"   {it.id}: {e}")
            continue
        if not np.isfinite(r["lst_estacao"]):
            continue
        vistos.add(dia)
        linhas.append({"data": dia, "cena": it.id, **r, "tar_13utc": ar13.get(dia, np.nan), "tar_max_dia": tmax.get(dia, np.nan)})
    d = pd.DataFrame(linhas).dropna(subset=["tar_13utc"])
    d["seca"] = d["data"].dt.month.isin([6, 7, 8, 9, 10])
    a, b = np.polyfit(d["tar_13utc"], d["lst_estacao"], 1)
    res = {
        "n_dias": int(len(d)), "periodo": [str(d["data"].min().date()), str(d["data"].max().date())],
        "pearson_lst_estacao_x_ar": round(float(d["lst_estacao"].corr(d["tar_13utc"])), 3),
        "pearson_lst_caixa_x_ar": round(float(d["lst_caixa"].corr(d["tar_13utc"])), 3),
        "pearson_lst_estacao_x_tmax": round(float(d["lst_estacao"].corr(d["tar_max_dia"])), 3),
        "inclinacao_lst_por_grau_ar": round(float(a), 2),
        "lst_menos_ar_media_c": round(float((d["lst_estacao"] - d["tar_13utc"]).mean()), 2),
        "lst_menos_ar_seca_c": round(float((d["lst_estacao"] - d["tar_13utc"])[d["seca"]].mean()), 2),
        "lst_menos_ar_chuvosa_c": round(float((d["lst_estacao"] - d["tar_13utc"])[~d["seca"]].mean()), 2),
        "media_seca": {"lst": round(float(d.loc[d["seca"], "lst_estacao"].mean()), 1), "ar": round(float(d.loc[d["seca"], "tar_13utc"].mean()), 1)},
        "media_chuvosa": {"lst": round(float(d.loc[~d["seca"], "lst_estacao"].mean()), 1), "ar": round(float(d.loc[~d["seca"], "tar_13utc"].mean()), 1)},
        "pontos": [{"data": str(r.data.date()), "lst": round(r.lst_estacao, 2), "ar": round(r.tar_13utc, 1), "seca": bool(r.seca)} for r in d.itertuples()],
        "estacao": ESTACAO,
    }
    return res, d
