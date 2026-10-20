"""
Figuras do artigo (versão 4), geradas a partir dos resultados finais (v1.0.0) e das validações

Uso: python -m arbormap.analysis.figuras_artigo_v4 DADOS SAIDA
  DADOS: pasta _dados (drive_final/arbormap_final, derivados, derivados_ampliado, ibge_malhas, inmet)
  SAIDA: pasta das figuras (PNG, 300 dpi)
"""

import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from rasterio.features import rasterize
from rasterio.transform import from_bounds

matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                            "legend.fontsize": 8, "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight"})
COR = {"verde": "#1f7a4d", "azul": "#2f6fae", "vermelho": "#c2412d", "cinza": "#7a7a7a", "laranja": "#d98a1c"}
FOLD_CORES = ["#c2412d", "#2f6fae", "#1f7a4d", "#8e5bb5", "#d98a1c"]
CM = 1 / 2.54
LARG = 17.5 * CM          # largura útil da página do artigo (21,6 cm − 2 × 1,7 cm − folga)


def salvar(fig, out, nome):
    fig.savefig(out / nome)
    plt.close(fig)
    print("  ", nome)


def fig_municipios(D, out):
    M = json.load(open(D / "ibge_malhas" / "municipios.json", encoding="utf-8"))
    est = gpd.read_file(D / "ibge_malhas" / "estados.geojson")
    fig, ax = plt.subplots(figsize=(LARG * 0.62, LARG * 0.62))
    est.boundary.plot(ax=ax, color="#9a9a9a", linewidth=0.5)
    for m in M:
        g = gpd.read_file(D / "ibge_malhas" / f"mun_{m['codigo']}.geojson")
        g.plot(ax=ax, color=FOLD_CORES[m["fold"] - 1], edgecolor="black", linewidth=0.3)
        c = g.geometry.iloc[0].representative_point()
        dx, dy = {"05": (6, -2), "12": (-14, -6), "19": (6, 0)}.get(m["id"], (3, 3))
        ax.annotate(m["id"], (c.x, c.y), xytext=(dx, dy), textcoords="offset points", fontsize=7, fontweight="bold")
    ax.set_xlim(-50, -34.5); ax.set_ylim(-15.6, -1.8)
    ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
    for i, c in enumerate(FOLD_CORES, 1):
        ax.scatter([], [], color=c, marker="s", s=40, label=f"Fold {i} (IDs {4 * i - 3:02d} a {4 * i:02d})")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
    ax.set_aspect("equal"); ax.grid(alpha=0.25, linewidth=0.4)
    salvar(fig, out, "Figura01_municipios_por_fold.png")


def _hist(F):
    return [json.load(open(F / f"fold_{k}.json", encoding="utf-8")) for k in range(5)]


def fig_curvas(F, out):
    H = _hist(F)
    fig, axs = plt.subplots(1, 2, figsize=(LARG, LARG * 0.36))
    h = H[1]["history"]
    ep = [x["epoch"] for x in h]
    axs[0].plot(ep, [x["train_loss"] for x in h], color=COR["azul"], label="treino")
    axs[0].plot(ep, [x["val_loss"] for x in h], color=COR["vermelho"], label="validação")
    axs[0].set_title("(a) Perda no Fold 2"); axs[0].set_xlabel("Época"); axs[0].set_ylabel("Perda"); axs[0].legend()
    for k, Hk in enumerate(H):
        hk = Hk["history"]
        axs[1].plot([x["epoch"] for x in hk], [x["mae_c"] for x in hk], color=FOLD_CORES[k], label=f"Fold {k + 1}", linewidth=1.2)
    axs[1].set_title("(b) MAE de validação por fold"); axs[1].set_xlabel("Época"); axs[1].set_ylabel("MAE (°C)")
    axs[1].legend(ncol=2)
    for a in axs:
        a.grid(alpha=0.3); a.set_xticks(range(0, 21, 5))
    salvar(fig, out, "Figura02_curvas_treinamento.png")


def fig_desvio(F, out):
    H = _hist(F)
    fig, ax = plt.subplots(figsize=(LARG * 0.6, LARG * 0.33))
    for k, Hk in enumerate(H):
        hk = Hk["history"]
        ax.plot([x["epoch"] for x in hk], [x["std_pred"] for x in hk], color=FOLD_CORES[k], label=f"Fold {k + 1}", linewidth=1.1)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="alvo")
    ax.set_xlabel("Época"); ax.set_ylabel("Desvio-padrão (anomalia padronizada)")
    ax.set_ylim(0.75, 1.03); ax.grid(alpha=0.3); ax.legend(ncol=3, loc="lower right")
    salvar(fig, out, "Figura03_desvio_padrao.png")


def fig_mapa_teresina(D, out):
    A = D / "derivados_ampliado"
    L = dict(np.load(A / "teresina_camadas.npz"))
    meta = json.load(open(A / "teresina_camadas.json", encoding="utf-8"))
    w, s, e, n = meta["bounds"]
    ext = [w, e, s, n]
    b = gpd.read_file(A / "teresina_bairros.geojson")
    obs, pred = L["lst_obs"], L["lst_pred"]
    err = np.abs(pred - obs)
    vmin, vmax = np.nanpercentile(obs, [1, 99])
    fig, axs = plt.subplots(1, 3, figsize=(LARG, LARG * 0.55), constrained_layout=True)
    for a, arr, t, cm, lim in ((axs[0], obs, "(a) LST observada (°C)", "inferno", (vmin, vmax)),
                               (axs[1], pred, "(b) LST prevista (°C)", "inferno", (vmin, vmax)),
                               (axs[2], err, "(c) Erro absoluto (°C)", "Greys", (0, 4))):
        im = a.imshow(arr, extent=ext, cmap=cm, vmin=lim[0], vmax=lim[1], interpolation="nearest")
        b.boundary.plot(ax=a, color="#30a0ff" if cm == "inferno" else "#c2412d", linewidth=0.25)
        a.set_title(t); a.set_xticks([]); a.set_yticks([]); a.set_xlabel(""); a.set_ylabel("")
        fig.colorbar(im, ax=a, orientation="horizontal", fraction=0.05, pad=0.02, shrink=0.85)
    fig.text(0.5, -0.03, f"Composição da estação seca (13 meses, 2018 a 2024), 123 bairros; MAE = {meta['mae_composicao_c']:.2f} °C, ".replace(".", ",") +
             f"correlação = {meta['pearson_composicao']:.2f}".replace(".", ",") + f". Em branco: sem LST observada no produto Landsat.", ha="center", fontsize=7.5)
    salvar(fig, out, "Figura05_teresina_observado_previsto_erro.png")


def fig_referencias(F, out):
    v = json.load(open(F / "verificacao" / "verificacao.json", encoding="utf-8"))["resumo"]["final"]
    nomes = ["UrbanHeatNet\n(4 índices)", "Regressão linear\n(4 índices)", "Média da cena", "Persistência*\n(usa a LST anterior)"]
    chaves = ["oraculo", "linear_oraculo", "media_cena", "persistencia"]
    vals = [v[k][0] for k in chaves]; dps = [v[k][1] for k in chaves]
    fig, ax = plt.subplots(figsize=(LARG * 0.6, LARG * 0.32))
    cores = [COR["verde"], "#9fb7a8", "#9fb7a8", "#d8d2bf"]
    ax.barh(nomes[::-1], vals[::-1], xerr=dps[::-1], color=cores[::-1], capsize=3)
    for i, (val, dp) in enumerate(zip(vals[::-1], dps[::-1])):
        ax.text(val + dp + 0.05, i, f"{val:.2f}".replace(".", ","), va="center", fontsize=8)
    ax.set_xlabel("MAE da anomalia de LST nas cidades de validação (°C)")
    ax.set_xlim(0, 3.4); ax.grid(axis="x", alpha=0.3)
    salvar(fig, out, "Figura04_comparacao_referencias.png")


def fig_validacao_real(D, out):
    v = json.load(open(D / "derivados" / "teresina_validacao_real.json", encoding="utf-8"))
    n, p = v["experimento_natural"], v["parques"]
    fig, axs = plt.subplots(1, 2, figsize=(LARG, LARG * 0.38))
    x = [c["dndvi"] for c in n["curva"]]
    axs[0].plot(x, [c["obs"] for c in n["curva"]], "o-", color="black", ms=3.5, label="medido (satélite)")
    axs[0].plot(x, [c["pred"] for c in n["curva"]], "s--", color=COR["verde"], ms=3.5, label="previsto (modelo)")
    axs[0].axhline(0, color=COR["cinza"], linewidth=0.6); axs[0].axvline(0, color=COR["cinza"], linewidth=0.6)
    axs[0].set_xlabel("Mudança de NDVI (2018–21 → 2023–24)"); axs[0].set_ylabel("Mudança da anomalia de LST (°C)")
    axs[0].set_title(f"(a) Experimento natural (r = {n['pearson_pixels_que_mudaram']:.2f})".replace(".", ","))
    axs[0].legend(); axs[0].grid(alpha=0.3)
    o = np.array([q["obs_c"] for q in p["parques"]]); pr = np.array([q["pred_c"] for q in p["parques"]])
    lim = np.ceil(max(np.abs(o).max(), np.abs(pr).max()))
    axs[1].plot([-lim, lim], [-lim, lim], "--", color=COR["cinza"], linewidth=0.8, label="1:1")
    axs[1].scatter(o, pr, s=14, color=COR["verde"], edgecolor="black", linewidth=0.3, label=f"{p['n_parques']} parques e praças")
    axs[1].set_xlim(-lim, lim); axs[1].set_ylim(-lim, lim); axs[1].set_aspect("equal")
    axs[1].set_xlabel("Medido: entorno − interior (°C)"); axs[1].set_ylabel("Previsto (°C)")
    axs[1].set_title(f"(b) Parques existentes (r = {p['pearson']:.2f})".replace(".", ","))
    axs[1].legend(loc="upper left"); axs[1].grid(alpha=0.3)
    salvar(fig, out, "Figura06_validacao_mundo_real.png")


def fig_inmet(D, out):
    i = json.load(open(D / "derivados" / "teresina_validacao_inmet.json", encoding="utf-8"))
    pts = i["pontos"]
    fig, ax = plt.subplots(figsize=(LARG * 0.55, LARG * 0.42))
    for seca, cor, nome in ((True, COR["vermelho"], "estação seca (jun a out)"), (False, COR["azul"], "estação chuvosa")):
        q = [x for x in pts if x["seca"] == seca]
        ax.scatter([x["ar"] for x in q], [x["lst"] for x in q], s=14, color=cor, label=nome, edgecolor="white", linewidth=0.3)
    lo = min(min(x["ar"] for x in pts), min(x["lst"] for x in pts)) - 1
    hi = max(max(x["ar"] for x in pts), max(x["lst"] for x in pts)) + 1
    ax.plot([lo, hi], [lo, hi], "--", color=COR["cinza"], linewidth=0.8, label="superfície = ar")
    ax.set_xlabel("Temperatura do ar às 10h (°C, INMET A312)"); ax.set_ylabel("LST no pixel da estação (°C, Landsat)")
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_title(f"{i['n_dias']} passagens sem nuvens, 2018–2024 (r = {i['pearson_lst_estacao_x_ar']:.2f})".replace(".", ","))
    salvar(fig, out, "Figura07_superficie_x_ar_inmet.png")


def fig_bairros(D, out):
    A = D / "derivados_ampliado"
    b = gpd.read_file(A / "teresina_bairros.geojson")
    b["vuln_dens"] = b["vulneraveis"] / b["area_km2"]
    rk = lambda c: b[c].rank(pct=True)
    b["indice"] = 100 * (0.6 * rk("anomalia_c") + 0.4 * rk("vuln_dens"))
    fig, axs = plt.subplots(1, 2, figsize=(LARG, LARG * 0.66), gridspec_kw={"width_ratios": [1, 1], "wspace": 0.25})
    norm = TwoSlopeNorm(vmin=np.nanmin(b.anomalia_c), vcenter=0, vmax=np.nanmax(b.anomalia_c))
    b.plot(column="anomalia_c", ax=axs[0], cmap="RdYlBu_r", norm=norm, edgecolor="white", linewidth=0.3, legend=True,
           legend_kwds={"label": "°C em relação à média", "shrink": 0.6})
    axs[0].set_title("(a) Temperatura da superfície", fontsize=9); axs[0].set_axis_off()
    b.plot(column="indice", ax=axs[1], cmap="YlOrRd", edgecolor="white", linewidth=0.3, legend=True,
           legend_kwds={"label": "índice (0 a 100)", "shrink": 0.6})
    top = b.sort_values("indice", ascending=False).head(10)
    for k, (_, r) in enumerate(top.iterrows(), 1):
        c = r.geometry.representative_point()
        axs[1].annotate(str(k), (c.x, c.y), ha="center", va="center", fontsize=6.5, fontweight="bold",
                        bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec="black", lw=0.4))
    axs[1].set_title("(b) Prioridade: calor 60% × vulnerabilidade 40%", fontsize=9); axs[1].set_axis_off()
    legenda = "; ".join(f"{k}. {r.bairro}" for k, (_, r) in enumerate(top.iterrows(), 1))
    fig.text(0.5, 0.03, "10 bairros prioritários: " + legenda, ha="center", fontsize=7, wrap=True)
    salvar(fig, out, "Figura08_bairros_calor_prioridade.png")
    return top[["bairro", "anomalia_c", "vulneraveis", "indice"]]


def fig_estrategias(D, out):
    A = D / "derivados_ampliado"
    fc = json.load(open(A / "teresina_areas.geojson", encoding="utf-8"))["features"]
    razao = min(json.load(open(A / "teresina_areas_info.json", encoding="utf-8"))["aditividade"]["razao_conjunto_soma"], 1)
    P = [f["properties"] for f in fc if f["properties"]["tipo"] in ("praca", "calcada") and f["properties"]["resfriamento_c"] >= 0.05]
    arv = lambda p: max(1, round(p["area_plantavel_ha"] * (10000 / 64 if p["tipo"] == "praca" else 10000 / 12.5)))
    def curva(ordem):
        x, a05 = [0], [0]
        for p in ordem:
            x.append(x[-1] + arv(p)); a05.append(a05[-1] + p["area_05c_ha"])
        return np.array(x), np.array(a05) * razao
    ef = sorted(P, key=lambda p: -p["resfriamento_int"] / max(p["area_plantavel_ha"], 1e-6))
    qu = sorted(P, key=lambda p: -p["lst_media_c"])
    grade = np.linspace(0, sum(arv(p) for p in P), 200)
    rng = np.random.default_rng(42)
    al = np.array([np.interp(grade, *curva([P[i] for i in rng.permutation(len(P))])) for _ in range(50)])
    fig, ax = plt.subplots(figsize=(LARG * 0.62, LARG * 0.36))
    for o, nome, cor in ((ef, "maior resfriamento por árvore (proposta)", COR["verde"]), (qu, "mais quentes primeiro", COR["vermelho"])):
        x, y = curva(o)
        ax.plot(x / 1000, y, color=cor, label=nome, linewidth=1.6)
    ax.plot(grade / 1000, al.mean(0), color=COR["cinza"], label="aleatória (média de 50 sorteios)")
    ax.fill_between(grade / 1000, al.mean(0) - al.std(0), al.mean(0) + al.std(0), color=COR["cinza"], alpha=0.2)
    ax.axvline(7.595, color="black", linestyle=":", linewidth=0.9)
    ax.set_xlabel("Árvores plantadas (mil)"); ax.set_ylabel("Área ≥ 0,5 °C mais fresca (ha)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False); ax.grid(alpha=0.3)
    ax.text(8.3, ax.get_ylim()[1] * 0.62, "plano padrão\n(7.595 árvores)", fontsize=7)
    salvar(fig, out, "Figura09_area_beneficiada_x_arvores.png")
    x, y = curva(ef); xq, yq = curva(qu)
    return {"n_areas": len(P), "efic_7595": float(np.interp(7595, x, y)), "quente_7595": float(np.interp(7595, xq, yq)),
            "aleat_7595": float(np.interp(7595, grade, al.mean(0)))}


def main(D, out):
    D, out = Path(D), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    F = D / "drive_final" / "arbormap_final"
    fig_municipios(D, out)
    fig_curvas(F, out)
    fig_desvio(F, out)
    fig_mapa_teresina(D, out)
    fig_referencias(F, out)
    fig_validacao_real(D, out)
    fig_inmet(D, out)
    top = fig_bairros(D, out)
    est = fig_estrategias(D, out)
    print(top.to_string(index=False))
    print(est)
    json.dump({"top10_bairros": top.drop(columns="geometry", errors="ignore").to_dict("records"), "estrategias": est},
              open(out / "numeros_figuras.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
