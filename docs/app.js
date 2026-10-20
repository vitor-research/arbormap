/* ARBORMAP: mapa interativo de ilhas de calor e planejamento de arborização (Teresina)
   Página estática: lê docs/data/ (gerado por arbormap.decision.export_pagina) e faz todo o
   cálculo do plano no navegador, somando os efeitos individuais pré-simulados de cada área. */

const DATA = "data/";
const CIDADE = "teresina";
const PALETAS = {
  RdYlBu_r: ["#313695", "#4575b4", "#74add1", "#abd9e9", "#e0f3f8", "#ffffbf", "#fee090", "#fdae61", "#f46d43", "#d73027", "#a50026"],
  magma_r: ["#fcfdbf", "#fe9f6d", "#de4968", "#8c2981", "#3b0f70", "#000004"],
  YlGn: ["#ffffe5", "#d9f0a3", "#78c679", "#238443", "#004529"],
  YlOrRd: ["#ffffcc", "#fed976", "#fd8d3c", "#e31a1c", "#800026"],
};
const TIPOS = { praca: "Praça/parque", calcada: "Calçada/canteiro", livre: "Área livre" };
const COR_TIPO = { praca: "#1d8a45", calcada: "#2a6fbf", livre: "#c98a00" };
const EQUIP = { escola: "escola", creche: "creche", saude: "unidade de saúde" };
const ESTRATEGIAS = { eficiencia: "Maior resfriamento por árvore", multicriterio: "Multicritério do modelo", quente: "Mais quentes primeiro" };
const METRICAS = {
  indice: { nome: "Índice calor × vulnerabilidade", un: "", pal: "YlOrRd", dec: 0 },
  anomalia_c: { nome: "Temperatura acima da média", un: " °C", pal: "YlOrRd", dec: 1 },
  hotspot_pct: { nome: "Área em ilha de calor", un: "%", pal: "YlOrRd", dec: 0 },
  vuln_dens: { nome: "Pessoas vulneráveis por km²", un: "", pal: "YlOrRd", dec: 0 },
  vegetacao_densa_pct: { nome: "Vegetação densa", un: "%", pal: "YlGn", dec: 0 },
};

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const fmt = (v, d = 0) => (v == null || !isFinite(v) ? "–" : Number(v).toLocaleString("pt-BR", { minimumFractionDigits: d, maximumFractionDigits: d }));
const sinal = (v, d = 1) => (v > 0 ? "+" : "") + fmt(v, d);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function corEscala(pal, t) {
  const p = PALETAS[pal];
  t = Math.max(0, Math.min(1, t));
  const x = t * (p.length - 1), i = Math.min(Math.floor(x), p.length - 2), f = x - i;
  const a = p[i].match(/\w\w/g).map((h) => parseInt(h, 16)), b = p[i + 1].match(/\w\w/g).map((h) => parseInt(h, 16));
  return "#" + a.map((v, k) => Math.round(v + (b[k] - v) * f).toString(16).padStart(2, "0")).join("");
}
const gradiente = (pal) => `linear-gradient(to right, ${PALETAS[pal].join(",")})`;

/* ---------------- estado ---------------- */
const S = {
  meta: null, bairros: null, areas: [], equip: [],
  estrategia: "eficiencia", orcArvores: 0, unidade: "arvores",
  tipos: new Set(["praca", "calcada"]), bairro: "", filtroEquip: false, dist: 300, filtroInc: false,
  custoArvore: null, custoManut: null, m2: 36, manual: new Set(),
  pesoCalor: 0.6, metrica: "indice", sortB: { k: "indice", asc: false }, bairroSel: null,
  modo: "gestor", aba: "planejar",
};
let map, camadas = {}, layerAreas, layerBairros, layerEquip, legenda, chartCurva, chartRef;
let legendaAberta = window.innerWidth > 820;

/* ---------------- carregamento ---------------- */
async function carregar() {
  const get = (f) => fetch(DATA + f).then((r) => { if (!r.ok) throw new Error(f); return r.json(); });
  const [meta, bairros, areas, equip] = await Promise.all([
    get("meta.json"), get(`${CIDADE}_bairros.geojson`), get(`${CIDADE}_areas.geojson`),
    get(`${CIDADE}_equipamentos.geojson`).catch(() => ({ features: [] })),
  ]);
  S.meta = meta; S.bairros = bairros; S.equip = equip.features;
  S.areas = areas.features.map((f) => {
    const p = f.properties;
    p.eficiencia = p.resfriamento_int / Math.max(p.area_plantavel_ha, 1e-6);
    return f;
  });
  prepararBairros();
  lerHash();
  iniciarMapa();
  iniciarControles();
  preencherConfiabilidade();
  preencherSobre();
  atualizarTudo();
}

/* ---------------- bairros: índice calor × vulnerabilidade ---------------- */
function prepararBairros() {
  const fs = S.bairros.features;
  fs.forEach((f) => {
    const p = f.properties;
    p.vuln_dens = p.area_km2 > 0 ? p.vulneraveis / p.area_km2 : null;
    p.tem = p.anomalia_c != null;
  });
  const com = fs.filter((f) => f.properties.tem);
  const rank = (k) => {
    const v = com.map((f) => f.properties[k]).sort((a, b) => a - b);
    com.forEach((f) => {
      const x = f.properties[k];
      let lo = 0; while (lo < v.length && v[lo] < x) lo++;
      let hi = lo; while (hi < v.length && v[hi] === x) hi++;
      f.properties["r_" + k] = v.length > 1 ? (lo + hi - 1) / 2 / (v.length - 1) : 0.5;
    });
  };
  rank("anomalia_c"); rank("vuln_dens");
  calcularIndice();
}
function calcularIndice() {
  S.bairros.features.forEach((f) => {
    const p = f.properties;
    p.indice = p.tem ? 100 * (S.pesoCalor * p.r_anomalia_c + (1 - S.pesoCalor) * p.r_vuln_dens) : null;
  });
  const ord = S.bairros.features.filter((f) => f.properties.tem).sort((a, b) => b.properties.anomalia_c - a.properties.anomalia_c);
  ord.forEach((f, i) => (f.properties.pos_calor = i + 1));
}

/* ---------------- plano ---------------- */
const arvoresDe = (p) => Math.max(1, Math.round((p.area_plantavel_ha * 10000) / S.m2));
const custoUnit = () => (S.custoArvore || 0) + (S.custoManut || 0);

function areasFiltradas() {
  return S.areas.filter((f) => {
    const p = f.properties;
    if (!S.tipos.has(p.tipo)) return false;
    if (S.bairro && p.bairro !== S.bairro) return false;
    if (S.filtroEquip && !(p.equipamento_proximo && p.equipamento_proximo.dist_m <= S.dist)) return false;
    if (S.filtroInc && !(p.incerteza_c < 0.5)) return false;
    return true;
  });
}
function ordenar(lista, estr) {
  const l = [...lista];
  if (estr === "eficiencia") return l.sort((a, b) => b.properties.eficiencia - a.properties.eficiencia);
  if (estr === "quente") return l.sort((a, b) => b.properties.lst_media_c - a.properties.lst_media_c);
  // multicritério: 60% potencial de resfriamento estimado pela rede, 40% calor do local (posições relativas)
  const r = (k) => { const s = [...l].sort((a, b) => a.properties[k] - b.properties[k]); const m = new Map(); s.forEach((f, i) => m.set(f, i / Math.max(1, s.length - 1))); return m; };
  const rp = r("potencial"), rl = r("lst_media_c");
  return l.sort((a, b) => 0.6 * rp.get(b) + 0.4 * rl.get(b) - (0.6 * rp.get(a) + 0.4 * rl.get(a)));
}
function montarPlano() {
  const filtradas = areasFiltradas();
  if (S.estrategia === "manual") return { filtradas, plano: S.areas.filter((f) => S.manual.has(f.properties.id)) };
  const plano = [];
  let usado = 0;
  for (const f of ordenar(filtradas, S.estrategia)) {
    const n = arvoresDe(f.properties);
    if (usado + n <= S.orcArvores) { plano.push(f); usado += n; }
  }
  return { filtradas, plano };
}
function resumoPlano(plano) {
  const razao = plano.length > 1 ? S.meta.areas?.aditividade?.razao_conjunto_soma ?? 1 : 1;
  let arv = 0, ha = 0, integ = 0, a05 = 0, wsum = 0, dsum = 0, perto = 0;
  plano.forEach((f) => {
    const p = f.properties, n = arvoresDe(p);
    arv += n; ha += p.area_plantavel_ha; integ += p.resfriamento_int; a05 += p.area_05c_ha;
    dsum += p.resfriamento_c * p.area_plantavel_ha; wsum += p.area_plantavel_ha;
    if (p.equipamento_proximo && p.equipamento_proximo.dist_m <= 300) perto++;
  });
  return { n: plano.length, arv, ha, integ: integ * Math.min(razao, 1), integBruto: integ, razao,
           a05: a05 * Math.min(razao, 1), dt: wsum ? dsum / wsum : 0, perto, custo: custoUnit() ? arv * custoUnit() : null };
}

/* ---------------- mapa ---------------- */
function iniciarMapa() {
  const m = S.meta;
  map = L.map("map", { zoomControl: true, preferCanvas: false }).fitBounds(m.bounds_leaflet);
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19, className: "basemap",
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
  }).addTo(map);
  const ov = (k, op) => L.imageOverlay(DATA + m.camadas[k].arquivo, m.bounds_leaflet, { opacity: op, className: "pix" });
  camadas = {
    anomalia: ov("anomalia", 0.7), incerteza: ov("incerteza", 0.75), plantavel: ov("plantavel", 0.8), inviavel: ov("inviavel", 0.6),
  };
  camadas.anomalia.addTo(map);
  layerBairros = L.geoJSON(S.bairros, { style: estiloBairro, onEachFeature: (f, l) => l.on("click", () => selecionarBairro(f.properties.cd_bairro, true)) }).addTo(map);
  layerAreas = L.geoJSON(null, { style: estiloArea, onEachFeature: ligarArea }).addTo(map);
  layerEquip = L.layerGroup(S.equip.map((f) => {
    const [lon, lat] = f.geometry.coordinates, c = f.properties.categoria;
    return L.circleMarker([lat, lon], { radius: 4, weight: 1, color: "#333", fillColor: c === "saude" ? "#d23" : c === "creche" ? "#e98a00" : "#6a3fb5", fillOpacity: 0.9 })
      .bindTooltip(`${esc(f.properties.nome)} (${EQUIP[c]})`);
  }));
  L.control.layers(null, {
    "Temperatura prevista (anomalia)": camadas.anomalia, "Incerteza do modelo": camadas.incerteza,
    "Espaço livre para plantio": camadas.plantavel, "Inviável (água, encosta, sem dado)": camadas.inviavel,
    "Bairros": layerBairros, "Áreas candidatas": layerAreas, "Escolas, creches e saúde": layerEquip,
  }, { collapsed: true }).addTo(map);
  legenda = L.control({ position: "bottomleft" });
  legenda.onAdd = () => { const d = L.DomUtil.create("div", "legend"); L.DomEvent.disableClickPropagation(d); return d; };
  legenda.addTo(map);
  map.on("overlayadd overlayremove", desenharLegenda);
  const st = document.createElement("style");
  st.textContent = ".pix{image-rendering:pixelated;image-rendering:crisp-edges}";
  document.head.appendChild(st);
}
function desenharLegenda() {
  const d = $(".legend"), m = S.meta.camadas;
  let h = "";
  if (map.hasLayer(camadas.anomalia)) h += `<b>Temperatura da superfície</b><br><span class="small">em relação à média da cidade (${fmt(S.meta.lst_media_cidade_c, 1)} °C)</span>
    <div class="bar" style="background:${gradiente("RdYlBu_r")}"></div><div class="ticks"><span>${sinal(m.anomalia.min, 0)} °C</span><span>0</span><span>${sinal(m.anomalia.max, 0)} °C</span></div>`;
  if (map.hasLayer(camadas.incerteza)) h += `<b>Incerteza (entre 5 modelos)</b><div class="bar" style="background:${gradiente("magma_r")}"></div><div class="ticks"><span>0</span><span>${fmt(m.incerteza.max, 1)} °C</span></div>`;
  if (map.hasLayer(camadas.plantavel)) h += `<b>Espaço livre para plantio</b><div class="bar" style="background:${gradiente("YlGn")}"></div><div class="ticks"><span>0%</span><span>100% do pixel</span></div>`;
  if (map.hasLayer(layerAreas) && S.modo === "gestor") h += `<div class="item"><span class="sw" style="background:#0b6b3a"></span>no plano</div>` +
    Object.entries(TIPOS).map(([k, v]) => `<div class="item"><span class="sw" style="border:2px solid ${COR_TIPO[k]}"></span>${v}</div>`).join("");
  if (map.hasLayer(layerBairros) && S.aba === "bairros" && S.modo === "gestor") {
    const mt = METRICAS[S.metrica], [lo, hi] = faixaMetrica();
    h += `<b>${mt.nome}</b><div class="bar" style="background:${gradiente(mt.pal)}"></div><div class="ticks"><span>${fmt(lo, mt.dec)}${mt.un}</span><span>${fmt(hi, mt.dec)}${mt.un}</span></div><span class="small">cinza: sem cobertura do satélite</span>`;
  }
  const atual = camadas.anomalia;
  h += `<div class="opacity"><span class="small">Transparência das camadas</span><input type="range" min="0" max="1" step="0.05" value="${atual.options.opacity}" id="opac"></div>`;
  d.innerHTML = `<details ${legendaAberta ? "open" : ""}><summary>Legenda</summary>${h}</details>`;
  $("details", d).ontoggle = (e) => (legendaAberta = e.target.open);
  $("#opac", d).oninput = (e) => Object.values(camadas).forEach((c) => c.setOpacity(+e.target.value));
}
function faixaMetrica() {
  const v = S.bairros.features.map((f) => f.properties[S.metrica]).filter((x) => x != null && isFinite(x)).sort((a, b) => a - b);
  if (!v.length) return [0, 1];
  return [v[Math.floor(v.length * 0.02)], v[Math.ceil(v.length * 0.98) - 1]];
}
function estiloBairro(f) {
  const p = f.properties, sel = S.bairroSel === p.cd_bairro;
  if (S.aba !== "bairros" || S.modo !== "gestor") {
    return { color: sel ? "#111" : "#555", weight: sel ? 2.5 : 0.7, opacity: sel ? 1 : 0.5, fill: true, fillOpacity: 0, dashArray: sel ? null : "2 3" };
  }
  const v = p[S.metrica], [lo, hi] = faixaMetrica();
  return { color: sel ? "#111" : "#fff", weight: sel ? 3 : 0.8, fillOpacity: v == null ? 0.25 : 0.72,
           fillColor: v == null ? "#bbb" : corEscala(METRICAS[S.metrica].pal, (v - lo) / (hi - lo || 1)) };
}

/* ---------------- áreas no mapa ---------------- */
let planoAtual = new Set();
function estiloArea(f) {
  const p = f.properties, no = planoAtual.has(p.id);
  return { color: no ? "#0b6b3a" : COR_TIPO[p.tipo], weight: no ? 1.5 : 1.2, fillColor: no ? "#0b6b3a" : COR_TIPO[p.tipo], fillOpacity: no ? 0.75 : 0.12 };
}
function popupArea(p) {
  const no = planoAtual.has(p.id), eq = p.equipamento_proximo;
  return `<b>${TIPOS[p.tipo]} #${p.id}</b>${p.bairro ? " · " + esc(p.bairro) : ""}<br>
    ${fmt(arvoresDe(p))} árvores em ${fmt(p.area_plantavel_ha, 2)} ha plantáveis<br>
    Temperatura atual: <b>${fmt(p.lst_media_c, 1)} °C</b> (incerteza ±${fmt(p.incerteza_c, 1)})<br>
    Resfriamento local: <b>${fmt(p.resfriamento_c, 2)} °C</b><br>
    Área com ≥ 0,5 °C a menos: ${fmt(p.area_05c_ha, 1)} ha<br>
    Resfriamento integrado: ${fmt(p.resfriamento_int, 1)} °C·ha
    ${eq ? `<br>A ${fmt(eq.dist_m)} m de ${EQUIP[eq.categoria]}: ${esc(eq.nome)}` : ""}
    ${p.tipo === "livre" ? '<br><span class="warn">Área possivelmente privada: verificar.</span>' : ""}
    ${S.modo === "gestor" ? `<br><button class="btn ${no ? "ghost" : ""}" style="margin-top:6px" onclick="alternarArea(${p.id})">${no ? "Retirar do plano" : "Incluir no plano"}</button>` : ""}`;
}
function ligarArea(f, l) {
  l.on("click", () => l.bindPopup(popupArea(f.properties)).openPopup());
}
window.alternarArea = (id) => {
  if (S.estrategia !== "manual") {
    S.manual = new Set(planoAtual);
    S.estrategia = "manual";
    $(`input[name=estr][value=manual]`).checked = true;
  }
  S.manual.has(id) ? S.manual.delete(id) : S.manual.add(id);
  map.closePopup();
  atualizarTudo();
};

/* ---------------- controles ---------------- */
function iniciarControles() {
  $$("#tabs button").forEach((b) => (b.onclick = () => abrirAba(b.dataset.tab)));
  $("#modo-gestor").onclick = () => mudarModo("gestor");
  $("#modo-cidadao").onclick = () => mudarModo("cidadao");

  $("#orc-unidade").value = S.unidade;
  $("#orc-unidade").onchange = (e) => { S.unidade = e.target.value; atualizarTudo(); };
  $("#orc-valor").oninput = (e) => {
    const v = +e.target.value || 0;
    S.orcArvores = S.unidade === "arvores" ? v : S.unidade === "ha" ? Math.round((v * 10000) / S.m2) : custoUnit() ? Math.floor(v / custoUnit()) : 0;
    atualizarTudo(false);
  };
  $("#orc-slider").oninput = (e) => { S.orcArvores = +e.target.value; atualizarTudo(); };
  const num = (id, k) => { const el = $(id); el.value = S[k] ?? ""; el.oninput = () => { S[k] = el.value === "" ? null : +el.value; atualizarTudo(); }; };
  num("#custo-arvore", "custoArvore"); num("#custo-manut", "custoManut");
  $("#m2-arvore").value = S.m2;
  $("#m2-arvore").oninput = (e) => { S.m2 = Math.max(4, +e.target.value || 36); atualizarTudo(); };

  $$("input[name=estr]").forEach((r) => { r.checked = r.value === S.estrategia; r.onchange = () => {
    if (r.value === "manual" && !S.manual.size) S.manual = new Set(planoAtual);
    S.estrategia = r.value; atualizarTudo(); }; });
  $$(".f-tipo").forEach((c) => { c.checked = S.tipos.has(c.value); c.onchange = () => { c.checked ? S.tipos.add(c.value) : S.tipos.delete(c.value); atualizarTudo(); }; });
  const nomes = [...new Set(S.areas.map((f) => f.properties.bairro).filter(Boolean))].sort((a, b) => a.localeCompare(b, "pt-BR"));
  $("#f-bairro").insertAdjacentHTML("beforeend", nomes.map((n) => `<option>${esc(n)}</option>`).join(""));
  $("#f-bairro").value = S.bairro;
  $("#f-bairro").onchange = (e) => { S.bairro = e.target.value; atualizarTudo(); if (S.bairro) zoomBairroNome(S.bairro); };
  $("#f-equip").checked = S.filtroEquip; $("#f-equip").onchange = (e) => { S.filtroEquip = e.target.checked; atualizarTudo(); };
  $("#f-dist").value = S.dist; $("#f-dist").oninput = (e) => { S.dist = +e.target.value || 300; atualizarTudo(); };
  $("#f-incerteza").checked = S.filtroInc; $("#f-incerteza").onchange = (e) => { S.filtroInc = e.target.checked; atualizarTudo(); };
  $("#btn-link").onclick = () => {
    escreverHash();
    navigator.clipboard?.writeText(location.href).then(() => ($("#link-ok").textContent = "Link copiado."), () => ($("#link-ok").textContent = "Copie o endereço da barra do navegador."));
  };

  $("#b-metrica").onchange = (e) => { S.metrica = e.target.value; $("#peso-box").style.display = S.metrica === "indice" ? "" : "none"; atualizarBairros(); };
  $("#peso-calor").value = Math.round(S.pesoCalor * 100);
  $("#peso-calor").oninput = (e) => { S.pesoCalor = e.target.value / 100; $("#peso-lbl").textContent = e.target.value + "%"; calcularIndice(); atualizarBairros(); };
  $$("#tab-bairros-t th").forEach((th) => (th.onclick = () => {
    const k = th.dataset.k; S.sortB = { k, asc: S.sortB.k === k ? !S.sortB.asc : k === "bairro" }; atualizarBairros();
  }));

  const lista = S.bairros.features.map((f) => f.properties.bairro).sort((a, b) => a.localeCompare(b, "pt-BR"));
  $("#c-lista").innerHTML = lista.map((n) => `<option value="${esc(n)}">`).join("");
  $("#c-busca").oninput = (e) => {
    const f = S.bairros.features.find((x) => norm(x.properties.bairro) === norm(e.target.value));
    if (f) selecionarBairro(f.properties.cd_bairro, true);
  };
}
const norm = (s) => String(s || "").normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase().trim();

function abrirAba(aba) {
  S.aba = aba;
  $$("#tabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === aba));
  $$(".panel").forEach((p) => p.classList.toggle("on", p.id === "tab-" + aba));
  if (aba === "bairros" && !map.hasLayer(layerBairros)) layerBairros.addTo(map);
  layerBairros.setStyle(estiloBairro);
  if (aba === "bairros") layerBairros.bringToFront(); else layerAreas.bringToFront();
  desenharLegenda();
  setTimeout(() => map.invalidateSize(), 50);
}
function mudarModo(modo) {
  S.modo = modo;
  $("#modo-gestor").classList.toggle("on", modo === "gestor");
  $("#modo-cidadao").classList.toggle("on", modo === "cidadao");
  $("#tabs").style.display = modo === "gestor" ? "" : "none";
  if (modo === "cidadao") {
    S.aba = "cidadao";
    $$(".panel").forEach((p) => p.classList.toggle("on", p.id === "tab-cidadao"));
    Object.entries(camadas).forEach(([k, c]) => (k === "anomalia" ? c.addTo(map) : map.removeLayer(c)));
    map.removeLayer(layerEquip);
    layerAreas.setStyle(() => ({ color: "#1d8a45", weight: 1, fillColor: "#1d8a45", fillOpacity: 0.25 }));
    layerBairros.setStyle(estiloBairro);
    if (S.bairroSel) mostrarCidadao(S.bairroSel);
  } else abrirAba("planejar");
  desenharLegenda();
}

/* ---------------- atualização ---------------- */
function atualizarTudo(reescreverValor = true) {
  const { filtradas, plano } = montarPlano();
  planoAtual = new Set(plano.map((f) => f.properties.id));
  const maxArv = filtradas.reduce((s, f) => s + arvoresDe(f.properties), 0);
  const sl = $("#orc-slider");
  sl.max = Math.max(maxArv, 1); sl.step = Math.max(1, Math.round(maxArv / 400));
  sl.value = Math.min(S.orcArvores, maxArv);
  sl.disabled = S.estrategia === "manual";
  $("#orc-valor").disabled = S.estrategia === "manual";
  if (reescreverValor) {
    const v = S.unidade === "arvores" ? S.orcArvores : S.unidade === "ha" ? (S.orcArvores * S.m2) / 10000 : custoUnit() ? S.orcArvores * custoUnit() : "";
    $("#orc-valor").value = v === "" ? "" : S.unidade === "ha" ? (+v).toFixed(1) : Math.round(v);
  }
  const av = $("#orc-aviso");
  av.hidden = !(S.unidade === "reais" && !custoUnit());
  av.textContent = "Informe o custo por árvore em \"Custos e premissas\" (ata de registro de preços ou SINAPI-PI) para planejar em reais.";

  layerAreas.clearLayers();
  const visiveis = new Set(filtradas.map((f) => f.properties.id));
  plano.forEach((f) => visiveis.add(f.properties.id));
  layerAreas.addData({ type: "FeatureCollection", features: S.areas.filter((f) => visiveis.has(f.properties.id)) });
  if (S.modo === "cidadao") layerAreas.setStyle(() => ({ color: "#1d8a45", weight: 1, fillColor: "#1d8a45", fillOpacity: 0.25 }));

  const r = resumoPlano(plano);
  $("#cards").innerHTML = [
    ["hl", fmt(r.arv), "árvores em " + fmt(r.n) + " áreas"],
    ["", fmt(r.ha, 1) + " ha", "área plantada"],
    ["hl", fmt(r.dt, 2) + " °C", "resfriamento médio no local plantado"],
    ["", fmt(r.a05, 0) + " ha", "ficam ao menos 0,5 °C mais frescos"],
    ["", fmt(r.integ, 0), "resfriamento integrado (°C·ha)"],
    ["", r.custo != null ? "R$ " + fmt(r.custo) : "informar custo", r.custo != null ? "custo estimado (plantio + manutenção)" : "custo: abra \"Custos e premissas\""],
  ].map(([c, v, k]) => `<div class="card ${c}"><div class="v">${v}</div><div class="k">${k}</div></div>`).join("");
  $("#resumo-nota").innerHTML = r.n > 1
    ? `${fmt(r.perto)} das ${fmt(r.n)} áreas ficam a até 300 m de escola, creche ou unidade de saúde. Áreas vizinhas não somam integralmente: os totais aplicam a razão medida entre o efeito conjunto e a soma dos individuais (${fmt(r.razao, 2)}). Efeito para copas formadas (5 a 10 anos).`
    : "Efeito para copas formadas (5 a 10 anos).";

  const tb = $("#tab-areas tbody");
  tb.innerHTML = plano.slice().sort((a, b) => b.properties.resfriamento_int - a.properties.resfriamento_int).map((f) => {
    const p = f.properties, eq = p.equipamento_proximo;
    return `<tr class="click" data-id="${p.id}"><td>${p.id}</td><td><span class="pill ${p.tipo}">${TIPOS[p.tipo]}</span></td><td>${esc(p.bairro || "–")}</td>
      <td class="num">${fmt(arvoresDe(p))}</td><td class="num">${fmt(p.resfriamento_c, 2)}</td><td>${eq && eq.dist_m <= S.dist ? `${EQUIP[eq.categoria]} (${fmt(eq.dist_m)} m)` : ""}</td></tr>`;
  }).join("") || `<tr><td colspan="6" class="small">Nenhuma área no plano. Aumente o número de árvores ou ajuste os filtros.</td></tr>`;
  $$("tr.click", tb).forEach((tr) => (tr.onclick = () => zoomArea(+tr.dataset.id)));

  desenharCurva(filtradas, r.arv);
  atualizarBairros();
  desenharLegenda();
  escreverHash();
}
function zoomArea(id) {
  layerAreas.eachLayer((l) => {
    if (l.feature.properties.id === id) { map.fitBounds(l.getBounds(), { maxZoom: 16 }); l.bindPopup(popupArea(l.feature.properties)).openPopup(); }
  });
}

function desenharCurva(filtradas, atual) {
  const razao = S.meta.areas?.aditividade?.razao_conjunto_soma ?? 1;
  const cores = { eficiencia: "#1f7a4d", multicriterio: "#2a6fbf", quente: "#c2412d" };
  const ds = Object.keys(ESTRATEGIAS).map((e) => {
    let x = 0, y = 0;
    const pts = [{ x: 0, y: 0 }];
    ordenar(filtradas, e).forEach((f, i) => {
      x += arvoresDe(f.properties); y += f.properties.resfriamento_int;
      pts.push({ x, y: y * (i ? Math.min(razao, 1) : 1) });
    });
    return { label: ESTRATEGIAS[e], data: pts, borderColor: cores[e], backgroundColor: cores[e], pointRadius: 0, borderWidth: e === S.estrategia ? 3 : 1.5, tension: 0 };
  });
  const ymax = Math.max(1, ...ds.flatMap((d) => d.data.map((p) => p.y)));
  ds.push({ label: "Plano atual", data: [{ x: atual, y: 0 }, { x: atual, y: ymax }], borderColor: "#111", borderDash: [4, 4], borderWidth: 1, pointRadius: 0 });
  if (chartCurva) { chartCurva.data.datasets = ds; chartCurva.update("none"); return; }
  chartCurva = new Chart($("#chart-curva"), {
    type: "line", data: { datasets: ds },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false, parsing: false,
      plugins: { legend: { labels: { boxWidth: 10, font: { size: 10 } } }, tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${fmt(c.parsed.y, 0)} °C·ha com ${fmt(c.parsed.x)} árvores` } } },
      scales: { x: { type: "linear", title: { display: true, text: "árvores" }, ticks: { callback: (v) => fmt(v) } },
                y: { title: { display: true, text: "°C·ha" }, ticks: { callback: (v) => fmt(v) } } },
    },
  });
}

/* ---------------- bairros ---------------- */
function atualizarBairros() {
  layerBairros.setStyle(estiloBairro);
  const { k, asc } = S.sortB;
  const fs = S.bairros.features.slice().sort((a, b) => {
    const x = a.properties[k], y = b.properties[k];
    if (x == null) return 1; if (y == null) return -1;
    const c = typeof x === "string" ? x.localeCompare(y, "pt-BR") : x - y;
    return asc ? c : -c;
  });
  $$("#tab-bairros-t th").forEach((th) => { th.classList.toggle("sorted", th.dataset.k === k); th.classList.toggle("asc", th.dataset.k === k && asc); });
  $("#tab-bairros-t tbody").innerHTML = fs.map((f) => {
    const p = f.properties;
    return `<tr class="click ${S.bairroSel === p.cd_bairro ? "sel" : ""}" data-cd="${p.cd_bairro}"><td>${esc(p.bairro)}</td><td class="num">${fmt(p.indice)}</td>
      <td class="num">${p.anomalia_c == null ? "–" : sinal(p.anomalia_c)}</td><td class="num">${fmt(p.hotspot_pct)}</td><td class="num">${fmt(p.vuln_dens)}</td><td class="num">${fmt(p.vegetacao_densa_pct)}</td></tr>`;
  }).join("");
  $$("#tab-bairros-t tbody tr").forEach((tr) => (tr.onclick = () => selecionarBairro(tr.dataset.cd, true)));
  const b = S.meta.bairros;
  $("#b-cobertura").textContent = `${b.n_com_metricas} de ${b.n_bairros} bairros têm ao menos ${Math.round(b.cobertura_minima * 100)}% da área coberta pela imagem de satélite usada; os demais aparecem sem métricas de temperatura. Ilha de calor: pixels acima de ${fmt(b.lst_p90_c, 1)} °C (10% mais quentes da cidade).`;
  if (S.bairroSel && S.modo === "gestor") fichaBairro(S.bairroSel);
}
function bairroPorCd(cd) { return S.bairros.features.find((f) => f.properties.cd_bairro === cd); }
function zoomBairroNome(nome) {
  const f = S.bairros.features.find((x) => x.properties.bairro === nome);
  if (f) map.fitBounds(L.geoJSON(f).getBounds(), { maxZoom: 15 });
}
function selecionarBairro(cd, zoom) {
  S.bairroSel = cd;
  const f = bairroPorCd(cd);
  if (zoom && f) map.fitBounds(L.geoJSON(f).getBounds(), { maxZoom: 15 });
  if (S.modo === "cidadao") { mostrarCidadao(cd); layerBairros.setStyle(estiloBairro); return; }
  if (S.aba !== "bairros") abrirAba("bairros");
  atualizarBairros();
  $("#ficha").scrollIntoView({ block: "nearest" });
}
function fichaBairro(cd) {
  const p = bairroPorCd(cd).properties;
  const areas = S.areas.filter((f) => f.properties.bairro === p.bairro);
  const integ = areas.reduce((s, f) => s + f.properties.resfriamento_int, 0);
  const nCom = S.bairros.features.filter((f) => f.properties.tem).length;
  $("#ficha").innerHTML = `<div class="ficha"><h2>${esc(p.bairro)} <span class="small">· região ${esc(p.regiao || "–")}</span></h2>
    ${p.tem ? `<div class="big ${p.anomalia_c >= 0 ? "hot" : "cool"}">${sinal(p.anomalia_c)} °C</div>
      <p class="small">em relação à média da cidade na estação seca · ${p.pos_calor}º mais quente de ${nCom} · incerteza ±${fmt(p.incerteza_c, 1)} °C</p>
      <div class="cards">
        <div class="card"><div class="v">${fmt(p.indice)}</div><div class="k">índice calor × vulnerabilidade (0 a 100)</div></div>
        <div class="card"><div class="v">${fmt(p.hotspot_pct)}%</div><div class="k">da área em ilha de calor</div></div>
        <div class="card"><div class="v">${fmt(p.vegetacao_densa_pct)}%</div><div class="k">com vegetação densa</div></div>
        <div class="card"><div class="v">${fmt(p.lst_media_c, 1)} °C</div><div class="k">temperatura média da superfície</div></div>
      </div>` : `<p class="note">Este bairro fica fora (ou quase fora) da imagem de satélite usada; as métricas de temperatura não são calculadas. Os dados do Censo estão abaixo.</p>`}
    <div class="cards">
      <div class="card"><div class="v">${fmt(p.populacao)}</div><div class="k">moradores (Censo 2022)</div></div>
      <div class="card"><div class="v">${fmt(p.vulneraveis)}</div><div class="k">crianças de 0 a 4 anos e pessoas com 60+ (${fmt(p.vulneraveis_pct)}%)</div></div>
    </div>
    <p class="small">${areas.length ? `${areas.length} áreas candidatas no bairro, com resfriamento integrado somado de ${fmt(integ, 0)} °C·ha.` : "Nenhuma área candidata identificada no bairro."}</p>
    ${areas.length ? `<button class="btn" id="btn-planejar-b">Planejar neste bairro</button>` : ""}</div>`;
  const b = $("#btn-planejar-b");
  if (b) b.onclick = () => { S.bairro = p.bairro; $("#f-bairro").value = p.bairro; abrirAba("planejar"); atualizarTudo(); };
}

/* ---------------- cidadão ---------------- */
function mostrarCidadao(cd) {
  const p = bairroPorCd(cd).properties;
  $("#c-busca").value = p.bairro;
  const nCom = S.bairros.features.filter((f) => f.properties.tem).length;
  const areas = S.areas.filter((f) => f.properties.bairro === p.bairro && f.properties.tipo !== "livre");
  let frase;
  if (!p.tem) frase = "Este bairro fica fora da imagem de satélite usada nesta versão do mapa.";
  else if (p.anomalia_c >= 2) frase = "É uma das áreas mais quentes da cidade: a superfície esquenta bem mais que a média.";
  else if (p.anomalia_c >= 0.5) frase = "É mais quente que a média da cidade.";
  else if (p.anomalia_c > -0.5) frase = "Fica perto da média de temperatura da cidade.";
  else frase = "É mais fresco que a média da cidade, em geral por ter mais vegetação ou estar perto de rios.";
  $("#c-resultado").innerHTML = `<div class="ficha"><h2>${esc(p.bairro)}</h2>
    ${p.tem ? `<div class="big ${p.anomalia_c >= 0 ? "hot" : "cool"}">${sinal(p.anomalia_c)} °C</div><p class="small">que a média de Teresina na estação seca · ${p.pos_calor}º mais quente de ${nCom} bairros</p>` : ""}
    <p>${frase}</p>
    ${p.tem ? `<p>${fmt(p.vegetacao_densa_pct)}% do bairro tem vegetação densa${p.hotspot_pct > 0 ? ` e ${fmt(p.hotspot_pct)}% está entre os pontos mais quentes da cidade` : ""}.</p>` : ""}
    ${areas.length ? `<p>O mapa indica <b>${areas.length}</b> ${areas.length > 1 ? "lugares públicos" : "lugar público"} (praças e calçadas) onde plantar árvores pode deixar o entorno mais fresco. Eles aparecem em verde no mapa.</p>` : ""}</div>`;
}

/* ---------------- confiabilidade e sobre ---------------- */
function preencherConfiabilidade() {
  const m = S.meta, r = m.modelo.resumo, v = m.verificacao.final || m.verificacao, c = m.composicao;
  $("#conf-cards").innerHTML = [
    ["hl", `${fmt(r.mae_c.media, 2)} ± ${fmt(r.mae_c.dp, 2)} °C`, "erro médio em cidades não vistas no treino (5 folds)"],
    ["hl", `${fmt(c.mae_c, 2)} °C`, `erro em Teresina, composição de ${c.cenas.length} cenas secas (modelo sem Teresina no treino)`],
    ["", fmt(v.pearson_anomalia[0], 2), "correlação com a temperatura observada (folds)"],
    ["", fmt(c.pearson, 2), "correlação em Teresina (composição)"],
    ["", `${fmt(r.hotspot_mae_c.media, 2)} °C`, "erro médio nos pontos mais quentes"],
    ["", `±${fmt(c.incerteza_media_c, 2)} °C`, "incerteza média em Teresina (desacordo entre os 5 modelos)"],
  ].map(([k, a, b]) => `<div class="card ${k}"><div class="v">${a}</div><div class="k">${b}</div></div>`).join("");
  const ref = [["UrbanHeatNet", v.oraculo], ["Regressão linear", v.linear_oraculo], ["Média da cena", v.media_cena], ["Persistência (usa o mês anterior)", v.persistencia]];
  chartRef = new Chart($("#chart-ref"), {
    type: "bar",
    data: { labels: ref.map((x) => x[0]), datasets: [{ data: ref.map((x) => x[1][0]), backgroundColor: ["#1f7a4d", "#9fb7a8", "#9fb7a8", "#d8d2bf"] }] },
    options: { indexAxis: "y", responsive: true, maintainAspectRatio: false, animation: false, plugins: { legend: { display: false },
      tooltip: { callbacks: { label: (c) => `${fmt(ref[c.dataIndex][1][0], 2)} ± ${fmt(ref[c.dataIndex][1][1], 2)} °C` } } },
      scales: { x: { beginAtZero: true, title: { display: true, text: "MAE (°C)" } } } },
  });
  const cidade = (c) => c.replace(/_([A-Z]{2})$/, " ($1)").replace(/_/g, " ");
  $("#tab-folds tbody").innerHTML = m.modelo.folds.map((f, i) => `<tr><td>${i + 1}</td><td>${f.val_cities.map((c) => esc(cidade(c))).join(", ")}</td>
    <td class="num">${fmt(f.mae_c, 2)}</td><td class="num">${fmt(f.hotspot_mae_c, 2)}</td><td class="num">${fmt(f.iou, 2)}</td></tr>`).join("");
  const ad = m.areas?.aditividade;
  $("#limitacoes").innerHTML = [
    "A temperatura é a da superfície (chão, telhados, copas), medida por satélite às 10h da manhã, e não a temperatura do ar.",
    "O modelo prevê quanto cada ponto fica acima ou abaixo da média da própria cidade. Por isso o resfriamento simulado é local: mostra onde a arborização refresca, não quanto cai a média de Teresina.",
    "A persistência (repetir o mês anterior) erra menos, mas precisa da temperatura medida; o modelo prevê só com vegetação, construções, água e relevo, o que permite simular cenários.",
    "O efeito vale para árvores adultas, com copa formada, o que leva de 5 a 10 anos.",
    "O espaço para plantio foi estimado com mapas abertos (OpenStreetMap e Overture). Fiação, largura real da calçada, tubulações e propriedade dos lotes precisam ser verificadas em campo.",
    ad ? `Áreas vizinhas não somam integralmente: plantando juntas as ${ad.n_areas} áreas de maior efeito, o resultado foi ${fmt(ad.razao_conjunto_soma * 100)}% da soma dos efeitos individuais; os totais do plano usam essa razão.` : null,
    `A imagem cobre ${m.bairros.n_com_metricas} dos ${m.bairros.n_bairros} bairros; o restante será incluído numa próxima versão.`,
    "Áreas hachuradas no mapa (água, encostas e falhas fixas do dado de satélite) não recebem previsão nem plantio.",
  ].filter(Boolean).map((t) => `<li>${t}</li>`).join("");
}
function preencherSobre() {
  const m = S.meta;
  $("#fontes").innerHTML = m.fontes.map((f) => `<li>${esc(f)}</li>`).join("") +
    `<li>Composição da estação seca: mediana de ${m.composicao.cenas.length} cenas sem nuvens (${esc(m.composicao.cenas.map((c) => c.replace(".npy", "").replace("_", "/")).join(", "))}).</li>`;
  const p = m.premissas;
  $("#premissas-lista").innerHTML = [
    `Pixels de ${p.resolucao_m} m; em cada pixel só a fração livre (calçada, praça ou lote sem construção) recebe árvores.`,
    `Largura das pistas: ${esc(p.largura_pista_m)}; faixa de calçada de ${fmt(p.faixa_calcada_m, 1)} m de cada lado.`,
    `Pixels com NDVI acima de ${fmt(p.ndvi_ja_vegetado, 1)} já são considerados vegetados.`,
    `Excluídos: água (NDWI ≥ ${fmt(p.agua_ndwi_min, 1)}) e declividade acima de ${p.declividade_max_graus}°.`,
    "Plantar = aproximar os índices do pixel (NDVI, NDBI, NDWI) da assinatura da vegetação densa de Teresina, na proporção da fração plantada. Nenhuma temperatura é editada.",
    "Resfriamento com espalhamento para a vizinhança (média de vizinhanças de 150 m e 450 m).",
  ].map((t) => `<li>${t}</li>`).join("");
}

/* ---------------- link do plano (hash) ---------------- */
function escreverHash() {
  const h = new URLSearchParams({ e: S.estrategia, n: S.orcArvores, t: [...S.tipos].join(","), m2: S.m2 });
  if (S.bairro) h.set("b", S.bairro);
  if (S.filtroEquip) h.set("eq", S.dist);
  if (S.filtroInc) h.set("inc", 1);
  if (S.custoArvore != null) h.set("c", S.custoArvore);
  if (S.custoManut != null) h.set("cm", S.custoManut);
  if (S.estrategia === "manual") h.set("a", [...S.manual].join(","));
  history.replaceState(null, "", "#" + h.toString());
}
function lerHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  const tot = S.areas.filter((f) => S.tipos.has(f.properties.tipo)).reduce((s, f) => s + arvoresDe(f.properties), 0);
  S.orcArvores = Math.round(tot * 0.1 / 100) * 100;
  if (!h.has("e")) return;
  S.estrategia = h.get("e") in ESTRATEGIAS || h.get("e") === "manual" ? h.get("e") : "eficiencia";
  S.orcArvores = +h.get("n") || S.orcArvores;
  if (h.has("t")) S.tipos = new Set(h.get("t").split(",").filter((t) => t in TIPOS));
  S.m2 = +h.get("m2") || 36;
  S.bairro = h.get("b") || "";
  if (h.has("eq")) { S.filtroEquip = true; S.dist = +h.get("eq") || 300; }
  S.filtroInc = h.has("inc");
  if (h.has("c")) S.custoArvore = +h.get("c");
  if (h.has("cm")) S.custoManut = +h.get("cm");
  if (h.has("a")) S.manual = new Set(h.get("a").split(",").map(Number).filter(Boolean));
}

carregar().catch((e) => {
  document.querySelector("main").innerHTML = `<p style="padding:20px">Não foi possível carregar os dados (${esc(e.message)}). Abra a página por um servidor (por exemplo, GitHub Pages ou <code>python -m http.server</code> na pasta docs).</p>`;
});
