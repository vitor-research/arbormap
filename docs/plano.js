/* ARBORMAP: plano de arborização em página própria (aberta pelo botão "Gerar plano completo").
   Recebe o plano da página principal (window.opener ou localStorage) e monta o documento para a
   prefeitura: mapa do trecho, resumo, custos, efeito ao longo do crescimento, cenários, espécies,
   áreas com coordenadas, bairros atendidos, confiabilidade e fontes. */

const DATA = "data/";
const $ = (s, r = document) => r.querySelector(s);
const fmt = (v, d = 0) => (v == null || !isFinite(v) ? "–" : Number(v).toLocaleString("pt-BR", { minimumFractionDigits: d, maximumFractionDigits: d }));
const sinal = (v, d = 1) => (v > 0 ? "+" : "") + fmt(v, d);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const brl = (v) => "R$ " + fmt(v, 0);

const TIPOS = { praca: "Praça/parque", calcada: "Calçada/canteiro", livre: "Área livre", desenho: "Área desenhada" };
const EQUIP = { escola: "escola", creche: "creche", saude: "unidade de saúde" };
const NOME_PORTE = { pequeno: "pequeno", medio: "médio", grande: "grande" };
const ESPAC = { pequeno: 5, medio: 8, grande: 12 };          // Manual de Fortaleza (2020), p. 68
const FAIXA_CALCADA_M = 2.5;
// Premissa (sem fonte local): anos até a copa adulta, por porte. Editável na página.
const ANOS_COPA = { pequeno: 5, medio: 8, grande: 12 };
// Rega: 10 a 20 L por muda, 3 vezes por semana, nos 3 primeiros meses (Manual de Fortaleza, p. 93)
const REGA = { litros_min: 10, litros_max: 20, vezes_semana: 3, semanas: 13 };

function porteDe(tipo, porte) {
  if (tipo === "calcada") return porte === "pequeno" || porte === "misto" ? "pequeno" : "medio";
  return porte === "misto" ? "medio" : porte;
}
function arvoresPorHa(tipo, porte) {
  const s = ESPAC[porteDe(tipo, porte)];
  return tipo === "calcada" ? 10000 / (FAIXA_CALCADA_M * s) : 10000 / (s * s);
}
const fracaoEfeito = (t, T) => Math.min(1, (t / T) ** 2);   // área de copa ~ (diâmetro)², diâmetro cresce ~ linear

let P, M, B, EQ, ESP, R = 1, chartTempo;

function lerPlano() {
  try { if (window.opener && window.opener.ARBORMAP_PLANO) return window.opener.ARBORMAP_PLANO; } catch (e) { /* outra origem */ }
  try { const s = localStorage.getItem("arbormap_plano"); if (s) return JSON.parse(s); } catch (e) { /* sem armazenamento */ }
  return null;
}

async function iniciar() {
  P = lerPlano();
  if (!P) { $("#p-corpo").innerHTML = `<section><h2>Nenhum plano encontrado</h2><p>Abra o <a href="index.html">mapa</a>, monte um plano e clique em "Gerar plano completo".</p></section>`; return; }
  const get = (f) => fetch(DATA + f).then((r) => (r.ok ? r.json() : null)).catch(() => null);
  [M, B, EQ, ESP] = await Promise.all([get("meta.json"), get("teresina_bairros.geojson"), get("teresina_equipamentos.geojson"), get("especies.json")]);
  R = M.areas?.aditividade?.razao_conjunto_soma ?? 1;
  $("#p-voltar").href = P.link || "index.html";
  $("#p-sub").textContent = `Teresina (PI) · gerado em ${new Date(P.gerado_em).toLocaleString("pt-BR")}`;
  const areas = P.origem === "desenho" ? [areaDoDesenho(P.desenho)] : P.areas.map((f) => ({ ...f, properties: { ...f.properties } }));
  areas.forEach((f, i) => {
    const p = f.properties;
    p.n = i + 1;
    p.porte = porteDe(p.tipo === "desenho" ? "praca" : p.tipo, P.params.porte);
    if (p.arvores == null) p.arvores = Math.max(1, Math.round(p.area_plantavel_ha * arvoresPorHa(p.tipo, P.params.porte)));
    if (!p.centro) p.centro = centroide(f.geometry);
  });
  montar(areas);
}

function areaDoDesenho(d) {
  const ring = d.pts.map(([lat, lon]) => [lon, lat]);
  ring.push(ring[0]);
  const geometry = { type: "Polygon", coordinates: [ring] };
  const c = centroide(geometry);
  const bairro = B.features.find((b) => dentroGeom(c, b.geometry));
  let prox = null;
  EQ?.features.forEach((e) => {
    const [lon, lat] = e.geometry.coordinates, dm = distM(c, [lon, lat]);
    if (!prox || dm < prox.dist_m) prox = { nome: e.properties.nome, categoria: e.properties.categoria, dist_m: Math.round(dm) };
  });
  return { type: "Feature", geometry, properties: { id: "D", tipo: "desenho", bairro: bairro?.properties.bairro || null, centro: c,
    area_plantavel_ha: d.area_plantavel_ha, arvores: d.arvores, resfriamento_c: d.resfriamento_c, area_05c_ha: d.area_05c_ha,
    resfriamento_int: d.resfriamento_int, equipamento_proximo: prox, lst_media_c: null, incerteza_c: null } };
}

/* ---------- geometria simples ---------- */
function aneis(g) { return g.type === "Polygon" ? [g.coordinates[0]] : g.coordinates.map((p) => p[0]); }
function centroide(g) {
  let sx = 0, sy = 0, n = 0;
  aneis(g).forEach((r) => r.forEach(([x, y]) => { sx += x; sy += y; n++; }));
  return [sx / n, sy / n];
}
function dentroAnel([x, y], r) {
  let d = false;
  for (let i = 0, j = r.length - 1; i < r.length; j = i++) {
    const [xi, yi] = r[i], [xj, yj] = r[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) d = !d;
  }
  return d;
}
const dentroGeom = (pt, g) => (g.type === "Polygon" ? [g.coordinates] : g.coordinates).some((pol) => dentroAnel(pt, pol[0]));
const distM = ([x1, y1], [x2, y2]) => Math.hypot((x1 - x2) * Math.cos((y1 * Math.PI) / 180), y1 - y2) * 111320;

/* ---------- cálculos ---------- */
function resumo(areas, porte, custoUnit) {
  let arv = 0, ha = 0, integ = 0, a05 = 0, dt = 0;
  areas.forEach((f) => {
    const p = f.properties, n = p.tipo === "desenho" ? p.arvores : Math.max(1, Math.round(p.area_plantavel_ha * arvoresPorHa(p.tipo, porte)));
    arv += n; ha += p.area_plantavel_ha; integ += p.resfriamento_int; a05 += p.area_05c_ha; dt += p.resfriamento_c * p.area_plantavel_ha;
  });
  const r = areas.length > 1 ? Math.min(R, 1) : 1;
  return { arv, ha, integ: integ * r, a05: a05 * r, dt: ha ? dt / ha : 0, custo: custoUnit ? arv * custoUnit : null,
           agua_m3: [arv * REGA.litros_min * REGA.vezes_semana * REGA.semanas / 1000, arv * REGA.litros_max * REGA.vezes_semana * REGA.semanas / 1000] };
}
function curvaTempo(areas, porte, anos = 20) {
  const pts = [];
  const tot = areas.reduce((s, f) => s + f.properties.area_plantavel_ha, 0);
  for (let t = 0; t <= anos; t++) {
    let dt = 0, integ = 0;
    areas.forEach((f) => {
      const p = f.properties, T = ANOS_COPA[porteDe(p.tipo === "desenho" ? "praca" : p.tipo, porte)], k = fracaoEfeito(t, T);
      dt += p.resfriamento_c * p.area_plantavel_ha * k; integ += p.resfriamento_int * k;
    });
    pts.push({ t, dt: tot ? dt / tot : 0, integ: integ * (areas.length > 1 ? Math.min(R, 1) : 1) });
  }
  return pts;
}
function anoPara(curva, frac) {
  const alvo = curva[curva.length - 1].dt * frac;
  const p = curva.find((c) => c.dt >= alvo - 1e-9);
  return p ? p.t : null;
}

/* ---------- montagem ---------- */
function montar(areas) {
  const pr = P.params, ref = M.custos_referencia?.opcoes || [];
  const sinapi = Object.fromEntries(ref.map((o) => [o.codigo, o.valor]));
  const custoUnit = pr.custoArvore ? pr.custoArvore + (pr.custoManut || 0) : null;
  const rs = resumo(areas, pr.porte, custoUnit);
  const bairros = bairrosAtendidos(areas);
  const equipPerto = equipamentosPerto(areas, 300);
  const curva = curvaTempo(areas, pr.porte);
  const a50 = anoPara(curva, 0.5), a90 = anoPara(curva, 0.9);
  const vuln = bairros.reduce((s, b) => s + (b.vulneraveis || 0), 0), mor = bairros.reduce((s, b) => s + (b.populacao || 0), 0);
  const titulo = P.origem === "desenho" ? "Área desenhada no mapa" : `${fmt(areas.length)} áreas escolhidas pela estratégia "${nomeEstrategia(pr.estrategia)}"`;

  $("#p-corpo").innerHTML = `
  <section>
    <h2>Resumo</h2>
    <p>${esc(titulo)}${pr.bairro ? `, no bairro ${esc(pr.bairro)}` : ""}. Porte das árvores: ${esc(nomePorte(pr.porte))}.</p>
    <div class="cards">
      ${card("hl", fmt(rs.arv), "árvores a plantar")}
      ${card("", fmt(rs.ha, 1) + " ha", "área plantada")}
      ${card("", fmt(bairros.length), "bairros atendidos")}
      ${card("", fmt(equipPerto.length), "escolas, creches e unidades de saúde a até 300 m")}
      ${card("hl", fmt(rs.dt, 2) + " °C", "resfriamento médio da superfície no local plantado, com as copas adultas")}
      ${card("", fmt(rs.a05, 0) + " ha", "ficam ao menos 0,5 °C mais frescos")}
      ${card("", a50 != null ? `${a50} anos` : "–", "para metade do efeito (premissa de crescimento abaixo)")}
      ${card("", a90 != null ? `${a90} anos` : "–", "para 90% do efeito")}
      ${card("hl", rs.custo != null ? brl(rs.custo) : (sinapi["98511"] ? brl(rs.arv * sinapi["98511"]) : "–"),
             rs.custo != null ? "custo informado (plantio" + (pr.custoManut ? " + manutenção)" : ")") : "plantio com muda de 2 a 4 m (SINAPI 98511, Teresina)")}
      ${card("", `${fmt(rs.agua_m3[0])} a ${fmt(rs.agua_m3[1])} m³`, "de água para a rega dos 3 primeiros meses")}
      ${card("", fmt(mor), "moradores nos bairros atendidos (Censo 2022)")}
      ${card("", fmt(vuln), "crianças de 0 a 4 anos e pessoas com 60 anos ou mais nesses bairros")}
    </div>
    <p class="small">Temperaturas da superfície (satélite Landsat, estação seca). Os totais com várias áreas aplicam a razão medida entre o efeito conjunto e a soma dos individuais (${fmt(R, 2)}). Nos parques que já existem, o modelo previu mais frescor do que o satélite mede: trate os valores como limite superior.</p>
  </section>

  <section>
    <h2>Mapa do trecho planejado</h2>
    <canvas id="mapa-print" class="mapa-print"></canvas>
    <div class="legenda-print"><span>Temperatura da superfície em relação à média da cidade:</span>
      <span class="bar" style="background:linear-gradient(to right,#313695,#74add1,#ffffbf,#fdae61,#a50026)"></span>
      <span>${sinal(M.camadas.anomalia.min, 0)} a ${sinal(M.camadas.anomalia.max, 0)} °C</span>
      <span><b style="color:#0b6b3a">■</b> áreas do plano (número = linha da tabela)</span><span>Mapa de fundo © OpenStreetMap</span></div>
  </section>

  <section>
    <h2>Quando o efeito aparece</h2>
    <p>O resfriamento calculado vale para as copas adultas. Ele cresce com a área de sombra: como o diâmetro da copa aumenta de forma aproximadamente constante, a área coberta cresce com o quadrado do tempo até a copa ficar adulta. No teste com o satélite, a temperatura respondeu de forma proporcional à vegetação (NDVI), o que sustenta escalar o efeito pela cobertura.</p>
    <div id="premissas-tempo">
      ${Object.keys(ANOS_COPA).map((k) => `<label class="premissa">Porte ${NOME_PORTE[k]}: copa adulta em <input type="number" min="2" max="40" data-porte="${k}" value="${ANOS_COPA[k]}"> anos</label>`).join("")}
    </div>
    <p class="aviso">Premissa sem fonte local: o tempo até a copa adulta depende da espécie, da muda, da rega e do solo. Substitua pelos dados do viveiro municipal ou de plantios anteriores; os gráficos e o resumo se ajustam.</p>
    <div class="chartbox" style="height:260px"><canvas id="chart-tempo"></canvas></div>
    <table id="tab-tempo"></table>
  </section>

  <section>
    <h2>Custos</h2>
    <table>
      <thead><tr><th>Item</th><th>Base</th><th class="num">Unitário</th><th class="num">Total</th></tr></thead>
      <tbody>
        ${ref.map((o) => `<tr><td>Plantio, SINAPI ${esc(o.codigo)}</td><td>${esc(o.descricao)}</td><td class="num">${brl(o.valor)}</td><td class="num">${brl(o.valor * rs.arv)}</td></tr>`).join("")}
        ${pr.custoArvore ? `<tr><td>Plantio (valor informado no mapa)</td><td>informado pelo gestor</td><td class="num">${brl(pr.custoArvore)}</td><td class="num">${brl(pr.custoArvore * rs.arv)}</td></tr>` : ""}
        <tr><td>Manutenção (2 a 3 anos)</td><td>${pr.custoManut ? "informada pelo gestor" : "sem referência no SINAPI: informar"}</td><td class="num">${pr.custoManut ? brl(pr.custoManut) : "–"}</td><td class="num">${pr.custoManut ? brl(pr.custoManut * rs.arv) : "–"}</td></tr>
        <tr><td>Água da rega (3 primeiros meses)</td><td>10 a 20 L por muda, 3 vezes por semana (Manual de Fortaleza, p. 93)</td><td class="num">${fmt(REGA.litros_min * REGA.vezes_semana * REGA.semanas)} a ${fmt(REGA.litros_max * REGA.vezes_semana * REGA.semanas)} L</td><td class="num">${fmt(rs.agua_m3[0])} a ${fmt(rs.agua_m3[1])} m³</td></tr>
      </tbody>
    </table>
    <p class="small">${esc(M.custos_referencia?.fonte || "")}; ${esc(M.custos_referencia?.base || "")}. ${esc(M.custos_referencia?.observacao || "")} O manual de Fortaleza pede muda com fuste de pelo menos 1,80 m (p. 14), o que corresponde à composição 98511. Não incluídos: adubação, tutor (mínimo 2,30 m), protetor (até 1,20 m), reposição de mudas perdidas e eventual obra na calçada.</p>
  </section>

  <section>
    <h2>Cenários e recomendações</h2>
    <p>As mesmas áreas, com escolhas diferentes de porte e de muda. O resfriamento com as copas adultas é o mesmo em todos (a área coberta é a mesma); muda o número de árvores, o custo e o tempo até o efeito.</p>
    <div class="tablewrap"><table class="cenarios" id="tab-cenarios"></table></div>
    <h3>Recomendação</h3>
    <ul id="recomendacoes"></ul>
  </section>

  <section>
    <h2>Espécies nativas recomendadas</h2>
    <div id="especies"></div>
  </section>

  <section>
    <h2>Regras de plantio</h2>
    <ul id="regras"></ul>
  </section>

  <section>
    <h2>Áreas do plano</h2>
    <div class="tablewrap"><table id="tab-areas-plano"></table></div>
    <p class="small">Coordenadas do centro de cada área. Conferir em campo: largura real da calçada, fiação, tubulações, acesso e propriedade (áreas livres podem ser privadas).</p>
  </section>

  <section>
    <h2>Bairros atendidos</h2>
    <div class="tablewrap"><table id="tab-bairros-plano"></table></div>
  </section>

  <section>
    <h2>Confiabilidade e limitações</h2>
    <ul id="confiab"></ul>
  </section>

  <section>
    <h2>Fontes</h2>
    <ul id="fontes-plano" class="small"></ul>
    <p class="small">Plano reproduzível pelo link: <a href="${esc(P.link)}">${esc(P.link)}</a></p>
  </section>`;

  desenharMapa(areas);
  atualizarTempo(areas);
  $$("#premissas-tempo input").forEach((i) => (i.oninput = () => { ANOS_COPA[i.dataset.porte] = Math.max(2, +i.value || ANOS_COPA[i.dataset.porte]); atualizarTempo(areas); }));
  cenarios(areas, sinapi);
  especies(areas);
  regras();
  tabelaAreas(areas);
  tabelaBairros(bairros);
  confiabilidade();
  fontes();
}
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const card = (k, v, t) => `<div class="card ${k}"><div class="v">${v}</div><div class="k">${t}</div></div>`;
const nomeEstrategia = (e) => ({ eficiencia: "maior resfriamento por árvore", multicriterio: "multicritério do modelo", quente: "mais quentes primeiro", manual: "seleção manual" }[e] || e);
const nomePorte = (p) => ({ misto: "misto (praças e lotes: médio; calçadas: pequeno)", pequeno: "pequeno", medio: "médio", grande: "grande nas praças e lotes, médio nas calçadas" }[p] || p);

function bairrosAtendidos(areas) {
  const m = new Map();
  areas.forEach((f) => {
    const nome = f.properties.bairro;
    if (!nome) return;
    const b = B.features.find((x) => x.properties.bairro === nome)?.properties;
    const r = m.get(nome) || { bairro: nome, areas: 0, arvores: 0, ...(b ? { anomalia_c: b.anomalia_c, populacao: b.populacao, vulneraveis: b.vulneraveis, vegetacao_densa_pct: b.vegetacao_densa_pct } : {}) };
    r.areas++; r.arvores += f.properties.arvores;
    m.set(nome, r);
  });
  return [...m.values()].sort((a, b) => b.arvores - a.arvores);
}
function equipamentosPerto(areas, raio) {
  const out = new Map();
  (EQ?.features || []).forEach((e) => {
    const c = e.geometry.coordinates;
    let dmin = Infinity;
    areas.forEach((f) => { dmin = Math.min(dmin, distM(c, f.properties.centro)); });
    if (dmin <= raio) out.set(e.properties.nome + c.join(), { ...e.properties, dist_m: Math.round(dmin) });
  });
  return [...out.values()];
}

/* ---------- mapa estático (impressão) ---------- */
function desenharMapa(areas) {
  const cv = $("#mapa-print"), ctx = cv.getContext("2d");
  let w = Infinity, s = Infinity, e = -Infinity, n = -Infinity;
  areas.forEach((f) => aneis(f.geometry).forEach((r) => r.forEach(([x, y]) => { w = Math.min(w, x); e = Math.max(e, x); s = Math.min(s, y); n = Math.max(n, y); })));
  const mx = Math.max((e - w) * 0.12, 0.004), my = Math.max((n - s) * 0.12, 0.004);
  w -= mx; e += mx; s -= my; n += my;
  const W = 1000;
  const X = (lon, z) => ((lon + 180) / 360) * 256 * 2 ** z;
  const Y = (lat, z) => { const r = (lat * Math.PI) / 180; return ((1 - Math.log(Math.tan(r) + 1 / Math.cos(r)) / Math.PI) / 2) * 256 * 2 ** z; };
  let z = 18;
  while (z > 10 && X(e, z) - X(w, z) > W) z--;
  const x0 = X(w, z), y0 = Y(n, z), wpx = X(e, z) - x0, hpx = Y(s, z) - y0;
  const esc2 = W / wpx;
  cv.width = W; cv.height = Math.round(hpx * esc2);
  const px = ([lon, lat]) => [(X(lon, z) - x0) * esc2, (Y(lat, z) - y0) * esc2];
  ctx.fillStyle = "#eef0ec"; ctx.fillRect(0, 0, cv.width, cv.height);
  const tiles = [];
  for (let tx = Math.floor(x0 / 256); tx <= Math.floor((x0 + wpx) / 256); tx++)
    for (let ty = Math.floor(y0 / 256); ty <= Math.floor((y0 + hpx) / 256); ty++) tiles.push([tx, ty]);
  const carregar = (src) => new Promise((ok) => { const im = new Image(); im.crossOrigin = "anonymous"; im.onload = () => ok(im); im.onerror = () => ok(null); im.src = src; });
  Promise.all(tiles.map(([tx, ty]) => carregar(`https://tile.openstreetmap.org/${z}/${tx}/${ty}.png`).then((im) => [tx, ty, im]))).then(async (ims) => {
    ctx.filter = "grayscale(0.8) brightness(1.05)";
    ims.forEach(([tx, ty, im]) => { if (im) ctx.drawImage(im, (tx * 256 - x0) * esc2, (ty * 256 - y0) * esc2, 256 * esc2 + 1, 256 * esc2 + 1); });
    ctx.filter = "none";
    const an = await carregar(DATA + M.camadas.anomalia.arquivo);
    if (an) {
      const [[bs, bw], [bn, be]] = M.bounds_leaflet;
      const [ax0, ay0] = px([bw, bn]), [ax1, ay1] = px([be, bs]);
      ctx.globalAlpha = 0.5; ctx.drawImage(an, ax0, ay0, ax1 - ax0, ay1 - ay0); ctx.globalAlpha = 1;
    }
    ctx.lineWidth = 1; ctx.strokeStyle = "rgba(40,40,40,.45)"; ctx.setLineDash([3, 3]);
    B.features.forEach((b) => aneis(b.geometry).forEach((r) => { ctx.beginPath(); r.forEach((c, i) => { const [a, bb] = px(c); i ? ctx.lineTo(a, bb) : ctx.moveTo(a, bb); }); ctx.stroke(); }));
    ctx.setLineDash([]);
    areas.forEach((f) => {
      aneis(f.geometry).forEach((r) => { ctx.beginPath(); r.forEach((c, i) => { const [a, bb] = px(c); i ? ctx.lineTo(a, bb) : ctx.moveTo(a, bb); }); ctx.closePath();
        ctx.fillStyle = "rgba(11,107,58,.7)"; ctx.fill(); ctx.strokeStyle = "#06331c"; ctx.lineWidth = 1.2; ctx.stroke(); });
      const [a, bb] = px(f.properties.centro);
      ctx.font = "bold 12px system-ui, sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.lineWidth = 3; ctx.strokeStyle = "#fff"; ctx.strokeText(String(f.properties.n), a, bb - 10); ctx.fillStyle = "#06331c"; ctx.fillText(String(f.properties.n), a, bb - 10);
    });
  });
}

/* ---------- tempo ---------- */
function atualizarTempo(areas) {
  const c = curvaTempo(areas, P.params.porte);
  const ds = [{ label: "Resfriamento médio no local (°C)", data: c.map((p) => p.dt), borderColor: "#1f7a4d", backgroundColor: "#1f7a4d", yAxisID: "y", tension: 0.2, pointRadius: 2 },
              { label: "Resfriamento integrado (°C·ha)", data: c.map((p) => p.integ), borderColor: "#2a6fbf", backgroundColor: "#2a6fbf", yAxisID: "y2", borderDash: [5, 4], tension: 0.2, pointRadius: 0 }];
  if (chartTempo) { chartTempo.data.datasets = ds; chartTempo.update("none"); }
  else chartTempo = new Chart($("#chart-tempo"), { type: "line", data: { labels: c.map((p) => p.t), datasets: ds },
    options: { responsive: true, maintainAspectRatio: false, animation: false, plugins: { legend: { labels: { boxWidth: 10, font: { size: 11 } } } },
      scales: { x: { title: { display: true, text: "anos após o plantio" } }, y: { title: { display: true, text: "°C" }, beginAtZero: true },
                y2: { position: "right", title: { display: true, text: "°C·ha" }, beginAtZero: true, grid: { drawOnChartArea: false } } } } });
  const marcos = [1, 2, 3, 5, 8, 10, 12, 15, 20];
  $("#tab-tempo").innerHTML = `<thead><tr><th>Ano</th>${marcos.map((t) => `<th class="num">${t}</th>`).join("")}</tr></thead><tbody>
    <tr><td>°C no local</td>${marcos.map((t) => `<td class="num">${fmt(c[t].dt, 2)}</td>`).join("")}</tr>
    <tr><td>% do efeito</td>${marcos.map((t) => `<td class="num">${fmt((100 * c[t].dt) / (c[20].dt || 1))}%</td>`).join("")}</tr></tbody>`;
}

/* ---------- cenários ---------- */
function cenarios(areas, sinapi) {
  const linhas = [
    { nome: "Menor custo de implantação", porte: "grande", muda: "98510", obs: "menos árvores (espaçamento de 12 m nas praças) e muda menor; efeito demora mais e a muda pequena pede mais cuidado" },
    { nome: "Padrão técnico (recomendado)", porte: "misto", muda: "98511", rec: true, obs: "médio porte nas praças e lotes, pequeno nas calçadas; muda com fuste de 1,80 m, como pede o manual" },
    { nome: "Sombra mais rápida", porte: "pequeno", muda: "98511", obs: "mais árvores (5 m), copa adulta mais cedo; maior custo e mais mudas para regar" },
    { nome: "Menor manutenção", porte: "medio", muda: "98511", obs: "médio porte em todos os locais (menos mudas para regar que o pequeno) e espécies da Caatinga, adaptadas à seca; nas calçadas exige 2,00 m de largura e sem fiação" },
  ];
  const rows = linhas.map((l) => {
    const rs = resumo(areas, l.porte, sinapi[l.muda]);
    const c = curvaTempo(areas, l.porte);
    return { ...l, rs, a50: anoPara(c, 0.5), a90: anoPara(c, 0.9) };
  });
  $("#tab-cenarios").innerHTML = `<thead><tr><th>Cenário</th><th class="num">Árvores</th><th class="num">Plantio (SINAPI)</th><th class="num">Água 3 meses</th><th class="num">50% do efeito</th><th class="num">90%</th><th>Observação</th></tr></thead><tbody>` +
    rows.map((r) => `<tr><td class="${r.rec ? "rec" : ""}"><b>${esc(r.nome)}</b><br><span class="small">muda ${r.muda}</span></td><td class="num">${fmt(r.rs.arv)}</td>
      <td class="num">${r.rs.custo != null ? brl(r.rs.custo) : "–"}</td><td class="num">${fmt(r.rs.agua_m3[0])}–${fmt(r.rs.agua_m3[1])} m³</td>
      <td class="num">${r.a50 ?? "–"} anos</td><td class="num">${r.a90 ?? "–"} anos</td><td class="small">${esc(r.obs)}</td></tr>`).join("") + "</tbody>";
  const tipos = new Set(areas.map((f) => f.properties.tipo));
  const rec = [
    "Prefira o cenário padrão técnico: o manual de Fortaleza adota como muda padrão a de fuste com pelo menos 1,80 m e copa formada (p. 14), que corresponde à composição SINAPI 98511.",
    `Para gastar menos com rega e reposição, use espécies que ocorrem naturalmente na Caatinga e no Piauí (marcadas abaixo) e plante em dezembro ou janeiro: assim os 3 meses de rega obrigatória caem na estação chuvosa${chuvaTexto()}.`,
    tipos.has("calcada") ? "Nas calçadas, confira a largura e a fiação antes de escolher o porte: abaixo de 1,50 m não plante árvores; sob rede elétrica, só pequeno porte." : null,
    tipos.has("praca") ? "Nas praças, médio e grande porte dão mais sombra por árvore e precisam de menos mudas; combine com pequeno porte nas bordas para sombra mais rápida." : null,
    tipos.has("livre") ? "Áreas livres podem ser privadas: confirme a propriedade antes de incluir no orçamento público." : null,
    "Comece pelas áreas perto de escolas, creches e unidades de saúde e nos bairros com mais crianças pequenas e idosos (tabela de bairros abaixo).",
  ].filter(Boolean);
  $("#recomendacoes").innerHTML = rec.map((t) => `<li>${t}</li>`).join("");
}

function chuvaTexto() {
  const c = M.validacao_inmet?.chuva_media_mensal_mm;
  if (!c) return "";
  const m = ["", "jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"];
  return ` (chuva média na estação do INMET em Teresina, 2018 a 2024: ${[12, 1, 2, 3, 4].map((k) => `${m[k]} ${fmt(c[k])} mm`).join(", ")}; de junho a outubro, ${fmt(Math.min(...[6, 7, 8, 9, 10].map((k) => c[k])))} a ${fmt(Math.max(...[6, 7, 8, 9, 10].map((k) => c[k])))} mm por mês)`;
}

/* ---------- espécies ---------- */
function especies(areas) {
  if (!ESP) { $("#especies").innerHTML = "<p>Lista de espécies indisponível.</p>"; return; }
  const ok = (e) => e.caatinga_ou_cerrado && e.ocorre_no_piaui && e.origem === "native";
  const tag = (d) => `<span class="tag ${d === "Caatinga" ? "caatinga" : d === "Cerrado" ? "cerrado" : "mata"}">${esc(d)}</span>`;
  const item = (e) => `<li><b>${esc(e.nome_popular)}</b> <i>${esc(e.nome_cientifico)}</i> ${e.dominios.filter((d) => ["Caatinga", "Cerrado"].includes(d)).map(tag).join("")}</li>`;
  const por = (p) => ESP.especies.filter((e) => e.porte === p && ok(e)).sort((a, b) => a.nome_popular.localeCompare(b.nome_popular, "pt-BR"));
  const caat = (p) => por(p).filter((e) => e.dominios.includes("Caatinga")).length;
  const tipos = new Set(areas.map((f) => f.properties.tipo));
  const blocos = [];
  if (tipos.has("calcada") || tipos.has("desenho")) blocos.push(["Calçadas e canteiros estreitos (pequeno porte; a única opção sob fiação)", por("pequeno")]);
  blocos.push(["Praças, canteiros largos e áreas livres (médio porte)", por("medio")]);
  if (tipos.has("praca") || tipos.has("livre") || tipos.has("desenho")) blocos.push(["Praças e parques com espaço (grande porte, espaçamento de 12 m)", por("grande")]);
  $("#especies").innerHTML = `<p>Espécies da lista oficial do Manual de Arborização Urbana de Fortaleza (2020), filtradas pela lista oficial da flora brasileira (Flora e Funga do Brasil): só as <b>nativas da Caatinga ou do Cerrado que ocorrem no Piauí</b>. As da Caatinga vêm do clima semiárido; são as primeiras a considerar quando o objetivo é reduzir a rega, o que deve ser confirmado com o viveiro municipal.</p>` +
    blocos.map(([t, l]) => `<h3>${esc(t)} · ${l.length} espécies</h3><ul class="especies">${l.map(item).join("")}</ul>`).join("") +
    `<p class="small">Para Teresina, a referência local é Machado et al. (2006), "Árvores nativas para a arborização de Teresina, Piauí" (REVSBAU 1(1): 10-18), ainda não conferida no texto completo. Antes de comprar, confirme a disponibilidade das mudas no viveiro municipal.</p>`;
}
function regras() {
  const r = ESP?.regras;
  if (!r) return;
  $("#regras").innerHTML = [
    `Calçada mínima: ${esc(r.calcada_minima)}.`,
    `Pequeno porte: ${esc(r.calcada_largura_m.pequeno)}; médio: ${esc(r.calcada_largura_m.medio)}; grande: ${esc(r.calcada_largura_m.grande)}.`,
    `Fiação: ${esc(r.sob_fiacao)}.`,
    `Espaçamento mínimo entre árvores: ${r.espacamento_m.pequeno} m (pequeno), ${r.espacamento_m.medio} m (médio), ${r.espacamento_m.grande} m (grande).`,
    `Distâncias: ${Object.entries(r.distancias_m).map(([p, d]) => `${NOME_PORTE[p]} porte a ${d.esquina} m da esquina, ${d.poste} m de postes e ${d.transformador} m de transformadores`).join("; ")}.`,
    `Muda: ${esc(r.muda_padrao)}.`, `Rega: ${esc(r.rega)}.`, `Tutor: ${esc(r.tutor)}. Protetor: ${esc(r.protetor)}.`,
    `Fonte: ${esc(r.fonte)}, p. 14, 67 a 73, 87, 88 e 93. Teresina deve conferir com as normas municipais vigentes.`,
  ].map((t) => `<li>${t}</li>`).join("");
}

/* ---------- tabelas ---------- */
function tabelaAreas(areas) {
  $("#tab-areas-plano").innerHTML = `<thead><tr><th>Nº</th><th>Tipo</th><th>Bairro</th><th class="num">Árvores</th><th>Porte</th><th class="num">ha</th><th class="num">ΔT local</th><th class="num">≥ 0,5 °C</th><th>Mais perto de</th><th>Localização</th></tr></thead><tbody>` +
    areas.map((f) => {
      const p = f.properties, [lon, lat] = p.centro, eq = p.equipamento_proximo;
      return `<tr><td>${p.n}</td><td>${TIPOS[p.tipo]}</td><td>${esc(p.bairro || "–")}</td><td class="num">${fmt(p.arvores)}</td><td>${NOME_PORTE[p.porte]}</td>
        <td class="num">${fmt(p.area_plantavel_ha, 2)}</td><td class="num">${fmt(p.resfriamento_c, 2)} °C</td><td class="num">${fmt(p.area_05c_ha, 1)} ha</td>
        <td>${eq ? `${EQUIP[eq.categoria]} (${fmt(eq.dist_m)} m): ${esc(eq.nome)}` : "–"}</td>
        <td><a href="https://www.google.com/maps?q=${lat.toFixed(6)},${lon.toFixed(6)}" target="_blank" rel="noopener">${lat.toFixed(5)}, ${lon.toFixed(5)}</a></td></tr>`;
    }).join("") + "</tbody>";
}
function tabelaBairros(bs) {
  $("#tab-bairros-plano").innerHTML = `<thead><tr><th>Bairro</th><th class="num">Áreas</th><th class="num">Árvores</th><th class="num">°C acima da média</th><th class="num">Moradores</th><th class="num">0-4 anos e 60+</th><th class="num">Vegetação densa</th></tr></thead><tbody>` +
    bs.map((b) => `<tr><td>${esc(b.bairro)}</td><td class="num">${b.areas}</td><td class="num">${fmt(b.arvores)}</td><td class="num">${b.anomalia_c == null ? "–" : sinal(b.anomalia_c)}</td>
      <td class="num">${fmt(b.populacao)}</td><td class="num">${fmt(b.vulneraveis)}</td><td class="num">${b.vegetacao_densa_pct == null ? "–" : fmt(b.vegetacao_densa_pct) + "%"}</td></tr>`).join("") + "</tbody>";
}
function confiabilidade() {
  const r = M.modelo.resumo, c = M.composicao, v = M.validacao_real, i = M.validacao_inmet;
  $("#confiab").innerHTML = [
    `Erro médio do modelo em cidades que ele não viu no treino: ${fmt(r.mae_c.media, 2)} ± ${fmt(r.mae_c.dp, 2)} °C; em Teresina, ${fmt(c.mae_c, 2)} °C (correlação ${fmt(c.pearson, 2)}).`,
    v ? `Onde a vegetação aumentou entre ${v.experimento_natural.anos_a[0]} e ${v.experimento_natural.anos_b.slice(-1)[0]}, o satélite mediu ${sinal(v.experimento_natural.grupos.ganhou_vegetacao.dT_observado_c, 2)} °C e o modelo previu ${sinal(v.experimento_natural.grupos.ganhou_vegetacao.dT_previsto_c, 2)} °C.` : null,
    v ? `Nos ${v.parques.n_parques} parques existentes, o modelo acerta quais refrescam mais (correlação ${fmt(v.parques.pearson, 2)}), mas superestima o frescor (${fmt(v.parques.pred_media_c, 2)} × ${fmt(v.parques.obs_media_c, 2)} °C): os valores do plano são um limite superior.` : null,
    i ? `A temperatura é a da superfície. Na estação do INMET em Teresina, no horário do satélite, a superfície acompanha o ar (correlação ${fmt(i.pearson_lst_estacao_x_ar, 2)} em ${i.n_dias} dias), mas na seca fica em média ${fmt(i.lst_menos_ar_seca_c, 1)} °C acima dele; ao longo da seca, cada 1 °C de superfície correspondeu a cerca de ${fmt(i.seca.inclinacao_ar_por_grau_lst, 2)} °C no ar. O frescor sentido no ar é, portanto, bem menor que o da superfície.` : null,
    "O resfriamento é local: o modelo mostra onde a arborização refresca, não quanto cai a média da cidade.",
    "Espaço plantável estimado com mapas abertos (OpenStreetMap e Overture); verificar em campo.",
  ].filter(Boolean).map((t) => `<li>${t}</li>`).join("");
}
function fontes() {
  const f = [...(M.fontes || []),
    "Custos: SINAPI (Caixa), Teresina, 09/2026, sem desoneração",
    "Espécies, porte, espaçamento, rega e regras: Prefeitura de Fortaleza (2020), Manual de Arborização Urbana de Fortaleza, SEUMA",
    "Bioma e ocorrência no Piauí: Flora e Funga do Brasil (Jardim Botânico do Rio de Janeiro), via GBIF",
    "Temperatura do ar: INMET, estação automática A312 (Teresina)"];
  $("#fontes-plano").innerHTML = f.map((t) => `<li>${esc(t)}</li>`).join("");
}

iniciar();
