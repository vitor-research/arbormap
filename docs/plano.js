/* ARBORMAP: plano de arborização em página própria (aberta pelo botão "Gerar plano completo").
   Recebe o plano da página principal (window.opener ou localStorage) e monta: mapa do trecho,
   resumo, áreas com localização, bairros atendidos, espécies nativas por local, confiabilidade e fontes. */

const DATA = "data/";
const $ = (s, r = document) => r.querySelector(s);
const fmt = (v, d = 0) => (v == null || !isFinite(v) ? "–" : Number(v).toLocaleString("pt-BR", { minimumFractionDigits: d, maximumFractionDigits: d }));
const sinal = (v, d = 1) => (v > 0 ? "+" : "") + fmt(v, d);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const card = (k, v, t) => `<div class="card ${k}"><div class="v">${v}</div><div class="k">${t}</div></div>`;

const TIPOS = { praca: "Praça/parque", calcada: "Calçada/canteiro" };
const EQUIP = { escola: "escola", creche: "creche", saude: "unidade de saúde" };

let P, M, B, EQ, ESP, R = 1;

function lerPlano() {
  try { if (window.opener && window.opener.ARBORMAP_PLANO) return window.opener.ARBORMAP_PLANO; } catch (e) { /* outra origem */ }
  try { const s = localStorage.getItem("arbormap_plano"); if (s) return JSON.parse(s); } catch (e) { /* sem armazenamento */ }
  return null;
}

async function iniciar() {
  P = lerPlano();
  if (!P || !P.areas?.length) { $("#p-corpo").innerHTML = `<section><h2>Nenhum plano encontrado</h2><p>Abra o <a href="index.html">mapa</a>, monte um plano e clique em "Gerar plano completo".</p></section>`; return; }
  const get = (f) => fetch(DATA + f).then((r) => (r.ok ? r.json() : null)).catch(() => null);
  [M, B, EQ, ESP] = await Promise.all([get("meta.json"), get("teresina_bairros.geojson"), get("teresina_equipamentos.geojson"), get("especies.json")]);
  R = Math.min(M.areas?.aditividade?.razao_conjunto_soma ?? 1, 1);
  $("#p-voltar").href = P.link || "index.html";
  $("#p-sub").textContent = `Teresina (PI) · gerado em ${new Date(P.gerado_em).toLocaleString("pt-BR")}`;
  const areas = P.areas.map((f, i) => {
    const p = { ...f.properties, n: i + 1 };
    if (!p.centro) p.centro = centroide(f.geometry);
    return { ...f, properties: p };
  });
  montar(areas);
}

/* ---------- geometria simples ---------- */
function aneis(g) { return g.type === "Polygon" ? [g.coordinates[0]] : g.coordinates.map((p) => p[0]); }
function centroide(g) {
  let sx = 0, sy = 0, n = 0;
  aneis(g).forEach((r) => r.forEach(([x, y]) => { sx += x; sy += y; n++; }));
  return [sx / n, sy / n];
}
const distM = ([x1, y1], [x2, y2]) => Math.hypot((x1 - x2) * Math.cos((y1 * Math.PI) / 180), y1 - y2) * 111320;

/* ---------- montagem ---------- */
function montar(areas) {
  let arv = 0, ha = 0, a05 = 0, dt = 0;
  areas.forEach((f) => { const p = f.properties; arv += p.arvores; ha += p.area_plantavel_ha; a05 += p.area_05c_ha; dt += p.resfriamento_c * p.area_plantavel_ha; });
  const r = areas.length > 1 ? R : 1;
  // faixa: teto = área que mais refresca; piso = teto × (frescor medido ÷ previsto nos parques existentes)
  const pq = M.validacao_real?.parques, inm = M.validacao_inmet;
  const correcao = pq ? Math.min(1, pq.obs_media_c / pq.pred_media_c) : 1;
  const teto = Math.max(...areas.map((f) => f.properties.resfriamento_c)), arK = inm?.seca?.inclinacao_ar_por_grau_lst ?? NaN;
  const fx = { teto, piso: teto * correcao, correcao, arPiso: teto * correcao * arK, arTeto: teto * arK };
  const bairros = bairrosAtendidos(areas), equipPerto = equipamentosPerto(areas, 300);
  const vuln = bairros.reduce((s, b) => s + (b.vulneraveis || 0), 0), mor = bairros.reduce((s, b) => s + (b.populacao || 0), 0);
  const pr = P.params;
  const titulo = pr.estrategia === "manual" ? `${fmt(areas.length)} áreas escolhidas no mapa` : `${fmt(areas.length)} áreas escolhidas pelo maior resfriamento por árvore`;

  $("#p-corpo").innerHTML = `
  <section>
    <h2>Resumo</h2>
    <p>${esc(titulo)}${pr.bairro ? `, no bairro ${esc(pr.bairro)}` : ""}.</p>
    <div class="cards">
      ${card("hl", fmt(arv), "árvores a plantar")}
      ${card("", fmt(ha, 1) + " ha", "área plantada")}
      ${card("hl", `${fmt(fx.piso, 1)} a ${fmt(fx.teto, 1)} °C`, `a menos na superfície nos pontos que mais se beneficiam, com as copas adultas (média das áreas: ${fmt(ha ? dt / ha : 0, 2)} °C)`)}
      ${card("", fmt(a05 * r, 0) + " ha", "ficam ao menos 0,5 °C mais frescos")}
      ${card("", fmt(bairros.length), "bairros atendidos")}
      ${card("", fmt(equipPerto.length), "escolas, creches e unidades de saúde a até 300 m")}
      ${card("", fmt(mor), "moradores nos bairros atendidos (Censo 2022)")}
      ${card("", fmt(vuln), "crianças de 0 a 4 anos e pessoas com 60 anos ou mais nesses bairros")}
    </div>
    <p class="small">Temperatura da superfície medida por satélite (estação seca). O maior valor da faixa é o que o modelo prevê para a área que mais refresca; o menor aplica a correção medida nos parques que já existem em Teresina (frescor real de ${fmt(fx.correcao * 100)}% do previsto). No ar que se respira, o efeito é menor: cerca de ${fmt(fx.arPiso, 2)} a ${fmt(fx.arTeto, 2)} °C.</p>
  </section>

  <section>
    <h2>Mapa do trecho planejado</h2>
    <canvas id="mapa-print" class="mapa-print"></canvas>
    <div class="legenda-print"><span>Temperatura da superfície em relação à média:</span>
      <span class="bar" style="background:linear-gradient(to right,#313695,#74add1,#ffffbf,#fdae61,#a50026)"></span>
      <span>${sinal(M.camadas.anomalia.min, 0)} a ${sinal(M.camadas.anomalia.max, 0)} °C</span>
      <span><b style="color:#0b6b3a">■</b> áreas do plano (número = linha da tabela)</span><span>Mapa de fundo © OpenStreetMap</span></div>
  </section>

  <section>
    <h2>Áreas do plano</h2>
    <div class="tablewrap"><table id="tab-areas-plano"></table></div>
    <p class="small">Coordenadas do centro de cada área. Conferir em campo a largura da calçada, a fiação e as tubulações.</p>
  </section>

  <section>
    <h2>Bairros atendidos</h2>
    <div class="tablewrap"><table id="tab-bairros-plano"></table></div>
  </section>

  <section>
    <h2>Espécies nativas recomendadas</h2>
    <div id="especies"></div>
  </section>

  <section>
    <h2>Confiabilidade</h2>
    <ul id="confiab"></ul>
  </section>

  <section>
    <h2>Fontes</h2>
    <ul id="fontes-plano" class="small"></ul>
    <p class="small">Plano reproduzível pelo link: <a href="${esc(P.link)}">${esc(P.link)}</a></p>
  </section>`;

  desenharMapa(areas);
  tabelaAreas(areas);
  tabelaBairros(bairros);
  especies(areas);
  confiabilidade();
  fontes();
}

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
  const k = W / wpx;
  cv.width = W; cv.height = Math.round(hpx * k);
  const px = ([lon, lat]) => [(X(lon, z) - x0) * k, (Y(lat, z) - y0) * k];
  ctx.fillStyle = "#eef0ec"; ctx.fillRect(0, 0, cv.width, cv.height);
  const tiles = [];
  for (let tx = Math.floor(x0 / 256); tx <= Math.floor((x0 + wpx) / 256); tx++)
    for (let ty = Math.floor(y0 / 256); ty <= Math.floor((y0 + hpx) / 256); ty++) tiles.push([tx, ty]);
  const carregar = (src) => new Promise((ok) => { const im = new Image(); im.crossOrigin = "anonymous"; im.onload = () => ok(im); im.onerror = () => ok(null); im.src = src; });
  const contorno = (r) => { ctx.beginPath(); r.forEach((c, i) => { const [a, b] = px(c); i ? ctx.lineTo(a, b) : ctx.moveTo(a, b); }); };
  Promise.all(tiles.map(([tx, ty]) => carregar(`https://tile.openstreetmap.org/${z}/${tx}/${ty}.png`).then((im) => [tx, ty, im]))).then(async (ims) => {
    ctx.filter = "grayscale(0.8) brightness(1.05)";
    ims.forEach(([tx, ty, im]) => { if (im) ctx.drawImage(im, (tx * 256 - x0) * k, (ty * 256 - y0) * k, 256 * k + 1, 256 * k + 1); });
    ctx.filter = "none";
    const an = await carregar(DATA + M.camadas.anomalia.arquivo);
    if (an) {
      const [[bs, bw], [bn, be]] = M.bounds_leaflet;
      const [ax0, ay0] = px([bw, bn]), [ax1, ay1] = px([be, bs]);
      ctx.globalAlpha = 0.5; ctx.drawImage(an, ax0, ay0, ax1 - ax0, ay1 - ay0); ctx.globalAlpha = 1;
    }
    ctx.lineWidth = 1; ctx.strokeStyle = "rgba(40,40,40,.45)"; ctx.setLineDash([3, 3]);
    B.features.forEach((b) => aneis(b.geometry).forEach((r) => { contorno(r); ctx.stroke(); }));
    ctx.setLineDash([]);
    areas.forEach((f) => {
      aneis(f.geometry).forEach((r) => { contorno(r); ctx.closePath(); ctx.fillStyle = "rgba(11,107,58,.7)"; ctx.fill(); ctx.strokeStyle = "#06331c"; ctx.lineWidth = 1.2; ctx.stroke(); });
      const [a, b] = px(f.properties.centro);
      ctx.font = "bold 12px system-ui, sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.lineWidth = 3; ctx.strokeStyle = "#fff"; ctx.strokeText(String(f.properties.n), a, b - 10); ctx.fillStyle = "#06331c"; ctx.fillText(String(f.properties.n), a, b - 10);
    });
  });
}

/* ---------- espécies (lista compacta) ---------- */
function especies(areas) {
  if (!ESP) { $("#especies").innerHTML = "<p>Lista de espécies indisponível.</p>"; return; }
  const ok = (e) => e.caatinga_ou_cerrado && e.ocorre_no_piaui && e.origem === "native";
  const tag = (d) => `<span class="tag ${d === "Caatinga" ? "caatinga" : "cerrado"}">${esc(d)}</span>`;
  const item = (e) => `<li><b>${esc(e.nome_popular)}</b> <i>${esc(e.nome_cientifico)}</i> ${e.dominios.filter((d) => ["Caatinga", "Cerrado"].includes(d)).map(tag).join("")}</li>`;
  const por = (...portes) => ESP.especies.filter((e) => portes.includes(e.porte) && ok(e)).sort((a, b) => a.nome_popular.localeCompare(b.nome_popular, "pt-BR"));
  const tipos = new Set(areas.map((f) => f.properties.tipo));
  const blocos = [];
  if (tipos.has("calcada")) blocos.push(["Calçadas e canteiros (pequeno porte)", por("pequeno")]);
  if (tipos.has("praca")) blocos.push(["Praças e parques (médio e grande porte)", por("medio", "grande")]);
  $("#especies").innerHTML = `<p>Árvores nativas da Caatinga ou do Cerrado que ocorrem no Piauí, da lista oficial do Manual de Arborização Urbana de Fortaleza (2020), com o bioma conferido na Flora e Funga do Brasil.</p>` +
    blocos.map(([t, l]) => `<h3>${esc(t)} · ${l.length} espécies</h3><ul class="especies">${l.map(item).join("")}</ul>`).join("") +
    `<p class="small">Antes de comprar, confirme a disponibilidade das mudas no viveiro municipal.</p>`;
}

/* ---------- tabelas ---------- */
function tabelaAreas(areas) {
  $("#tab-areas-plano").innerHTML = `<thead><tr><th>Nº</th><th>Tipo</th><th>Bairro</th><th class="num">Árvores</th><th class="num">ha</th><th class="num">ΔT no local</th><th class="num">≥ 0,5 °C</th><th>Mais perto de</th><th>Localização</th></tr></thead><tbody>` +
    areas.map((f) => {
      const p = f.properties, [lon, lat] = p.centro, eq = p.equipamento_proximo;
      return `<tr><td>${p.n}</td><td>${TIPOS[p.tipo] || esc(p.tipo)}</td><td>${esc(p.bairro || "–")}</td><td class="num">${fmt(p.arvores)}</td>
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
    `Erro médio do modelo em cidades que ele não viu no treino: ${fmt(r.mae_c.media, 2)} ± ${fmt(r.mae_c.dp, 2)} °C; em Teresina, ${fmt(c.mae_c, 2)} °C.`,
    v ? `Onde a vegetação aumentou entre ${v.experimento_natural.anos_a[0]} e ${v.experimento_natural.anos_b.slice(-1)[0]}, o satélite mediu ${sinal(v.experimento_natural.grupos.ganhou_vegetacao.dT_observado_c, 2)} °C e o modelo previu ${sinal(v.experimento_natural.grupos.ganhou_vegetacao.dT_previsto_c, 2)} °C.` : null,
    v ? `Nos ${v.parques.n_parques} parques existentes, o modelo acerta quais refrescam mais, mas superestima o frescor (${fmt(v.parques.pred_media_c, 2)} × ${fmt(v.parques.obs_media_c, 2)} °C): os valores do plano são um limite superior.` : null,
    i ? `A temperatura é a da superfície. Na estação do INMET em Teresina, na seca, a superfície fica em média ${fmt(i.lst_menos_ar_seca_c, 1)} °C acima do ar, e cada 1 °C de superfície correspondeu a cerca de ${fmt(i.seca.inclinacao_ar_por_grau_lst, 1)} °C no ar.` : null,
    "O resfriamento é local: o modelo mostra onde a arborização refresca, não quanto cai a média da cidade.",
    "O espaço para plantio foi estimado com mapas abertos (OpenStreetMap e Overture); verificar em campo.",
  ].filter(Boolean).map((t) => `<li>${t}</li>`).join("");
}
function fontes() {
  const f = [...(M.fontes || []),
    "Espécies e espaçamento: Prefeitura de Fortaleza (2020), Manual de Arborização Urbana de Fortaleza, SEUMA",
    "Bioma e ocorrência no Piauí: Flora e Funga do Brasil (Jardim Botânico do Rio de Janeiro), via GBIF",
    "Temperatura do ar: INMET, estação automática A312 (Teresina)"];
  $("#fontes-plano").innerHTML = f.map((t) => `<li>${esc(t)}</li>`).join("");
}

iniciar();
