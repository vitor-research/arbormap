/* ARBORMAP: modo "desenhar área livre". O gestor desenha um polígono; só a parte plantável dele
   (praça, calçada ou lote livre, fora de água, encosta e áreas já vegetadas) recebe árvores, e a
   UrbanHeatNet roda no navegador (onnxruntime-web, WebGPU quando houver) sobre a cidade inteira.
   Mesma conta de arbormap.decision.areas_candidatas.Simulador.effect. */

const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.webgpu.min.js";
const MOD = DATA + "modelo/";
const DES = { pts: [], linha: null, poly: null, ativo: false, sess: null, ep: null, x: null, yBase: null,
              pl: null, valido: null, info: null, overlay: null, carregando: null };

function carregarScript(src) {
  return new Promise((ok, erro) => { const s = document.createElement("script"); s.src = src; s.onload = ok; s.onerror = () => erro(new Error("falha ao carregar " + src)); document.head.appendChild(s); });
}
const buscar = (f, tipo) => fetch(MOD + f).then((r) => { if (!r.ok) throw new Error(f); return tipo === "json" ? r.json() : r.arrayBuffer(); });

async function carregarModelo(status) {
  if (DES.sess) return;
  if (DES.carregando) return DES.carregando;
  DES.carregando = (async () => {
    status("Baixando o modelo (26 MB, só na primeira vez)...");
    if (!window.ort) await carregarScript(ORT_URL);
    const [modelo, x, pl, val, info] = await Promise.all([buscar("urbanheatnet_fp16.onnx"), buscar("entrada.bin"), buscar("plantavel.bin"), buscar("valido.bin"), buscar("sim.json", "json")]);
    DES.x = new Float32Array(x); DES.pl = new Uint8Array(pl); DES.valido = new Uint8Array(val); DES.info = info;
    status("Preparando o modelo...");
    const eps = navigator.gpu ? ["webgpu", "wasm"] : ["wasm"];
    for (const ep of eps) {
      try { DES.sess = await ort.InferenceSession.create(modelo, { executionProviders: [ep] }); DES.ep = ep; break; }
      catch (e) { console.warn("ARBORMAP: " + ep + " indisponível", e); }
    }
    if (!DES.sess) throw new Error("o navegador não conseguiu executar o modelo");
    status(DES.ep === "webgpu" ? "Calculando o cenário atual (placa de vídeo)..." : "Calculando o cenário atual (sem placa de vídeo: cerca de 30 s)...");
    DES.yBase = await prever(DES.x);
  })();
  try { await DES.carregando; } finally { DES.carregando = null; }
}

async function prever(x) {
  const [, c, h, w] = DES.info.shape;
  const r = await DES.sess.run({ x: new ort.Tensor("float32", x, [1, c, h, w]) });
  return Float32Array.from(r.y.data);
}

/* ---- desenho no mapa ---- */
function iniciarDesenho() {
  limparDesenho();
  DES.ativo = true;
  map.getContainer().classList.add("desenhando");
  map.doubleClickZoom.disable();
  map.on("click", adicionarPonto);
  map.on("dblclick", concluirDesenho);
  atualizarBotoesDesenho();
  $("#d-status").textContent = "Clique no mapa para marcar os cantos da área. Clique duas vezes (ou em Concluir) para fechar.";
}
function adicionarPonto(e) {
  DES.pts.push(e.latlng);
  if (DES.linha) map.removeLayer(DES.linha);
  DES.linha = L.polyline(DES.pts, { color: "#111", weight: 2, dashArray: "4 4" }).addTo(map);
  atualizarBotoesDesenho();
}
function concluirDesenho() {
  if (!DES.ativo) return;
  map.off("click", adicionarPonto); map.off("dblclick", concluirDesenho);
  map.getContainer().classList.remove("desenhando");
  setTimeout(() => map.doubleClickZoom.enable(), 300);
  DES.ativo = false;
  if (DES.linha) { map.removeLayer(DES.linha); DES.linha = null; }
  const pts = DES.pts.filter((p, i) => i === 0 || !p.equals(DES.pts[i - 1]));
  DES.pts = pts;
  if (pts.length < 3) { $("#d-status").textContent = "Marque pelo menos 3 pontos."; DES.pts = []; atualizarBotoesDesenho(); return; }
  DES.poly = L.polygon(pts, { color: "#111", weight: 2, fillOpacity: 0.05 }).addTo(map);
  const f = fracaoDesenho();
  const ha = f.soma * DES_PIXEL_HA();
  $("#d-status").textContent = ha > 0
    ? `Área desenhada com ${fmt(ha, 2)} ha plantáveis (cerca de ${fmt(ha * 10000 / S.m2)} árvores). Clique em Simular.`
    : "A área desenhada não tem espaço plantável nas classes escolhidas (ou já é vegetada). Desenhe outra.";
  atualizarBotoesDesenho(ha > 0);
}
function limparDesenho() {
  [DES.linha, DES.poly, DES.overlay].forEach((l) => l && map.removeLayer(l));
  DES.linha = DES.poly = DES.overlay = null; DES.pts = [];
  if (DES.ativo) { map.off("click", adicionarPonto); map.off("dblclick", concluirDesenho); map.getContainer().classList.remove("desenhando"); map.doubleClickZoom.enable(); DES.ativo = false; }
  $("#d-resultado").innerHTML = ""; $("#d-status").textContent = "";
  atualizarBotoesDesenho();
}
function atualizarBotoesDesenho(podeSimular = false) {
  $("#d-iniciar").disabled = DES.ativo;
  $("#d-concluir").disabled = !(DES.ativo && DES.pts.length >= 3);
  $("#d-simular").disabled = !podeSimular;
}
const DES_PIXEL_HA = () => (DES.info ? DES.info.pixel_ha : 0.09);

/* fração plantável dentro do polígono, pixel a pixel (centro do pixel dentro do polígono) */
function fracaoDesenho() {
  const N = 400, [[s, w], [n, e]] = S.meta.bounds_leaflet;
  const frac = new Float32Array(N * N);
  if (!DES.pts.length) return { frac, soma: 0 };
  const poly = DES.pts.map((p) => [(p.lng - w) / (e - w) * N - 0.5, (n - p.lat) / (n - s) * N - 0.5]);
  const xs = poly.map((p) => p[0]), ys = poly.map((p) => p[1]);
  const c0 = Math.max(0, Math.floor(Math.min(...xs))), c1 = Math.min(N - 1, Math.ceil(Math.max(...xs)));
  const r0 = Math.max(0, Math.floor(Math.min(...ys))), r1 = Math.min(N - 1, Math.ceil(Math.max(...ys)));
  const classes = $$(".d-classe").map((c) => c.checked);
  const pl = DES.pl;
  let soma = 0;
  for (let r = r0; r <= r1; r++) for (let c = c0; c <= c1; c++) {
    let dentro = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const [xi, yi] = poly[i], [xj, yj] = poly[j];
      if ((yi > r) !== (yj > r) && c < ((xj - xi) * (r - yi)) / (yj - yi) + xi) dentro = !dentro;
    }
    if (!dentro) continue;
    const k = r * N + c;
    let f = 0;
    if (pl) classes.forEach((on, ci) => { if (on) f += pl[ci * N * N + k] / 255; });
    else f = 1;
    f = Math.min(1, f);
    frac[k] = f; soma += f;
  }
  return { frac, soma };
}

/* desfoque gaussiano separável, como cv2.GaussianBlur (borda BORDER_REFLECT_101) */
function gauss(src, N, k, sigma) {
  const h = (k - 1) / 2, w = [];
  let t = 0;
  for (let i = -h; i <= h; i++) { const v = Math.exp(-(i * i) / (2 * sigma * sigma)); w.push(v); t += v; }
  for (let i = 0; i < w.length; i++) w[i] /= t;
  const refl = (i) => (i < 0 ? -i : i >= N ? 2 * N - 2 - i : i);
  const tmp = new Float32Array(N * N), out = new Float32Array(N * N);
  for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) { let s = 0; for (let d = -h; d <= h; d++) s += w[d + h] * src[r * N + refl(c + d)]; tmp[r * N + c] = s; }
  for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) { let s = 0; for (let d = -h; d <= h; d++) s += w[d + h] * tmp[refl(r + d) * N + c]; out[r * N + c] = s; }
  return out;
}

async function efeito(frac) {
  const N = 400, NN = N * N, info = DES.info;
  const xp = Float32Array.from(DES.x);
  for (const [k, alvo] of Object.entries(info.alvo_normalizado)) {
    const o = +k * NN;
    for (let i = 0; i < NN; i++) if (frac[i] > 0) xp[o + i] += frac[i] * (alvo - xp[o + i]);
  }
  const y = await prever(xp);
  const raw = new Float32Array(NN);
  for (let i = 0; i < NN; i++) raw[i] = (DES.yBase[i] - y[i]) * info.lst_dp_c;
  const g5 = gauss(raw, N, 5, 1.0), g15 = gauss(raw, N, 15, 3.0);
  const cool = new Float32Array(NN);
  let sw = 0, sc = 0, n05 = 0, integ = 0;
  for (let i = 0; i < NN; i++) {
    const v = 0.7 * g5[i] + 0.3 * g15[i];
    cool[i] = v;
    if (frac[i] > 0) { sw += frac[i]; sc += v * frac[i]; }
    if (DES.valido[i]) { if (v >= 0.5) n05++; if (v > 0) integ += v; }
  }
  return { cool, resfriamento_c: sc / Math.max(sw, 1e-9), area_05c_ha: n05 * info.pixel_ha, resfriamento_int: integ * info.pixel_ha };
}

function overlayResfriamento(cool) {
  const N = 400, cv = document.createElement("canvas");
  cv.width = cv.height = N;
  const ctx = cv.getContext("2d"), img = ctx.createImageData(N, N);
  for (let i = 0; i < N * N; i++) {
    const v = cool[i];
    if (v < 0.1) continue;
    const t = Math.min(1, (v - 0.1) / 1.9);
    const [r, g, b] = corEscala("YlGn", 0.25 + 0.75 * t).match(/\w\w/g).map((h) => parseInt(h, 16));
    img.data.set([r, g, b, Math.round(120 + 135 * t)], i * 4);
  }
  ctx.putImageData(img, 0, 0);
  if (DES.overlay) map.removeLayer(DES.overlay);
  DES.overlay = L.imageOverlay(cv.toDataURL(), S.meta.bounds_leaflet, { className: "pix", opacity: 0.9 }).addTo(map);
}

async function simularDesenho() {
  const st = (t) => ($("#d-status").textContent = t);
  $("#d-simular").disabled = true;
  try {
    await carregarModelo(st);
    const { frac, soma } = fracaoDesenho();
    if (!soma) { st("Sem espaço plantável na área."); return; }
    st("Simulando o plantio...");
    const t0 = performance.now();
    const r = await efeito(frac);
    const seg = (performance.now() - t0) / 1000;
    overlayResfriamento(r.cool);
    const ha = soma * DES.info.pixel_ha, arv = Math.max(1, Math.round((ha * 10000) / S.m2));
    const custo = custoUnit() ? arv * custoUnit() : null;
    const vr = S.meta.validacao_real?.parques;
    const fracMedia = soma / Math.max(1, frac.reduce((n, f) => n + (f > 0), 0));
    const semEfeito = r.resfriamento_c < 0.05;
    $("#d-resultado").innerHTML = (semEfeito ? `<p class="note warn">O modelo não detecta resfriamento neste plantio: em média só ${fmt(fracMedia * 100)}% de cada pixel de 30 m pode receber árvores aqui, uma mudança pequena demais para a resolução do satélite. Inclua praças ou áreas livres maiores.</p>` : "") + `<div class="cards">
      <div class="card hl"><div class="v">${fmt(r.resfriamento_c, 2)} °C</div><div class="k">resfriamento médio no local plantado</div></div>
      <div class="card"><div class="v">${fmt(arv)}</div><div class="k">árvores em ${fmt(ha, 2)} ha plantáveis</div></div>
      <div class="card"><div class="v">${fmt(r.area_05c_ha, 1)} ha</div><div class="k">ficam ao menos 0,5 °C mais frescos</div></div>
      <div class="card"><div class="v">${fmt(r.resfriamento_int, 1)}</div><div class="k">resfriamento integrado (°C·ha)</div></div>
      <div class="card"><div class="v">${custo != null ? "R$ " + fmt(custo) : "–"}</div><div class="k">${custo != null ? "custo estimado" : "informe o custo em \"Custos e premissas\""}</div></div>
      <div class="card"><div class="v">${fmt(seg, 1)} s</div><div class="k">tempo no seu navegador (${DES.ep === "webgpu" ? "placa de vídeo" : "processador"})</div></div></div>
      <p class="small">Em verde no mapa: onde a superfície fica ao menos 0,1 °C mais fresca, com as copas formadas.${vr ? ` Nos ${vr.n_parques} parques existentes, o modelo previu em média ${fmt(vr.pred_media_c, 2)} °C de frescor contra ${fmt(vr.obs_media_c, 2)} °C observados: trate o valor como um limite superior.` : ""}</p>`;
    st("Pronto. Desenhe outra área ou ajuste as classes e simule de novo.");
  } catch (e) {
    console.error(e);
    st("Não foi possível simular neste navegador (" + e.message + "). Use o Chrome ou o Edge atualizados.");
  } finally { $("#d-simular").disabled = !DES.poly; }
}

function iniciarControlesDesenho() {
  $("#d-iniciar").onclick = iniciarDesenho;
  $("#d-concluir").onclick = concluirDesenho;
  $("#d-limpar").onclick = limparDesenho;
  $("#d-simular").onclick = simularDesenho;
  $$(".d-classe").forEach((c) => (c.onchange = () => { if (DES.poly) { const f = fracaoDesenho(); $("#d-status").textContent = `${fmt(f.soma * DES_PIXEL_HA(), 2)} ha plantáveis com as classes escolhidas.`; atualizarBotoesDesenho(f.soma > 0); } }));
  // a fração plantável é leve (0,5 MB): carrega já, para medir a área desenhada antes do modelo
  Promise.all([buscar("plantavel.bin"), buscar("sim.json", "json")]).then(([pl, info]) => { DES.pl = new Uint8Array(pl); DES.info = DES.info || info; }).catch(() => {});
  atualizarBotoesDesenho();
}
