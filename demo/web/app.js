/* Cascadia OED SC26 demo — live full select + repair on the held GPU allocation. */

const DATA = window.DEMO_DATA_URL.replace(/\/$/, "");

const state = {
  sensors: [],
  order: [],
  precomputedOrder: [],
  frames: [],
  frameMeta: null,
  presentation: null,
  scenarios: [],
  scenario: null,
  repair: null,
  k: 0,
  playing: false,
  timer: null,
  view: "original",
  stage: "A",
  repairing: false,
  activeJob: null, // "full" | scenario id while a live job is active
  poll: null,
  drawnK: 0,
  lastHeatPng: null,
  framesReady: false,
};

const els = {
  status: document.getElementById("status"),
  kLabel: document.getElementById("kLabel"),
  kMax: document.getElementById("kMax"),
  kSlider: document.getElementById("kSlider"),
  meanStd: document.getElementById("meanStd"),
  scenarioButtons: document.getElementById("scenarioButtons"),
  btnPlay: document.getElementById("btnPlay"),
  btnPause: document.getElementById("btnPause"),
  btnJump: document.getElementById("btnJump"),
  btnFull: document.getElementById("btnFull"),
  btnFullPrepare: document.getElementById("btnFullPrepare"),
  btnFullSelect: document.getElementById("btnFullSelect"),
  btnFullCancel: document.getElementById("btnFullCancel"),
  fullHeadline: document.getElementById("fullHeadline"),
  fullProgress: document.getElementById("fullProgress"),
  fullStep: document.getElementById("fullStep"),
  fullElapsed: document.getElementById("fullElapsed"),
  fullBar: document.getElementById("fullBar"),
  fullLog: document.getElementById("fullLog"),
  btnRepair: document.getElementById("btnRepair"),
  btnPrepare: document.getElementById("btnPrepare"),
  btnSelect: document.getElementById("btnSelect"),
  btnCancel: document.getElementById("btnCancel"),
  repairHeadline: document.getElementById("repairHeadline"),
  btnOrig: document.getElementById("btnOrig"),
  btnRemainder: document.getElementById("btnRemainder"),
  btnRepaired: document.getElementById("btnRepaired"),
  btnPresent: document.getElementById("btnPresent"),
  btnResetAll: document.getElementById("btnResetAll"),
  repairMetrics: document.getElementById("repairMetrics"),
  retainedCount: document.getElementById("retainedCount"),
  replacedCount: document.getElementById("replacedCount"),
  relOrig: document.getElementById("relOrig"),
  relRem: document.getElementById("relRem"),
  recovered: document.getElementById("recovered"),
  consequence: document.getElementById("consequence"),
  precomputedNote: document.getElementById("precomputedNote"),
  origNote: document.getElementById("origNote"),
  compareRow: document.getElementById("compareRow"),
  uqBox: document.getElementById("uqBox"),
  uqHide: document.getElementById("uqHide"),
  cmin: document.getElementById("cmin"),
  cmax: document.getElementById("cmax"),
  stageA: document.getElementById("stageA"),
  stageB: document.getElementById("stageB"),
  stageC: document.getElementById("stageC"),
  repairProgress: document.getElementById("repairProgress"),
  repairStep: document.getElementById("repairStep"),
  repairElapsed: document.getElementById("repairElapsed"),
  repairBar: document.getElementById("repairBar"),
  repairLog: document.getElementById("repairLog"),
};

function isFullJob(st) {
  return (st && st.scenario === "full") || state.activeJob === "full";
}

function progressEls(full) {
  if (full) {
    return {
      progress: els.fullProgress,
      step: els.fullStep,
      elapsed: els.fullElapsed,
      bar: els.fullBar,
      log: els.fullLog,
      selectBtn: els.btnFullSelect,
      prepareBtn: els.btnFullPrepare,
      cancelBtn: els.btnFullCancel,
      primaryBtn: els.btnFull,
    };
  }
  return {
    progress: els.repairProgress,
    step: els.repairStep,
    elapsed: els.repairElapsed,
    bar: els.repairBar,
    log: els.repairLog,
    selectBtn: els.btnSelect,
    prepareBtn: els.btnPrepare,
    cancelBtn: els.btnCancel,
    primaryBtn: els.btnRepair,
  };
}

const map = L.map("map", {
  zoomControl: true,
  attributionControl: false,
  worldCopyJump: false,
  minZoom: 5,
  maxZoom: 10,
}).setView([46.1, -126.4], 6);

const tiles = L.tileLayer(
  "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
  { maxZoom: 13, attribution: "Esri" }
);
tiles.addTo(map);

let heatLayer = null;
let revealTimer = null;
let framed = false;

const candLayer = L.layerGroup().addTo(map);
const exclLayer = L.layerGroup().addTo(map);
const polyLayer = L.layerGroup().addTo(map);
const selLayer = L.layerGroup().addTo(map);

function setStatus(msg) {
  els.status.textContent = msg;
}

function setStage(id) {
  state.stage = id;
  for (const [key, el] of [
    ["A", els.stageA],
    ["B", els.stageB],
    ["C", els.stageC],
  ]) {
    el.classList.toggle("current", key === id);
  }
}

async function loadJSON(path) {
  const r = await fetch(`${DATA}/${path}`);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}

function nearestFrame(k) {
  if (!state.frames.length) return null;
  let best = state.frames[0];
  for (const f of state.frames) {
    if (Math.abs(f.k - k) < Math.abs(best.k - k)) best = f;
  }
  return best;
}

function showHeat(on) {
  if (heatLayer) heatLayer.setOpacity(on ? 0.72 : 0);
  els.uqBox.classList.toggle("dim", !on);
  if (els.uqHide) els.uqHide.hidden = on;
}

function setHeatUrl(url, meanStd, { force } = {}) {
  if (!state.frameMeta) {
    showHeat(false);
    return;
  }
  if (!force && url === state.lastHeatPng) {
    if (meanStd != null) {
      els.meanStd.textContent = `mean ${Number(meanStd).toFixed(2)} m`;
    }
    showHeat(true);
    return;
  }
  const m = state.frameMeta;
  const bounds = [
    [m.lat_min, m.lon_min],
    [m.lat_max, m.lon_max],
  ];
  if (!heatLayer) {
    heatLayer = L.imageOverlay(url, bounds, {
      opacity: 0.72,
      interactive: false,
    });
    heatLayer.addTo(map);
  } else {
    heatLayer.setBounds(bounds);
    heatLayer.setUrl(url);
  }
  state.lastHeatPng = url;
  showHeat(true);
  els.meanStd.textContent =
    meanStd != null ? `mean ${Number(meanStd).toFixed(2)} m` : "—";
}

function updateHeat(k) {
  if (state.view === "repaired" && state.repair && state.repair.uq_available && state.repair.frame_png) {
    setHeatUrl(`${DATA}/${state.repair.frame_png}`, state.repair.uq_mean_std);
    return;
  }
  const useHeat = state.view === "original" || state.stage === "A";
  if (!useHeat) {
    showHeat(false);
    return;
  }
  const f = nearestFrame(k);
  if (!f) {
    showHeat(false);
    return;
  }
  setHeatUrl(`${DATA}/${f.png}`, f.mean_std);
}

function frameMap() {
  const m = state.frameMeta || {};
  const lats = state.sensors.map((s) => s.lat);
  const lons = state.sensors.map((s) => s.lon);
  if (!lats.length) return;
  const south = Math.min(m.lat_min != null ? m.lat_min : 90, ...lats) - 0.12;
  const north = Math.max(m.lat_max != null ? m.lat_max : -90, ...lats) + 0.12;
  const west = Math.min(m.lon_min != null ? m.lon_min : 0, ...lons) - 0.25;
  const east = Math.max(m.lon_max != null ? m.lon_max : -180, ...lons) + 0.7;
  map.fitBounds(
    [
      [south, west],
      [north, east],
    ],
    { padding: [16, 16], animate: false, maxZoom: 7 }
  );
  map.setMaxBounds([
    [south - 1.2, west - 1.8],
    [north + 1.2, east + 1.4],
  ]);
  framed = true;
  setTimeout(() => map.invalidateSize(), 50);
}

function drawCandidates() {
  candLayer.clearLayers();
  const forbidden = new Set(state.scenario ? state.scenario.exclude : []);
  for (const s of state.sensors) {
    if (forbidden.has(s.id)) continue;
    L.circleMarker([s.lat, s.lon], {
      radius: 2,
      color: "#5a6a7a",
      fillColor: "#5a6a7a",
      fillOpacity: 0.5,
      weight: 0,
    }).addTo(candLayer);
  }
}

function drawExclusion() {
  polyLayer.clearLayers();
  exclLayer.clearLayers();
  const sc = state.scenario;
  if (!sc) return;
  L.polygon(
    sc.polygon.map(([lon, lat]) => [lat, lon]),
    { color: "#e8a838", weight: 2, fillOpacity: 0.1 }
  ).addTo(polyLayer);
  const lost = new Set(sc.reference_hit || []);
  for (const id of sc.exclude) {
    if (lost.has(id)) continue;
    const s = state.sensors[id];
    if (!s) continue;
    L.circleMarker([s.lat, s.lon], {
      radius: 3.5,
      color: "#e8a838",
      fillColor: "#e8a838",
      fillOpacity: 0.22,
      weight: 1,
    }).addTo(exclLayer);
  }
}

function markLost(ids) {
  for (const id of ids) {
    const s = state.sensors[id];
    if (!s) continue;
    L.marker([s.lat, s.lon], {
      interactive: false,
      icon: L.divIcon({
        className: "",
        html: '<div style="color:#e4572e;font-weight:800;font-size:22px;line-height:1;transform:translate(-8px,-12px);text-shadow:0 0 4px #0b1015">×</div>',
      }),
    }).addTo(selLayer);
  }
}

function addSensor(id, color, radius = 6, extra = {}) {
  const s = state.sensors[id];
  if (!s) return;
  L.circleMarker([s.lat, s.lon], {
    radius,
    color: extra.stroke || "#111",
    weight: extra.weight || 1,
    fillColor: color,
    fillOpacity: 0.95,
  })
    .bindTooltip(`#${id}`)
    .addTo(selLayer);
}

function stopReveal() {
  if (revealTimer) {
    clearTimeout(revealTimer);
    revealTimer = null;
  }
}

function drawSelection({ hideNew, full } = {}) {
  const k = state.k;
  const sc = state.scenario;
  const repair = state.repair;
  const lost = sc ? sc.reference_hit || [] : [];

  if (state.view === "original") {
    const n = Math.min(k, state.order.length);
    // Incremental add during play — only 11 UQ keyframes exist, but sensors
    // advance every tick; clearing 175 markers each frame is what felt choppy.
    if (full || n < state.drawnK) {
      selLayer.clearLayers();
      state.drawnK = 0;
    }
    for (let i = state.drawnK; i < n; i++) addSensor(state.order[i], "#e4572e", 5);
    state.drawnK = n;
    if (sc && k >= state.order.length) markLost(lost);
    return;
  }

  selLayer.clearLayers();
  state.drawnK = 0;
  if (!repair || !sc) return;

  if (state.view === "remainder") {
    for (const id of repair.retained_ids) addSensor(id, "#c9d1d9", 6);
    markLost(lost);
    return;
  }

  for (const id of repair.retained_ids) addSensor(id, "#c9d1d9", 6);
  if (!hideNew) {
    for (const id of repair.replacement_ids) {
      addSensor(id, "#2a9d8f", 8, { stroke: "#7eeadc", weight: 2 });
    }
  }
  markLost(lost);
}

function revealReplacements() {
  stopReveal();
  const ids = (state.repair && state.repair.replacement_ids) || [];
  let i = 0;
  function step() {
    if (!state.repair || state.view !== "repaired") return;
    if (i >= ids.length) return;
    addSensor(ids[i], "#2a9d8f", 8, { stroke: "#7eeadc", weight: 2 });
    i += 1;
    revealTimer = setTimeout(step, 50);
  }
  step();
}

function setK(k, { fromSlider, full } = {}) {
  state.k = k;
  els.kLabel.textContent = String(k);
  if (!fromSlider) els.kSlider.value = String(k);
  updateHeat(k);
  drawSelection({ full: Boolean(full || fromSlider) });
}

function play() {
  if (state.playing) return;
  if (!state.framesReady) {
    setStatus("Still loading UQ frames…");
    return;
  }
  state.playing = true;
  els.btnPlay.disabled = true;
  els.btnPause.disabled = false;
  // One sensor per tick; frames_meta has a UQ PNG for every k=0..175.
  state.timer = setInterval(() => {
    if (state.k >= state.order.length) {
      pause();
      return;
    }
    setK(state.k + 1);
  }, 40);
}

function pause() {
  state.playing = false;
  els.btnPlay.disabled = false;
  els.btnPause.disabled = true;
  if (state.timer) clearInterval(state.timer);
  state.timer = null;
}

function fmtPct(x) {
  return `${(100 * x).toFixed(1)}%`;
}

function fmtDelta(x) {
  const ax = Math.abs(x);
  const exp = ax >= 1000 ? x.toExponential(2) : x.toFixed(1);
  return `${x >= 0 ? "+" : ""}${exp} nats`;
}

function fmtElapsed(s) {
  s = Math.max(0, Math.round(Number(s) || 0));
  const m = Math.floor(s / 60);
  const r = s % 60;
  return m ? `${m}m ${r}s` : `${r}s`;
}

function applyCompare(view, { animate } = {}) {
  stopReveal();
  state.view = view;
  els.btnOrig.classList.toggle("active", view === "original");
  els.btnRemainder.classList.toggle("active", view === "remainder");
  els.btnRepaired.classList.toggle("active", view === "repaired");
  if (view === "original") {
    setK(state.order.length);
    setStatus("Original unconstrained design (infeasible under this exclusion)");
  } else {
    updateHeat(state.order.length);
    drawSelection({ hideNew: Boolean(animate && view === "repaired") });
    if (animate && view === "repaired") revealReplacements();
    setStatus(
      view === "remainder"
        ? "Feasible remainder — lost sites not yet replaced"
        : "Repaired design — retained sensors fixed"
    );
  }
  els.origNote.hidden = view !== "original";
}

function selectScenario(sc) {
  if (state.repairing) return;
  pause();
  stopReveal();
  stopPoll();
  state.scenario = sc;
  state.repair = null;
  state.view = "original";
  state.drawnK = 0;
  state.lastHeatPng = null;
  setStage("B");
  els.btnRepair.disabled = false;
  if (els.btnPrepare) els.btnPrepare.disabled = false;
  if (els.btnSelect) els.btnSelect.disabled = true;
  setCancelVisible(false, false);
  els.compareRow.hidden = true;
  els.repairMetrics.hidden = true;
  els.origNote.hidden = true;
  els.precomputedNote.hidden = true;
  if (els.repairHeadline) els.repairHeadline.hidden = true;
  els.repairProgress.hidden = true;
  [...els.scenarioButtons.querySelectorAll("button")].forEach((b) => {
    b.classList.toggle("selected", b.dataset.id === sc.id);
  });
  drawExclusion();
  drawCandidates();
  setK(state.order.length);
  const nLost = sc.n_reference_lost;
  const label = sc.id.includes("south") ? "South" : "North";
  els.consequence.hidden = false;
  els.consequence.textContent = `${label} branch unavailable: ${nLost} selected locations must be replaced.`;
  setStatus(`${label} corridor excluded — ${nLost} original sites are infeasible`);
}

function fillRepairMetrics(repair, extraNote) {
  const ig = repair.information_gain_f;
  els.retainedCount.textContent = `${repair.n_retained} / 175`;
  els.replacedCount.textContent = String(repair.n_new);
  if (ig && ig.relative_to_original) {
    els.relRem.textContent = fmtPct(ig.relative_to_original.remainder);
    els.relOrig.textContent = fmtPct(ig.relative_to_original.repaired);
    els.recovered.textContent = fmtDelta(ig.recovered_by_replacements);
  } else {
    els.relRem.textContent = "—";
    els.relOrig.textContent = "—";
    els.recovered.textContent = "—";
  }
  if (extraNote) {
    els.precomputedNote.hidden = false;
    els.precomputedNote.textContent = extraNote;
  } else {
    els.precomputedNote.hidden = true;
  }
}

function showLiveFull(result) {
  const pe = progressEls(true);
  if (pe.progress) pe.progress.hidden = true;
  setCancelVisible(false, true);
  state.repairing = false;
  state.activeJob = null;
  setFullControlsEnabled(true);
  if (Array.isArray(result.order) && result.order.length) {
    state.order = result.order.slice();
    els.kMax.textContent = String(state.order.length);
    els.kSlider.max = String(state.order.length);
    setK(state.order.length, { full: true });
  }
  const sel =
    result.selection_seconds != null
      ? result.selection_seconds
      : result.wall_seconds;
  const n = result.n_selected != null ? result.n_selected : state.order.length;
  const headline = `Live design: ${n} sensors selected in ${fmtElapsed(sel)}`;
  if (els.fullHeadline) {
    els.fullHeadline.hidden = false;
    els.fullHeadline.textContent = headline;
  }
  const hw = result.hardware || "GPU allocation";
  setStatus(`${headline} · ${hw}`);
  setStage("A");
}

function showLiveRepair(repair) {
  state.repair = repair;
  setStage("C");
  els.repairProgress.hidden = true;
  els.compareRow.hidden = false;
  els.repairMetrics.hidden = false;
  els.btnRepair.disabled = false;
  if (els.btnPrepare) els.btnPrepare.disabled = false;
  if (els.btnSelect) els.btnSelect.disabled = true;
  setCancelVisible(false, false);
  setFullControlsEnabled(true);
  state.activeJob = null;
  const nNew = repair.n_new != null ? repair.n_new : 0;
  const sel =
    repair.selection_seconds != null
      ? repair.selection_seconds
      : repair.wall_seconds;
  const headline = `Network repaired: ${nNew} sensor${nNew === 1 ? "" : "s"} selected in ${fmtElapsed(sel)}`;
  if (els.repairHeadline) {
    els.repairHeadline.hidden = false;
    els.repairHeadline.textContent = headline;
  }
  const hw = repair.hardware || "GPU allocation";
  const note = repair.uq_note
    ? `Live on ${hw}. ${repair.uq_note}`
    : `Live on ${hw}.`;
  fillRepairMetrics(repair, note);
  applyCompare("repaired", { animate: true });
  setStatus(headline);
}

function setFullControlsEnabled(on) {
  if (els.btnFull) els.btnFull.disabled = !on;
  if (els.btnFullPrepare) els.btnFullPrepare.disabled = !on;
  if (els.btnFullSelect) els.btnFullSelect.disabled = true;
  if (els.btnPlay) els.btnPlay.disabled = !on;
}

function restorePrecomputedOrder() {
  if (!state.precomputedOrder.length) return;
  state.order = state.precomputedOrder.slice();
  els.kMax.textContent = String(state.order.length);
  els.kSlider.max = String(state.order.length);
  state.drawnK = 0;
  state.lastHeatPng = null;
  setK(0, { full: true });
}

function setCancelVisible(on, full) {
  const useFull = full == null ? isFullJob({ scenario: state.activeJob }) : full;
  const pe = progressEls(useFull);
  if (!pe.cancelBtn) return;
  pe.cancelBtn.hidden = !on;
  pe.cancelBtn.disabled = !on;
  // Hide the other job's cancel if present.
  const other = progressEls(!useFull);
  if (other.cancelBtn && other.cancelBtn !== pe.cancelBtn) {
    other.cancelBtn.hidden = true;
    other.cancelBtn.disabled = true;
  }
}

async function cancelJob() {
  if (!state.repairing) return;
  const full = state.activeJob === "full";
  const pe = progressEls(full);
  setCancelVisible(false, full);
  if (pe.step) pe.step.textContent = "Stopping…";
  setStatus("Stopping…");
  try {
    const r = await fetch("/api/repair/cancel", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    const body = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(body.error || `HTTP ${r.status}`);
  } catch (e) {
    failJob(e.message || e, full);
    return;
  }
  stopPoll();
  state.repairing = false;
  state.activeJob = null;
  if (pe.primaryBtn) pe.primaryBtn.disabled = false;
  if (pe.prepareBtn) pe.prepareBtn.disabled = false;
  if (pe.selectBtn) pe.selectBtn.disabled = true;
  if (pe.progress) pe.progress.hidden = true;
  setFullControlsEnabled(true);
  els.btnRepair.disabled = !state.scenario;
  if (els.btnPrepare) els.btnPrepare.disabled = !state.scenario;
  if (full) restorePrecomputedOrder();
  setStatus("Cancelled");
}

function stopPoll() {
  if (state.poll) {
    clearInterval(state.poll);
    state.poll = null;
  }
}

function renderProgress(st) {
  const full = isFullJob(st);
  const pe = progressEls(full);
  if (!pe.progress) return;
  pe.progress.hidden = false;
  const k = st.k != null ? st.k : 0;
  const budget = st.budget || 175;
  let step = "Running on GPUs…";
  if (st.state === "queued") step = st.message || "Waiting for GPUs…";
  else if (st.state === "preparing")
    step = full ? "Loading K…" : "Loading K and factorizing…";
  else if (st.state === "ready") step = "Ready — press Start selection";
  else if (st.state === "selecting" || k) step = `Selecting ${k} / ${budget}`;
  else if (st.message && !/^srun |^====/.test(st.message)) step = st.message;
  pe.step.textContent = step;
  pe.elapsed.textContent = fmtElapsed(st.elapsed_s);
  pe.bar.style.width = `${Math.min(100, (100 * k) / budget)}%`;
  if (pe.selectBtn) pe.selectBtn.disabled = st.state !== "ready";
  if (st.log_tail && pe.log) {
    const clean = st.log_tail
      .split("\n")
      .map((ln) => ln.replace(/[█▉▊▋▌▍▎▏\u2588-\u2593]+/g, "").trim())
      .filter((ln) => ln && !ln.startsWith("srun") && !ln.startsWith("===="));
    pe.log.textContent = clean.slice(-6).join("\n");
  }
  // Stream selected sensors onto the map during full live select.
  if (full && Array.isArray(st.order) && st.order.length) {
    state.order = st.order.slice();
    const nk = st.order.length;
    if (nk !== state.k) setK(nk);
    else {
      updateHeat(nk);
      drawSelection({});
    }
  }
}

async function fetchStatus() {
  const r = await fetch("/api/repair/status", { cache: "no-store" });
  if (!r.ok) throw new Error(`status ${r.status}`);
  return r.json();
}

function beginJobUi(opts) {
  const { full, statusMsg, stepMsg } = opts;
  pause();
  state.repairing = true;
  state.activeJob = full ? "full" : state.scenario && state.scenario.id;
  const pe = progressEls(full);
  if (pe.primaryBtn) pe.primaryBtn.disabled = true;
  if (pe.prepareBtn) pe.prepareBtn.disabled = true;
  if (pe.selectBtn) pe.selectBtn.disabled = true;
  setCancelVisible(true, full);
  if (full) {
    setFullControlsEnabled(false);
    if (els.btnFull) els.btnFull.disabled = true;
    if (els.btnFullPrepare) els.btnFullPrepare.disabled = true;
    if (els.btnPlay) els.btnPlay.disabled = true;
    if (els.fullHeadline) els.fullHeadline.hidden = true;
    els.btnRepair.disabled = true;
    if (els.btnPrepare) els.btnPrepare.disabled = true;
    if (els.btnSelect) els.btnSelect.disabled = true;
    // Clear prior design so sensors appear live as they are chosen.
    state.order = [];
    state.drawnK = 0;
    state.lastHeatPng = null;
    setK(0, { full: true });
    setStage("A");
  } else {
    if (els.btnFull) els.btnFull.disabled = true;
    if (els.btnFullPrepare) els.btnFullPrepare.disabled = true;
    if (els.btnFullSelect) els.btnFullSelect.disabled = true;
    els.compareRow.hidden = true;
    els.repairMetrics.hidden = true;
    if (els.repairHeadline) els.repairHeadline.hidden = true;
    setStage("C");
  }
  if (pe.progress) pe.progress.hidden = false;
  if (pe.log) pe.log.textContent = "";
  if (pe.bar) pe.bar.style.width = "0%";
  if (pe.step) pe.step.textContent = stepMsg;
  if (pe.elapsed) pe.elapsed.textContent = "0s";
  setStatus(statusMsg);
  try {
    if (pe.primaryBtn)
      pe.primaryBtn.scrollIntoView({ block: "center", behavior: "smooth" });
  } catch (e) {}
}

function failJob(err, full) {
  const useFull = full == null ? state.activeJob === "full" : full;
  const pe = progressEls(useFull);
  state.repairing = false;
  state.activeJob = null;
  if (pe.primaryBtn) pe.primaryBtn.disabled = false;
  if (pe.prepareBtn) pe.prepareBtn.disabled = false;
  if (pe.selectBtn) pe.selectBtn.disabled = true;
  setCancelVisible(false, useFull);
  setFullControlsEnabled(true);
  els.btnRepair.disabled = !state.scenario;
  if (els.btnPrepare) els.btnPrepare.disabled = !state.scenario;
  if (useFull) restorePrecomputedOrder();
  if (pe.step) pe.step.textContent = useFull ? "Selection failed" : "Repair failed";
  if (pe.log) pe.log.textContent = String(err);
  setStatus(`${useFull ? "Selection" : "Repair"} failed: ${err}`);
}

function pollJob() {
  stopPoll();
  const tick = async () => {
    try {
      const st = await fetchStatus();
      const full = isFullJob(st);
      renderProgress(st);
      if (st.state === "ready") {
        stopPoll();
        setStatus(
          full
            ? "K loaded — start selection when ready"
            : "K loaded and factorized — start selection when ready"
        );
        const pe = progressEls(full);
        if (pe.step) pe.step.textContent = "Ready — press Start selection";
        if (pe.selectBtn) pe.selectBtn.disabled = false;
        if (pe.prepareBtn) pe.prepareBtn.disabled = false;
        setCancelVisible(true, full);
        return;
      }
      if (st.state === "idle" || st.state === "cancelled") {
        stopPoll();
        if (state.repairing) {
          const pe = progressEls(full);
          state.repairing = false;
          state.activeJob = null;
          setCancelVisible(false, full);
          if (pe.primaryBtn) pe.primaryBtn.disabled = false;
          if (pe.prepareBtn) pe.prepareBtn.disabled = false;
          if (pe.selectBtn) pe.selectBtn.disabled = true;
          if (pe.progress) pe.progress.hidden = true;
          setFullControlsEnabled(true);
          els.btnRepair.disabled = !state.scenario;
          if (els.btnPrepare) els.btnPrepare.disabled = !state.scenario;
          if (full) restorePrecomputedOrder();
          setStatus(st.message || "Cancelled");
        }
        return;
      }
      if (st.state === "done" && st.result) {
        stopPoll();
        state.repairing = false;
        setCancelVisible(false, full);
        if (full) showLiveFull(st.result);
        else showLiveRepair(st.result);
        return;
      }
      if (st.state === "error") {
        stopPoll();
        failJob(st.error || "Job failed", full);
      }
    } catch (e) {
      stopPoll();
      failJob(e.message || e);
    }
  };
  state.poll = setInterval(tick, 800);
  tick();
}

async function startFull(mode) {
  if (state.repairing) return;
  const prepare = mode === "prepare";
  beginJobUi({
    full: true,
    statusMsg: prepare
      ? "Loading K on the GPU allocation…"
      : "Full 175-sensor selection running on the GPU allocation…",
    stepMsg: prepare ? "Loading K…" : "Submitting to GPUs…",
  });

  try {
    const r = await fetch("/api/repair", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scenario: "full", mode: mode || "all", gpus: gpuCount() }),
    });
    const body = await r.json().catch(() => ({}));
    if (!r.ok) {
      throw new Error(body.error || `HTTP ${r.status}`);
    }
  } catch (e) {
    const msg = String(e && e.message ? e.message : e);
    if (/failed to fetch|networkerror|load failed/i.test(msg)) {
      failJob(
        "Server unreachable (Failed to fetch). Check that serve_demo.py is still running on the compute node and the SSH tunnel points at that node.",
        true
      );
    } else {
      failJob(msg, true);
    }
    return;
  }
  pollJob();
}

function gpuCount() {
  const el = document.getElementById("gpuCount");
  const v = el ? parseInt(el.value, 10) : NaN;
  return Number.isFinite(v) && v > 0 ? v : null; // null = whole allocation
}

async function startRepair(mode) {
  if (!state.scenario || state.repairing) return;
  const prepare = mode === "prepare";
  beginJobUi({
    full: false,
    statusMsg: prepare
      ? "Loading K and factorizing retained sensors…"
      : "Repair running on the GPU allocation…",
    stepMsg: prepare ? "Loading K…" : "Submitting to GPUs…",
  });

  try {
    const r = await fetch("/api/repair", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scenario: state.scenario.id, mode: mode || "all", gpus: gpuCount() }),
    });
    const body = await r.json().catch(() => ({}));
    if (!r.ok) {
      throw new Error(body.error || `HTTP ${r.status}`);
    }
  } catch (e) {
    const msg = String(e && e.message ? e.message : e);
    if (/failed to fetch|networkerror|load failed/i.test(msg)) {
      failJob(
        "Server unreachable (Failed to fetch). Check that serve_demo.py is still running on the compute node and the SSH tunnel points at that node.",
        false
      );
    } else {
      failJob(msg, false);
    }
    return;
  }
  pollJob();
}

async function startSelection() {
  if (!state.activeJob && !state.scenario) return;
  const full = state.activeJob === "full";
  const pe = progressEls(full);
  if (pe.selectBtn) pe.selectBtn.disabled = true;
  setStatus("Selection running…");
  if (pe.step) pe.step.textContent = "Starting selection…";
  try {
    const r = await fetch("/api/repair/go", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        scenario: full ? "full" : state.scenario.id,
      }),
    });
    const body = await r.json().catch(() => ({}));
    if (!r.ok) {
      throw new Error(body.error || `HTTP ${r.status}`);
    }
  } catch (e) {
    failJob(e.message || e, full);
    return;
  }
  pollJob();
}

function resetDemo() {
  if (state.repairing) return;
  pause();
  stopReveal();
  stopPoll();
  state.scenario = null;
  state.repair = null;
  state.activeJob = null;
  state.view = "original";
  // Restore precomputed reference order for playback until the next live run.
  if (state.precomputedOrder.length) {
    state.order = state.precomputedOrder.slice();
  }
  setStage("A");
  polyLayer.clearLayers();
  exclLayer.clearLayers();
  drawCandidates();
  els.consequence.hidden = true;
  els.compareRow.hidden = true;
  els.repairMetrics.hidden = true;
  els.origNote.hidden = true;
  els.precomputedNote.hidden = true;
  if (els.repairHeadline) els.repairHeadline.hidden = true;
  if (els.fullHeadline) els.fullHeadline.hidden = true;
  els.repairProgress.hidden = true;
  if (els.fullProgress) els.fullProgress.hidden = true;
  els.btnRepair.disabled = true;
  if (els.btnPrepare) els.btnPrepare.disabled = true;
  if (els.btnSelect) els.btnSelect.disabled = true;
  setCancelVisible(false, false);
  setCancelVisible(false, true);
  setFullControlsEnabled(true);
  state.drawnK = 0;
  state.lastHeatPng = null;
  [...els.scenarioButtons.querySelectorAll("button")].forEach((b) =>
    b.classList.remove("selected")
  );
  if (framed) frameMap();
  setK(0, { full: true });
  setStatus("Ready — compute the reference live, then choose a branch restriction");
}

els.btnPlay.addEventListener("click", play);
els.btnPause.addEventListener("click", pause);
els.btnJump.addEventListener("click", () => {
  pause();
  setK(state.order.length);
});
els.kSlider.addEventListener("input", () => {
  pause();
  setK(Number(els.kSlider.value), { fromSlider: true });
});
if (els.btnFull) {
  els.btnFull.addEventListener("click", () => startFull("all"));
}
if (els.btnFullPrepare) {
  els.btnFullPrepare.addEventListener("click", () => startFull("prepare"));
}
if (els.btnFullSelect) {
  els.btnFullSelect.addEventListener("click", startSelection);
}
if (els.btnFullCancel) {
  els.btnFullCancel.addEventListener("click", cancelJob);
}
els.btnRepair.addEventListener("click", () => startRepair("all"));
if (els.btnPrepare) {
  els.btnPrepare.addEventListener("click", () => startRepair("prepare"));
}
if (els.btnSelect) {
  els.btnSelect.addEventListener("click", startSelection);
}
if (els.btnCancel) {
  els.btnCancel.addEventListener("click", cancelJob);
}
els.btnOrig.addEventListener("click", () => applyCompare("original"));
els.btnRemainder.addEventListener("click", () => applyCompare("remainder"));
els.btnRepaired.addEventListener("click", () => applyCompare("repaired"));
els.btnResetAll.addEventListener("click", resetDemo);
els.btnPresent.addEventListener("click", () => {
  document.body.classList.add("present");
  const root = document.documentElement;
  if (root.requestFullscreen) root.requestFullscreen().catch(() => {});
});
document.addEventListener("fullscreenchange", () => {
  if (!document.fullscreenElement) document.body.classList.remove("present");
  setTimeout(() => map.invalidateSize(), 80);
});
window.addEventListener("resize", () => map.invalidateSize());

(async function init() {
  try {
    const [manifest, presentation] = await Promise.all([
      loadJSON("manifest.json"),
      loadJSON("presentation.json"),
    ]);
    state.presentation = presentation;
    const [sensors, reference, framesMeta] = await Promise.all([
      loadJSON(manifest.sensors),
      loadJSON(manifest.reference),
      loadJSON(manifest.frames_meta),
    ]);
    state.sensors = sensors.sensors;
    state.precomputedOrder = reference.order.slice();
    state.order = reference.order.slice();
    state.frameMeta = framesMeta;
    state.frames = framesMeta.frames;
    els.kMax.textContent = String(state.order.length);
    els.kSlider.max = String(state.order.length);
    if (els.cmin && framesMeta.vmin != null) {
      els.cmin.textContent = `${framesMeta.vmin.toFixed(2)} m`;
    }
    if (els.cmax && framesMeta.vmax != null) {
      els.cmax.textContent = `${framesMeta.vmax.toFixed(2)} m`;
    }
    await Promise.all(
      state.frames.map(
        (f) =>
          new Promise((resolve) => {
            const img = new Image();
            img.onload = img.onerror = () => resolve();
            img.src = `${DATA}/${f.png}`;
          })
      )
    );
    state.framesReady = true;

    const scenList = await Promise.all(
      (manifest.scenarios || []).map((p) => loadJSON(p))
    );

    frameMap();
    drawCandidates();
    setK(0, { full: true });

    for (const sc of scenList) {
      state.scenarios.push(sc);
      const nLost = sc.n_reference_lost;
      const b = document.createElement("button");
      b.type = "button";
      b.className = "choice";
      b.dataset.id = sc.id;
      const name = sc.id.includes("south") ? "South branch" : "North branch";
      b.innerHTML = `<span>${name}</span><span class="count">${nLost} to replace</span>`;
      b.addEventListener("click", () => selectScenario(sc));
      els.scenarioButtons.appendChild(b);
    }

    try {
      const alloc = await fetch("/api/alloc", { cache: "no-store" }).then((r) =>
        r.ok ? r.json() : null
      );
      if (alloc && alloc.jobid) {
        const gi = document.getElementById("gpuCount");
        if (gi && alloc.gpus) {
          gi.max = alloc.gpus;
          gi.value = alloc.gpus;
          const gm = document.getElementById("gpuMax");
          if (gm) gm.textContent = `of ${alloc.gpus} in allocation`;
        }
        setStatus(
          `Ready · GPU job ${alloc.jobid} (${alloc.nodes} node${
            alloc.nodes === 1 ? "" : "s"
          }) — Compute 175 live`
        );
      } else {
        setStatus(
          "Ready — hold a GPU allocation, then Compute 175 (or Playback)"
        );
      }
    } catch (e) {
      setStatus(
        "Ready — hold a GPU allocation, then Compute 175 (or Playback)"
      );
    }
  } catch (e) {
    setStatus(`Failed to load demo data: ${e.message}`);
    console.error(e);
  }
})();
