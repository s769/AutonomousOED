/* Cascadia OED SC26 demo — live repair on the held GPU allocation. */

const DATA = window.DEMO_DATA_URL.replace(/\/$/, "");

const state = {
  sensors: [],
  order: [],
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
  poll: null,
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
  btnRepair: document.getElementById("btnRepair"),
  btnPrepare: document.getElementById("btnPrepare"),
  btnSelect: document.getElementById("btnSelect"),
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

function setHeatUrl(url, meanStd) {
  if (!state.frameMeta) {
    showHeat(false);
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

function drawSelection({ hideNew } = {}) {
  selLayer.clearLayers();
  const k = state.k;
  const sc = state.scenario;
  const repair = state.repair;
  const lost = sc ? sc.reference_hit || [] : [];

  if (state.view === "original") {
    const n = Math.min(k, state.order.length);
    for (let i = 0; i < n; i++) addSensor(state.order[i], "#e4572e", 5);
    if (sc && k >= state.order.length) markLost(lost);
    return;
  }

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

function setK(k, { fromSlider } = {}) {
  state.k = k;
  els.kLabel.textContent = String(k);
  if (!fromSlider) els.kSlider.value = String(k);
  updateHeat(k);
  drawSelection();
}

function play() {
  if (state.playing) return;
  state.playing = true;
  els.btnPlay.disabled = true;
  els.btnPause.disabled = false;
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
  pause();
  stopReveal();
  stopPoll();
  state.scenario = sc;
  state.repair = null;
  state.view = "original";
  setStage("B");
  els.btnRepair.disabled = state.repairing;
  if (els.btnPrepare) els.btnPrepare.disabled = state.repairing;
  if (els.btnSelect) els.btnSelect.disabled = true;
  els.compareRow.hidden = true;
  els.repairMetrics.hidden = true;
  els.origNote.hidden = true;
  els.precomputedNote.hidden = true;
  if (els.repairHeadline) els.repairHeadline.hidden = true;
  if (!state.repairing) els.repairProgress.hidden = true;
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

function showLiveRepair(repair) {
  state.repair = repair;
  setStage("C");
  els.repairProgress.hidden = true;
  els.compareRow.hidden = false;
  els.repairMetrics.hidden = false;
  els.btnRepair.disabled = false;
  if (els.btnPrepare) els.btnPrepare.disabled = false;
  if (els.btnSelect) els.btnSelect.disabled = true;
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
  fillRepairMetrics(repair, `Live on ${hw}.`);
  applyCompare("repaired", { animate: true });
  setStatus(headline);
}

function stopPoll() {
  if (state.poll) {
    clearInterval(state.poll);
    state.poll = null;
  }
}

function renderProgress(st) {
  els.repairProgress.hidden = false;
  const k = st.k != null ? st.k : 0;
  const budget = st.budget || 175;
  let step = "Running on GPUs…";
  if (st.state === "queued") step = st.message || "Waiting for GPUs…";
  else if (st.state === "preparing") step = "Loading K and factorizing…";
  else if (st.state === "ready") step = "Ready — press Start selection";
  else if (st.state === "selecting" || k) step = `Selecting ${k} / ${budget}`;
  else if (st.message && !/^srun |^====/.test(st.message)) step = st.message;
  els.repairStep.textContent = step;
  els.repairElapsed.textContent = fmtElapsed(st.elapsed_s);
  els.repairBar.style.width = `${Math.min(100, (100 * k) / budget)}%`;
  if (els.btnSelect) {
    els.btnSelect.disabled = st.state !== "ready";
  }
  if (st.log_tail) {
    const clean = st.log_tail
      .split("\n")
      .map((ln) => ln.replace(/[█▉▊▋▌▍▎▏\u2588-\u2593]+/g, "").trim())
      .filter((ln) => ln && !ln.startsWith("srun") && !ln.startsWith("===="));
    els.repairLog.textContent = clean.slice(-6).join("\n");
  }
}

async function fetchStatus() {
  const r = await fetch("/api/repair/status", { cache: "no-store" });
  if (!r.ok) throw new Error(`status ${r.status}`);
  return r.json();
}

function beginRepairUi(statusMsg, stepMsg) {
  pause();
  state.repairing = true;
  els.btnRepair.disabled = true;
  if (els.btnPrepare) els.btnPrepare.disabled = true;
  if (els.btnSelect) els.btnSelect.disabled = true;
  els.compareRow.hidden = true;
  els.repairMetrics.hidden = true;
  if (els.repairHeadline) els.repairHeadline.hidden = true;
  els.repairProgress.hidden = false;
  els.repairLog.textContent = "";
  els.repairBar.style.width = "0%";
  els.repairStep.textContent = stepMsg;
  els.repairElapsed.textContent = "0s";
  setStage("C");
  setStatus(statusMsg);
  try {
    els.btnRepair.scrollIntoView({ block: "center", behavior: "smooth" });
  } catch (e) {}
}

function failRepair(err) {
  state.repairing = false;
  els.btnRepair.disabled = false;
  if (els.btnPrepare) els.btnPrepare.disabled = false;
  if (els.btnSelect) els.btnSelect.disabled = true;
  els.repairStep.textContent = "Repair failed";
  els.repairLog.textContent = String(err);
  setStatus(`Repair failed: ${err}`);
}

function pollRepair() {
  stopPoll();
  const tick = async () => {
    try {
      const st = await fetchStatus();
      renderProgress(st);
      if (st.state === "ready") {
        setStatus("K loaded and factorized — start selection when ready");
        if (els.btnSelect) els.btnSelect.disabled = false;
        return;
      }
      if (st.state === "done" && st.result) {
        stopPoll();
        state.repairing = false;
        showLiveRepair(st.result);
        return;
      }
      if (st.state === "error") {
        stopPoll();
        failRepair(st.error || "Repair failed");
      }
    } catch (e) {
      stopPoll();
      failRepair(e.message || e);
    }
  };
  state.poll = setInterval(tick, 800);
  tick();
}

async function startRepair(mode) {
  if (!state.scenario || state.repairing) return;
  const prepare = mode === "prepare";
  beginRepairUi(
    prepare
      ? "Loading K and factorizing retained sensors…"
      : "Repair running on the GPU allocation…",
    prepare ? "Loading K…" : "Submitting to GPUs…"
  );

  try {
    const r = await fetch("/api/repair", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scenario: state.scenario.id, mode: mode || "all" }),
    });
    const body = await r.json().catch(() => ({}));
    if (!r.ok) {
      throw new Error(body.error || `HTTP ${r.status}`);
    }
  } catch (e) {
    failRepair(e.message || e);
    return;
  }
  pollRepair();
}

async function startSelection() {
  if (!state.scenario) return;
  if (els.btnSelect) els.btnSelect.disabled = true;
  setStatus("Selection running…");
  els.repairStep.textContent = "Starting selection…";
  try {
    const r = await fetch("/api/repair/go", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scenario: state.scenario.id }),
    });
    const body = await r.json().catch(() => ({}));
    if (!r.ok) {
      throw new Error(body.error || `HTTP ${r.status}`);
    }
  } catch (e) {
    failRepair(e.message || e);
    return;
  }
  pollRepair();
}

function resetDemo() {
  pause();
  stopReveal();
  stopPoll();
  state.scenario = null;
  state.repair = null;
  state.view = "original";
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
  if (!state.repairing) {
    els.repairProgress.hidden = true;
    els.btnRepair.disabled = true;
    if (els.btnPrepare) els.btnPrepare.disabled = true;
    if (els.btnSelect) els.btnSelect.disabled = true;
  }
  [...els.scenarioButtons.querySelectorAll("button")].forEach((b) =>
    b.classList.remove("selected")
  );
  if (framed) frameMap();
  setK(0);
  setStatus("Ready — play the reference, then choose a branch restriction");
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
els.btnRepair.addEventListener("click", () => startRepair("all"));
if (els.btnPrepare) {
  els.btnPrepare.addEventListener("click", () => startRepair("prepare"));
}
if (els.btnSelect) {
  els.btnSelect.addEventListener("click", startSelection);
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
    state.order = reference.order;
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
    for (const f of state.frames) {
      const img = new Image();
      img.src = `${DATA}/${f.png}`;
    }

    const scenList = await Promise.all(
      (manifest.scenarios || []).map((p) => loadJSON(p))
    );

    frameMap();
    drawCandidates();
    setK(0);

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
        setStatus(
          `Ready · GPU job ${alloc.jobid} (${alloc.nodes} node${alloc.nodes === 1 ? "" : "s"})`
        );
      } else {
        setStatus("Ready — play the reference, then choose a branch restriction");
      }
    } catch (e) {
      setStatus("Ready — play the reference, then choose a branch restriction");
    }
  } catch (e) {
    setStatus(`Failed to load demo data: ${e.message}`);
    console.error(e);
  }
})();
