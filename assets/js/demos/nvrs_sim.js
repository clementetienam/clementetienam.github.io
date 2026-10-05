/* nvrs_sim.js -- the simulator app on the mixed-precision page: runs the
   miniature simulator of nvrs_engine.js (fully implicit, Newton, FP64
   FGMRES with two-stage CPR in FP64, FP32 or FP16) and shows the fields,
   the iteration counts, the time per solver stage and the convergence.
   An FP64 run of the same model goes alongside as the reference (red), so
   the well curves of the chosen precision (blue) can be checked against it.
   Both runs live in Web Workers (nvrs_worker.js), one per core, and the app
   plays them as a movie: each frame advances a short time and redraws the
   fields and the well curves.                                             */
(function () {
  "use strict";
  const root = document.getElementById("demo-nvrs");
  if (!root || !window.NVRS) return;
  const E = window.NVRS;
  const $ = s => root.querySelector(s);
  const SRC = (document.currentScript && document.currentScript.src) || "";
  const fieldCv = $("canvas.field"), convCv = $("canvas.conv"), wellCv = $("canvas.wells");
  const fc = fieldCv.getContext("2d"), cc = convCv.getContext("2d"), wc_ = wellCv.getContext("2d");
  const P_ = {}, MIN_FRAME_MS = 70;
  let G0, A, B, snapA, snapB, last = null, tsplit = [0, 0, 0, 0], hist = [], running = false, stepping = false, gen = 0;

  function readParams() {
    root.querySelectorAll("input,select").forEach(el => {
      P_[el.name] = el.tagName === "SELECT" && isNaN(+el.value) ? el.value : parseFloat(el.value);
      const v = root.querySelector('[data-for="' + el.name + '"]');
      if (v) v.textContent = el.value;
    });
  }
  // a worker running the engine; on the main thread when workers are not available
  function engine() {
    if (window.Worker && SRC) {
      try {
        const w = new Worker(SRC.replace(/nvrs_sim\.js/, "nvrs_worker.js")), pend = new Map(); let seq = 0;
        w.onmessage = e => { const f = pend.get(e.data.id); pend.delete(e.data.id); if (f) f(e.data); };
        return { call: (cmd, payload) => new Promise(res => { const id = ++seq; pend.set(id, res); w.postMessage(Object.assign({ cmd, id }, payload)); }) };
      } catch (e) { /* fall through */ }
    }
    let M = null;
    const snap = res => { const r = M.prods.map(k => E.wellRates(M, k));
      return { t: M.t, dt: M.dt, steps: M.steps, newton: M.newton, krylov: M.krylov, p: M.p, S: M.S, qo: r.map(x => x.qo / M.Q), qw: r.map(x => x.qw / M.Q),
               wct: r.map(x => x.wcut), bhp: M.p[M.inj], res: res ? { newtonHist: res.newtonHist, krylovHists: res.krylovHists, tm: res.tm } : null }; };
    return { call: async (cmd, d) => {
      if (cmd === "build") { M = E.model(d.opts); return { ok: true, snap: snap(null) }; }
      if (cmd === "step") { const r = E.step(M, d.o); return { ok: !!r, snap: snap(r) }; }
      return { ok: false, rows: [] };
    } };
  }
  A = engine(); B = engine();
  async function build() {
    readParams(); running = false; $(".run").textContent = "Run";
    const g = ++gen, o = { n: P_.n | 0, sigma: P_.sigma, seed: P_.seed | 0, visc: P_.visc, wells: P_.wells };
    G0 = E.model(o);                          // geometry and permeability for drawing
    last = null; tsplit = [0, 0, 0, 0]; hist = [];
    $(".cmp").innerHTML = "";
    const [a, b] = await Promise.all([A.call("build", { opts: o }), B.call("build", { opts: o })]);
    if (g !== gen) return;
    snapA = a.snap; snapB = b.snap;
    drawField(); drawConv(); stats(); drawWells();
  }
  const rates = s => ({ qo: s.qo, qw: s.qw, wct: s.wct, bhp: s.bhp });
  async function frame() {
    if (stepping || !snapA) return;
    stepping = true;
    const g = gen, t0 = performance.now(), h = P_.frame || 3;
    const o = { restart: P_.restart | 0, dtrep: h, dtmax: h, dt0: Math.min(1, h), tstop: Math.min(P_.tend, snapA.t + h) };
    const [a, b] = await Promise.all([A.call("step", { o: Object.assign({ prec: P_.prec }, o) }), B.call("step", { o: Object.assign({ prec: "fp64" }, o) })]);
    stepping = false;
    if (g !== gen) return;
    if (!a.ok || !b.ok) { running = false; $(".run").textContent = "Run"; return; }
    snapA = a.snap; snapB = b.snap; last = snapA.res; tsplit = last.tm;
    hist.push({ t: snapA.t, a: rates(snapA), b: rates(snapB) });
    drawField(); drawConv(); stats(); drawWells();
    if (running && snapA.t < P_.tend - 1e-9) setTimeout(frame, Math.max(0, MIN_FRAME_MS - (performance.now() - t0)));
    else { running = false; $(".run").textContent = "Run"; }
  }
  // well curves: chosen precision (blue) against the FP64 reference (red, dashed), drawn as the run proceeds
  function drawWells() {
    const c = wc_, W = wellCv.width, H = wellCv.height; c.fillStyle = "#07070a"; c.fillRect(0, 0, W, H);
    c.font = "12px DM Mono, monospace"; c.fillStyle = "#e8e6e3";
    c.fillText("well curves: " + String(P_.prec).toUpperCase() + " preconditioner (blue) against the FP64 reference (red, dashed)", 10, 16);
    const panels = [["WOPR (oil rate / injection)", h => h.qo, 0], ["WWPR (water rate / injection)", h => h.qw, 0],
                    ["WWCT (water cut)", h => h.wct, 1.05], ["WBHP (injector bottom-hole pressure)", h => [h.bhp], 0]];
    const pw = (W - 20) / 4 - 12, t0 = 30, ph = H - t0 - 26, tEnd = Math.max(P_.tend, 1);
    let maxDiff = 0;
    panels.forEach(([lab, f, top], pi) => {
      const x0 = 16 + pi * (pw + 12);
      let hi = top; if (!hi) { hi = pi === 3 ? 1 : 0.05; for (const h of hist) hi = Math.max(hi, ...f(h.b).map(v => v * 1.15), ...f(h.a).map(v => v * 1.15)); }
      const X = t => x0 + pw * t / tEnd, Y = v => t0 + ph * (1 - Math.min(1.05, Math.max(0, v / hi)));
      c.strokeStyle = "rgba(255,255,255,.12)"; c.strokeRect(x0, t0, pw, ph);
      c.fillStyle = "#9a9997"; c.font = "11px DM Mono, monospace"; c.fillText(lab, x0 + 4, t0 + 13);
      c.fillText("0", x0 - 9, t0 + ph + 4); c.fillText(hi.toFixed(hi < 2 ? 2 : 1), x0 + 2, t0 + 26); c.fillText("time " + tEnd, x0 + pw - 62, H - 8);
      if (!hist.length) return;
      const nl = f(hist[0].a).length;
      for (let w = 0; w < nl; w++) {
        for (const [key, col, dash, lw] of [["a", "#4fa0ff", [], 2.2], ["b", "#ff4d4d", [5, 4], 1.6]]) {
          c.beginPath(); c.strokeStyle = col; c.setLineDash(dash); c.lineWidth = lw;
          c.moveTo(X(0), Y(pi === 3 ? 1 : 0));
          hist.forEach(h => c.lineTo(X(h.t), Y(f(h[key])[w])));
          c.stroke(); c.setLineDash([]); c.lineWidth = 1;
        }
        const lastH = hist[hist.length - 1];
        if (nl > 1) { c.fillStyle = "#e8e6e3"; c.fillText("P" + (w + 1), Math.min(X(lastH.t) + 3, x0 + pw - 18), Y(f(lastH.a)[w]) + 4); }
      }
      for (const h of hist) f(h.a).forEach((v, w) => { maxDiff = Math.max(maxDiff, Math.abs(v - f(h.b)[w]) / (Math.abs(f(h.b)[w]) + 1e-3)); });
    });
    if (hist.length) { c.fillStyle = "#e8e6e3"; c.font = "12px DM Mono, monospace";
      c.fillText("largest relative difference from FP64 over all curves: " + maxDiff.toExponential(1), W - 470, 16); }
  }
  const jet = t => { const x = Math.min(1, Math.max(0, t)); return [1.5 - Math.abs(4 * x - 3), 1.5 - Math.abs(4 * x - 2), 1.5 - Math.abs(4 * x - 1)].map(v => Math.round(255 * Math.min(1, Math.max(0, v)))); };
  function color(v, kind) {                  // jet colour map
    return kind === "S" ? jet((v - E.swc) / (1 - E.swc - E.sor)) : jet(v);
  }
  function drawField() {
    if (!snapA) return;
    const { n, N } = G0, W = fieldCv.width, H = fieldCv.height, img = fc.createImageData(n, n);
    const view = P_.view, val = k => view === "p" ? snapA.p[k] : view === "K" ? Math.log(G0.K[k]) : snapA.S[k];
    let lo = Infinity, hi = -Infinity;
    for (let k = 0; k < N; k++) { lo = Math.min(lo, val(k)); hi = Math.max(hi, val(k)); }
    for (let k = 0; k < N; k++) {
      const c = view === "S" ? color(snapA.S[k], "S") : color((val(k) - lo) / (hi - lo || 1), "v");
      img.data.set([c[0], c[1], c[2], 255], 4 * k);
    }
    const off = document.createElement("canvas"); off.width = n; off.height = n;
    off.getContext("2d").putImageData(img, 0, 0);
    fc.imageSmoothingEnabled = true; fc.imageSmoothingQuality = "high"; fc.drawImage(off, 0, 0, W, H);   // smooth, as the waterflood demo
    for (const [k, col, lab] of [...G0.injs.map(k => [k, "#ffffff", "I"]), ...G0.prods.map((k, i) => [k, "#ff00ff", "P" + (G0.prods.length > 1 ? i + 1 : "")])]) {
      const x = ((k % n) + 0.5) * W / n, y = (Math.floor(k / n) + 0.5) * H / n;
      fc.beginPath(); fc.arc(x, y, 7, 0, 7); fc.fillStyle = col; fc.fill(); fc.strokeStyle = "#000"; fc.lineWidth = 2; fc.stroke();
      fc.fillStyle = "#fff"; fc.font = "12px DM Mono, monospace"; fc.fillText(lab, Math.min(W - 44, x + 11), Math.max(14, y - 9));
    }
    const bx = W / 2 - 165;                   // status box at the bottom centre, clear of the corner wells
    fc.fillStyle = "rgba(0,0,0,.6)"; fc.fillRect(bx, H - 30, 330, 22);
    fc.fillStyle = "#fff"; fc.font = "12px DM Mono, monospace";
    fc.fillText(n + " x " + n + "   t = " + snapA.t.toFixed(1) + "   step " + snapA.steps + "   next dt = " + snapA.dt.toFixed(2), bx + 6, H - 14);
  }
  function drawConv() {
    const W = convCv.width, H = convCv.height, l = 50, r = 12, t = 26, b = 30;
    cc.fillStyle = "#07070a"; cc.fillRect(0, 0, W, H);
    cc.font = "11px DM Mono, monospace"; cc.fillStyle = "#e8e6e3";
    cc.fillText("last time step: Newton residual (dots), FGMRES residual (lines)", l, 15);
    if (!last) return;
    const ks = last.krylovHists, tot = ks.reduce((a, h) => a + h.length, 0) + ks.length + 1;
    const X = v => l + (W - l - r) * v / Math.max(tot, 10), Y = v => t + (H - t - b) * (-Math.log10(Math.max(v, 1e-12))) / 12;
    cc.strokeStyle = "rgba(255,255,255,.1)"; cc.fillStyle = "#9a9997";
    for (let e = 0; e <= 12; e += 3) { const y = Y(Math.pow(10, -e)); cc.beginPath(); cc.moveTo(l, y); cc.lineTo(W - r, y); cc.stroke(); cc.fillText("1e-" + e, 8, y + 4); }
    let x0 = 0;
    ks.forEach((h, i) => {
      cc.beginPath(); cc.strokeStyle = "#00e5a0"; cc.lineWidth = 1.6;
      h.forEach((v, j) => j ? cc.lineTo(X(x0 + j + 1), Y(v)) : cc.moveTo(X(x0 + 1), Y(v)));
      cc.stroke();
      cc.beginPath(); cc.arc(X(x0), Y(last.newtonHist[i]), 4, 0, 7); cc.fillStyle = "#ffb74d"; cc.fill();
      x0 += h.length + 1;
    });
    cc.beginPath(); cc.arc(X(x0), Y(last.newtonHist[last.newtonHist.length - 1]), 4, 0, 7); cc.fillStyle = "#ffb74d"; cc.fill();
    cc.fillStyle = "#9a9997"; cc.fillText("FGMRES iterations of every Newton step, in sequence", l, H - 8);
  }
  function stats() {
    if (!snapA) return;
    const ms = tsplit, tot = ms.reduce((a, b) => a + b, 0) || 1, cols = ["#9a9997", "#00e5a0", "#4fc3ff", "#7b61ff"];
    const lh = hist.length ? hist[hist.length - 1].a : null;
    const wc = lh ? lh.qw.reduce((a, b) => a + b, 0) / Math.max(1e-12, lh.qw.reduce((a, b) => a + b, 0) + lh.qo.reduce((a, b) => a + b, 0)) : 0;
    $(".sim-stats").innerHTML =
      '<div><b>' + snapA.steps + '</b><span>time steps</span></div>' +
      '<div><b>' + snapA.newton + '</b><span>Newton iterations</span></div>' +
      '<div><b>' + snapA.krylov + '</b><span>FGMRES iterations</span></div>' +
      '<div><b>' + (snapA.newton ? (snapA.krylov / snapA.newton).toFixed(1) : "-") + '</b><span>Krylov per Newton</span></div>' +
      '<div><b>' + (100 * wc).toFixed(0) + ' %</b><span>field water cut</span></div>';
    $(".split").innerHTML = '<div class="splitbar">' + ms.map((v, i) => '<i style="width:' + (100 * v / tot).toFixed(1) +
      '%;background:' + cols[i] + '"></i>').join("") + '</div><div class="legend">' +
      ["assembly (FP64)", "CPR stage 1: V-cycle", "CPR stage 2: 2x2 block Jacobi", "FGMRES (FP64)"].map((s, i) =>
        '<span><i style="background:' + cols[i] + '"></i>' + s + " " + ms[i].toFixed(0) + " ms</span>").join("") + "</div>";
    root.querySelectorAll(".stage").forEach(el => {
      el.querySelector("em").textContent = el.getAttribute("data-stage") === "pre" ? String(P_.prec).toUpperCase() : "FP64";
    });
  }
  $(".run").addEventListener("click", async () => {
    if (running) { running = false; $(".run").textContent = "Run"; return; }
    if (snapA && snapA.t >= P_.tend - 1e-9) await build();
    running = true; $(".run").textContent = "Pause"; frame();
  });
  $(".step").addEventListener("click", () => { running = false; $(".run").textContent = "Run"; frame(); });
  $(".reset").addEventListener("click", () => { build(); });
  $(".compare").addEventListener("click", async () => {
    if (stepping) return;
    $(".cmp").innerHTML = '<p class="dnote">solving the next step three times...</p>';
    const r = await A.call("compare", { restart: P_.restart | 0 });
    $(".cmp").innerHTML = "<table><tr><th>CPR preconditioner in</th><th>Newton</th><th>FGMRES</th><th>time</th><th>max |S_w - S_w(FP64)|</th></tr>" +
      (r.rows || []).map(row => "<tr>" + row.map(c => "<td>" + c + "</td>").join("") + "</tr>").join("") + "</table>" +
      '<p class="dnote">The same next time step solved three times from the current state. With FGMRES in FP64 all three are converged to the same Newton and linear tolerances, so they agree to within those tolerances; the last column is the measured difference. The preconditioner precision changes the iteration counts.</p>';
  });
  root.querySelectorAll("input,select").forEach(el => el.addEventListener("input", () => {
    readParams();
    if (el.name === "view") { drawField(); return; }
    if (el.name === "prec") { stats(); drawWells(); return; }
    if (["restart", "frame", "tend"].includes(el.name)) return;
    build();
  }));
  build();
})();
