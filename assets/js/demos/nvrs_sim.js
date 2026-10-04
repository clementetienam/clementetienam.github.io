/* nvrs_sim.js -- the simulator app on the mixed-precision page: runs the
   miniature simulator of nvrs_engine.js (fully implicit, Newton, FP64
   FGMRES with two-stage CPR in FP64, FP32 or FP16) and shows the fields,
   the iteration counts, the time per solver stage and the convergence. */
(function () {
  "use strict";
  const root = document.getElementById("demo-nvrs");
  if (!root || !window.NVRS) return;
  const E = window.NVRS;
  const $ = s => root.querySelector(s);
  const fieldCv = $("canvas.field"), convCv = $("canvas.conv");
  const fc = fieldCv.getContext("2d"), cc = convCv.getContext("2d");
  const P_ = {};
  let M, last = null, tsplit = [0, 0, 0, 0], hist = [], running = false, raf = 0;

  function readParams() {
    root.querySelectorAll("input,select").forEach(el => {
      P_[el.name] = el.tagName === "SELECT" && isNaN(+el.value) ? el.value : parseFloat(el.value);
      const v = root.querySelector('[data-for="' + el.name + '"]');
      if (v) v.textContent = el.value;
    });
  }
  function build() {
    readParams();
    M = E.model({ n: P_.n | 0, sigma: P_.sigma, seed: P_.seed | 0, visc: P_.visc });
    last = null; tsplit = [0, 0, 0, 0]; hist = [];
    $(".cmp").innerHTML = "";
    drawField(); drawConv(); stats();
  }
  function advance() {
    const res = E.step(M, { prec: P_.prec, restart: P_.restart | 0, dtrep: P_.dtmax, dtmax: P_.dtmax, dt0: 1, tstop: Math.min(P_.tend, M.t + P_.dtmax) });
    if (!res) { running = false; return; }
    last = res; tsplit = res.tm;
    hist.push([M.t, E.wellRates(M).wcut]);
  }
  function color(v, kind) {
    if (kind === "S") {
      const x = Math.min(1, Math.max(0, (v - E.swc) / (1 - E.swc - E.sor)));
      const a = [74, 52, 26], b = [40, 110, 210], c = [150, 210, 255];
      const [u, w, f] = x < 0.6 ? [a, b, x / 0.6] : [b, c, (x - 0.6) / 0.4];
      return u.map((q, i) => Math.round(q + (w[i] - q) * f));
    }
    const st = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
    const x = Math.min(0.999, Math.max(0, v)) * 4, i = Math.floor(x), f = x - i;
    return st[i].map((q, c) => Math.round(q + (st[i + 1][c] - q) * f));
  }
  function drawField() {
    const { n, N } = M, W = fieldCv.width, H = fieldCv.height, img = fc.createImageData(n, n);
    const view = P_.view, val = k => view === "p" ? M.p[k] : view === "K" ? Math.log(M.K[k]) : M.S[k];
    let lo = Infinity, hi = -Infinity;
    for (let k = 0; k < N; k++) { lo = Math.min(lo, val(k)); hi = Math.max(hi, val(k)); }
    for (let k = 0; k < N; k++) {
      const c = view === "S" ? color(M.S[k], "S") : color((val(k) - lo) / (hi - lo || 1), "v");
      img.data.set([c[0], c[1], c[2], 255], 4 * k);
    }
    const off = document.createElement("canvas"); off.width = n; off.height = n;
    off.getContext("2d").putImageData(img, 0, 0);
    fc.imageSmoothingEnabled = false; fc.drawImage(off, 0, 0, W, H);
    for (const [k, col, lab] of [[M.inj, "#4fc3ff", "INJ"], [M.prod, "#00e5a0", "PROD"]]) {
      const x = ((k % n) + 0.5) * W / n, y = (Math.floor(k / n) + 0.5) * H / n;
      fc.beginPath(); fc.arc(x, y, 8, 0, 7); fc.fillStyle = col; fc.fill(); fc.strokeStyle = "#000"; fc.lineWidth = 2; fc.stroke();
      fc.fillStyle = "#fff"; fc.font = "12px DM Mono, monospace"; fc.fillText(lab, Math.min(W - 44, x + 11), Math.max(14, y - 9));
    }
    fc.fillStyle = "rgba(0,0,0,.6)"; fc.fillRect(8, H - 30, 270, 22);
    fc.fillStyle = "#fff"; fc.font = "12px DM Mono, monospace";
    fc.fillText("t = " + M.t.toFixed(1) + "   step " + M.steps + "   next dt = " + M.dt.toFixed(2), 14, H - 14);
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
    const ms = tsplit, tot = ms.reduce((a, b) => a + b, 0) || 1, cols = ["#9a9997", "#00e5a0", "#4fc3ff", "#7b61ff"];
    const wc = hist.length ? hist[hist.length - 1][1] : 0;
    $(".sim-stats").innerHTML =
      '<div><b>' + M.steps + '</b><span>time steps</span></div>' +
      '<div><b>' + M.newton + '</b><span>Newton iterations</span></div>' +
      '<div><b>' + M.krylov + '</b><span>FGMRES iterations</span></div>' +
      '<div><b>' + (M.newton ? (M.krylov / M.newton).toFixed(1) : "-") + '</b><span>Krylov per Newton</span></div>' +
      '<div><b>' + (100 * wc).toFixed(0) + ' %</b><span>producer water cut</span></div>';
    $(".split").innerHTML = '<div class="splitbar">' + ms.map((v, i) => '<i style="width:' + (100 * v / tot).toFixed(1) +
      '%;background:' + cols[i] + '"></i>').join("") + '</div><div class="legend">' +
      ["assembly (FP64)", "CPR stage 1: V-cycle", "CPR stage 2: 2x2 block Jacobi", "FGMRES (FP64)"].map((s, i) =>
        '<span><i style="background:' + cols[i] + '"></i>' + s + " " + ms[i].toFixed(0) + " ms</span>").join("") + "</div>";
    root.querySelectorAll(".stage").forEach(el => {
      el.querySelector("em").textContent = el.getAttribute("data-stage") === "pre" ? P_.prec.toUpperCase() : "FP64";
    });
  }
  function frame() {
    advance(); drawField(); drawConv(); stats();
    if (running && M.t < P_.tend) raf = requestAnimationFrame(frame);
    else { running = false; $(".run").textContent = "Run"; }
  }
  $(".run").addEventListener("click", () => {
    running = !running; $(".run").textContent = running ? "Pause" : "Run";
    if (running) { if (M.t >= P_.tend) build(); raf = requestAnimationFrame(frame); }
  });
  $(".step").addEventListener("click", () => { running = false; $(".run").textContent = "Run"; frame(); });
  $(".reset").addEventListener("click", () => { running = false; cancelAnimationFrame(raf); $(".run").textContent = "Run"; build(); });
  $(".compare").addEventListener("click", () => {
    const out = []; let ref = null;
    for (const prec of ["fp64", "fp32", "fp16"]) {
      const t0 = performance.now();
      let res = null, dt = M.dt;
      while (!res && dt > 1e-4) { res = E.timeStep(M, dt, prec, P_.restart | 0); if (!res) dt *= 0.5; }
      const ms = performance.now() - t0;
      if (!res) { out.push([prec.toUpperCase(), "-", "-", "-", "failed"]); continue; }
      if (!ref) ref = res;
      let d = 0; for (let k = 0; k < M.N; k++) d = Math.max(d, Math.abs(res.S[k] - ref.S[k]));
      out.push([prec.toUpperCase(), res.newton, res.krylov, ms.toFixed(0) + " ms", prec === "fp64" ? "reference" : d.toExponential(1)]);
    }
    $(".cmp").innerHTML = "<table><tr><th>CPR preconditioner in</th><th>Newton</th><th>FGMRES</th><th>time</th><th>max |S_w - S_w(FP64)|</th></tr>" +
      out.map(r => "<tr>" + r.map(c => "<td>" + c + "</td>").join("") + "</tr>").join("") + "</table>" +
      '<p class="dnote">The same next time step solved three times from the current state. With FGMRES in FP64 the solutions agree to the Newton tolerance; the preconditioner precision changes the iteration counts.</p>';
  });
  root.querySelectorAll("input,select").forEach(el => el.addEventListener("input", () => {
    readParams();
    if (el.name === "view") { drawField(); return; }
    if (el.name === "prec") { stats(); return; }
    if (["restart", "dtmax", "tend"].includes(el.name)) return;
    running = false; cancelAnimationFrame(raf); $(".run").textContent = "Run"; build();
  }));
  build();
})();
