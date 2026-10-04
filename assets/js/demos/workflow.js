/* workflow.js -- the end-to-end surrogate workflow, in miniature, in the
   browser:
     1 geology   permeability fields log K = sigma * sum_m xi_m phi_m(x),
                 xi ~ N(0, I), six smooth basis functions;
     2 simulate  each realisation run to the end of the schedule with the
                 mixed-precision simulator (nvrs_engine.js: Newton, FP64
                 FGMRES, two-stage CPR with an FP32 multigrid V-cycle);
                 five-spot (injector in the centre, a producer in each
                 corner); recorded: the water cut of each producer, the field
                 oil rate and the injector pressure;
     3 train     a CCR surrogate xi -> production (cluster-classify-regress,
                 ccr.js), live;
     4 validate  on held-out simulations: accuracy and measured speed-up;
     5 invert    (options: covariance localisation, made on the grid with a
                 Gaspari-Cohn taper and projected back onto the basis, and
                 inflation against ensemble collapse)
                 alpha-REKI (1/alpha from the mean and variance of the data
                 misfit, Iglesias and Yang; stopped when sum 1/alpha = 1) or ES-MDA (alpha = number of assimilations) on
                 xi, the surrogate as forward model, against noisy observations
                 of a hidden "true" field;
     6 verify    the simulator run on the matched model.
   The same structure as the PhysicsNeMo workflow on Norne, at a size a
   browser can run in seconds.                                            */
(function () {
  "use strict";
  const root = document.getElementById("demo-workflow");
  if (!root || !window.NVRS) return;
  const E = window.NVRS, $ = s => root.querySelector(s);
  const mapCv = $("canvas.maps"), crvCv = $("canvas.curves"), lossCv = $("canvas.loss"), alphaCv = $("canvas.alpha");
  const mc = mapCv.getContext("2d"), cc = crvCv.getContext("2d"), lc = lossCv.getContext("2d"), ac = alphaCv.getContext("2d");
  const drawAlpha = series => window.PLOTS && window.PLOTS.alpha(ac, alphaCv.width, alphaCv.height, series,
    (P_.method === 1 ? "ES-MDA: alpha = N_a at every assimilation" : "alpha-REKI: alpha against iteration") + " (left, log) and sum of 1/alpha (right)");
  const P_ = {};
  const n = 16, MD = 6, NR = 20, T_END = 300, SIG = 1.3, VISC = 5;
  const RT = Array.from({ length: NR }, (_, i) => (i + 1) * T_END / NR);
  // outputs per run: water cut of P1..P4, field oil rate / injection, log injector pressure
  const NP = 4, NY = (NP + 2) * NR;
  const C = n >> 1, WELLS = [[C * n + C, "#ffffff"], [n + 1, "#ff00ff"], [n + n - 2, "#ff00ff"], [(n - 2) * n + 1, "#ff00ff"], [(n - 2) * n + n - 2, "#ff00ff"]];
  let state = null, busy = false, truthSeed = 11;

  function rng(seed) {
    let a = seed >>> 0;
    return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  }
  function gaussFrom(r) { let u = 0; while (!u) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); }
  function readParams() {
    root.querySelectorAll("input,select").forEach(el => {
      P_[el.name] = el.tagName === "SELECT" ? +el.selectedIndex : parseFloat(el.value);
      const v = root.querySelector('[data-for="' + el.name + '"]'); if (v) v.textContent = el.value;
    });
  }
  // ---- geology: smooth basis
  const WAVES = [[1, 0], [0, 1], [1, 1], [2, 0], [0, 2], [2, 1]];
  const BASIS = WAVES.map(([kx, ky]) => {
    const b = new Float64Array(n * n); let s = 0;
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const v = Math.cos(Math.PI * kx * (i + 0.5) / n) * Math.cos(Math.PI * ky * (j + 0.5) / n);
      b[j * n + i] = v; s += v * v;
    }
    s = Math.sqrt(s / (n * n)); return b.map(v => v / s);
  });
  function project(f) {                      // ln K field -> basis coefficients (exact for fields in the span)
    return BASIS.map(b => { let s = 0; for (let k = 0; k < n * n; k++) s += f[k] * b[k]; return s / (n * n) * Math.sqrt(MD) / SIG; });
  }
  // Gaspari-Cohn taper, zero beyond the radius L (cells)
  function gc(d, L) {
    const r = 2 * d / L;
    if (r >= 2) return 0;
    if (r <= 1) return (((-0.25 * r + 0.5) * r + 0.625) * r - 5 / 3) * r * r + 1;
    return ((((r / 12 - 0.5) * r + 0.625) * r + 5 / 3) * r - 5) * r + 4 - 2 / (3 * r);
  }
  function logK(xi) {
    const f = new Float64Array(n * n);
    for (let m = 0; m < MD; m++) for (let k = 0; k < n * n; k++) f[k] += SIG * xi[m] * BASIS[m][k] / Math.sqrt(MD);
    return f;
  }
  // ---- simulate one realisation: 3 x NR outputs
  function simulate(xi) {
    const M = E.model({ n, logK: logK(xi), visc: VISC, dt0: 0.5, wells: "fivespot" });
    const y = new Float64Array(NY), t0 = performance.now();
    let kry = 0, newt = 0;
    for (let r = 0; r < NR; r++) {
      while (M.t < RT[r] - 1e-9) {
        const res = E.step(M, { prec: "fp32", restart: 30, dtmax: 15, tstop: RT[r] });
        if (!res) return null;
        kry += res.krylov; newt += res.newton;
      }
      let qo = 0;
      M.prods.forEach((k, w) => { const q = E.wellRates(M, k); y[w * NR + r] = q.wcut; qo += q.qo; });
      y[NP * NR + r] = qo / M.Q; y[(NP + 1) * NR + r] = Math.log(Math.max(1e-6, M.p[M.inj]));
    }
    return { y, ms: performance.now() - t0, kry, newt };
  }
  // ---- drawing
  function viridis(t) {   // jet colour map (name kept for its callers)
    const x = Math.min(1, Math.max(0, t)); return [1.5 - Math.abs(4 * x - 3), 1.5 - Math.abs(4 * x - 2), 1.5 - Math.abs(4 * x - 1)].map(v => Math.round(255 * Math.min(1, Math.max(0, v)))); }
  function drawMaps(fields, titles) {
    const W = mapCv.width, H = mapCv.height;
    mc.fillStyle = "#07070a"; mc.fillRect(0, 0, W, H);
    const k = fields.length, gap = 14, size = Math.min((W - gap * (k + 1)) / k, H - 40);
    fields.forEach((f, idx) => {
      const img = mc.createImageData(n, n);
      for (let q = 0; q < n * n; q++) { const c = viridis((f[q] + 2.5 * SIG / 1.2) / (5 * SIG / 1.2)); img.data.set([c[0], c[1], c[2], 255], 4 * q); }
      const off = document.createElement("canvas"); off.width = n; off.height = n; off.getContext("2d").putImageData(img, 0, 0);
      const x = gap + idx * (size + gap);
      mc.imageSmoothingEnabled = true; mc.drawImage(off, x, 28, size, size);
      mc.fillStyle = "#e8e6e3"; mc.font = "12px DM Mono, monospace"; mc.fillText(titles[idx], x, 18);
      for (const [cell, col] of WELLS) {
        mc.beginPath(); mc.arc(x + ((cell % n) + 0.5) * size / n, 28 + (Math.floor(cell / n) + 0.5) * size / n, 5, 0, 7);
        mc.fillStyle = col; mc.fill(); mc.strokeStyle = "#000"; mc.stroke();
      }
    });
  }
  function drawCurves(sets, title) {        // sets: [{y, col, w, dots}]; six panels
    const W = crvCv.width, H = crvCv.height, t0 = 24;
    const panels = [["P1 water cut", 0], ["P2 water cut", NR], ["P3 water cut", 2 * NR], ["P4 water cut", 3 * NR],
                    ["field oil rate / injection", NP * NR], ["log injector pressure", (NP + 1) * NR]];
    cc.fillStyle = "#07070a"; cc.fillRect(0, 0, W, H);
    cc.font = "11px DM Mono, monospace"; cc.fillStyle = "#e8e6e3"; cc.fillText(title, 8, 15);
    const cols = 3, pw = (W - 16) / cols - 8, ph = (H - t0 - 6) / 2 - 8;
    // pressure axis from the data on show
    let plo = Infinity, phi = -Infinity;
    for (const s of sets) for (let i = 0; i < NR; i++) { const v = s.y[(NP + 1) * NR + i]; if (isFinite(v)) { plo = Math.min(plo, v); phi = Math.max(phi, v); } }
    if (!isFinite(plo)) { plo = -1; phi = 2; } const pad = 0.1 * (phi - plo || 1); plo -= pad; phi += pad;
    panels.forEach(([lab, off], pi) => {
      const x0 = 8 + (pi % cols) * (pw + 8), y0 = t0 + Math.floor(pi / cols) * (ph + 8);
      const lo = pi === 5 ? plo : 0, hi = pi === 5 ? phi : 1;
      const X = v => x0 + pw * v / T_END, Y = v => y0 + ph * (1 - Math.min(1, Math.max(0, (v - lo) / (hi - lo))));
      cc.strokeStyle = "rgba(255,255,255,.12)"; cc.strokeRect(x0, y0, pw, ph);
      cc.fillStyle = "#9a9997"; cc.fillText(lab, x0 + 4, y0 + 12);
      for (const s of sets) {
        if (!s.dots) { cc.beginPath(); cc.strokeStyle = s.col; cc.lineWidth = s.w || 1;
          RT.forEach((tt, i) => i ? cc.lineTo(X(tt), Y(s.y[off + i])) : cc.moveTo(X(tt), Y(s.y[off + i]))); cc.stroke(); }
        else RT.forEach((tt, i) => { cc.beginPath(); cc.arc(X(tt), Y(s.y[off + i]), 2.4, 0, 7); cc.fillStyle = "#ff4d4d"; cc.fill(); cc.strokeStyle = "#000"; cc.lineWidth = 1; cc.stroke(); });
      }
    });
  }
  function drawLoss(sse, K, title) {          // the elbow: within-cluster sum of squares against K
    const W = lossCv.width, H = lossCv.height, l = 50, r = 14, t = 26, b = 26;
    lc.fillStyle = "#07070a"; lc.fillRect(0, 0, W, H);
    lc.font = "11px DM Mono, monospace"; lc.fillStyle = "#e8e6e3"; lc.fillText(title, l, 14);
    if (!sse || !sse.length) return;
    const hi = sse[0] * 1.05, X = k => l + (W - l - r) * (k - 1) / Math.max(sse.length - 1, 1), Y = v => t + (H - t - b) * (1 - v / hi);
    lc.strokeStyle = "rgba(255,255,255,.12)"; lc.strokeRect(l, t, W - l - r, H - t - b);
    lc.beginPath(); lc.strokeStyle = "#4fa0ff"; lc.lineWidth = 2; sse.forEach((v, i) => i ? lc.lineTo(X(i + 1), Y(v)) : lc.moveTo(X(1), Y(v))); lc.stroke(); lc.lineWidth = 1;
    sse.forEach((v, i) => { lc.beginPath(); lc.arc(X(i + 1), Y(v), i + 1 === K ? 6 : 3.5, 0, 7); lc.fillStyle = i + 1 === K ? "#ff4d4d" : "#4fa0ff"; lc.fill();
      lc.fillStyle = "#9a9997"; lc.fillText("K=" + (i + 1), X(i + 1) - 12, H - 8); });
    lc.fillStyle = "#e8e6e3"; lc.fillText("chosen K = " + K, W - r - 100, t + 14);
  }
  function stage(k, status) {
    root.querySelectorAll(".stage").forEach(el => {
      const s = +el.getAttribute("data-step");
      el.classList.toggle("active", s === k && status !== "done");
      el.classList.toggle("done", s < k || (s === k && status === "done"));
    });
  }
  function log(msg) { const el = $(".wlog"); el.innerHTML += "<div>" + msg + "</div>"; el.scrollTop = el.scrollHeight; }
  const tick = () => new Promise(res => setTimeout(res, 0));
  async function runAll() {
    if (busy) return; busy = true; readParams();
    $(".wlog").innerHTML = ""; $(".run").disabled = true;
    const r = rng(1234 + (P_.runs | 0));
    // 1-2 sample and simulate
    const NV = 20;                             // held-out runs
    stage(1); log("1. Sampling " + ((P_.runs | 0) + NV) + " permeability realisations from the prior (" + (P_.runs | 0) + " to train on, " + NV + " to validate).");
    const X = [], Y = [], simMs = []; let kry = 0, newt = 0;
    stage(2); log("2. Simulating each one with the mixed-precision simulator (FP64 FGMRES, CPR with an FP32 V-cycle).");
    const total = (P_.runs | 0) + NV;
    for (let s = 0; s < total; s++) {
      const xi = Array.from({ length: MD }, () => gaussFrom(r));
      const out = simulate(xi);
      if (!out) continue;
      X.push(xi); Y.push(out.y); simMs.push(out.ms); kry += out.kry; newt += out.newt;
      if (s % 4 === 0 || s === total - 1) {
        drawMaps([logK(xi)], ["realisation " + (s + 1) + " of " + total]);
        drawCurves([{ y: out.y, col: "#4fc3ff", w: 1.5 }], "simulated production, realisation " + (s + 1));
        $(".prog").style.width = (100 * (s + 1) / total).toFixed(0) + "%";
        await tick();
      }
    }
    const simAvg = simMs.reduce((a, b) => a + b, 0) / simMs.length;
    log("&nbsp;&nbsp; " + X.length + " runs (" + Math.max(0, X.length - NV) + " for training, " + NV + " held out), " + newt + " Newton and " + kry + " FGMRES iterations, " + simAvg.toFixed(0) + " ms per run.");
    stage(2, "done");
    // 3 train: CCR, cluster-classify-regress
    stage(3); log("3. Training the CCR surrogate on " + (X.length - NV) + " runs: k-means on joint (parameters, production), a random-forest classifier, Gaussian-process experts.");
    await tick();
    const Xt = X.slice(0, X.length - NV), Yt = Y.slice(0, Y.length - NV), Xv = X.slice(-NV), Yv = Y.slice(-NV);
    const trainCCR = () => window.CCR.train(Xt, Yt, { Kmax: 5, expert: "gp", overlap: 0.1 });
    let ccr = trainCCR();
    drawLoss(ccr.sse, ccr.K, "CCR clustering: within-cluster sum of squares against K (elbow)");
    let g = xi => ccr.predict(xi);
    log("&nbsp;&nbsp; elbow picks K = " + ccr.K + " regimes; random-forest classifier accuracy " + (100 * ccr.acc).toFixed(1) + " % on the training runs.");
    $(".prog").style.width = "100%";
    stage(3, "done");
    // 4 validate
    stage(4); log("4. Validating on " + NV + " held-out simulations.");
    let ss = 0, st = 0; const mu = new Float64Array(NY);
    for (const y of Yv) for (let j = 0; j < NY; j++) mu[j] += y[j] / Yv.length;
    const t0 = performance.now(); const preds = Xv.map(g); const surMs = (performance.now() - t0) / Xv.length;
    preds.forEach((p, s) => { for (let j = 0; j < NY; j++) { ss += (p[j] - Yv[s][j]) ** 2; st += (Yv[s][j] - mu[j]) ** 2; } });
    const R2 = 1 - ss / st;
    let ss1 = 0; Xv.forEach((x, s) => { const p = ccr.single(x); for (let j = 0; j < NY; j++) ss1 += (p[j] - Yv[s][j]) ** 2; });
    drawCurves([{ y: Yv[0], col: "#ff4d4d", w: 2.2 }, { y: preds[0], col: "#4fa0ff", w: 2.2 }], "held-out run: simulator (red) against surrogate (blue)");
    log("&nbsp;&nbsp; R2 = " + R2.toFixed(3) + " on unseen runs (one global expert: " + (1 - ss1 / st).toFixed(3) + "); surrogate " + surMs.toFixed(3) + " ms against simulator " + simAvg.toFixed(0) +
        " ms per run, " + Math.round(simAvg / Math.max(surMs, 1e-3)) + " times faster.");
    stage(4, "done"); await tick();
    // 5 history match: alpha-REKI (adaptive alpha, stop when sum 1/alpha = 1) or ES-MDA (alpha = Na)
    const method = P_.method === 1 ? "esmda" : "areki";
    stage(5); log("5. History matching with " + (method === "areki" ? "alpha-REKI (adaptive regularised ensemble Kalman inversion)" : "ES-MDA") +
                  " and the surrogate, " + (P_.ne | 0) + " members.");
    const rt = rng(truthSeed * 77 + 3), xiTrue = Array.from({ length: MD }, () => gaussFrom(rt));
    const truth = simulate(xiTrue), sdObs = P_.noise;
    const obsIdx = Array.from({ length: NY }, (_, i) => i);                 // every water cut, oil rate and injector pressure
    const dobs = obsIdx.map(i => truth.y[i] + sdObs * gaussFrom(rt));
    const Ne = P_.ne | 0, Na = P_.na | 0, m = obsIdx.length;
    const meanOf = A => A[0].map((_, j) => A.reduce((s, a) => s + a[j], 0) / A.length);
    const obsY = new Float64Array(NY); obsIdx.forEach((i, p) => { obsY[i] = dobs[p]; });
    let re, ens, priorMeanK; const alphaSeries = [];
    // localisation: the distance from each cell to the well of each datum (field oil rate: the nearest producer)
    const Lr = P_.loc | 0, beta = P_.infl || 1;
    const dist = (k, w) => Math.hypot((k % n) - (w % n), ((k / n) | 0) - ((w / n) | 0));
    const RHO = Lr > 0 ? Array.from({ length: n * n }, (_, k) => Float64Array.from(obsIdx, i => {
      const grp = Math.floor(i / NR);
      const d = grp < NP ? dist(k, WELLS[1 + grp][0]) : grp === NP ? Math.min(...WELLS.slice(1).map(w => dist(k, w[0]))) : dist(k, WELLS[0][0]);
      return gc(d, Lr); })) : null;
    const spread = E_ => { const F = E_.map(logK), mm = meanOf(F); let s = 0; for (let k = 0; k < n * n; k++) s += Math.sqrt(F.reduce((a, f) => a + (f[k] - mm[k]) ** 2, 0) / F.length); return s / (n * n); };
    let spread0 = 0;
    async function historyMatch(tag) {
    re = rng(99);                              // the same starting ensemble in every round
    ens = Array.from({ length: Ne }, () => Array.from({ length: MD }, () => gaussFrom(re)));
    priorMeanK = logK(meanOf(ens)); spread0 = spread(ens);
    log("&nbsp;&nbsp; " + (Lr > 0 ? "localisation radius " + Lr + " cells (update on the grid, projected onto the basis)" : "no localisation") +
        (beta > 1 ? ", inflation " + beta.toFixed(2) + " whenever the spread falls below half the prior spread" : ", no inflation") +
        "; prior spread of ln K " + spread0.toFixed(3) + ".");
    drawMaps([logK(xiTrue), priorMeanK], ["true field", "prior mean"]);
    drawCurves([...ens.slice(0, 40).map(u => ({ y: g(u), col: "rgba(79,160,255,.25)" })), { y: truth.y, col: "#ff4d4d", w: 2 }, { y: obsY, dots: true }],
               "prior ensemble (blue), true model (red), observed (red dots)" + tag);
    await new Promise(res => setTimeout(res, 700));
    function kalman(alpha) {
      const D = ens.map(u => { const y = g(u); return obsIdx.map(i => y[i]); });
      const um = meanOf(ens), dm = meanOf(D);
      const Cud = Array.from({ length: MD }, () => new Float64Array(m)), Cdd = Array.from({ length: m }, () => new Float64Array(m));
      for (let j = 0; j < Ne; j++) for (let p = 0; p < m; p++) {
        const dp = D[j][p] - dm[p];
        for (let q = 0; q < MD; q++) Cud[q][p] += (ens[j][q] - um[q]) * dp / (Ne - 1);
        for (let q = p; q < m; q++) Cdd[p][q] += dp * (D[j][q] - dm[q]) / (Ne - 1);
      }
      for (let p = 0; p < m; p++) { for (let q = 0; q < p; q++) Cdd[p][q] = Cdd[q][p]; Cdd[p][p] += alpha * sdObs * sdObs; }
      const L = Cdd.map(row => Float64Array.from(row));
      for (let i = 0; i < m; i++) { for (let j = 0; j <= i; j++) { let s = L[i][j]; for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k]; L[i][j] = i === j ? Math.sqrt(Math.max(s, 1e-12)) : s / L[j][j]; } for (let j = i + 1; j < m; j++) L[i][j] = 0; }
      const solve = v => { const y = Float64Array.from(v); for (let i = 0; i < m; i++) { for (let k = 0; k < i; k++) y[i] -= L[i][k] * y[k]; y[i] /= L[i][i]; } for (let i = m - 1; i >= 0; i--) { for (let k = i + 1; k < m; k++) y[i] -= L[k][i] * y[k]; y[i] /= L[i][i]; } return y; };
      if (RHO) {
        // localised update on the grid: (rho o C_fd) (C_dd + alpha C_d)^-1, then back onto the basis
        const F = ens.map(logK), fm = meanOf(F), N2 = n * n, Cfd = Array.from({ length: N2 }, () => new Float64Array(m));
        for (let j = 0; j < Ne; j++) for (let k = 0; k < N2; k++) { const df = (F[j][k] - fm[k]) / (Ne - 1); for (let p = 0; p < m; p++) Cfd[k][p] += df * (D[j][p] - dm[p]); }
        for (let k = 0; k < N2; k++) for (let p = 0; p < m; p++) Cfd[k][p] *= RHO[k][p];
        ens = F.map((f, j) => {
          const w = solve(obsIdx.map((_, p) => dobs[p] + Math.sqrt(alpha) * sdObs * gaussFrom(re) - D[j][p]));
          return project(f.map((v, k) => v + Cfd[k].reduce((s, c, p) => s + c * w[p], 0)));
        });
      } else {
        ens = ens.map((u, j) => {
          const w = solve(obsIdx.map((_, p) => dobs[p] + Math.sqrt(alpha) * sdObs * gaussFrom(re) - D[j][p]));
          return u.map((v, q) => v + Cud[q].reduce((s, c, p) => s + c * w[p], 0));
        });
      }
      inflated = false;
      if (beta > 1 && spread(ens) < 0.5 * spread0) {    // inflation, only against collapse
        const mm = meanOf(ens); ens = ens.map(u => u.map((v, q) => mm[q] + beta * (v - mm[q]))); inflated = true;
      }
    }
    let inflated = false;
    function phis() {                          // 1/2 || C_d^-1/2 (G(m_j) - d) ||^2 per member
      return ens.map(u => { const y = g(u); let s = 0; obsIdx.forEach((i, p) => { s += 0.5 * ((y[i] - dobs[p]) / sdObs) ** 2; }); return s; });
    }
    let sumInv = 0, it = 0;
    const AS = { alphas: [], label: tag ? tag.replace(/^, /, "") : "first match", col: ["#4fa0ff", "#ffb74d", "#c792ea", "#ff6b6b"][alphaSeries.length % 4] };
    alphaSeries.push(AS);
    const maxIt = method === "areki" ? 20 : Na;
    while (it < maxIt) {
      let alpha;
      if (method === "esmda") alpha = Na;
      else {
        // Iglesias and Yang: 1/alpha = max(n_d / (2 mean Phi), sqrt(n_d / (2 var Phi))), capped at 1 - sum 1/alpha
        const ph = phis(), mu = ph.reduce((a, b) => a + b, 0) / Ne, va = ph.reduce((a, b) => a + (b - mu) ** 2, 0) / Ne;
        alpha = 1 / Math.min(Math.max(m / (2 * mu), Math.sqrt(m / (2 * va))), 1 - sumInv);
        if (it === maxIt - 1) alpha = 1 / (1 - sumInv);       // the last allowed iteration completes sum 1/alpha = 1
      }
      kalman(alpha); sumInv += 1 / alpha; it++; AS.alphas.push(alpha); drawAlpha(alphaSeries);
      const pm = g(meanOf(ens));
      drawMaps([logK(xiTrue), priorMeanK, logK(meanOf(ens))], ["true field", "prior mean", "mean, iteration " + it]);
      drawCurves([...ens.slice(0, 40).map(u => ({ y: g(u), col: "rgba(79,160,255,.25)" })), { y: pm, col: "#4fa0ff", w: 2.2 }, { y: truth.y, col: "#ff4d4d", w: 2 }, { y: obsY, dots: true }],
                 "ensemble (blue), true model (red), observed (red dots), iteration " + it + tag);
      let mis = 0; obsIdx.forEach((i, p) => { mis += (pm[i] - dobs[p]) ** 2; });
      log("&nbsp;&nbsp; iteration " + it + ": alpha = " + alpha.toFixed(2) + ", sum 1/alpha = " + Math.min(1, sumInv).toFixed(3) +
          ", RMS misfit of the posterior mean " + Math.sqrt(mis / m).toFixed(3) + ", spread of ln K " + spread(ens).toFixed(3) + (inflated ? " (inflated)" : ""));
      $(".prog").style.width = (100 * Math.min(1, sumInv)).toFixed(0) + "%";
      await new Promise(res => setTimeout(res, 350));
      if (sumInv >= 1 - 1e-9) { log("&nbsp;&nbsp; converged: sum of 1/alpha reached 1."); break; }
    }
    }
    const rounds = P_.refine | 0;
    await historyMatch("");
    for (let k = 1; k <= rounds; k++) {
      // refinement: simulate posterior members, add them to the training set, retrain CCR, match again
      log("5." + k + " Refinement round " + k + ": the simulator on 24 posterior members, added to the training set; CCR retrained; history matched again.");
      const newX = [], newY = []; let ssl = 0, stl = 0;
      for (let j = 0; j < 24; j++) { const u = ens[Math.floor(j * Ne / 24)], o = simulate(u); if (o) { newX.push(u.slice()); newY.push(o.y); } await tick(); }
      const mu2 = new Float64Array(NY); newY.forEach(y => { for (let j = 0; j < NY; j++) mu2[j] += y[j] / newY.length; });
      newX.forEach((u, q) => { const p = g(u); for (let j = 0; j < NY; j++) { ssl += (p[j] - newY[q][j]) ** 2; stl += (newY[q][j] - mu2[j]) ** 2; } });
      Xt.push(...newX); Yt.push(...newY);
      ccr = trainCCR(); g = xi => ccr.predict(xi);
      let ssa = 0; newX.forEach((u, q) => { const p = g(u); for (let j = 0; j < NY; j++) ssa += (p[j] - newY[q][j]) ** 2; });
      log("&nbsp;&nbsp; surrogate error on these posterior runs: RMS " + Math.sqrt(ssl / (newX.length * NY)).toFixed(4) + " before, " +
          Math.sqrt(ssa / (newX.length * NY)).toFixed(4) + " after retraining (" + Xt.length + " training runs, K = " + ccr.K + ").");
      drawLoss(ccr.sse, ccr.K, "CCR clustering after refinement round " + k);
      await historyMatch(", refinement round " + k);
    }
    stage(5, "done");
    // 6 verify with the simulator
    stage(6); log("6. Verifying with the simulator: the posterior mean, 12 posterior members and the same 12 prior members.");
    await tick();
    const xm = meanOf(ens), ver = simulate(xm), post = [], pri = [], re0 = rng(99);
    const ens0 = Array.from({ length: Ne }, () => Array.from({ length: MD }, () => gaussFrom(re0)));     // the starting ensemble again
    for (let j = 0; j < 12; j++) { const q = Math.floor(j * Ne / 12), a = simulate(ens[q]), b = simulate(ens0[q]); if (a) post.push(a.y); if (b) pri.push(b.y); await tick(); }
    const priorMean = g(meanOf(Array.from({ length: Ne }, () => Array.from({ length: MD }, () => 0))));
    drawCurves([...pri.map(y => ({ y, col: "rgba(170,170,180,.35)" })), ...post.map(y => ({ y, col: "rgba(79,160,255,.45)" })),
                { y: ver.y, col: "#2f7dff", w: 2.6 }, { y: truth.y, col: "#ff4d4d", w: 2 }, { y: obsY, dots: true }],
               "simulator: prior (grey), posterior (blue), posterior mean (thick), true (red)");
    drawMaps([logK(xiTrue), priorMeanK, logK(xm)], ["true field", "prior mean", "matched (posterior mean)"]);
    let e0 = 0, e1 = 0; obsIdx.forEach((i, p) => { e0 += (priorMean[i] - dobs[p]) ** 2; e1 += (ver.y[i] - dobs[p]) ** 2; });
    let fe = 0, f0 = 0; const lt = logK(xiTrue), lm = logK(xm), lp = new Float64Array(n * n);
    for (let k = 0; k < n * n; k++) { fe += (lm[k] - lt[k]) ** 2; f0 += (lp[k] - lt[k]) ** 2; }
    log("&nbsp;&nbsp; data misfit " + Math.sqrt(e0 / obsIdx.length).toFixed(3) + " (prior) → " + Math.sqrt(e1 / obsIdx.length).toFixed(3) +
        " (matched); permeability error " + Math.sqrt(f0 / (n * n)).toFixed(2) + " → " + Math.sqrt(fe / (n * n)).toFixed(2) + " (log units).");
    stage(6, "done");
    busy = false; $(".run").disabled = false;
  }
  $(".run").addEventListener("click", runAll);
  $(".truth").addEventListener("click", () => { truthSeed++; runAll(); });
  root.querySelectorAll("input").forEach(el => el.addEventListener("input", readParams));
  readParams();
  drawMaps([logK(new Float64Array(MD))], ["prior mean field"]);
  drawCurves([], "press Run the workflow");
  drawLoss([], 0, "CCR clustering (elbow)");
  drawAlpha([]);
})();
