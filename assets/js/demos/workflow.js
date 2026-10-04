/* workflow.js -- the end-to-end workflow of the paper, in miniature, in the
   browser, on a five-spot (injector in the centre, a producer in each corner):
     1 geology   ln K = sigma * sum_m xi_m phi_m(x), xi ~ N(0, I), six smooth
                 basis functions (a Gaussian prior);
     2 simulate  the hidden truth and 10 validation runs with the mixed-precision
                 simulator (nvrs_engine.js: Newton, FP64 FGMRES, two-stage CPR
                 with an FP32 multigrid V-cycle);
     3 surrogate two stages, as in the paper: an FNO or a PINO (an FNO trained
                 with the discrete mass-balance residual as an extra loss) maps
                 ln K to pressure and water saturation at 20 report times; the
                 states at each producer are then indexed and passed to the
                 Peaceman well model, a 1-D FNO over time or CCR (ccr.js,
                 trained here); the FNOs were trained offline on 900 runs of
                 the same simulator and are evaluated here (fno.js);
     4 validate  surrogate against simulator on the validation runs;
     5 invert    alpha-REKI (Iglesias and Yang) or ES-MDA on xi, with the
                 surrogate or the simulator itself as the forward model;
                 options: covariance localisation (on the grid, Gaspari-Cohn,
                 projected back onto the basis) and inflation;
     6 verify    the simulator on the matched model.                       */
(function () {
  "use strict";
  const root = document.getElementById("demo-workflow");
  if (!root || !window.NVRS) return;
  const E = window.NVRS, $ = s => root.querySelector(s);
  const BASE = (document.currentScript && document.currentScript.src) || "";
  const DATA = BASE.replace(/js\/demos\/workflow\.js.*$/, "data/");
  const mapCv = $("canvas.maps"), crvCv = $("canvas.curves"), lossCv = $("canvas.loss"), alphaCv = $("canvas.alpha");
  const mc = mapCv.getContext("2d"), cc = crvCv.getContext("2d"), lc = lossCv.getContext("2d"), ac = alphaCv.getContext("2d");
  const P_ = {};
  const drawAlpha = series => window.PLOTS && window.PLOTS.alpha(ac, alphaCv.width, alphaCv.height, series,
    (P_.method === 1 ? "ES-MDA: alpha = N_a at every assimilation" : "alpha-REKI: alpha against iteration") + " (left, log) and sum of 1/alpha (right)");
  const n = 16, NC = n * n, MD = 6, NR = 20, T_END = 300, SIG = 1.3, VISC = 5, NV = 10;
  const RT = Array.from({ length: NR }, (_, i) => (i + 1) * T_END / NR);
  // outputs per run: water cut of P1..P4, field oil rate / injection, log injector pressure
  const NP = 4, NY = (NP + 2) * NR;
  const C = n >> 1, INJ = C * n + C, PRODS = [n + 1, n + n - 2, (n - 2) * n + 1, (n - 2) * n + n - 2];
  const WELLS = [[INJ, "#ffffff"], ...PRODS.map(k => [k, "#ff00ff"])];
  let busy = false, truthSeed = 11, NET = null, netPromise = null, CCRM = null;

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
  const tick = () => new Promise(res => setTimeout(res, 0));
  // ---- geology: smooth basis
  const WAVES = [[1, 0], [0, 1], [1, 1], [2, 0], [0, 2], [2, 1]];
  const BASIS = WAVES.map(([kx, ky]) => {
    const b = new Float64Array(NC); let s = 0;
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const v = Math.cos(Math.PI * kx * (i + 0.5) / n) * Math.cos(Math.PI * ky * (j + 0.5) / n);
      b[j * n + i] = v; s += v * v;
    }
    s = Math.sqrt(s / NC); return b.map(v => v / s);
  });
  function logK(xi) {
    const f = new Float64Array(NC);
    for (let m = 0; m < MD; m++) for (let k = 0; k < NC; k++) f[k] += SIG * xi[m] * BASIS[m][k] / Math.sqrt(MD);
    return f;
  }
  function project(f) {                      // ln K field -> basis coefficients (exact for fields in the span)
    return BASIS.map(b => { let s = 0; for (let k = 0; k < NC; k++) s += f[k] * b[k]; return s / NC * Math.sqrt(MD) / SIG; });
  }
  function gc(d, L) {                        // Gaspari-Cohn taper, zero beyond the radius L (cells)
    const r = 2 * d / L;
    if (r >= 2) return 0;
    if (r <= 1) return (((-0.25 * r + 0.5) * r + 0.625) * r - 5 / 3) * r * r + 1;
    return ((((r / 12 - 0.5) * r + 0.625) * r + 5 / 3) * r - 5) * r + 4 - 2 / (3 * r);
  }
  // ---- the simulator
  function simulate(xi) {
    const M = E.model({ n, logK: logK(xi), visc: VISC, dt0: 0.5, wells: "fivespot" });
    const y = new Float64Array(NY), t0 = performance.now();
    for (let r = 0; r < NR; r++) {
      if (!E.step(M, { prec: "fp32", restart: 30, dtmax: 15, tstop: RT[r] })) return null;
      let qo = 0;
      M.prods.forEach((k, w) => { const q = E.wellRates(M, k); y[w * NR + r] = q.wcut; qo += q.qo; });
      y[NP * NR + r] = qo / M.Q; y[(NP + 1) * NR + r] = Math.log(Math.max(1e-6, M.p[M.inj]));
    }
    return { y, ms: performance.now() - t0 };
  }
  // ---- the surrogate: states (FNO or PINO), then the Peaceman well model (1-D FNO or CCR)
  function loadNets() {
    if (!netPromise) netPromise = (async () => {
      const W = await window.FNO.load(DATA + "wf_models.json", DATA + "wf_models.bin");
      const [cmeta, cbin] = await Promise.all([window.FNO.grab(DATA + "wf_ccr.json", "json"), window.FNO.grab(DATA + "wf_ccr.bin")]);
      NET = { meta: W.meta, fno: window.FNO.net2(W.get, "s.", { n, m: 6, L: 3, width: 16 }), pino: window.FNO.net2(W.get, "p.", { n, m: 6, L: 3, width: 16 }),
              well: window.FNO.net1(W.get, "w.", { len: NR, m: 8, L: 3, width: 24 }), ccr: { meta: cmeta, X: new Float32Array(cbin) } };
      return NET;
    })().catch(e => { netPromise = null; throw e; });
    return netPromise;
  }
  const NBS = PRODS.map(k => { const i = k % n, j = (k / n) | 0, r = []; for (const [di, dj] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) if (i + di >= 0 && i + di < n && j + dj >= 0 && j + dj < n) r.push((j + dj) * n + i + di); return r; });
  function states(lk, which) {               // -> P[t][cell], S[t][cell]
    const m = NET.meta, out = window.FNO.run2(which === 1 ? NET.pino : NET.fno, [lk.map(v => v / 1.8)]);
    return { P: out.slice(0, NR).map(a => a.map(v => Math.exp(v * m.psd + m.pmu) - 0.1)), S: out.slice(NR).map(a => a.map(v => Math.min(1, Math.max(0, (v + 1) / 2)))) };
  }
  function wellFeatures(st, lk, w) {         // five channels over time at producer w (as in training)
    const m = NET.meta, k = PRODS[w], nb = NBS[w], L = v => (Math.log(v + 0.1) - m.pmu) / m.psd;
    const ch = Array.from({ length: 5 }, () => new Float64Array(NR));
    for (let t = 0; t < NR; t++) {
      ch[0][t] = L(st.P[t][k]); ch[1][t] = st.S[t][k]; ch[2][t] = lk[k] / 1.8;
      ch[3][t] = nb.reduce((s, q) => s + st.S[t][q], 0) / nb.length;
      ch[4][t] = (nb.reduce((s, q) => s + Math.log(st.P[t][q] + 0.1), 0) / nb.length - m.pmu) / m.psd;
    }
    return ch;
  }
  function surrogate(xi) {
    const lk = logK(xi), st = states(lk, P_.states), y = new Float64Array(NY), qoT = new Float64Array(NR);
    for (let w = 0; w < NP; w++) {
      const ch = wellFeatures(st, lk, w);
      let qo, qw;
      if (P_.wells === 1) {                 // CCR, row by row
        qo = new Float64Array(NR); qw = new Float64Array(NR);
        for (let t = 0; t < NR; t++) { const r = ccrRates([ch[0][t], ch[1][t], ch[2][t], ch[3][t], ch[4][t]]); qo[t] = r[0]; qw[t] = r[1]; }
      } else {                              // 1-D FNO over the 20 report times
        const o = window.FNO.run1(NET.well, ch), s = NET.meta.wsc;
        qo = o[0].map(v => Math.max(0, v / s)); qw = o[1].map(v => Math.max(0, v / s));
      }
      for (let t = 0; t < NR; t++) { y[w * NR + t] = qo[t] + qw[t] > 1e-9 ? qw[t] / (qo[t] + qw[t]) : 0; qoT[t] += qo[t]; }
    }
    for (let t = 0; t < NR; t++) { y[NP * NR + t] = qoT[t]; y[(NP + 1) * NR + t] = Math.log(Math.max(1e-6, st.P[t][INJ])); }
    return y;
  }
  // physics-informed CCR features: the Peaceman rate is a well index times a Corey mobility times the drawdown,
  // so the mobilities of the well cell and of its neighbours are given to the experts explicitly
  const SWC = E.swc, SOR = E.sor;
  const mobs = S => { const se = Math.min(1, Math.max(0, (S - SWC) / (1 - SWC - SOR))); return [se * se, (1 - se) * (1 - se) / VISC]; };
  const aug = x => { const a = mobs(x[1]), b = mobs(x[3]); return [x[0], x[1], x[2], x[3], x[4], a[0], a[1], b[0], b[1], a[0] * Math.exp(x[2] * 1.8), a[1] * Math.exp(x[2] * 1.8)]; };
  const ccrRates = x => { const r = CCRM.predict(aug(x)); return [Math.max(0, r[0]), mobs(x[1])[0] > 0 ? Math.max(0, r[1]) : 0]; };   // no water before it is mobile
  async function trainCCR() {
    const Cd = NET.ccr, cols = Cd.meta.cols, A = Cd.X, X = [], Y = [];
    for (let i = 0; i < Cd.meta.ntrain; i += 2) { const o = i * cols; X.push(aug(Array.from(A.subarray(o, o + 5)))); Y.push(Array.from(A.subarray(o + 5, o + 7))); }
    await tick();
    CCRM = window.CCR.train(X, Y, { Kmax: 5, lam: 1e-3 });
    let ss = 0, st = 0; const te = [];
    for (let i = Cd.meta.ntrain; i < Cd.meta.ntrain + Cd.meta.ntest; i++) { const o = i * cols; te.push([Array.from(A.subarray(o, o + 5)), [A[o + 5], A[o + 6]]]); }
    const mu = [0, 1].map(j => te.reduce((s, r) => s + r[1][j], 0) / te.length);
    te.forEach(([x, y]) => { const p = ccrRates(x); for (let j = 0; j < 2; j++) { ss += (p[j] - y[j]) ** 2; st += (y[j] - mu[j]) ** 2; } });
    return { r2: 1 - ss / st, n: X.length };
  }
  // ---- drawing
  function jet(t) {
    const x = Math.min(1, Math.max(0, t)); return [1.5 - Math.abs(4 * x - 3), 1.5 - Math.abs(4 * x - 2), 1.5 - Math.abs(4 * x - 1)].map(v => Math.round(255 * Math.min(1, Math.max(0, v)))); }
  function drawMaps(fields, titles) {
    const W = mapCv.width, H = mapCv.height;
    mc.fillStyle = "#07070a"; mc.fillRect(0, 0, W, H);
    const k = fields.length, gap = 14, size = Math.min((W - gap * (k + 1)) / k, H - 40);
    fields.forEach((f, idx) => {
      const img = mc.createImageData(n, n);
      for (let q = 0; q < NC; q++) { const c = jet((f[q] + 2.5 * SIG / 1.2) / (5 * SIG / 1.2)); img.data.set([c[0], c[1], c[2], 255], 4 * q); }
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
  function drawParity(pairs, title) {        // surrogate against simulator, water cut and oil rate
    const W = lossCv.width, H = lossCv.height, l = 40, r = 12, t = 24, b = 26;
    lc.fillStyle = "#07070a"; lc.fillRect(0, 0, W, H);
    lc.font = "11px DM Mono, monospace"; lc.fillStyle = "#e8e6e3"; lc.fillText(title, 8, 14);
    const s = Math.min(W - l - r, H - t - b), X = v => l + s * v, Y = v => t + s * (1 - v);
    lc.strokeStyle = "rgba(255,255,255,.14)"; lc.strokeRect(l, t, s, s);
    lc.setLineDash([4, 4]); lc.beginPath(); lc.moveTo(X(0), Y(0)); lc.lineTo(X(1), Y(1)); lc.stroke(); lc.setLineDash([]);
    for (const [a, p] of pairs) { lc.fillStyle = "rgba(79,160,255,.55)"; lc.fillRect(X(Math.min(1, a)) - 1.2, Y(Math.min(1, Math.max(0, p))) - 1.2, 2.4, 2.4); }
    lc.fillStyle = "#9a9997"; lc.fillText("simulator", l + s / 2 - 28, H - 6); lc.fillText("surrogate", l + s + 8, t + 12);
    lc.fillText("0", l - 10, t + s + 4); lc.fillText("1", l - 10, t + 8);
  }
  function stage(k, status) {
    root.querySelectorAll(".stage").forEach(el => {
      const s = +el.getAttribute("data-step");
      el.classList.toggle("active", s === k && status !== "done");
      el.classList.toggle("done", s < k || (s === k && status === "done"));
    });
  }
  function log(msg) { const el = $(".wlog"); el.innerHTML += "<div>" + msg + "</div>"; el.scrollTop = el.scrollHeight; }
  const meanOf = A => A[0].map((_, j) => A.reduce((s, a) => s + a[j], 0) / A.length);

  async function runAll() {
    if (busy) return; busy = true; readParams();
    $(".wlog").innerHTML = ""; root.querySelectorAll("button").forEach(b => { b.disabled = true; });
    try {
      const useSim = P_.forward === 1, sName = P_.states === 1 ? "PINO" : "FNO", wName = P_.wells === 1 ? "CCR" : "1-D FNO";
      // 1-2 the truth and the validation runs
      stage(1); log("1. Prior: ln K from six smooth basis functions with Gaussian coefficients; the hidden truth is one more draw.");
      stage(2); log("2. The mixed-precision simulator (FP64 FGMRES, CPR with an FP32 V-cycle) on the hidden field and " + NV + " validation fields.");
      const rt = rng(truthSeed * 77 + 3), xiTrue = Array.from({ length: MD }, () => gaussFrom(rt)), truth = simulate(xiTrue);
      const rv = rng(4321), Xv = [], Yv = []; let simMs = truth.ms;
      for (let s = 0; s < NV; s++) {
        const xi = Array.from({ length: MD }, () => gaussFrom(rv)), o = simulate(xi);
        if (o) { Xv.push(xi); Yv.push(o.y); simMs += o.ms; }
        drawMaps([logK(xi)], ["validation field " + (s + 1)]); drawCurves([{ y: o ? o.y : truth.y, col: "#ff4d4d", w: 1.5 }], "simulated production, validation run " + (s + 1));
        $(".prog").style.width = (100 * (s + 1) / NV).toFixed(0) + "%"; await tick();
      }
      simMs /= Xv.length + 1;
      log("&nbsp;&nbsp; " + simMs.toFixed(0) + " ms per simulator run.");
      stage(2, "done");
      // 3 the surrogate
      stage(3); log("3. Surrogate: " + sName + " for pressure and saturation, then the Peaceman well model as " + wName + ".");
      await loadNets(); const st = NET.meta.stats;
      log("&nbsp;&nbsp; " + sName + " trained offline on " + NET.meta.ntrain + " simulator runs; on 60 held-out runs R2 of pressure " + st[sName.toLowerCase()].r2p.toFixed(3) +
          ", saturation " + st[sName.toLowerCase()].r2s.toFixed(3) + ", RMS mass-balance residual " + st[sName.toLowerCase()].residual.toExponential(2) +
          " (FNO " + st.fno.residual.toExponential(2) + ", PINO " + st.pino.residual.toExponential(2) + ").");
      if (P_.wells === 1) {
        if (!CCRM) { const c = await trainCCR(); log("&nbsp;&nbsp; CCR trained here on " + c.n + " (well, time) samples, with the Corey mobilities as physics features: K = " + CCRM.K + " regimes, held-out R2 of (q_o, q_w) " + c.r2.toFixed(3) + "."); }
      } else log("&nbsp;&nbsp; 1-D FNO well model trained offline; held-out R2 of (q_o, q_w) " + st.well1d.r2.toFixed(3) + ".");
      stage(3, "done");
      // 4 validate
      stage(4); log("4. Validating the surrogate against the simulator on the " + Xv.length + " validation runs.");
      const t0 = performance.now(), preds = Xv.map(surrogate), surMs = (performance.now() - t0) / Xv.length;
      let ss = 0, sst = 0; const mu = new Float64Array(NY), pairs = [];
      Yv.forEach(y => { for (let j = 0; j < NY; j++) mu[j] += y[j] / Yv.length; });
      preds.forEach((p, s) => { for (let j = 0; j < NY; j++) { ss += (p[j] - Yv[s][j]) ** 2; sst += (Yv[s][j] - mu[j]) ** 2; if (j < (NP + 1) * NR) pairs.push([Yv[s][j], p[j]]); } });
      drawParity(pairs, sName + " + " + wName + ": water cut and oil rate, surrogate against simulator");
      drawCurves([{ y: Yv[0], col: "#ff4d4d", w: 2.2 }, { y: preds[0], col: "#4fa0ff", w: 2.2 }], "validation run: simulator (red) against " + sName + " + " + wName + " (blue)");
      log("&nbsp;&nbsp; R2 = " + (1 - ss / sst).toFixed(3) + "; surrogate " + surMs.toFixed(1) + " ms against simulator " + simMs.toFixed(0) + " ms per run, " +
          Math.round(simMs / Math.max(surMs, 1e-3)) + " times faster.");
      stage(4, "done"); await tick();
      // 5 history match
      const method = P_.method === 1 ? "esmda" : "areki";
      const Ne = useSim ? Math.min(P_.ne | 0, 40) : P_.ne | 0, Na = P_.na | 0, sdObs = P_.noise;
      const g = useSim ? (xi => { const o = simulate(xi); return o ? o.y : null; }) : surrogate;
      stage(5); log("5. History matching with " + (method === "areki" ? "alpha-REKI" : "ES-MDA") + ", " + Ne + " members, forward model: " +
                    (useSim ? "the simulator itself (ensemble capped at 40)" : sName + " + " + wName) + ".");
      const obsIdx = Array.from({ length: NY }, (_, i) => i), m = NY;
      const dobs = obsIdx.map(i => truth.y[i] + sdObs * gaussFrom(rt));
      const obsY = Float64Array.from(dobs);
      const Lr = P_.loc | 0, beta = P_.infl || 1;
      const dist = (k, w) => Math.hypot((k % n) - (w % n), ((k / n) | 0) - ((w / n) | 0));
      const RHO = Lr > 0 ? Array.from({ length: NC }, (_, k) => Float64Array.from(obsIdx, i => {
        const grp = Math.floor(i / NR);
        const d = grp < NP ? dist(k, PRODS[grp]) : grp === NP ? Math.min(...PRODS.map(w => dist(k, w))) : dist(k, INJ);
        return gc(d, Lr); })) : null;
      const spread = E_ => { const F = E_.map(logK), mm = meanOf(F); let s = 0; for (let k = 0; k < NC; k++) s += Math.sqrt(F.reduce((a, f) => a + (f[k] - mm[k]) ** 2, 0) / F.length); return s / NC; };
      const re = rng(99);
      let ens = Array.from({ length: Ne }, () => Array.from({ length: MD }, () => gaussFrom(re)));
      const ens0 = ens.map(u => u.slice()), priorMeanK = logK(meanOf(ens)), spread0 = spread(ens);
      log("&nbsp;&nbsp; " + (Lr > 0 ? "localisation radius " + Lr + " cells (update on the grid, projected onto the basis)" : "no localisation") +
          (beta > 1 ? ", inflation " + beta.toFixed(2) + " whenever the spread falls below half the prior spread" : ", no inflation") + "; prior spread of ln K " + spread0.toFixed(3) + ".");
      let fwdN = 0, fwdMs = 0;
      async function evalAll(E_) {           // one forward evaluation per member per iteration
        const D = [], t1 = performance.now();
        for (let j = 0; j < E_.length; j++) {
          let y = g(E_[j]); if (!y) y = surrogateOr(E_[j]); D.push(y);
          if (useSim || j % 20 === 19) { $(".prog").style.width = (100 * (j + 1) / E_.length).toFixed(0) + "%"; await tick(); }
        }
        fwdN += E_.length; fwdMs += performance.now() - t1; return D;
      }
      const surrogateOr = xi => { const o = simulate(xi.map(v => v * 0.9)); return o ? o.y : new Float64Array(NY); };   // a failed run: shrink towards the prior mean
      let D = await evalAll(ens);
      drawMaps([logK(xiTrue), priorMeanK], ["true field", "prior mean"]);
      drawCurves([...D.slice(0, 40).map(y => ({ y, col: "rgba(79,160,255,.25)" })), { y: truth.y, col: "#ff4d4d", w: 2 }, { y: obsY, dots: true }],
                 "prior ensemble (blue), true model (red), observed (red dots)");
      const AS = { alphas: [], label: useSim ? "simulator" : sName + " + " + wName, col: "#4fa0ff" }; drawAlpha([AS]);
      let sumInv = 0, it = 0; const maxIt = method === "areki" ? 20 : Na;
      while (it < maxIt) {
        const ph = D.map(y => { let s = 0; for (let p = 0; p < m; p++) s += 0.5 * ((y[p] - dobs[p]) / sdObs) ** 2; return s; });
        let alpha = Na;
        if (method === "areki") {
          // Iglesias and Yang: 1/alpha = max(n_d / (2 mean Phi), sqrt(n_d / (2 var Phi))), capped at 1 - sum 1/alpha
          const mu_ = ph.reduce((a, b) => a + b, 0) / Ne, va = ph.reduce((a, b) => a + (b - mu_) ** 2, 0) / Ne;
          alpha = 1 / Math.min(Math.max(m / (2 * mu_), Math.sqrt(m / (2 * va))), 1 - sumInv);
          if (it === maxIt - 1) alpha = 1 / (1 - sumInv);     // the last allowed iteration completes sum 1/alpha = 1
        }
        // the ensemble Kalman update with inflated noise
        const dm = meanOf(D), Cdd = Array.from({ length: m }, () => new Float64Array(m));
        for (let j = 0; j < Ne; j++) for (let p = 0; p < m; p++) { const dp = D[j][p] - dm[p]; for (let q = p; q < m; q++) Cdd[p][q] += dp * (D[j][q] - dm[q]) / (Ne - 1); }
        for (let p = 0; p < m; p++) { for (let q = 0; q < p; q++) Cdd[p][q] = Cdd[q][p]; Cdd[p][p] += alpha * sdObs * sdObs; }
        const L = Cdd;
        for (let i = 0; i < m; i++) { for (let j = 0; j <= i; j++) { let s = L[i][j]; for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k]; L[i][j] = i === j ? Math.sqrt(Math.max(s, 1e-12)) : s / L[j][j]; } }
        const solve = v => { const y = Float64Array.from(v); for (let i = 0; i < m; i++) { for (let k = 0; k < i; k++) y[i] -= L[i][k] * y[k]; y[i] /= L[i][i]; } for (let i = m - 1; i >= 0; i--) { for (let k = i + 1; k < m; k++) y[i] -= L[k][i] * y[k]; y[i] /= L[i][i]; } return y; };
        const W_ = D.map(y => solve(obsIdx.map((_, p) => dobs[p] + Math.sqrt(alpha) * sdObs * gaussFrom(re) - y[p])));
        if (RHO) {                            // localised update on the grid, projected back onto the basis
          const F = ens.map(logK), fm = meanOf(F), Cfd = Array.from({ length: NC }, () => new Float64Array(m));
          for (let j = 0; j < Ne; j++) for (let k = 0; k < NC; k++) { const df = (F[j][k] - fm[k]) / (Ne - 1); for (let p = 0; p < m; p++) Cfd[k][p] += df * (D[j][p] - dm[p]) * RHO[k][p]; }
          ens = F.map((f, j) => project(f.map((v, k) => { let s = 0; const c = Cfd[k], w = W_[j]; for (let p = 0; p < m; p++) s += c[p] * w[p]; return v + s; })));
        } else {
          const um = meanOf(ens), Cud = Array.from({ length: MD }, () => new Float64Array(m));
          for (let j = 0; j < Ne; j++) for (let q = 0; q < MD; q++) { const du = (ens[j][q] - um[q]) / (Ne - 1); for (let p = 0; p < m; p++) Cud[q][p] += du * (D[j][p] - dm[p]); }
          ens = ens.map((u, j) => u.map((v, q) => { let s = 0; for (let p = 0; p < m; p++) s += Cud[q][p] * W_[j][p]; return v + s; }));
        }
        let inflated = false;
        if (beta > 1 && spread(ens) < 0.5 * spread0) { const mm = meanOf(ens); ens = ens.map(u => u.map((v, q) => mm[q] + beta * (v - mm[q]))); inflated = true; }
        sumInv += 1 / alpha; it++; AS.alphas.push(alpha); drawAlpha([AS]);
        D = await evalAll(ens);
        const dmn = meanOf(D);
        drawMaps([logK(xiTrue), priorMeanK, logK(meanOf(ens))], ["true field", "prior mean", "mean, iteration " + it]);
        drawCurves([...D.slice(0, 40).map(y => ({ y, col: "rgba(79,160,255,.25)" })), { y: dmn, col: "#4fa0ff", w: 2.2 }, { y: truth.y, col: "#ff4d4d", w: 2 }, { y: obsY, dots: true }],
                   "ensemble (blue), true model (red), observed (red dots), iteration " + it);
        let mis = 0; for (let p = 0; p < m; p++) mis += (dmn[p] - dobs[p]) ** 2;
        log("&nbsp;&nbsp; iteration " + it + ": alpha = " + alpha.toFixed(2) + ", sum 1/alpha = " + Math.min(1, sumInv).toFixed(3) +
            ", RMS misfit of the ensemble mean prediction " + Math.sqrt(mis / m).toFixed(3) + ", spread of ln K " + spread(ens).toFixed(3) + (inflated ? " (inflated)" : ""));
        await tick();
        if (method === "areki" && sumInv >= 1 - 1e-9) { log("&nbsp;&nbsp; converged: sum of 1/alpha reached 1."); break; }
      }
      log("&nbsp;&nbsp; " + fwdN + " forward evaluations in " + (fwdMs / 1000).toFixed(1) + " s (" + (fwdMs / fwdN).toFixed(1) + " ms each)" +
          (useSim ? "" : "; with the simulator they would take about " + (fwdN * simMs / 1000).toFixed(0) + " s."));
      stage(5, "done");
      // 6 verify with the simulator
      stage(6); log("6. Verifying with the simulator: the posterior mean, 12 posterior members and the same 12 prior members.");
      await tick();
      const xm = meanOf(ens), ver = simulate(xm), post = [], pri = [];
      for (let j = 0; j < 12; j++) { const q = Math.floor(j * Ne / 12), a = simulate(ens[q]), b = simulate(ens0[q]); if (a) post.push(a.y); if (b) pri.push(b.y); await tick(); }
      drawCurves([...pri.map(y => ({ y, col: "rgba(170,170,180,.35)" })), ...post.map(y => ({ y, col: "rgba(79,160,255,.45)" })),
                  { y: ver.y, col: "#2f7dff", w: 2.6 }, { y: truth.y, col: "#ff4d4d", w: 2 }, { y: obsY, dots: true }],
                 "simulator: prior (grey), posterior (blue), posterior mean (thick), true (red)");
      drawMaps([logK(xiTrue), priorMeanK, logK(xm)], ["true field", "prior mean", "matched (posterior mean)"]);
      const pm0 = simulate(new Array(MD).fill(0));
      let e0 = 0, e1 = 0; for (let p = 0; p < m; p++) { e0 += (pm0.y[p] - dobs[p]) ** 2; e1 += (ver.y[p] - dobs[p]) ** 2; }
      let fe = 0, f0 = 0; const lt = logK(xiTrue), lm = logK(xm);
      for (let k = 0; k < NC; k++) { fe += (lm[k] - lt[k]) ** 2; f0 += lt[k] ** 2; }
      log("&nbsp;&nbsp; data misfit " + Math.sqrt(e0 / m).toFixed(3) + " (prior mean) → " + Math.sqrt(e1 / m).toFixed(3) +
          " (matched); permeability error " + Math.sqrt(f0 / NC).toFixed(2) + " → " + Math.sqrt(fe / NC).toFixed(2) + " (log units).");
      stage(6, "done");
    } catch (e) { log("Stopped: " + (e && e.message ? e.message : e) + ". Please try again."); }
    finally { busy = false; root.querySelectorAll("button").forEach(b => { b.disabled = false; }); }
  }
  $(".run").addEventListener("click", runAll);
  $(".truth").addEventListener("click", () => { truthSeed++; runAll(); });
  root.querySelectorAll("input,select").forEach(el => el.addEventListener("input", readParams));
  readParams();
  drawMaps([logK(new Float64Array(MD))], ["prior mean field"]);
  drawCurves([], "press Run the workflow");
  drawParity([], "surrogate against simulator (after validation)");
  drawAlpha([]);
  window.__WF = { loadNets, surrogate, states, get NET() { return NET; } };
})();
