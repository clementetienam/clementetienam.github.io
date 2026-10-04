/* workflow.js -- the end-to-end surrogate workflow, in miniature, in the
   browser:
     1 geology   permeability fields log K = sigma * sum_m xi_m phi_m(x),
                 xi ~ N(0, I), six smooth basis functions;
     2 simulate  each realisation run to the end of the schedule with the
                 mixed-precision simulator (nvrs_engine.js: Newton, FP64
                 FGMRES, two-stage CPR with an FP32 multigrid V-cycle);
                 recorded: producer water cut, oil rate, injector pressure;
     3 train     a neural surrogate xi -> production (an MLP, Adam), live;
     4 validate  on held-out simulations: accuracy and measured speed-up;
     5 invert    alpha-REKI (adaptive alpha from the data misfit, stopped when
                 sum 1/alpha = 1) or ES-MDA (alpha = number of assimilations) on
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
  const mapCv = $("canvas.maps"), crvCv = $("canvas.curves"), lossCv = $("canvas.loss");
  const mc = mapCv.getContext("2d"), cc = crvCv.getContext("2d"), lc = lossCv.getContext("2d");
  const P_ = {};
  const n = 16, MD = 6, NR = 20, T_END = 300, SIG = 1.3, VISC = 5;
  const RT = Array.from({ length: NR }, (_, i) => (i + 1) * T_END / NR);
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
  function logK(xi) {
    const f = new Float64Array(n * n);
    for (let m = 0; m < MD; m++) for (let k = 0; k < n * n; k++) f[k] += SIG * xi[m] * BASIS[m][k] / Math.sqrt(MD);
    return f;
  }
  // ---- simulate one realisation: 3 x NR outputs
  function simulate(xi) {
    const M = E.model({ n, logK: logK(xi), visc: VISC, dt0: 0.5 });
    const y = new Float64Array(3 * NR), t0 = performance.now();
    let kry = 0, newt = 0;
    for (let r = 0; r < NR; r++) {
      while (M.t < RT[r] - 1e-9) {
        const res = E.step(M, { prec: "fp32", restart: 30, dtmax: 15, tstop: RT[r] });
        if (!res) return null;
        kry += res.krylov; newt += res.newton;
      }
      const w = E.wellRates(M);
      y[r] = w.wcut; y[NR + r] = w.qo / M.Q; y[2 * NR + r] = Math.log(Math.max(1e-6, w.pinj));
    }
    return { y, ms: performance.now() - t0, kry, newt };
  }
  // ---- MLP surrogate
  function mlpInit(sizes, seed) {
    const r = rng(seed), L = [];
    for (let l = 0; l < sizes.length - 1; l++) {
      const a = sizes[l], b = sizes[l + 1], sc = Math.sqrt(2 / (a + b));
      L.push({ W: Float64Array.from({ length: a * b }, () => gaussFrom(r) * sc), b: new Float64Array(b), a, o: b });
    }
    for (const l of L) { l.mW = new Float64Array(l.W.length); l.vW = new Float64Array(l.W.length); l.mb = new Float64Array(l.o); l.vb = new Float64Array(l.o); }
    return { L, t: 0 };
  }
  function mlpFwd(net, x) {
    const acts = [x];
    let h = x;
    net.L.forEach((l, li) => {
      const z = new Float64Array(l.o);
      for (let j = 0; j < l.o; j++) { let s = l.b[j]; for (let i = 0; i < l.a; i++) s += l.W[j * l.a + i] * h[i]; z[j] = li < net.L.length - 1 ? Math.tanh(s) : s; }
      acts.push(z); h = z;
    });
    return acts;
  }
  function mlpEpoch(net, X, Y, lr) {
    const gW = net.L.map(l => new Float64Array(l.W.length)), gb = net.L.map(l => new Float64Array(l.o));
    let loss = 0;
    for (let s = 0; s < X.length; s++) {
      const acts = mlpFwd(net, X[s]), out = acts[acts.length - 1];
      let d = out.map((v, j) => v - Y[s][j]);
      for (const v of d) loss += v * v;
      for (let li = net.L.length - 1; li >= 0; li--) {
        const l = net.L[li], hin = acts[li];
        if (li < net.L.length - 1) d = d.map((v, j) => v * (1 - acts[li + 1][j] ** 2));
        const dn = new Float64Array(l.a);
        for (let j = 0; j < l.o; j++) {
          gb[li][j] += d[j];
          for (let i = 0; i < l.a; i++) { gW[li][j * l.a + i] += d[j] * hin[i]; dn[i] += l.W[j * l.a + i] * d[j]; }
        }
        d = dn;
      }
    }
    net.t++;
    const b1 = 0.9, b2 = 0.999, c1 = 1 - Math.pow(b1, net.t), c2 = 1 - Math.pow(b2, net.t), sc = 2 / X.length;
    net.L.forEach((l, li) => {
      const upd = (P, G, Mm, V) => { for (let k = 0; k < P.length; k++) { const g = G[k] * sc; Mm[k] = b1 * Mm[k] + (1 - b1) * g; V[k] = b2 * V[k] + (1 - b2) * g * g; P[k] -= lr * (Mm[k] / c1) / (Math.sqrt(V[k] / c2) + 1e-8); } };
      upd(l.W, gW[li], l.mW, l.vW); upd(l.b, gb[li], l.mb, l.vb);
    });
    return loss / (X.length * Y[0].length);
  }
  // ---- drawing
  function viridis(x) {
    const st = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
    x = Math.min(0.999, Math.max(0, x)) * 4; const i = Math.floor(x), f = x - i;
    return st[i].map((q, c) => Math.round(q + (st[i + 1][c] - q) * f));
  }
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
      for (const [cell, col] of [[2 * n + 2, "#4fc3ff"], [(n - 3) * n + (n - 3), "#00e5a0"]]) {
        mc.beginPath(); mc.arc(x + ((cell % n) + 0.5) * size / n, 28 + (Math.floor(cell / n) + 0.5) * size / n, 5, 0, 7);
        mc.fillStyle = col; mc.fill(); mc.strokeStyle = "#000"; mc.stroke();
      }
    });
  }
  function drawCurves(sets, title) {        // sets: [{y, col, w, dots}]
    const W = crvCv.width, H = crvCv.height, l = 44, r = 10, t = 26, b = 26, panels = [["water cut", 0], ["oil rate / injection", NR]];
    cc.fillStyle = "#07070a"; cc.fillRect(0, 0, W, H);
    cc.font = "11px DM Mono, monospace"; cc.fillStyle = "#e8e6e3"; cc.fillText(title, l, 15);
    const pw = (W - l - r) / 2 - 10;
    panels.forEach(([lab, off], pi) => {
      const x0 = l + pi * (pw + 20), X = v => x0 + pw * v / T_END, Y = v => t + (H - t - b) * (1 - v);
      cc.strokeStyle = "rgba(255,255,255,.12)"; cc.strokeRect(x0, t, pw, H - t - b);
      cc.fillStyle = "#9a9997"; cc.fillText(lab, x0 + 4, t + 14); cc.fillText("0", x0 - 10, H - b + 4); cc.fillText("1", x0 - 10, t + 4);
      cc.fillText("time", x0 + pw - 30, H - 8);
      for (const s of sets) {
        cc.beginPath(); cc.strokeStyle = s.col; cc.lineWidth = s.w || 1;
        RT.forEach((tt, i) => { const v = Math.min(1.05, Math.max(0, s.y[off + i])); i ? cc.lineTo(X(tt), Y(v)) : cc.moveTo(X(tt), Y(v)); });
        if (!s.dots) cc.stroke();
        if (s.dots) RT.forEach((tt, i) => { cc.beginPath(); cc.arc(X(tt), Y(Math.min(1.05, Math.max(0, s.y[off + i]))), 3, 0, 7); cc.fillStyle = "#fff"; cc.fill(); });
      }
    });
  }
  function drawLoss(h, title) {
    const W = lossCv.width, H = lossCv.height, l = 50, r = 10, t = 24, b = 24;
    lc.fillStyle = "#07070a"; lc.fillRect(0, 0, W, H);
    lc.font = "11px DM Mono, monospace"; lc.fillStyle = "#e8e6e3"; lc.fillText(title, l, 14);
    if (!h.length) return;
    const lo = -4, hi = 0.5, X = i => l + (W - l - r) * i / Math.max(h.length - 1, 1), Y = v => t + (H - t - b) * (hi - Math.log10(Math.max(v, 1e-4))) / (hi - lo);
    lc.strokeStyle = "rgba(255,255,255,.1)"; lc.fillStyle = "#9a9997";
    for (let e = 0; e >= -4; e--) { lc.beginPath(); lc.moveTo(l, Y(Math.pow(10, e))); lc.lineTo(W - r, Y(Math.pow(10, e))); lc.stroke(); lc.fillText("1e" + e, 10, Y(Math.pow(10, e)) + 4); }
    lc.beginPath(); lc.strokeStyle = "#00e5a0"; lc.lineWidth = 2;
    h.forEach((v, i) => i ? lc.lineTo(X(i), Y(v)) : lc.moveTo(X(i), Y(v))); lc.stroke();
    lc.fillStyle = "#9a9997"; lc.fillText("epochs", W - r - 50, H - 6);
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
  // standardisation helpers
  function stats(Y) {
    const m = Y[0].length, mu = new Float64Array(m), sd = new Float64Array(m);
    for (const y of Y) for (let j = 0; j < m; j++) mu[j] += y[j] / Y.length;
    for (const y of Y) for (let j = 0; j < m; j++) sd[j] += (y[j] - mu[j]) ** 2 / Y.length;
    for (let j = 0; j < m; j++) sd[j] = Math.sqrt(sd[j]) || 1;
    return { mu, sd };
  }
  async function runAll() {
    if (busy) return; busy = true; readParams();
    $(".wlog").innerHTML = ""; $(".run").disabled = true;
    const r = rng(1234 + (P_.runs | 0));
    // 1-2 sample and simulate
    stage(1); log("1. Sampling " + ((P_.runs | 0) + 10) + " permeability realisations from the prior (" + (P_.runs | 0) + " to train on, 10 to validate).");
    const X = [], Y = [], simMs = []; let kry = 0, newt = 0;
    stage(2); log("2. Simulating each one with the mixed-precision simulator (FP64 FGMRES, CPR with an FP32 V-cycle).");
    const total = (P_.runs | 0) + 10;
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
    log("&nbsp;&nbsp; " + X.length + " runs (" + Math.max(0, X.length - 10) + " for training, 10 held out), " + newt + " Newton and " + kry + " FGMRES iterations, " + simAvg.toFixed(0) + " ms per run.");
    stage(2, "done");
    // 3 train
    stage(3); log("3. Training the surrogate (MLP 6 -> 48 -> 48 -> " + 3 * NR + ", Adam) on " + (X.length - 10) + " runs.");
    const Xt = X.slice(0, X.length - 10), Yt = Y.slice(0, Y.length - 10), Xv = X.slice(-10), Yv = Y.slice(-10);
    const sY = stats(Yt), norm = y => y.map((v, j) => (v - sY.mu[j]) / sY.sd[j]), denorm = z => z.map((v, j) => v * sY.sd[j] + sY.mu[j]);
    const Ytn = Yt.map(norm), net = mlpInit([MD, 48, 48, 3 * NR], 7);
    const lossH = [], E_ = P_.epochs | 0;
    for (let ep = 0; ep < E_; ep++) {
      lossH.push(mlpEpoch(net, Xt, Ytn, 3e-3));
      if (ep % 25 === 0 || ep === E_ - 1) { drawLoss(lossH, "training loss (normalised MSE), epoch " + (ep + 1)); $(".prog").style.width = (100 * (ep + 1) / E_).toFixed(0) + "%"; await tick(); }
    }
    const g = xi => denorm(mlpFwd(net, xi).pop());
    log("&nbsp;&nbsp; final training loss " + lossH[lossH.length - 1].toExponential(2) + ".");
    stage(3, "done");
    // 4 validate
    stage(4); log("4. Validating on 10 held-out simulations.");
    let ss = 0, st = 0; const mu = new Float64Array(3 * NR);
    for (const y of Yv) for (let j = 0; j < 3 * NR; j++) mu[j] += y[j] / Yv.length;
    const t0 = performance.now(); const preds = Xv.map(g); const surMs = (performance.now() - t0) / Xv.length;
    preds.forEach((p, s) => { for (let j = 0; j < 3 * NR; j++) { ss += (p[j] - Yv[s][j]) ** 2; st += (Yv[s][j] - mu[j]) ** 2; } });
    const R2 = 1 - ss / st;
    drawCurves([{ y: Yv[0], col: "#4fc3ff", w: 2.2 }, { y: preds[0], col: "#ffb74d", w: 2.2 }], "held-out run: simulator (blue) against surrogate (orange)");
    log("&nbsp;&nbsp; R2 = " + R2.toFixed(3) + " on unseen runs; surrogate " + surMs.toFixed(3) + " ms against simulator " + simAvg.toFixed(0) +
        " ms per run, " + Math.round(simAvg / Math.max(surMs, 1e-3)) + " times faster.");
    stage(4, "done"); await tick();
    // 5 history match: alpha-REKI (adaptive alpha, stop when sum 1/alpha = 1) or ES-MDA (alpha = Na)
    const method = P_.method === 1 ? "esmda" : "areki";
    stage(5); log("5. History matching with " + (method === "areki" ? "alpha-REKI (adaptive regularised ensemble Kalman inversion)" : "ES-MDA") +
                  " and the surrogate, " + (P_.ne | 0) + " members.");
    const rt = rng(truthSeed * 77 + 3), xiTrue = Array.from({ length: MD }, () => gaussFrom(rt));
    const truth = simulate(xiTrue), sdObs = P_.noise;
    const obsIdx = Array.from({ length: 2 * NR }, (_, i) => i);              // water cut and oil rate
    const dobs = obsIdx.map(i => truth.y[i] + sdObs * gaussFrom(rt));
    const re = rng(99), Ne = P_.ne | 0, Na = P_.na | 0, m = obsIdx.length;
    let ens = Array.from({ length: Ne }, () => Array.from({ length: MD }, () => gaussFrom(re)));
    const meanOf = A => A[0].map((_, j) => A.reduce((s, a) => s + a[j], 0) / A.length);
    const obsY = new Float64Array(3 * NR); obsIdx.forEach((i, p) => { obsY[i] = dobs[p]; });
    drawMaps([logK(xiTrue), logK(meanOf(ens))], ["true field (hidden)", "prior mean"]);
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
      ens = ens.map((u, j) => {
        const w = solve(obsIdx.map((_, p) => dobs[p] + Math.sqrt(alpha) * sdObs * gaussFrom(re) - D[j][p]));
        return u.map((v, q) => v + Cud[q].reduce((s, c, p) => s + c * w[p], 0));
      });
    }
    function phiBar() {                        // mean of 1/2 || C_d^-1/2 (G(m_j) - d) ||^2
      let s = 0;
      for (const u of ens) { const y = g(u); obsIdx.forEach((i, p) => { s += 0.5 * ((y[i] - dobs[p]) / sdObs) ** 2; }); }
      return s / Ne;
    }
    let sumInv = 0, alphaPrev = Infinity, it = 0;
    const maxIt = method === "areki" ? 20 : Na;
    while (it < maxIt) {
      let alpha;
      if (method === "esmda") alpha = Na;
      else {
        alpha = Math.min(m / (2 * phiBar()), 0.9 * alphaPrev);
        alpha = Math.max(alpha, 1);
        if (sumInv + 1 / alpha >= 1) alpha = 1 / (1 - sumInv);   // land exactly on sum 1/alpha = 1
      }
      kalman(alpha); sumInv += 1 / alpha; alphaPrev = alpha; it++;
      const pm = g(meanOf(ens));
      drawMaps([logK(xiTrue), logK(meanOf(ens))], ["true field (hidden)", "posterior mean, iteration " + it]);
      drawCurves([...ens.slice(0, 40).map(u => ({ y: g(u), col: "rgba(0,229,160,.18)" })), { y: pm, col: "#00e5a0", w: 2.2 }, { y: obsY, dots: true }],
                 "observed (dots) and ensemble predictions, iteration " + it);
      let mis = 0; obsIdx.forEach((i, p) => { mis += (pm[i] - dobs[p]) ** 2; });
      log("&nbsp;&nbsp; iteration " + it + ": alpha = " + alpha.toFixed(2) + ", sum 1/alpha = " + Math.min(1, sumInv).toFixed(3) +
          ", RMS misfit of the posterior mean " + Math.sqrt(mis / m).toFixed(3));
      $(".prog").style.width = (100 * Math.min(1, sumInv)).toFixed(0) + "%";
      await new Promise(res => setTimeout(res, 350));
      if (sumInv >= 1 - 1e-9) { log("&nbsp;&nbsp; converged: sum of 1/alpha reached 1."); break; }
    }
    stage(5, "done");
    // 6 verify with the simulator
    stage(6); log("6. Verifying: the simulator run on the matched model.");
    const xm = meanOf(ens), ver = simulate(xm);
    const priorMean = g(meanOf(Array.from({ length: Ne }, () => Array.from({ length: MD }, () => 0))));
    drawCurves([{ y: priorMean, col: "rgba(255,255,255,.35)", w: 1.5 }, { y: ver.y, col: "#ffb74d", w: 2.4 }, { y: obsY, dots: true }],
               "simulator on the matched model (orange), prior mean (grey), observed (dots)");
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
  drawLoss([], "training loss");
})();
