/* waterflood.js -- a two-phase (water/oil) waterflood on a heterogeneous
   permeability field, solved in the browser with IMPES: an implicit pressure
   equation  -div(lambda_t K grad p) = q  (SOR, warm-started), then an explicit
   upwind saturation update  phi dS/dt + div(f_w(S) u) = q_w.  Quarter
   five-spot: injector in one corner, producer in the other.  This is the
   forward problem a neural-operator surrogate learns to replace.          */
(function () {
  "use strict";
  const root = document.getElementById("demo-waterflood");
  if (!root) return;
  const $ = s => root.querySelector(s);
  const map = $("canvas.map"), plot = $("canvas.plot");
  const mc = map.getContext("2d"), pc = plot.getContext("2d");
  const N = 56, NN = N * N;

  let K, phi, P, S, lamT, Tx, Ty, inj, prod, t, pvi, hist, running = false, raf = 0;
  const P_ = {};

  function rng(seed) {           // mulberry32
    let a = seed >>> 0;
    return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  }
  function readParams() {
    root.querySelectorAll("input,select").forEach(el => {
      P_[el.name] = el.type === "range" ? parseFloat(el.value) : el.value;
      const v = root.querySelector('[data-for="' + el.name + '"]');
      if (v) v.textContent = el.type === "range" ? (+el.value).toFixed(el.step && el.step < 1 ? 1 : 0) : el.value;
    });
  }
  function field() {
    // log-permeability: a sum of random Fourier modes with a Gaussian
    // spectrum of correlation length L (cells), plus a channel option.
    const r = rng(P_.seed * 7919 + 13), L = P_.corr, sig = P_.sigma;
    const M = 48, modes = [];
    for (let m = 0; m < M; m++) {
      const kx = (r() * 2 - 1), ky = (r() * 2 - 1);
      const kk = Math.hypot(kx, ky) + 1e-9;
      const w = Math.exp(-0.5 * (kk * L / 6) ** 2);
      modes.push([kx * Math.PI * 2 / 6, ky * Math.PI * 2 / 6, r() * Math.PI * 2, w * (0.5 + r())]);
    }
    const g = new Float64Array(NN);
    let mean = 0;
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
      let v = 0;
      for (const [kx, ky, ph, w] of modes) v += w * Math.cos(kx * i + ky * j + ph);
      g[j * N + i] = v; mean += v;
    }
    mean /= NN; let sd = 0;
    for (let k = 0; k < NN; k++) sd += (g[k] - mean) ** 2;
    sd = Math.sqrt(sd / NN) || 1;
    K = new Float64Array(NN); phi = new Float64Array(NN);
    for (let k = 0; k < NN; k++) {
      const z = (g[k] - mean) / sd;
      K[k] = Math.exp(sig * z);
      phi[k] = 0.18 + 0.04 * Math.tanh(0.5 * z);
    }
    if (P_.barrier === "fault") {               // a sealing fault with a gap
      for (let j = 0; j < N; j++) if (j < N * 0.7) { const i = Math.floor(N * 0.5); K[j * N + i] *= 1e-4; }
    }
  }
  function reset() {
    readParams(); field();
    S = new Float64Array(NN).fill(0.1);       // connate water
    P = new Float64Array(NN);
    lamT = new Float64Array(NN);
    Tx = new Float64Array(NN); Ty = new Float64Array(NN);
    inj = 2 * N + 2; prod = (N - 3) * N + (N - 3);
    t = 0; pvi = 0; hist = [];
    draw(); drawPlot();
  }
  const swc = 0.1, sor = 0.15;
  function fw(s) {
    const se = Math.min(1, Math.max(0, (s - swc) / (1 - swc - sor)));
    const krw = se * se, kro = (1 - se) * (1 - se);
    const mw = krw, mo = kro / P_.visc;
    return [mw + mo > 0 ? mw / (mw + mo) : 0, mw + mo];
  }
  function step() {
    // mobilities and transmissibilities (harmonic averages)
    for (let k = 0; k < NN; k++) lamT[k] = fw(S[k])[1] * K[k];
    for (let j = 0; j < N; j++) for (let i = 0; i < N - 1; i++) {
      const a = j * N + i, b = a + 1;
      Tx[a] = 2 * lamT[a] * lamT[b] / (lamT[a] + lamT[b] + 1e-30);
    }
    for (let j = 0; j < N - 1; j++) for (let i = 0; i < N; i++) {
      const a = j * N + i, b = a + N;
      Ty[a] = 2 * lamT[a] * lamT[b] / (lamT[a] + lamT[b] + 1e-30);
    }
    // pressure: SOR with p = 0 at the producer, unit rate at the injector
    const Q = 1, w = 1.85;
    for (let it = 0; it < 70; it++) {
      for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
        const k = j * N + i;
        if (k === prod) { P[k] = 0; continue; }
        let num = k === inj ? Q : 0, den = 0, T;
        if (i > 0) { T = Tx[k - 1]; num += T * P[k - 1]; den += T; }
        if (i < N - 1) { T = Tx[k]; num += T * P[k + 1]; den += T; }
        if (j > 0) { T = Ty[k - N]; num += T * P[k - N]; den += T; }
        if (j < N - 1) { T = Ty[k]; num += T * P[k + N]; den += T; }
        if (den > 0) P[k] += w * (num / den - P[k]);
      }
    }
    // explicit upwind saturation with a CFL step
    const out = new Float64Array(NN), wIn = new Float64Array(NN);
    const fl = new Float64Array(NN);
    for (let k = 0; k < NN; k++) fl[k] = fw(S[k])[0];
    function edge(a, b, T) {
      const F = T * (P[a] - P[b]);              // flux a -> b
      if (F > 0) { out[a] += F; wIn[b] += F * fl[a]; wIn[a] -= F * fl[a]; }
      else { out[b] -= F; wIn[a] += -F * fl[b]; wIn[b] -= -F * fl[b]; }
    }
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
      const k = j * N + i;
      if (i < N - 1) edge(k, k + 1, Tx[k]);
      if (j < N - 1) edge(k, k + N, Ty[k]);
    }
    out[inj] += 0; wIn[inj] += Q;              // injected water
    const qprod = Q;                             // producer withdraws Q
    let dt = Infinity;
    for (let k = 0; k < NN; k++) {
      const o = k === prod ? qprod : out[k];
      if (o > 0) dt = Math.min(dt, phi[k] / o);
    }
    dt *= 0.25;
    const fprod = fl[prod];
    wIn[prod] -= qprod * fprod;
    for (let k = 0; k < NN; k++) S[k] = Math.min(1 - sor, Math.max(swc, S[k] + dt * wIn[k] / phi[k]));
    let pv = 0;
    for (let k = 0; k < NN; k++) pv += phi[k];
    t += dt; pvi += Q * dt / pv;
    let oip = 0;
    for (let k = 0; k < NN; k++) oip += phi[k] * (1 - S[k]);
    hist.push([pvi, fprod, 1 - oip / (pv * (1 - swc))]);
  }
  const jet = t => { const x = Math.min(1, Math.max(0, t)); return [1.5 - Math.abs(4 * x - 3), 1.5 - Math.abs(4 * x - 2), 1.5 - Math.abs(4 * x - 1)].map(v => Math.round(255 * Math.min(1, Math.max(0, v)))); }
  const cmapW = s => jet((s - swc) / (1 - swc - sor));     // jet: oil (blue) to water (red)
  const cmapK = z => jet(z);
  function draw() {
    const W = map.width, H = map.height, img = mc.createImageData(N, N);
    let lo = Infinity, hi = -Infinity;
    for (let k = 0; k < NN; k++) { const l = Math.log(K[k]); if (l > -8) { lo = Math.min(lo, l); hi = Math.max(hi, l); } }
    for (let k = 0; k < NN; k++) {
      const c = P_.view === "perm" ? cmapK((Math.log(K[k]) - lo) / (hi - lo + 1e-9)) :
                P_.view === "pressure" ? cmapK(P[k] / (P[inj] || 1)) : cmapW(S[k]);
      img.data[4 * k] = c[0]; img.data[4 * k + 1] = c[1]; img.data[4 * k + 2] = c[2]; img.data[4 * k + 3] = 255;
    }
    const off = document.createElement("canvas"); off.width = N; off.height = N;
    off.getContext("2d").putImageData(img, 0, 0);
    mc.imageSmoothingEnabled = true;
    mc.drawImage(off, 0, 0, W, H);
    const cx = k => ((k % N) + 0.5) * W / N, cy = k => (Math.floor(k / N) + 0.5) * H / N;
    for (const [k, col, lab] of [[inj, "#ffffff", "INJ"], [prod, "#ff00ff", "PROD"]]) {
      mc.beginPath(); mc.arc(cx(k), cy(k), 9, 0, 7); mc.fillStyle = col; mc.fill();
      mc.lineWidth = 2; mc.strokeStyle = "#000"; mc.stroke();
      mc.fillStyle = "#fff"; mc.font = "12px DM Mono, monospace";
      mc.fillText(lab, Math.min(W - 44, cx(k) + 12), Math.max(14, cy(k) - 10));
    }
    mc.fillStyle = "rgba(0,0,0,.55)"; mc.fillRect(8, H - 28, 230, 20);
    mc.fillStyle = "#fff"; mc.font = "12px DM Mono, monospace";
    mc.fillText("pore volumes injected: " + pvi.toFixed(2), 14, H - 14);
  }
  function drawPlot() {
    const W = plot.width, H = plot.height, l = 46, r = 12, tp = 16, b = 34;
    pc.fillStyle = "#07070a"; pc.fillRect(0, 0, W, H);
    pc.strokeStyle = "rgba(255,255,255,.12)"; pc.lineWidth = 1;
    pc.fillStyle = "#9a9997"; pc.font = "11px DM Mono, monospace";
    const xmax = Math.max(1, Math.ceil((hist.length ? hist[hist.length - 1][0] : 1) * 2) / 2);
    for (let i = 0; i <= 5; i++) {
      const y = tp + (H - tp - b) * i / 5;
      pc.beginPath(); pc.moveTo(l, y); pc.lineTo(W - r, y); pc.stroke();
      pc.fillText((1 - i / 5).toFixed(1), 12, y + 4);
    }
    for (let i = 0; i <= 4; i++) {
      const x = l + (W - l - r) * i / 4;
      pc.fillText((xmax * i / 4).toFixed(1), x - 8, H - 14);
    }
    pc.fillText("pore volumes injected", W / 2 - 70, H - 2);
    const X = v => l + (W - l - r) * v / xmax, Y = v => tp + (H - tp - b) * (1 - v);
    for (const [idx, col] of [[1, "#4fc3ff"], [2, "#00e5a0"]]) {
      pc.beginPath(); pc.strokeStyle = col; pc.lineWidth = 2;
      hist.forEach((h, i) => i ? pc.lineTo(X(h[0]), Y(h[idx])) : pc.moveTo(X(h[0]), Y(h[idx])));
      pc.stroke();
    }
    const h = hist[hist.length - 1];
    $(".readout").textContent = h ? ("water cut " + (100 * h[1]).toFixed(1) + " %   ·   oil recovered " +
      (100 * h[2]).toFixed(1) + " %   ·   breakthrough " + (bt() || "not yet")) : "Press Run.";
  }
  function bt() {
    for (const h of hist) if (h[1] > 0.01) return "at " + h[0].toFixed(2) + " PVI";
    return null;
  }
  function loop() {
    for (let s = 0; s < 3; s++) step();
    draw(); drawPlot();
    if (running && pvi < 2.5) raf = requestAnimationFrame(loop);
    else { running = false; $(".run").textContent = "Run"; }
  }
  $(".run").addEventListener("click", () => {
    running = !running;
    $(".run").textContent = running ? "Pause" : "Run";
    if (running) { if (pvi >= 2.5) reset(); raf = requestAnimationFrame(loop); }
  });
  $(".reset").addEventListener("click", () => { running = false; cancelAnimationFrame(raf); $(".run").textContent = "Run"; reset(); });
  root.querySelectorAll("input,select").forEach(el => el.addEventListener("input", () => {
    if (el.name === "view") { readParams(); draw(); return; }
    running = false; cancelAnimationFrame(raf); $(".run").textContent = "Run"; reset();
  }));
  reset();
})();
