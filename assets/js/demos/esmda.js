/* esmda.js -- ensemble history matching (ES-MDA) on a two-parameter toy
   reservoir, live.  The unknowns are u1 = log near-well permeability and
   u2 = inter-well connectivity (prior N(0, I)).  The forward model gives the
   water cut of two producers over time: breakthrough time and sharpness
   depend on (u1, u2) nonlinearly.  Noisy observations come from a hidden
   "true" u.  ES-MDA assimilates them Na times with inflated noise alpha = Na,
     u_j <- u_j + C_ud (C_dd + alpha C_e)^{-1} (d_obs + sqrt(alpha) e_j - g(u_j)),
   which is the ensemble Kalman update used, with a neural-operator forward
   model in place of g, in the PhysicsNeMo reservoir examples.             */
(function () {
  "use strict";
  const root = document.getElementById("demo-esmda");
  if (!root) return;
  const $ = s => root.querySelector(s);
  const pa = $("canvas.params"), pb = $("canvas.data");
  const ca = pa.getContext("2d"), cb = pb.getContext("2d");
  const P_ = {};
  const T = 24;                                 // report times
  let truth, dobs, ens, stepNo, timer = 0, seed = 3, hist = [];

  function rng(s) {
    let a = s >>> 0;
    return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  }
  let R = rng(1);
  const gauss = () => { let u = 0, v = 0; while (!u) u = R(); v = R(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
  function readParams() {
    root.querySelectorAll("input,select").forEach(el => {
      P_[el.name] = el.tagName === "SELECT" ? el.value : parseFloat(el.value);
      const v = root.querySelector('[data-for="' + el.name + '"]');
      if (v) v.textContent = el.value;
    });
  }
  // forward model: water cut of two wells at T times (2T data)
  function g(u) {
    const out = new Float64Array(2 * T);
    const tb1 = 6 * Math.exp(-0.45 * u[0]) * (1 + 0.25 * Math.tanh(u[1]));
    const tb2 = 9 * Math.exp(-0.2 * u[0] - 0.4 * u[1]);
    const s1 = 1.2 + 0.4 * Math.exp(-0.3 * u[1]), s2 = 1.6 + 0.3 * Math.exp(0.2 * u[0]);
    for (let k = 0; k < T; k++) {
      const t = k + 1;
      out[k] = 1 / (1 + Math.exp(-(t - tb1) / s1));
      out[T + k] = 1 / (1 + Math.exp(-(t - tb2) / s2));
    }
    return out;
  }
  function newTruth() {
    const r = rng(seed * 101 + 7);
    truth = [(r() * 2 - 1) * 1.6, (r() * 2 - 1) * 1.6];
    const d = g(truth);
    R = rng(seed * 31 + 5);
    dobs = d.map(v => v + P_.noise * gauss());
  }
  function reset() {
    clearInterval(timer); timer = 0;
    readParams(); newTruth();
    R = rng(seed * 977 + 1);
    ens = [];
    for (let j = 0; j < (P_.ne | 0); j++) ens.push([gauss(), gauss()]);
    stepNo = 0; hist = [misfit()]; sumInv = 0; alphaPrev = Infinity; lastAlpha = 0;
    draw();
  }
  function misfit() {
    let s = 0;
    for (const u of ens) { const d = g(u); for (let k = 0; k < 2 * T; k++) s += (d[k] - dobs[k]) ** 2; }
    return Math.sqrt(s / (ens.length * 2 * T));
  }
  // one ES-MDA update with inflation alpha
  function update(alpha) {
    const Ne = ens.length, m = 2 * T;
    const D = ens.map(g);
    const um = [0, 0], dm = new Float64Array(m);
    for (let j = 0; j < Ne; j++) { um[0] += ens[j][0] / Ne; um[1] += ens[j][1] / Ne; for (let k = 0; k < m; k++) dm[k] += D[j][k] / Ne; }
    // C_ud (2 x m), C_dd (m x m)
    const Cud = [new Float64Array(m), new Float64Array(m)], Cdd = [];
    for (let a = 0; a < m; a++) Cdd.push(new Float64Array(m));
    for (let j = 0; j < Ne; j++) {
      const du = [ens[j][0] - um[0], ens[j][1] - um[1]];
      for (let a = 0; a < m; a++) {
        const da = D[j][a] - dm[a];
        Cud[0][a] += du[0] * da / (Ne - 1); Cud[1][a] += du[1] * da / (Ne - 1);
        for (let b = a; b < m; b++) Cdd[a][b] += da * (D[j][b] - dm[b]) / (Ne - 1);
      }
    }
    const s2 = P_.noise * P_.noise;
    for (let a = 0; a < m; a++) { for (let b = 0; b < a; b++) Cdd[a][b] = Cdd[b][a]; Cdd[a][a] += alpha * s2; }
    // Cholesky of (C_dd + alpha C_e)
    const L = Cdd.map(r => Float64Array.from(r));
    for (let i = 0; i < m; i++) {
      for (let j = 0; j <= i; j++) {
        let s = L[i][j];
        for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
        L[i][j] = i === j ? Math.sqrt(Math.max(s, 1e-14)) : s / L[j][j];
      }
      for (let j = i + 1; j < m; j++) L[i][j] = 0;
    }
    const solve = v => {
      const y = Float64Array.from(v);
      for (let i = 0; i < m; i++) { for (let k = 0; k < i; k++) y[i] -= L[i][k] * y[k]; y[i] /= L[i][i]; }
      for (let i = m - 1; i >= 0; i--) { for (let k = i + 1; k < m; k++) y[i] -= L[k][i] * y[k]; y[i] /= L[i][i]; }
      return y;
    };
    for (let j = 0; j < Ne; j++) {
      const inn = new Float64Array(m);
      for (let a = 0; a < m; a++) inn[a] = dobs[a] + Math.sqrt(alpha) * P_.noise * gauss() - D[j][a];
      const w = solve(inn);
      let d0 = 0, d1 = 0;
      for (let a = 0; a < m; a++) { d0 += Cud[0][a] * w[a]; d1 += Cud[1][a] * w[a]; }
      ens[j] = [ens[j][0] + d0, ens[j][1] + d1];
    }
  }
  function axes(c, W, H, l, b, xr, yr, xl, yl) {
    c.fillStyle = "#07070a"; c.fillRect(0, 0, W, H);
    c.strokeStyle = "rgba(255,255,255,.1)"; c.fillStyle = "#9a9997"; c.font = "11px DM Mono, monospace";
    for (let i = 0; i <= 4; i++) {
      const x = l + (W - l - 10) * i / 4, y = 10 + (H - 10 - b) * i / 4;
      c.beginPath(); c.moveTo(x, 10); c.lineTo(x, H - b); c.stroke();
      c.beginPath(); c.moveTo(l, y); c.lineTo(W - 10, y); c.stroke();
      c.fillText((xr[0] + (xr[1] - xr[0]) * i / 4).toFixed(1), x - 10, H - b + 15);
      c.fillText((yr[1] - (yr[1] - yr[0]) * i / 4).toFixed(1), 6, y + 4);
    }
    c.fillText(xl, l + 4, H - 4);
    c.save(); c.translate(12, 14); c.fillText(yl, 22, 0); c.restore();
    return [v => l + (W - l - 10) * (v - xr[0]) / (xr[1] - xr[0]), v => 10 + (H - 10 - b) * (yr[1] - v) / (yr[1] - yr[0])];
  }
  function draw() {
    // parameter space
    let W = pa.width, H = pa.height;
    let [X, Y] = axes(ca, W, H, 40, 30, [-3, 3], [-3, 3], "u1  (log near-well permeability)", "u2 (connectivity)");
    for (const u of ens) { ca.beginPath(); ca.arc(X(u[0]), Y(u[1]), 3, 0, 7); ca.fillStyle = "rgba(79,160,255,.75)"; ca.fill(); }
    ca.fillStyle = "#ff4d4d"; ca.font = "20px serif"; ca.fillText("★", X(truth[0]) - 8, Y(truth[1]) + 7);
    ca.fillStyle = "#e8e6e3"; ca.font = "11px DM Mono, monospace";
    ca.fillText((P_.method === "areki" ? "alpha-REKI, iteration " + stepNo + (stepNo ? ", alpha " + lastAlpha.toFixed(1) + ", sum 1/alpha " + Math.min(1, sumInv).toFixed(2) : "")
                 : "ES-MDA, " + stepNo + " of " + (P_.na | 0) + " assimilations") + "   ★ truth", 46, 24);
    // data space
    W = pb.width; H = pb.height;
    [X, Y] = axes(cb, W, H, 40, 30, [0, T], [0, 1], "time (report steps)", "water cut");
    for (const u of ens) {
      const d = g(u);
      for (const [o, col] of [[0, "rgba(79,160,255,.22)"], [T, "rgba(79,160,255,.22)"]]) {
        cb.beginPath(); cb.strokeStyle = col; cb.lineWidth = 1;
        for (let k = 0; k < T; k++) k ? cb.lineTo(X(k + 1), Y(d[o + k])) : cb.moveTo(X(k + 1), Y(d[o + k]));
        cb.stroke();
      }
    }
    const dt = g(truth);
    for (const o of [0, T]) {
      cb.beginPath(); cb.strokeStyle = "#ff4d4d"; cb.lineWidth = 2;
      for (let k = 0; k < T; k++) k ? cb.lineTo(X(k + 1), Y(dt[o + k])) : cb.moveTo(X(k + 1), Y(dt[o + k]));
      cb.stroke(); cb.lineWidth = 1;
      for (let k = 0; k < T; k++) { cb.beginPath(); cb.arc(X(k + 1), Y(dobs[o + k]), 3, 0, 7); cb.fillStyle = "#ff4d4d"; cb.fill(); cb.strokeStyle = "#000"; cb.stroke(); }
    }
    cb.fillStyle = "#e8e6e3"; cb.fillText("ensemble (blue), true model (red line), observed (red dots); producers 1 and 2", 46, 24);
    $(".readout").textContent = "data misfit (RMS): " + hist.map(v => v.toFixed(3)).join(" → ");
  }
  let sumInv = 0, alphaPrev = Infinity, lastAlpha = 0;
  function phis() {
    return ens.map(u => { const d = g(u); let s = 0; for (let k = 0; k < 2 * T; k++) s += 0.5 * ((d[k] - dobs[k]) / P_.noise) ** 2; return s; });
  }
  function finished() { return P_.method === "areki" ? sumInv >= 1 - 1e-9 || stepNo >= 20 : stepNo >= (P_.na | 0); }
  function next() {
    if (finished()) { clearInterval(timer); timer = 0; $(".run").textContent = "Assimilate"; return; }
    let alpha = P_.na;
    if (P_.method === "areki") {             // alpha-REKI (Iglesias and Yang), stopped at sum 1/alpha = 1
      const ph = phis(), Ne = ph.length, mu = ph.reduce((a, b) => a + b, 0) / Ne, va = ph.reduce((a, b) => a + (b - mu) ** 2, 0) / Ne;
      alpha = 1 / Math.min(Math.max(2 * T / (2 * mu), Math.sqrt(2 * T / (2 * va))), 1 - sumInv);
    }
    update(alpha); sumInv += 1 / alpha; alphaPrev = alpha; lastAlpha = alpha;
    stepNo++; hist.push(misfit()); draw();
  }
  $(".run").addEventListener("click", () => {
    if (timer) { clearInterval(timer); timer = 0; $(".run").textContent = "Assimilate"; return; }
    if (finished()) reset();
    $(".run").textContent = "Pause";
    timer = setInterval(next, 650);
  });
  $(".truth").addEventListener("click", () => { seed++; reset(); });
  $(".reset").addEventListener("click", reset);
  root.querySelectorAll("input,select").forEach(el => el.addEventListener("input", reset));
  reset();
})();
