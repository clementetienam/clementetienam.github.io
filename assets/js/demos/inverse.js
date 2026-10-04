/* inverse.js -- "Find the hidden channel": the inverse problem of the Norne
   workflow as a game, on a 16 x 16 grid.

   Wells: an inverted nine-spot, one injector and eight producers.
   Forward model, as in the paper's workflow:
     states  a Fourier neural operator, ln K -> p, S at 12 report times;
     wells   the Peaceman well model replaced either by a second FNO,
             (ln K, p, S) -> (q_o, q_w) read at the producers, or by CCR
             (cluster-classify-regress: k-means on joint inputs and outputs,
             a random-forest classifier, polynomial ridge experts), trained
             live in the browser.
   Priors:   a Gaussian field in a truncated DCT basis (36 of 256
             coefficients) or the latent space of a convolutional variational
             autoencoder (VCAE) trained on channel images.
   Inversion alpha-REKI (1/alpha from the mean and variance of the data
             misfit, stopped when sum 1/alpha = 1) or ES-MDA, on the DCT coefficients or the VCAE latent.
             Options: diagonal C_d set by the user; covariance localisation
             (Gaspari-Cohn taper on C_md by distance from each cell to the well
             of each datum; the update is then made on the grid and mapped back
             by DCT projection or by the VCAE encoder); multiplicative inflation.
   The observed data and every verification come from the in-browser
   simulator (nvrs_engine.js). The two FNOs and the VCAE were trained offline
   on 600 runs of that simulator and 4,000 channel images; only their weights
   are loaded here.                                                          */
(function () {
  "use strict";
  const root = document.getElementById("demo-inverse");
  if (!root || !window.NVRS || !window.GEO) return;
  const E = window.NVRS, G = window.GEO, $ = s => root.querySelector(s);
  const n = 16, NC = 256, NR = 12, RT = Array.from({ length: NR }, (_, i) => (i + 1) * 20);
  // inverted nine-spot: injector in the centre, producers at the corners and edge midpoints
  const PRODS = [1 * n + 1, 1 * n + 8, 1 * n + 14, 8 * n + 1, 8 * n + 14, 14 * n + 1, 14 * n + 8, 14 * n + 14], INJ = 8 * n + 8;
  const NW = PRODS.length, DW = 2 * NW + 1, ND = DW * NR;   // per time: q_o x8, q_w x8, p_inj
  const BASE = (document.currentScript && document.currentScript.src) || "";
  const DATA = BASE.replace(/js\/demos\/inverse\.js.*$/, "data/");
  let NET = null, CCR = null, busy = false, caseNo = 1, truth = null, dobs = null, sig = null;
  const board = { guess: new Float64Array(NC).fill(G.SHALE), results: {} };
  const P_ = {};

  // ------------------------------------------------------------ utilities
  const rng = G.rng, gauss = G.gauss;
  const tick = (ms) => new Promise(res => setTimeout(res, ms || 0));
  function readParams() {
    root.querySelectorAll("input,select").forEach(el => {
      P_[el.name] = el.tagName === "SELECT" ? el.value : parseFloat(el.value);
      const v = root.querySelector('[data-for="' + el.name + '"]'); if (v) v.textContent = el.value;
    });
  }
  function log(msg) { const el = $(".wlog"); el.innerHTML += "<div>" + msg + "</div>"; el.scrollTop = el.scrollHeight; }
  function stage(k, status) {
    root.querySelectorAll(".stage").forEach(el => {
      const s = +el.getAttribute("data-step");
      el.classList.toggle("active", s === k && status !== "done");
      el.classList.toggle("done", s < k || (s === k && status === "done"));
    });
  }
  function erf(x) {
    const s = x < 0 ? -1 : 1; x = Math.abs(x); const t = 1 / (1 + 0.3275911 * x);
    return s * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x));
  }
  const gelu = x => 0.5 * x * (1 + erf(x / Math.SQRT2));

  // ------------------------------------------------------------ neural nets (inference only)
  const TW = [], TH = [];                  // twiddles for the 16-point DFT
  for (let k = 0; k < n; k++) for (let j = 0; j < n; j++) { TW.push(Math.cos(2 * Math.PI * k * j / n)); TH.push(Math.sin(2 * Math.PI * k * j / n)); }
  function fnoNet(get, pre, ci, co, width, m, L) {
    const g = Array.from({ length: n }, (_, i) => (i + 0.5) / n);
    const gx = new Float64Array(NC), gy = new Float64Array(NC);
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) { gx[j * n + i] = g[i]; gy[j * n + i] = g[j]; }
    return { ci, co, width, m, L, gx, gy, lift: [get(pre + "lift.weight"), get(pre + "lift.bias")],
             spec: Array.from({ length: L }, (_, l) => [get(pre + "spec." + l + ".w1"), get(pre + "spec." + l + ".w2")]),
             pw: Array.from({ length: L }, (_, l) => [get(pre + "pw." + l + ".weight"), get(pre + "pw." + l + ".bias")]),
             p1: [get(pre + "p1.weight"), get(pre + "p1.bias")], p2: [get(pre + "p2.weight"), get(pre + "p2.bias")] };
  }
  function conv1(X, Wb, act) {                // 1 x 1 convolution, X: array of channels
    const [W, b] = Wb, ci = X.length, co = b.length, Y = [];
    for (let o = 0; o < co; o++) {
      const y = new Float64Array(NC).fill(b[o]);
      for (let c = 0; c < ci; c++) { const w = W[o * ci + c], x = X[c]; if (w) for (let k = 0; k < NC; k++) y[k] += w * x[k]; }
      if (act) for (let k = 0; k < NC; k++) y[k] = act(y[k]);
      Y.push(y);
    }
    return Y;
  }
  function spectral(X, w1, w2, m, co) {
    const ci = X.length, rows = [];
    for (let a = 0; a < m; a++) rows.push(a);
    for (let a = n - m; a < n; a++) rows.push(a);
    const nr = rows.length;
    // forward: along x (m bins), then along y (selected rows); scale 1/16 (ortho)
    const Fr = [], Fi = [];
    for (let c = 0; c < ci; c++) {
      const x = X[c], Rr = new Float64Array(n * m), Ri = new Float64Array(n * m);
      for (let j = 0; j < n; j++) for (let b = 0; b < m; b++) {
        let sr = 0, si = 0;
        for (let i = 0; i < n; i++) { const v = x[j * n + i]; sr += v * TW[b * n + i]; si -= v * TH[b * n + i]; }
        Rr[j * m + b] = sr; Ri[j * m + b] = si;
      }
      const fr = new Float64Array(nr * m), fi = new Float64Array(nr * m);
      for (let q = 0; q < nr; q++) { const a = rows[q]; for (let b = 0; b < m; b++) {
        let sr = 0, si = 0;
        for (let j = 0; j < n; j++) { const c_ = TW[a * n + j], s_ = TH[a * n + j], ur = Rr[j * m + b], ui = Ri[j * m + b]; sr += ur * c_ + ui * s_; si += ui * c_ - ur * s_; }
        fr[q * m + b] = sr / n; fi[q * m + b] = si / n;
      } }
      Fr.push(fr); Fi.push(fi);
    }
    const Y = [];
    for (let o = 0; o < co; o++) {
      const or_ = new Float64Array(nr * m), oi = new Float64Array(nr * m);
      for (let c = 0; c < ci; c++) for (let q = 0; q < nr; q++) {
        const w = q < m ? w1 : w2, a = q < m ? q : q - m;
        for (let b = 0; b < m; b++) {
          const p = ((((c * co + o) * m + a) * m + b) * 2), wr = w[p], wi = w[p + 1], xr = Fr[c][q * m + b], xi = Fi[c][q * m + b];
          or_[q * m + b] += xr * wr - xi * wi; oi[q * m + b] += xr * wi + xi * wr;
        }
      }
      // inverse: along y, then the real inverse along x
      const Zr = new Float64Array(n * m), Zi = new Float64Array(n * m);
      for (let j = 0; j < n; j++) for (let b = 0; b < m; b++) {
        let sr = 0, si = 0;
        for (let q = 0; q < nr; q++) { const a = rows[q], c_ = TW[a * n + j], s_ = TH[a * n + j], ur = or_[q * m + b], ui = oi[q * m + b]; sr += ur * c_ - ui * s_; si += ur * s_ + ui * c_; }
        Zr[j * m + b] = sr; Zi[j * m + b] = si;
      }
      const y = new Float64Array(NC);
      for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
        let s = Zr[j * m];
        for (let b = 1; b < m; b++) s += 2 * (Zr[j * m + b] * TW[b * n + i] - Zi[j * m + b] * TH[b * n + i]);
        y[j * n + i] = s / n;
      }
      Y.push(y);
    }
    return Y;
  }
  function fnoRun(F, inp) {
    let h = conv1([...inp, F.gx, F.gy], F.lift);
    for (let l = 0; l < F.L; l++) {
      const s = spectral(h, F.spec[l][0], F.spec[l][1], F.m, F.width), p = conv1(h, F.pw[l]);
      h = s.map((v, c) => { const r = new Float64Array(NC); for (let k = 0; k < NC; k++) { const t = v[k] + p[c][k]; r[k] = l < F.L - 1 ? gelu(t) : t; } return r; });
    }
    return conv1(conv1(h, F.p1, gelu), F.p2);
  }
  function vcaeEncode(prob) {               // facies probability (256) -> latent mean (8)
    const V = NET.v;
    const conv = (x, C, S, Wb, CO) => { const [W, b] = Wb, S2 = S / 2, y = new Float64Array(CO * S2 * S2);
      for (let o = 0; o < CO; o++) for (let j = 0; j < S2; j++) for (let i = 0; i < S2; i++) {
        let s = b[o];
        for (let c = 0; c < C; c++) for (let kj = 0; kj < 3; kj++) { const jj = 2 * j + kj - 1; if (jj < 0 || jj >= S) continue;
          for (let ki = 0; ki < 3; ki++) { const ii = 2 * i + ki - 1; if (ii < 0 || ii >= S) continue;
            s += W[((o * C + c) * 3 + kj) * 3 + ki] * x[(c * S + jj) * S + ii]; } }
        y[(o * S2 + j) * S2 + i] = Math.max(0, s);
      } return y; };
    const h = conv(conv(prob, 1, 16, V.e1, 16), 16, 8, V.e2, 32), [W, b] = V.mu, z = new Float64Array(8);
    for (let o = 0; o < 8; o++) { let t = b[o]; for (let i = 0; i < 512; i++) t += W[o * 512 + i] * h[i]; z[o] = t; }
    return z;
  }
  function vcaeDecode(z) {                  // z (8) -> facies probability (256)
    const V = NET.v, [W0, b0] = V.d0, h = new Float64Array(512);
    for (let o = 0; o < 512; o++) { let s = b0[o]; for (let i = 0; i < 8; i++) s += W0[o * 8 + i] * z[i]; h[o] = Math.max(0, s); }
    const up = (x, C, S) => { const S2 = 2 * S, y = new Float64Array(C * S2 * S2);
      for (let c = 0; c < C; c++) for (let j = 0; j < S2; j++) for (let i = 0; i < S2; i++) y[(c * S2 + j) * S2 + i] = x[(c * S + (j >> 1)) * S + (i >> 1)]; return y; };
    const conv3 = (x, C, S, Wb, CO, act) => { const [W, b] = Wb, y = new Float64Array(CO * S * S);
      for (let o = 0; o < CO; o++) for (let j = 0; j < S; j++) for (let i = 0; i < S; i++) {
        let s = b[o];
        for (let c = 0; c < C; c++) for (let dj = -1; dj <= 1; dj++) { const jj = j + dj; if (jj < 0 || jj >= S) continue;
          for (let di = -1; di <= 1; di++) { const ii = i + di; if (ii < 0 || ii >= S) continue;
            s += W[((o * C + c) * 3 + dj + 1) * 3 + di + 1] * x[(c * S + jj) * S + ii]; } }
        y[(o * S + j) * S + i] = act(s);
      } return y; };
    const h1 = conv3(up(h, 32, 4), 32, 8, V.d1, 16, v => Math.max(0, v));
    return conv3(up(h1, 16, 8), 16, 16, V.d2, 1, v => 1 / (1 + Math.exp(-v)));
  }
  let netPromise = null;
  function loadNets() {                      // one download, shared; retried on failure
    if (!netPromise) netPromise = fetchNets().catch(e => { netPromise = null; throw e; });
    return netPromise;
  }
  async function grab(url, kind) {           // fetch with a timeout and up to three attempts
    for (let a = 0; ; a++) {
      const ctl = new AbortController(), timer = setTimeout(() => ctl.abort(), 15000);
      try { const r = await fetch(url, { signal: ctl.signal }); if (!r.ok) throw new Error(r.status); return await (kind === "json" ? r.json() : r.arrayBuffer()); }
      catch (e) { if (a >= 2) throw e; }
      finally { clearTimeout(timer); }
    }
  }
  async function fetchNets() {
    if (NET) return NET;
    const [meta, bin, cmeta, cbin] = await Promise.all([
      grab(DATA + "inverse_models.json", "json"), grab(DATA + "inverse_models.bin"),
      grab(DATA + "ccr_train.json", "json"), grab(DATA + "ccr_train.bin")]);
    const [vmeta, vbin] = await Promise.all([grab(DATA + "vcae.json", "json"), grab(DATA + "vcae.bin")]);
    const all = new Float32Array(bin), get = k => { const [o, len] = meta.index[k]; return all.subarray(o, o + len); };
    const vall = new Float32Array(vbin), vget = k => { const [o, len] = vmeta.index[k]; return vall.subarray(o, o + len); };
    NET = { meta, s: fnoNet(get, "s.", 1, 2 * NR, 16, 6, 3), w: fnoNet(get, "w.", 3, 2, 12, 4, 2),
            v: Object.fromEntries(["e1", "e2", "mu", "d0", "d1", "d2"].map(k => [k, [vget("v." + k + ".weight"), vget("v." + k + ".bias")]])), vmeta,
            ccr: { meta: cmeta, X: new Float32Array(cbin) } };
    return NET;
  }

  // ------------------------------------------------------------ forward model
  function nb(k) { const i = k % n, j = (k / n) | 0, r = []; for (const [di, dj] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) if (i + di >= 0 && i + di < n && j + dj >= 0 && j + dj < n) r.push((j + dj) * n + i + di); return r; }
  const NB = PRODS.map(nb);
  function states(lnK) {                     // -> {P: [NR][256], S: [NR][256]}
    const m = NET.meta, out = fnoRun(NET.s, [lnK.map(v => v / 1.8)]);
    // the FNO learns log(p + 0.1), standardised
    return { P: out.slice(0, NR).map(a => a.map(v => Math.exp(v * m.psd + m.pmu) - m.plog)), S: out.slice(NR).map(a => a.map(v => Math.min(1, Math.max(0, (v + 1) / 2)))) };
  }
  function wells(lnK, st, kind) {            // -> data vector (ND)
    const m = NET.meta, cm = NET.ccr.meta, d = new Float64Array(ND);
    for (let t = 0; t < NR; t++) {
      if (kind === "ccr") {
        PRODS.forEach((k, w) => {
          const f = [(st.P[t][k] - cm.pmu) / cm.psd, st.S[t][k], lnK[k], NB[w].reduce((s, q) => s + st.S[t][q], 0) / NB[w].length,
                     (NB[w].reduce((s, q) => s + st.P[t][q], 0) / NB[w].length - cm.pmu) / cm.psd];
          const q = ccrPredict(f); d[DW * t + w] = q[0]; d[DW * t + NW + w] = q[1];
        });
      } else {
        const o = fnoRun(NET.w, [lnK.map(v => v / 1.8), st.P[t].map(v => (Math.log(v + m.plog) - m.pmu) / m.psd), st.S[t].map(v => 2 * v - 1)]);
        PRODS.forEach((k, w) => { d[DW * t + w] = o[0][k] / m.wsc; d[DW * t + NW + w] = o[1][k] / m.wsc; });
      }
      d[DW * t + 2 * NW] = st.P[t][INJ];
    }
    return d;
  }
  const surrogate = (lnK, kind) => wells(lnK, states(lnK), kind);
  function simulate(lnK) {                   // the simulator itself
    const M = E.model({ n, logK: lnK, visc: 5, wells: "ninespot" }), d = new Float64Array(ND);
    for (let t = 0; t < NR; t++) {
      if (!E.step(M, { prec: "fp32", restart: 30, dtmax: 10, tstop: RT[t] })) return null;
      PRODS.forEach((k, w) => { const q = E.wellRates(M, k); d[DW * t + w] = q.qo / M.Q; d[DW * t + NW + w] = q.qw / M.Q; });
      d[DW * t + 2 * NW] = M.p[M.inj];
    }
    return { d, S: Float64Array.from(M.S) };
  }

  // ------------------------------------------------------------ CCR well model (ccr.js)
  async function trainCCR() {
    const C = NET.ccr, cols = C.meta.cols, ntr = C.meta.ntrain, nte = C.meta.ntest, A = C.X;
    const row = i => Array.from(A.subarray(i * cols, i * cols + cols));
    const Xtr = [], Ytr = [], Xte = [], Yte = [];
    for (let i = 0; i < ntr; i += 2) { const v = row(i); Xtr.push(v.slice(0, 5)); Ytr.push(v.slice(5)); }
    for (let i = ntr; i < ntr + nte; i++) { const v = row(i); Xte.push(v.slice(0, 5)); Yte.push(v.slice(5)); }
    await tick();
    const m = window.CCR.train(Xtr, Ytr, { Kmax: 5, lam: 1e-2 });
    log("&nbsp;&nbsp; cluster: k-means on joint (inputs, rates), elbow picks K = " + m.K + " (SSE " + m.sse.map(v => v.toFixed(0)).join(", ") + ").");
    const clip = y => y.map(v => Math.max(0, v));
    const r2 = pred => { let ss = 0, st = 0; const mu = [0, 1].map(j => Yte.reduce((s, y) => s + y[j], 0) / Yte.length);
      Xte.forEach((x, s) => { const p = pred(x); for (let j = 0; j < 2; j++) { ss += (p[j] - Yte[s][j]) ** 2; st += (Yte[s][j] - mu[j]) ** 2; } }); return 1 - ss / st; };
    const predict = x => clip(m.predict(x));
    CCR = { K: m.K, predict, lab: m.labels, Xtr, Ytr, acc: m.acc, r2: r2(predict), r2single: r2(x => clip(m.single(x))), Xte, Yte };
    return CCR;
  }
  const ccrPredict = f => CCR.predict(f);

  // ------------------------------------------------------------ drawing
  const CV = { board: $("canvas.board"), ens: $("canvas.ens"), data: $("canvas.data"), side: $("canvas.side"), evo: $("canvas.evo") };
  const CX = Object.fromEntries(Object.entries(CV).map(([k, c]) => [k, c.getContext("2d")]));
  const jet = t => { const x = Math.min(1, Math.max(0, t)); return [1.5 - Math.abs(4 * x - 3), 1.5 - Math.abs(4 * x - 2), 1.5 - Math.abs(4 * x - 1)].map(v => Math.round(255 * Math.min(1, Math.max(0, v)))); }
  const facCol = v => jet((v - G.SHALE) / (G.SAND - G.SHALE));   // ln K on the jet colour map
  function drawField(c, f, x, y, s, opt) {
    opt = opt || {};
    const img = c.createImageData(n, n);
    for (let k = 0; k < NC; k++) {
      let col;
      if (opt.std) col = jet(Math.min(1, f[k] / 2));
      else col = facCol(f[k]);
      img.data.set([col[0], col[1], col[2], 255], 4 * k);
    }
    const off = document.createElement("canvas"); off.width = n; off.height = n; off.getContext("2d").putImageData(img, 0, 0);
    c.imageSmoothingEnabled = false; c.drawImage(off, x, y, s, s);   // cell by cell, as the simulator sees it
    if (opt.fog) {
      c.fillStyle = "rgba(10,10,14,.86)"; c.fillRect(x, y, s, s);
      c.fillStyle = "#e8e6e3"; c.font = Math.round(s / 3) + "px Syne, sans-serif"; c.textAlign = "center"; c.fillText("?", x + s / 2, y + s * 0.62); c.textAlign = "left";
    }
    if (!opt.nowells) {
      const cell = s / n, dot = (k, col) => { c.beginPath(); c.arc(x + ((k % n) + 0.5) * cell, y + (((k / n) | 0) + 0.5) * cell, Math.max(2.5, cell * 0.35), 0, 7); c.fillStyle = col; c.fill(); c.strokeStyle = "#000"; c.stroke(); };
      dot(INJ, "#ffffff"); PRODS.forEach(k => dot(k, "#ff00ff"));
    }
    c.strokeStyle = "rgba(255,255,255,.15)"; c.strokeRect(x + 0.5, y + 0.5, s - 1, s - 1);
  }
  const BOARD = [["hidden reservoir", "truth"], ["your guess (paint me)", "guess"], ["grid cells (no exotic prior)", "grid"], ["DCT prior", "dct"], ["VCAE prior", "vcae"]];
  function boardLayout() { const W = CV.board.width, gap = 16, k = BOARD.length, s = Math.min((W - gap * (k + 1)) / k, CV.board.height - 40); return { gap, s, x: i => gap + i * (s + gap), y: 30 }; }
  function drawBoard() {
    const c = CX.board, L = boardLayout(); c.fillStyle = "#07070a"; c.fillRect(0, 0, CV.board.width, CV.board.height);
    BOARD.forEach(([title, key], i) => {
      c.fillStyle = "#e8e6e3"; c.font = "12px DM Mono, monospace"; c.fillText(title, L.x(i), 18);
      if (key === "truth") drawField(c, truth ? truth.lnK : new Float64Array(NC), L.x(i), L.y, L.s, { fog: P_.play === "on" && !board.revealed, pixel: true });
      else if (key === "guess") drawField(c, board.guess, L.x(i), L.y, L.s, { pixel: true });
      else if (board.results[key]) drawField(c, pick(board.results[key].mean, board.results[key].best), L.x(i), L.y, L.s);
      else { c.fillStyle = "#101016"; c.fillRect(L.x(i), L.y, L.s, L.s); c.fillStyle = "#6b6a68"; c.fillText("not run yet", L.x(i) + 10, L.y + L.s / 2); }
    });
  }
  function drawEnsemble(fields, title, stdField) {
    const c = CX.ens, W = CV.ens.width, H = CV.ens.height; c.fillStyle = "#07070a"; c.fillRect(0, 0, W, H);
    c.fillStyle = "#e8e6e3"; c.font = "12px DM Mono, monospace"; c.fillText(title, 10, 16);
    const cols = 12, rows = 2, gap = 6, s = Math.min((W - gap * (cols + 1)) / cols, (H - 30 - gap * (rows + 1)) / rows);
    fields.slice(0, cols * rows).forEach((f, q) => drawField(c, f, gap + (q % cols) * (s + gap), 26 + Math.floor(q / cols) * (s + gap), s, { nowells: true }));
  }
  // data panels: water rate of the eight producers and the injector pressure; bands = P10-P90
  function drawData(sets, title) {
    const c = CX.data, W = CV.data.width, H = CV.data.height; c.fillStyle = "#07070a"; c.fillRect(0, 0, W, H);
    c.fillStyle = "#e8e6e3"; c.font = "12px DM Mono, monospace"; c.fillText(title, 10, 16);
    const panels = PRODS.map((_, w) => ["P" + (w + 1) + " water rate", NW + w]).concat([["injector pressure", 2 * NW]]);
    const cols = 5, pw = (W - 20) / cols - 10, ph = (H - 40) / 2 - 22;
    const pTop = truth ? 1.4 * Math.max(...Array.from({ length: NR }, (_, t) => truth.d[DW * t + 2 * NW])) : 10;
    const qTop = truth ? Math.max(0.1, 1.3 * Math.max(...Array.from({ length: NR * NW }, (_, q) => truth.d[DW * Math.floor(q / NW) + NW + (q % NW)]))) : 0.6;
    panels.forEach(([lab, j], pi) => {
      const x0 = 14 + (pi % cols) * (pw + 10), t0 = 28 + Math.floor(pi / cols) * (ph + 22), hi = j === 2 * NW ? pTop : qTop;
      const X = t => x0 + pw * t / RT[NR - 1], Y = v => t0 + ph * (1 - Math.min(1, Math.max(0, (isFinite(v) ? v : 0) / hi)));
      c.strokeStyle = "rgba(255,255,255,.12)"; c.strokeRect(x0, t0, pw, ph);
      c.fillStyle = "#9a9997"; c.font = "11px DM Mono, monospace"; c.fillText(lab, x0 + 4, t0 + 13);
      for (const s of sets) {
        if (!s) continue;
        if (s.band) {
          c.beginPath(); for (let t = 0; t < NR; t++) c.lineTo(X(RT[t]), Y(s.band[1][DW * t + j]));
          for (let t = NR - 1; t >= 0; t--) c.lineTo(X(RT[t]), Y(s.band[0][DW * t + j])); c.closePath(); c.fillStyle = s.col; c.fill();
        } else if (s.dots) {
          for (let t = 0; t < NR; t++) { c.beginPath(); c.arc(X(RT[t]), Y(s.y[DW * t + j]), 2.6, 0, 7); c.fillStyle = "#ff4d4d"; c.fill(); c.strokeStyle = "#000"; c.lineWidth = 1; c.stroke(); }
        } else if (s.members) {
          c.strokeStyle = s.col; c.lineWidth = 1;
          for (const y of s.members) { c.beginPath(); for (let t = 0; t < NR; t++) t ? c.lineTo(X(RT[t]), Y(y[DW * t + j])) : c.moveTo(X(RT[t]), Y(y[DW * t + j])); c.stroke(); }
        } else {
          c.beginPath(); c.strokeStyle = s.col; c.lineWidth = s.w || 1.5; c.setLineDash(s.dash || []);
          for (let t = 0; t < NR; t++) t ? c.lineTo(X(RT[t]), Y(s.y[DW * t + j])) : c.moveTo(X(RT[t]), Y(s.y[DW * t + j]));
          c.stroke(); c.setLineDash([]);
        }
      }
    });
  }
  function drawSide(kind) {                  // alpha schedule or CCR clusters
    const c = CX.side, W = CV.side.width, H = CV.side.height; c.fillStyle = "#07070a"; c.fillRect(0, 0, W, H);
    c.font = "12px DM Mono, monospace"; c.fillStyle = "#e8e6e3";
    if (kind === "ccr" && CCR) {
      c.fillText("CCR: water rate against well-cell saturation, coloured by regime", 10, 16);
      const cols = ["#4fc3ff", "#00e5a0", "#ffb74d", "#ff6b6b", "#c792ea"], X = v => 40 + (W - 60) * v, Y = v => H - 24 - (H - 50) * Math.min(1, v / 0.6);
      c.strokeStyle = "rgba(255,255,255,.12)"; c.strokeRect(40, 26, W - 60, H - 50);
      CCR.Xtr.forEach((x, s) => { if (s % 3) return; c.fillStyle = cols[CCR.lab[s] % 5]; c.globalAlpha = 0.5; c.fillRect(X(x[1]) - 1, Y(CCR.Ytr[s][1]) - 1, 2.4, 2.4); });
      c.globalAlpha = 1; c.fillStyle = "#9a9997"; c.fillText("S_w at the well cell", W - 170, H - 6); c.fillText("q_w", 8, 40);
      return;
    }
    const hist = kind && kind.alphas;
    if (window.PLOTS) { window.PLOTS.alpha(c, W, H, hist ? [{ alphas: hist, label: kind.label || "", col: "#4fa0ff" }] : [],
      (P_.method === "esmda" ? "ES-MDA: alpha = N_a" : "alpha-REKI: alpha against iteration") + " (left, log) and sum of 1/alpha (right)"); return; }
    c.fillText("alpha per iteration (bars) and the running sum of 1/alpha (line)", 10, 16);
    if (!hist || !hist.length) return;
    const nb_ = Math.max(hist.length, 4), bw = (W - 60) / nb_, amax = Math.max(...hist) * 1.1, Y = v => H - 24 - (H - 50) * v;
    c.strokeStyle = "rgba(255,255,255,.12)"; c.strokeRect(40, 26, W - 60, H - 50);
    let s = 0; c.beginPath();
    hist.forEach((a, i) => { c.fillStyle = "rgba(79,195,255,.6)"; c.fillRect(42 + i * bw, Y(a / amax), bw - 4, H - 24 - Y(a / amax)); c.fillStyle = "#9a9997"; c.fillText(a.toFixed(1), 44 + i * bw, Y(a / amax) - 4);
      s += 1 / a; const yy = Y(Math.min(1, s)); i ? c.lineTo(42 + (i + 0.5) * bw, yy) : c.moveTo(42 + (i + 0.5) * bw, yy); });
    c.strokeStyle = "#00e5a0"; c.lineWidth = 2; c.stroke(); c.lineWidth = 1;
    c.fillStyle = "#00e5a0"; c.fillText("sum 1/alpha = 1", W - 140, Y(1) + 14);
    c.setLineDash([4, 4]); c.strokeStyle = "rgba(0,229,160,.5)"; c.beginPath(); c.moveTo(40, Y(1)); c.lineTo(W - 20, Y(1)); c.stroke(); c.setLineDash([]);
  }
  // permeability through the iterations: truth, prior mean, posterior mean at each iteration, final spread
  // which estimate to show: the best-matching member (crisp), the posterior mean (smooth), or the mean thresholded to facies
  function pick(mean, best) {
    if (P_.est === "mean" || !best) return mean;
    if (P_.est === "facies") return mean.map(v => (v > 0 ? G.SAND : G.SHALE));
    return best;
  }
  const estName = () => P_.est === "mean" ? "posterior mean" : P_.est === "facies" ? "thresholded mean" : "best member";
  function drawEvo(list, std) {
    board.evo = list; board.evoStd = std;
    const c = CX.evo, W = CV.evo.width, H = CV.evo.height; c.fillStyle = "#07070a"; c.fillRect(0, 0, W, H);
    c.fillStyle = "#e8e6e3"; c.font = "12px DM Mono, monospace";
    c.fillText("permeability: the truth, then the " + estName() + " after each update, then the spread", 10, 16);
    const items = [{ f: truth ? truth.lnK : new Float64Array(NC), t: "truth", fog: P_.play === "on" && !board.revealed }, ...list.slice(-8).map(it => ({ t: it.t, f: pick(it.f, it.b) }))];
    if (std) items.push({ f: std, t: "spread (std)", std: true });
    const k = Math.max(items.length, 6), gap = 10, s = Math.min((W - gap * (k + 1)) / k, H - 46);
    items.forEach((it, i) => {
      const x = gap + i * (s + gap);
      drawField(c, it.f, x, 26, s, { std: it.std, fog: it.fog, pixel: i === 0 });
      c.fillStyle = i === 0 ? "#ff4d4d" : "#9a9997"; c.font = "11px DM Mono, monospace"; c.fillText(it.t, x, 26 + s + 14);
    });
    if (truth && list.length && !(P_.play === "on" && !board.revealed)) {
      const last = pick(list[list.length - 1].f, list[list.length - 1].b); let e = 0; for (let q = 0; q < NC; q++) e += (last[q] - truth.lnK[q]) ** 2;
      c.fillStyle = "#e8e6e3"; c.textAlign = "right"; c.fillText("RMS error in ln K of the " + estName() + ": " + Math.sqrt(e / NC).toFixed(2), W - 12, 16); c.textAlign = "left";
    }
  }
  function score() {
    const rows = [["You", "guess"], ["alpha-REKI / ES-MDA on the grid cells (no exotic prior)", "grid"], ["alpha-REKI / ES-MDA + DCT prior", "dct"], ["alpha-REKI / ES-MDA + VCAE prior", "vcae"]];
    const misOf = (r, k) => !r ? NaN : k === "guess" || P_.est === "mean" ? r.mis : P_.est === "facies" ? r.misF : r.misB;
    const fm = f => { if ((P_.play === "on" && !board.revealed) || !f) return "?"; let s = 0; for (let k = 0; k < NC; k++) if ((f[k] > 0) === (truth.lnK[k] > 0)) s++; return (100 * s / NC).toFixed(0) + " %"; };
    $(".score").innerHTML = "<table><tr><th>contestant</th><th>data misfit (simulator, in noise units)</th><th>sand/shale cells right</th><th>forward runs used</th></tr>" +
      rows.map(([lab, k]) => { const r = board.results[k]; const name = k === "guess" ? lab : lab.replace("alpha-REKI / ES-MDA", r ? r.method : "alpha-REKI / ES-MDA");
        return "<tr><td>" + name + "</td><td>" + (isFinite(misOf(r, k)) ? misOf(r, k).toFixed(2) : "-") + "</td><td>" + (r ? fm(k === "guess" ? r.mean : pick(r.mean, r.best)) : "-") + "</td><td>" + (r ? r.runs : "-") + "</td></tr>"; }).join("") + "</table>";
  }

  // ------------------------------------------------------------ the case
  function newCase() {
    const seed = 900000 + caseNo * 17;
    const lnK = G.channel(seed);
    const sim = simulate(lnK);
    if (!sim) { caseNo++; return newCase(); }
    const r = rng(seed + 3);
    // C_d diagonal: sigma = (relative noise) x |d| + floor, separately for rates and injector pressure
    const rq = (P_.noiseq || 10) / 100, rp = (P_.noisep || 5) / 100;
    sig = sim.d.map((v, j) => (j % DW === 2 * NW ? rp * Math.abs(v) + 0.02 : rq * Math.abs(v) + 0.01));
    dobs = sim.d.map((v, j) => v + sig[j] * gauss(r));
    truth = { lnK, d: sim.d, seed };
    board.results = {}; board.revealed = false;
    $(".casehdr").textContent = "Case #" + caseNo + ": a sand channel runs somewhere under this field. Eight producers have reported their oil and water rates for 240 days.";
    drawBoard(); score(); drawData([{ y: sim.d, col: "#ff4d4d", w: 2 }, { y: dobs, dots: true }], "true model (red line) and the noisy observations from it (red dots)");
    drawEvo([]);
    drawEnsemble([], "ensemble members appear here"); drawSide(null);
    log("Case #" + caseNo + ": observation noise sigma_d = " + (P_.noiseq || 10) + " % of each rate + 0.01, " + (P_.noisep || 5) +
        " % of the injector pressure + 0.02; C_d diagonal; " + ND + " observations.");
  }
  // Gaspari-Cohn taper, zero beyond the radius L (cells)
  function gc(d, L) {
    const r = 2 * d / L;
    if (r >= 2) return 0;
    if (r <= 1) return (((-0.25 * r + 0.5) * r + 0.625) * r - 5 / 3) * r * r + 1;
    return ((((r / 12 - 0.5) * r + 0.625) * r + 5 / 3) * r - 5) * r + 4 - 2 / (3 * r);
  }
  function taper(L) {                        // rho[k][p]: cell k against datum p, at the well that measured it
    const wellOf = p => { const w = p % DW; return w < NW ? PRODS[w] : w < 2 * NW ? PRODS[w - NW] : INJ; };
    return Array.from({ length: NC }, (_, k) => Float64Array.from({ length: ND }, (_, p) => {
      const q = wellOf(p); return gc(Math.hypot((k % n) - (q % n), ((k / n) | 0) - ((q / n) | 0)), L); }));
  }
  function misfit(d) { let s = 0; for (let j = 0; j < ND; j++) s += ((d[j] - dobs[j]) / sig[j]) ** 2; return Math.sqrt(s / ND); }

  // ------------------------------------------------------------ inversion
  async function invert(prior) {
    if (busy) return; busy = true; readParams(); setBusy(true);
    try {
      if (!truth) newCase();
      await loadNets();
      const well = P_.well, method = P_.method, Ne = P_.ne | 0, Na = P_.na | 0;
      if (well === "ccr" && !CCR) {
        stage(3); log("3. Training the CCR well model on 28,800 (well, time) samples from 300 simulator runs.");
        await tick(); await trainCCR(); drawSide("ccr");
        log("&nbsp;&nbsp; random forest classifier accuracy " + (100 * CCR.acc).toFixed(1) + " %; held-out R2 of (q_o, q_w): CCR " + CCR.r2.toFixed(3) +
            " against " + CCR.r2single.toFixed(3) + " for one global polynomial.");
        stage(3, "done"); await tick(600);
      }
      // grid: the 256 cell values of ln K are the parameters (no reduced basis, no learned prior)
      const dim = prior === "vcae" ? 8 : prior === "grid" ? NC : 36;
      const toField = u => prior === "vcae" ? vcaeDecode(u).map(v => G.SHALE + (G.SAND - G.SHALE) * v) : prior === "grid" ? Float64Array.from(u) : G.dctField(u.map((v, i) => v * G.PSTD[i]));
      const fwd = u => surrogate(toField(u), well);
      // back from a grid field to the parameters: DCT projection, or the VCAE encoder
      const fromField = f => prior === "grid" ? Array.from(f) : prior === "vcae"
        ? Array.from(vcaeEncode(f.map(v => Math.min(1, Math.max(0, (v - G.SHALE) / (G.SAND - G.SHALE))))))
        : G.project(f).map((c, i) => (G.PSTD[i] > 0 ? c / G.PSTD[i] : 0));
      // localisation needs parameters with a position: the grid cells, or the DCT field; the 8 VCAE
      // latent variables are global, so the VCAE is updated in its latent space without a taper
      const Lr = prior === "vcae" ? 0 : P_.loc | 0, beta = P_.infl || 1, RHO = Lr > 0 ? taper(Lr) : null;
      const spread = F => { const m = meanOf(F); let s = 0; for (let k = 0; k < NC; k++) s += Math.sqrt(F.reduce((a, f) => a + (f[k] - m[k]) ** 2, 0) / F.length); return s / NC; };
      stage(4); log("4. " + (method === "areki" ? "alpha-REKI" : "ES-MDA") + " with the " + (prior === "vcae" ? "VCAE prior (8 latent variables)" : prior === "grid" ? "grid-cell parameters (256 values of ln K, Gaussian prior ensemble, no reduced basis)" : "DCT prior (36 coefficients)") +
                    ", forward model FNO (states) + " + (well === "ccr" ? "CCR" : "FNO") + " (wells), " + Ne + " members, " +
                    (prior === "vcae" ? "no localisation (the latent variables are global)" : Lr > 0 ? "localisation radius " + Lr + " cells" + (prior === "grid" ? "" : " (update on the grid, projected back onto the DCT basis)") : "no localisation") +
                    (beta > 1 ? ", inflation " + beta.toFixed(2) + " whenever the spread falls below half the prior spread." : ", no inflation."));
      const re = rng(caseNo * 1000 + (prior === "vcae" ? 7 : prior === "grid" ? 5 : 3));
      let ens = prior === "grid"
        ? Array.from({ length: Ne }, (_, j) => Array.from(G.dctField(G.dctPrior(caseNo * 7919 + j))))   // Gaussian fields, then free cell values
        : Array.from({ length: Ne }, () => Array.from({ length: dim }, () => gauss(re)));
      const meanOf = A => A[0].map((_, j) => A.reduce((s, a) => s + a[j], 0) / A.length);
      let runs = 0, D = null, priorBand = null; const evo = [];
      const evalEns = async () => { D = []; for (let j = 0; j < Ne; j++) { D.push(fwd(ens[j])); runs++; if (j % 10 === 9) await tick(); } };
      const show = (it, extra) => {
        const F = ens.map(toField), mean = meanOf(F), std = mean.map((m, k) => Math.sqrt(F.reduce((s, f) => s + (f[k] - m) ** 2, 0) / F.length));
        let bi = 0, bv = Infinity;                 // the member that matches the data best (surrogate)
        D.forEach((d, j) => { let s = 0; for (let p = 0; p < ND; p++) s += ((d[p] - dobs[p]) / sig[p]) ** 2; if (s < bv) { bv = s; bi = j; } });
        const best = F[bi];
        board.results[prior] = Object.assign(board.results[prior] || {}, { mean, best, std, runs, method: method === "areki" ? "alpha-REKI" : "ES-MDA" });
        drawBoard(); drawEnsemble(F, (prior === "vcae" ? "VCAE" : prior === "grid" ? "Grid-cell" : "DCT") + " ensemble, " + (it ? "iteration " + it : "prior") + extra);
        const q = j => { const v = D.map(d => d[j]).sort((a, b) => a - b); return [v[Math.floor(0.1 * (Ne - 1))], v[Math.ceil(0.9 * (Ne - 1))]]; };
        const lo = new Float64Array(ND), hi = new Float64Array(ND), md = meanOf(D);
        for (let j = 0; j < ND; j++) [lo[j], hi[j]] = q(j);
        if (!it) priorBand = [lo, hi];
        drawData([{ band: priorBand, col: "rgba(160,160,170,.18)" }, { band: [lo, hi], col: "rgba(79,160,255,.30)" },
                  { members: D.slice(0, 30), col: "rgba(79,160,255,.35)" }, { y: md, col: "#4fa0ff", w: 2 },
                  { y: truth.d, col: "#ff4d4d", w: 2 }, { y: dobs, dots: true }],
                 "ensemble (blue: members, P10 to P90 band, mean), prior band (grey), true model (red), " + (it ? "iteration " + it : "prior"));
        evo.push({ f: mean, b: best, t: it ? "iteration " + it : "prior" }); drawEvo(evo, std);
        score();
      };
      await evalEns(); show(0, "");
      const spread0 = spread(ens.map(toField)), ens0 = ens.map(u => u.slice());
      log("&nbsp;&nbsp; prior: spread of ln K (ensemble std, mean over cells) " + spread0.toFixed(3));
      const phis = () => D.map(d => d.reduce((a, v, j) => a + 0.5 * ((v - dobs[j]) / sig[j]) ** 2, 0));
      const phi = () => phis().reduce((a, b) => a + b, 0) / Ne;
      let sumInv = 0, aPrev = Infinity, it = 0; const alphas = [];
      const maxIt = method === "areki" ? 20 : Na;
      while (it < maxIt) {
        let alpha = Na;
        if (method === "areki") {
          // Iglesias and Yang: 1/alpha = max(n_d / (2 mean Phi), sqrt(n_d / (2 var Phi))), capped at 1 - sum 1/alpha
          const ph = phis(), mu = ph.reduce((a, b) => a + b, 0) / Ne, va = ph.reduce((a, b) => a + (b - mu) ** 2, 0) / Ne;
          alpha = 1 / Math.min(Math.max(ND / (2 * mu), Math.sqrt(ND / (2 * va))), 1 - sumInv);
          if (it === maxIt - 1) alpha = 1 / (1 - sumInv);     // the last allowed iteration completes sum 1/alpha = 1
        }
        const um = meanOf(ens), dm = meanOf(D), Cud = Array.from({ length: dim }, () => new Float64Array(ND)), Cdd = Array.from({ length: ND }, () => new Float64Array(ND));
        for (let j = 0; j < Ne; j++) for (let p = 0; p < ND; p++) {
          const dp = D[j][p] - dm[p];
          for (let q = 0; q < dim; q++) Cud[q][p] += (ens[j][q] - um[q]) * dp / (Ne - 1);
          for (let q = p; q < ND; q++) Cdd[p][q] += dp * (D[j][q] - dm[q]) / (Ne - 1);
        }
        for (let p = 0; p < ND; p++) { for (let q = 0; q < p; q++) Cdd[p][q] = Cdd[q][p]; Cdd[p][p] += alpha * sig[p] * sig[p]; }
        const L = Cdd;
        for (let i = 0; i < ND; i++) { for (let j = 0; j <= i; j++) { let s = L[i][j]; for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k]; L[i][j] = i === j ? Math.sqrt(Math.max(s, 1e-12)) : s / L[j][j]; } }
        const solve = v => { const y = Float64Array.from(v); for (let i = 0; i < ND; i++) { for (let k = 0; k < i; k++) y[i] -= L[i][k] * y[k]; y[i] /= L[i][i]; } for (let i = ND - 1; i >= 0; i--) { for (let k = i + 1; k < ND; k++) y[i] -= L[k][i] * y[k]; y[i] /= L[i][i]; } return y; };
        if (RHO) {
          // localised update on the grid: K = (rho o C_fd) (C_dd + alpha C_d)^-1
          const F = ens.map(toField), fm = meanOf(F), Cfd = Array.from({ length: NC }, () => new Float64Array(ND));
          for (let j = 0; j < Ne; j++) for (let k = 0; k < NC; k++) { const df = (F[j][k] - fm[k]) / (Ne - 1); if (df) for (let p = 0; p < ND; p++) Cfd[k][p] += df * (D[j][p] - dm[p]); }
          for (let k = 0; k < NC; k++) for (let p = 0; p < ND; p++) Cfd[k][p] *= RHO[k][p];
          ens = F.map((f, j) => { const w = solve(dobs.map((v, p) => v + Math.sqrt(alpha) * sig[p] * gauss(re) - D[j][p]));
            return fromField(f.map((v, k) => v + Cfd[k].reduce((s, c, p) => s + c * w[p], 0))); });
        } else {
          ens = ens.map((u, j) => { const w = solve(dobs.map((v, p) => v + Math.sqrt(alpha) * sig[p] * gauss(re) - D[j][p])); return u.map((v, q) => v + Cud[q].reduce((s, c, p) => s + c * w[p], 0)); });
        }
        let inflated = false;
        if (beta > 1 && spread(ens.map(toField)) < 0.5 * spread0) {    // inflation, only against collapse
          const m = meanOf(ens); ens = ens.map(u => u.map((v, q) => m[q] + beta * (v - m[q]))); inflated = true;
        }
        sumInv += 1 / alpha; aPrev = alpha; it++; alphas.push(alpha);
        await evalEns(); show(it, ", alpha " + alpha.toFixed(1)); drawSide({ alphas, label: (prior === "vcae" ? "VCAE" : prior === "grid" ? "grid cells" : "DCT") + (method === "areki" ? ", alpha-REKI" : ", ES-MDA") });
        log("&nbsp;&nbsp; iteration " + it + ": alpha = " + alpha.toFixed(2) + ", sum 1/alpha = " + Math.min(1, sumInv).toFixed(3) + ", mean data misfit " + Math.sqrt(2 * phi() / ND).toFixed(2) + " noise units, spread of ln K " + spread(ens.map(toField)).toFixed(3) + (inflated ? " (inflated)" : ""));
        $(".prog").style.width = (100 * (method === "areki" ? Math.min(1, sumInv) : it / Na)).toFixed(0) + "%";
        await tick(450);
        if (method === "areki" && sumInv >= 1 - 1e-9) break;
      }
      stage(4, "done");
      // verify with the simulator on the posterior mean field
      stage(5); log("5. Verifying with the simulator: the posterior mean, 16 posterior members and the same 16 members of the starting (prior) ensemble.");
      await tick();
      const mean = board.results[prior].mean, ver = simulate(mean), post = [], pri = [];
      for (let j = 0; j < Ne && post.length < 16; j += Math.max(1, Math.floor(Ne / 16))) {
        const r = simulate(toField(ens[j])), r0 = simulate(toField(ens0[j]));
        if (r) post.push(r.d); if (r0) pri.push(r0.d); await tick();
      }
      const pb = [new Float64Array(ND), new Float64Array(ND)];
      for (let j = 0; j < ND; j++) { const v = pri.map(d => d[j]).sort((a, b) => a - b); pb[0][j] = v[Math.floor(0.1 * (v.length - 1))]; pb[1][j] = v[Math.ceil(0.9 * (v.length - 1))]; }
      board.results[prior].mis = ver ? misfit(ver.d) : NaN; board.results[prior].runs = runs;
      const R_ = board.results[prior], vb = simulate(R_.best), vf = simulate(R_.mean.map(v => (v > 0 ? G.SAND : G.SHALE)));
      R_.misB = vb ? misfit(vb.d) : NaN; R_.misF = vf ? misfit(vf.d) : NaN;
      log("&nbsp;&nbsp; simulated misfit of the best-matching member " + R_.misB.toFixed(2) + ", of the thresholded mean " + R_.misF.toFixed(2) + " noise units.");
      const plo = new Float64Array(ND), phi_ = new Float64Array(ND);
      for (let j = 0; j < ND; j++) { const v = post.map(d => d[j]).sort((a, b) => a - b); plo[j] = v[Math.floor(0.1 * (v.length - 1))]; phi_[j] = v[Math.ceil(0.9 * (v.length - 1))]; }
      let inside = 0; for (let j = 0; j < ND; j++) if (truth.d[j] >= plo[j] - 1e-9 && truth.d[j] <= phi_[j] + 1e-9) inside++;
      drawData([{ band: pb, col: "rgba(160,160,170,.16)" }, { members: pri, col: "rgba(170,170,180,.30)" },
                { band: [plo, phi_], col: "rgba(79,160,255,.32)" }, { members: post, col: "rgba(79,160,255,.40)" },
                { y: ver.d, col: "#2f7dff", w: 2.6 }, { y: truth.d, col: "#ff4d4d", w: 2 }, { y: dobs, dots: true }],
               "simulator: prior members (grey), posterior members and P10 to P90 band (blue), posterior mean (thick blue), true model (red)");
      log("&nbsp;&nbsp; the posterior band from the simulator contains the true model at " + (100 * inside / ND).toFixed(0) + " % of the data points.");
      log("&nbsp;&nbsp; data misfit of the matched model, simulated: " + board.results[prior].mis.toFixed(2) + " noise units, " + runs + " surrogate evaluations.");
      stage(5, "done"); score(); drawBoard();
    } catch (e) { log("Stopped: " + (e && e.message ? e.message : e) + ". Please try again."); }
    finally { busy = false; setBusy(false); }
  }
  function setBusy(b) { root.querySelectorAll("button").forEach(x => { x.disabled = b; }); }

  // ------------------------------------------------------------ the player's guess
  let painting = false;
  function paintAt(ev) {
    const L = boardLayout(), rect = CV.board.getBoundingClientRect(), sx = CV.board.width / rect.width;
    const x = (ev.clientX - rect.left) * sx - L.x(1), y = (ev.clientY - rect.top) * sx - L.y;
    if (x < 0 || y < 0 || x >= L.s || y >= L.s) return;
    const i = Math.floor(x / L.s * n), j = Math.floor(y / L.s * n), v = P_.brush === "shale" ? G.SHALE : G.SAND, rad = P_.brushsize | 0;
    for (let dj = -rad + 1; dj < rad; dj++) for (let di = -rad + 1; di < rad; di++) {
      const ii = i + di, jj = j + dj; if (ii >= 0 && ii < n && jj >= 0 && jj < n && di * di + dj * dj < rad * rad) board.guess[jj * n + ii] = v;
    }
    drawBoard();
  }
  CV.board.addEventListener("pointerdown", ev => { readParams(); painting = true; CV.board.setPointerCapture(ev.pointerId); paintAt(ev); });
  CV.board.addEventListener("pointermove", ev => { if (painting) paintAt(ev); });
  CV.board.addEventListener("pointerup", () => { painting = false; });
  $(".tryguess").addEventListener("click", async () => {
    if (busy) return; busy = true; setBusy(true);
    if (!truth) newCase();
    log("Your guess goes to the simulator.");
    await tick();
    const sim = simulate(board.guess);
    const r = board.results.guess = { mean: board.guess.slice(), mis: sim ? misfit(sim.d) : NaN, runs: ((board.results.guess && board.results.guess.runs) || 0) + 1 };
    drawData([{ y: sim.d, col: "#ffd166", w: 2.4 }, { y: truth.d, col: "#ff4d4d", w: 2 }, { y: dobs, dots: true }], "simulator on your guess (yellow) against the true model (red)");
    log("&nbsp;&nbsp; your data misfit: " + r.mis.toFixed(2) + " noise units (1 is as good as the noise allows).");
    score(); busy = false; setBusy(false);
  });
  $(".clearguess").addEventListener("click", () => { board.guess.fill(G.SHALE); drawBoard(); });
  $(".reveal").addEventListener("click", () => { board.revealed = true; drawBoard(); score(); drawEvo([]); log("Revealed: the hidden reservoir of case #" + caseNo + "."); });
  $(".newcase").addEventListener("click", () => { if (busy) return; caseNo++; $(".wlog").innerHTML = ""; stage(0); newCase(); });
  $(".run-dct").addEventListener("click", () => invert("dct"));
  $(".run-vcae").addEventListener("click", () => invert("vcae"));
  $(".run-grid").addEventListener("click", () => invert("grid"));
  root.querySelectorAll("input,select").forEach(el => el.addEventListener("input", () => { readParams(); if (el.name === "play" || el.name === "est") { drawBoard(); score(); if (board.evo) drawEvo(board.evo, board.evoStd); } if ((el.name === "noiseq" || el.name === "noisep") && !busy) newCase(); }));
  readParams();
  // the case is built on first view, so the page loads without a simulator run
  const io = new IntersectionObserver(es => { if (es.some(e => e.isIntersecting)) { io.disconnect(); newCase(); loadNets().catch(() => log("Could not load the trained networks.")); } });
  io.observe(root);
  window.__INV = { vcaeEncode, loadNets, states, wells, vcaeDecode, fnoRun, simulate, trainCCR, get NET() { return NET; } };
})();
