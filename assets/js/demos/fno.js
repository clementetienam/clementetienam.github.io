/* fno.js -- inference for Fourier neural operators trained offline in PyTorch
   (rfft conventions, norm="ortho"), in plain JavaScript:
     2-D  lift (1x1) -> L x [spectral conv on the lowest m x m modes (both
          signs of the first axis) + 1x1 conv, GELU] -> 1x1 -> GELU -> 1x1,
          with the grid coordinates appended to the input channels;
     1-D  the same along one axis (time), m lowest modes.
   window.FNO = { net2, run2, net1, run1, load }                            */
(function () {
  "use strict";
  function erf(x) {
    const s = x < 0 ? -1 : 1; x = Math.abs(x); const t = 1 / (1 + 0.3275911 * x);
    return s * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x));
  }
  const gelu = x => 0.5 * x * (1 + erf(x / Math.SQRT2));
  function pointwise(X, Wb, N, act) {        // 1 x 1 convolution on channels of length N
    const [W, b] = Wb, ci = X.length, co = b.length, Y = [];
    for (let o = 0; o < co; o++) {
      const y = new Float64Array(N).fill(b[o]);
      for (let c = 0; c < ci; c++) { const w = W[o * ci + c], x = X[c]; if (w) for (let k = 0; k < N; k++) y[k] += w * x[k]; }
      if (act) for (let k = 0; k < N; k++) y[k] = act(y[k]);
      Y.push(y);
    }
    return Y;
  }
  function tables(L) {
    const C = new Float64Array(L * L), S = new Float64Array(L * L);
    for (let k = 0; k < L; k++) for (let j = 0; j < L; j++) { C[k * L + j] = Math.cos(2 * Math.PI * k * j / L); S[k * L + j] = Math.sin(2 * Math.PI * k * j / L); }
    return { C, S };
  }
  function layers(get, pre, L) {
    return { lift: [get(pre + "lift.weight"), get(pre + "lift.bias")],
             pw: Array.from({ length: L }, (_, l) => [get(pre + "pw." + l + ".weight"), get(pre + "pw." + l + ".bias")]),
             p1: [get(pre + "p1.weight"), get(pre + "p1.bias")], p2: [get(pre + "p2.weight"), get(pre + "p2.bias")] };
  }
  // ---------------------------------------------------------------- 2-D
  function net2(get, pre, o) {
    const n = o.n, NC = n * n, g = Array.from({ length: n }, (_, i) => (i + 0.5) / n);
    const gx = new Float64Array(NC), gy = new Float64Array(NC);
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) { gx[j * n + i] = g[i]; gy[j * n + i] = g[j]; }
    return Object.assign(layers(get, pre, o.L), { n, NC, m: o.m, L: o.L, width: o.width, gx, gy, T: tables(n),
      spec: Array.from({ length: o.L }, (_, l) => [get(pre + "spec." + l + ".w1"), get(pre + "spec." + l + ".w2")]) });
  }
  function spectral2(F, X, w1, w2) {
    const n = F.n, m = F.m, co = F.width, ci = X.length, { C: TW, S: TH } = F.T, rows = [];
    for (let a = 0; a < m; a++) rows.push(a);
    for (let a = n - m; a < n; a++) rows.push(a);
    const nr = rows.length, Fr = [], Fi = [];
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
          const p = (((c * co + o) * m + a) * m + b) * 2, wr = w[p], wi = w[p + 1], xr = Fr[c][q * m + b], xi = Fi[c][q * m + b];
          or_[q * m + b] += xr * wr - xi * wi; oi[q * m + b] += xr * wi + xi * wr;
        }
      }
      const Zr = new Float64Array(n * m), Zi = new Float64Array(n * m);
      for (let j = 0; j < n; j++) for (let b = 0; b < m; b++) {
        let sr = 0, si = 0;
        for (let q = 0; q < nr; q++) { const a = rows[q], c_ = TW[a * n + j], s_ = TH[a * n + j], ur = or_[q * m + b], ui = oi[q * m + b]; sr += ur * c_ - ui * s_; si += ur * s_ + ui * c_; }
        Zr[j * m + b] = sr; Zi[j * m + b] = si;
      }
      const y = new Float64Array(n * n);
      for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
        let s = Zr[j * m];
        for (let b = 1; b < m; b++) s += 2 * (Zr[j * m + b] * TW[b * n + i] - Zi[j * m + b] * TH[b * n + i]);
        y[j * n + i] = s / n;
      }
      Y.push(y);
    }
    return Y;
  }
  function run2(F, inp) {
    const N = F.NC;
    let h = pointwise([...inp, F.gx, F.gy], F.lift, N);
    for (let l = 0; l < F.L; l++) {
      const s = spectral2(F, h, F.spec[l][0], F.spec[l][1]), p = pointwise(h, F.pw[l], N);
      h = s.map((v, c) => { const r = new Float64Array(N); for (let k = 0; k < N; k++) { const t = v[k] + p[c][k]; r[k] = l < F.L - 1 ? gelu(t) : t; } return r; });
    }
    return pointwise(pointwise(h, F.p1, N, gelu), F.p2, N);
  }
  // ---------------------------------------------------------------- 1-D
  function net1(get, pre, o) {
    const Ln = o.len, g = Float64Array.from({ length: Ln }, (_, i) => (i + 0.5) / Ln);
    return Object.assign(layers(get, pre, o.L), { len: Ln, m: o.m, L: o.L, width: o.width, g, T: tables(Ln),
      spec: Array.from({ length: o.L }, (_, l) => get(pre + "spec." + l + ".w")) });
  }
  function spectral1(F, X, w) {
    const Ln = F.len, m = F.m, co = F.width, ci = X.length, { C, S } = F.T, sq = Math.sqrt(Ln);
    const Fr = [], Fi = [];
    for (let c = 0; c < ci; c++) {
      const x = X[c], fr = new Float64Array(m), fi = new Float64Array(m);
      for (let b = 0; b < m; b++) { let sr = 0, si = 0; for (let t = 0; t < Ln; t++) { sr += x[t] * C[b * Ln + t]; si -= x[t] * S[b * Ln + t]; } fr[b] = sr / sq; fi[b] = si / sq; }
      Fr.push(fr); Fi.push(fi);
    }
    const Y = [];
    for (let o = 0; o < co; o++) {
      const or_ = new Float64Array(m), oi = new Float64Array(m);
      for (let c = 0; c < ci; c++) for (let b = 0; b < m; b++) {
        const p = ((c * co + o) * m + b) * 2, wr = w[p], wi = w[p + 1];
        or_[b] += Fr[c][b] * wr - Fi[c][b] * wi; oi[b] += Fr[c][b] * wi + Fi[c][b] * wr;
      }
      const y = new Float64Array(Ln);
      for (let t = 0; t < Ln; t++) { let s = or_[0]; for (let b = 1; b < m; b++) s += 2 * (or_[b] * C[b * Ln + t] - oi[b] * S[b * Ln + t]); y[t] = s / sq; }
      Y.push(y);
    }
    return Y;
  }
  function run1(F, inp) {
    const N = F.len;
    let h = pointwise([...inp, F.g], F.lift, N);
    for (let l = 0; l < F.L; l++) {
      const s = spectral1(F, h, F.spec[l]), p = pointwise(h, F.pw[l], N);
      h = s.map((v, c) => { const r = new Float64Array(N); for (let k = 0; k < N; k++) { const t = v[k] + p[c][k]; r[k] = l < F.L - 1 ? gelu(t) : t; } return r; });
    }
    return pointwise(pointwise(h, F.p1, N, gelu), F.p2, N);
  }
  // ---------------------------------------------------------------- loading
  async function grab(url, kind) {           // fetch with a timeout and up to three attempts
    for (let a = 0; ; a++) {
      const ctl = new AbortController(), timer = setTimeout(() => ctl.abort(), 15000);
      try { const r = await fetch(url, { signal: ctl.signal }); if (!r.ok) throw new Error(r.status); return await (kind === "json" ? r.json() : r.arrayBuffer()); }
      catch (e) { if (a >= 2) throw e; }
      finally { clearTimeout(timer); }
    }
  }
  async function load(jsonUrl, binUrl) {      // -> { meta, get(name) }
    const [meta, bin] = await Promise.all([grab(jsonUrl, "json"), grab(binUrl)]);
    const all = new Float32Array(bin);
    return { meta, get: k => { const [o, len] = meta.index[k]; return all.subarray(o, o + len); } };
  }
  window.FNO = { net2, run2, net1, run1, load, grab, gelu };
})();
