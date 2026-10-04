/* precision.js -- the precision argument of mixed-precision CPR, run live.
   The pressure equation  -div(K grad p) = q  on an n x n grid (5-point
   stencil, log-normal K with a chosen contrast, injector and producer) is
   the system CPR's first stage works on.  It is solved with restarted
   flexible GMRES, preconditioned by one aggregation-multigrid V-cycle
   (2 x 2 aggregates, Galerkin coarse operators, damped-Jacobi smoothing,
   exact coarsest solve) -- the role AMG plays in CPR stage 1.  Four arms:
     FP64            FGMRES and V-cycle in double precision;
     mixed (FP32)    FGMRES in FP64, V-cycle in FP32  (the paper's design);
     mixed (FP16)    FGMRES in FP64, V-cycle in FP16 on the Jacobi-scaled
                     matrix (half precision cannot hold field-unit entries);
     all FP32        FGMRES recurrences and V-cycle both in FP32.
   The plotted quantity is the TRUE relative residual ||b - A x|| / ||b||
   evaluated in FP64 at every iteration: with an FP64 outer method the
   precision of the preconditioner changes how many iterations are needed,
   not the accuracy reached; when the outer method is reduced too, the
   residual stalls near that precision's unit roundoff.                   */
(function () {
  "use strict";
  const root = document.getElementById("demo-precision");
  if (!root) return;
  const $ = s => root.querySelector(s);
  const cv = $("canvas.plot"), cx = cv.getContext("2d");
  const P_ = {};

  const f32 = Math.fround;
  const f16 = x => {                         // round to nearest IEEE binary16
    if (!isFinite(x) || x === 0) return x;
    const a = Math.abs(x);
    if (a >= 65520) return x > 0 ? Infinity : -Infinity;
    let e = Math.floor(Math.log2(a));
    if (e < -14) e = -14;                     // subnormals: spacing 2^-24
    const q = Math.pow(2, e - 10);
    return Math.round(x / q) * q;
  };
  const id = x => x;

  function rng(seed) {
    let a = seed >>> 0;
    return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  }
  function readParams() {
    root.querySelectorAll("input,select").forEach(el => {
      P_[el.name] = parseFloat(el.value);
      const v = root.querySelector('[data-for="' + el.name + '"]');
      if (v) v.textContent = el.name === "contrast" ? ("10^" + (+el.value).toFixed(0)) : el.value;
    });
  }

  // ---- operators: 5-point, stored as tx (i,i+1), ty (j,j+1), diag ----------
  function fineOperator(n, contrast) {
    const N = n * n, r = rng(17 + n);
    const g = new Float64Array(N), modes = [];
    for (let m = 0; m < 30; m++) modes.push([(r() * 2 - 1) * 24 / n, (r() * 2 - 1) * 24 / n, r() * 6.283]);
    let lo = Infinity, hi = -Infinity;
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      let v = 0;
      for (const [a, b, c] of modes) v += Math.cos(a * i + b * j + c);
      g[j * n + i] = v; lo = Math.min(lo, v); hi = Math.max(hi, v);
    }
    const K = new Float64Array(N);
    for (let k = 0; k < N; k++) K[k] = 50 * Math.pow(10, contrast * ((g[k] - lo) / (hi - lo) - 0.5));
    const tx = new Float64Array(N), ty = new Float64Array(N), diag = new Float64Array(N);
    const h = (a, b) => 2 * a * b / (a + b);
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const k = j * n + i;
      if (i < n - 1) tx[k] = h(K[k], K[k + 1]);
      if (j < n - 1) ty[k] = h(K[k], K[k + n]);
    }
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const k = j * n + i;
      diag[k] = (i > 0 ? tx[k - 1] : 2 * K[k]) + (i < n - 1 ? tx[k] : 2 * K[k]) +
                (j > 0 ? ty[k - n] : 2 * K[k]) + (j < n - 1 ? ty[k] : 2 * K[k]);
    }
    const b = new Float64Array(N);
    b[Math.floor(n / 4) * n + Math.floor(n / 4)] = 1e3;            // injector
    b[Math.floor(3 * n / 4) * n + Math.floor(3 * n / 4)] = -1e3;   // producer
    for (let k = 0; k < N; k++) b[k] += r() - 0.5;
    return { n, N, tx, ty, diag, b };
  }
  // Galerkin coarse operator for 2 x 2 aggregates (P piecewise constant)
  function coarsen(A) {
    const n = A.n, m = n / 2, M = m * m;
    const tx = new Float64Array(M), ty = new Float64Array(M), diag = new Float64Array(M);
    const f = (i, j) => j * n + i;
    for (let J = 0; J < m; J++) for (let I = 0; I < m; I++) {
      const K = J * m + I, i = 2 * I, j = 2 * J;
      let d = A.diag[f(i, j)] + A.diag[f(i + 1, j)] + A.diag[f(i, j + 1)] + A.diag[f(i + 1, j + 1)];
      d -= 2 * (A.tx[f(i, j)] + A.tx[f(i, j + 1)] + A.ty[f(i, j)] + A.ty[f(i + 1, j)]);
      diag[K] = d;
      if (I < m - 1) tx[K] = A.tx[f(i + 1, j)] + A.tx[f(i + 1, j + 1)];
      if (J < m - 1) ty[K] = A.ty[f(i, j + 1)] + A.ty[f(i + 1, j + 1)];
    }
    return { n: m, N: M, tx, ty, diag };
  }
  function matvec(A, x, y, rd) {
    const { n, tx, ty, diag } = A;
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const k = j * n + i;
      let s = rd(diag[k] * x[k]);
      if (i > 0) s = rd(s - rd(tx[k - 1] * x[k - 1]));
      if (i < n - 1) s = rd(s - rd(tx[k] * x[k + 1]));
      if (j > 0) s = rd(s - rd(ty[k - n] * x[k - n]));
      if (j < n - 1) s = rd(s - rd(ty[k] * x[k + n]));
      y[k] = s;
    }
  }
  // hierarchy stored in precision rd (optionally Jacobi-scaled: D^-1/2 A D^-1/2)
  function hierarchy(A0, rd, scaled) {
    let A = A0, dsc = null;
    if (scaled) {
      dsc = A0.diag.map(d => 1 / Math.sqrt(d));
      const n = A0.n;
      A = { n, N: A0.N, tx: A0.tx.map((t, k) => t * dsc[k] * (k % n < n - 1 ? dsc[k + 1] : 0)),
            ty: A0.ty.map((t, k) => t * dsc[k] * (k + n < A0.N ? dsc[k + n] : 0)), diag: A0.diag.map(() => 1) };
    }
    const lev = [A];
    while (lev[lev.length - 1].n > 4) lev.push(coarsen(lev[lev.length - 1]));
    const L = lev.map(a => ({ n: a.n, N: a.N, tx: a.tx.map(rd), ty: a.ty.map(rd), diag: a.diag.map(rd),
                              dinv: a.diag.map(d => rd(0.8 / d)) }));
    // dense LU (FP64) of the coarsest level
    const c = lev[lev.length - 1], Nc = c.N, D = [];
    for (let a = 0; a < Nc; a++) { D.push(new Float64Array(Nc)); }
    for (let k = 0; k < Nc; k++) {
      const e = new Float64Array(Nc); e[k] = 1; const y = new Float64Array(Nc);
      matvec(c, e, y, id); for (let a = 0; a < Nc; a++) D[a][k] = y[a];
    }
    for (let k = 0; k < Nc; k++) for (let a = k + 1; a < Nc; a++) {
      const f = D[a][k] / D[k][k]; D[a][k] = f; for (let b = k + 1; b < Nc; b++) D[a][b] -= f * D[k][b];
    }
    return { L, LU: D, dsc, rd };
  }
  function vcycle(H, l, b, rd) {
    const A = H.L[l], N = A.N, x = new Float64Array(N);
    if (l === H.L.length - 1) {                               // coarsest: LU
      const y = Float64Array.from(b);
      for (let a = 0; a < N; a++) for (let k = 0; k < a; k++) y[a] -= H.LU[a][k] * y[k];
      for (let a = N - 1; a >= 0; a--) { for (let k = a + 1; k < N; k++) y[a] -= H.LU[a][k] * y[k]; y[a] /= H.LU[a][a]; }
      return y.map(rd);
    }
    const r = new Float64Array(N), Ax = new Float64Array(N);
    const smooth = () => {
      matvec(A, x, Ax, rd);
      for (let k = 0; k < N; k++) x[k] = rd(x[k] + rd(A.dinv[k] * rd(b[k] - Ax[k])));
    };
    smooth(); smooth();
    matvec(A, x, Ax, rd);
    for (let k = 0; k < N; k++) r[k] = rd(b[k] - Ax[k]);
    const n = A.n, m = n / 2, rc = new Float64Array(m * m);
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++)
      rc[(j >> 1) * m + (i >> 1)] = rd(rc[(j >> 1) * m + (i >> 1)] + r[j * n + i]);
    const ec = vcycle(H, l + 1, rc, rd);
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) x[j * n + i] = rd(x[j * n + i] + ec[(j >> 1) * m + (i >> 1)]);
    smooth(); smooth();
    return x;
  }
  function precond(H, v) {                 // z = M^{-1} v in H's precision
    const rd = H.rd;
    let b = H.dsc ? v.map((x, k) => rd(x * H.dsc[k])) : v.map(rd);
    let z = vcycle(H, 0, b, rd);
    if (H.dsc) z = z.map((x, k) => x * H.dsc[k]);
    return Float64Array.from(z);
  }
  // restarted FGMRES; outer arithmetic ro; cb(x, it) after every iteration
  function fgmres(A, H, ro, maxit, restart, cb) {
    const N = A.N, b = A.b, x = new Float64Array(N);
    let bn = 0; for (let k = 0; k < N; k++) bn += b[k] * b[k]; bn = Math.sqrt(bn);
    let it = 0;
    const tmp = new Float64Array(N);
    while (it < maxit) {
      matvec(A, x, tmp, ro);
      const r = new Float64Array(N); let beta = 0;
      for (let k = 0; k < N; k++) { r[k] = ro(b[k] - tmp[k]); beta = ro(beta + ro(r[k] * r[k])); }
      beta = Math.sqrt(beta);
      if (beta / bn < 1e-14) break;
      const V = [r.map(v => ro(v / beta))], Z = [], Hm = [], cs = [], sn = [], g = [beta];
      let j = 0;
      for (; j < restart && it < maxit; j++, it++) {
        const z = precond(H, V[j]).map(ro); Z.push(z);
        const w = new Float64Array(N); matvec(A, z, w, ro);
        const h = new Float64Array(j + 2);
        for (let i = 0; i <= j; i++) {             // modified Gram-Schmidt
          let s = 0; for (let k = 0; k < N; k++) s = ro(s + ro(w[k] * V[i][k]));
          h[i] = s; for (let k = 0; k < N; k++) w[k] = ro(w[k] - ro(s * V[i][k]));
        }
        let wn = 0; for (let k = 0; k < N; k++) wn = ro(wn + ro(w[k] * w[k]));
        h[j + 1] = Math.sqrt(wn);
        V.push(w.map(v => ro(v / (h[j + 1] || 1))));
        for (let i = 0; i < j; i++) {               // apply earlier rotations
          const t = cs[i] * h[i] + sn[i] * h[i + 1]; h[i + 1] = -sn[i] * h[i] + cs[i] * h[i + 1]; h[i] = t;
        }
        const den = Math.hypot(h[j], h[j + 1]) || 1;
        cs.push(h[j] / den); sn.push(h[j + 1] / den);
        h[j] = den; h[j + 1] = 0;
        g.push(-sn[j] * g[j]); g[j] = cs[j] * g[j];
        Hm.push(h);
        // current iterate x + Z y (y from the triangular system) for the plot
        const y = new Float64Array(j + 1);
        for (let a = j; a >= 0; a--) { let s = g[a]; for (let c = a + 1; c <= j; c++) s -= Hm[c][a] * y[c]; y[a] = s / Hm[a][a]; }
        const xi = Float64Array.from(x);
        for (let c = 0; c <= j; c++) for (let k = 0; k < N; k++) xi[k] = ro(xi[k] + ro(y[c] * Z[c][k]));
        const tr = trueRes(A, xi, bn);
        cb(tr, it + 1);
        if (tr < 1e-13 || j === restart - 1 || it + 1 >= maxit || !isFinite(tr)) {
          for (let k = 0; k < N; k++) x[k] = xi[k];
          it++; j++;
          break;
        }
      }
      if (cbStop) break;
    }
  }
  let cbStop = false;
  function trueRes(A, x, bn) {
    const y = new Float64Array(A.N); matvec(A, x, y, id);
    let r = 0; for (let k = 0; k < A.N; k++) r += (A.b[k] - y[k]) ** 2;
    return Math.sqrt(r) / bn;
  }
  const series = {};
  const ARMS = [
    ["fp64", "FP64 FGMRES + FP64 V-cycle", "#e8e6e3", id, id, false],
    ["mix32", "FP64 FGMRES + FP32 V-cycle (paper)", "#00e5a0", id, f32, false],
    ["mix16", "FP64 FGMRES + FP16 V-cycle", "#ffb74d", id, f16, true],
    ["all32", "FP32 FGMRES + FP32 V-cycle", "#4fc3ff", f32, f32, false],
  ];
  function run() {
    readParams();
    const n = P_.n | 0, A = fineOperator(n, P_.contrast);
    for (const [key, , , ro, rv, sc] of ARMS) {
      const H = hierarchy(A, rv, sc), h = [[0, 0]];
      cbStop = false;
      fgmres(A, H, ro, P_.maxit | 0, P_.restart | 0, (tr, it) => {
        h.push([it, Math.log10(Math.max(1e-16, isFinite(tr) ? tr : 1))]);
        if (tr < 1e-13) cbStop = true;
      });
      series[key] = h;
    }
    draw();
    const fin = k => series[k][series[k].length - 1];
    const its = k => { const s = series[k]; for (const p of s) if (p[1] < -10) return p[0]; return "-"; };
    $(".readout").innerHTML = ARMS.map(([k, lab, col]) =>
      '<span style="color:' + col + '">' + lab + "</span>: residual 1e" + fin(k)[1].toFixed(1) +
      ", iterations to 1e-10: " + its(k)).join("<br>");
  }
  function draw() {
    const W = cv.width, H = cv.height, l = 52, r = 14, t = 14, b = 36;
    cx.fillStyle = "#07070a"; cx.fillRect(0, 0, W, H);
    let xmax = 10;
    for (const k in series) xmax = Math.max(xmax, series[k][series[k].length - 1][0]);
    const ymin = -15, ymax = 0;
    const X = v => l + (W - l - r) * v / xmax, Y = v => t + (H - t - b) * (ymax - v) / (ymax - ymin);
    cx.font = "11px DM Mono, monospace"; cx.fillStyle = "#9a9997"; cx.strokeStyle = "rgba(255,255,255,.1)";
    for (let e = 0; e >= -15; e -= 3) {
      cx.beginPath(); cx.moveTo(l, Y(e)); cx.lineTo(W - r, Y(e)); cx.stroke();
      cx.fillText("1e" + e, 8, Y(e) + 4);
    }
    for (let i = 0; i <= 5; i++) cx.fillText(Math.round(xmax * i / 5), X(xmax * i / 5) - 8, H - 18);
    cx.fillText("FGMRES iterations (one V-cycle each)", l, H - 3);
    for (const [u, lab] of [[Math.log10(5.96e-8), "FP32 unit roundoff"], [Math.log10(1.11e-16), "FP64"]]) {
      cx.setLineDash([4, 4]); cx.strokeStyle = "rgba(255,255,255,.22)";
      cx.beginPath(); cx.moveTo(l, Y(u)); cx.lineTo(W - r, Y(u)); cx.stroke(); cx.setLineDash([]);
      cx.fillStyle = "rgba(255,255,255,.35)"; cx.fillText(lab, W - r - 130, Y(u) - 4);
    }
    for (const [k, , col] of ARMS) {
      const h = series[k]; if (!h) continue;
      cx.beginPath(); cx.strokeStyle = col; cx.lineWidth = k === "mix32" ? 2.6 : 1.8;
      if (k === "fp64") cx.setLineDash([6, 4]);
      h.forEach((p, i) => i ? cx.lineTo(X(p[0]), Y(p[1])) : cx.moveTo(X(p[0]), Y(p[1])));
      cx.stroke(); cx.setLineDash([]);
    }
  }
  $(".run").addEventListener("click", () => { $(".readout").textContent = "solving…"; setTimeout(run, 20); });
  root.querySelectorAll("input,select").forEach(el => el.addEventListener("input", readParams));
  readParams();
  setTimeout(run, 60);
})();
