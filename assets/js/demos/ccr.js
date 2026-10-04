/* ccr.js -- cluster-classify-regress (CCR), the mixture of experts used as a
   surrogate in the Norne workflow:
     cluster   k-means (k-means++ start) on the joint, standardised (x, y),
               K chosen by the elbow of the within-cluster sum of squares;
     classify  a random forest (CART, Gini, bootstrap, sqrt(D) features per
               split) predicts the cluster from x alone;
     regress   one expert per cluster, a quadratic ridge regression or a
               Gaussian process (RBF kernel; length scale and noise chosen by
               leave-one-out cross-validation); the prediction is the hard
               mixture, the expert of the predicted cluster.

   window.CCR.train(X, Y, opts) -> { K, sse, acc, predict(x), single(x), labels } */
(function () {
  "use strict";
  function rng(seed) {
    let a = seed >>> 0;
    return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  }
  function kmeans(Z, K, r) {
    const N = Z.length, D = Z[0].length, C = [Z[Math.floor(r() * N)].slice()];
    while (C.length < K) {
      const d2 = Z.map(z => Math.min(...C.map(c => c.reduce((s, v, j) => s + (v - z[j]) ** 2, 0))));
      let u = r() * d2.reduce((a, b) => a + b, 0), i = 0; while (u > d2[i] && i < N - 1) u -= d2[i++];
      C.push(Z[i].slice());
    }
    const lab = new Int32Array(N); let sse = 0;
    for (let it = 0; it < 25; it++) {
      sse = 0;
      for (let i = 0; i < N; i++) { let b = 0, bd = Infinity; for (let k = 0; k < K; k++) { let s = 0; for (let j = 0; j < D; j++) s += (C[k][j] - Z[i][j]) ** 2; if (s < bd) { bd = s; b = k; } } lab[i] = b; sse += bd; }
      const S = Array.from({ length: K }, () => new Float64Array(D)), cnt = new Float64Array(K);
      for (let i = 0; i < N; i++) { cnt[lab[i]]++; for (let j = 0; j < D; j++) S[lab[i]][j] += Z[i][j]; }
      for (let k = 0; k < K; k++) if (cnt[k]) for (let j = 0; j < D; j++) C[k][j] = S[k][j] / cnt[k];
    }
    return { lab, sse };
  }
  function tree(X, y, K, idx, depth, r) {
    const cnt = new Float64Array(K); for (const i of idx) cnt[y[i]]++;
    const maj = cnt.indexOf(Math.max(...cnt));
    if (depth === 0 || idx.length < 6 || Math.max(...cnt) === idx.length) return { leaf: maj };
    const D = X[0].length, nf = Math.max(1, Math.round(Math.sqrt(D)));
    let best = null;
    for (let f = 0; f < nf; f++) {
      const j = Math.floor(r() * D);
      for (let c = 0; c < 8; c++) {
        const thr = X[idx[Math.floor(r() * idx.length)]][j], L = new Float64Array(K), R = new Float64Array(K); let nl = 0;
        for (const i of idx) { if (X[i][j] <= thr) { L[y[i]]++; nl++; } else R[y[i]]++; }
        const nr = idx.length - nl; if (!nl || !nr) continue;
        const gini = (A, m) => 1 - A.reduce((s, v) => s + (v / m) ** 2, 0), sc = nl * gini(L, nl) + nr * gini(R, nr);
        if (!best || sc < best.sc) best = { sc, j, thr };
      }
    }
    if (!best) return { leaf: maj };
    const li = idx.filter(i => X[i][best.j] <= best.thr), ri = idx.filter(i => X[i][best.j] > best.thr);
    return { j: best.j, thr: best.thr, l: tree(X, y, K, li, depth - 1, r), r: tree(X, y, K, ri, depth - 1, r) };
  }
  function treePredict(t, x) { while (t.leaf === undefined) t = x[t.j] <= t.thr ? t.l : t.r; return t.leaf; }
  const poly = x => { const f = [1, ...x]; for (let a = 0; a < x.length; a++) for (let b = a; b < x.length; b++) f.push(x[a] * x[b]); return f; };
  function ridge(Fs, Ys, lam) {
    const p = Fs[0].length, q = Ys[0].length, A = Array.from({ length: p }, () => new Float64Array(p)), B = Array.from({ length: p }, () => new Float64Array(q));
    Fs.forEach((f, s) => { for (let a = 0; a < p; a++) { for (let b = a; b < p; b++) A[a][b] += f[a] * f[b]; for (let c = 0; c < q; c++) B[a][c] += f[a] * Ys[s][c]; } });
    for (let a = 0; a < p; a++) { for (let b = 0; b < a; b++) A[a][b] = A[b][a]; A[a][a] += lam; }
    for (let k = 0; k < p; k++) for (let a = k + 1; a < p; a++) { const f = A[a][k] / A[k][k]; for (let b = k; b < p; b++) A[a][b] -= f * A[k][b]; for (let c = 0; c < q; c++) B[a][c] -= f * B[k][c]; }
    const W = Array.from({ length: p }, () => new Float64Array(q));
    for (let a = p - 1; a >= 0; a--) for (let c = 0; c < q; c++) { let s = B[a][c]; for (let b = a + 1; b < p; b++) s -= A[a][b] * W[b][c]; W[a][c] = s / A[a][a]; }
    return W;
  }
  const ridgePredict = (W, f) => { const q = W[0].length, y = new Float64Array(q); for (let a = 0; a < f.length; a++) { const v = f[a], w = W[a]; for (let c = 0; c < q; c++) y[c] += v * w[c]; } return y; };
  // Gaussian-process expert (kernel ridge with an RBF kernel), hyperparameters by closed-form LOO
  function chol(A) {
    const n = A.length, L = A.map(r => Float64Array.from(r));
    for (let i = 0; i < n; i++) { for (let j = 0; j <= i; j++) { let s = L[i][j]; for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      if (i === j) { if (s <= 0) return null; L[i][i] = Math.sqrt(s); } else L[i][j] = s / L[j][j]; } for (let j = i + 1; j < n; j++) L[i][j] = 0; }
    return L;
  }
  function cholInv(L) {                    // (L L^T)^-1
    const n = L.length, Li = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) { Li[i][i] = 1 / L[i][i]; for (let j = 0; j < i; j++) { let s = 0; for (let k = j; k < i; k++) s -= L[i][k] * Li[k][j]; Li[i][j] = s / L[i][i]; } }
    const A = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) { let s = 0; for (let k = i; k < n; k++) s += Li[k][i] * Li[k][j]; A[i][j] = A[j][i] = s; }
    return A;
  }
  function gp(Xs, Ys) {
    const n = Xs.length, q = Ys[0].length, d = Xs[0].length;
    const D2 = Xs.map(a => Xs.map(b => a.reduce((s, v, j) => s + (v - b[j]) ** 2, 0)));
    let best = null;
    for (const ls of [0.5, 0.75, 1, 1.5, 2, 3].map(v => v * Math.sqrt(d) / 2)) for (const nz of [1e-4, 1e-3, 1e-2, 3e-2, 1e-1]) {
      const K = D2.map((r, i) => r.map((v, j) => Math.exp(-v / (2 * ls * ls)) + (i === j ? nz : 0)));
      const L = chol(K); if (!L) continue;
      const Ki = cholInv(L), A = Ki.map(r => { const a = new Float64Array(q); for (let j = 0; j < n; j++) { const w = r[j], y = Ys[j]; for (let c = 0; c < q; c++) a[c] += w * y[c]; } return a; });
      let loo = 0; for (let i = 0; i < n; i++) for (let c = 0; c < q; c++) loo += (A[i][c] / Ki[i][i]) ** 2;     // LOO residual = alpha_i / (K^-1)_ii
      if (!best || loo < best.loo) best = { loo, ls, A };
    }
    const { ls, A } = best;
    return x => { const y = new Float64Array(q); for (let i = 0; i < n; i++) { const k = Math.exp(-Xs[i].reduce((s, v, j) => s + (v - x[j]) ** 2, 0) / (2 * ls * ls)), a = A[i]; for (let c = 0; c < q; c++) y[c] += k * a[c]; } return y; };
  }
  const stats = A => { const m = A[0].map((_, j) => A.reduce((s, x) => s + x[j], 0) / A.length);
    return [m, m.map((mu, j) => Math.sqrt(A.reduce((s, x) => s + (x[j] - mu) ** 2, 0) / A.length) || 1)]; };

  // X: inputs, Y: outputs (arrays of arrays); opts: Kmax, trees, depth, lam, yWeight, seed, minCluster
  function train(X, Y, opts) {
    const o = Object.assign({ Kmax: 5, trees: 25, depth: 9, lam: 1e-2, yWeight: 1.5, seed: 7, minCluster: 0, expert: "poly", overlap: 0 }, opts || {});
    const [xm, xs] = stats(X), [ym, ys] = stats(Y);
    const nx = x => x.map((v, j) => (v - xm[j]) / xs[j]);
    const Xn = X.map(nx), Yn = Y.map(y => y.map((v, j) => (v - ym[j]) / ys[j]));
    const wy = o.yWeight / Math.sqrt(Y[0].length / X[0].length);       // balance x and y in the joint space
    const Z = Xn.map((x, s) => [...x, ...Yn[s].map(v => wy * v)]);
    const minC = o.minCluster || Math.max(8, (X[0].length + 1) * (X[0].length + 2) / 2 / 2);
    const sse = [];
    for (let K = 1; K <= o.Kmax; K++) sse.push(kmeans(Z, K, rng(o.seed + K)).sse);
    let K = 1, bestCurv = -Infinity;                                  // elbow: largest second difference
    for (let k = 2; k < o.Kmax; k++) { const c = (sse[k - 2] - sse[k - 1]) - (sse[k - 1] - sse[k]); if (c > bestCurv) { bestCurv = c; K = k; } }
    let lab = kmeans(Z, K, rng(o.seed + K)).lab;
    while (K > 1) {                                                   // every expert needs enough samples
      const cnt = new Float64Array(K); for (const l of lab) cnt[l]++;
      if (Math.min(...cnt) >= minC) break;
      K--; lab = K > 1 ? kmeans(Z, K, rng(o.seed + K)).lab : new Int32Array(X.length);
    }
    const r = rng(o.seed * 31 + 1), forest = [];
    if (K > 1) for (let t = 0; t < o.trees; t++) forest.push(tree(Xn, lab, K, Array.from({ length: Xn.length }, () => Math.floor(r() * Xn.length)), o.depth, r));
    const votes = z => { const v = new Float64Array(K); if (K === 1) { v[0] = 1; return v; } for (const t of forest) v[treePredict(t, z)] += 1 / forest.length; return v; };
    const classify = z => { const v = votes(z); return v.indexOf(Math.max(...v)); };
    let acc = 0; Xn.forEach((z, s) => { if (classify(z) === lab[s]) acc++; });
    const fit = (Xs, Ts) => { if (o.expert === "gp") return gp(Xs, Ts); const W = ridge(Xs.map(poly), Ts, o.lam * Xs.length + 1e-6); return z => ridgePredict(W, poly(z)); };
    const experts = [];
    // each expert trains on its cluster and, with overlap > 0, on the samples the forest places near its
    // boundary (vote share >= overlap), so neighbouring experts agree where the regimes meet
    const V = Xn.map(votes);
    for (let k = 0; k < K; k++) { const Xs = [], T = []; Xn.forEach((z, s) => { if (lab[s] === k || (o.overlap > 0 && V[s][k] >= o.overlap)) { Xs.push(z); T.push(Yn[s]); } }); experts.push(fit(Xs, T)); }
    const single = fit(Xn, Yn);
    const back = yn => Array.from(yn, (v, j) => v * ys[j] + ym[j]);
    return { K, sse, acc: acc / X.length, labels: lab,
             predict: x => { const z = nx(x); return back(experts[classify(z)](z)); },
             single: x => back(single(nx(x))) };
  }
  window.CCR = { train };
})();
