/* nvrs_engine.js -- the miniature mixed-precision simulator used by the demos.
   Fully implicit two-phase (water/oil) flow on an n x n grid with slightly
   compressible rock; unknowns (p, S_w) per cell, 2 x 2 Jacobian blocks.
   Newton's method; each Newton step solves J du = -R with restarted
   flexible GMRES in FP64, preconditioned by two-stage CPR:
     stage 1  quasi-IMPES weights -> pressure matrix A_p -> one aggregation-
              multigrid V-cycle (2 x 2 aggregates, Galerkin coarse operators,
              damped Jacobi, exact coarsest solve), the role AmgX plays;
     stage 2  2 x 2 block Jacobi on the full system.
   The preconditioner runs in FP64, FP32 or emulated FP16; the state,
   residual, Jacobian and Krylov recurrences stay in FP64.

   window.NVRS = { model(opts), step(M, opts), timeStep(M, dt, prec, restart),
                   mob(s, visc), wellRates(M), swc, sor }                   */
(function () {
  "use strict";
  const f32 = Math.fround, id = x => x;
  const f16 = x => {
    if (!isFinite(x) || x === 0) return x;
    const a = Math.abs(x);
    if (a >= 65520) return x > 0 ? Infinity : -Infinity;
    let e = Math.floor(Math.log2(a)); if (e < -14) e = -14;
    const q = Math.pow(2, e - 10); return Math.round(x / q) * q;
  };
  const RD = { fp64: id, fp32: f32, fp16: f16 };
  const swc = 0.1, sor = 0.1, cr = 2e-3, phi0 = 0.2, pref = 0;
  const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());

  function rng(seed) {
    let a = seed >>> 0;
    return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  }
  function mob(s, visc) {                    // [lw, lo, dlw/dS, dlo/dS]
    const d = 1 - swc - sor;
    const se = Math.min(1, Math.max(0, (s - swc) / d));
    const inside = s > swc && s < 1 - sor ? 1 : 0;
    return [se * se, (1 - se) * (1 - se) / visc, inside * 2 * se / d, -inside * 2 * (1 - se) / d / visc];
  }
  // a random log-permeability field (sum of Fourier modes), standardised
  function randomLogK(n, sigma, seed) {
    const r = rng(31 + seed * 7), modes = [], N = n * n;
    for (let m = 0; m < 36; m++) modes.push([(r() * 2 - 1) * 14 / n, (r() * 2 - 1) * 14 / n, r() * 6.283, 0.5 + r()]);
    const g = new Float64Array(N); let mu = 0, sd = 0;
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      let v = 0; for (const [a, b, c, w] of modes) v += w * Math.cos(a * i + b * j + c);
      g[j * n + i] = v; mu += v;
    }
    mu /= N; for (let k = 0; k < N; k++) sd += (g[k] - mu) ** 2; sd = Math.sqrt(sd / N);
    return g.map(v => sigma * (v - mu) / sd);
  }
  function model(o) {
    const n = o.n, N = n * n;
    const lnK = o.logK || randomLogK(n, o.sigma, o.seed | 0);
    const K = lnK.map(Math.exp);
    const tx = new Float64Array(N), ty = new Float64Array(N);
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const k = j * n + i;
      if (i < n - 1) tx[k] = 2 * K[k] * K[k + 1] / (K[k] + K[k + 1]);
      if (j < n - 1) ty[k] = 2 * K[k] * K[k + n] / (K[k] + K[k + n]);
    }
    const prod = (n - 3) * n + (n - 3);
    return { n, N, K, tx, ty, visc: o.visc, inj: 2 * n + 2, prod, Q: N / 1000,          // one pore volume (phi = 0.2) every 200 time units
             // productivity index chosen so that the initial oil rate at the
             // producer (p - p_bhp = 1, lambda_o = 1/visc) balances injection
             pbhp: 0, PI: (N / 1000) * o.visc,
             p: new Float64Array(N).fill(1), S: new Float64Array(N).fill(swc),
             t: 0, dt: o.dt0 || 0.5, chop: 0, clean: 0, steps: 0, newton: 0, krylov: 0 };
  }
  function assemble(M, p, S, pn, Sn, dt, wantJ) {
    const { n, N, tx, ty } = M;
    const R = new Float64Array(2 * N);
    const J = wantJ ? { D: new Float64Array(4 * N), W: new Float64Array(4 * N), E: new Float64Array(4 * N),
                        Sd: new Float64Array(4 * N), Nn: new Float64Array(4 * N) } : null;
    for (let k = 0; k < N; k++) {
      const ph = phi0 * (1 + cr * (p[k] - pref)), phn = phi0 * (1 + cr * (pn[k] - pref));
      R[2 * k] = (ph * S[k] - phn * Sn[k]) / dt;
      R[2 * k + 1] = (ph * (1 - S[k]) - phn * (1 - Sn[k])) / dt;
      if (J) {
        const o = 4 * k;
        J.D[o] += phi0 * cr * S[k] / dt;           J.D[o + 1] += ph / dt;
        J.D[o + 2] += phi0 * cr * (1 - S[k]) / dt; J.D[o + 3] += -ph / dt;
      }
    }
    const mb = new Array(N);
    for (let k = 0; k < N; k++) mb[k] = mob(S[k], M.visc);
    function face(a, b, T, offAB, offBA) {
      const dp = p[a] - p[b], up = dp >= 0 ? a : b, m = mb[up];
      const Fw = T * m[0] * dp, Fo = T * m[1] * dp;
      R[2 * a] += Fw; R[2 * a + 1] += Fo; R[2 * b] -= Fw; R[2 * b + 1] -= Fo;
      if (!J) return;
      const oa = 4 * a, ob = 4 * b;
      J.D[oa] += T * m[0]; J.D[oa + 2] += T * m[1];
      offAB[oa] -= T * m[0]; offAB[oa + 2] -= T * m[1];
      J.D[ob] += T * m[0]; J.D[ob + 2] += T * m[1];
      offBA[ob] -= T * m[0]; offBA[ob + 2] -= T * m[1];
      const gw = T * m[2] * dp, go = T * m[3] * dp;
      if (up === a) { J.D[oa + 1] += gw; J.D[oa + 3] += go; offBA[ob + 1] -= gw; offBA[ob + 3] -= go; }
      else { offAB[oa + 1] += gw; offAB[oa + 3] += go; J.D[ob + 1] -= gw; J.D[ob + 3] -= go; }
    }
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const k = j * n + i;
      if (i < n - 1) face(k, k + 1, tx[k], J ? J.E : null, J ? J.W : null);
      if (j < n - 1) face(k, k + n, ty[k], J ? J.Nn : null, J ? J.Sd : null);
    }
    R[2 * M.inj] -= M.Q;                       // water injected at a fixed rate
    const k = M.prod, m = mb[k], dpw = p[k] - M.pbhp;   // producer at fixed BHP
    R[2 * k] += M.PI * m[0] * dpw; R[2 * k + 1] += M.PI * m[1] * dpw;
    if (J) {
      const o = 4 * k;
      J.D[o] += M.PI * m[0]; J.D[o + 2] += M.PI * m[1];
      J.D[o + 1] += M.PI * m[2] * dpw; J.D[o + 3] += M.PI * m[3] * dpw;
    }
    return { R, J };
  }
  function jmul(M, J, x, y) {
    const { n } = M;
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const k = j * n + i, o = 4 * k;
      let a = J.D[o] * x[2 * k] + J.D[o + 1] * x[2 * k + 1];
      let b = J.D[o + 2] * x[2 * k] + J.D[o + 3] * x[2 * k + 1];
      const add = (B, c) => { a += B[o] * x[2 * c] + B[o + 1] * x[2 * c + 1]; b += B[o + 2] * x[2 * c] + B[o + 3] * x[2 * c + 1]; };
      if (i > 0) add(J.W, k - 1); if (i < n - 1) add(J.E, k + 1);
      if (j > 0) add(J.Sd, k - n); if (j < n - 1) add(J.Nn, k + n);
      y[2 * k] = a; y[2 * k + 1] = b;
    }
  }
  // ---- multigrid on a 5-point (possibly non-symmetric) pressure operator
  function coarsen(A) {
    const n = A.n, m = n / 2, Nc = m * m;
    const C = { n: m, N: Nc, d: new Float64Array(Nc), w: new Float64Array(Nc), e: new Float64Array(Nc),
                s: new Float64Array(Nc), nn: new Float64Array(Nc) };
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const k = j * n + i, I = i >> 1, Jc = j >> 1, K = Jc * m + I;
      C.d[K] += A.d[k];
      const put = (c, ii, jj) => {
        const I2 = ii >> 1, J2 = jj >> 1;
        if (I2 === I && J2 === Jc) C.d[K] += c;
        else if (I2 === I - 1) C.w[K] += c; else if (I2 === I + 1) C.e[K] += c;
        else if (J2 === Jc - 1) C.s[K] += c; else C.nn[K] += c;
      };
      if (i > 0) put(A.w[k], i - 1, j); if (i < n - 1) put(A.e[k], i + 1, j);
      if (j > 0) put(A.s[k], i, j - 1); if (j < n - 1) put(A.nn[k], i, j + 1);
    }
    return C;
  }
  function amul(A, x, y, rd) {
    const n = A.n;
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const k = j * n + i;
      let v = rd(A.d[k] * x[k]);
      if (i > 0) v = rd(v + rd(A.w[k] * x[k - 1])); if (i < n - 1) v = rd(v + rd(A.e[k] * x[k + 1]));
      if (j > 0) v = rd(v + rd(A.s[k] * x[k - n])); if (j < n - 1) v = rd(v + rd(A.nn[k] * x[k + n]));
      y[k] = v;
    }
  }
  function mgSetup(Ap, rd) {
    const lev = [Ap];
    while (lev[lev.length - 1].n > 4 && lev[lev.length - 1].n % 2 === 0) lev.push(coarsen(lev[lev.length - 1]));
    const L = lev.map(a => ({ n: a.n, N: a.N, d: a.d.map(rd), w: a.w.map(rd), e: a.e.map(rd), s: a.s.map(rd),
                              nn: a.nn.map(rd), dinv: a.d.map(v => rd(0.7 / v)) }));
    const c = lev[lev.length - 1], Nc = c.N, LU = [];
    for (let a = 0; a < Nc; a++) LU.push(new Float64Array(Nc));
    for (let k = 0; k < Nc; k++) {
      const e = new Float64Array(Nc); e[k] = 1; const y = new Float64Array(Nc);
      amul(c, e, y, id); for (let a = 0; a < Nc; a++) LU[a][k] = y[a];
    }
    for (let k = 0; k < Nc; k++) for (let a = k + 1; a < Nc; a++) {
      const f = LU[a][k] / LU[k][k]; LU[a][k] = f; for (let b = k + 1; b < Nc; b++) LU[a][b] -= f * LU[k][b];
    }
    return { L, LU, rd };
  }
  function vcycle(H, l, b) {
    const A = H.L[l], N = A.N, rd = H.rd, x = new Float64Array(N);
    if (l === H.L.length - 1) {
      const y = Float64Array.from(b);
      for (let a = 0; a < N; a++) for (let k = 0; k < a; k++) y[a] -= H.LU[a][k] * y[k];
      for (let a = N - 1; a >= 0; a--) { for (let k = a + 1; k < N; k++) y[a] -= H.LU[a][k] * y[k]; y[a] /= H.LU[a][a]; }
      return y.map(rd);
    }
    const Ax = new Float64Array(N);
    const smooth = () => { amul(A, x, Ax, rd); for (let k = 0; k < N; k++) x[k] = rd(x[k] + rd(A.dinv[k] * rd(b[k] - Ax[k]))); };
    smooth(); smooth();
    amul(A, x, Ax, rd);
    const n = A.n, m = n / 2, rc = new Float64Array(m * m);
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const K = (j >> 1) * m + (i >> 1); rc[K] = rd(rc[K] + rd(b[j * n + i] - Ax[j * n + i]));
    }
    const ec = vcycle(H, l + 1, rc);
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) x[j * n + i] = rd(x[j * n + i] + ec[(j >> 1) * m + (i >> 1)]);
    smooth(); smooth();
    return x;
  }
  function cprSetup(M, J, prec) {
    const { n, N } = M, rd = RD[prec];
    const wt = new Float64Array(2 * N);
    for (let k = 0; k < N; k++) {
      const o = 4 * k, a = -J.D[o + 3], b = J.D[o + 1], s = Math.abs(a) + Math.abs(b) || 1;
      wt[2 * k] = a / s; wt[2 * k + 1] = b / s;           // w^T D[:, S] = 0 (quasi-IMPES)
    }
    const Ap = { n, N, d: new Float64Array(N), w: new Float64Array(N), e: new Float64Array(N),
                 s: new Float64Array(N), nn: new Float64Array(N) };
    const pcol = (B, k) => wt[2 * k] * B[4 * k] + wt[2 * k + 1] * B[4 * k + 2];
    for (let k = 0; k < N; k++) {
      Ap.d[k] = pcol(J.D, k); Ap.w[k] = pcol(J.W, k); Ap.e[k] = pcol(J.E, k);
      Ap.s[k] = pcol(J.Sd, k); Ap.nn[k] = pcol(J.Nn, k);
    }
    const sc = Ap.d.map(v => 1 / v);                       // row scaling, needed for FP16
    for (let k = 0; k < N; k++) { Ap.d[k] = 1; Ap.w[k] *= sc[k]; Ap.e[k] *= sc[k]; Ap.s[k] *= sc[k]; Ap.nn[k] *= sc[k]; }
    const H = mgSetup(Ap, rd);
    const Dinv = new Float64Array(4 * N);
    for (let k = 0; k < N; k++) {
      const o = 4 * k, a = J.D[o], b = J.D[o + 1], c = J.D[o + 2], d = J.D[o + 3], det = a * d - b * c;
      Dinv[o] = rd(d / det); Dinv[o + 1] = rd(-b / det); Dinv[o + 2] = rd(-c / det); Dinv[o + 3] = rd(a / det);
    }
    return { M, wt, sc, H, Dinv, rd, J };
  }
  function cprApply(C, r, tm) {
    const M = C.M, N = M.N, rd = C.rd;
    const t0 = now();
    const rp = new Float64Array(N);
    for (let k = 0; k < N; k++) rp[k] = rd(C.sc[k] * (C.wt[2 * k] * r[2 * k] + C.wt[2 * k + 1] * r[2 * k + 1]));
    const xp = vcycle(C.H, 0, rp);
    const z = new Float64Array(2 * N);
    for (let k = 0; k < N; k++) z[2 * k] = xp[k];
    const t1 = now(); tm[1] += t1 - t0;
    const Jz = new Float64Array(2 * N); jmul(M, C.J, z, Jz);
    for (let k = 0; k < N; k++) {
      const o = 4 * k, a = rd(r[2 * k] - Jz[2 * k]), b = rd(r[2 * k + 1] - Jz[2 * k + 1]);
      z[2 * k] += rd(rd(C.Dinv[o] * a) + rd(C.Dinv[o + 1] * b));
      z[2 * k + 1] += rd(rd(C.Dinv[o + 2] * a) + rd(C.Dinv[o + 3] * b));
    }
    tm[2] += now() - t1;
    return z;
  }
  function fgmres(M, J, rhs, C, tol, maxit, restart, tm) {
    const N2 = rhs.length, x = new Float64Array(N2), hist = [];
    let bn = 0; for (let k = 0; k < N2; k++) bn += rhs[k] * rhs[k]; bn = Math.sqrt(bn) || 1;
    let it = 0;
    const t = new Float64Array(N2);
    while (it < maxit) {
      jmul(M, J, x, t);
      const r = rhs.map((v, k) => v - t[k]); let beta = 0; for (const v of r) beta += v * v; beta = Math.sqrt(beta);
      if (beta / bn < tol) break;
      const V = [r.map(v => v / beta)], Z = [], Hh = [], cs = [], sn = [], g = [beta];
      let done = false;
      for (let j = 0; j < restart && it < maxit; j++) {
        const z = cprApply(C, V[j], tm); Z.push(z);
        const t0 = now();
        const w = new Float64Array(N2); jmul(M, J, z, w);
        const h = new Float64Array(j + 2);
        for (let i = 0; i <= j; i++) { let s = 0; for (let k = 0; k < N2; k++) s += w[k] * V[i][k]; h[i] = s; for (let k = 0; k < N2; k++) w[k] -= s * V[i][k]; }
        let wn = 0; for (const v of w) wn += v * v; h[j + 1] = Math.sqrt(wn);
        V.push(w.map(v => v / (h[j + 1] || 1)));
        for (let i = 0; i < j; i++) { const q = cs[i] * h[i] + sn[i] * h[i + 1]; h[i + 1] = -sn[i] * h[i] + cs[i] * h[i + 1]; h[i] = q; }
        const den = Math.hypot(h[j], h[j + 1]) || 1; cs.push(h[j] / den); sn.push(h[j + 1] / den);
        h[j] = den; h[j + 1] = 0; g.push(-sn[j] * g[j]); g[j] = cs[j] * g[j]; Hh.push(h);
        it++;
        tm[3] += now() - t0;
        const rel = Math.abs(g[j + 1]) / bn;
        hist.push(rel);
        if (rel < tol || j === restart - 1 || it >= maxit) {
          const y = new Float64Array(j + 1);
          for (let a = j; a >= 0; a--) { let s = g[a]; for (let c = a + 1; c <= j; c++) s -= Hh[c][a] * y[c]; y[a] = s / Hh[a][a]; }
          for (let c = 0; c <= j; c++) for (let k = 0; k < N2; k++) x[k] += y[c] * Z[c][k];
          if (rel < tol) done = true;
          break;
        }
      }
      if (done) break;
    }
    return { x, it, hist };
  }
  // Newton convergence follows the simulator in the paper, which uses OPM
  // Flow's criteria: per phase, CNV = max_k dt |R_k| / PV_k below 1e-2 and
  // MB = dt |sum_k R_k| / sum_k PV_k below 1e-6, with at least one Newton
  // update; up to 20 Newton iterations; saturation updates limited to 0.2
  // per iteration; linear tolerance 1e-6 with FGMRES(30), 100 iterations.
  const TOL = { cnv: 1e-2, mb: 1e-6, newtonMax: 20, dsMax: 0.2, lin: 1e-6, linMax: 100,
                maxChop: 6, dtGrow: 2 };
  // one time step from (M.p, M.S) over dt; returns the new state and stats or null
  function timeStep(M, dt, prec, restart) {
    const p = Float64Array.from(M.p), S = Float64Array.from(M.S);
    const tm = [0, 0, 0, 0], newtonHist = [], krylovHists = [], mbHist = [];
    let kry = 0;
    for (let it = 0; it <= TOL.newtonMax; it++) {
      const t0 = now();
      const { R, J } = assemble(M, p, S, M.p, M.S, dt, true);
      tm[0] += now() - t0;
      let cnv = 0, mb = 0, pvs = 0;
      const sum = [0, 0];
      for (let k = 0; k < M.N; k++) {
        const pv = phi0 * (1 + cr * (p[k] - pref)); pvs += pv;
        for (let a = 0; a < 2; a++) { const r = R[2 * k + a]; sum[a] += r; cnv = Math.max(cnv, dt * Math.abs(r) / pv); }
      }
      for (let a = 0; a < 2; a++) mb = Math.max(mb, dt * Math.abs(sum[a]) / pvs);
      if (!isFinite(cnv)) return null;
      newtonHist.push(cnv); mbHist.push(mb);
      if (it > 0 && cnv < TOL.cnv && mb < TOL.mb)
        return { p, S, newton: it, krylov: kry, newtonHist, krylovHists, tm, dt, cnv, mb };
      if (it === TOL.newtonMax) break;
      const C = cprSetup(M, J, prec);
      const sol = fgmres(M, J, R.map(v => -v), C, TOL.lin, TOL.linMax, restart || 30, tm);
      kry += sol.it; krylovHists.push(sol.hist);
      for (let k = 0; k < M.N; k++) {
        p[k] += sol.x[2 * k];
        const ds = Math.max(-TOL.dsMax, Math.min(TOL.dsMax, sol.x[2 * k + 1]));
        let sn = S[k] + ds;
        // Appleyard chop: a cell crossing a relative-permeability endpoint
        // stops at it for this iteration
        if (S[k] > swc && sn < swc) sn = swc; else if (S[k] < 1 - sor && sn > 1 - sor) sn = 1 - sor;
        S[k] = Math.min(1, Math.max(0, sn));                // immobile water may compress below swc
      }
    }
    M.lastFail = { dt, newtonHist, mbHist, krylov: krylovHists.map(h => h.length) };
    return null;
  }
  // advance one report step of length dtRep by dyadic chopping, as in the
  // simulator: the step is covered by 2^c equal sub-steps, c starting from the
  // remembered level (and at least what dtmax requires); if any sub-step
  // fails the report step restarts from its beginning at the next level, up
  // to maxChop extra halvings. A level that needed chopping is kept; after
  // dtGrow consecutive report steps that converged at the first attempt one
  // halving is released.
  function step(M, o) {
    const dtRep = o.tstop !== undefined ? o.tstop - M.t : (o.dtrep || o.dtmax || 10);
    if (!(dtRep > 1e-9)) return null;
    const dtmax = o.dtmax || Infinity, prec = o.prec || "fp32", restart = o.restart || 30;
    let c0 = 0; while (c0 < 24 && dtRep / (1 << c0) > dtmax * (1 + 1e-9)) c0++;
    // the first step starts from the initial time step, as simulators do
    if (!M.steps && o.dt0) while (c0 < 24 && dtRep / (1 << c0) > o.dt0 * (1 + 1e-9)) c0++;
    c0 = Math.max(c0, M.chop || 0);
    const p0 = M.p, S0 = M.S, t0 = M.t;
    const agg = { newton: 0, krylov: 0, tm: [0, 0, 0, 0], subs: 0, chop: 0, lostNewton: 0 };
    for (let c = c0; c <= c0 + TOL.maxChop; c++) {
      const nsub = 1 << c, dt = dtRep / nsub;
      let ok = true, last = null, nw = 0, kr = 0;
      for (let s = 0; s < nsub; s++) {
        const r = timeStep(M, dt, prec, restart);
        if (!r) { ok = false; break; }
        M.p = r.p; M.S = r.S; last = r; nw += r.newton; kr += r.krylov;
        for (let i = 0; i < 4; i++) agg.tm[i] += r.tm[i];
      }
      if (!ok) { M.p = p0; M.S = S0; agg.lostNewton += nw; continue; }
      if (c === c0) {
        M.clean = (M.clean || 0) + 1;
        if (M.clean >= TOL.dtGrow) { M.chop = Math.max(0, c - 1); M.clean = 0; } else M.chop = c;
      } else { M.chop = c; M.clean = 0; }
      M.t = o.tstop !== undefined ? o.tstop : t0 + dtRep;
      M.steps += nsub; M.newton += nw; M.krylov += kr;
      M.dt = dtRep / (1 << M.chop);
      return Object.assign({}, last, { newton: nw, krylov: kr, tm: agg.tm, subs: nsub, chop: c, dt });
    }
    M.t = t0;
    return null;
  }
  function wellRates(M) {
    const k = M.prod, m = mob(M.S[k], M.visc), dp = M.p[k] - M.pbhp;
    const qw = M.PI * m[0] * dp, qo = M.PI * m[1] * dp;
    return { qw, qo, wcut: qw + qo > 0 ? qw / (qw + qo) : 0, pinj: M.p[M.inj] };
  }
  window.NVRS = { model, step, timeStep, mob: (s, v) => mob(s, v), wellRates, randomLogK, swc, sor };
})();
