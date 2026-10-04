/* geology.js -- permeability fields for the inverse-problem demo on a 16 x 16
   grid: sinuous sand channels in shale (a non-Gaussian, "exotic" geology) and
   Gaussian fields in a truncated DCT basis (6 x 6 of 256 coefficients, about
   15 %). Every field is reproducible from its seed.

   window.GEO = { n, KD, rng, gauss, channel(seed), dctField(c), project(f),
                  dctPrior(seed), SAND, SHALE }                               */
(function () {
  "use strict";
  const n = 16, N = n * n, KD = 6, SAND = 1.8, SHALE = -1.8;
  function rng(seed) {
    let a = seed >>> 0;
    return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  }
  function gauss(r) { let u = 0; while (!u) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); }
  // one to three sinuous channels with random direction, amplitude, wavelength and width
  function channel(seed) {
    const r = rng(seed * 2654435761 + 17), f = new Float64Array(N).fill(0);
    const nc = 1 + Math.floor(r() * 3), base = r() * Math.PI;
    for (let c = 0; c < nc; c++) {
      const th = base + (r() - 0.5) * 0.9, ct = Math.cos(th), st = Math.sin(th);
      const off = (r() - 0.5) * n * 0.8, amp = 0.8 + r() * 2.6, lam = 7 + r() * 12, ph = r() * 6.283;
      const w = 0.9 + r() * 1.1;
      for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
        const x = i + 0.5 - n / 2, y = j + 0.5 - n / 2;
        const u = x * ct + y * st, v = -x * st + y * ct;
        const d = Math.abs(v - off - amp * Math.sin(2 * Math.PI * u / lam + ph));
        const s = 1 / (1 + Math.exp((d - w) * 3));
        f[j * n + i] = Math.max(f[j * n + i], s);
      }
    }
    return f.map(s => SHALE + (SAND - SHALE) * s);
  }
  // orthonormal 2-D DCT-II basis, low-frequency KD x KD block
  const BASIS = [];
  for (let ky = 0; ky < KD; ky++) for (let kx = 0; kx < KD; kx++) {
    const b = new Float64Array(N), ax = kx ? Math.sqrt(2 / n) : Math.sqrt(1 / n), ay = ky ? Math.sqrt(2 / n) : Math.sqrt(1 / n);
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++)
      b[j * n + i] = ax * ay * Math.cos(Math.PI * kx * (i + 0.5) / n) * Math.cos(Math.PI * ky * (j + 0.5) / n);
    BASIS.push({ b, kx, ky });
  }
  function dctField(c) {
    const f = new Float64Array(N);
    for (let m = 0; m < BASIS.length; m++) { const b = BASIS[m].b, v = c[m]; if (v) for (let k = 0; k < N; k++) f[k] += v * b[k]; }
    return f;
  }
  function project(f) { return BASIS.map(({ b }) => { let s = 0; for (let k = 0; k < N; k++) s += f[k] * b[k]; return s; }); }
  // Gaussian prior on the DCT coefficients, variance decaying with frequency
  const PSD = BASIS.map(({ kx, ky }) => (kx + ky === 0 ? 0 : 1) * 9 / (1 + 0.35 * (kx * kx + ky * ky)));
  const psdNorm = Math.sqrt(PSD.reduce((a, b) => a + b, 0) / N);
  const PSTD = PSD.map(v => Math.sqrt(v) / psdNorm * 1.6);
  function dctPrior(seed) { const r = rng(seed * 40503 + 11); return PSTD.map(s => s * gauss(r)); }
  window.GEO = { n, N, KD, rng, gauss, channel, dctField, project, dctPrior, PSTD, SAND, SHALE };
})();
