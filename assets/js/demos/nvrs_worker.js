/* nvrs_worker.js -- runs the miniature simulator (nvrs_engine.js) off the
   page's main thread for the simulator app, so large grids do not freeze the
   page. Messages: build {opts}, step {o}, compare {restart}; every reply
   carries the request id and a snapshot of the state and the well rates.  */
"use strict";
self.window = self;
importScripts("nvrs_engine.js" + self.location.search);
const E = self.NVRS;
let M = null;

function snapshot(res) {
  const r = M.prods.map(k => E.wellRates(M, k));
  return { t: M.t, dt: M.dt, steps: M.steps, newton: M.newton, krylov: M.krylov, p: M.p, S: M.S,
           qo: r.map(x => x.qo / M.Q), qw: r.map(x => x.qw / M.Q), wct: r.map(x => x.wcut), bhp: M.p[M.inj],
           res: res ? { newtonHist: res.newtonHist, krylovHists: res.krylovHists, tm: res.tm, subs: res.subs } : null };
}
function compare(restart) {                 // the next time step solved with the preconditioner in FP64, FP32 and FP16
  const rows = []; let ref = null;
  for (const prec of ["fp64", "fp32", "fp16"]) {
    const t0 = performance.now();
    let res = null, dt = M.dt;
    while (!res && dt > 1e-4) { res = E.timeStep(M, dt, prec, restart); if (!res) dt *= 0.5; }
    const ms = performance.now() - t0;
    if (!res) { rows.push([prec.toUpperCase(), "-", "-", "-", "failed"]); continue; }
    if (!ref) ref = res;
    let d = 0; for (let k = 0; k < M.N; k++) d = Math.max(d, Math.abs(res.S[k] - ref.S[k]));
    rows.push([prec.toUpperCase(), res.newton, res.krylov, ms.toFixed(0) + " ms", prec === "fp64" ? "reference" : d.toExponential(1)]);
  }
  return rows;
}
self.onmessage = e => {
  const d = e.data;
  try {
    if (d.cmd === "build") { M = E.model(d.opts); self.postMessage({ id: d.id, ok: true, snap: snapshot(null) }); }
    else if (d.cmd === "step") { const res = E.step(M, d.o); self.postMessage({ id: d.id, ok: !!res, snap: snapshot(res) }); }
    else if (d.cmd === "compare") self.postMessage({ id: d.id, ok: true, rows: compare(d.restart) });
  } catch (err) { self.postMessage({ id: d.id, ok: false, error: String(err) }); }
};
