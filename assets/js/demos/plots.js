/* plots.js -- shared plots for the history-matching demos.
   PLOTS.alpha(ctx, W, H, series, title): alpha against iteration (log axis,
   left) and the running sum of 1/alpha (right axis, 0 to 1), one line per
   series; series = [{ alphas: [..], label, col }].                        */
(function () {
  "use strict";
  function alpha(c, W, H, series, title) {
    const l = 52, r = 46, t = 28, b = 30;
    c.fillStyle = "#07070a"; c.fillRect(0, 0, W, H);
    c.font = "11px DM Mono, monospace"; c.fillStyle = "#e8e6e3"; c.fillText(title || "alpha against iteration", 10, 16);
    c.strokeStyle = "rgba(255,255,255,.14)"; c.strokeRect(l, t, W - l - r, H - t - b);
    const all = series.flatMap(s => s.alphas);
    const nIt = Math.max(4, ...series.map(s => s.alphas.length));
    const lo = all.length ? Math.max(0.5, Math.min(...all) / 1.5) : 1, hi = all.length ? Math.max(...all) * 1.5 : 100;
    const X = i => l + (W - l - r) * (i - 1) / Math.max(nIt - 1, 1);
    const Y = a => t + (H - t - b) * (Math.log10(hi) - Math.log10(a)) / (Math.log10(hi) - Math.log10(lo));
    const Ys = v => t + (H - t - b) * (1 - v);
    // axes: alpha (log) on the left, sum 1/alpha on the right, iterations below
    c.fillStyle = "#9a9997";
    for (let e = Math.ceil(Math.log10(lo)); e <= Math.floor(Math.log10(hi)); e++) {
      const y = Y(Math.pow(10, e)); c.strokeStyle = "rgba(255,255,255,.07)"; c.beginPath(); c.moveTo(l, y); c.lineTo(W - r, y); c.stroke();
      c.fillText(String(Math.pow(10, e)), l - 8 - c.measureText(String(Math.pow(10, e))).width, y + 4);
    }
    for (const v of [0, 0.5, 1]) c.fillText(v.toFixed(1), W - r + 6, Ys(v) + 4);
    for (let i = 1; i <= nIt; i += Math.max(1, Math.round(nIt / 10))) c.fillText(String(i), X(i) - 3, H - b + 14);
    c.fillText("iteration", (l + W - r) / 2 - 24, H - 4);
    c.save(); c.translate(12, (t + H - b) / 2 + 16); c.rotate(-Math.PI / 2); c.fillText("alpha (log)", 0, 0); c.restore();
    c.save(); c.translate(W - 8, (t + H - b) / 2 - 30); c.rotate(Math.PI / 2); c.fillStyle = "#00e5a0"; c.fillText("sum 1/alpha", 0, 0); c.restore();
    c.setLineDash([4, 4]); c.strokeStyle = "rgba(0,229,160,.45)"; c.beginPath(); c.moveTo(l, Ys(1)); c.lineTo(W - r, Ys(1)); c.stroke(); c.setLineDash([]);
    series.forEach((s, si) => {
      if (!s.alphas.length) return;
      c.strokeStyle = s.col; c.lineWidth = 2; c.beginPath();
      s.alphas.forEach((a, i) => i ? c.lineTo(X(i + 1), Y(a)) : c.moveTo(X(i + 1), Y(a))); c.stroke();
      s.alphas.forEach((a, i) => { c.beginPath(); c.arc(X(i + 1), Y(a), 3.2, 0, 7); c.fillStyle = s.col; c.fill(); });
      let sum = 0; c.strokeStyle = "#00e5a0"; c.setLineDash(si ? [3, 3] : []); c.beginPath();
      s.alphas.forEach((a, i) => { sum += 1 / a; const y = Ys(Math.min(1, sum)); i ? c.lineTo(X(i + 1), y) : c.moveTo(X(i + 1), y); }); c.stroke(); c.setLineDash([]);
      c.lineWidth = 1; c.fillStyle = s.col; c.fillText(s.label, l + 8, t + 14 + 14 * si);
    });
  }
  window.PLOTS = { alpha };
})();
