/* scaling.js -- the Million family on one, two and four GPUs, GB200 and VR72,
   AmgX and BoomerAMG stage 1 (wall clock in seconds, from the paper). */
(function () {
  "use strict";
  const root = document.getElementById("demo-scaling");
  if (!root) return;
  const cv = root.querySelector("canvas"), c = cv.getContext("2d");
  const DATA = {
    GB200: { "1 M": { AmgX: [21.17, 18.96, 18.26], BoomerAMG: [26.23, 33.85, 43.27] },
             "2 M": { AmgX: [39.38, 31.05, 30.84], BoomerAMG: [49.94, 52.84, 65.32] },
             "4 M": { AmgX: [77.67, 58.01, 51.78], BoomerAMG: [81.87, 80.00, 86.67] },
             "5 M": { AmgX: [95.61, 71.14, 61.71], BoomerAMG: [102.32, 96.16, 96.64] } },
    VR72:  { "1 M": { AmgX: [14.87, 14.02, 16.17], BoomerAMG: [19.94, 28.88, 49.61] },
             "2 M": { AmgX: [27.17, 21.24, 23.62], BoomerAMG: [38.23, 43.83, 72.11] },
             "4 M": { AmgX: [52.33, 36.58, 41.14], BoomerAMG: [60.97, 60.96, 85.21] },
             "5 M": { AmgX: [63.77, 43.91, 42.26], BoomerAMG: [76.65, 77.49, 93.30] } },
  };
  const COL = { AmgX: ["#00e5a0", "#00b37d", "#007a56"], BoomerAMG: ["#7b61ff", "#5b45d6", "#3e2ea3"] };
  let mach = "GB200", mode = "wall", bars = [];
  function draw() {
    const W = cv.width, H = cv.height, l = 56, r = 14, t = 36, b = 44;
    c.fillStyle = "#07070a"; c.fillRect(0, 0, W, H);
    const D = DATA[mach], sizes = Object.keys(D);
    const val = (be, s, g) => mode === "wall" ? D[s][be][g] : D[s][be][0] / D[s][be][g];
    let ymax = 0;
    for (const s of sizes) for (const be of ["AmgX", "BoomerAMG"]) for (let g = 0; g < 3; g++) ymax = Math.max(ymax, val(be, s, g));
    ymax = mode === "wall" ? Math.ceil(ymax / 20) * 20 : Math.max(2, Math.ceil(ymax * 2) / 2);
    const Y = v => t + (H - t - b) * (1 - v / ymax);
    c.font = "11px DM Mono, monospace"; c.strokeStyle = "rgba(255,255,255,.1)"; c.fillStyle = "#9a9997";
    for (let i = 0; i <= 5; i++) {
      const v = ymax * i / 5; c.beginPath(); c.moveTo(l, Y(v)); c.lineTo(W - r, Y(v)); c.stroke();
      c.fillText(mode === "wall" ? v.toFixed(0) + " s" : v.toFixed(1) + "x", 6, Y(v) + 4);
    }
    if (mode !== "wall") { c.setLineDash([4, 4]); c.strokeStyle = "rgba(255,255,255,.4)"; c.beginPath(); c.moveTo(l, Y(1)); c.lineTo(W - r, Y(1)); c.stroke(); c.setLineDash([]); }
    const gw = (W - l - r) / sizes.length, bw = gw / 8;
    bars = [];
    sizes.forEach((s, si) => {
      const x0 = l + si * gw + bw * 0.6;
      ["AmgX", "BoomerAMG"].forEach((be, bi) => {
        for (let g = 0; g < 3; g++) {
          const x = x0 + (bi * 3.4 + g) * bw, v = val(be, s, g);
          c.fillStyle = COL[be][g]; c.fillRect(x, Y(v), bw * 0.9, Y(0) - Y(v));
          bars.push({ x, y: Y(v), w: bw * 0.9, h: Y(0) - Y(v), txt: be + ", " + s + " cells, " + [1, 2, 4][g] + " GPU" + (g ? "s" : "") + ": " +
            (mode === "wall" ? D[s][be][g].toFixed(2) + " s" : (D[s][be][0] / D[s][be][g]).toFixed(2) + "x against its own 1-GPU time") });
        }
      });
      c.fillStyle = "#e8e6e3"; c.fillText(s + " cells", l + si * gw + gw / 2 - 26, H - 22);
    });
    c.fillStyle = "#e8e6e3";
    c.fillText(mode === "wall" ? "Wall clock, " + mach + " (lower is better)" : "Speed-up over one GPU, " + mach + " (higher is better)", l, 18);
    c.fillStyle = "#9a9997"; c.fillText("bars: 1, 2, 4 GPUs", W - r - 140, 18);
  }
  cv.addEventListener("mousemove", e => {
    const rc = cv.getBoundingClientRect(), x = (e.clientX - rc.left) * cv.width / rc.width, y = (e.clientY - rc.top) * cv.height / rc.height;
    const hit = bars.find(bb => x >= bb.x && x <= bb.x + bb.w && y >= bb.y && y <= bb.y + bb.h);
    root.querySelector(".readout").textContent = hit ? hit.txt : "Hover a bar for its value.";
  });
  root.querySelectorAll("button[data-m]").forEach(bt => bt.addEventListener("click", () => {
    mach = bt.getAttribute("data-m"); root.querySelectorAll("button[data-m]").forEach(o => o.classList.toggle("ghost", o !== bt)); draw();
  }));
  root.querySelectorAll("button[data-v]").forEach(bt => bt.addEventListener("click", () => {
    mode = bt.getAttribute("data-v"); root.querySelectorAll("button[data-v]").forEach(o => o.classList.toggle("ghost", o !== bt)); draw();
  }));
  draw();
})();
