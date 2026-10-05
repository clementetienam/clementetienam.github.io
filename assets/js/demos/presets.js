/* presets.js -- preset buttons for the demos: each sets several controls of
   its demo at once and shows a note on what to watch. */
(function () {
  "use strict";
  document.querySelectorAll(".presets").forEach(box => {
    const demo = box.closest(".demo"), note = box.querySelector(".preset-note");
    box.querySelectorAll("button[data-preset]").forEach(b => b.addEventListener("click", () => {
      const p = JSON.parse(b.getAttribute("data-preset"));
      for (const [k, v] of Object.entries(p)) {
        const el = demo.querySelector('[name="' + k + '"]');
        if (!el) continue;
        el.value = String(v);
        el.dispatchEvent(new Event("input", { bubbles: true }));
      }
      box.querySelectorAll("button").forEach(x => x.classList.toggle("ghost", x !== b));
      if (note) note.textContent = b.getAttribute("data-note");
    }));
  });
})();
