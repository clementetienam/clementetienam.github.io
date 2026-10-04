/* lightbox.js -- click any [data-zoom] element to open its full-size image or
   movie in a viewer: wheel or pinch to zoom, drag to pan, double-click to
   toggle 2.5x, arrow keys or the side buttons to step through the group,
   Esc to close.  No dependencies. */
(function () {
  "use strict";
  const items = () => Array.from(document.querySelectorAll("[data-zoom]"));
  let group = [], index = 0, scale = 1, tx = 0, ty = 0;
  let drag = null, pinch = null;

  const box = document.createElement("div");
  box.className = "lb";
  box.setAttribute("role", "dialog");
  box.setAttribute("aria-modal", "true");
  box.innerHTML =
    '<div class="lb-stage"><img class="lb-img" alt=""></div>' +
    '<div class="lb-bar">' +
    '<span class="lb-cap"></span>' +
    '<span class="lb-tools">' +
    '<button type="button" data-a="out" aria-label="Zoom out">&minus;</button>' +
    '<button type="button" data-a="reset" aria-label="Fit to screen">fit</button>' +
    '<button type="button" data-a="in" aria-label="Zoom in">+</button>' +
    '<a class="lb-open" target="_blank" rel="noopener" aria-label="Open original">&#8599;</a>' +
    '<button type="button" data-a="close" aria-label="Close">&times;</button>' +
    "</span></div>" +
    '<button type="button" class="lb-nav lb-prev" data-a="prev" aria-label="Previous">&#8249;</button>' +
    '<button type="button" class="lb-nav lb-next" data-a="next" aria-label="Next">&#8250;</button>';
  const img = box.querySelector(".lb-img");
  const stage = box.querySelector(".lb-stage");
  const cap = box.querySelector(".lb-cap");
  const open = box.querySelector(".lb-open");

  function apply() {
    img.style.transform = "translate(" + tx + "px," + ty + "px) scale(" + scale + ")";
    box.classList.toggle("lb-zoomed", scale > 1.01);
  }
  function reset() { scale = 1; tx = 0; ty = 0; apply(); }
  function zoomAt(f, cx, cy) {
    const r = stage.getBoundingClientRect();
    const px = cx - r.left - r.width / 2, py = cy - r.top - r.height / 2;
    const ns = Math.min(12, Math.max(1, scale * f));
    const k = ns / scale;
    tx = px - (px - tx) * k; ty = py - (py - ty) * k; scale = ns;
    if (scale === 1) { tx = 0; ty = 0; }
    apply();
  }
  function show(i) {
    index = (i + group.length) % group.length;
    const el = group[index];
    const src = el.getAttribute("data-zoom");
    img.src = src;
    img.alt = el.getAttribute("data-caption") || "";
    cap.textContent = el.getAttribute("data-caption") || "";
    open.href = src;
    box.classList.toggle("lb-single", group.length < 2);
    reset();
  }
  function openAt(el) {
    const g = el.getAttribute("data-group");
    group = g ? items().filter(e => e.getAttribute("data-group") === g) : [el];
    if (!box.isConnected) document.body.appendChild(box);
    show(group.indexOf(el));
    box.classList.add("lb-on");
    document.documentElement.classList.add("lb-lock");
  }
  function close() {
    box.classList.remove("lb-on");
    document.documentElement.classList.remove("lb-lock");
    img.removeAttribute("src");
  }

  document.addEventListener("click", e => {
    const el = e.target.closest("[data-zoom]");
    if (el && !box.contains(el)) { e.preventDefault(); openAt(el); }
  });
  document.addEventListener("keydown", e => {
    if (!box.classList.contains("lb-on")) {
      const el = document.activeElement;
      if ((e.key === "Enter" || e.key === " ") && el && el.hasAttribute && el.hasAttribute("data-zoom")) {
        e.preventDefault(); openAt(el);
      }
      return;
    }
    if (e.key === "Escape") close();
    else if (e.key === "ArrowRight") show(index + 1);
    else if (e.key === "ArrowLeft") show(index - 1);
    else if (e.key === "+" || e.key === "=") zoomAt(1.4, innerWidth / 2, innerHeight / 2);
    else if (e.key === "-") zoomAt(1 / 1.4, innerWidth / 2, innerHeight / 2);
    else if (e.key === "0") reset();
  });
  box.addEventListener("click", e => {
    const a = e.target.closest("[data-a]");
    if (a) {
      const act = a.getAttribute("data-a");
      if (act === "close") close();
      else if (act === "next") show(index + 1);
      else if (act === "prev") show(index - 1);
      else if (act === "in") zoomAt(1.5, innerWidth / 2, innerHeight / 2);
      else if (act === "out") zoomAt(1 / 1.5, innerWidth / 2, innerHeight / 2);
      else if (act === "reset") reset();
      return;
    }
    if (e.target === stage && scale <= 1.01) close();
  });
  stage.addEventListener("wheel", e => {
    e.preventDefault();
    zoomAt(e.deltaY < 0 ? 1.18 : 1 / 1.18, e.clientX, e.clientY);
  }, { passive: false });
  img.addEventListener("dblclick", e => {
    if (scale > 1.01) reset(); else zoomAt(2.5, e.clientX, e.clientY);
  });
  img.addEventListener("dragstart", e => e.preventDefault());
  stage.addEventListener("pointerdown", e => {
    if (e.pointerType === "touch") return;
    drag = { x: e.clientX, y: e.clientY, tx: tx, ty: ty };
    stage.setPointerCapture(e.pointerId);
  });
  stage.addEventListener("pointermove", e => {
    if (!drag) return;
    tx = drag.tx + e.clientX - drag.x; ty = drag.ty + e.clientY - drag.y;
    if (scale <= 1.01) { tx = 0; ty = 0; }
    apply();
  });
  stage.addEventListener("pointerup", () => { drag = null; });
  stage.addEventListener("touchstart", e => {
    if (e.touches.length === 2) {
      const [a, b] = e.touches;
      pinch = { d: Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY), s: scale };
    } else if (e.touches.length === 1) {
      drag = { x: e.touches[0].clientX, y: e.touches[0].clientY, tx: tx, ty: ty };
    }
  }, { passive: true });
  stage.addEventListener("touchmove", e => {
    if (pinch && e.touches.length === 2) {
      e.preventDefault();
      const [a, b] = e.touches;
      const d = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
      const f = (pinch.s * d / pinch.d) / scale;
      zoomAt(f, (a.clientX + b.clientX) / 2, (a.clientY + b.clientY) / 2);
    } else if (drag && e.touches.length === 1 && scale > 1.01) {
      e.preventDefault();
      tx = drag.tx + e.touches[0].clientX - drag.x; ty = drag.ty + e.touches[0].clientY - drag.y;
      apply();
    }
  }, { passive: false });
  stage.addEventListener("touchend", e => {
    if (e.touches.length < 2) pinch = null;
    if (e.touches.length === 0) drag = null;
  });

  // Movies: load the animation only when it scrolls into view.
  const io = "IntersectionObserver" in window ? new IntersectionObserver(es => {
    es.forEach(en => {
      if (en.isIntersecting) {
        const m = en.target;
        if (m.dataset.anim && m.src.indexOf(m.dataset.anim) < 0) m.src = m.dataset.anim;
        io.unobserve(m);
      }
    });
  }, { rootMargin: "300px" }) : null;
  document.querySelectorAll("img[data-anim]").forEach(m => {
    if (io) io.observe(m); else m.src = m.dataset.anim;
  });
})();
