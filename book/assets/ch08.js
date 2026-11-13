/* Chapter 8: decompositions. Four interactive figures:
 *   #c8-tree   how Tensor methods expand into a handful of primitive UOps
 *   #c8-exp2   tinygrad's xexp2 step by step, in float32, with the IEEE-754 exponent trick animated
 *   #c8-sin    sine range reduction on a wheel, plus polynomial and reduction error plots
 *   #c8-idiv   magic-number division (magicgu from codegen/decomp/op.py) checked exhaustively
 * All arithmetic mirrors the generated kernels: float32 everywhere (Math.fround), and fused
 * multiply-add for the Horner steps, which is what ptxas emits (FFMA) for the captured CUDA C.
 */
(function () {
  const f = Math.fround;
  const fma = (a, b, c) => f(a * b + c); // a*b of two float32 values is exact in float64
  const NS = "http://www.w3.org/2000/svg";
  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;");
  const dv = new DataView(new ArrayBuffer(4));
  const bitsOf = (x) => { dv.setFloat32(0, x); return dv.getUint32(0); };
  const floatOf = (u) => { dv.setUint32(0, u >>> 0); return dv.getFloat32(0); };
  function ulp32(y) { // spacing of float32 numbers around y
    y = Math.abs(y); if (!isFinite(y)) return NaN; if (y < 1.1754943508222875e-38) return 1.401298464324817e-45;
    return Math.pow(2, Math.floor(Math.log2(y)) - 23);
  }
  const fmt = (x, p = 9) => (x === Infinity ? "+inf" : x === -Infinity ? "−inf" : Number.isNaN(x) ? "nan" : x === 0 ? "0" :
    Math.abs(x) >= 1e6 || Math.abs(x) < 1e-4 ? x.toExponential(p - 3).replace("e", "·10^").replace("+", "") : x.toPrecision(p).replace(/(\.\d*?)0+$/, "$1").replace(/\.$/, ""));

  // ---------- coefficients, copied from tinygrad/codegen/decomp/transcendental.py ----------
  const EXP2_C = [0.1535920892e-3, 0.1339262701e-2, 0.9618384764e-2, 0.5550347269e-1, 0.2402264476e+0, 0.6931471825e+0, 1.0]; // line 209
  const SIN_C = [2.6083159809786593541503e-06, -0.0001981069071916863322258, 0.00833307858556509017944336, -0.166666597127914428710938, 1.0]; // line 153
  const CW = [-3.1414794921875, -0.00011315941810607910156, -1.9841872589410058936e-09, -1.2154201256553420762e-10]; // lines 138-141
  const LOG2E_F = f(1 / Math.log(2)); // rendered as 1.4426950216293335f in the kernel
  const pow2if = (q) => floatOf((q + 127) << 23); // transcendental.py:27-30

  function xexp2(d) { // transcendental.py:193-217, float32 path
    d = f(d);
    const x = Number.isFinite(d) ? d : 0;
    const q = Math.trunc(f(x + (x < 0 ? -0.5 : 0.5))); // rintk: round half away from zero
    const s = f(x - q);
    let u = 0; const steps = [];
    for (const c of EXP2_C) { u = fma(u, s, f(c)); steps.push(u); }
    const a = q >> 1, b = q - a;
    let r = f(f(u * pow2if(a)) * pow2if(b));
    let clamp = "";
    if (d >= 128) { r = Infinity; clamp = "d ≥ 128 → +inf"; }
    if (d < -150) { r = 0; clamp = "d < −150 → 0"; }
    if (Number.isNaN(d)) { r = NaN; clamp = "nan → nan"; }
    return { d, q, s, u, steps, a, b, r, clamp };
  }
  function sinPoly(dd) { const d = f(dd), d2 = f(d * d); let p = 0; for (const c of SIN_C) p = fma(p, d2, f(c)); return f(d * p); }
  function cwReduce(ax) { // cody_waite_reduction, float32 branch (transcendental.py:115-147)
    const t = f(ax * f(0.318309886183790671537767526745028724));
    const q = Math.trunc(f(t + (t < 0 ? -0.5 : 0.5)));
    let d = ax; for (const c of CW) d = fma(q, f(c), d);
    return [d, q];
  }
  function tgSinSmall(x0) {
    const x = f(x0), sign = x === 0 ? 0 : x < 0 ? -1 : 1, ax = f(x * sign);
    const [d, q] = cwReduce(ax);
    return { q, d, poly: sinPoly(d), r: f(sinPoly(d) * (q & 1 ? -1 : 1) * sign) };
  }
  function naiveSin(x0) { // one-constant reduction, for comparison
    const x = f(x0), sign = x < 0 ? -1 : 1, ax = f(x * sign);
    const q = Math.round(f(ax * f(1 / Math.PI))); const d = f(ax - f(q * f(Math.PI)));
    return f(sinPoly(d) * (q & 1 ? -1 : 1) * sign);
  }

  // ---------- small SVG helpers ----------
  function svgEl(tag, attrs, parent) { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; }
  // minimal line/point plot. series: [{pts:[[x,y]], color, kind:'line'|'dots', label}]
  function plot(svg, { W = 520, H = 220, xDom, yDom, logX = false, series, xLabel, yLabel, vlines = [], hlines = [], marker }) {
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.innerHTML = "";
    const L = 58, R = 12, T = 12, B = 34, tx = logX ? Math.log10 : (v) => v;
    const X = (v) => L + (W - L - R) * (tx(v) - tx(xDom[0])) / (tx(xDom[1]) - tx(xDom[0]));
    const Y = (v) => T + (H - T - B) * (1 - (v - yDom[0]) / (yDom[1] - yDom[0]));
    svgEl("rect", { x: L, y: T, width: W - L - R, height: H - T - B, fill: "#0a0e14", stroke: "#2a3441" }, svg);
    const ticks = (lo, hi, n) => { const st = Math.pow(10, Math.floor(Math.log10((hi - lo) / n))); const m = [1, 2, 5, 10].find((k) => (hi - lo) / (k * st) <= n) * st; const out = []; for (let v = Math.ceil(lo / m) * m; v <= hi + 1e-12; v += m) out.push(+v.toPrecision(10)); return out; };
    const xt = logX ? Array.from({ length: Math.round(Math.log10(xDom[1]) - Math.log10(xDom[0])) + 1 }, (_, i) => Math.pow(10, Math.round(Math.log10(xDom[0])) + i)) : ticks(xDom[0], xDom[1], 6);
    xt.forEach((v) => { svgEl("line", { x1: X(v), x2: X(v), y1: T, y2: H - B, stroke: "#1c2430" }, svg); const t = svgEl("text", { x: X(v), y: H - B + 14, "text-anchor": "middle", class: "c8-ax" }, svg); t.textContent = logX ? (v >= 1000 ? "1e" + Math.round(Math.log10(v)) : String(v)) : String(v); });
    ticks(yDom[0], yDom[1], 4).forEach((v) => { svgEl("line", { x1: L, x2: W - R, y1: Y(v), y2: Y(v), stroke: v === 0 ? "#33445a" : "#1c2430" }, svg); const t = svgEl("text", { x: L - 6, y: Y(v) + 4, "text-anchor": "end", class: "c8-ax" }, svg); t.textContent = Math.abs(v) < 1e-3 && v !== 0 ? v.toExponential(0) : String(v); });
    vlines.forEach(([v, c, lab]) => { if (v < xDom[0] || v > xDom[1]) return; svgEl("line", { x1: X(v), x2: X(v), y1: T, y2: H - B, stroke: c, "stroke-dasharray": "4 3" }, svg); const t = svgEl("text", { x: X(v) + 4, y: H - B - 6, class: "c8-ax", fill: c }, svg); t.textContent = lab; t.style.fill = c; });
    hlines.forEach(([v, c]) => svgEl("line", { x1: L, x2: W - R, y1: Y(v), y2: Y(v), stroke: c, "stroke-dasharray": "2 3" }, svg));
    const clampY = (v) => Math.max(T, Math.min(H - B, Y(v)));
    series.forEach((s) => {
      if (s.kind === "dots") s.pts.forEach(([x, y]) => svgEl("circle", { cx: X(x), cy: clampY(y), r: 1.3, fill: s.color, opacity: 0.75 }, svg));
      else svgEl("path", { d: s.pts.map(([x, y], i) => (i ? "L" : "M") + X(x).toFixed(1) + " " + clampY(y).toFixed(1)).join(""), fill: "none", stroke: s.color, "stroke-width": 1.8 }, svg);
    });
    if (xLabel) { const t = svgEl("text", { x: (L + W - R) / 2, y: H - 4, "text-anchor": "middle", class: "c8-ax" }, svg); t.textContent = xLabel; }
    if (yLabel) { const t = svgEl("text", { x: 12, y: (T + H - B) / 2, "text-anchor": "middle", class: "c8-ax", transform: `rotate(-90 12 ${(T + H - B) / 2})` }, svg); t.textContent = yLabel; }
    let legX = L + 8; series.filter((s) => s.label).forEach((s) => { svgEl("rect", { x: legX - 3, y: T + 3, width: s.label.length * 6.2 + 20, height: 15, fill: "#0a0e14", opacity: 0.85 }, svg); svgEl("rect", { x: legX, y: T + 6, width: 10, height: 10, fill: s.color }, svg); const t = svgEl("text", { x: legX + 14, y: T + 15, class: "c8-ax" }, svg); t.textContent = s.label; legX += 22 + s.label.length * 6.2; });
    if (marker) { const m = svgEl("line", { x1: X(marker), x2: X(marker), y1: T, y2: H - B, stroke: "#fff", "stroke-width": 1.5 }, svg); m.classList.add("c8-marker"); }
    return { X, Y };
  }

  // ---------- 1. decomposition trees ----------
  const TREES = {
    exp: { e: ["EXP2", ["MUL", "x", "1.4427"]], src: "elementwise.py:519", note: "exp(x) = 2^(x·log₂e). 1.4427 is 1/ln 2, the constant you will find in every exp kernel below." },
    log: { e: ["MUL", ["LOG2", "x"], "0.6931"], src: "elementwise.py:837", note: "log(x) = log₂(x)·ln 2." },
    sigmoid: { e: ["RECIPROCAL", ["ADD", "1", ["EXP2", ["MUL", "x", "−1.4427"]]]], src: "elementwise.py:692", note: "σ(x) = 1/(1+2^(−x·log₂e)): one EXP2 and one RECIPROCAL." },
    tanh: { e: ["ADD", ["MUL", "2", ["RECIPROCAL", ["ADD", "1", ["EXP2", ["MUL", ["MUL", "2", "x"], "−1.4427"]]]]], "−1"], src: "elementwise.py:764", note: "tanh(x) = 2·σ(2x) − 1, so it is built on sigmoid." },
    gelu: { e: ["MUL", ["RECIPROCAL", ["ADD", "1", ["EXP2", ["MUL", ["ADD", "x", ["MUL", ["MUL", ["MUL", "0.0447", "x"], "x"], "x"]], "−2.3022"]]]], "x"], src: "elementwise.py:787 + symbolic", note: "Written as 0.5·x·(1+tanh(…)) with x**3; after symbolic simplification the captured kernel computes x/(1+2^(−2.3022·(x+0.0447·x³))). x**3 became x·x·x (symbolic.py:20) and 0.5·(1+(2σ−1)) folded to σ." },
    cos: { e: ["SIN", ["SUB", "π/2", "x"]], src: "elementwise.py:506", note: "There is no COS op: cos(x) = sin(π/2 − x)." },
    tan: { e: ["MUL", ["SIN", "x"], ["RECIPROCAL", ["SIN", ["SUB", "π/2", "x"]]]], src: "elementwise.py:925", note: "tan = sin/cos, and division by a tensor is multiplication by its RECIPROCAL." },
    pow: { e: ["WHERE", "fix-ups", "…", ["EXP2", ["MUL", ["LOG2", ["WHERE", ["CMPLT", "b", "0"], ["NEG", "b"], "b"]], "e"]]], src: "elementwise.py:564 + symbolic.py:450", note: "The frontend emits one POW op (with its own derivative, gradient.py:61). It is not a renderer primitive: in codegen symbolic rewrites it with xpow (transcendental.py:257-265) into 2^(e·log₂|b|), then WHERE fix-ups for negative bases, odd integer exponents and e = 0. A constant integer exponent never gets here: x**3 is rewritten to x·x·x first." },
    rsqrt: { e: ["RECIPROCAL", ["SQRT", "x"]], src: "elementwise.py:825", note: "SQRT is native in CUDA C (sqrt) and PTX (sqrt.approx). With TRANSCENDENTAL=2 it becomes xpow(x, 0.5)." },
    relu: { e: ["WHERE", ["CMPLT", "0", "x"], "x", "0"], src: "elementwise.py:680", note: "Not maximum(x, 0): the comment says that would give the wrong gradient at 0." },
    maximum: { e: ["WHERE", ["CMPLT", "a", "b"], "b", "a"], src: "op.py:89", note: "Ops.MAX itself is a primitive, but CUDA C has no MAX in code_for_op, so the early decompositions rewrite it to CMPLT + WHERE. PTX keeps it as max.f32." },
  };
  const PRIM = { EXP2: 1, LOG2: 1, SIN: 1, SQRT: 1, RECIPROCAL: 1 };
  function mountTree(root) {
    root.classList.add("widget");
    root.innerHTML = `<div class="widget-head"><span class="title">What a Tensor method becomes</span><div class="tabs">${Object.keys(TREES).map((k) => `<button class="btn" data-k="${k}">${k}</button>`).join("")}</div></div>
      <div class="widget-body"><svg class="c8-tree-svg"></svg><div class="c8-tree-note"></div>
      <div class="c8-legend"><span><i style="background:var(--s-render)"></i>transcendental primitive (EXP2, LOG2, SIN, SQRT, RECIPROCAL)</span><span><i style="background:var(--s-codegen)"></i>ordinary ALU op</span><span><i style="background:#33445a"></i>input or constant</span></div></div>`;
    const svg = root.querySelector("svg"), note = root.querySelector(".c8-tree-note");
    function show(k) {
      root.querySelectorAll("[data-k]").forEach((b) => b.classList.toggle("on", b.dataset.k === k));
      const t = TREES[k]; svg.innerHTML = "";
      // layout: leaves get consecutive slots, parents sit over the middle of their children
      let slot = 0; const nodes = [], edges = [];
      (function walk(n, depth, parent) {
        const me = { label: Array.isArray(n) ? n[0] : n, depth, leaf: !Array.isArray(n) };
        nodes.push(me); if (parent) edges.push([parent, me]);
        if (me.leaf) me.x = slot++; else { const kids = n.slice(1).map((c) => walk(c, depth + 1, me)); me.x = (kids[0].x + kids[kids.length - 1].x) / 2; }
        return me;
      })(t.e, 0, null);
      const maxD = Math.max(...nodes.map((n) => n.depth)), gw = 92, gh = 54, W = Math.max(560, slot * gw + 20), H = (maxD + 1) * gh + 16;
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.style.maxHeight = H * 1.15 + "px";
      const off = (W - slot * gw) / 2 + gw / 2, P = (n) => [off + n.x * gw, 22 + n.depth * gh];
      edges.forEach(([a, b]) => { const [x1, y1] = P(a), [x2, y2] = P(b); const l = svgEl("path", { d: `M${x1} ${y1 + 12} C${x1} ${y1 + 32} ${x2} ${y2 - 32} ${x2} ${y2 - 12}`, class: "c8-edge" }, svg); l.style.transitionDelay = (b.depth * 0.12) + "s"; });
      nodes.forEach((n) => {
        const [x, y] = P(n), g = svgEl("g", { class: "c8-node", transform: `translate(${x} ${y})` }, svg);
        const w = Math.max(30, n.label.length * 7.6 + 14), col = n.leaf ? "#33445a" : PRIM[n.label] ? "var(--s-render)" : "var(--s-codegen)";
        svgEl("rect", { x: -w / 2, y: -12, width: w, height: 24, rx: n.leaf ? 12 : 5, fill: n.leaf ? "#1c2430" : "#10161e", stroke: col, "stroke-width": PRIM[n.label] ? 2.4 : 1.4 }, g);
        const tx = svgEl("text", { y: 4, "text-anchor": "middle", class: n.leaf ? "c8-leaf" : "c8-op" }, g); tx.textContent = n.label;
        g.style.transitionDelay = (n.depth * 0.12) + "s";
      });
      requestAnimationFrame(() => requestAnimationFrame(() => svg.classList.add("grown")));
      svg.classList.remove("grown");
      note.innerHTML = `${esc(t.note)} <span class="chip">${esc(t.src)}</span>`;
    }
    root.querySelectorAll("[data-k]").forEach((b) => (b.onclick = () => show(b.dataset.k)));
    show("gelu");
  }

  // ---------- 2. the exp2 lab ----------
  function bitRow(label, u, cls = "") {
    const b = (u >>> 0).toString(2).padStart(32, "0");
    return `<div class="c8-bits ${cls}"><span class="c8-bl">${label}</span><span class="c8-bf">${[...b].map((c, i) => `<i class="${i === 0 ? "s" : i < 9 ? "e" : "m"}${c === "1" ? " one" : ""}">${c}</i>`).join("")}</span></div>`;
  }
  function mountExp2(root) {
    root.classList.add("widget");
    root.innerHTML = `<div class="widget-head"><span class="title">exp(x) the way TRANSCENDENTAL=2 computes it (float32, as on the GPU)</span>
      <label class="c8-ctl">x = <input type="number" class="btn c8-num" step="0.01" value="3.7"></label>
      <input type="range" class="c8-x" min="-105" max="90" step="0.01" value="3.7" style="flex:1;min-width:160px">
      <button class="btn c8-play">▶ animate the steps</button></div>
      <div class="widget-body c8-exp-grid">
        <ol class="c8-steps">
          <li data-s="0"><b>Scale.</b> <code>d = x · 1.4426950216293335f</code> <span class="v v0"></span><div class="why">exp(x) = 2^(x·log₂e); the frontend already wrote this multiply (elementwise.py:519).</div></li>
          <li data-s="1"><b>Range reduction.</b> <code>q = (int)(d + (d&lt;0 ? −0.5 : 0.5))</code>, <code>s = d − q</code> <span class="v v1"></span><div class="why">2^d = 2^q · 2^s with q an integer and s ∈ [−½, ½]. Only the small factor 2^s needs a polynomial.</div></li>
          <li data-s="2"><b>Polynomial.</b> Horner with the 7 coefficients from transcendental.py:209, each step one <code>FFMA</code>: <span class="v v2"></span><div class="c8-horner"></div></li>
          <li data-s="3"><b>Build 2^q from bits.</b> <code>pow2if(n)</code> puts <code>n+127</code> into the 8 exponent bits: <code>(n+127)&lt;&lt;23</code> reinterpreted as a float is exactly 2^n. q is split as a = q&gt;&gt;1 and b = q − a so that q = 128 or q = −150 does not overflow the field. <span class="v v3"></span><div class="c8-bitbox"></div></li>
          <li data-s="4"><b>Scale and clamp.</b> <code>u · 2^a · 2^b</code>, then d ≥ 128 → +inf, d &lt; −150 → 0, NaN → NaN. <span class="v v4"></span></li>
        </ol>
        <div><svg class="c8-plot-exp"></svg><div class="c8-cap">Error of the float32 Horner result for 2^s over the whole reduced range, in units in the last place (ulps) of the true value. Dots: float32 with FMA, as the GPU runs it. Line: the same polynomial in float64, which isolates the approximation error of the coefficients themselves. White line: the current s.</div></div>
      </div>`;
    const xs = root.querySelector(".c8-x"), num = root.querySelector(".c8-num"), svg = root.querySelector("svg");
    // static plot data
    const dots = [], line = [];
    for (let i = 0; i <= 600; i++) {
      const s = f(-0.5 + i / 600), ex = Math.pow(2, s); let u = 0, ud = 0;
      for (const c of EXP2_C) { u = fma(u, s, f(c)); ud = ud * s + c; }
      dots.push([s, (u - ex) / ulp32(ex)]); line.push([s, (ud - ex) / ulp32(ex)]);
    }
    function render(x, animate) {
      const R = xexp2(f(x * LOG2E_F)), d = R.d, ref = Math.exp(f(x)), ref32 = f(ref);
      const v = (k, h) => (root.querySelector(".v" + k).innerHTML = h);
      v(0, `→ d = <b>${fmt(d)}</b>`);
      v(1, `→ q = <b>${R.q}</b>, s = <b>${fmt(R.s)}</b>`);
      const ex = Math.pow(2, R.s), e2 = (R.u - ex) / ulp32(ex);
      v(2, `→ u = <b>${fmt(R.u)}</b> (2^s = ${fmt(ex, 10)}, error ${e2.toFixed(2)} ulp)`);
      root.querySelector(".c8-horner").innerHTML = R.steps.map((u, i) => `<span>${i ? "·s + " : ""}${fmt(f(EXP2_C[i]), 6)}</span><span class="hv">→ ${fmt(u, 7)}</span>`).join("");
      v(3, `→ a = ${R.a}, b = ${R.b}`);
      const ua = ((R.a + 127) & 0xff) << 23, ub = ((R.b + 127) & 0xff) << 23;
      const inRange = R.a + 127 > 0 && R.a + 127 < 255 && R.b + 127 > 0 && R.b + 127 < 255;
      root.querySelector(".c8-bitbox").innerHTML = (inRange ? "" : `<div class="c8-warn">a+127 or b+127 is outside 1..254, so these bit patterns are not 2^a, 2^b; the clamp in the next step replaces the result.</div>`) +
        bitRow(`a+127 = ${R.a + 127}`, (R.a + 127) & 0xff, "pre") + bitRow("&lt;&lt; 23 → 2^" + R.a, ua, "post") +
        bitRow(`b+127 = ${R.b + 127}`, (R.b + 127) & 0xff, "pre") + bitRow("&lt;&lt; 23 → 2^" + R.b, ub, "post") + bitRow("u", bitsOf(R.u), "u") + bitRow("result", bitsOf(R.r), "res") +
        `<div class="c8-cap">Sign bit, <span style="color:var(--s-render)">exponent</span>, <span style="color:var(--s-schedule)">mantissa</span>. Multiplying u by 2^a·2^b only adds a+b to u's exponent field; the mantissa bits of u pass through untouched. In the SASS the shift and the +127 fuse into one instruction: <code>LEA R10, R5, 0x3f800000, 0x17</code> is (R5 &lt;&lt; 23) + 0x3f800000, and 0x3f800000 is 127 &lt;&lt; 23.</div>`;
      const err = Number.isFinite(ref32) && Number.isFinite(R.r) ? (R.r - ref) / ulp32(ref) : NaN;
      v(4, `→ <b>${fmt(R.r)}</b> ${R.clamp ? `<span class="chip">${R.clamp}</span>` : ""}<br>exp(x) in float64: ${fmt(ref, 12)}${Number.isFinite(err) ? ` · error <b>${err.toFixed(2)} ulp</b>` : ""}`);
      plot(svg, { W: 460, H: 250, xDom: [-0.5, 0.5], yDom: [-1, 1], series: [{ pts: dots, color: "#60a5fa", kind: "dots", label: "float32 + FMA" }, { pts: line, color: "#fbbf24", label: "float64" }], xLabel: "s", yLabel: "error (ulp)", marker: R.s });
      if (animate) {
        const lis = [...root.querySelectorAll(".c8-steps li")];
        lis.forEach((li) => li.classList.remove("lit", "done"));
        root.querySelectorAll(".c8-bits.post .c8-bf, .c8-bits.pre .c8-bf").forEach((b) => b.classList.remove("shifted"));
        let i = 0; const tick = () => {
          if (i > 0) lis[i - 1].classList.replace("lit", "done");
          if (i >= lis.length) return;
          lis[i].classList.add("lit");
          if (i === 3) root.querySelectorAll(".c8-bits.post .c8-bf").forEach((b) => { b.classList.add("from-low"); requestAnimationFrame(() => requestAnimationFrame(() => { b.classList.remove("from-low"); })); });
          i++; setTimeout(tick, 1100);
        };
        tick();
      }
    }
    xs.oninput = () => { num.value = xs.value; render(+xs.value); };
    num.onchange = () => { xs.value = num.value; render(+num.value); };
    root.querySelector(".c8-play").onclick = () => render(+num.value, true);
    render(3.7);
  }

  // ---------- 3. the sine lab ----------
  function mountSin(root) {
    root.classList.add("widget");
    root.innerHTML = `<div class="widget-head"><span class="title">Range reduction for sin(x)</span>
      <label class="c8-ctl">x = <input type="number" class="btn c8-num" step="0.01" value="7.5"></label>
      <input type="range" class="c8-x" min="-40" max="40" step="0.01" value="7.5" style="flex:1;min-width:160px">
      <div class="tabs"><button class="btn on" data-m="pi">tinygrad: mod π</button><button class="btn" data-m="half">libdevice sinf: mod π/2</button></div>
      <button class="btn c8-spin">▶ unwind</button></div>
      <div class="widget-body c8-sin-grid"><svg class="c8-wheel" viewBox="-130 -130 260 260"></svg><div class="c8-sin-info"></div></div>
      <div class="widget-body c8-sin-plots"><div><svg class="c8-p1"></svg><div class="c8-cap">Error of tinygrad's <code>sin_poly</code> (5 coefficients, transcendental.py:153) on the reduced range [−π/2, π/2], against Math.sin. The polynomial alone is good to about 1e-7.</div></div>
      <div><svg class="c8-p2"></svg><div class="c8-cap">Worst error of the whole float32 sine as |x| grows (log scale), for tinygrad's 4-constant Cody–Waite reduction and for a naive x − q·π with one float32 π. Dashed: tinygrad switches to Payne–Hanek at |x| = 30 (xsin's switch_over); libdevice's sinf branches to its own Payne–Hanek loop at 105615.</div></div></div>`;
    const svg = root.querySelector(".c8-wheel"), info = root.querySelector(".c8-sin-info"), xs = root.querySelector(".c8-x"), num = root.querySelector(".c8-num");
    let mode = "pi", x = 7.5, anim = null;
    // static plots
    const p1 = [], p1d = [];
    for (let i = 0; i <= 500; i++) { const r = f(-Math.PI / 2 + Math.PI * i / 500); p1.push([r, sinPoly(r) - Math.sin(r)]); let p = 0; const d2 = r * r; for (const c of SIN_C) p = p * d2 + c; p1d.push([r, r * p - Math.sin(r)]); }
    plot(root.querySelector(".c8-p1"), { W: 460, H: 230, xDom: [-1.6, 1.6], yDom: [-1.5e-7, 1.5e-7], series: [{ pts: p1, color: "#60a5fa", kind: "dots", label: "float32 + FMA" }, { pts: p1d, color: "#fbbf24", label: "float64" }], xLabel: "reduced argument r", yLabel: "sin_poly(r) − sin(r)" });
    const cw = [], nv = []; let seed = 12345; const rnd = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
    for (let i = 0; i < 90; i++) {
      const lo = Math.pow(10, -1 + 7 * i / 90), hi = Math.pow(10, -1 + 7 * (i + 1) / 90); let e1 = 0, e2 = 0;
      for (let j = 0; j < 160; j++) { const xx = f(lo + (hi - lo) * rnd()), ref = Math.sin(xx); e1 = Math.max(e1, Math.abs(tgSinSmall(xx).r - ref)); e2 = Math.max(e2, Math.abs(naiveSin(xx) - ref)); }
      const xm = Math.sqrt(lo * hi); cw.push([xm, Math.log10(Math.max(e1, 1e-9))]); nv.push([xm, Math.log10(Math.max(e2, 1e-9))]);
    }
    plot(root.querySelector(".c8-p2"), { W: 460, H: 230, logX: true, xDom: [0.1, 1e6], yDom: [-8, 0], series: [{ pts: cw, color: "#34d399", label: "4-constant Cody–Waite" }, { pts: nv, color: "#f87171", label: "naive x − q·π" }], vlines: [[30, "#fbbf24", "30"], [105615, "#fb923c", "105615"]], xLabel: "|x|", yLabel: "log10 max |error|" });

    function draw(xv) {
      const xf = f(xv), period = mode === "pi" ? Math.PI : Math.PI / 2, k = mode === "pi" ? 2 : 4, ax = Math.abs(xf);
      let q, r, val, how;
      if (mode === "pi") { const t = tgSinSmall(xf); q = t.q; r = t.d; val = t.r; how = `sin(x) = sign · (−1)^q · sin_poly(r)`; }
      else { q = Math.round(ax / period); r = ax - q * period; const qq = q & 3; val = (xf < 0 ? -1 : 1) * [Math.sin(r), Math.cos(r), -Math.sin(r), -Math.cos(r)][qq]; how = `quadrant q mod 4 = ${qq} → ${["+sin(r)", "+cos(r)", "−sin(r)", "−cos(r)"][qq]}`; }
      const ref = Math.sin(xf), R = 100;
      svg.innerHTML = "";
      for (let i = 0; i < k; i++) { // sectors centred on the multiples of the period
        const a0 = (i - 0.5) * period, a1 = (i + 0.5) * period, on = ((q % k) + k) % k === i;
        const P = (a) => `${R * Math.cos(a)} ${-R * Math.sin(a)}`;
        svgEl("path", { d: `M0 0 L${P(a0)} A${R} ${R} 0 0 0 ${P(a1)} Z`, fill: on ? "rgba(251,191,36,.18)" : i % 2 ? "rgba(96,165,250,.06)" : "rgba(52,211,153,.06)", stroke: "#2a3441" }, svg);
        const t = svgEl("text", { x: 0.78 * R * Math.cos(i * period), y: -0.78 * R * Math.sin(i * period) + 4, "text-anchor": "middle", class: "c8-ax" }, svg); t.textContent = mode === "pi" ? (i ? "odd q: −" : "even q: +") : ["+sin", "+cos", "−sin", "−cos"][i];
      }
      svgEl("circle", { r: R, fill: "none", stroke: "#33445a" }, svg);
      const base = q * period, end = ax, pt = (a, rr = R) => [rr * Math.cos(a), -rr * Math.sin(a)];
      const [bx, by] = pt(base), [ex, ey] = pt(end);
      svgEl("line", { x1: 0, y1: 0, x2: bx, y2: by, stroke: "#fbbf24", "stroke-dasharray": "3 3" }, svg);
      const large = Math.abs(end - base) > Math.PI ? 1 : 0, sweep = end > base ? 0 : 1;
      svgEl("path", { d: `M${bx * 1.08} ${by * 1.08} A${R * 1.08} ${R * 1.08} 0 ${large} ${sweep} ${ex * 1.08} ${ey * 1.08}`, fill: "none", stroke: "#f472b6", "stroke-width": 4, "stroke-linecap": "round" }, svg);
      svgEl("line", { x1: 0, y1: 0, x2: ex, y2: ey, stroke: "#fff", "stroke-width": 2 }, svg);
      svgEl("circle", { cx: ex, cy: ey, r: 5, fill: "#fff" }, svg);
      svgEl("line", { x1: -R, x2: R, y1: 0, y2: 0, stroke: "#1c2430" }, svg);
      const turns = Math.floor(ax / (2 * Math.PI));
      const lab = svgEl("text", { x: 0, y: R + 24, "text-anchor": "middle", class: "c8-ax" }, svg); lab.textContent = `|x| wraps the circle ${turns} time${turns === 1 ? "" : "s"}`;
      const path = ax < 30 ? `|x| &lt; 30: Cody–Waite, then sin_poly_small` : `|x| ≥ 30: Payne–Hanek, then sin_poly_large (Cody–Waite is computed too, and discarded by the WHERE)`;
      const ld = ax < 105615 ? "|x| &lt; 105615: the short path (3-term Cody–Waite with π/2 split)" : "|x| ≥ 105615: the Payne–Hanek loop over __cudart_i2opi_f (the STL/LDL in the SASS)";
      info.innerHTML = `<table class="c8-t"><tr><th>x (float32)</th><td>${fmt(xf, 10)}</td></tr>
        <tr><th>q = round(|x| / ${mode === "pi" ? "π" : "(π/2)"})</th><td>${q}</td></tr>
        <tr><th>r = |x| − q·${mode === "pi" ? "π" : "π/2"}</th><td><span style="color:#f472b6">${fmt(r, 8)}</span>${mode === "pi" ? " ∈ [−π/2, π/2]" : " ∈ [−π/4, π/4]"}</td></tr>
        <tr><th>reconstruction</th><td>${how}</td></tr>
        <tr><th>result</th><td><b>${fmt(val, 9)}</b> (float64 sin: ${fmt(ref, 9)})</td></tr>
        <tr><th>error</th><td>${(val - ref).toExponential(2)}</td></tr></table>
        <div class="c8-path"><b>tinygrad xsin</b> takes: ${path}.<br><b>libdevice sinf</b> (the CUDA C default and PyTorch): ${ld}.</div>
        ${mode === "pi" ? `<div class="c8-cap">Values on the left are tinygrad's float32 arithmetic: q·(−3.1414794921875) + x, then three more FMAs with the smaller pieces of π (transcendental.py:138-141). Each piece has few enough bits that q·piece is exact in float32 for moderate q.</div>` : `<div class="c8-cap">Quadrant view in float64. libdevice multiplies by 2/π (the <code>FMUL … 0.63661974668502807617</code> in the SASS), reduces with three pieces of π/2, and picks sine or cosine coefficients with <code>FSEL</code> on the low bit of q.</div>`}`;
    }
    function set(v) { x = v; xs.value = Math.max(-40, Math.min(40, v)); num.value = v; draw(v); }
    xs.oninput = () => set(+xs.value);
    num.onchange = () => set(+num.value);
    root.querySelectorAll("[data-m]").forEach((b) => (b.onclick = () => { mode = b.dataset.m; root.querySelectorAll("[data-m]").forEach((c) => c.classList.toggle("on", c === b)); draw(x); }));
    root.querySelector(".c8-spin").onclick = () => { // animate x from 0 to its value so the reader sees the wrapping
      if (anim) cancelAnimationFrame(anim);
      const target = x, t0 = performance.now(), dur = 2200;
      const step = (t) => { const p = Math.min(1, (t - t0) / dur), e = p < 0.5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2; draw(target * e); if (p < 1) anim = requestAnimationFrame(step); else set(target); };
      anim = requestAnimationFrame(step);
    };
    set(7.5);
  }

  // ---------- 4. magic division ----------
  function magicgu(vmax, d) { // codegen/decomp/op.py:9-18, in BigInt
    vmax = BigInt(vmax); d = BigInt(d);
    const nc = (vmax + 1n) / d * d - 1n, nbits = vmax.toString(2).length;
    for (let s = 0; s <= 2 * nbits; s++) { const p = 1n << BigInt(s); if (p > nc * (d - 1n - (p - 1n) % d)) return [(p + d - 1n - (p - 1n) % d) / d, s]; }
    return null;
  }
  function mountIdiv(root) {
    root.classList.add("widget");
    root.innerHTML = `<div class="widget-head"><span class="title">Division by a constant as multiply and shift (fast_idiv)</span>
      <label class="c8-ctl">x // <input type="number" class="btn c8-d" min="1" max="100000" value="7" style="width:80px"></label>
      <label class="c8-ctl">with 0 ≤ x ≤ <select class="btn c8-v"><option value="1023">1023</option><option value="65535">65535</option><option value="1048575" selected>2^20 − 1 (the int_divmod inputs)</option><option value="2147483647">2^31 − 1</option></select></label></div>
      <div class="widget-body c8-idiv-out"></div>`;
    const out = root.querySelector(".c8-idiv-out");
    function run() {
      const d = Math.max(1, Math.floor(+root.querySelector(".c8-d").value || 1)), vmax = +root.querySelector(".c8-v").value;
      if ((d & (d - 1)) === 0) { out.innerHTML = `<p><b>${d} is a power of two.</b> The early decompositions already turned <code>x // ${d}</code> into <code>x &gt;&gt; ${Math.log2(d)}</code> and <code>x % ${d}</code> into <code>x &amp; ${d - 1}</code> (op.py:80-84). Both are exact floor division and floor modulo for any sign in two's complement, so no magic number is needed.</p>`; return; }
      const [m, s] = magicgu(vmax, d), mv = m * BigInt(vmax), fits32 = mv <= 2147483647n, fits64 = mv <= 9223372036854775807n;
      // check every x (or a dense sample above 2^22 values) with BigInt-free arithmetic when it is exact
      const N = Math.min(vmax, 1 << 22), stride = vmax > N ? Math.floor(vmax / N) : 1, mNum = Number(m), exact = Number(mv) < 2 ** 53;
      let bad = 0, checked = 0, firstBad = null;
      for (let x = 0; x <= vmax; x += stride) {
        checked++;
        const q = exact ? Math.floor(x * mNum / 2 ** s) : Number((BigInt(x) * m) >> BigInt(s));
        if (q !== Math.floor(x / d)) { bad++; if (firstBad === null) firstBad = x; }
        if (!exact && checked > 20000) break;
      }
      const emit = fits32 ? `((x*${m})&gt;&gt;${s})` : fits64 ? `((int)(((long)(x)*${m}l)&gt;&gt;${s}))` : "(no rewrite: keeps x/" + d + ")";
      out.innerHTML = `<div class="c8-idiv-grid"><div class="kstat"><span>magic m</span><b>${m}</b></div><div class="kstat"><span>shift s</span><b>${s}</b></div>
        <div class="kstat"><span>m · vmax</span><b>${mv}</b></div><div class="kstat"><span>fits in int32?</span><b>${fits32 ? "yes" : "no → widen to int64"}</b></div></div>
        <p>For every 0 ≤ x ≤ ${vmax}: <code>x // ${d} == (x · ${m}) &gt;&gt; ${s}</code>. Checked ${checked.toLocaleString()} values${stride > 1 || !exact ? " (sampled)" : " (all of them)"}: <b class="${bad ? "c8-bad" : "c8-good"}">${bad ? bad + " wrong, first at x = " + firstBad : "all correct"}</b>.</p>
        <p>Roughly what tinygrad would emit if it knew this range and <code>DISABLE_FAST_IDIV=0</code>: <code>${emit}</code>. The product must not overflow the integer type (op.py:30); if it would, tinygrad first tries to divide out a power of two from d, then a wider type (op.py:34-43).</p>`;
    }
    root.querySelectorAll("input,select").forEach((e) => (e.oninput = run));
    run();
  }

  BOOK.ready(() => {
    const m = [["#c8-tree", mountTree], ["#c8-exp2", mountExp2], ["#c8-sin", mountSin], ["#c8-idiv", mountIdiv]];
    for (const [sel, fn] of m) { const n = document.querySelector(sel); if (!n) continue; try { fn(n); } catch (e) { console.error(e); n.innerHTML = `<div class="callout warn">figure failed: ${esc(e.message)}</div>`; } }
  });
})();
