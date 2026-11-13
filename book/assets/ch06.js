/* Chapter 6: the optimizer. Custom widgets:
 *   #opt-viz     apply the captured applied_opts one at a time (shift_to semantics from postrange.py)
 *   #name-decode the three captured matmul kernel names, decoded axis by axis
 *   #blockmap    which outputs each thread of block (0,0) computes (derived from the same simulation)
 *   #regtile     register tiling: one thread, k step by k step
 *   #beam-tree   schematic of the beam search (winning path = captured applied_opts)
 *   #tflops      achieved TFLOP/s vs the L4's approximate fp32 peak
 */
BOOK.ready(() => {
  const esc = BOOK.esc;
  // ---- the scheduler model (mirrors tinygrad/codegen/opt/postrange.py) ----
  const COL = { WEAK: "#e5e7eb", GLOBAL: "#60a5fa", LOCAL: "#22d3ee", WARP: "#a5f3fc", UPCAST: "#facc15", REDUCE: "#f87171", UNROLL: "#e879f9", GROUP_REDUCE: "#fb7185" };
  const POS = { WEAK: -1, GLOBAL: 0, WARP: 1, LOCAL: 2, GROUP_REDUCE: 2, UPCAST: 3, REDUCE: 4, UNROLL: 5 };  // axis_to_pos
  const LET = { WEAK: "L", GLOBAL: "g", WARP: "w", LOCAL: "l", GROUP_REDUCE: "G", UPCAST: "u", REDUCE: "R", UNROLL: "r" };  // axis_letters
  const ORIG = { i: "row i of C (and A)", j: "column j of C (and B)", k: "reduce index k" };
  const parseOpt = (s) => { const m = /OptOps\.(\w+), axis=(-?\d+|None), arg=(\(.*\)|-?\d+|None)/.exec(s); return { op: m[1], axis: m[2] === "None" ? null : +m[2], arg: /^-?\d+$/.test(m[3]) ? +m[3] : m[3], text: s }; };
  const optShort = (o) => o.op === "GLOBALIZE" ? "convert_loop_to_global()" : `${o.op}(axis=${o.axis}, arg=${o.arg})`;

  function init(N = 1024) {
    const k = { id: 0, size: N, type: "REDUCE", orig: "k" }, i = { id: 1, size: N, type: "WEAK", orig: "i" }, j = { id: 2, size: N, type: "WEAK", orig: "j" };
    return { rngs: [k, i, j], terms: { i: [[i, 1]], j: [[j, 1]], k: [[k, 1]] }, next: 3 };
  }
  // Scheduler.rngs: ranges with size > 1, sorted by (axis_to_pos, range id)
  const rngs = (s) => s.rngs.filter((r) => r.size > 1).sort((a, b) => POS[a.type] - POS[b.type] || a.id - b.id);
  const shapeStr = (L) => { const c = {}; return L.map((r) => { c[r.type] = (c[r.type] ?? -1) + 1; return LET[r.type] + c[r.type]; }); };
  // Scheduler.shift_to: rng -> replaced*amount + new (top=False)
  function shiftTo(s, r, amt, type) {
    if (r.size % amt) throw new Error(`${amt} can't divide ${r.size}`);
    const nr = { id: s.next++, size: amt, type, orig: r.orig };
    r.size /= amt;
    for (const ax in s.terms) for (const t of s.terms[ax]) if (t[0] === r) { s.terms[ax].push([nr, t[1]]); t[1] *= amt; }
    s.rngs.push(nr);
    return nr;
  }
  function apply(s, o) {
    const L = rngs(s);
    if (o.op === "GLOBALIZE") { s.rngs.forEach((r) => { if (r.type === "WEAK") r.type = "GLOBAL"; }); return "The output loops become GLOBAL axes (convert_loop_to_global). This alone is the NOOPT kernel: one thread per output, one reduce loop."; }
    if (o.op === "UNROLL") {
      const un = L.filter((r) => r.type === "REDUCE" || r.type === "GROUP_REDUCE"), r = un[o.axis];
      if (!r) throw new Error("no such reduce axis");
      const amt = o.arg || r.size; if (amt > 32) throw new Error("don't unroll more than 32");
      const old = r.size, nr = shiftTo(s, r, amt, "UNROLL");
      return `UNROLL axis ${o.axis} counts reduce axes only, so it is R (${old}). It becomes ${r.size} loop iterations × ${amt} unrolled steps: each iteration of the loop now does ${amt} values of k, written out in straight-line code.`;
    }
    const r = L[o.axis]; if (!r) throw new Error("no such axis");
    const name = shapeStr(L)[o.axis], old = r.size;
    if (o.op === "UPCAST") {
      if (!["GLOBAL", "LOCAL", "WEAK"].includes(r.type)) throw new Error("upcast is for GLOBAL/LOCAL"); if (o.arg > 16) throw new Error("don't upcast more than 16");
      shiftTo(s, r, o.arg || r.size, "UPCAST");
      return `${name} (${old}, ${ORIG[r.orig]}) splits into ${r.size} × ${o.arg}. The ${o.arg} part is UPCAST: it is no longer a loop or a thread index, it is ${o.arg} separate registers in every thread. Adjacent upcast values are adjacent along ${r.orig} (stride 1 in the new sub-axis).`;
    }
    if (o.op === "LOCAL") {
      if (!["GLOBAL", "WEAK"].includes(r.type)) throw new Error("local is for globals");
      shiftTo(s, r, o.arg, "LOCAL");
      return `${name} (${old}, ${ORIG[r.orig]}) splits into ${r.size} × ${o.arg}. The ${o.arg} part is LOCAL: it becomes a threadIdx dimension, so ${o.arg} threads of one block share this axis. Its stride inside ${r.orig} is whatever was innermost before the split.`;
    }
    if (o.op === "SWAP") {
      const a = L[o.arg]; if (r.type !== "GLOBAL" || a.type !== "GLOBAL") throw new Error("swap only for globals");
      [r.id, a.id] = [a.id, r.id];
      return `SWAP exchanges the range ids of ${name} and ${shapeStr(L)[o.arg]}. Sizes and strides do not change, only the order: the axes are sorted by id, so the two globals trade places in the name and in the grid (which one becomes blockIdx.x).`;
    }
    throw new Error("not simulated: " + o.op);
  }
  // gpudims: globals reversed onto x,y,z (get_grouped_dims reverse=True), locals in order
  function launch(s) {
    const L = rngs(s), g = L.filter((r) => r.type === "GLOBAL"), l = L.filter((r) => r.type === "LOCAL" || r.type === "WARP");
    if (!g.length) return null;
    const grid = g.map((r) => r.size).reverse(), block = l.map((r) => r.size);
    while (grid.length < 3) grid.push(1); while (block.length < 3) block.push(1);
    return { grid, block };
  }
  const kname = (s) => "r_" + rngs(s).map((r) => r.size).join("_");
  const nameHTML = (s) => `<span class="nm">r${rngs(s).map((r) => `<i>_</i><b style="color:${COL[r.type]}">${r.size}</b>`).join("")}</span>`;
  function formulas(s) {
    const L = rngs(s), names = new Map(L.map((r, i) => [r, shapeStr(L)[i]]));
    return ["i", "j", "k"].map((ax) => {
      const t = s.terms[ax].filter(([r]) => r.size > 1).sort((a, b) => b[1] - a[1]);
      return `<div><b>${ax}</b> = ${t.map(([r, st]) => `${st > 1 ? st + "·" : ""}<span style="color:${COL[r.type]}">${names.get(r)}</span>`).join(" + ")}</div>`;
    }).join("");
  }
  function states(opts, N) {  // replay: state after every step
    const out = [{ s: init(N), msg: "The kernel AST as the scheduler hands it over: two output loops (WEAK, printed L) and one REDUCE loop, all 1024." }];
    [{ op: "GLOBALIZE" }, ...opts].forEach((o) => {
      const s = init(N); let msg = "";
      for (const p of [{ op: "GLOBALIZE" }, ...opts].slice(0, out.length)) msg = apply(s, p);
      out.push({ s, msg, opt: o });
    });
    return out;
  }
  BOOK.ch06 = { init, apply, rngs, shapeStr, states, launch, kname, COL, parseOpt };

  // ---- #opt-viz ----
  const viz = document.getElementById("opt-viz");
  if (viz) {
    const presets = { "matmul/default": "hand-coded heuristic", "matmul/beam2": "BEAM=2" };
    let key = "matmul/default", steps, cur = 0, timer = null;
    viz.classList.add("widget");
    viz.innerHTML = `<div class="widget-head"><span class="title">Apply the opts one at a time: 1024×1024×1024 fp32 matmul</span>
      <select class="btn preset">${Object.entries(presets).map(([k, v]) => `<option value="${k}">${v} (${k})</option>`).join("")}</select>
      <button class="btn reset">⟲ reset</button><button class="btn back">◀</button><button class="btn fwd">apply next ▶</button><button class="btn play">▶ play</button></div>
      <div class="widget-body"><div class="ov-opts"></div><div class="ov-bar"></div><div class="ov-grid"><div><div class="ov-label">kernel name</div><div class="ov-name"></div>
      <div class="ov-label">index math (what each original index is made of)</div><div class="ov-form mono"></div></div>
      <div><div class="ov-label">launch and per-thread work</div><div class="ov-launch"></div></div></div><div class="ov-msg"></div><div class="ov-check"></div></div>`;
    const $ = (c) => viz.querySelector(c);
    function load() {
      const k = BOOK.get(key).kernels[0];
      steps = states(k.applied_opts.map(parseOpt));
      cur = 0; render();
    }
    function render() {
      const { s, msg } = steps[cur], L = rngs(s), names = shapeStr(L), k = BOOK.get(key).kernels[0];
      $(".ov-opts").innerHTML = ["start", "globalize", ...k.applied_opts.map((x) => optShort(parseOpt(x)))].map((t, i) =>
        `<span class="chip ov-o ${i < cur ? "done" : i === cur ? "now" : ""}" data-i="${i}">${i ? i - 1 ? i - 1 + ". " : "" : ""}${esc(t)}</span>`).join("");
      const tot = L.reduce((a, r) => a + Math.log2(r.size), 0);
      $(".ov-bar").innerHTML = L.map((r, i) => `<div class="ov-ax" style="flex:${Math.log2(r.size)} 1 0;background:${COL[r.type]}">
        <b>${names[i]}</b><span>${r.size}</span><em>${r.orig} · ${r.type.toLowerCase()}</em></div>`).join("") + `<div class="ov-tot">width ∝ log₂(size); total ${tot} bits = 1024³</div>`;
      $(".ov-name").innerHTML = s.rngs.some((r) => r.type === "WEAK") ? `<span class="muted">(not named yet: on a GPU, output loops are globalized first)</span>` : nameHTML(s);
      $(".ov-form").innerHTML = formulas(s);
      const ln = launch(s), up = L.filter((r) => r.type === "UPCAST").reduce((a, r) => a * r.size, 1), unr = L.filter((r) => r.type === "UNROLL").reduce((a, r) => a * r.size, 1);
      const loop = L.filter((r) => r.type === "REDUCE").reduce((a, r) => a * r.size, 1);
      $(".ov-launch").innerHTML = ln ? `<table class="ov-t"><tr><td>grid (blocks)</td><td>${ln.grid.join(" × ")}</td></tr><tr><td>block (threads)</td><td>${ln.block.join(" × ")} = ${ln.block.reduce((a, b) => a * b)}</td></tr>
        <tr><td>outputs per thread (accumulators)</td><td>${up}</td></tr><tr><td>reduce loop trips</td><td>${loop}${unr > 1 ? ` × ${unr} unrolled` : ""}</td></tr>
        <tr><td>multiply-adds per loop iteration</td><td>${up * unr}</td></tr></table>` : `<span class="muted">no GPU axes yet</span>`;
      $(".ov-msg").innerHTML = (cur ? `<b>${esc(optShort(steps[cur].opt))}</b>: ` : "") + esc(msg);
      if (cur === steps.length - 1) {
        const ok = kname(s) === k.name && ln.grid.slice(0, 2).join() === k.global_size.slice(0, 2).join() && ln.block.slice(0, 2).join() === k.local_size.slice(0, 2).join();
        $(".ov-check").innerHTML = `<span class="chip ${ok ? "ok" : "bad"}">${ok ? "✓" : "✗"} simulated ${esc(kname(s))}, grid ${ln.grid.join("×")}, block ${ln.block.join("×")} · captured on the L4: ${esc(k.name)}, grid ${k.global_size.join("×")}, block ${k.local_size.join("×")}</span>`;
      } else $(".ov-check").innerHTML = "";
    }
    const go = (i) => { cur = Math.max(0, Math.min(steps.length - 1, i)); render(); };
    $(".fwd").onclick = () => go(cur + 1); $(".back").onclick = () => go(cur - 1); $(".reset").onclick = () => go(0);
    $(".ov-opts").onclick = (e) => { if (e.target.dataset.i) go(+e.target.dataset.i); };
    $(".preset").onchange = (e) => { key = e.target.value; load(); };
    $(".play").onclick = (e) => {
      if (timer) { clearInterval(timer); timer = null; e.target.textContent = "▶ play"; return; }
      if (cur === steps.length - 1) go(0);
      e.target.textContent = "❚❚ pause";
      timer = setInterval(() => { if (cur >= steps.length - 1) { clearInterval(timer); timer = null; e.target.textContent = "▶ play"; } else go(cur + 1); }, 1600);
    };
    load();
  }

  // ---- #name-decode ----
  const nd = document.getElementById("name-decode");
  if (nd) {
    const rows = ["matmul/noopt", "matmul/default", "matmul/beam2"].map((key) => {
      const k = BOOK.get(key).kernels[0], st = states(k.applied_opts.map(parseOpt)), s = st[st.length - 1].s, L = rngs(s), nm = shapeStr(L);
      return `<tr><td class="mono">${esc(key)}</td><td>${nameHTML(s)} ${kname(s) === k.name ? '<span class="chip ok">= capture</span>' : '<span class="chip bad">≠ capture</span>'}</td>
        <td>${L.map((r, i) => `<span class="axchip" style="border-color:${COL[r.type]}"><b style="color:${COL[r.type]}">${nm[i]}</b> ${r.size} <em>${r.orig}</em></span>`).join("")}</td></tr>`;
    });
    nd.innerHTML = `<table class="nd"><tr><th>capture</th><th>kernel name</th><th>axes in name order (letter, size, which original index)</th></tr>${rows.join("")}</table>`;
  }

  // ---- #blockmap ----
  const bm = document.getElementById("blockmap");
  if (bm) {
    let key = "matmul/default", hover = 0;
    bm.classList.add("widget");
    bm.innerHTML = `<div class="widget-head"><span class="title">Who computes what inside block (0,0)</span>
      <div class="tabs"><button class="btn on" data-k="matmul/default">heuristic</button><button class="btn" data-k="matmul/beam2">BEAM=2</button></div></div>
      <div class="widget-body bm-body"><canvas></canvas><div class="bm-info"></div></div>`;
    const cv = bm.querySelector("canvas"), info = bm.querySelector(".bm-info");
    let model;
    function build() {
      const k = BOOK.get(key).kernels[0], st = states(k.applied_opts.map(parseOpt)), s = st[st.length - 1].s, L = rngs(s);
      const loc = L.filter((r) => r.type === "LOCAL"), ups = L.filter((r) => r.type === "UPCAST");
      const stride = (r, ax) => (s.terms[ax].find((t) => t[0] === r) || [0, 0])[1];
      const cells = [];  // [row, col, tid]
      const nthr = loc.reduce((a, r) => a * r.size, 1), nup = ups.reduce((a, r) => a * r.size, 1);
      for (let t = 0; t < nthr; t++) for (let u = 0; u < nup; u++) {
        let row = 0, col = 0, tt = t, uu = u;
        for (const r of loc) { const v = tt % r.size; tt = Math.floor(tt / r.size); row += v * stride(r, "i"); col += v * stride(r, "j"); }
        for (const r of ups) { const v = uu % r.size; uu = Math.floor(uu / r.size); row += v * stride(r, "i"); col += v * stride(r, "j"); }
        cells.push([row, col, t]);
      }
      const R = Math.max(...cells.map((c) => c[0])) + 1, C = Math.max(...cells.map((c) => c[1])) + 1;
      const own = new Int32Array(R * C); cells.forEach(([r, c, t]) => (own[r * C + c] = t));
      const tx = loc[0], ty = loc[1];
      model = { R, C, own, loc, ups, stride, k, tx, ty, nthr };
      const cs = Math.max(2, Math.min(9, Math.floor(560 / C), Math.floor(420 / R)));
      cv.width = C * cs; cv.height = R * cs; model.cs = cs; draw();
    }
    function color(t, on) {
      const w = t >> 5, lane = t & 31, hues = [210, 150, 35, 320];
      return on ? "#ffffff" : `hsl(${hues[w % 4]} 70% ${28 + (lane % 8) * 5}%)`;
    }
    function draw() {
      const { R, C, own, cs } = model, g = cv.getContext("2d");
      for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) { const t = own[r * C + c]; g.fillStyle = color(t, t === hover); g.fillRect(c * cs, r * cs, cs - (cs > 3), cs - (cs > 3)); }
      const { tx, ty, stride, k } = model, x = hover % tx.size, y = Math.floor(hover / tx.size);
      const mine = []; for (let i = 0; i < R * C; i++) if (own[i] === hover) mine.push([Math.floor(i / C), i % C]);
      const rows = [...new Set(mine.map((m) => m[0]))], cols = [...new Set(mine.map((m) => m[1]))];
      const dir = (r) => stride(r, "i") ? `row i, stride ${stride(r, "i")}` : `column j, stride ${stride(r, "j")}`;
      info.innerHTML = `<div><b>${esc(k.name)}</b>: block tile ${R} rows × ${C} columns = ${R * C} outputs, ${model.nthr} threads (4 warps, one hue each).</div>
        <div>Hover a cell. Thread <b>threadIdx = (${x}, ${y})</b>, warp ${hover >> 5}, lane ${hover & 31}: computes ${mine.length} outputs, rows {${rows.join(", ")}} × columns {${cols.join(", ")}}.</div>
        <div>threadIdx.x moves along ${dir(tx)}; threadIdx.y moves along ${dir(ty)}.</div>
        <div class="muted">Derived from the same opt simulation as above; the strides match the captured CUDA source.</div>`;
    }
    cv.onmousemove = (e) => { const b = cv.getBoundingClientRect(), c = Math.floor((e.clientX - b.left) * cv.width / b.width / model.cs), r = Math.floor((e.clientY - b.top) * cv.height / b.height / model.cs);
      if (r >= 0 && c >= 0 && r < model.R && c < model.C) { const t = model.own[r * model.C + c]; if (t !== hover) { hover = t; draw(); } } };
    bm.querySelectorAll("[data-k]").forEach((b) => (b.onclick = () => { key = b.dataset.k; hover = 0; bm.querySelectorAll("[data-k]").forEach((x) => x.classList.toggle("on", x === b)); build(); }));
    build();
  }

  // ---- #regtile ----
  const rt = document.getElementById("regtile");
  if (rt) {
    const MODES = { "matmul/noopt": [1, 1, "NOOPT: 1×1"], "matmul/default": [4, 4, "heuristic: 4×4"], "matmul/beam2": [16, 4, "BEAM=2: 16×4"] };
    let key = "matmul/default", k = 0, sub = -1, timer = null;
    const K = 8;
    rt.classList.add("widget");
    rt.innerHTML = `<div class="widget-head"><span class="title">Register tiling: one thread, one k step at a time</span>
      <div class="tabs">${Object.entries(MODES).map(([kk, m]) => `<button class="btn ${kk === key ? "on" : ""}" data-k="${kk}">${m[2]}</button>`).join("")}</div>
      <button class="btn step">step k ▶</button><button class="btn play">▶ play</button><button class="btn reset">⟲</button></div>
      <div class="widget-body rt-body"><svg></svg><div class="rt-info"></div></div>`;
    const svg = rt.querySelector("svg"), info = rt.querySelector(".rt-info");
    function draw() {
      const [R, C] = MODES[key], cs = R > 8 ? 16 : 26, gap = 30, ax = 40, bx = ax + K * cs + gap, by = 30, ay = by + K * cs + gap;
      const W = bx + C * cs + 170, H = ay + R * cs + 20;
      let h = `<text x="${ax}" y="${ay - 8}" class="rt-l">A: ${R} rows × k</text><text x="${bx}" y="${by - 10}" class="rt-l">B: k × ${C} cols</text><text x="${bx}" y="${ay - 8}" class="rt-l">C tile: ${R * C} accumulators (registers)</text>`;
      for (let r = 0; r < R; r++) for (let kk = 0; kk < K; kk++) h += `<rect x="${ax + kk * cs}" y="${ay + r * cs}" width="${cs - 2}" height="${cs - 2}" class="rt-a ${kk === k && sub >= 0 ? "hot" : kk < k ? "used" : ""}"/>`;
      for (let kk = 0; kk < K; kk++) for (let c = 0; c < C; c++) h += `<rect x="${bx + c * cs}" y="${by + kk * cs}" width="${cs - 2}" height="${cs - 2}" class="rt-b ${kk === k && sub >= 0 ? "hot" : kk < k ? "used" : ""}"/>`;
      for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) {
        const idx = r * C + c, done = k + (sub >= 0 && idx <= sub ? 1 : 0);
        h += `<rect x="${bx + c * cs}" y="${ay + r * cs}" width="${cs - 2}" height="${cs - 2}" class="rt-c" style="fill:rgba(52,211,153,${0.08 + 0.9 * done / K})"/>`;
        if (sub >= 0 && idx === sub) h += `<rect x="${bx + c * cs - 2}" y="${ay + r * cs - 2}" width="${cs + 2}" height="${cs + 2}" class="rt-now"/>` +
          `<line x1="${ax + k * cs + cs / 2}" y1="${ay + r * cs + cs / 2}" x2="${bx + c * cs + cs / 2}" y2="${ay + r * cs + cs / 2}" class="rt-ln"/><line x1="${bx + c * cs + cs / 2}" y1="${by + k * cs + cs / 2}" x2="${bx + c * cs + cs / 2}" y2="${ay + r * cs + cs / 2}" class="rt-ln"/>`;
      }
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.style.maxHeight = H + "px"; svg.innerHTML = h;
      const steps = k + (sub >= R * C - 1 ? 1 : 0), loads = (k + (sub >= 0 ? 1 : 0)) * (R + C), ffma = k * R * C + (sub >= 0 ? sub + 1 : 0);
      const cap = BOOK.get(key).kernels[0], mix = cap.sass_mix || {};
      info.innerHTML = `<div class="rt-n"><span>k steps</span><b>${steps}</b></div><div class="rt-n"><span>floats loaded</span><b>${loads}</b></div><div class="rt-n"><span>FFMAs</span><b>${ffma}</b></div>
        <div class="rt-n"><span>FFMA per float loaded</span><b>${(R * C / (R + C)).toFixed(2)}</b></div>
        <p>Each k step loads ${R} values of A (a column slice) and ${C} of B (a row slice), then does all ${R}×${C} = ${R * C} multiply-adds between them: the outer product. Every A value is reused ${C}×, every B value ${R}×, from registers.</p>
        <p class="muted">The real kernel <code>${esc(cap.name)}</code>: ${cap.regs} registers per thread (ptxas), ${mix.FFMA} FFMA in its SASS listing, ${BOOK.fmtUs(cap.time_us)} on the L4.</p>`;
    }
    function tick() {
      const [R, C] = MODES[key];
      if (sub < R * C - 1) sub += R * C > 16 ? 4 : 1; else { k++; sub = -1; if (k >= K) { k = 0; } }
      if (sub > R * C - 1) sub = R * C - 1;
      draw();
    }
    function stepK() { const [R, C] = MODES[key]; if (sub >= 0) { k = (k + 1) % K; sub = -1; } else sub = R * C - 1; draw(); }
    rt.querySelector(".step").onclick = stepK;
    rt.querySelector(".reset").onclick = () => { k = 0; sub = -1; draw(); };
    rt.querySelector(".play").onclick = (e) => {
      if (timer) { clearInterval(timer); timer = null; e.target.textContent = "▶ play"; return; }
      e.target.textContent = "❚❚ pause"; timer = setInterval(tick, 140);
    };
    rt.querySelectorAll("[data-k]").forEach((b) => (b.onclick = () => { key = b.dataset.k; k = 0; sub = -1; rt.querySelectorAll("[data-k]").forEach((x) => x.classList.toggle("on", x === b)); draw(); }));
    draw();
    BOOK.whenVisible(rt, () => { if (!timer) rt.querySelector(".play").click(); });
  }

  // ---- #beam-tree: schematic ----
  const bt = document.getElementById("beam-tree");
  if (bt) {
    const opts = BOOK.get("matmul/beam2").kernels[0].applied_opts.map(parseOpt), n = opts.length;
    const W = 120 + n * 118, H = 250;
    let h = `<text x="50" y="20" class="bt-l">round →</text>`;
    const node = (x, y, cls, label) => `<g class="bt-n ${cls}"><circle cx="${x}" cy="${y}" r="${cls === "win" ? 9 : 5}"/>${label ? `<text x="${x}" y="${y + 24}">${label}</text>` : ""}</g>`;
    h += node(50, 120, "win", "start");
    for (let r = 0; r < n; r++) {
      const x = 120 + r * 118;
      h += `<text x="${x}" y="20" class="bt-l">${r + 1}</text>`;
      for (let c = 0; c < 9; c++) { const y = 45 + c * 19; if (c === 4) continue; h += `<line x1="${x - 118 + 50 * (r === 0)}" y1="${r === 0 ? 120 : 120}" x2="${x}" y2="${y}" class="bt-e" style="animation-delay:${r * 0.5}s"/>` + node(x, y, "cand", ""); }
      h += `<line x1="${r === 0 ? 50 : x - 118}" y1="120" x2="${x}" y2="120" class="bt-w" style="animation-delay:${r * 0.5 + 0.3}s"/>` + node(x, 120, "win", "");
      h += `<text x="${x}" y="228" class="bt-o" style="animation-delay:${r * 0.5 + 0.3}s">${esc(opts[r].op)}</text><text x="${x}" y="243" class="bt-o" style="animation-delay:${r * 0.5 + 0.3}s">${opts[r].axis},${esc(String(opts[r].arg))}</text>`;
    }
    bt.innerHTML = `<svg viewBox="0 0 ${W} ${H}">${h}</svg><div class="code-note">Schematic, not timing data: each round expands every kept kernel by every legal single opt (gray), compiles and times them, and keeps the fastest 2 (BEAM=2). The highlighted path is the captured result: one opt per round, in the order of <code>applied_opts</code>.</div>`;
    BOOK.whenVisible(bt, () => bt.classList.add("go"));
  }

  // ---- #tflops ----
  const tf = document.getElementById("tflops");
  if (tf) {
    const FL = 2 * 1024 ** 3, PEAK = 30.3;
    const t = BOOK.get("matmul/torch").eager_kernels.find((e) => /gemm/.test(e.name));
    const rows = [["NOOPT", BOOK.get("matmul/noopt").kernels[0].time_us], ["heuristic", BOOK.get("matmul/default").kernels[0].time_us], ["BEAM=2", BOOK.get("matmul/beam2").kernels[0].time_us], ["cuBLAS " + t.name, t.us]];
    tf.classList.add("widget", "bars");
    tf.innerHTML = `<div class="widget-head"><span class="title">Achieved TFLOP/s = 2·1024³ / time, against the L4's ~30.3 TFLOP/s fp32 peak</span></div><div class="widget-body">${rows.map(([l, us], i) => {
      const v = FL / (us * 1e-6) / 1e12;
      return `<div class="row"><span>${esc(l)}</span><div class="track"><div class="fill" data-w="${100 * v / PEAK}" style="background:${i === 3 ? "#ee4c2c" : "var(--s-codegen)"}"></div></div><span class="val">${v.toFixed(v < 1 ? 3 : 2)} · ${(100 * v / PEAK).toFixed(1)}%</span></div>`; }).join("")}
      <div class="code-note">Times from the captures (tinygrad DEBUG=2 timing; torch.profiler for cuBLAS). The bar's full width is the approximate fp32 peak from NVIDIA's L4 datasheet (30.3 TFLOPS at boost clock).</div></div>`;
    BOOK.whenVisible(tf, () => tf.querySelectorAll(".fill").forEach((f) => (f.style.width = Math.max(0.5, f.dataset.w) + "%")));
  }
});
