/* Chapter 4: movement ops become index math.
 * - .tg-graph[data-key]            : prints the captured lazy tensor graph
 * - .ir-snip[data-key][data-pass]  : prints the IR after one named pass (optionally filtered / highlighted)
 * - #idxmap                        : interactive index-mapping grid (miniature shapes, real formulas from the captures)
 * - #coalesce                      : one warp's addresses for one load/store instruction, grouped into 32-byte sectors
 */
BOOK.ready((B) => {
  const esc = B.esc;

  // ---------- small helpers that print captured artifacts ----------
  document.querySelectorAll(".tg-graph[data-key]").forEach((n) => {
    const d = B.get(n.dataset.key);
    n.classList.add("widget");
    n.innerHTML = `<div class="widget-head"><span class="title">${esc(n.dataset.title || "Lazy tensor graph: " + n.dataset.key)}</span></div>` +
      B.codeBlock(d.tensor_graph.text, "uop", { highlight: n.dataset.highlight, note: "Printed by tinygrad's <code>pyrender</code> before scheduling (captured on the L4 run)." });
  });
  function passIr(d, name, kernel) {
    let last = null;
    for (const p of d.passes) {
      if (kernel != null && p.kernel != null && String(p.kernel) !== String(kernel)) continue;
      if (p.ir) last = p.ir.text;
      if (p.name === name && (kernel == null || p.kernel == null || String(p.kernel) === String(kernel))) return last;
    }
    return null;
  }
  B.passIr = B.passIr || passIr;
  document.querySelectorAll(".ir-snip[data-key][data-pass]").forEach((n) => {
    const d = B.get(n.dataset.key);
    let text = passIr(d, n.dataset.pass, n.dataset.kernel) || "(pass not found)";
    if (n.dataset.grep) text = text.split("\n").filter((l) => new RegExp(n.dataset.grep).test(l)).join("\n");
    n.classList.add("widget");
    n.innerHTML = `<div class="widget-head"><span class="title">${esc(n.dataset.title || "IR after “" + n.dataset.pass + "”: " + n.dataset.key)}</span></div>` +
      B.codeBlock(text, "uop", { highlight: n.dataset.highlight, note: n.dataset.note || "Captured IR (pyrender) after the named pass." });
  });

  // pull one line from a captured kernel's CUDA source
  const srcLine = (key, re, k = 0) => (B.get(key).kernels[k].source.split("\n").find((l) => re.test(l)) || "").trim();
  const irLines = (key, pass, re) => (passIr(B.get(key), pass) || "").split("\n").filter((l) => re.test(l)).join("\n");

  // ---------- 1. index-mapping grid ----------
  (function indexMap() {
    const root = document.getElementById("idxmap");
    if (!root) return;
    // Each mode: a miniature of the real example. out = [rows, cols]; inputs = [{name, rows, cols}];
    // read(i,j) -> {src, r, c} or null (padding); mini = formula string in miniature sizes; real = text pulled from the captures.
    const MODES = {
      transpose: {
        label: "transpose  a.T", key: "transpose/default", out: [6, 6], inputs: [{ name: "a", rows: 6, cols: 6 }],
        read: (i, j) => ({ src: 0, r: j, c: i }),
        mini: (i, j) => `in = j*6 + i = ${j}*6 + ${i} = ${j * 6 + i}`,
        idea: "PERMUTE (1,0) swaps the two ranges: the output's row range indexes the input's columns.",
        real: () => irLines("transpose/default", "apply rangeify", /\.index\(/),
        cuda: () => srcLine("transpose/default", /float val0/) + "\n" + srcLine("transpose/default", /int alu0/),
      },
      broadcast: {
        label: "broadcast  a + b.reshape(1, N)", key: "broadcast_add/default", out: [5, 6], inputs: [{ name: "a", rows: 5, cols: 6 }, { name: "b", rows: 1, cols: 6 }],
        read: (i, j) => [{ src: 0, r: i, c: j }, { src: 1, r: 0, c: j }],
        mini: (i, j) => `a: i*6 + j = ${i * 6 + j}      b: j = ${j}   (the row range is dropped)`,
        idea: "No EXPAND node at all: broadcast_rngs replaces the range of a broadcast axis with the constant 0, so b's index only uses the column range.",
        real: () => irLines("broadcast_add/default", "apply rangeify", /\.index\(|c6 = /),
        cuda: () => srcLine("broadcast_add/default", /val4 =/) + "\n" + srcLine("broadcast_add/default", /int alu0/),
      },
      slice: {
        label: "strided slice  a[::2, 1::3]", key: "pad_slice/default", out: [4, 3], inputs: [{ name: "a", rows: 8, cols: 9 }],
        read: (i, j) => ({ src: 0, r: 2 * i, c: 3 * j + 1 }),
        mini: (i, j) => `in = (2i)*9 + (3j+1) = ${2 * i}*9 + ${3 * j + 1} = ${2 * i * 9 + 3 * j + 1}`,
        idea: "A step is SHRINK + RESHAPE (n, step) + SHRINK to (n, 1): after rangeify, the step is a multiplier on the range and the start is an offset.",
        derived: true,
        real: () => "real kernel (1000×1000 input, inside the pad below):\n  in = (2*i)*1000 + (3*j + 1)",
        cuda: () => "(see the pad mode: the slice and the pad fuse into one expression)",
      },
      pad: {
        label: "slice + pad  a[::2,1::3].pad(((1,1),(2,2)))", key: "pad_slice/default", out: [6, 7], inputs: [{ name: "a", rows: 8, cols: 9 }],
        read: (i, j) => (i >= 1 && i < 5 && j >= 2 && j < 5 ? { src: 0, r: 2 * (i - 1), c: 3 * (j - 2) + 1 } : null),
        mini: (i, j) => {
          const v = i >= 1 && i < 5 && j >= 2 && j < 5;
          return `valid = (1 ≤ i < 5) & (2 ≤ j < 5) = ${v}\n` + (v ? `in = 2(i-1)*9 + 3(j-2)+1 = ${2 * (i - 1) * 9 + 3 * (j - 2) + 1}` : "value = 0.0 (no load)");
        },
        idea: "PAD turns the index into WHERE(valid, i - offset, Invalid). Out-of-range cells read nothing and become 0.0.",
        real: () => irLines("pad_slice/default", "apply rangeify", /c15 =|c22 =|c35 =|c38 =/),
        cuda: () => srcLine("pad_slice/default", /float val0/),
      },
      permute4d: {
        label: "permute(0,2,1,3) on (2,3,3,2)", key: "permute4d/default", out: [6, 6], inputs: [{ name: "a", rows: 6, cols: 6 }],
        // out[n][c][h][w] = a[n][h][c][w]; out displayed as rows (n,c) cols (h,w); input as rows (n,h) cols (c,w)
        read: (i, j) => { const n = Math.floor(i / 3), c = i % 3, h = Math.floor(j / 2), w = j % 2; return { src: 0, r: n * 3 + h, c: c * 2 + w }; },
        mini: (i, j) => { const n = Math.floor(i / 3), c = i % 3, h = Math.floor(j / 2), w = j % 2;
          return `out[n=${n}][${c}][${h}][w=${w}] ← a[${n}][${h}][${c}][${w}]\nin = n*18 + h*6 + c*2 + w = ${n * 18 + h * 6 + c * 2 + w}`; },
        idea: "Only the middle two ranges swap. The last axis keeps stride 1 on both sides, so runs along w stay contiguous (in the real kernel, runs of 32 floats: both sides are float4).",
        real: () => irLines("permute4d/default", "apply rangeify", /\.index\(|c13 =/),
        cuda: () => srcLine("permute4d/default", /float4 val0/),
      },
    };
    const CS = 30, GAP = 2;
    let mode = "transpose", timer = null, cur = -1;
    root.classList.add("widget");
    root.innerHTML = `<div class="widget-head"><span class="title">Which input cell does each output cell read?</span>
      <div class="tabs">${Object.entries(MODES).map(([k, m]) => `<button class="btn" data-m="${k}">${esc(m.label)}</button>`).join("")}</div>
      <button class="btn play">▶ sweep in store order</button></div>
      <div class="widget-body"><div class="im-idea"></div><div class="im-svg"></div>
      <div class="im-grid2"><div><div class="im-h">Miniature: this cell</div><pre class="code im-mini"></pre></div>
      <div><div class="im-h im-real-h"></div><pre class="code im-real"></pre>
      <div class="im-h">…and the final CUDA C</div><pre class="code im-cuda"></pre></div></div></div>
      <div class="code-note">Grids are miniatures of the captured examples (the real shapes are 1024×1024, 1000×1000 and 32⁴). Hover an output cell, or sweep: output cells are visited in memory order (the order a store walks them), and the trail shows where the reads land in the input's memory. Numbers in the input cells are flat offsets.</div>`;
    const svgBox = root.querySelector(".im-svg");
    function layout(m) {
      const ow = m.out[1] * (CS + GAP), oh = m.out[0] * (CS + GAP);
      let x = 20 + ow + 70, y = 30, boxes = [];
      m.inputs.forEach((inp) => { boxes.push({ x, y, ...inp }); y += inp.rows * (CS + GAP) + 34; });
      const W = Math.max(...boxes.map((b) => b.x + b.cols * (CS + GAP))) + 20, H = Math.max(oh + 50, y);
      return { ow, oh, boxes, W, H };
    }
    function draw() {
      const m = MODES[mode], L = layout(m);
      root.querySelectorAll("[data-m]").forEach((b) => b.classList.toggle("on", b.dataset.m === mode));
      root.querySelector(".im-idea").textContent = m.idea;
      let s = `<svg viewBox="0 0 ${L.W} ${L.H}" style="width:100%;max-width:${L.W * 1.25}px;display:block;margin:auto">`;
      s += `<text x="20" y="20" class="im-lbl">output (${m.out[0]}×${m.out[1]})</text>`;
      for (let i = 0; i < m.out[0]; i++) for (let j = 0; j < m.out[1]; j++) {
        const pad = m.read(i, j) === null;
        s += `<g class="im-o${pad ? " pad" : ""}" data-i="${i}" data-j="${j}"><rect x="${20 + j * (CS + GAP)}" y="${30 + i * (CS + GAP)}" width="${CS}" height="${CS}" rx="4"/>` +
          `<text x="${20 + j * (CS + GAP) + CS / 2}" y="${30 + i * (CS + GAP) + CS / 2 + 4}">${pad ? "0" : i * m.out[1] + j}</text></g>`;
      }
      L.boxes.forEach((b, bi) => {
        s += `<text x="${b.x}" y="${b.y - 10}" class="im-lbl">input ${b.name} (${b.rows}×${b.cols})</text>`;
        for (let r = 0; r < b.rows; r++) for (let c = 0; c < b.cols; c++)
          s += `<g class="im-i" data-b="${bi}" data-r="${r}" data-c="${c}"><rect x="${b.x + c * (CS + GAP)}" y="${b.y + r * (CS + GAP)}" width="${CS}" height="${CS}" rx="4"/>` +
            `<text x="${b.x + c * (CS + GAP) + CS / 2}" y="${b.y + r * (CS + GAP) + CS / 2 + 4}">${r * b.cols + c}</text></g>`;
      });
      s += `<polyline class="im-trail" points=""/><line class="im-link" x1="0" y1="0" x2="0" y2="0"/></svg>`;
      svgBox.innerHTML = s;
      root.querySelector(".im-real-h").innerHTML = m.derived ? "Real kernel: derived from the shapes (not in the capture: the slice only exists fused into the pad)" : "Real kernel: captured IR after <code>apply rangeify</code>";
      root.querySelector(".im-real").innerHTML = B.highlight(m.real(), "uop");
      root.querySelector(".im-cuda").innerHTML = B.highlight(m.cuda(), "c");
      svgBox.querySelectorAll(".im-o").forEach((g) => g.addEventListener("mouseenter", () => { stop(); show(+g.dataset.i, +g.dataset.j, false); }));
      trail = []; show(0, 0, false);
    }
    let trail = [];
    function center(m, L, bi, r, c) { const b = L.boxes[bi]; return [b.x + c * (CS + GAP) + CS / 2, b.y + r * (CS + GAP) + CS / 2]; }
    function show(i, j, keepTrail) {
      const m = MODES[mode], L = layout(m), rd = m.read(i, j), reads = rd === null ? [] : Array.isArray(rd) ? rd : [rd];
      svgBox.querySelectorAll(".im-o").forEach((g) => g.classList.toggle("cur", +g.dataset.i === i && +g.dataset.j === j));
      if (!keepTrail) { trail = []; svgBox.querySelectorAll(".im-i.seen").forEach((g) => g.classList.remove("seen")); }
      svgBox.querySelectorAll(".im-i").forEach((g) => {
        const hit = reads.some((x) => x.src === +g.dataset.b && x.r === +g.dataset.r && x.c === +g.dataset.c);
        g.classList.toggle("cur", hit); if (hit && keepTrail) g.classList.add("seen");
      });
      const link = svgBox.querySelector(".im-link"), oc = [20 + j * (CS + GAP) + CS / 2, 30 + i * (CS + GAP) + CS / 2];
      if (reads.length) { const [x2, y2] = center(m, L, reads[0].src, reads[0].r, reads[0].c); link.setAttribute("x1", oc[0]); link.setAttribute("y1", oc[1]); link.setAttribute("x2", x2); link.setAttribute("y2", y2); link.style.opacity = 1;
        if (keepTrail && reads[0].src === 0) { trail.push(`${x2},${y2}`); svgBox.querySelector(".im-trail").setAttribute("points", trail.join(" ")); } }
      else link.style.opacity = 0;
      if (!keepTrail) svgBox.querySelector(".im-trail").setAttribute("points", "");
      root.querySelector(".im-mini").textContent = `out[${i}][${j}]  (output offset ${i * m.out[1] + j})\n` + m.mini(i, j);
    }
    function stop() { if (timer) { clearInterval(timer); timer = null; root.querySelector(".play").textContent = "▶ sweep in store order"; } }
    root.querySelector(".play").onclick = () => {
      if (timer) return stop();
      const m = MODES[mode], n = m.out[0] * m.out[1];
      cur = -1; trail = []; svgBox.querySelectorAll(".im-i.seen").forEach((g) => g.classList.remove("seen"));
      root.querySelector(".play").textContent = "❚❚ pause";
      timer = setInterval(() => { cur++; if (cur >= n) return stop(); show(Math.floor(cur / m.out[1]), cur % m.out[1], true); }, 260);
    };
    root.querySelectorAll("[data-m]").forEach((b) => (b.onclick = () => { stop(); mode = b.dataset.m; draw(); }));
    draw();
  })();

  // ---------- 2. memory coalescing ----------
  (function coalesce() {
    const root = document.getElementById("coalesce");
    if (!root) return;
    // Each pattern: lanes(l) -> {elem, bytes} (elem = float offset) or null when the lane does not exist / is inactive.
    // Formulas are copied from the captured CUDA source (block indices fixed to small example values), except where noted.
    const P = {
      add: { label: "add: float4 load", key: "add/default", re: /int alu0/,
        lane: (l) => ({ elem: (3 << 7) + (l << 2), bytes: 16 }), ctx: "gidx0 = 3, lane = lidx0",
        note: "E_8192_32_4: 32 lanes × 16 B, all consecutive: 512 B = 4 full 128-byte lines. The ideal case." },
      t_ld: { label: "transpose (tinygrad): load val0", key: "transpose/default", re: /int alu0/,
        lane: (l) => { const lidx0 = l % 8, lidx1 = l >> 3; return { elem: lidx0 + (5 << 3) + (2 << 16) + (lidx1 << 12), bytes: 4 }; },
        ctx: "block 8×16 → warp 0 = lidx0 0..7 × lidx1 0..3; gidx0 = 2, gidx1 = 5",
        note: "Four scalar LDG.E per thread. Each one: 8 lanes read 32 consecutive bytes (one whole sector) in 4 different rows. Every byte fetched is used." },
      t_st: { label: "transpose (tinygrad): float4 store", key: "transpose/default", re: /float4\*/,
        lane: (l) => { const lidx0 = l % 8, lidx1 = l >> 3; return { elem: (2 << 6) + (lidx1 << 2) + (5 << 13) + (lidx0 << 10), bytes: 16 }; },
        ctx: "same warp: lidx0 0..7 (stride 1024 floats), lidx1 0..3 (stride 4)",
        note: "One STG.E.128 per thread. 4 lanes cover 64 consecutive bytes (two sectors) in each of 8 output rows. Again no wasted bytes." },
      t_eager: { label: "transpose (eager, one thread per output)", key: null,
        lane: (l) => ({ elem: l * 1024 + 7, bytes: 4 }), ctx: "lane = output column j; reads a[j][i]: stride 1024 floats",
        note: "Not from a capture: the access pattern a one-element-per-thread copy over TensorIterator strides produces (PyTorch's elementwise_kernel<128, 2> direct_copy). 32 lanes hit 32 sectors and use 4 of every 32 bytes." },
      t_triton: { label: "transpose (Triton): load, #blocked1 layout", key: null,
        lane: (l) => ({ elem: 4 * (l % 8) + 1024 * (l >> 3), bytes: 16 }), ctx: "sizePerThread [4,1], threadsPerWarp [8,4], order [0,1]",
        note: "From the captured TTGIR layout: 8 lanes × 4 floats run along the input's contiguous axis, so each warp reads 4 full 128-byte lines (LDG.E.128). The transposition happens in shared memory (convert_layout)." },
      bcast: { label: "broadcast_add: float4 load of a", key: "broadcast_add/default", re: /int alu1/,
        lane: (l) => ({ elem: (1 << 4) + (0 << 2) + (2 << 17) + (l << 12), bytes: 16 }), ctx: "lane = lidx0 (stride 4096 floats); lidx1 = 0, gidx0 = 1, gidx1 = 2",
        note: "threadIdx.x walks down rows (4 rows apart), so the 32 lanes hit 32 sectors and use 16 of each 32 bytes. The neighbouring warp (lidx1 + 1) reads the other halves." },
      pad: { label: "pad_slice: gated load, 2-thread blocks", key: "pad_slice/default", re: /float val0/,
        lane: (l) => (l < 2 ? { elem: 10 * 4000 + l * 2000 + 10 * 3 - 2005, bytes: 4 } : null), ctx: "block = 2 threads; gidx0 = gidx1 = 10",
        note: "local size 2: each warp has only 2 live lanes (the other 30 do not exist), each reading 4 bytes from a different row." },
      rows: { label: "sum_rows: load (Ridx0 = 0)", key: "sum_rows/default", re: /float val0/,
        lane: (l) => (l < 16 ? { elem: (l << 6) + 0 + (7 << 10), bytes: 4 } : null), ctx: "block = 16 threads; gidx0 = 7",
        note: "16 lanes, 256 bytes apart: 16 sectors for 64 useful bytes. The next 7 loads of each thread fall in the same sectors, so caches catch most of the waste (ptxas unrolled all 64 loads)." },
      cols: { label: "max_cols: load (Ridx0 = 0)", key: "max_cols/default", re: /float val0/,
        lane: (l) => (l < 16 ? { elem: 7 + (l << 16) + 0, bytes: 4 } : null), ctx: "block = 16 threads; gidx0 = 7 (one column)",
        note: "16 lanes read the same column, 64 rows (256 KB) apart. Adjacent columns are read by adjacent blocks." },
      sumall: { label: "sum_all kernel 0: float4 load val3", key: "sum_all/default", re: /int alu4/,
        lane: (l) => ({ elem: (1 << 19) + (l << 14) + (0 << 2), bytes: 16 }), ctx: "lane = lidx0 (stride 16384 floats = 4 rows); gidx0 = 1, Ridx0 = 0",
        note: "Each thread owns 4 whole rows, so lanes are 64 KB apart: 32 sectors, half used. The next unrolled load (+16 B) uses the other halves." },
    };
    let pat = "add", timer = null;
    root.classList.add("widget");
    root.innerHTML = `<div class="widget-head"><span class="title">One warp, one memory instruction: which 32-byte sectors does it touch?</span>
      <select class="btn psel">${Object.entries(P).map(([k, p]) => `<option value="${k}">${esc(p.label)}</option>`).join("")}</select>
      <button class="btn play">▶ issue</button></div>
      <div class="widget-body"><div class="co-svg"></div><div class="co-stats"></div><div class="co-note"></div><pre class="code co-src"></pre></div>
      <div class="code-note">Top: the 32 lanes of a warp (grey = no such thread in this block). Bottom: every 128-byte line the instruction touches, split into four 32-byte sectors (the unit DRAM and L2 transfer). Coloured bytes are bytes a lane asked for; the rest of a lit sector is fetched anyway.</div>`;
    const box = root.querySelector(".co-svg");
    const HUE = (l) => `hsl(${(l * 360) / 32},75%,62%)`;
    function render(animate) {
      const p = P[pat], reqs = [];
      for (let l = 0; l < 32; l++) { const q = p.lane(l); if (q) reqs.push({ l, a: q.elem * 4, n: q.bytes }); }
      const lines = [...new Set(reqs.map((q) => Math.floor(q.a / 128)))].sort((a, b) => a - b);
      const sectors = new Set(reqs.flatMap((q) => { const s = []; for (let b = q.a; b < q.a + q.n; b += 32) s.push(Math.floor(b / 32)); return s; }));
      const cols = lines.length <= 4 ? lines.length : lines.length <= 8 ? 4 : 4, LW = 192, LG = 22, rowsN = Math.ceil(lines.length / cols);
      const W = 900, top = 92, H = top + rowsN * 46 + 10, x0 = (W - (cols * LW + (cols - 1) * LG)) / 2;
      let s = `<svg viewBox="0 0 ${W} ${H}" style="width:100%;display:block">`;
      for (let l = 0; l < 32; l++) {
        const on = reqs.some((q) => q.l === l), x = 18 + l * 27;
        s += `<rect class="co-lane" x="${x}" y="8" width="24" height="24" rx="4" style="fill:${on ? HUE(l) : "#1c2430"};opacity:${on ? 1 : 0.5}"/><text class="co-ln" x="${x + 12}" y="24">${l}</text>`;
      }
      const pos = {}; lines.forEach((ln, i) => { pos[ln] = [x0 + (i % cols) * (LW + LG), top + Math.floor(i / cols) * 46]; });
      lines.forEach((ln) => {
        const [x, y] = pos[ln];
        s += `<text class="co-addr" x="${x}" y="${y - 3}">+0x${(ln * 128 - lines[0] * 128).toString(16)}</text>`;
        for (let k = 0; k < 4; k++) { const sid = ln * 4 + k; s += `<rect class="co-sec ${sectors.has(sid) ? "hit" : ""}" data-s="${sid}" x="${x + k * (LW / 4)}" y="${y}" width="${LW / 4 - 2}" height="22" rx="3"/>`; }
      });
      reqs.forEach((q, i) => {
        const ln = Math.floor(q.a / 128), [x, y] = pos[ln], bx = x + ((q.a % 128) / 128) * LW, bw = Math.max(2, (q.n / 128) * LW - 1);
        const lx = 18 + q.l * 27 + 12;
        s += `<path class="co-path" data-i="${i}" d="M${lx} 34 C ${lx} 60, ${bx + bw / 2} ${y - 30}, ${bx + bw / 2} ${y}" style="stroke:${HUE(q.l)}"/>`;
        s += `<rect class="co-byte" data-i="${i}" x="${bx}" y="${y + 3}" width="${bw}" height="16" rx="2" style="fill:${HUE(q.l)}"/>`;
      });
      s += "</svg>";
      box.innerHTML = s;
      const reqB = reqs.reduce((a, q) => a + q.n, 0), moved = sectors.size * 32;
      root.querySelector(".co-stats").innerHTML = `<span class="chip">${reqs.length} live lanes</span> <span class="chip">${reqB} B requested</span>
        <span class="chip">${lines.length} × 128 B lines</span> <span class="chip">${sectors.size} sectors = ${moved} B moved</span>
        <span class="chip ${reqB / moved > 0.99 ? "ok" : reqB / moved < 0.3 ? "bad" : ""}">efficiency ${Math.round((100 * reqB) / moved)}%</span>`;
      root.querySelector(".co-note").innerHTML = esc(p.note) + ` <span style="color:var(--muted)">(${esc(p.ctx)})</span>`;
      root.querySelector(".co-src").innerHTML = p.key ? B.highlight(srcLine(p.key, p.re), "c") : B.highlight("// pattern derived from layout / strides, see note", "c");
      const els = [...box.querySelectorAll(".co-path,.co-byte")], secs = [...box.querySelectorAll(".co-sec.hit")];
      if (!animate) { els.forEach((e) => e.classList.add("on")); secs.forEach((e) => e.classList.add("on")); return; }
      let i = 0; clearInterval(timer);
      timer = setInterval(() => {
        if (i >= reqs.length) { clearInterval(timer); timer = null; secs.forEach((e) => e.classList.add("on")); return; }
        box.querySelectorAll(`[data-i="${i}"]`).forEach((e) => e.classList.add("on")); i++;
      }, 70);
    }
    root.querySelector(".psel").onchange = (e) => { pat = e.target.value; render(true); };
    root.querySelector(".play").onclick = () => render(true);
    render(false);
    B.whenVisible(root, () => render(true));
  })();
});
