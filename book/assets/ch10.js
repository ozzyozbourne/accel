/* Chapter 10: autograd animation, annotated DEBUG=2 line, TinyJit call table + timeline, symbolic-size logs. */
BOOK.ready((B) => {
  const esc = B.esc;
  const SRC = (path, line) => `<a class="src" target="_blank" href="https://github.com/tinygrad/tinygrad/blob/${B.commit}/${path}#L${line}">${path}:${line}</a>`;
  const NS = "http://www.w3.org/2000/svg";
  const svgEl = (tag, attrs = {}, parent) => {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (parent) parent.appendChild(e);
    return e;
  };
  const head = (title, extra = "") => `<div class="widget-head"><span class="title">${title}</span><span class="grow"></span>${extra}</div>`;

  // =====================================================================================
  // 1. Autograd: forward graph, then gradient nodes attached rule by rule, then kernels
  // =====================================================================================
  (function autograd() {
    const root = document.getElementById("ag-anim");
    if (!root) return;
    const W = 150, H = 40;
    // forward graph of (x @ w).relu().sum(), as printed by the CPU toposort experiment
    const N = {
      f_sum: [195, 44, "REDUCE ADD", "() loss = .sum()", "fwd", 0],
      f_where: [150, 104, "WHERE", "(256,256) relu", "fwd", 0],
      f_cmp: [340, 104, "CMPLT", "(256,256) c = 0<s", "fwd", 0],
      f_red: [195, 164, "REDUCE ADD", "(256,256) s = x @ w", "fwd", 0],
      f_perm: [195, 224, "PERMUTE", "(512,256,256)", "fwd", 0],
      f_mul: [195, 284, "MUL", "(256,256,512)", "fwd", 0],
      f_pw: [300, 344, "PERMUTE", "(1,256,512) wᵀ", "fwd", 0],
      f_rx: [90, 404, "RESHAPE", "(256,1,512)", "fwd", 0],
      f_rw: [300, 404, "RESHAPE", "(1,512,256)", "fwd", 0],
      f_x: [90, 464, "x", "(256,512) buffer", "fwd", 0],
      f_w: [300, 464, "w", "(512,256) buffer", "fwd", 0],
      g1: [610, 44, "CONST 1.0", "() seed dL/dL", "grad", 1],
      g2: [610, 104, "EXPAND", "(256,256) dL/d relu", "grad", 2],
      g3: [610, 164, "WHERE", "(256,256) dL/ds", "grad", 3],
      g4: [610, 224, "EXPAND", "(512,256,256)", "grad", 4],
      g5: [610, 284, "PERMUTE", "(256,256,512)", "grad", 5],
      g6: [610, 344, "MUL", "(256,256,512) x * g", "grad", 6],
      g7: [810, 344, "REDUCE ADD+RESHAPE", "(1,256,512) dL/dwᵀ", "grad", 7],
      g8: [810, 404, "PERMUTE", "(1,512,256)", "grad", 8],
      g9: [810, 464, "RESHAPE", "(512,256) dL/dw", "grad", 8],
    };
    const E = [ // [src, dst, step, kind]
      ["f_x", "f_rx", 0], ["f_w", "f_rw", 0], ["f_rw", "f_pw", 0], ["f_rx", "f_mul", 0], ["f_pw", "f_mul", 0], ["f_mul", "f_perm", 0],
      ["f_perm", "f_red", 0], ["f_red", "f_cmp", 0], ["f_red", "f_where", 0], ["f_cmp", "f_where", 0], ["f_where", "f_sum", 0],
      ["g1", "g2", 2, "g"], ["g2", "g3", 3, "g"], ["f_cmp", "g3", 3, "cross"], ["g3", "g4", 4, "g"], ["g4", "g5", 5, "g"],
      ["g5", "g6", 6, "g"], ["f_rx", "g6", 6, "cross"], ["g6", "g7", 7, "g"], ["g7", "g8", 8, "g"], ["g8", "g9", 8, "g"],
    ];
    const MOPS = new Set(["f_perm", "f_pw", "f_rx", "f_rw", "g2", "g4", "g5", "g8", "g9"]);
    const K0 = new Set(["f_w", "f_rw", "f_pw", "f_mul", "f_perm", "f_red", "f_cmp", "g1", "g2", "g3"]), K01 = new Set(["f_x", "f_rx"]);
    const DEAD = new Set(["f_where", "f_sum"]);
    const STEPS = [
      ["the forward graph", `<code>(x @ w).relu().sum()</code> before anything runs. <code>dot</code> made the matmul a broadcasting <b>MUL</b> of
        <code>x.reshape(256,1,512)</code> and <code>w</code> transposed to <code>(1,256,512)</code>, then a <b>REDUCE ADD</b> whose reduced axis was permuted to the front.
        relu is <code>(s &gt; 0).where(s, 0)</code>: a <b>CMPLT</b> and a <b>WHERE</b>.`, [["tinygrad/mixin/op.py", 390], ["tinygrad/uop/ops.py", 636], ["tinygrad/mixin/elementwise.py", 680]]],
      ["seed", `<code>gradient()</code> starts from <code>self.const_like(1.0)</code>: dL/dL = 1. <code>_deepwalk</code> has already kept only the nodes on a path from the loss to
        <code>w</code>.`, [["tinygrad/mixin/op.py", 471], ["tinygrad/mixin/gradient.py", 89]]],
      ["REDUCE(ADD) rule", `<code>reduce_gradient</code>: the gradient of a sum is the incoming gradient broadcast back to the input shape, an <b>EXPAND</b> of 1.0 to (256,256).`,
        [["tinygrad/mixin/gradient.py", 9], ["tinygrad/mixin/gradient.py", 67]]],
      ["WHERE rule (relu)", `WHERE returns <code>(None, c.where(ctx, 0), c.where(0, ctx))</code>: the condition gets nothing, <code>s</code> gets the gradient where <code>c</code> holds,
        and the constant 0 is off the path. The new WHERE <i>reads the forward</i> <code>c = 0 &lt; s</code> (dashed edge). <b>CMPLT</b> itself has no gradient.`,
        [["tinygrad/mixin/gradient.py", 66], ["tinygrad/mixin/gradient.py", 59]]],
      ["REDUCE(ADD) rule (matmul)", `The matmul's sum over k is a REDUCE too: the gradient (256,256) is broadcast to the reduce input's shape (512,256,256).`,
        [["tinygrad/mixin/gradient.py", 9]]],
      ["PERMUTE rule", `A PERMUTE's gradient is the inverse permutation, <code>ctx.permute(argsort(order))</code>: (2,0,1) becomes (1,2,0), giving (256,256,512).`,
        [["tinygrad/mixin/gradient.py", 74]]],
      ["MUL rule", `<code>(ret.src[1]*ctx, ret.src[0]*ctx)</code>. Only the w side is on the path, so one new <b>MUL</b>: <code>x.reshape(256,1,512) * g</code>, again
        <i>reading a forward node</i>. The gradient toward x is never built.`, [["tinygrad/mixin/gradient.py", 65]]],
      ["shape fix: sum to the source shape", `The gradient is (256,256,512), but the MUL source it belongs to is w transposed, (1,256,512). compute_gradient sums the broadcast axis
        away (REDUCE ADD over axis 0) and reshapes. This is where the second matmul's reduction comes from.`, [["tinygrad/mixin/gradient.py", 122], ["tinygrad/mixin/gradient.py", 123]]],
      ["PERMUTE and RESHAPE rules", `Undo w's transpose (<code>(0,2,1)</code>) and its reshape: dL/dw, shape (512,256). That is the whole gradient: 9 new UOps, all lazy.`,
        [["tinygrad/mixin/gradient.py", 74], ["tinygrad/mixin/gradient.py", 70]]],
      ["scheduled into 2 kernels", `<span style="color:#fbbf24">■ kernel 0</span> <code>r_8_4_8_16_4_4_128_4</code> computes x@w in registers and stores only
        <code>c.where(1,0)</code>, a 256×256 mask: a REDUCE feeds a broadcast, so it is realized. <span style="color:#34d399">■ kernel 1</span> <code>r_8_8_8_16_4_4_64_4</code>
        is the MUL + REDUCE, <code>mask.T @ x</code>. <span style="color:#a3e635">■</span> x is read by both. Dashed boxes (movement ops) become index math, not work. The faded
        relu output and loss were never asked for, so they are never computed.`, [["tinygrad/schedule/indexing.py", 220], ["tinygrad/schedule/rangeify.py", 259]]],
    ];
    root.innerHTML = head("Autograd as graph construction: d/dw of (x @ w).relu().sum()",
      `<button class="btn prev">◀</button><button class="btn play">▶ play</button><button class="btn next">▶</button><span class="chip stepno"></span>`) +
      `<svg viewBox="0 0 960 512"></svg><div class="caption"></div>`;
    const svg = root.querySelector("svg"), cap = root.querySelector(".caption");
    const defs = svgEl("defs", {}, svg);
    [["agArrow", "#4b5d73"], ["agArrowP", "#f472b6"]].forEach(([id, c]) => {
      const m = svgEl("marker", { id, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto" }, defs);
      svgEl("path", { d: "M0 0L10 5L0 10z", fill: c }, m);
    });
    svgEl("text", { x: 195, y: 15, "text-anchor": "middle", class: "ag-colhead" }, svg).textContent = "FORWARD (data flows up)";
    svgEl("text", { x: 710, y: 15, "text-anchor": "middle", class: "ag-colhead" }, svg).textContent = "GRADIENT (built top down)";
    const clip = (a, b) => { // point on a's box border toward b
      const hw = (a[2].length > 12 ? W + 26 : W) / 2;
      const dx = b[0] - a[0], dy = b[1] - a[1], t = Math.min(dx ? hw / Math.abs(dx) : 1e9, dy ? (H / 2) / Math.abs(dy) : 1e9);
      return [a[0] + dx * t, a[1] + dy * t];
    };
    const edgeEls = E.map(([s, d, st, kind]) => {
      const a = N[s], b = N[d];
      let dstr;
      if (s === "f_rx" && d === "g6") dstr = `M90 ${404 + H / 2} V500 H480 Q505 500 505 475 V360 Q505 350 ${610 - W / 2} 350`;
      else { const p = clip(a, b), q = clip(b, a); dstr = `M${p[0]} ${p[1]} L${q[0]} ${q[1]}`; }
      const e = svgEl("path", { d: dstr, class: `ag-edge ${kind || ""}` }, svg);
      if (kind) e.setAttribute("marker-end", "url(#agArrowP)");
      e.dataset.step = st;
      return e;
    });
    const nodeEls = {};
    for (const [id, [x, y, label, sub, kind, st]] of Object.entries(N)) {
      const g = svgEl("g", { class: `ag-node ${kind}` }, svg);
      const w = label.length > 12 ? W + 26 : W;
      svgEl("rect", { x: x - w / 2, y: y - H / 2, width: w, height: H, rx: 7 }, g);
      svgEl("text", { x, y: y - 3 }, g).textContent = label;
      svgEl("text", { x, y: y + 12, class: "s" }, g).textContent = sub;
      g.dataset.step = st;
      nodeEls[id] = g;
    }
    const tags = [
      svgEl("text", { x: 694, y: 168, "text-anchor": "start", class: "ag-tag", fill: "#fbbf24" }, svg),
      svgEl("text", { x: 810, y: 497, "text-anchor": "middle", class: "ag-tag", fill: "#34d399" }, svg),
    ];
    tags[0].textContent = "← STORE: mask (256×256)";
    tags[1].textContent = "STORE → out (stored as 256×512)";
    let cur = 0, timer = null;
    const playBtn = root.querySelector(".play");
    function show(i) {
      cur = Math.max(0, Math.min(STEPS.length - 1, i));
      const fin = cur === STEPS.length - 1;
      for (const [id, g] of Object.entries(nodeEls)) {
        const st = +g.dataset.step;
        g.classList.toggle("hidden", st > cur);
        g.classList.toggle("new", st === cur && st > 0 && !fin);
        g.classList.toggle("k0", fin && K0.has(id));
        g.classList.toggle("k1", fin && id.startsWith("g") && !K0.has(id));
        g.classList.toggle("k01", fin && K01.has(id));
        g.classList.toggle("dead", fin && DEAD.has(id));
        g.classList.toggle("mop", fin && MOPS.has(id));
      }
      edgeEls.forEach((e) => e.classList.toggle("hidden", +e.dataset.step > cur));
      tags.forEach((t) => (t.style.opacity = fin ? 1 : 0));
      const [t, body, srcs] = STEPS[cur];
      cap.innerHTML = `<b class="rule">${esc(t)}</b>: ${body}<span class="srcs">source: ${srcs.map(([p, l]) => SRC(p, l)).join(" · ")}</span>`;
      root.querySelector(".stepno").textContent = `step ${cur} / ${STEPS.length - 1}`;
    }
    const stop = () => { if (timer) clearInterval(timer); timer = null; playBtn.textContent = "▶ play"; };
    root.querySelector(".prev").onclick = () => { stop(); show(cur - 1); };
    root.querySelector(".next").onclick = () => { stop(); show(cur + 1); };
    playBtn.onclick = () => {
      if (timer) return stop();
      if (cur >= STEPS.length - 1) show(0);
      playBtn.textContent = "❚❚ pause";
      timer = setInterval(() => { if (cur >= STEPS.length - 1) stop(); else show(cur + 1); }, 2600);
    };
    show(0);
    B.whenVisible(root, () => { if (cur === 0 && !timer) playBtn.click(); });
  })();

  // =====================================================================================
  // 2. Annotated DEBUG=2 line (first JIT call from the capture)
  // =====================================================================================
  const J = B.get("jit/default");
  (function dbgLine() {
    const root = document.getElementById("dbg-line");
    if (!root) return;
    const line = J.calls[0].log.split("\n").find((l) => l.startsWith("***")) || "";
    const re = /^(\*\*\* \S+\s+\d+)(\s+)(\S+)(\s+)(arg\s+\d+)(\s+)(mem\s+\S+ GB)(\s+)(tm\s+\S+\/\s+\S+)(\s+)(\(\s*\d+ GFLOPS)(\s+)(\d+\|\d+\s+GB\/s\))\s*$/;
    const m = re.exec(line);
    const EXPL = [
      `<b>device and running kernel count.</b> <code>***</code>, the device name padded to 7 characters, then <code>GlobalCounters.kernel_count</code> (reset by the harness before each call).
       In a terminal this part is colored: green on the first run of a kernel, magenta when the JIT executes it. ${SRC("tinygrad/engine/realize.py", 86)}`,
      `<b>kernel name.</b> <code>r_</code> = it has a reduction (<code>E_</code> = elementwise); the numbers are the sizes of the kernel's axes after the opts, colored by axis type in a terminal.
       Copies show as <code>copy N, DST &lt;- SRC</code>, CUDA graphs as <code>batched N</code>. ${SRC("tinygrad/engine/realize.py", 35)}`,
      `<b>number of buffers</b> passed to the kernel: output, x, w. ${SRC("tinygrad/engine/realize.py", 93)}`,
      `<b>device memory in use</b> by tinygrad (<code>GlobalCounters.mem_used</code>), in GB. These buffers total about 1.5 MB, so it rounds to 0.00. ${SRC("tinygrad/engine/realize.py", 93)}`,
      `<b>kernel time / running total.</b> Measured with two CUDA events around <code>cuLaunchKernel</code>, because DEBUG≥2 forces <code>wait=True</code>. Yellow above 10 ms.
       ${SRC("tinygrad/runtime/ops_cuda.py", 26)} ${SRC("tinygrad/engine/realize.py", 318)}`,
      `<b>GFLOPS</b> = the kernel's estimated op count / time. The estimate is computed from the uops at compile time, not by counters. ${SRC("tinygrad/engine/realize.py", 88)}`,
      `<b>A|B GB/s.</b> A = <code>estimates.mem</code> (unique bytes the kernel must touch: x, w, out ≈ 1.5 MB) / time. B = <code>estimates.lds</code> (every load and store the code issues,
       including re-reads that hit cache) / time. A far below B means heavy reuse, or wasted re-reads. ${SRC("tinygrad/engine/realize.py", 90)}`,
    ];
    root.innerHTML = head("A DEBUG=2 line from the L4 (TinyJit call 0)", `<span class="chip">click a part</span>`) +
      `<div class="dbg"><pre></pre></div><div class="dbg-expl">Click any highlighted part of the line.</div>`;
    const pre = root.querySelector("pre"), ex = root.querySelector(".dbg-expl");
    if (!m) { pre.textContent = line; return; }
    let html = "", k = 0;
    for (let g = 1; g < m.length; g++) html += g % 2 ? `<span class="seg" data-i="${k++}">${esc(m[g])}</span>` : esc(m[g]);
    pre.innerHTML = html;
    const segs = [...pre.querySelectorAll(".seg")];
    const pick = (i) => { segs.forEach((s, j) => s.classList.toggle("on", j === i)); ex.innerHTML = EXPL[i]; };
    segs.forEach((s) => (s.onclick = () => pick(+s.dataset.i)));
    pick(4);
  })();

  // =====================================================================================
  // 3. The four TinyJit calls (table straight from the capture)
  // =====================================================================================
  (function jitCalls() {
    const root = document.getElementById("jit-calls");
    if (!root) return;
    const what = ["cnt 0: runs normally", "cnt 1: captures, then runs the capture", "cnt 2: replay", "cnt 3: replay"];
    root.innerHTML = `<div class="tablewrap"><table class="data"><tr><th>call</th><th>what happens</th><th>jit_cnt_after</th><th>kernels</th><th>y[0]</th><th>DEBUG=2 output (captured on the L4)</th></tr>` +
      J.calls.map((c) => `<tr><td>${c.call}</td><td class="what">${what[c.call] || ""}</td><td>${c.jit_cnt_after}</td><td>${c.kernel_count}</td><td class="mono" style="white-space:nowrap">${c.y0.toFixed(2)}</td>` +
        `<td class="mono">${esc(c.log.replace(/\s+$/, ""))}</td></tr>`).join("") + `</table></div>`;
  })();

  // =====================================================================================
  // 4. TinyJit timeline: 4 calls, CPU lanes vs GPU lane
  // =====================================================================================
  (function timeline() {
    const root = document.getElementById("jit-timeline");
    if (!root) return;
    const times = J.calls.map((c) => { const m = /tm\s+([\d.]+)us/.exec(c.log); return m ? m[1] + " µs" : ""; });
    const LANES = [["Python: build Tensor graph", "var(--s-tensor)"], ["Schedule", "var(--s-schedule)"], ["Codegen + compile", "var(--s-codegen)"],
      ["Launch (CPU → driver)", "var(--s-compile)"], ["GPU", "var(--s-run)"]];
    const X0 = 175, CW = [290, 250, 100, 100], GAP = 14, LY = 52, LH = 40;
    const cx = CW.reduce((acc, w, i) => (acc.push(i ? acc[i - 1] + CW[i - 1] + GAP : X0), acc), []);
    // blocks: [call, lane, start fraction, width fraction, label]
    const MODES = {
      one: {
        title: "this capture: 1 kernel per call",
        blocks: [
          [0, 0, 0, .18, "step(x)"], [0, 1, .18, .2, "schedule"], [0, 2, .38, .22, "compile"], [0, 3, .6, .04, ""], [0, 4, .64, .36, times[0]],
          [1, 0, 0, .2, "step(x)"], [1, 1, .2, .2, "→ JIT"], [1, 2, .4, .25, "jit_lower"], [1, 3, .65, .04, ""], [1, 4, .69, .31, times[1]],
          [2, 0, 0, .15, ""], [2, 3, .15, .1, ""], [2, 4, .25, .75, times[2]],
          [3, 0, 0, .15, ""], [3, 3, .15, .1, ""], [3, 4, .25, .75, times[3]],
        ],
        text: [
          `<b>Call 0, cnt 0.</b> <code>step(x)</code> runs as plain tinygrad: Python builds the UOp graph, the scheduler makes a 1-kernel LINEAR, codegen and NVRTC compile
           <code>r_256_16_512_32</code>, <code>cuModuleLoadData</code> loads it, and one <code>cuLaunchKernel</code> runs it (${esc(times[0])}).`,
          `<b>Call 1, cnt 1: capture.</b> <code>step(x)</code> runs again and the graph is scheduled again, but the scheduler hands the LINEAR to the JIT instead of running it.
           <code>jit_lower</code> swaps x for a PARAM, memory-plans, compiles (an in-memory cache hit) and tries to graph-batch (one kernel: no graph). Then the
           CapturedJit runs it (${esc(times[1])}).`,
          `<b>Call 2, cnt 2: replay.</b> The Python function is not called. tinygrad checks the input's shape, dtype and device, swaps in the new x pointer, and calls
           <code>run_linear(jit=True)</code>: one <code>cuLaunchKernel</code> (${esc(times[2])}).`,
          `<b>Call 3, cnt 3: replay.</b> Same as call 2 (${esc(times[3])}). Calls 2 and 3 do no Python graph building, no scheduling and no compiling.`,
        ],
      },
      many: {
        title: "illustrative: a step with 3 kernels (not measured)",
        blocks: [
          [0, 0, 0, .19, "step(x)"], [0, 1, .19, .2, "schedule"], [0, 2, .39, .23, "compile"],
          [0, 3, .62, .04, ""], [0, 4, .66, .1, "k0"], [0, 3, .76, .04, ""], [0, 4, .8, .09, "k1"], [0, 3, .89, .03, ""], [0, 4, .92, .08, "k2"],
          [1, 0, 0, .2, "step(x)"], [1, 1, .2, .18, "→ JIT"], [1, 2, .38, .3, "lower+graph"], [1, 3, .68, .05, ""],
          [1, 4, .73, .09, "k0"], [1, 4, .82, .09, "k1"], [1, 4, .91, .09, "k2"],
          [2, 0, 0, .15, ""], [2, 3, .15, .12, ""], [2, 4, .27, .25, "k0"], [2, 4, .52, .24, "k1"], [2, 4, .76, .24, "k2"],
          [3, 0, 0, .15, ""], [3, 3, .15, .12, ""], [3, 4, .27, .25, "k0"], [3, 4, .52, .24, "k1"], [3, 4, .76, .24, "k2"],
        ],
        text: [
          `<b>Call 0.</b> Three kernels are compiled and launched one by one: three <code>cuLaunchKernel</code>s, with Python dispatching between them.`,
          `<b>Call 1: capture.</b> <code>graph_split_rewrite</code> sees three consecutive PROGRAMs and makes one batch (DEBUG=2 prints <code>JIT GRAPHing batch with 3 kernels</code>).
           At first execution <code>CUDAGraph</code> creates a CUgraph with three kernel nodes, instantiates it, and launches it with one <code>cuGraphLaunch</code>.`,
          `<b>Call 2: replay.</b> Patch the input pointers (and any Variable values) into the instantiated graph's nodes, then one <code>cuGraphLaunch</code>. The GPU runs
           k0, k1 and k2 back to back without the CPU in between. DEBUG=2 shows a single <code>batched 3</code> line.`,
          `<b>Call 3: replay.</b> Same. For a 300-kernel training step, that is 300 kernels from a few graph launches (batches of 32, then 64, 128, …).`,
        ],
      },
    };
    root.innerHTML = head("TinyJit over four calls",
      `<div class="tabs"><button class="btn on" data-m="one">this capture (1 kernel)</button><button class="btn" data-m="many">if the step had 3 kernels</button></div>` +
      `<button class="btn prev">◀</button><button class="btn play">▶ play</button><button class="btn next">▶</button>`) +
      `<svg viewBox="0 0 960 262"></svg><div class="caption"></div><div class="tl-log"></div>`;
    const svg = root.querySelector("svg"), cap = root.querySelector(".caption"), logEl = root.querySelector(".tl-log");
    LANES.forEach(([name, col], i) => {
      const y = LY + i * LH;
      svgEl("rect", { x: X0 - 6, y: y, width: 960 - X0, height: LH - 6, class: "tl-lane" }, svg);
      svgEl("rect", { x: 8, y: y + 10, width: 10, height: 10, fill: col, rx: 2 }, svg);
      svgEl("text", { x: 24, y: y + 20, class: "tl-lanelabel" }, svg).textContent = name;
    });
    svgEl("line", { x1: X0 - 8, x2: 952, y1: LY + 4 * LH - 9, y2: LY + 4 * LH - 9, stroke: "#33445a", "stroke-dasharray": "2 3" }, svg);
    const heads = ["call 0: run", "call 1: capture", "call 2: replay", "call 3: replay"];
    heads.forEach((h, i) => {
      svgEl("text", { x: cx[i], y: 22, class: "tl-callhead" }, svg).textContent = h;
      svgEl("text", { x: cx[i], y: 38, class: "tl-callsub" }, svg).textContent = `cnt ${i} → ${J.calls[i].jit_cnt_after}`;
      if (i) svgEl("line", { x1: cx[i] - GAP / 2, x2: cx[i] - GAP / 2, y1: 8, y2: LY + 5 * LH - 6, stroke: "#2a3441" }, svg);
    });
    const foot = svgEl("text", { x: 952, y: 256, "text-anchor": "end", class: "tl-callsub" }, svg);
    const layer = svgEl("g", {}, svg);
    const cursor = svgEl("line", { x1: X0, x2: X0, y1: LY - 4, y2: LY + 5 * LH - 6, class: "tl-cursor" }, svg);
    let mode = "one", blocks = [], pos = 0, raf = null, curCall = -1;
    const END = cx[3] + CW[3];
    function build() {
      layer.innerHTML = "";
      foot.textContent = mode === "one" ? "CPU lanes not to scale · GPU labels = measured L4 kernel time" : "illustrative only: nothing in this mode is measured";
      blocks = MODES[mode].blocks.map(([c, lane, s, w, label]) => {
        const x = cx[c] + s * CW[c], bw = Math.max(3, w * CW[c] - 2), y = LY + lane * LH + 3;
        const g = svgEl("g", { class: "tl-block off" }, layer);
        svgEl("rect", { x, y, width: bw, height: LH - 12, fill: LANES[lane][1] }, g);
        if (label && label.length * 6.2 + 6 <= bw) svgEl("text", { x: x + 4, y: y + 18 }, g).textContent = label;
        return { g, x0: x };
      });
    }
    function setPos(p) {
      pos = Math.max(X0, Math.min(END, p));
      cursor.setAttribute("x1", pos); cursor.setAttribute("x2", pos);
      blocks.forEach((b) => b.g.classList.toggle("off", b.x0 > pos));
      let c = 0; for (let i = 0; i < 4; i++) if (pos >= cx[i]) c = i;
      if (pos <= X0) c = 0;
      if (c !== curCall) {
        curCall = c;
        cap.innerHTML = MODES[mode].text[c];
        logEl.innerHTML = mode === "one"
          ? B.codeBlock(J.calls[c].log.replace(/\s+$/, ""), "text", { note: `DEBUG=2 output of call ${c}, captured on the L4 (ANSI colors stripped).` })
          : `<div class="code-note">Illustrative mode: no capture exists for a multi-kernel step; the log lines named in the text come from the source.</div>`;
      }
    }
    const playBtn = root.querySelector(".play");
    const stop = () => { if (raf) cancelAnimationFrame(raf); raf = null; playBtn.textContent = "▶ play"; };
    function play() {
      if (raf) return stop();
      if (pos >= END) setPos(X0);
      playBtn.textContent = "❚❚ pause";
      let last = null;
      const tick = (t) => {
        if (last != null) setPos(pos + (t - last) * 0.11);
        last = t;
        if (pos >= END) return stop();
        raf = requestAnimationFrame(tick);
      };
      raf = requestAnimationFrame(tick);
    }
    playBtn.onclick = play;
    root.querySelector(".next").onclick = () => { stop(); const n = Math.min(3, pos <= X0 ? 0 : curCall + 1); setPos(pos <= X0 ? cx[0] + CW[0] : cx[n] + CW[n]); };
    root.querySelector(".prev").onclick = () => { stop(); const n = curCall - 1; setPos(n < 0 ? X0 : cx[n] + CW[n]); };
    root.querySelectorAll("[data-m]").forEach((b) => (b.onclick = () => {
      stop(); mode = b.dataset.m;
      root.querySelectorAll("[data-m]").forEach((x) => x.classList.toggle("on", x === b));
      build(); curCall = -1; setPos(END);
    }));
    build(); setPos(X0); curCall = -1; setPos(X0);
    B.whenVisible(root, () => { if (pos <= X0) play(); });
  })();

  // =====================================================================================
  // 5. Symbolic-size logs (DEBUG=4, captured on the L4)
  // =====================================================================================
  (function symbolic() {
    const root = document.getElementById("sym-log");
    if (!root) return;
    root.innerHTML = J.symbolic.map((s) => `<div class="widget"><div class="widget-head"><span class="title">n = ${s.n}: (big[:n] * 2).sum() at DEBUG=4 on the L4</span>
      <span class="chip ${Math.abs(s.sum - s.expected) < 1e-3 ? "ok" : "bad"}">sum ${s.sum} (numpy ${s.expected})</span></div>
      ${B.codeBlock(s.log.replace(/\s+$/, ""), "c", { highlight: "r_n|data2_|cache hit|CACHE MISS" })}</div>`).join("");
  })();
});
