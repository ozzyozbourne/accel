/* Chapter 3: Scheduling and fusion.
 *   #pass-map       the nesting of realize -> callify -> schedule, with every captured pass, its time and node count
 *   #rangeify-anim  softmax: ranges are assigned node by node, STAGEs appear and disappear, kernels are cut out
 *   #traffic        memory traffic of eager (one kernel per op) vs fused kernels, bytes computed from the shapes
 */
BOOK.ready((B) => {
  const esc = B.esc;

  // ---------- tweening graph scene: nodes, edges, labeled boxes ----------
  function scene(svg) {
    let cur = { nodes: new Map(), edges: new Map(), boxes: new Map() }, raf = null;
    const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);
    const lerp = (a, b, t) => a + (b - a) * t;
    function border(a, b) {
      const dx = b.x - a.x, dy = b.y - a.y;
      if (!dx && !dy) return { x: a.x, y: a.y };
      const s = Math.min((a.w / 2) / Math.abs(dx || 1e-9), (a.h / 2) / Math.abs(dy || 1e-9));
      return { x: a.x + dx * s, y: a.y + dy * s };
    }
    function draw(nodes, edges, boxes) {
      let h = "";
      for (const b of boxes) if (b.o > 0.02) h += `<g class="box ${b.cls || ""}" style="opacity:${b.o}"><rect x="${b.x}" y="${b.y}" width="${b.w}" height="${b.h}" rx="10"/><text x="${b.x + 10}" y="${b.y + 16}">${esc(b.label || "")}</text>${b.sub ? `<text class="bsub" x="${b.x + 10}" y="${b.y + b.h - 8}">${esc(b.sub)}</text>` : ""}</g>`;
      for (const e of edges) {
        const a = nodes.get(e.from), b = nodes.get(e.to);
        if (!a || !b) continue;
        const op = Math.min(e.o, a.o, b.o);
        if (op < 0.02) continue;
        if (e.via) { // routed edge: a polyline through fixed waypoints
          const p = border(a, { x: e.via[0][0], y: e.via[0][1] }), q = border(b, { x: e.via.at(-1)[0], y: e.via.at(-1)[1] });
          h += `<polyline points="${[[p.x, p.y], ...e.via, [q.x, q.y]].join(" ")}" class="e ${e.cls || ""}" style="opacity:${op}" marker-end="url(#${svg.id}-arr)"/>`;
          continue;
        }
        const p = border(a, b), q = border(b, a);
        h += `<line x1="${p.x}" y1="${p.y}" x2="${q.x}" y2="${q.y}" class="e ${e.cls || ""}" style="opacity:${op}" marker-end="url(#${svg.id}-arr)"/>`;
      }
      for (const n of nodes.values()) {
        if (n.o < 0.02) continue;
        h += `<g class="n ${n.cls || ""}" style="opacity:${n.o}"><rect x="${n.x - n.w / 2}" y="${n.y - n.h / 2}" width="${n.w}" height="${n.h}" rx="${n.cls && n.cls.includes("buf") ? 3 : 7}"/>` +
          `<text x="${n.x}" y="${n.y + (n.sub ? -2 : 4)}">${esc(n.label)}</text>` + (n.sub ? `<text class="sub" x="${n.x}" y="${n.y + 11}">${esc(n.sub)}</text>` : "") +
          (n.chip ? `<text class="chip2" x="${n.x}" y="${n.y + n.h / 2 + 14}">${esc(n.chip)}</text>` : "") + `</g>`;
      }
      svg.querySelector(".layer").innerHTML = h;
    }
    svg.innerHTML = `<defs><marker id="${svg.id}-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="#6b7c93"/></marker></defs><g class="layer"></g>`;
    return {
      set(state, animate = true) {
        const nxt = {
          nodes: new Map(state.nodes.map((n) => [n.id, { w: 76, h: 32, ...n, o: n.o ?? 1 }])),
          edges: new Map(state.edges.map((e) => [e.from + ">" + e.to, { ...e, o: 1 }])),
          boxes: new Map((state.boxes || []).map((b) => [b.id, { ...b, o: b.o ?? 1 }])),
        };
        if (raf) cancelAnimationFrame(raf);
        const prev = cur, t0 = performance.now(), D = animate ? 750 : 0;
        const mix = (pm, nm, t, geo) => {
          const out = [];
          for (const id of new Set([...pm.keys(), ...nm.keys()])) {
            const a = pm.get(id), b = nm.get(id), s = a || b, f = b || a, r = { ...f, o: lerp(a ? a.o : 0, b ? b.o : 0, t) };
            for (const g of geo) r[g] = lerp(s[g], f[g], t);
            if (t < 0.5 && a && !b) r.cls = a.cls;
            out.push([id, r]);
          }
          return out;
        };
        const frame = (now) => {
          const t = D ? ease(Math.min(1, (now - t0) / D)) : 1;
          const ns = new Map(mix(prev.nodes, nxt.nodes, t, ["x", "y", "w"]));
          const es = mix(prev.edges, nxt.edges, t, []).map((x) => x[1]);
          const bs = mix(prev.boxes, nxt.boxes, t, ["x", "y", "w", "h"]).map((x) => x[1]);
          draw(ns, es, bs);
          if (t < 1) raf = requestAnimationFrame(frame); else { raf = null; cur = nxt; }
        };
        if (D) raf = requestAnimationFrame(frame); else frame(t0);
      },
    };
  }
  function stepper(root, n, show, ms = 2200) {
    let cur = 0, timer = null;
    const go = (i, anim = true) => { cur = Math.max(0, Math.min(n - 1, i)); show(cur, anim); root.querySelector(".stepc").textContent = `step ${cur + 1}/${n}`; };
    root.querySelector(".prev").onclick = () => go(cur - 1);
    root.querySelector(".next").onclick = () => go(cur + 1);
    root.querySelector(".play").onclick = (e) => {
      if (timer) { clearInterval(timer); timer = null; e.target.textContent = "▶ play"; return; }
      if (cur >= n - 1) go(0);
      e.target.textContent = "❚❚ pause";
      timer = setInterval(() => { if (cur >= n - 1) { clearInterval(timer); timer = null; e.target.textContent = "▶ play"; } else go(cur + 1); }, ms);
    };
    go(0, false);
  }

  // ---------- 1. pass map ----------
  (function passMap() {
    const root = document.getElementById("pass-map");
    if (!root) return;
    const d = B.get("softmax/default"), ps = d.passes.filter((p) => p.stage === "tensor" || p.stage === "schedule");
    const info = (name) => {
      const hits = ps.map((p, i) => ({ ...p, i })).filter((p) => p.name === name);
      if (!hits.length) return null;
      return { n: hits.length, ms: hits.reduce((s, p) => s + p.ms, 0), nodes: hits[hits.length - 1].nodes, changed: hits.some((p) => p.changed), order: hits.map((p) => d.passes.indexOf(ps[p.i]) + 1) };
    };
    const maxMs = Math.max(...ps.map((p) => p.ms));
    const T = [
      ["fn", "Tensor.realize()", "tensor.py#L420", [
        ["fn", "Tensor.linear_with_vars()", "tensor.py#L404", [
          ["grp", "transform_to_call: “Callify”", "tensor.py#L224", [
            ["pass", "number the uops", "tensor.py#L233"], ["pass", "early transform tensor graph", "tensor.py#L236"],
            ["pass", "finalize call", "tensor.py#L239"], ["pass", "replace bufs", "tensor.py#L240"]]],
          ["grp", "create_linear_with_vars: “Schedule”", "schedule/__init__.py#L181", [
            ["pass", "schedule to linear", "schedule/__init__.py#L183", [
              ["fn", "lower_sink_to_linear (schedule cache lookup)", "schedule/__init__.py#L119", [
                ["fn", "get_kernel_graph", "schedule/rangeify.py#L585", [
                  ["pass", "multi_pm", "schedule/rangeify.py#L587"], ["pass", "earliest rewrites", "schedule/rangeify.py#L589"],
                  ["pass", "convert copy to store", "schedule/rangeify.py#L590"],
                  ["fn", "run_rangeify", "schedule/indexing.py#L188", [
                    ["pass", "get realize", "schedule/indexing.py#L193"], ["note", "consumer map + reverse-toposort walk: assigns ranges, decides what to realize (not a rewrite)", "schedule/indexing.py#L205"],
                    ["pass", "apply rangeify", "schedule/indexing.py#L320"], ["pass", "add device to deviceless", "schedule/indexing.py#L322"]]],
                  ["pass", "symbolic+reduce_collapse+debuf", "schedule/rangeify.py#L596"], ["pass", "limit buffers", "schedule/rangeify.py#L600"],
                  ["pass", "stage to store", "schedule/rangeify.py#L606"],
                  ["pass", "split kernels", "schedule/rangeify.py#L607", [["pass", "kernel split", "schedule/rangeify.py#L556"]]],
                  ["pass", "remove indexing from call args", "schedule/rangeify.py#L608"]]],
                ["note", "create_schedule: toposort of kernels by their AFTER dependencies (not a rewrite)", "schedule/__init__.py#L28"]]]]],
            ["pass", "resolve linear call", "schedule/__init__.py#L186", [["pass", "params to buffers", "schedule/__init__.py#L101"]]],
            ["pass", "create COPY kernels for SDMA", "schedule/__init__.py#L189"]]]]],
        ["fn", "run_linear: codegen + launch for each kernel (later chapters)", "engine/realize.py#L315", []]]],
    ];
    const link = (src) => { const [p, l] = src.split("#"); return `<a class="src" href="https://github.com/tinygrad/tinygrad/blob/${B.commit}/tinygrad/${p}#${l}" target="_blank">${p}:${l.slice(1)}</a>`; };
    function row(item) {
      const [kind, name, src, kids] = item;
      if (kind === "pass") {
        const s = info(name);
        const stats = s ? `<span class="pm-bar"><i style="width:${Math.max(1, 100 * s.ms / maxMs)}%"></i></span><span class="pm-ms">${s.ms.toFixed(1)} ms</span>` +
          `<span class="pm-nodes">${s.nodes} UOps</span><span class="chip ${s.changed ? "ok" : ""}">${s.changed ? "changed" : "no change"}</span>` +
          `<span class="pm-ord" title="position in the captured pass list (completion order)">#${s.order.join(",#")}</span>` : `<span class="pm-ms">(not in capture)</span>`;
        return `<div class="pm-item pass" data-name="${esc(name)}"><div class="pm-line"><b class="mono">${esc(name)}${s && s.n > 1 ? ` ×${s.n}` : ""}</b>${link(src)}${stats}</div>${kids ? `<div class="pm-kids">${kids.map(row).join("")}</div>` : ""}</div>`;
      }
      if (kind === "note") return `<div class="pm-item note"><div class="pm-line"><i>${esc(name)}</i>${link(src)}</div></div>`;
      return `<div class="pm-item ${kind}"><div class="pm-line"><b>${esc(name)}</b>${link(src)}</div><div class="pm-kids">${(kids || []).map(row).join("")}</div></div>`;
    }
    root.classList.add("widget");
    root.innerHTML = `<div class="widget-head"><span class="title">softmax/default: who calls which pass (click a pass)</span><span class="chip">times from the instrumented capture run; a parent includes its children</span></div>
      <div class="widget-body pm-tree">${T.map(row).join("")}</div><div class="pm-desc"></div>`;
    root.querySelectorAll(".pm-item.pass > .pm-line").forEach((l) => (l.onclick = (e) => {
      if (e.target.tagName === "A") return;
      root.querySelectorAll(".pm-line.sel").forEach((x) => x.classList.remove("sel"));
      l.classList.add("sel");
      const name = l.parentNode.dataset.name;
      root.querySelector(".pm-desc").innerHTML = `<b class="mono">${esc(name)}</b>: ${esc(B.passDesc[name] || "")}`;
    }));
    root.querySelector(".pm-desc").innerHTML = "Passes are listed where they are called. The <b>#</b> numbers are their positions in the captured list, which records a pass when it <i>finishes</i>: a nested pass is recorded before the pass that contains it.";
  })();

  // ---------- 2. softmax: rangeify, STAGEs, kernels ----------
  (function rangeifyAnim() {
    const root = document.getElementById("rangeify-anim");
    if (!root) return;
    const d = B.get("softmax/default"), K = d.kernels;
    const N = {
      in: [60, 215, "PARAM 1", "input x"], rx: [160, 215, "RESHAPE", "1024×1024"], p1: [255, 90, "PERMUTE", "(1,0)"], mx: [355, 90, "REDUCE", "MAX"],
      r1: [455, 90, "RESHAPE", "1024×1"], ng: [555, 90, "MUL", "× -1"], ad: [610, 215, "ADD", "x - max"], m2: [705, 215, "MUL", "× log2(e)"], ex: [800, 215, "EXP2", ""],
      p2: [840, 340, "PERMUTE", "(1,0)"], sm: [935, 340, "REDUCE", "ADD"], r2: [1030, 340, "RESHAPE", "1024×1"], rc: [1120, 340, "RECIP", "1/sum"],
      fm: [1030, 215, "MUL", "e × 1/sum"], st: [1120, 215, "STORE", ""], out: [1120, 90, "PARAM 0", "output"],
    };
    const MOV = ["rx", "p1", "r1", "p2", "r2"];
    const E0 = [["in", "rx"], ["rx", "p1"], ["p1", "mx"], ["mx", "r1"], ["r1", "ng"], ["rx", "ad"], ["ng", "ad"], ["ad", "m2"], ["m2", "ex"], ["ex", "p2"], ["p2", "sm"], ["sm", "r2"], ["r2", "rc"], ["ex", "fm"], ["rc", "fm"], ["fm", "st"], ["out", "st"]];
    const E1 = [["in", "mx"], ["in", "ad"], ["mx", "ng"], ["ng", "ad"], ["ad", "m2"], ["m2", "ex"], ["ex", "sm"], ["sm", "rc"], ["ex", "fm"], ["rc", "fm"], ["fm", "st"], ["out", "st"]];
    const E2 = [["in", "mx"], ["in", "ad"], ["mx", "b0"], ["b0", "ng"], ["ng", "ad"], ["ad", "m2"], ["m2", "ex"], ["ex", "sm"], ["sm", "b1"], ["b1", "rc"], ["ex", "fm"], ["rc", "fm"], ["fm", "st"], ["out", "st"]];
    const CH = { st: "[r0,r1]", fm: "[r0,r1]", rc: "[r0,0]", r2: "[r0]→[r0,0]", sm: "[r3,r2]→[r2]", p2: "[r2,r3]", ex: "[r4,r5]", m2: "[r4,r5]", ad: "[r4,r5]",
      ng: "[r4,0]", r1: "[r4]→[r4,0]", mx: "[r7,r6]→[r6]", p1: "[r6,r7]", rx: "[r8,r9]", in: "r8*1024+r9", out: "r0*1024+r1" };
    const ORDER = [["st", "out"], ["fm", "rc", "r2"], ["sm", "p2"], ["ex"], ["m2", "ad", "ng", "r1", "mx", "p1"], ["rx", "in"]];
    const REAL = [["st"], [], ["sm"], ["ex"], ["mx"], ["rx"]];
    const TRACE = {
      st: "***  0 STORE      (1024, 1024)  [r0][r1]", fm: "     1 MUL        (1024, 1024)  [r0][r1]", rc: "     1 RECIPROCAL (1024, 1)     [r0][0]",
      r2: "     1 RESHAPE    (1024, 1)     [r0] -> [r0][0]", sm: "***  1 REDUCE     (1024,)       [r3][r2] -> [r2]", p2: "     1 PERMUTE    (1024, 1024)  [r2 -> r3][r3 -> r2]",
      ex: "***  2 EXP2       (1024, 1024)  [r4][r5]", m2: "     1 MUL        (1024, 1024)  [r4][r5]", ad: "     1 ADD        (1024, 1024)  [r4][r5]",
      ng: "     1 MUL        (1024, 1)     [r4][0]", r1: "     1 RESHAPE    (1024, 1)     [r4] -> [r4][0]", mx: "***  1 REDUCE     (1024,)       [r7][r6] -> [r6]",
      p1: "     1 PERMUTE    (1024, 1024)  [r6 -> r7][r7 -> r6]", rx: "***  2 RESHAPE    (1024, 1024)  [(r8*1024+r9)] -> [r8][r9]", in: "     1 PARAM      (1048576,)    [(r8*1024+r9)]",
    };
    const kname = (i) => `kernel ${i}: ${K[i].name}`, ksub = (i) => `${B.fmtUs(K[i].time_us)} on the L4 · ${K[i].regs} registers · grid ${K[i].global_size.join("×")}`;
    const LOOPS = {
      7: `# after apply rangeify: movement ops are gone, every value lives in loops (pseudocode)
for r8, r9:  stage_x[r8, r9]  = x[r8*1024 + r9]                                   # STAGE
for r6:      stage_mx[r6]     = MAX over r7 of stage_x[r6, r7]                    # STAGE
for r4, r5:  stage_e[r4, r5]  = exp2((stage_x[r4, r5] - stage_mx[r4]) * log2(e))  # STAGE
for r2:      stage_sum[r2]    = SUM over r3 of stage_e[r2, r3]                    # STAGE
for r0, r1:  out[r0*1024+r1]  = stage_e[r0, r1] * (1 / stage_sum[r0])             # STORE`,
      8: `# after symbolic+reduce_collapse+debuf: two STAGEs were cheaper to recompute
for r6:      stage_mx[r6]    = MAX over r7 of x[r6*1024 + r7]
for r2:      stage_sum[r2]   = SUM over r3 of exp2((x[r2*1024 + r3] - stage_mx[r2]) * log2(e))
for r0, r1:  out[r0*1024+r1] = exp2((x[r0*1024 + r1] - stage_mx[r0]) * log2(e)) * (1 / stage_sum[r0])`,
      9: `# after stage to store: each STAGE is a BUFFER + STORE + END, read through AFTER
buf0 = BUFFER(1024 floats)
for r6:      buf0[r6] = MAX over r7 of x[r6*1024 + r7]                    # END(r6)
buf1 = BUFFER(1024 floats)
for r2:      buf1[r2] = SUM over r3 of exp2((x[r2*1024+r3] - buf0[r2]) * log2(e))   # reads buf0.after(...)
for r0, r1:  out[r0*1024+r1] = exp2((x[...] - buf0[r0]) * log2(e)) * (1 / buf1[r0]) # reads buf1.after(...)`,
      10: `# after split kernels: one SINK per closed STORE/END, buffers become PARAMs, ranges renumbered
kernel 0:  for r1:     data0[r1] = MAX over r0 of data1[r1*1024 + r0]
kernel 1:  for r1:     data0[r1] = SUM over r0 of exp2((data1[r1*1024+r0] - data2[r1]) * log2(e))
kernel 2:  for r0, r1: data0[r0*1024+r1] = exp2((data1[r0*1024+r1] - data2[r0]) * log2(e)) * (1 / data3[r0])`,
    };
    LOOPS[11] = LOOPS[10];
    const steps = [
      "The tensor graph after callify: whole-tensor ops, no loops. The scheduler walks it in reverse toposort order, from the output back to the input, and gives every node a tuple of RANGEs: loop variables, one per axis.",
      "STORE is always realized (it is in the realize map), so it gets <b>new ranges r0, r1</b>: one loop over rows, one over columns. The output PARAM will be indexed by <code>r0*1024+r1</code>.",
      "Single consumers inherit their consumer's ranges, so the final MUL fuses: <code>[r0, r1]</code>. RECIPROCAL has shape 1024×1, so for it the column range is broadcast to the <b>constant 0</b>. The RESHAPE maps <code>[r0, 0]</code> back to <code>[r0]</code>.",
      "The sum's result is broadcast back over r1, a range its consumer iterates and it does not: an <b>ending range</b>. Rule: <i>REDUCE before the broadcast</i> is realized, so it gets a new output range <b>r2</b>, and the REDUCE creates a fresh <code>AxisType.REDUCE</code> range <b>r3</b>. The PERMUTE swaps them.",
      "EXP2 has <b>two consumers that index it differently</b>: the final MUL wants it at <code>[r0, r1]</code>, the sum at <code>[r2, r3]</code>. The ranges don't agree, so EXP2 gets new ranges <b>r4, r5</b> and is marked for realization.",
      "The exp chain inherits <code>[r4, r5]</code>. The max is again a reduce broadcast back over an axis (r5), so it is realized: output range <b>r6</b>, reduce range <b>r7</b>.",
      "The input RESHAPE has two consumers with different ranges (<code>[r4, r5]</code> and <code>[r6, r7]</code>): new ranges <b>r8, r9</b>, realized. Every node now has ranges; four nodes plus the STORE are marked for realization.",
      "<b>apply rangeify</b> rewrites the graph with these ranges. Movement ops disappear: their effect is now in the index expressions. Every realized node is wrapped in a <b>STAGE</b> over its closed ranges: 4 STAGEs (input, max, exp, sum) plus the output STORE.",
      "<b>symbolic+reduce_collapse+debuf</b> runs the cost model <code>remove_bufferize</code>. The input STAGE and the EXP2 STAGE contain no REDUCE that reads a buffer, so they are removed and their values are <b>recomputed</b> in each consumer. The max and sum STAGEs stay: each wraps a REDUCE over a buffer.",
      "<b>stage to store</b>: each remaining STAGE becomes a real buffer (<code>BUFFER</code> of 1024 floats), a <code>STORE</code> into it, an <code>END</code> that closes its loop, and an <code>AFTER</code> that readers go through. That AFTER is the dependency edge between kernels.",
      "<b>split kernels</b> cuts the graph at every STORE/END with no open ranges: three kernels. EXP2 appears in two of them: tinygrad computes exp twice rather than writing a 4 MiB intermediate.",
      "Codegen then optimizes and names each kernel. These are the three kernels the L4 ran, from the capture.",
    ];
    root.classList.add("widget");
    root.innerHTML = `<div class="widget-head"><span class="title">softmax: from tensor graph to three kernels</span>
      <button class="btn prev">◀</button><button class="btn play">▶ play</button><button class="btn next">▶</button><span class="chip stepc"></span></div>
      <svg id="rgsvg" class="ch3svg" viewBox="0 0 1180 440"></svg><div class="rg-cap"></div><pre class="code rg-code"></pre>`;
    const sc = scene(root.querySelector("svg"));
    function show(i, anim) {
      const seen = new Set(ORDER.slice(0, Math.min(i, 6)).flat()), realized = new Set(REAL.slice(0, Math.min(i, 6)).flat()), now = new Set(i >= 1 && i <= 6 ? ORDER[i - 1] : []);
      const nodes = [], boxes = [];
      for (const [id, [x, y, l, s]] of Object.entries(N)) {
        const n = { id, x, y, label: l, sub: s, cls: now.has(id) ? "hot" : realized.has(id) && i <= 6 ? "real" : "" };
        if (i <= 6) { if (seen.has(id)) n.chip = CH[id]; }
        else {
          if (MOV.includes(id)) { n.o = 0; const tgt = { rx: "in", p1: "mx", r1: "ng", p2: "sm", r2: "rc" }[id]; n.x = N[tgt][0]; n.y = N[tgt][1]; }
          if (id === "mx") n.x = 300; if (id === "sm") n.x = 900;
        }
        nodes.push(n);
      }
      const at = (id) => nodes.find((n) => n.id === id);
      if (i === 7) for (const [id, lab] of [["in", "STAGE (input)"], ["mx", "STAGE (max)"], ["ex", "STAGE (exp)"], ["sm", "STAGE (sum)"]]) { const n = at(id); boxes.push({ id: "s" + id, x: n.x - 52, y: n.y - 38, w: 104, h: 64, label: lab, cls: "stage" }); }
      if (i === 8 || i === 9) for (const [id, lab] of [["mx", "STAGE (max)"], ["sm", "STAGE (sum)"]]) { const n = at(id); boxes.push({ id: "s" + id, x: n.x - 52, y: n.y - 38, w: 104, h: 64, label: lab, cls: "stage" + (i === 9 ? " gone" : ""), o: i === 9 ? 0 : 1 }); }
      if (i === 8) for (const id of ["in", "ex"]) { const n = at(id); boxes.push({ id: "s" + id, x: n.x - 52, y: n.y - 38, w: 104, h: 64, label: "", cls: "stage", o: 0 }); }
      if (i >= 8 && i <= 9) { at("in").cls = "recomp"; at("ex").cls = "recomp"; }
      let edges = i <= 6 ? E0 : i <= 8 ? E1 : E2;
      if (i >= 9 && i <= 9) {
        nodes.push({ id: "b0", x: 440, y: 150, label: "BUFFER", sub: "buf0 · 1024 f32", cls: "buf", w: 110 }, { id: "b1", x: 1000, y: 400, label: "BUFFER", sub: "buf1 · 1024 f32", cls: "buf", w: 110 });
      }
      if (i >= 10) {
        // kernel layout: three rows
        const P = { in: [60, 215], mx: [300, 70], b0: [700, 70], ex1: [300, 215], sm: [500, 215], b1: [700, 215], ex: [300, 360], rc: [500, 360], fm: [700, 360], st: [870, 360], out: [1060, 360] };
        nodes.length = 0;
        for (const [id, [x, y]] of Object.entries(P)) {
          const base = N[id] || N.ex;
          nodes.push({ id, x, y, label: id === "ex1" || id === "ex" ? "EXP2" : base[2], sub: id === "ex1" || id === "ex" ? "(x − max)·log2e" : base[3], w: id === "ex" || id === "ex1" ? 120 : 84, cls: id === "ex1" || id === "ex" ? "recomp" : "" });
        }
        nodes.find((n) => n.id === "b0").label = "BUFFER"; nodes.find((n) => n.id === "b0").sub = "buf0 · 1024 f32"; nodes.find((n) => n.id === "b0").cls = "buf"; nodes.find((n) => n.id === "b0").w = 110;
        nodes.find((n) => n.id === "b1").label = "BUFFER"; nodes.find((n) => n.id === "b1").sub = "buf1 · 1024 f32"; nodes.find((n) => n.id === "b1").cls = "buf"; nodes.find((n) => n.id === "b1").w = 110;
        for (const id of ["ng", "ad", "m2"]) nodes.push({ id, x: P.ex[0], y: P.ex[1], label: "", o: 0 });
        for (const id of MOV) nodes.push({ id, x: 60, y: 215, label: "", o: 0 });
        edges = [["in", "mx"], ["mx", "b0"], ["in", "ex1"], ["b0", "ex1"], ["ex1", "sm"], ["sm", "b1"], ["in", "ex"], ["b0", "ex", [[790, 70], [790, 290], [300, 290]]], ["b1", "rc"], ["ex", "fm"], ["rc", "fm"], ["fm", "st"], ["out", "st"]];
        const full = i === 11;
        boxes.push({ id: "k0", x: 200, y: 25, w: 400, h: 92, label: full ? kname(0) : "kernel 0: max", sub: full ? ksub(0) : "", cls: "kern k0" },
          { id: "k1", x: 200, y: 165, w: 400, h: 100, label: full ? kname(1) : "kernel 1: sum of exp", sub: full ? ksub(1) : "", cls: "kern k1" },
          { id: "k2", x: 20, y: 312, w: 940, h: 100, label: full ? kname(2) : "kernel 2: exp × 1/sum", sub: full ? ksub(2) : "", cls: "kern k2" });
      }
      sc.set({ nodes, edges: edges.map(([f, t, via]) => ({ from: f, to: t, via })), boxes }, anim);
      root.querySelector(".rg-cap").innerHTML = steps[i];
      const code = root.querySelector(".rg-code");
      if (i === 0) code.innerHTML = B.codeHTML("# DEBUG_RANGEIFY-style trace: *** = realized, then #consumers, op, shape, ranges (in -> out)\n# (reconstructed; range numbers are the ones in the captured IR)", "python");
      else if (i <= 6) {
        const rows = ORDER.slice(0, i).flat().filter((id) => TRACE[id]).map((id) => TRACE[id]);
        const nowRows = new Set(ORDER[i - 1].filter((id) => TRACE[id]).map((id) => TRACE[id]));
        code.innerHTML = B.codeHTML(rows.join("\n"), "text", rows.map((r) => (nowRows.has(r) ? "add" : "")));
      } else code.innerHTML = B.codeHTML(LOOPS[i], "python");
    }
    stepper(root, steps.length, show, 3200);
  })();

  // ---------- 3. memory traffic ----------
  (function traffic() {
    const root = document.getElementById("traffic");
    if (!root) return;
    const MiB = 1024 * 1024, T = 4 * MiB, KB4 = 4096;
    const fcT = B.get("fused_chain/torch"), fcG = B.get("fused_chain/default"), smT = B.get("softmax/torch"), smG = B.get("softmax/default");
    const ek = fcT.eager_kernels;
    const short = (n) => n.includes("MulFunctor") && n.includes("BinaryFunctor") ? "eager mul (a*b)" : n.includes("CUDAFunctor_add") ? "eager add (+c)" : n.includes("clamp") ? "eager clamp (relu)" : n.includes("AUnaryFunctor") ? "eager mul (×2.0)" : n.slice(0, 30);
    const fcEagerIO = [[["a", "b"], ["t1"]], [["t1", "c"], ["t2"]], [["t2"], ["t3"]], [["t3"], ["out"]]];
    const SC = [
      { name: "fused_chain · PyTorch eager", slots: { a: T, b: T, c: T, t1: T, t2: T, t3: T, out: T },
        kernels: ek.map((k, i) => ({ name: short(k.name), us: k.us, r: fcEagerIO[i][0], w: fcEagerIO[i][1] })) },
      { name: "fused_chain · tinygrad", slots: { a: T, b: T, c: T, out: T }, kernels: fcG.kernels.map((k) => ({ name: k.name, us: k.time_us, r: ["a", "b", "c"], w: ["out"] })) },
      { name: "fused_chain · Inductor", slots: { a: T, b: T, c: T, out: T }, kernels: fcT.inductor_kernels.map((k) => ({ name: k.name, us: k.us, r: ["a", "b", "c"], w: ["out"] })) },
      { name: "softmax · tinygrad", slots: { x: T, buf0: KB4, buf1: KB4, out: T },
        kernels: [[["x"], ["buf0"]], [["x", "buf0"], ["buf1"]], [["x", "buf0", "buf1"], ["out"]]].map(([r, w], i) => ({ name: smG.kernels[i].name, us: smG.kernels[i].time_us, r, w, est: +smG.kernels[i].estimates.mem })) },
      { name: "softmax · PyTorch eager", slots: { x: T, out: T }, kernels: smT.eager_kernels.map((k) => ({ name: "softmax_warp_forward", us: k.us, r: ["x"], w: ["out"] })) },
      { name: "softmax · Inductor", slots: { x: T, out: T }, kernels: smT.inductor_kernels.map((k) => ({ name: k.name, us: k.us, r: ["x"], w: ["out"] })) },
    ];
    for (const s of SC) { s.bytes = s.kernels.reduce((t, k) => t + [...k.r, ...k.w].reduce((u, n) => u + s.slots[n], 0), 0); s.us = s.kernels.reduce((t, k) => t + (k.us || 0), 0); }
    const fmtB = (b) => b >= MiB ? (b / MiB).toFixed(b % MiB ? 2 : 0) + " MiB" : (b / 1024).toFixed(0) + " KiB";
    root.classList.add("widget");
    root.innerHTML = `<div class="widget-head"><span class="title">Round trips to memory</span>
      <select class="btn scen">${SC.map((s, i) => `<option value="${i}">${esc(s.name)}</option>`).join("")}</select>
      <button class="btn play">▶ run</button><span class="chip tc"></span></div>
      <svg class="ch3svg" id="trsvg" viewBox="0 0 900 300"></svg>
      <div class="widget-body tr-bars"></div>`;
    const svg = root.querySelector("svg");
    let anim = null;
    function bars(sel) {
      const max = Math.max(...SC.map((s) => s.bytes));
      root.querySelector(".tr-bars").innerHTML = `<div class="bars">${SC.map((s, i) => `<div class="row ${i === sel ? "sel" : ""}"><span>${esc(s.name)}</span><div class="track"><div class="fill" style="width:${100 * s.bytes / max}%;background:${s.name.includes("tinygrad") ? "var(--s-schedule)" : "#ee4c2c"}"></div></div><span class="val">${fmtB(s.bytes)} · ${s.kernels.length} kernel${s.kernels.length > 1 ? "s" : ""}</span></div>`).join("")}</div>
        <div class="tr-note">Bytes each kernel must read and write in global memory, from the shapes: one 1024×1024 float32 tensor is 4 MiB, a 1024-float row statistic is 4 KiB.${SC[sel].kernels[0].est ? " For tinygrad's softmax these equal codegen's own memory estimates (<code>estimates.mem</code>: " + SC[sel].kernels.map((k) => k.est.toLocaleString()).join(", ") + " bytes)." : ""}</div>`;
    }
    function render(s, ki, phase, t, read, written) {
      const names = Object.keys(s.slots), sh = Math.min(34, 230 / names.length);
      let h = `<rect class="dram" x="20" y="20" width="250" height="260" rx="10"/><text class="lbl" x="34" y="40">global memory</text>`;
      const slotY = {};
      names.forEach((n, i) => {
        const y = 52 + i * sh; slotY[n] = y + sh / 2 - 3;
        const k = s.kernels[ki], isR = k && phase === 0 && k.r.includes(n), isW = k && phase === 1 && k.w.includes(n);
        const exists = !written.has("?" + n) && (["a", "b", "c", "x"].includes(n) || written.has(n));
        h += `<rect class="slot ${isR ? "rd" : isW ? "wr" : ""} ${exists ? "" : "empty"}" x="34" y="${y}" width="222" height="${sh - 6}" rx="4"/><text class="slab" x="44" y="${y + sh / 2}">${n}</text><text class="ssz" x="248" y="${y + sh / 2}">${fmtB(s.slots[n])}</text>`;
      });
      const k = s.kernels[ki];
      h += `<rect class="sm" x="620" y="90" width="260" height="120" rx="10"/><text class="lbl" x="634" y="110">SMs: kernel ${k ? ki + 1 : "-"} of ${s.kernels.length}</text><text class="kn" x="750" y="150">${esc(k ? k.name.slice(0, 34) : "done")}</text>`;
      if (k) {
        const list = phase === 0 ? k.r : k.w;
        list.forEach((n, j) => {
          const dots = Math.max(1, Math.round(s.slots[n] / MiB));
          for (let q = 0; q < dots; q++) {
            const tt = Math.max(0, Math.min(1, t * 1.6 - q * 0.12 - j * 0.08));
            if (tt <= 0 || tt >= 1) continue;
            const x0 = 256, y0 = slotY[n], x1 = 620, y1 = 150, x = phase === 0 ? x0 + (x1 - x0) * tt : x1 + (x0 - x1) * tt, y = phase === 0 ? y0 + (y1 - y0) * tt : y1 + (y0 - y1) * tt;
            h += `<circle class="${phase === 0 ? "pr" : "pw"}" cx="${x}" cy="${y}" r="${s.slots[n] < MiB ? 3 : 6}"/>`;
          }
        });
      }
      h += `<text class="cnt" x="300" y="268">read ${fmtB(read)} · written ${fmtB(written.bytes || 0)} · launches ${Math.min(ki + 1, s.kernels.length)}</text>`;
      svg.innerHTML = h;
    }
    function run(sel, play) {
      if (anim) cancelAnimationFrame(anim);
      const s = SC[sel];
      bars(sel);
      root.querySelector(".tc").textContent = `${s.kernels.length} kernel${s.kernels.length > 1 ? "s" : ""} · ${fmtB(s.bytes)} moved · measured ${B.fmtUs(s.us)} total`;
      const written = new Set(); written.bytes = 0;
      if (!play) { render(s, -1, 0, 0, 0, written); return; }
      let ki = 0, phase = 0, t0 = performance.now(), read = 0;
      const D = 900;
      const frame = (now) => {
        const t = Math.min(1, (now - t0) / D), k = s.kernels[ki];
        render(s, ki, phase, t, read, written);
        if (t >= 1) {
          if (phase === 0) { read += k.r.reduce((u, n) => u + s.slots[n], 0); phase = 1; }
          else { k.w.forEach((n) => written.add(n)); written.bytes += k.w.reduce((u, n) => u + s.slots[n], 0); phase = 0; ki++; }
          t0 = now;
          if (ki >= s.kernels.length) { render(s, s.kernels.length - 1, 2, 0, read, written); anim = null; return; }
        }
        anim = requestAnimationFrame(frame);
      };
      anim = requestAnimationFrame(frame);
    }
    const sel = root.querySelector(".scen");
    sel.onchange = () => run(+sel.value, true);
    root.querySelector(".play").onclick = () => run(+sel.value, true);
    run(0, false);
    B.whenVisible(root, () => run(+sel.value, true));
  })();
});
