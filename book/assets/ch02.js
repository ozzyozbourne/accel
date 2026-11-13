/* Chapter 2: UOps and the rewrite engine.
 * Three custom diagrams:
 *   #dag-demo        hash-consing: a softmax-shaped expression tree collapses into a DAG
 *   #pyrender-map    the real fused_chain tensor graph (17 UOps) linked to its pyrender lines
 *   #toy-rewriter    a teaching model of graph_rewrite: rules fire on an SVG tree, step by step
 */
BOOK.ready((B) => {
  const esc = B.esc;

  // ---------- a tiny tweening graph scene (nodes, edges, boxes) ----------
  function scene(svg) {
    let cur = { nodes: new Map(), edges: new Map() }, raf = null;
    const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);
    function border(a, b) { // point on rect a's border toward b's center
      const dx = b.x - a.x, dy = b.y - a.y;
      if (!dx && !dy) return { x: a.x, y: a.y };
      const sx = (a.w / 2) / Math.abs(dx || 1e-9), sy = (a.h / 2) / Math.abs(dy || 1e-9), s = Math.min(sx, sy);
      return { x: a.x + dx * s, y: a.y + dy * s };
    }
    function draw(nodes, edges) {
      let h = "";
      for (const e of edges) {
        const a = nodes.get(e.from), b = nodes.get(e.to);
        if (!a || !b) continue;
        const p = border(a, b), q = border(b, a), op = Math.min(e.o, a.o, b.o);
        if (op < 0.02) continue;
        h += `<line x1="${p.x}" y1="${p.y}" x2="${q.x}" y2="${q.y}" class="e ${e.cls || ""}" style="opacity:${op}" marker-end="url(#${svg.id}-arr)"/>`;
        if (e.label) h += `<text class="elabel" x="${(p.x + q.x) / 2 + 6}" y="${(p.y + q.y) / 2}" style="opacity:${op}">${esc(e.label)}</text>`;
      }
      for (const n of nodes.values()) {
        if (n.o < 0.02) continue;
        h += `<g class="n ${n.cls || ""}" data-id="${n.id}" style="opacity:${n.o}"><rect x="${n.x - n.w / 2}" y="${n.y - n.h / 2}" width="${n.w}" height="${n.h}" rx="7"/>` +
          `<text x="${n.x}" y="${n.y + (n.sub ? -2 : 4)}">${esc(n.label)}</text>` + (n.sub ? `<text class="sub" x="${n.x}" y="${n.y + 12}">${esc(n.sub)}</text>` : "") + `</g>`;
      }
      svg.querySelector(".layer").innerHTML = h;
    }
    svg.innerHTML = `<defs><marker id="${svg.id}-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="#6b7c93"/></marker></defs><g class="layer"></g>`;
    return {
      set(state, animate = true) {
        const nxt = { nodes: new Map(state.nodes.map((n) => [n.id, { w: 64, h: 30, ...n, o: n.o ?? 1 }])), edges: new Map(state.edges.map((e) => [e.from + ">" + e.to + (e.k || ""), { ...e, o: 1 }])) };
        if (raf) cancelAnimationFrame(raf);
        const prev = cur, ids = new Set([...prev.nodes.keys(), ...nxt.nodes.keys()]), eids = new Set([...prev.edges.keys(), ...nxt.edges.keys()]);
        const t0 = performance.now(), D = animate ? 650 : 0;
        const frame = (now) => {
          const t = D ? ease(Math.min(1, (now - t0) / D)) : 1, ns = new Map(), es = [];
          for (const id of ids) {
            const a = prev.nodes.get(id), b = nxt.nodes.get(id), s = a || b, f = b || a;
            const oa = a ? a.o : 0, ob = b ? b.o : 0;
            ns.set(id, { ...f, x: s.x + (f.x - s.x) * t, y: s.y + (f.y - s.y) * t, o: oa + (ob - oa) * t, cls: (t < 0.5 && a && !b) ? a.cls : f.cls });
          }
          for (const k of eids) { const a = prev.edges.get(k), b = nxt.edges.get(k), f = b || a; es.push({ ...f, o: (a ? 1 : 0) + ((b ? 1 : 0) - (a ? 1 : 0)) * t }); }
          draw(ns, es);
          if (t < 1) raf = requestAnimationFrame(frame); else { raf = null; cur = nxt; }
        };
        if (D) raf = requestAnimationFrame(frame); else frame(t0);
      },
    };
  }
  B.ch02scene = scene;

  // ---------- 1. hash-consing: tree -> DAG ----------
  (function dagDemo() {
    const root = document.getElementById("dag-demo");
    if (!root) return;
    root.classList.add("widget");
    root.innerHTML = `<div class="widget-head"><span class="title">Hash-consing: the same expression as a tree and as UOps</span>
      <button class="btn prev">◀</button><button class="btn play">▶ play</button><button class="btn next">▶</button><span class="chip cnt"></span></div>
      <div class="widget-body"><svg id="dagsvg" class="ch2svg" viewBox="0 0 800 420"></svg><div class="cap"></div></div>`;
    const sc = scene(root.querySelector("svg"));
    // tree positions (as written) and DAG positions (as UOps)
    const T = {
      div: [400, 30, "DIV"], expa: [220, 100, "EXP"], suba: [220, 170, "SUB"], x1: [150, 245, "x"], maxa: [270, 300, "MAX"], x2: [270, 370, "x"],
      sum: [590, 100, "SUM"], expb: [590, 170, "EXP"], subb: [590, 245, "SUB"], x3: [520, 320, "x"], maxb: [660, 320, "MAX"], x4: [660, 390, "x"],
    };
    const DAG = { div: [400, 30], sum: [540, 105], expa: [330, 175], suba: [330, 245], maxa: [470, 320], x1: [330, 390] };
    const E = [["expa", "div"], ["sum", "div"], ["suba", "expa"], ["x1", "suba"], ["maxa", "suba"], ["x2", "maxa"], ["expb", "sum"], ["subb", "expb"], ["x3", "subb"], ["maxb", "subb"], ["x4", "maxb"]];
    // merge steps: which duplicate collapses into which canonical node
    const steps = [
      { m: {}, cap: "As written: <b>12 nodes</b>. <code>e = (x - x.max()).exp()</code> appears twice, once in the numerator and once inside the sum (simplified ops)." },
      { m: { x2: "x1", x3: "x1", x4: "x1" }, cap: "Leaves first. The four <code>x</code> nodes have the same <code>(op, dtype, src, arg, tag)</code> key, so the constructor returns the one cached UOp every time: <b>9 nodes</b>." },
      { m: { maxb: "maxa" }, cap: "Now both MAX nodes have the identical source <i>object</i> <code>x</code>, so their keys match too: <b>8 nodes</b>." },
      { m: { subb: "suba" }, cap: "Then both SUB(x, MAX(x)): <b>7 nodes</b>." },
      { m: { expb: "expa" }, cap: "Then both EXP: <b>6 nodes</b>. Because sources are compared by identity, deduplication works bottom-up in a single pass, with no tree comparison." },
      { m: {}, dag: true, cap: "The final DAG has 6 UOps. EXP has two consumers (DIV and SUM). In Chapter 3 that sharing matters: the scheduler must decide whether a node with two consumers gets its own buffer or is computed twice." },
    ];
    let cur = 0, timer = null;
    function state(i) {
      const merged = {};
      for (let s = 1; s <= Math.min(i, 4); s++) Object.assign(merged, steps[s].m);
      const canon = (id) => (merged[id] ? canon(merged[id]) : id), dag = steps[i].dag;
      const nodes = Object.entries(T).map(([id, [x, y, l]]) => {
        const c = canon(id), pos = dag && DAG[c] ? DAG[c] : (merged[id] ? T[c] : [x, y]);
        const justNow = steps[i].m[id] !== undefined;
        return { id, x: pos[0], y: pos[1], label: l, w: 58, o: merged[id] ? 0 : 1, cls: (Object.values(steps[i].m).includes(id) || justNow) ? "hot" : "" };
      });
      const seen = new Set(), edges = [];
      for (const [a, b] of E) { const ca = canon(a), cb = canon(b), k = ca + ">" + cb; if (merged[b] || seen.has(k)) continue; seen.add(k); edges.push({ from: ca, to: cb }); }
      return { nodes, edges, n: nodes.filter((n) => n.o > 0).length };
    }
    function show(i, anim = true) {
      cur = Math.max(0, Math.min(steps.length - 1, i));
      const s = state(cur);
      sc.set(s, anim);
      root.querySelector(".cnt").textContent = `step ${cur + 1}/${steps.length} · ${s.n} nodes`;
      root.querySelector(".cap").innerHTML = steps[cur].cap;
    }
    root.querySelector(".prev").onclick = () => show(cur - 1);
    root.querySelector(".next").onclick = () => show(cur + 1);
    root.querySelector(".play").onclick = (e) => {
      if (timer) { clearInterval(timer); timer = null; e.target.textContent = "▶ play"; return; }
      if (cur >= steps.length - 1) show(0);
      e.target.textContent = "❚❚ pause";
      timer = setInterval(() => { if (cur >= steps.length - 1) { clearInterval(timer); timer = null; e.target.textContent = "▶ play"; } else show(cur + 1); }, 1700);
    };
    show(0, false);
  })();

  // ---------- 2. the fused_chain tensor graph, linked to pyrender lines ----------
  (function pyrenderMap() {
    const root = document.getElementById("pyrender-map");
    if (!root) return;
    const d = B.get("fused_chain/default"), lines = d.tensor_graph.text.split("\n");
    // the 17 UOps of the graph (toposort order checked with a CPU run of the same expression), hand laid out
    const N = [
      ["k1m", 160, 30, "CONST", "1048576", "weakint", "1048576", "c2,c6,c9", "Inlined into every new_buffer(...) call: consts are never named. All three buffers share this one node."],
      ["k1024", 520, 30, "CONST", "1024", "weakint", "1024", "c4", "Used twice by the STACK (shape 1024×1024); still one node."],
      ["b1", 60, 105, "BUFFER", "slot 1", "float", "ParamArg(1, float, 'CUDA')", "c2", "Input a. BUFFER is always rendered on its own line."],
      ["b3", 190, 105, "BUFFER", "slot 3", "float", "ParamArg(3, float, 'CUDA')", "c6", "Input b."],
      ["b5", 320, 105, "BUFFER", "slot 5", "float", "ParamArg(5, float, 'CUDA')", "c9", "Input c."],
      ["stk", 520, 105, "STACK", "(1024,1024)", "weakint", "None", "c4", "A shape argument is itself a UOp: a STACK of CONSTs, used as src[1] of each RESHAPE."],
      ["r1", 60, 180, "RESHAPE", "", "float", "None", "c11", "src = (BUFFER, STACK). One consumer, so it is inlined as UOp(Ops.RESHAPE, ...) inside line c11."],
      ["r3", 190, 180, "RESHAPE", "", "float", "None", "c11", "Same shape node c4, different buffer."],
      ["r5", 320, 180, "RESHAPE", "", "float", "None", "c11", ""],
      ["mul1", 125, 255, "MUL", "a*b", "float", "None", "c11", "Inlined: rendered with the * operator."],
      ["add", 220, 330, "ADD", "a*b+c", "float", "None", "c11", "Two consumers (CMPLT and WHERE), so it gets a name: c11."],
      ["k0", 480, 405, "CONST", "0.0", "weakfloat", "0.0", "c13", "The 0 of relu, used by both CMPLT and WHERE: one node. weakfloat means 'a Python float, adopt the other operand's dtype'."],
      ["lt", 340, 405, "CMPLT", "0<x", "bool", "None", "c13", "x > 0 is written as 0 < x: there is only a less-than op."],
      ["wh", 260, 480, "WHERE", "", "float", "None", "c13", "WHERE is always rendered: this is relu, (x > 0).where(x, 0)."],
      ["k2", 450, 480, "CONST", "2.0", "weakfloat", "2.0", "c15", ""],
      ["mul2", 300, 555, "MUL", "×2", "float", "None", "c15", "Sources of the SINK are always named."],
      ["sink", 300, 620, "SINK", "", "void", "None", "ast", "The root. The last line is always called ast."],
    ];
    const E = [["k1m", "b1"], ["k1m", "b3"], ["k1m", "b5"], ["k1024", "stk", "×2"], ["b1", "r1"], ["b3", "r3"], ["b5", "r5"], ["stk", "r1"], ["stk", "r3"], ["stk", "r5"],
      ["r1", "mul1"], ["r3", "mul1"], ["mul1", "add"], ["r5", "add"], ["k0", "lt"], ["add", "lt"], ["lt", "wh"], ["add", "wh"], ["k0", "wh"], ["wh", "mul2"], ["k2", "mul2"], ["mul2", "sink"]];
    root.classList.add("widget");
    root.innerHTML = `<div class="widget-head"><span class="title">fused_chain/default: the lazy tensor graph, node by node</span><span class="chip">${N.length} UOps</span><span style="font-size:12px;color:var(--muted)">hover a line or a node</span></div>
      <div class="split pm-split"><div><svg id="pmsvg" class="ch2svg" viewBox="0 0 600 650"></svg></div>
      <div><pre class="code pm-lines">${lines.map((l, i) => `<span class="ln" data-l="${esc(l.split(" = ")[0])}">${B.highlight(l, "uop") || " "}</span>`).join("")}</pre>
      <div class="pm-info"></div></div></div>`;
    const svg = root.querySelector("svg"), sc = scene(svg);
    const base = { nodes: N.map(([id, x, y, l, s]) => ({ id, x, y, label: l, sub: s, w: id.startsWith("k") ? 74 : 80, h: 34 })), edges: E.map(([f, t, lab]) => ({ from: f, to: t, label: lab })) };
    const info = root.querySelector(".pm-info");
    function hi(names, nid) {
      sc.set({ nodes: base.nodes.map((n) => { const row = N.find((r) => r[0] === n.id); const on = row[7].split(",").some((x) => names.includes(x)) || n.id === nid; return { ...n, cls: on ? "hot" : (names.length || nid ? "dim" : "") }; }), edges: base.edges }, false);
      root.querySelectorAll(".pm-lines .ln").forEach((s) => s.classList.toggle("hl", names.includes(s.dataset.l)));
      const r = nid ? N.find((x) => x[0] === nid) : null;
      info.innerHTML = r ? `<b>${r[3]}</b> · dtype <code>${r[5]}</code> · arg <code>${esc(r[6])}</code> · rendered in <code>${r[7]}</code><br>${esc(r[8])}` :
        names.length ? `Line <code>${esc(names[0])}</code> defines the highlighted nodes. Everything inlined into it has exactly one consumer.` : "Each <code>cN</code> is a node that needs a name: it has several consumers or its op is always rendered. N is the node's index in the toposort, so the numbers skip.";
    }
    root.querySelectorAll(".pm-lines .ln").forEach((s) => { s.onmouseenter = () => hi([s.dataset.l]); s.onclick = s.onmouseenter; });
    svg.addEventListener("mouseover", (e) => { const g = e.target.closest("g.n"); if (g) { const r = N.find((x) => x[0] === g.dataset.id); hi(r[7].split(",").slice(0, 1), g.dataset.id); } });
    root.addEventListener("mouseleave", () => hi([]));
    hi([]);
  })();

  // ---------- 3. a toy graph_rewrite ----------
  (function toyRewriter() {
    const root = document.getElementById("toy-rewriter");
    if (!root) return;
    let nid = 0;
    const V = (name) => ({ id: nid++, op: "VAR", name, kids: [] });
    const C = (val) => ({ id: nid++, op: "CONST", val, kids: [] });
    const O = (op, ...kids) => ({ id: nid++, op, kids });
    const SYM = { ADD: "+", MUL: "×", CMPLT: "<", WHERE: "where", MAX: "max" };
    const label = (n) => n.op === "VAR" ? n.name : n.op === "CONST" ? String(n.val) : SYM[n.op];
    const key = (n) => n.op === "VAR" ? "v" + n.name : n.op === "CONST" ? "c" + n.val : n.op + "(" + n.kids.map(key).join(",") + ")";
    const same = (a, b) => key(a) === key(b);  // in tinygrad this is just `is`: hash-consing makes equal graphs the same object
    const isC = (n, v) => n.op === "CONST" && (v === undefined || n.val === v);
    const str = (n) => n.op === "VAR" ? n.name : n.op === "CONST" ? String(n.val) : n.op === "WHERE" ? `where(${str(n.kids[0])}, ${str(n.kids[1])}, ${str(n.kids[2])})` :
      n.op === "MAX" ? `max(${str(n.kids[0])}, ${str(n.kids[1])})` : `(${str(n.kids[0])}${SYM[n.op] === "×" ? "*" : SYM[n.op]}${str(n.kids[1])})`;
    // rules, in the order they appear in uop/symbolic.py (a PatternMatcher tries the patterns of a node's op in list order)
    const RULES = [
      { name: "x+0 → x", src: "symbolic.py#L111", code: `(UPat({Ops.ADD, Ops.XOR, Ops.OR}, src=[UPat.var("x"), UPat.const(0)]), lambda x: x)`,
        f: (n) => n.op === "ADD" ? (isC(n.kids[1], 0) ? n.kids[0] : isC(n.kids[0], 0) ? n.kids[1] : null) : null },
      { name: "x*1 → x", src: "symbolic.py#L113", code: `(UPat.var("x") * 1, lambda x: x)`,
        f: (n) => n.op === "MUL" ? (isC(n.kids[1], 1) ? n.kids[0] : isC(n.kids[0], 1) ? n.kids[1] : null) : null },
      { name: "constant folding", src: "symbolic.py#L150", code: `(UPat(GroupOp.Binary-{Ops.THREEFRY}, src=(const_folding_pat,)*2, name="a"), fold_const_alu)`,
        f: (n) => (n.op === "ADD" || n.op === "MUL") && isC(n.kids[0]) && isC(n.kids[1]) ? C(n.op === "ADD" ? n.kids[0].val + n.kids[1].val : n.kids[0].val * n.kids[1].val) : null },
      { name: "x*0 → 0", src: "symbolic.py#L165", code: `(UPat.var("x") * 0, lambda x: x.const_like(... 0))`,
        f: (n) => n.op === "MUL" && (isC(n.kids[0], 0) || isC(n.kids[1], 0)) && !(isC(n.kids[0]) && isC(n.kids[1])) ? C(0) : null },
      { name: "max folding: (c<b).where(b, c) → max(c, b)", src: "symbolic.py#L261", code: `((UPat.cvar("a") < UPat.var("b")).where(UPat.var("b"), UPat.cvar("c")), lambda a,b,c: UOp.maximum(a,b) if a.val == c.val else None)`,
        f: (n) => { if (n.op !== "WHERE") return null; const [g, t, e] = n.kids; return g.op === "CMPLT" && isC(g.kids[0]) && same(g.kids[1], t) && isC(e, g.kids[0].val) ? O("MAX", g.kids[0], t) : null; } },
      { name: "two-stage folding: (x op c1) op c2 → x op (c1 op c2)", src: "symbolic.py#L267", code: `(UPat.var("x").alu(op, UPat.cvar("c1")).alu(op, UPat.cvar("c2")).named("f"), lambda f,x,c1,c2: x.alu(f.op,c1.alu(f.op,c2)))`,
        f: (n) => { if (n.op !== "ADD" && n.op !== "MUL") return null; const [a, c2] = n.kids; return a.op === n.op && isC(c2) && isC(a.kids[1]) && !isC(a.kids[0]) ? O(n.op, a.kids[0], O(n.op, a.kids[1], c2)) : null; } },
    ];
    const PRESETS = [
      ["((x*1)+0)*2", () => O("MUL", O("ADD", O("MUL", V("x"), C(1)), C(0)), C(2))],
      ["((x*1)+0)*(3+4)", () => O("MUL", O("ADD", O("MUL", V("x"), C(1)), C(0)), O("ADD", C(3), C(4)))],
      ["((x*2)*3)+0", () => O("ADD", O("MUL", O("MUL", V("x"), C(2)), C(3)), C(0))],
      ["relu(y)*2, y = a*b+c", () => { const y = () => O("ADD", O("MUL", V("a"), V("b")), V("c")); return O("MUL", O("WHERE", O("CMPLT", C(0), y()), y(), C(0)), C(2)); }],
      ["x*0 + y*1", () => O("ADD", O("MUL", V("x"), C(0)), O("MUL", V("y"), C(1)))],
    ];
    root.classList.add("widget");
    root.innerHTML = `<div class="widget-head"><span class="title">A toy graph_rewrite (a teaching model, in JavaScript)</span>
      <select class="btn preset">${PRESETS.map((p, i) => `<option value="${i}">${esc(p[0])}</option>`).join("")}</select>
      <div class="tabs"><button class="btn on" data-m="td">default: sources first</button><button class="btn" data-m="bu">bottom_up=True: node first</button></div>
      <button class="btn prev">◀</button><button class="btn play">▶ play</button><button class="btn next">▶</button><span class="chip stepc"></span></div>
      <div class="toy-grid"><div><svg id="toysvg" class="ch2svg" viewBox="0 0 640 360"></svg><div class="toy-expr mono"></div></div>
      <div><div class="toy-rules">${RULES.map((r, i) => `<div class="rule" data-r="${i}"><b>${esc(r.name)}</b> <a class="src" data-src="tinygrad/uop/${r.src}"></a><code>${esc(r.code)}</code></div>`).join("")}</div></div></div>
      <div class="toy-msg"></div>`;
    root.querySelectorAll("a.src").forEach((a) => { const [p, l] = a.dataset.src.split("#"); a.href = `https://github.com/tinygrad/tinygrad/blob/${B.commit}/${p}#${l}`; a.target = "_blank"; a.textContent = p.replace("tinygrad/", "") + ":" + l.slice(1); });
    const sc = scene(root.querySelector("svg"));
    let mode = "td", events = [], cur = 0, timer = null;

    function clone(n) { return { id: n.id, op: n.op, name: n.name, val: n.val, kids: n.kids.map(clone) }; }
    function tryRules(n) { for (let i = 0; i < RULES.length; i++) { const out = RULES[i].f(n); if (out) return { i, out }; } return null; }
    function run() {
      nid = 0;
      const holder = { kids: [PRESETS[+root.querySelector(".preset").value][1]()] }, done = new Set();
      events = [];
      const ev = (focus, rule, msg, kind) => events.push({ tree: clone(holder.kids[0]), focus, rule, msg, kind });
      ev(null, null, "The starting expression. Press ▶ to step through the rewrite.", "start");
      let visits = 0;
      // default mode: rewrite all sources, rebuild, then try the rules on the rebuilt node (unified_rewrite stage 1)
      function td(n, set) {
        if (done.has(n.id) || ++visits > 200) return;
        if (n.kids.length) ev(n.id, null, `Visit ${label(n)}: its sources are rewritten first.`, "visit");
        n.kids.forEach((k, i) => td(k, (m) => { n.kids[i] = m; }));
        const r = tryRules(n);
        if (!r) { done.add(n.id); if (n.kids.length) ev(n.id, null, `Sources done. No rule matches ${str(n)}, so this node is final.`, "none"); return; }
        ev(n.id, r.i, `Rule “${RULES[r.i].name}” matches ${str(n)}.`, "match");
        set(r.out);
        ev(r.out.id, r.i, `Replaced by ${str(r.out)}. The result is rewritten again (until no rule applies), because a rule can create new opportunities.`, "result");
        td(r.out, set);
      }
      // bottom_up mode: apply rules to the node itself to a fixed point, then descend (unified_rewrite stage 0 with bpm)
      function bu(n, set) {
        if (done.has(n.id) || ++visits > 200) return;
        let cur = n;
        ev(cur.id, null, `Visit ${label(cur)}: bottom_up tries the rules on the node before its sources.`, "visit");
        for (;;) {
          const r = tryRules(cur);
          if (!r) break;
          ev(cur.id, r.i, `Rule “${RULES[r.i].name}” matches ${str(cur)}.`, "match");
          set(cur = r.out);
          ev(cur.id, r.i, `Replaced by ${str(cur)}; rules are retried on it (fixed point).`, "result");
          if (done.has(cur.id)) return;
        }
        const before = cur.kids.map(key).join("|");
        cur.kids.forEach((k, i) => bu(k, (m) => { cur.kids[i] = m; }));
        if (cur.kids.map(key).join("|") !== before && tryRules(cur)) { ev(cur.id, null, `Its sources changed, so the rebuilt ${label(cur)} goes through the rules again.`, "visit"); return bu(cur, set); }
        done.add(cur.id);
        if (cur.kids.length) ev(cur.id, null, `${str(cur)} is final.`, "none");
      }
      (mode === "td" ? td : bu)(holder.kids[0], (m) => { holder.kids[0] = m; });
      ev(null, null, `Fixed point: no rule matches anywhere. Result: ${str(holder.kids[0])}. Rules fired: ${events.filter((e) => e.kind === "match").length}.`, "end");
      show(0, false);
    }
    function layout(t) {
      const nodes = [], edges = [];
      let leaf = 0;
      const depthOf = (n) => 1 + Math.max(0, ...n.kids.map(depthOf));
      const D = depthOf(t), leaves = (function c(n) { return n.kids.length ? n.kids.reduce((s, k) => s + c(k), 0) : 1; })(t);
      const dx = Math.min(90, 600 / Math.max(1, leaves)), dy = Math.min(80, 300 / Math.max(1, D - 1 || 1)), x0 = 320 - (leaves - 1) * dx / 2;
      (function place(n, d) {
        let x;
        if (!n.kids.length) x = x0 + dx * leaf++;
        else { const xs = n.kids.map((k) => place(k, d + 1)); x = (Math.min(...xs) + Math.max(...xs)) / 2; }
        nodes.push({ id: n.id, x, y: 24 + d * dy, label: label(n), w: n.op === "WHERE" ? 62 : 46, h: 28, cls: n.op === "CONST" ? "const" : n.op === "VAR" ? "var" : "" });
        n.kids.forEach((k, i) => edges.push({ from: k.id, to: n.id, k: ":" + i }));
        return x;
      })(t, 0);
      return { nodes, edges };
    }
    function show(i, anim = true) {
      cur = Math.max(0, Math.min(events.length - 1, i));
      const e = events[cur], st = layout(e.tree);
      st.nodes.forEach((n) => { if (n.id === e.focus) n.cls += e.kind === "match" ? " match" : e.kind === "result" ? " hot" : " focus"; });
      sc.set(st, anim);
      root.querySelector(".stepc").textContent = `step ${cur + 1}/${events.length}`;
      root.querySelector(".toy-msg").innerHTML = esc(e.msg);
      root.querySelector(".toy-expr").textContent = str(e.tree);
      root.querySelectorAll(".rule").forEach((r) => r.classList.toggle("on", e.rule != null && +r.dataset.r === e.rule && (e.kind === "match" || e.kind === "result")));
    }
    root.querySelector(".preset").onchange = run;
    root.querySelectorAll("[data-m]").forEach((b) => (b.onclick = () => { mode = b.dataset.m; root.querySelectorAll("[data-m]").forEach((x) => x.classList.toggle("on", x === b)); run(); }));
    root.querySelector(".prev").onclick = () => show(cur - 1);
    root.querySelector(".next").onclick = () => show(cur + 1);
    root.querySelector(".play").onclick = (e) => {
      if (timer) { clearInterval(timer); timer = null; e.target.textContent = "▶ play"; return; }
      if (cur >= events.length - 1) show(0);
      e.target.textContent = "❚❚ pause";
      timer = setInterval(() => { if (cur >= events.length - 1) { clearInterval(timer); timer = null; e.target.textContent = "▶ play"; } else show(cur + 1); }, 1300);
    };
    run();
  })();
});
