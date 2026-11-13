/* Shared runtime for "From Python to SASS".
 *
 * Data: every element with data-key / data-keys names captures ("matmul/default", "add/torch", ...).
 * book.js loads data/<example>__<variant>.js for each (a <script> tag, so file:// works), then mounts
 * the widgets below. Chapter-specific animations live in assets/chNN.js and run after BOOK.ready.
 *
 * Widgets (put the class on a <div>):
 *   .pipeline      data-active="tensor|schedule|codegen|render|compile|run"   animated overview
 *   .pass-stepper  data-key  [data-kernel=N|all] [data-stages="schedule,codegen"] [data-start="pass name"]
 *   .kernel-view   data-key  [data-kernel=N] [data-tabs="ast,uops,source,ptx,sass"] [data-tab=default]
 *                  [data-highlight="regex"]  marks matching lines
 *   .torch-view    data-key="x/torch" [data-tabs="eager,fx,pre,post,output,ttir,ttgir,llir,ptx,sass"]
 *   .bars          data-keys="a/default,b/torch:eager,b/torch:inductor" [data-labels="..|.."] [data-title]
 *   .sass-mix      data-key [data-kernel=N] [data-top=12]
 *   .code-compare  data-left="key|kernel|tab" data-right="key|kernel|tab" [data-labels="L|R"]
 *   pre.code.static[data-lang=python|c|ptx|sass|uop|llvm|mlir]   static code, highlighted in place
 *   a.src[data-src="tinygrad/codegen/__init__.py#L289"]  link to tinygrad at the captured commit
 */
(function () {
  const COMMIT = "6f87158d77f66a36d5f8bbe915170b24e2acabe8";
  const CHAPTERS = [
    ["index.html", "Preface: the whole pipeline"],
    ["ch01-journey.html", "1. The journey of a + b"],
    ["ch02-uops.html", "2. UOps and the rewrite engine"],
    ["ch03-scheduling.html", "3. Scheduling and fusion"],
    ["ch04-movement.html", "4. Movement ops become index math"],
    ["ch05-reductions.html", "5. Reductions"],
    ["ch06-optimizer.html", "6. The optimizer: opts, heuristics, BEAM"],
    ["ch07-tensor-cores.html", "7. Tensor cores"],
    ["ch08-decompositions.html", "8. Decompositions: exp, sin, integer math"],
    ["ch09-rendering.html", "9. Rendering and compiling: CUDA C, PTX, SASS"],
    ["ch10-runtime.html", "10. Runtime, TinyJit and autograd"],
    ["ch11-toolbox.html", "11. Appendix: the debug toolbox"],
  ];
  const STAGES = {
    tensor: ["Tensor graph", "var(--s-tensor)"], schedule: ["Schedule / rangeify", "var(--s-schedule)"],
    codegen: ["Codegen passes", "var(--s-codegen)"], program: ["Linearize / render", "var(--s-render)"],
  };
  // One line per named graph_rewrite pass, shown under the pass stepper. Kept to what the source says.
  const PASS_DESC = {
    "number the uops": "Tags each tensor UOp so later passes can map results back to the Tensors that asked for them.",
    "early transform tensor graph": "Cleans the lazy tensor graph and marks what must be realized (the outputs become STOREs into buffers).",
    "finalize call": "Collects the buffers the computation reads and writes.",
    "replace bufs": "Replaces concrete buffers with PARAMs and wraps the graph in a CALL, so the schedule can be cached and reused.",
    "multi_pm": "Multi-device rewrites. With one device this changes nothing.",
    "earliest rewrites": "Early simplifications on the tensor graph, including splitting very large reductions into two stages.",
    "convert copy to store": "Device-to-device COPYs become STOREs.",
    "get realize": "Decides which nodes must be materialized in memory (realized) and which can be fused into their consumers.",
    "minimum_valid": "Symbolic helper used by rangeify for padded regions.",
    "reshape": "Symbolic helper: rewrites index expressions through a RESHAPE.",
    "apply rangeify": "Rangeify: every node gets RANGE loop variables, and movement ops (reshape, permute, expand, pad, shrink) become index arithmetic.",
    "add device to deviceless": "Fills in the device on nodes that do not have one yet.",
    "symbolic+reduce_collapse+debuf": "Simplifies the index math and removes buffers that are cheaper to recompute than to store.",
    "limit buffers": "Splits kernels that would read more buffers than the device allows.",
    "stage to store": "Every remaining STAGE (a value that needs its own buffer) becomes BUFFER + STORE + END + AFTER.",
    "kernel split": "Cuts out one kernel: the STORE and its loops become a SINK with KernelInfo.",
    "split kernels": "Splits the graph at each store with no open ranges: one kernel per store.",
    "remove indexing from call args": "Cleans kernel arguments.",
    "schedule to linear": "Orders the kernels into a LINEAR list of CALLs, respecting their dependencies.",
    "params to buffers": "Binds the kernels' PARAMs back to real buffers.",
    "resolve linear call": "Inlines the cached schedule for this call.",
    "create COPY kernels for SDMA": "Turns plain copies into DMA copy calls where the device supports it.",
    "early movement ops": "Codegen starts: pushes any remaining movement ops into index arithmetic.",
    "load collapse": "Collapses reductions over loads that can be computed in closed form.",
    "split ranges": "Splits ranges where an index uses floor division or modulo, so each piece is a plain loop.",
    "initial symbolic": "Symbolic simplification of the index expressions.",
    "simplify ranges": "Merges adjacent loops that can be one loop (a contiguous 1024x1024 becomes one range of 1048576).",
    "postopt symbolic": "After the optimizer applied its opts (UPCAST, LOCAL, GROUP, UNROLL, TC): simplify again.",
    "expander": "UPCAST and UNROLL axes stop being loops: their values are expanded into vectors of constants.",
    "remove reduces": "REDUCE nodes become register accumulators (and, for GROUP reductions, a shared-memory buffer).",
    "add local buffers": "Allocates shared-memory (LOCAL) buffers the opts asked for.",
    "add gpudims": "GLOBAL and LOCAL ranges become SPECIAL uops: blockIdx (gidx) and threadIdx (lidx).",
    "*** expand broadcast / add loads": "Inserts explicit LOADs from memory and broadcasts mismatched vector widths.",
    "devectorize2": "Splits vector ops into scalar ops where the target cannot do them as vectors.",
    "early symbolic": "Symbolic simplification after devectorizing.",
    "add images": "Image-texture handling (only for image dtypes) and folding of vector loads.",
    "extra symbolic": "Another round of index simplification.",
    "lower all index dtypes": "The abstract index type becomes int32 (or int64 where values may not fit).",
    "final symbolic": "Last symbolic cleanup of index math.",
    "cast float alu operands": "Casts ALU operands to the dtype the hardware computes in.",
    "early decompositions": "Rewrites ops the target lacks into ops it has (e.g. MAX as compare + where).",
    "decomp dtypes": "Emulates dtypes the target lacks, and commits weak types to real ones.",
    "late decompositions": "Late op rewrites: transcendentals in software when needed, multiply by power of two as shift, and so on.",
    "move gates from index": "Moves validity masks off the INDEX onto the LOAD or STORE they guard.",
    "final rewrite": "Target-specific final rewrites (vector loads become SHRINK + LOAD, etc.).",
    "casted consts": "Folds casts of constants.",
    "add implicit barriers": "Inserts barriers between shared-memory writes and reads (__syncthreads).",
    "add control flow": "Orders loops and adds the ENDs: the graph is now ready to be linearized.",
    "number params with -1": "Numbers the kernel arguments.",
    "linearize/render": "Toposorts the graph into a list of uops, renders source (CUDA C or PTX) and compiles it.",
    "optimize local size": "Times a few local sizes when the kernel did not choose one.",
  };

  const BOOK = window.BOOK = { data: {}, commit: COMMIT, chapters: CHAPTERS, stages: STAGES, passDesc: PASS_DESC, _ready: [] };
  BOOK.register = (key, obj) => { BOOK.data[key] = obj; };
  BOOK.ready = (fn) => BOOK._ready.push(fn);
  BOOK.get = (key) => { const d = BOOK.data[key]; if (!d) throw new Error("missing capture " + key); return d; };
  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  BOOK.esc = esc;
  BOOK.fmtUs = (us) => us == null ? "n/a" : us >= 1000 ? (us / 1000).toFixed(2) + " ms" : us.toFixed(1) + " µs";

  // ---------- highlighting ----------
  const RULES = {
    python: [[/#.*$/m, "c"], [/("""[\s\S]*?"""|'[^'\n]*'|"[^"\n]*")/, "s"], [/\b(def|return|for|in|if|else|lambda|import|from|as|with|class|None|True|False|and|or|not)\b/, "k"],
      [/\bOps\.[A-Z_0-9]+\b/, "op"], [/\bdtypes\.\w+\b/, "t"], [/\b\d+(\.\d+)?(e-?\d+)?\b/, "n"], [/\b[A-Za-z_]\w*(?=\()/, "f"]],
    c: [[/\/\/.*$|\/\*[\s\S]*?\*\//m, "c"], [/"[^"\n]*"/, "s"], [/#\w+/, "k"],
      [/\b(extern|__global__|__device__|__shared__|__launch_bounds__|__forceinline__|for|if|else|return|const|template|class|union|typedef|struct|unsigned|asm|volatile)\b/, "k"],
      [/\b(float4?|half[248]?|int|uint|bool|void|float2|__half|__nv_bfloat16|double|long|char|unsigned char)\b/, "t"],
      [/\b(blockIdx|threadIdx|blockDim|__syncthreads|make_float4|make_half2)\b/, "r"], [/\b0x[0-9a-fA-F]+|\b\d+(\.\d+)?f?\b/, "n"], [/\b[A-Za-z_]\w*(?=\()/, "f"]],
    ptx: [[/\/\/.*$/m, "c"], [/^\s*\.[a-z_]+\b|\.(reg|param|global|shared|local|u32|s32|u64|s64|f32|f16|f16x2|b16|b32|b64|pred|v4|v2|approx|rn|lo|wide|ftz|sat|sync|aligned|row|col)\b/m, "t"],
      [/%[A-Za-z_][\w.]*|%r[dsf]?\d+|%p\d+/, "r"], [/\b(ld|st|mov|add|sub|mul|mad|fma|setp|selp|bra|cvt|cvta|shl|shr|and|or|xor|max|min|ex2|lg2|sin|cos|rcp|sqrt|mma|ret|bar|div|rem|neg|abs|not|shfl|atom|red)\b/, "op"],
      [/\b0[fFdD][0-9A-Fa-f]{8,16}\b|\b0x[0-9a-fA-F]+|-?\b\d+\b/, "n"], [/\$L__\w+|\w+:/, "k"]],
    sass: [[/\/\*[0-9a-f]{4}\*\//, "c"], [/Function :.*$|\.(target|headerflags).*$|code for.*$/m, "k"],
      [/\b[A-Z][A-Z0-9]*(?:\.[A-Z0-9_]+)*(?=\s)/, "op"], [/\b(R\d+|RZ|UR\d+|URZ|P\d|PT|UP\d|UPT|SR_\w+)\b/, "r"], [/c\[0x[0-9a-f]+\]\[0x[0-9a-f]+\]|\b0x[0-9a-f]+\b|-?\b\d+(\.\d+)?\b/, "n"]],
    llvm: [[/;.*$/m, "c"], [/"[^"\n]*"/, "s"], [/%[\w.]+|@[\w.]+/, "r"], [/\b(define|declare|ret|br|label|call|load|store|getelementptr|fadd|fmul|fsub|add|mul|icmp|fcmp|select|phi|bitcast|zext|sext|and|or|shl|lshr|tail|inbounds)\b/, "op"],
      [/\b(i1|i8|i16|i32|i64|float|half|double|ptr|void|<\d+ x \w+>)\b/, "t"], [/\b\d+(\.\d+)?\b/, "n"]],
    mlir: [[/\/\/.*$/m, "c"], [/"[^"\n]*"/, "s"], [/%[\w#]+|#\w+/, "r"], [/\b(tt|arith|scf|ttg|math|nvgpu|triton_gpu)\.\w+/, "op"], [/\b(tensor|i1|i32|i64|f32|f16|bf16|ptr)\b/, "t"], [/\b\d+(\.\d+)?\b/, "n"]],
  };
  RULES.uop = RULES.python; RULES.cuda = RULES.c; RULES.text = [];
  function highlight(text, lang) {
    const rules = RULES[lang] || [];
    if (!rules.length) return esc(text);
    const big = new RegExp(rules.map((r) => "(" + r[0].source + ")").join("|"), "gm");
    const counts = rules.map((r) => new RegExp(r[0].source + "|").exec("").length);  // groups per rule
    let out = "", last = 0, m;
    while ((m = big.exec(text))) {
      if (m[0] === "") { big.lastIndex++; continue; }
      out += esc(text.slice(last, m.index));
      let g = 1, cls = "";
      for (let i = 0; i < rules.length; i++) { if (m[g] !== undefined) { cls = rules[i][1]; break; } g += counts[i]; }
      out += `<span class="tk-${cls}">${esc(m[0])}</span>`;
      last = m.index + m[0].length;
    }
    return out + esc(text.slice(last));
  }
  BOOK.highlight = highlight;
  // Renders code as one span per line, so lines can be marked (diff additions, highlights).
  BOOK.codeHTML = (text, lang, marks) => highlight(text, lang).split("\n").map((l, i) => `<span class="ln${marks && marks[i] ? " " + marks[i] : ""}">${l || " "}</span>`).join("");
  BOOK.codeBlock = (text, lang, opts = {}) => {
    const lines = text.split("\n"), marks = lines.map((l) => (opts.highlight && new RegExp(opts.highlight).test(l) ? "hl" : ""));
    return `<pre class="code">${BOOK.codeHTML(text, lang, marks)}</pre>` + (opts.note ? `<div class="code-note">${opts.note}</div>` : "");
  };
  const clipNote = (c) => (c && c.clipped ? `Showing ${c.text.split("\n").length} of ${c.lines} lines.` : "");

  // line diff (LCS on lines): marks lines of `b` that are not in `a`
  function addedLines(a, b) {
    const A = a.split("\n"), B = b.split("\n");
    if (A.length * B.length > 4e6) return B.map((l) => (A.includes(l) ? "" : "add"));
    const n = A.length, m = B.length, dp = Array.from({ length: n + 1 }, () => new Uint16Array(m + 1));
    for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--) dp[i][j] = A[i] === B[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
    const marks = new Array(m).fill("add");
    let i = 0, j = 0;
    while (i < n && j < m) { if (A[i] === B[j]) { marks[j] = ""; i++; j++; } else if (dp[i + 1][j] >= dp[i][j + 1]) i++; else j++; }
    return marks;
  }
  BOOK.addedLines = addedLines;

  // ---------- widgets ----------
  function el(html) { const t = document.createElement("template"); t.innerHTML = html.trim(); return t.content.firstElementChild; }

  function mountPipeline(root) {
    const stages = [["tensor", "Tensor", "lazy UOp graph"], ["schedule", "Schedule", "rangeify, fusion"], ["codegen", "Codegen", "opts, expand, lower"],
      ["render", "Render", "CUDA C / PTX"], ["compile", "Compile", "NVRTC, ptxas"], ["run", "Run", "cuLaunchKernel"]];
    const W = 150, G = 34, H = 64, active = (root.dataset.active || "").split(",");
    let svg = `<svg viewBox="0 0 ${stages.length * (W + G) - G + 20} ${H + 40}"><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="#33445a"/></marker></defs>`;
    stages.forEach(([id, name, sub], i) => {
      const x = 10 + i * (W + G), on = active.includes(id) || !root.dataset.active;
      svg += `<g class="stage ${active.includes(id) ? "active" : ""}" style="opacity:${on ? 1 : 0.45}"><rect x="${x}" y="14" width="${W}" height="${H}" rx="10" style="stroke:var(--s-${id})"/>` +
        `<text x="${x + W / 2}" y="42">${name}</text><text class="sub" x="${x + W / 2}" y="60">${sub}</text></g>`;
      if (i) svg += `<path class="link" d="M${x - G + 2} ${14 + H / 2}H${x - 4}"/>`;
    });
    svg += `<circle class="packet" r="6" cy="${14 + H / 2}" fill="#fff"><animate attributeName="cx" values="10;${stages.length * (W + G) - G + 10}" dur="6s" repeatCount="indefinite"/>` +
      `<animate attributeName="fill" values="#c084fc;#60a5fa;#34d399;#fbbf24;#fb923c;#f87171" dur="6s" repeatCount="indefinite"/></circle></svg>`;
    root.classList.add("widget"); root.innerHTML = `<div class="widget-body">${svg}</div>`;
  }

  function mountPassStepper(root) {
    const d = BOOK.get(root.dataset.key), want = root.dataset.kernel ?? "all", stages = (root.dataset.stages || "tensor,schedule,codegen,program").split(",");
    // resolve the IR after each pass: unchanged passes carry no copy, so reuse the previous one in that kernel
    let lastIr = { null: d.tensor_graph.text };
    const all = d.passes.map((p) => {
      const k = p.kernel == null ? "s" : p.kernel;
      if (p.ir) lastIr[k] = p.ir.text;
      else if (lastIr[k] === undefined) lastIr[k] = p.kernel != null && d.kernels[p.kernel] ? d.kernels[p.kernel].ast.text : lastIr.s || lastIr[null];
      if (p.kernel == null) lastIr.s = lastIr[k];
      return { ...p, text: lastIr[k], clipped: p.ir && p.ir.clipped, lines: p.ir ? p.ir.lines : null };
    });
    const ps = all.filter((p) => stages.includes(p.stage) && (want === "all" || p.kernel == null || String(p.kernel) === want) && p.name !== "linearize/render");
    const maxN = Math.max(...ps.map((p) => p.nodes));
    root.classList.add("widget");
    root.innerHTML = `<div class="widget-head"><span class="title">${esc(root.dataset.title || "Every pass, in order: " + root.dataset.key)}</span>
      <button class="btn prev">◀</button><button class="btn play">▶ play</button><button class="btn next">▶</button>
      <input type="range" min="0" max="${ps.length - 1}" value="0" style="flex:1;min-width:120px">
      <label style="font-size:12px"><input type="checkbox" class="skip" checked> skip passes that change nothing</label></div>
      <div class="pass-strip">${ps.map((p, i) => `<div class="bar ${p.changed ? "" : "same"}" data-i="${i}" title="${esc(p.name)} (${p.nodes} nodes)" style="height:${8 + 58 * p.nodes / maxN}px;background:${STAGES[p.stage][1]}"></div>`).join("")}</div>
      <div class="pass-legend">${stages.map((s) => `<span><i style="background:${STAGES[s][1]}"></i>${STAGES[s][0]}</span>`).join("")}<span>bar height = number of UOps in the graph after the pass; faded = pass changed nothing</span></div>
      <div class="pass-info"></div><div class="pass-desc"></div><div class="ir"></div>`;
    const range = root.querySelector("input[type=range]"), bars = [...root.querySelectorAll(".bar")], skip = root.querySelector(".skip");
    let cur = 0, timer = null;
    function show(i) {
      cur = Math.max(0, Math.min(ps.length - 1, i)); range.value = cur;
      bars.forEach((b, j) => b.classList.toggle("cur", j === cur));
      const p = ps[cur], prev = cur > 0 ? ps[cur - 1].text : "";
      const kname = p.kernel != null && d.kernels[p.kernel] ? ` · kernel ${p.kernel} <span class="chip">${esc(d.kernels[p.kernel].name || "")}</span>` : "";
      root.querySelector(".pass-info").innerHTML = `<span class="chip" style="background:${STAGES[p.stage][1]};color:#081018">${STAGES[p.stage][0]}</span>
        <span class="pname">${esc(p.name)}</span><span>pass ${cur + 1} / ${ps.length}${kname}</span><span>${p.nodes} UOps</span>
        <span class="chip ${p.changed ? "ok" : ""}">${p.changed ? "changed the graph" : "no change"}</span><span>${p.ms} ms</span>`;
      root.querySelector(".pass-desc").textContent = PASS_DESC[p.name] || "";
      const marks = cur > 0 && p.changed ? addedLines(prev, p.text) : null;
      root.querySelector(".ir").innerHTML = `<pre class="code">${BOOK.codeHTML(p.text, "uop", marks)}</pre><div class="code-note">IR after this pass, printed by tinygrad's <code>pyrender</code>. Green lines are new since the previous pass. ${p.clipped ? `Showing 400 of ${p.lines} lines.` : ""}</div>`;
    }
    const step = (dir) => { let i = cur + dir; if (skip.checked) while (i > 0 && i < ps.length - 1 && !ps[i].changed) i += dir; show(i); };
    root.querySelector(".prev").onclick = () => step(-1);
    root.querySelector(".next").onclick = () => step(1);
    range.oninput = () => show(+range.value);
    root.querySelector(".pass-strip").onclick = (e) => { if (e.target.dataset.i) show(+e.target.dataset.i); };
    root.querySelector(".play").onclick = (e) => {
      if (timer) { clearInterval(timer); timer = null; e.target.textContent = "▶ play"; return; }
      e.target.textContent = "❚❚ pause";
      timer = setInterval(() => { if (cur >= ps.length - 1) { clearInterval(timer); timer = null; e.target.textContent = "▶ play"; } else step(1); }, 900);
    };
    const start = root.dataset.start ? ps.findIndex((p) => p.name === root.dataset.start) : 0;
    show(Math.max(0, start));
  }

  const KTABS = { ast: ["Kernel AST", "uop"], uops: ["Linear uops", "text"], source: ["Source", "c"], ptx: ["PTX", "ptx"], sass: ["SASS", "sass"], ptxas: ["ptxas -v", "text"] };
  function kernelText(k, tab) {
    if (tab === "ast") return [k.ast.text, clipNote(k.ast)];
    if (tab === "uops") return [k.uops ? k.uops.text : "", clipNote(k.uops) + " Columns: index, op, dtype, sources (by index), arg."];
    if (tab === "source") return [k.source || "", ""];
    if (tab === "ptx") return [k.ptx ? k.ptx.text : "(no PTX)", clipNote(k.ptx)];
    if (tab === "sass") return [k.sass ? k.sass.text : "(no SASS)", clipNote(k.sass) + " Disassembled with cuobjdump -sass after ptxas -arch=sm_89."];
    if (tab === "ptxas") return [k.ptxas_info || "", ""];
    return ["", ""];
  }
  BOOK.kernelText = kernelText;
  function mountKernelView(root) {
    const d = BOOK.get(root.dataset.key), tabs = (root.dataset.tabs || "ast,uops,source,ptx,sass").split(",");
    let ki = +(root.dataset.kernel || 0), tab = root.dataset.tab || tabs[0];
    root.classList.add("widget");
    function render() {
      const k = d.kernels[ki], isPtx = (d.env && d.env.DEV || "").includes("PTX");
      const lang = tab === "source" ? (isPtx ? "ptx" : "c") : KTABS[tab][1];
      const [text, note] = kernelText(k, tab);
      const gflops = k.estimates && k.time_us ? (Number(k.estimates.ops) / (k.time_us * 1e3)).toFixed(1) : null;
      const gbs = k.estimates && k.time_us ? (Number(k.estimates.mem) / (k.time_us * 1e3)).toFixed(0) : null;
      root.innerHTML = `<div class="widget-head"><span class="title">${esc(root.dataset.title || root.dataset.key)}</span>
        ${d.kernels.length > 1 ? `<select class="btn ksel">${d.kernels.map((x, i) => `<option value="${i}" ${i === ki ? "selected" : ""}>kernel ${i}: ${esc(x.name)}</option>`).join("")}</select>` : ""}
        <div class="tabs">${tabs.map((t) => `<button class="btn ${t === tab ? "on" : ""}" data-t="${t}">${t === "source" && isPtx ? "PTX (tinygrad)" : KTABS[t][0]}</button>`).join("")}</div>
        <span class="chip ${d.check.ok ? "ok" : "bad"}" title="max abs error ${d.check.max_abs_err}">${d.check.ok ? "matches numpy" : "MISMATCH"}</span></div>
        <div class="kstats"><div class="kstat"><span>kernel</span><b>${esc(k.name)}</b></div>
        <div class="kstat"><span>grid (blocks)</span><b>${k.global_size.join(" × ")}</b></div>
        <div class="kstat"><span>block (threads)</span><b>${k.local_size ? k.local_size.join(" × ") : "auto"}</b></div>
        <div class="kstat"><span>registers / thread</span><b>${k.regs ?? "n/a"}</b></div>
        <div class="kstat"><span>time on L4</span><b>${BOOK.fmtUs(k.time_us)}</b></div>
        ${gflops ? `<div class="kstat"><span>GFLOP/s · GB/s</span><b>${gflops} · ${gbs}</b></div>` : ""}
        <div class="kstat" style="grid-column:span 2"><span>applied opts</span><b style="font-size:12px">${esc((k.applied_opts || []).map((o) => o.replace("Opt(op=OptOps.", "").replace(/\)$/, "")).join(", ") || "none")}</b></div></div>
        ${BOOK.codeBlock(text, lang, { highlight: root.dataset.highlight, note })}`;
      root.querySelectorAll("[data-t]").forEach((b) => (b.onclick = () => { tab = b.dataset.t; render(); }));
      const s = root.querySelector(".ksel"); if (s) s.onchange = () => { ki = +s.value; render(); };
    }
    render();
  }

  const TTABS = { eager: "Eager kernels", fx: "FX graph", pre: "Inductor IR (pre-fusion)", post: "Inductor IR (post-fusion)", output: "Output code",
    ttir: "TTIR", ttgir: "TTGIR", llir: "LLVM IR", ptx: "PTX", sass: "SASS" };
  function mountTorchView(root) {
    const d = BOOK.get(root.dataset.key), tabs = (root.dataset.tabs || "eager,fx,post,output,ttir,ttgir,llir,ptx,sass").split(",");
    const tks = d.triton_kernels;
    let tab = root.dataset.tab || tabs[0], ki = 0;
    root.classList.add("widget");
    function render() {
      let text = "", lang = "text", note = "";
      const k = tks[ki], dbg = d.inductor_debug;
      if (tab === "eager") { text = d.eager_kernels.map((e) => `${BOOK.fmtUs(e.us).padStart(10)}  ${e.name}`).join("\n"); note = "CUDA kernels PyTorch eager mode launched, from torch.profiler, with device time on the L4."; }
      else if (tab === "fx") { text = dbg.fx_graph ? dbg.fx_graph.text : ""; lang = "python"; note = "The graph Dynamo traced from the Python function."; }
      else if (tab === "pre" || tab === "post") { const c = dbg[tab === "pre" ? "ir_pre_fusion" : "ir_post_fusion"]; text = c ? c.text : ""; note = "Inductor's scheduler nodes " + (tab === "pre" ? "before" : "after") + " fusion."; }
      else if (tab === "output") { text = dbg.output_code ? dbg.output_code.text : ""; lang = "python"; note = "Python module Inductor generated: Triton kernels as source strings plus the call() that launches them."; }
      else if (!k) { text = "(no Triton kernel: Inductor called an external library kernel, see Output code)"; }
      else { const c = k[tab]; text = c ? c.text : "(not captured)"; lang = { ttir: "mlir", ttgir: "mlir", llir: "llvm", ptx: "ptx", sass: "sass" }[tab]; note = clipNote(c); }
      const needK = ["ttir", "ttgir", "llir", "ptx", "sass"].includes(tab) && tks.length > 1;
      root.innerHTML = `<div class="widget-head"><span class="title">${esc(root.dataset.title || "PyTorch " + d.torch_version + ": " + d.example)}</span>
        ${needK ? `<select class="btn ksel">${tks.map((x, i) => `<option value="${i}" ${i === ki ? "selected" : ""}>${esc(x.name)} (${esc(x.cache_dir.slice(0, 6))})</option>`).join("")}</select>` : ""}
        <div class="tabs">${tabs.map((t) => `<button class="btn ${t === tab ? "on" : ""}" data-t="${t}">${TTABS[t]}</button>`).join("")}</div></div>
        ${BOOK.codeBlock(text, lang, { highlight: root.dataset.highlight, note })}`;
      root.querySelectorAll("[data-t]").forEach((b) => (b.onclick = () => { tab = b.dataset.t; render(); }));
      const s = root.querySelector(".ksel"); if (s) s.onchange = () => { ki = +s.value; render(); };
    }
    render();
  }

  // total time of a capture: tinygrad kernels, or torch eager / inductor launches
  BOOK.totalUs = (spec) => {
    const [key, part] = spec.split(":"), d = BOOK.get(key);
    if (d.kernels) return d.kernels.reduce((s, k) => s + (k.time_us || 0), 0);
    const ks = part === "inductor" ? d.inductor_kernels : d.eager_kernels;
    return ks.reduce((s, k) => s + (k.us || 0), 0);
  };
  function mountBars(root) {
    const specs = root.dataset.keys.split(","), labels = root.dataset.labels ? root.dataset.labels.split("|") : specs;
    const vals = specs.map(BOOK.totalUs), max = Math.max(...vals), colors = ["var(--s-codegen)", "var(--s-schedule)", "var(--s-render)", "var(--s-tensor)", "var(--s-compile)", "#ee4c2c"];
    root.classList.add("widget");
    root.innerHTML = `<div class="widget-head"><span class="title">${esc(root.dataset.title || "Time on the L4 (all kernels, best of 5 runs)")}</span></div><div class="widget-body">${specs.map((s, i) =>
      `<div class="row"><span>${esc(labels[i])}</span><div class="track"><div class="fill" data-w="${100 * vals[i] / max}" style="background:${s.includes("torch") ? "#ee4c2c" : colors[i % 5]}"></div></div><span class="val">${BOOK.fmtUs(vals[i])}</span></div>`).join("")}</div>`;
    whenVisible(root, () => root.querySelectorAll(".fill").forEach((f) => (f.style.width = Math.max(0.6, f.dataset.w) + "%")));
  }

  // instruction families for the SASS mix chart
  const FAMILY = (op) => /^(FFMA|FADD|FMUL|FMNMX|FSETP|FSEL|FCHK|FRND|F2F|F2I|I2F)/.test(op) ? ["float math", "#34d399"] : /^(HMMA|IMMA)/.test(op) ? ["tensor core", "#f472b6"] :
    /^MUFU/.test(op) ? ["special function (MUFU)", "#fbbf24"] : /^(LDG|STG|LD|ST|RED|ATOM)$/.test(op) ? ["global memory", "#60a5fa"] : /^(LDS|STS|LDSM)$/.test(op) ? ["shared memory", "#22d3ee"] :
    /^(BAR|BRA|EXIT|BSYNC|BSSY|WARPSYNC|RET|CALL|ISETP|PLOP3|SHFL|VOTE)/.test(op) ? ["control / sync", "#f87171"] : /^(IMAD|IADD3|LEA|SHF|LOP3|IABS|IMNMX|PRMT|SEL|MOV|IMUL|ULDC|UIADD3|UMOV|ULEA|USHF|ULOP3|UIMAD|S2R|CS2R|S2UR|SGXT)/.test(op) ? ["integer / address", "#a78bfa"] : ["other", "#64748b"];
  BOOK.family = FAMILY;
  function mountSassMix(root) {
    const d = BOOK.get(root.dataset.key), k = (d.kernels || d.triton_kernels)[+(root.dataset.kernel || 0)], top = +(root.dataset.top || 14);
    const mix = Object.entries(k.sass_mix || {}).filter(([op]) => op !== "NOP"), total = mix.reduce((s, [, c]) => s + c, 0), max = Math.max(...mix.map(([, c]) => c));
    root.classList.add("widget", "bars");
    root.innerHTML = `<div class="widget-head"><span class="title">${esc(root.dataset.title || "SASS instruction mix: " + (k.name || ""))}</span><span class="chip">${total} instructions (NOPs not counted)</span></div>
      <div class="widget-body">${mix.slice(0, top).map(([op, c]) => { const [fam, col] = FAMILY(op);
        return `<div class="row"><span class="mono">${op} <span style="color:var(--muted);font-family:var(--sans);font-size:12px">${fam}</span></span><div class="track"><div class="fill" data-w="${100 * c / max}" style="background:${col}"></div></div><span class="val">${c}</span></div>`; }).join("")}</div>`;
    whenVisible(root, () => root.querySelectorAll(".fill").forEach((f) => (f.style.width = f.dataset.w + "%")));
  }

  function mountCompare(root) {
    const side = (spec) => { const [key, kn, tab] = spec.split("|"), d = BOOK.get(key), k = d.kernels[+kn];
      const isPtx = (d.env && d.env.DEV || "").includes("PTX"), lang = tab === "source" ? (isPtx ? "ptx" : "c") : KTABS[tab][1];
      return [kernelText(k, tab)[0], lang]; };
    const [lt, ll] = side(root.dataset.left), [rt, rl] = side(root.dataset.right), labels = (root.dataset.labels || "left|right").split("|");
    root.classList.add("widget");
    root.innerHTML = `<div class="split"><div><div class="widget-head"><span class="title">${esc(labels[0])}</span><span class="chip">${lt.split("\n").length} lines</span></div>${BOOK.codeBlock(lt, ll, { highlight: root.dataset.highlight })}</div>
      <div><div class="widget-head"><span class="title">${esc(labels[1])}</span><span class="chip">${rt.split("\n").length} lines</span></div>${BOOK.codeBlock(rt, rl, { highlight: root.dataset.highlight })}</div></div>`;
  }

  function whenVisible(node, fn) {
    if (!("IntersectionObserver" in window)) return fn();
    const io = new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) { io.disconnect(); fn(); } }, { threshold: 0.2 });
    io.observe(node);
  }
  BOOK.whenVisible = whenVisible;

  function buildChrome() {
    const here = location.pathname.split("/").pop() || "index.html";
    const main = document.querySelector("main");
    if (!main || document.querySelector(".layout")) return;
    const layout = el(`<div class="layout"><nav class="toc"><a class="book-title" href="index.html">From Python to SASS</a><ol>${CHAPTERS.map(([href, t]) =>
      `<li><a href="${href}" class="${href === here ? "current" : ""}">${esc(t)}</a>${href === here ? '<div class="sections"></div>' : ""}</li>`).join("")}</ol></nav></div>`);
    main.parentNode.insertBefore(layout, main); layout.appendChild(main);
    const secs = layout.querySelector(".sections");
    main.querySelectorAll("h2").forEach((h, i) => { h.id = h.id || "s" + i; if (secs) secs.appendChild(el(`<a href="#${h.id}">${esc(h.textContent)}</a>`)); });
    const idx = CHAPTERS.findIndex(([h]) => h === here);
    if (idx >= 0) main.appendChild(el(`<div class="chapter-nav">${idx > 0 ? `<a href="${CHAPTERS[idx - 1][0]}">← ${esc(CHAPTERS[idx - 1][1])}</a>` : "<span></span>"}${idx < CHAPTERS.length - 1 ? `<a href="${CHAPTERS[idx + 1][0]}">${esc(CHAPTERS[idx + 1][1])} →</a>` : "<span></span>"}</div>`));
    document.body.prepend(el('<div class="progress"></div>'));
    const bar = document.querySelector(".progress"), links = secs ? [...secs.querySelectorAll("a")] : [];
    addEventListener("scroll", () => {
      const h = document.documentElement; bar.style.width = (100 * h.scrollTop / (h.scrollHeight - h.clientHeight || 1)) + "%";
      let cur = null; main.querySelectorAll("h2").forEach((x) => { if (x.getBoundingClientRect().top < 120) cur = x.id; });
      links.forEach((a) => a.classList.toggle("active", a.getAttribute("href") === "#" + cur));
    }, { passive: true });
  }

  function loadScript(src) { return new Promise((res, rej) => { const s = document.createElement("script"); s.src = src; s.onload = res; s.onerror = () => rej(new Error("failed to load " + src)); document.head.appendChild(s); }); }

  async function boot() {
    buildChrome();
    document.querySelectorAll("a.src[data-src]").forEach((a) => { const [p, l] = a.dataset.src.split("#"); a.href = `https://github.com/tinygrad/tinygrad/blob/${COMMIT}/${p}${l ? "#" + l : ""}`; a.target = "_blank"; if (!a.textContent) a.textContent = a.dataset.src.replace("#L", ":"); });
    document.querySelectorAll("pre.code.static").forEach((p) => { const t = p.textContent.replace(/^\n/, "").replace(/\s+$/, ""); p.innerHTML = BOOK.codeHTML(t, p.dataset.lang || "text"); });
    const keys = new Set();
    document.querySelectorAll("[data-key],[data-keys],[data-left],[data-right]").forEach((n) => {
      [n.dataset.key, ...(n.dataset.keys || "").split(","), (n.dataset.left || "").split("|")[0], (n.dataset.right || "").split("|")[0]]
        .filter(Boolean).forEach((k) => keys.add(k.split(":")[0]));
    });
    try { await Promise.all([...keys].map((k) => loadScript("data/" + k.replace("/", "__") + ".js"))); }
    catch (e) { document.querySelector("main").prepend(el(`<div class="callout warn">${esc(e.message)}</div>`)); }
    const mounts = [[".pipeline", mountPipeline], [".pass-stepper", mountPassStepper], [".kernel-view", mountKernelView], [".torch-view", mountTorchView],
      [".bars", mountBars], [".sass-mix", mountSassMix], [".code-compare", mountCompare]];
    for (const [sel, fn] of mounts) document.querySelectorAll(sel).forEach((n) => {
      try { fn(n); } catch (e) { console.error(e); n.innerHTML = `<div class="callout warn">widget failed: ${esc(e.message)}</div>`; }
    });
    const io = "IntersectionObserver" in window ? new IntersectionObserver((es) => es.forEach((e) => e.isIntersecting && e.target.classList.add("shown")), { threshold: 0.1 }) : null;
    document.querySelectorAll(".reveal").forEach((n) => (io ? io.observe(n) : n.classList.add("shown")));
    for (const fn of BOOK._ready) { try { fn(BOOK); } catch (e) { console.error(e); } }
    document.body.dataset.ready = "1";
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot); else boot();
})();
