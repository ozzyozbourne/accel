/* Chapter 9: rendering and compiling. Two custom figures, both built only from the captured text:
 *   #c9-linked  hover a line of the add kernel's CUDA C to light up its PTX and SASS (and back)
 *   #c9-ladder  one strand of the float4 add (index, load, add, store) lowered uop -> C -> PTX -> SASS
 * The correspondence tables below were written by hand by reading add_ptx/cuda and add_ptx/ptx; every
 * rule is a regex over the captured lines, and mount() warns if a rule matches nothing.
 */
(function () {
  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  // C lines of the kernel body (ids 0..7)
  const C_RULES = [/__global__/, /blockIdx\.x/, /threadIdx\.x/, /int alu0/, /float4 val0/, /float4 val1/, /make_float4/, /^}$/];
  const C_NOTE = [
    "Signature. The PARAMs become pointer arguments; __launch_bounds__(32) is the product of the local SPECIAL bounds.",
    "SPECIAL gidx0 → blockIdx.x. The comment is its bound (8192 blocks).",
    "SPECIAL lidx0 → threadIdx.x (32 threads per block).",
    "The index: (gidx0<<7)+(lidx0<<2), element offset of this thread's 4 floats. Multiplies by 128 and 4 were rewritten to shifts.",
    "float4 load from data1 (input a). One 16-byte load per thread.",
    "float4 load from data2 (input b).",
    "Four adds, packed with make_float4, stored with one 16-byte store into data0.",
    "End of kernel.",
  ];
  // [regex, C ids]; a rule may match several lines
  const MAP = {
    nvrtc_ptx: [[/\.visible \.entry|\.param \.u64 E_8192_32_4_param_0/, [0, 6]], [/\.param \.u64 E_8192_32_4_param_1/, [0, 4]], [/\.param \.u64 E_8192_32_4_param_2/, [0, 5]], [/\.maxntid/, [0]],
      [/ld\.param\.u64\s+%rd1,/, [0, 6]], [/ld\.param\.u64\s+%rd2,/, [0, 4]], [/ld\.param\.u64\s+%rd3,/, [0, 5]],
      [/cvta\.to\.global\.u64\s+%rd4, %rd1/, [6]], [/cvta\.to\.global\.u64\s+%rd5, %rd3/, [5]], [/cvta\.to\.global\.u64\s+%rd6, %rd2/, [4]],
      [/%r1, %ctaid\.x/, [1]], [/%r2, %tid\.x/, [2]], [/shl\.b32\s+%r[34],/, [3]], [/add\.s32\s+%r5,/, [3]], [/mul\.wide\.s32/, [3, 4, 5, 6]],
      [/add\.s64\s+%rd8,|ld\.global\.v4\.f32.*\[%rd8\]/, [4]], [/add\.s64\s+%rd9,|ld\.global\.v4\.f32.*\[%rd9\]/, [5]],
      [/add\.s64\s+%rd10,|add\.f32|st\.global/, [6]], [/^\s*ret;/, [7]]],
    nvrtc_sass: [[/MOV R1, c\[0x0\]\[0x28\]/, [0]], [/SR_CTAID\.X/, [1]], [/SR_TID\.X/, [2]], [/MOV R7, 0x4|ULDC\.64/, [4, 5, 6]],
      [/LEA R0, R0, R3, 0x5|SHF\.L\.U32 R6, R0, 0x2/, [3]], [/c\[0x0\]\[0x168\]|LDG\.E\.128 R8/, [4]], [/c\[0x0\]\[0x170\]|LDG\.E\.128 R12/, [5]],
      [/c\[0x0\]\[0x160\]|FADD|STG/, [6]], [/EXIT|BRA 0x/, [7]]],
    tg_ptx: [[/\.visible \.entry|\.maxntid|%cast_s32_0, 1048576/, [0]], [/\.param \.u64 data0|\[data0\+0\]/, [0, 6]], [/\.param \.u64 data1|\[data1\+0\]/, [0, 4]], [/\.param \.u64 data2|\[data2\+0\]/, [0, 5]],
      [/%cast_s32_1, 8192|%gidx0, %ctaid/, [1]], [/%cast_s32_2, 32;|%lidx0, %tid/, [2]], [/%cast_s32_3, 4;/, [4, 5, 6]],
      [/%cast_s32_4, 128|%cast_u32_0, 2U|shl\.b32|mad\.lo\.s32/, [3]], [/%bidx_u64_0[,\+]/, [4]], [/%bidx_u64_1[,\+]/, [5]],
      [/%cast_s32_[5-8],|add\.f32|%bidx_u64_2[,\+]/, [6]], [/^\s*ret;/, [7]]],
    tg_sass: [[/MOV R1, c\[0x0\]\[0x28\]/, [0]], [/SR_CTAID\.X/, [1]], [/SR_TID\.X/, [2]], [/MOV R7, 0x4|ULDC\.64|SHF\.R\.S32\.HI R0|SHF\.L\.U32 R17/, [4, 5, 6]],
      [/LEA R0, R0, R3, 0x5|SHF\.L\.U32 R6, R0, 0x2/, [3]], [/c\[0x0\]\[0x168\]|IADD3 R3, R3, R17|LDG\.E\.128 R8/, [4]], [/c\[0x0\]\[0x170\]|IADD3 R5, R17|LDG\.E\.128 R12/, [5]],
      [/c\[0x0\]\[0x160\]|IADD3 R7, R17|FADD|STG/, [6]], [/EXIT|BRA 0x/, [7]]],
  };
  const sassBody = (t) => t.split("\n").filter((l) => /\/\*[0-9a-f]{4}\*\//.test(l) && !/\bNOP\b/.test(l)).map((l) => l.replace(/^\s+/, "").replace(/\s{2,}/g, " "));
  const ptxBody = (t) => t.split("\n").filter((l) => l.trim() && !/^\/\//.test(l.trim()));
  function tag(lines, rules, missing, name) {
    const ids = lines.map(() => []);
    rules.forEach(([re, cs]) => { let hit = 0; lines.forEach((l, i) => { if (re.test(l)) { ids[i].push(...cs); hit++; } }); if (!hit) missing.push(name + " " + re); });
    return ids.map((a) => [...new Set(a)]);
  }

  function mountLinked(root) {
    const C = BOOK.get("add_ptx/cuda").kernels[0], T = BOOK.get("add_ptx/ptx").kernels[0];
    const cLines = C.source.split("\n"), missing = [];
    const cIds = cLines.map((l) => { const i = C_RULES.findIndex((r) => r.test(l.trim())); return i < 0 ? [] : [i]; });
    C_RULES.forEach((r, i) => { if (!cIds.some((x) => x[0] === i)) missing.push("C " + r); });
    const routes = {
      nvrtc: { ptx: ptxBody(C.ptx.text), sass: sassBody(C.sass.text), pl: "PTX from NVRTC", sl: "SASS (ptxas, sm_89)", regs: C.regs },
      tg: { ptx: ptxBody(T.ptx.text), sass: sassBody(T.sass.text), pl: "PTX from tinygrad's PTXRenderer", sl: "SASS (ptxas, sm_89)", regs: T.regs },
    };
    routes.nvrtc.pid = tag(routes.nvrtc.ptx, MAP.nvrtc_ptx, missing, "nvrtc ptx"); routes.nvrtc.sid = tag(routes.nvrtc.sass, MAP.nvrtc_sass, missing, "nvrtc sass");
    routes.tg.pid = tag(routes.tg.ptx, MAP.tg_ptx, missing, "tg ptx"); routes.tg.sid = tag(routes.tg.sass, MAP.tg_sass, missing, "tg sass");
    let route = "nvrtc", pinned = null;
    root.classList.add("widget");
    function col(title, lines, ids, lang, cls, chip) {
      return `<div class="c9-col ${cls}"><div class="widget-head"><span class="title">${esc(title)}</span>${chip ? `<span class="chip">${chip}</span>` : ""}</div><pre class="code">${lines.map((l, i) =>
        `<span class="ln c9-l${ids[i].length ? " c9-m" : ""}" data-ids="${ids[i].join(" ")}">${BOOK.highlight(l, lang) || " "}</span>`).join("")}</pre></div>`;
    }
    function render() {
      const R = routes[route];
      root.innerHTML = `<div class="widget-head"><span class="title">One kernel, three languages: hover a line (click to pin)</span>
        <div class="tabs"><button class="btn ${route === "nvrtc" ? "on" : ""}" data-r="nvrtc">CUDA C → NVRTC → ptxas</button><button class="btn ${route === "tg" ? "on" : ""}" data-r="tg">uops → tinygrad PTX → ptxas</button></div></div>
        <div class="c9-cols">${col("CUDA C (tinygrad's CUDARenderer)", cLines, cIds, "c", "c", "")}${col(R.pl, R.ptx, R.pid, "ptx", "p", R.ptx.length + " lines")}${col(R.sl, R.sass, R.sid, "sass", "s", R.sass.length + " instr · " + R.regs + " regs")}</div>
        <div class="c9-note">${route === "tg" ? "The tinygrad PTX is rendered from the same linear uops as the C on the left, not from the C; the left column stands for those uops. " : ""}Hover a CUDA C line of the kernel body.</div>`;
      root.querySelectorAll("[data-r]").forEach((b) => (b.onclick = () => { route = b.dataset.r; pinned = null; render(); }));
      const all = [...root.querySelectorAll(".c9-l")], note = root.querySelector(".c9-note");
      const light = (ids) => {
        all.forEach((l) => { const mine = l.dataset.ids ? l.dataset.ids.split(" ").map(Number) : []; l.classList.toggle("c9-on", ids.length > 0 && mine.some((x) => ids.includes(x))); });
        if (ids.length) {
          const np = root.querySelectorAll(".p .c9-on").length, ns = root.querySelectorAll(".s .c9-on").length;
          root.querySelectorAll(".c9-col pre").forEach((pre) => { // bring the first lit line of each column into view
            const on = pre.querySelector(".c9-on"); if (!on || pre.matches(":hover")) return;
            const top = on.offsetTop - pre.offsetTop; if (top < pre.scrollTop || top > pre.scrollTop + pre.clientHeight - 30) pre.scrollTop = Math.max(0, top - 60);
          });
          note.innerHTML = ids.map((i) => `<b>C line ${i + 1}.</b> ${esc(C_NOTE[i])}`).join(" ") + ` <span class="chip">${np} PTX · ${ns} SASS</span>`;
        }
      };
      all.forEach((l) => {
        const ids = l.dataset.ids ? l.dataset.ids.split(" ").map(Number) : [];
        l.onmouseenter = () => { if (pinned === null) light(ids); };
        l.onclick = () => { pinned = pinned === l ? null : l; light(pinned ? ids : []); };
      });
      root.querySelector(".c9-cols").onmouseleave = () => { if (pinned === null) light([]); };
      light([3]);
    }
    render();
    if (missing.length) root.insertAdjacentHTML("beforeend", `<div class="callout warn">mapping rules matched nothing: ${esc(missing.join("; "))}</div>`);
  }

  // ---------- the lowering ladder ----------
  const STRANDS = {
    "index": { uops: [17, 18, 19], c: /int alu0 = .*;/, ptx: [/shl\.b32\s+%r3.*/, /shl\.b32\s+%r4.*/, /add\.s32\s+%r5.*/, /mul\.wide\.s32.*/], sass: [/LEA R0, R0, R3, 0x5.*/, /SHF\.L\.U32 R6.*/],
      note: "Two shifts and an add in the uops. NVRTC keeps them as shl, shl, add plus a widening multiply by 4 bytes; ptxas notices (gidx·128 + lidx·4) = (gidx·32 + lidx)·4 and uses one LEA (shift-and-add) and one shift." },
    "load a": { uops: [20, 21], c: /float4 val0 = .*;/, ptx: [/cvta\.to\.global\.u64\s+%rd6.*/, /add\.s64\s+%rd8.*/, /ld\.global\.v4\.f32.*%rd8.*/], sass: [/IMAD\.WIDE R2, R6, R7, c\[0x0\]\[0x168\].*/, /LDG\.E\.128 R8.*/],
      note: "A SHRINK of 4 elements of the PARAM, then a LOAD of the whole float4. In SASS the pointer comes straight from constant bank 0 (c[0x0][0x168] is the second kernel argument) and one IMAD.WIDE computes base + index·4; the load is a single 128-bit LDG." },
    "add (lane x)": { uops: [26, 34, 38], c: /\(val0\.x\+val1\.x\)/, ptx: [/add\.f32\s+%f20, %f1, %f9.*/], sass: [/FADD R8, R8, R12.*/],
      note: "INDEX picks lane 0 of each loaded vector, ADD adds them. In C a lane is .x; in PTX each lane is its own scalar register (%f1, %f9); in SASS the 128-bit load wrote four consecutive registers R8..R11 and R12..R15, so lane x is R8 and R12." },
    "store": { uops: [42, 43, 44], c: /\*\(\(float4\*\)\(\(data0_1048576\+alu0\)\)\) = make_float4\(/, ptx: [/add\.s64\s+%rd10.*/, /st\.global\.v4\.f32.*/], sass: [/IMAD\.WIDE R6, R6, R7, c\[0x0\]\[0x160\].*/, /STG\.E\.128.*/],
      note: "STACK packs the four sums into a vector, SHRINK addresses data0, STORE writes it. make_float4 compiles to nothing: the four FADD results already sit in R8..R11, which is exactly the register quad STG.E.128 stores." },
  };
  const LEVELS = [["UOp (linearized)", "var(--s-codegen)", "text"], ["CUDA C", "var(--s-render)", "c"], ["PTX (NVRTC)", "var(--s-compile)", "ptx"], ["SASS", "var(--s-run)", "sass"]];
  function mountLadder(root) {
    const k = BOOK.get("add_ptx/cuda").kernels[0], uopL = k.uops.text.split("\n"), missing = [];
    const pick = (text, re) => { for (const l of text.split("\n")) { const m = l.match(re); if (m) return m[0].trim().replace(/\s{2,}/g, " "); } missing.push(String(re)); return "?"; };
    const data = {};
    for (const [name, s] of Object.entries(STRANDS)) {
      data[name] = [
        s.uops.map((i) => { const l = uopL.find((x) => new RegExp("^\\s*" + i + " ").test(x)); if (!l) missing.push("uop " + i); return (l || "?").trim().replace(/\s{2,}/g, "  ").replace(/\s+None$/, ""); }),
        [pick(k.source, s.c) + (name === "store" ? "…" : "")],
        s.ptx.map((r) => pick(k.ptx.text, r)), s.sass.map((r) => pick(k.sass.text, r))];
    }
    root.classList.add("widget");
    root.innerHTML = `<div class="widget-head"><span class="title">The lowering ladder: the float4 add, one strand at a time</span>
      <div class="tabs">${Object.keys(STRANDS).map((n) => `<button class="btn" data-s="${n}">${n}</button>`).join("")}</div>
      <button class="btn c9-play">▶ play all</button></div>
      <div class="widget-body c9-ladder">${LEVELS.map(([n, c], i) => `<div class="c9-rung" style="--c:${c}"><div class="c9-lab">${n}</div><pre class="c9-txt" data-l="${i}"></pre></div>`).join("")}</div>
      <div class="c9-note c9-lnote"></div>`;
    const lad = root.querySelector(".c9-ladder"), txt = [...root.querySelectorAll(".c9-txt")];
    let timers = [];
    const clear = () => { timers.forEach(clearTimeout); timers = []; lad.querySelectorAll(".c9-ghost").forEach((g) => g.remove()); };
    function show(name, then) {
      clear();
      root.querySelectorAll("[data-s]").forEach((b) => b.classList.toggle("on", b.dataset.s === name));
      root.querySelector(".c9-lnote").textContent = STRANDS[name].note;
      txt.forEach((t, i) => { t.innerHTML = BOOK.codeHTML(data[name][i].join("\n"), LEVELS[i][2]); t.classList.remove("in"); });
      timers.push(setTimeout(() => txt[0].classList.add("in"), 60));
      for (let i = 1; i < 4; i++) timers.push(setTimeout(() => {
        const from = txt[i - 1], to = txt[i], lb = lad.getBoundingClientRect(), fb = from.getBoundingClientRect(), tb = to.getBoundingClientRect();
        const g = from.cloneNode(true); g.classList.add("c9-ghost"); g.classList.remove("in");
        Object.assign(g.style, { left: fb.left - lb.left + "px", top: fb.top - lb.top + "px", width: fb.width + "px" });
        lad.appendChild(g);
        requestAnimationFrame(() => requestAnimationFrame(() => { g.style.transform = `translate(${tb.left - fb.left}px, ${tb.top - fb.top}px) scale(.96)`; g.style.opacity = "0"; }));
        timers.push(setTimeout(() => { to.classList.add("in"); }, 380));
        timers.push(setTimeout(() => g.remove(), 900));
      }, 200 + i * 950));
      if (then) timers.push(setTimeout(then, 200 + 4 * 950 + 1600));
    }
    root.querySelectorAll("[data-s]").forEach((b) => (b.onclick = () => show(b.dataset.s)));
    root.querySelector(".c9-play").onclick = () => { const names = Object.keys(STRANDS); let i = 0; const next = () => { if (i < names.length) show(names[i++], next); }; next(); };
    BOOK.whenVisible(root, () => show("index"));
    if (missing.length) root.insertAdjacentHTML("beforeend", `<div class="callout warn">ladder lines not found: ${esc(missing.join("; "))}</div>`);
  }

  BOOK.ready(() => {
    for (const [sel, fn] of [["#c9-linked", mountLinked], ["#c9-ladder", mountLadder]]) {
      const n = document.querySelector(sel); if (!n) continue;
      try { fn(n); } catch (e) { console.error(e); n.innerHTML = `<div class="callout warn">figure failed: ${esc(e.message)}</div>`; }
    }
  });
})();
