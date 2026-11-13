/* Chapter 1 animations. All text comes from captures add/default and add/torch. */
BOOK.ready(function (B) {
  const tg = B.get("add/default"), th = B.get("add/torch"), K = tg.kernels[0];
  const noAnsi = (s) => s.replace(/\\x1b\[[0-9;]*m/g, "").replace(/\x1b\[[0-9;]*m/g, "");
  const noNops = (s) => s.split("\n").filter((l) => !/NOP;|^\s*\.{3,}/.test(l) && l.trim()).join("\n");
  const ir = (name, nth = 0) => { const ps = tg.passes.filter((p) => p.name === name && p.ir); return ps[nth] ? noAnsi(ps[nth].ir.text) : ""; };

  // ---- tensor graph ----
  const tgp = document.getElementById("tgraph");
  if (tgp) tgp.innerHTML = B.codeHTML(tg.tensor_graph.text, "uop");

  // ---- 1. the journey: artifact morphing stage by stage ----
  (function journey() {
    const root = document.getElementById("journey"); if (!root) return;
    const ptxBody = K.ptx.text.split("\n").filter((l) => !/^\/\/|^\s*$/.test(l)).join("\n");
    const steps = [
      ["tensor", "tensor graph", "uop", tg.tensor_graph.text, "c = a + b: an ADD of two RESHAPEd BUFFERs. Nothing has run."],
      ["tensor", "early transform tensor graph", "uop", ir("early transform tensor graph"), "The result needs a home: a new BUFFER c1, and STORE the ADD into it. AFTER says 'c1, once the store has run'."],
      ["tensor", "replace bufs", "uop", ir("replace bufs"), "Concrete buffers become numbered PARAM slots (0 = output). The graph no longer depends on which memory it runs on."],
      ["schedule", "apply rangeify", "uop", ir("apply rangeify"), "Two RANGE loop variables. Each RESHAPE was pushed into the index as c3*1024+c5 and deleted."],
      ["schedule", "kernel split", "uop", ir("kernel split"), "The STORE closes both loops (END), so this is a complete kernel."],
      ["codegen", "kernel AST (input to codegen)", "uop", K.ast.text, "Codegen starts: a SINK with KernelInfo(name='test')."],
      ["codegen", "simplify ranges", "uop", ir("simplify ranges"), "r0*1024+r1 over 1024x1024 is just a counter: the two loops merge into one RANGE of 1048576."],
      ["codegen", "apply_opts → flatten range", "uop", ir("flatten range"), "UPCAST 4 then LOCAL 32 split the loop: 8192 GLOBAL x 32 LOCAL x 4 UPCAST."],
      ["codegen", "postopt symbolic", "uop", ir("postopt symbolic"), "Index simplified to g*128 + l*4 + u. The kernel now has its name: E_8192_32_4."],
      ["codegen", "expander", "uop", ir("expander"), "The UPCAST range is gone: replaced by the vector constant STACK(0,1,2,3). Every value is now 4 wide."],
      ["codegen", "add gpudims", "uop", ir("add gpudims"), "GLOBAL and LOCAL ranges become SPECIAL gidx0 (blockIdx.x) and lidx0 (threadIdx.x). No loops are left."],
      ["codegen", "devectorize2", "uop", ir("devectorize2"), "Every 4-wide op split into 4 scalar loads, adds and stores (50 UOps)."],
      ["codegen", "memory coalescing / add images", "uop", ir("add images"), "Four loads at base+0..3 glued back into one 4-wide SHRINK load: this will be a float4."],
      ["codegen", "late decompositions", "uop", ir("late decompositions"), "Multiply by a power of two becomes a shift: gidx0<<7, lidx0<<2."],
      ["render", "CUDA C", "c", K.source, "Rendered by the CUDA renderer. alu0 is the index, two float4 loads, one float4 store."],
      ["compile", "PTX (NVRTC)", "ptx", ptxBody, "NVRTC's PTX: shl, shl, add, mul.wide, ld.global.v4.f32 x2, add.f32 x4, st.global.v4.f32."],
      ["compile", "SASS (ptxas sm_89)", "sass", noNops(K.sass.text), "19 machine instructions: S2R, LEA, SHF, IMAD.WIDE, LDG.E.128 x2, FADD x4, STG.E.128, EXIT."],
    ];
    root.innerHTML = `<div class="widget-head"><span class="title">The journey of a + b: one capture, seventeen snapshots</span>
      <button class="btn jp">◀</button><button class="btn jplay">▶ play</button><button class="btn jn">▶</button><span class="chip jcount"></span></div>
      <div class="jr-map">${steps.map((s, i) => `<i data-i="${i}" title="${B.esc(s[1])}" style="background:var(--s-${s[0]})"></i>`).join("")}</div>
      <div class="jr-cap"></div><pre class="code"></pre>`;
    const pre = root.querySelector("pre"), cap = root.querySelector(".jr-cap"), marks = [...root.querySelectorAll(".jr-map i")];
    let cur = -1, timer = null;
    function show(i) {
      const prev = cur; cur = (i + steps.length) % steps.length; const s = steps[cur];
      marks.forEach((m, j) => { m.classList.toggle("on", j === cur); m.classList.toggle("past", j < cur); });
      root.querySelector(".jcount").textContent = `${cur + 1} / ${steps.length}`;
      cap.innerHTML = `<span class="chip" style="background:var(--s-${s[0]});color:#081018">${s[0]}</span> <b>${B.esc(s[1])}</b> · ${B.esc(s[4])}`;
      const sameLang = prev >= 0 && steps[prev][2] === s[2] && cur === prev + 1;
      const m = sameLang ? B.addedLines(steps[prev][3], s[3]) : s[3].split("\n").map(() => "add");
      pre.classList.add("out");
      setTimeout(() => { pre.innerHTML = B.codeHTML(s[3], s[2], m); pre.classList.remove("out");
        [...pre.querySelectorAll(".ln.add")].forEach((l, k) => (l.style.animationDelay = Math.min(k * 40, 900) + "ms")); }, 220);
    }
    const btn = root.querySelector(".jplay");
    const stop = () => { if (timer) { clearInterval(timer); timer = null; btn.textContent = "▶ play"; } };
    btn.onclick = () => { if (timer) return stop(); btn.textContent = "❚❚ pause"; if (cur === steps.length - 1) show(0); timer = setInterval(() => { if (cur >= steps.length - 1) stop(); else show(cur + 1); }, 3200); };
    root.querySelector(".jp").onclick = () => { stop(); show(cur - 1); };
    root.querySelector(".jn").onclick = () => { stop(); show(cur + 1); };
    marks.forEach((m) => (m.onclick = () => { stop(); show(+m.dataset.i); }));
    show(0);
  })();

  // ---- 2. range splitter: shift_to UPCAST 4, LOCAL 32 ----
  (function splitter() {
    const root = document.getElementById("splitter"); if (!root) return;
    const N = 1048576;
    const st = [
      { segs: [["L", N, "#cbd5e1", "WEAK"]], name: "", expr: "c2 = UOp.range(1048576, 0, AxisType.WEAK)   # after simplify ranges", cap: "One output loop of 1,048,576 iterations. Type WEAK: codegen may map it to hardware however it likes." },
      { segs: [["g", N, "#60a5fa", "GLOBAL"]], name: "", expr: "convert_loop_to_global(): WEAK → GLOBAL", cap: "On a GPU (has_local), output loops become GLOBAL: one grid index per iteration." },
      { segs: [["g", N / 4, "#60a5fa", "GLOBAL"], ["u", 4, "#fbbf24", "UPCAST"]], name: "", expr: "Opt(UPCAST, axis=0, arg=4):  r → r'*4 + u        r' ∈ [0,262144), u ∈ [0,4)", cap: "shift_to splits off 4 iterations as an UPCAST axis: 4 consecutive elements per thread, unrolled into a vector." },
      { segs: [["g", 8192, "#60a5fa", "GLOBAL"], ["l", 32, "#22d3ee", "LOCAL"], ["u", 4, "#fbbf24", "UPCAST"]], name: "", expr: "Opt(LOCAL, axis=0, arg=32):  r' → g*32 + l       index = (g*32 + l)*4 + u", cap: "The remaining 262,144 split again: 32 LOCAL threads per block, 8192 blocks." },
      { segs: [["g", 8192, "#60a5fa", "GLOBAL"], ["l", 32, "#22d3ee", "LOCAL"], ["u", 4, "#fbbf24", "UPCAST"]], name: "E_8192_32_4", expr: "postopt symbolic:  g*128 + l*4 + u   →  late decompositions:  (gidx0<<7) + (lidx0<<2) + lane", cap: "get_optimized_ast names the kernel: E (no reduce), then the axis sizes in order global, local, upcast." },
    ];
    root.innerHTML = `<div class="widget-head"><span class="title">shift_to: how one loop becomes 8192 × 32 × 4</span><button class="btn sp">◀</button><button class="btn sn">next ▶</button><span class="chip scount"></span></div>
      <div class="widget-body"><svg viewBox="0 0 900 120"></svg><div class="sp-name"></div><div class="sp-expr"></div><div class="jr-cap scap" style="padding:0 2px"></div></div>`;
    const svg = root.querySelector("svg");
    let cur = 0;
    function draw() {
      const s = st[cur]; let x = 10; const total = 880;
      // widths: log-ish so small axes stay visible
      const w = s.segs.map((g) => Math.max(110, Math.log2(g[1]) / 20 * total));
      const sc = total / w.reduce((a, b) => a + b, 0);
      let h = "";
      s.segs.forEach((g, i) => {
        const ww = w[i] * sc - 24;
        h += `<g style="transition:all .6s"><rect x="${x}" y="20" width="${ww}" height="58" rx="8" fill="${g[2]}22" stroke="${g[2]}" stroke-width="2"/>
          <text x="${x + ww / 2}" y="48" text-anchor="middle" fill="#fff" style="font:700 20px var(--mono)">${g[1].toLocaleString("en-US")}</text>
          <text x="${x + ww / 2}" y="68" text-anchor="middle" fill="${g[2]}" style="font:600 11px var(--sans)">${g[3]} (${g[0]})</text></g>`;
        if (i < s.segs.length - 1) h += `<text x="${x + ww + 6}" y="54" fill="#8b97a6" style="font:700 16px var(--mono)">×</text>`;
        x += w[i] * sc;
      });
      h += `<text x="450" y="106" text-anchor="middle" fill="#8b97a6" style="font:12px var(--mono)">product = ${s.segs.reduce((a, g) => a * g[1], 1).toLocaleString("en-US")} elements</text>`;
      svg.innerHTML = h;
      root.querySelector(".sp-name").innerHTML = s.name ? `<span>E</span><span class="c-k">_</span><span class="c-g">8192</span><span class="c-k">_</span><span class="c-l">32</span><span class="c-k">_</span><span class="c-u">4</span>` : `<span class="c-k">name: (not yet)</span>`;
      root.querySelector(".sp-expr").textContent = s.expr;
      root.querySelector(".scap").textContent = s.cap;
      root.querySelector(".scount").textContent = `${cur + 1} / ${st.length}`;
    }
    root.querySelector(".sn").onclick = () => { cur = (cur + 1) % st.length; draw(); };
    root.querySelector(".sp").onclick = () => { cur = (cur - 1 + st.length) % st.length; draw(); };
    draw();
  })();

  // ---- 3. GPU grid: which addresses does (gidx0, lidx0) touch ----
  (function grid() {
    const root = document.getElementById("gridanim"); if (!root) return;
    root.innerHTML = `<div class="widget-head"><span class="title">E_8192_32_4 on the L4: grid → block (= 1 warp) → thread → float4</span>
      <button class="btn grun">▶ run warp</button><span class="chip gphase">idle</span></div>
      <div class="widget-body"><svg viewBox="0 0 900 380"></svg></div>
      <div class="gd-ctl"><label>block gidx0 <input type="range" class="gb" min="0" max="8191" value="3"><b class="gbv mono">3</b></label>
        <label>thread lidx0 <input type="range" class="gt" min="0" max="31" value="7"><b class="gtv mono">7</b></label>
        <span style="color:var(--muted)">alu0 = (gidx0&lt;&lt;7) + (lidx0&lt;&lt;2), from the kernel's CUDA C</span></div>
      <div class="gd-math"></div>`;
    const svg = root.querySelector("svg"), gb = root.querySelector(".gb"), gt = root.querySelector(".gt");
    let g = 3, t = 7, busy = false;
    const C = { g: "#60a5fa", l: "#22d3ee", u: "#fbbf24", a: "#34d399", st: "#f87171" };
    function build() {
      let h = `<text x="10" y="16" fill="#8b97a6" style="font:12px var(--sans)">grid: 8192 blocks (showing 32 around gidx0)</text>`;
      const g0 = Math.max(0, Math.min(8192 - 32, g - 8));
      for (let i = 0; i < 32; i++) { const bi = g0 + i;
        h += `<rect class="blk" x="${10 + i * 27.5}" y="24" width="24" height="24" rx="4" fill="${bi === g ? C.g : "#1c2430"}" stroke="${C.g}" stroke-opacity="${bi === g ? 1 : .35}"/>`;
        if (i % 4 === 0 || bi === g) h += `<text x="${22 + i * 27.5}" y="62" text-anchor="middle" fill="${bi === g ? "#fff" : "#6e7b8b"}" style="font:9.5px var(--mono)">${bi}</text>`; }
      h += `<path d="M${22 + (g - g0) * 27.5} 66 L450 92" stroke="${C.g}" stroke-dasharray="3 4" fill="none"/>`;
      h += `<text x="10" y="106" fill="#8b97a6" style="font:12px var(--sans)">block ${g} = one warp of 32 threads (lidx0 = threadIdx.x)</text>`;
      for (let i = 0; i < 32; i++) h += `<g class="thr" data-t="${i}" style="cursor:pointer"><rect x="${10 + i * 27.5}" y="114" width="24" height="30" rx="4" fill="${i === t ? C.l : "#1c2430"}" stroke="${C.l}" stroke-opacity="${i === t ? 1 : .45}"/><text x="${22 + i * 27.5}" y="133" text-anchor="middle" fill="${i === t ? "#081018" : "#9fb0c2"}" style="font:10px var(--mono)">${i}</text></g>`;
      h += `<text x="10" y="172" fill="#8b97a6" style="font:12px var(--sans)">data1 (and data2, data0): floats ${g * 128} … ${g * 128 + 127} owned by this block · each row = 32 floats = 128 bytes</text>`;
      for (let r = 0; r < 4; r++) for (let c = 0; c < 32; c++) { const e = r * 32 + c, own = e >> 2;
        h += `<rect class="cell" data-e="${e}" x="${10 + c * 27.5}" y="${182 + r * 30}" width="25" height="26" rx="2" fill="${own === t ? C.u : "#151b23"}" stroke="${own % 2 ? "#33445a" : "#2a3441"}"/>`;
        if (c % 4 === 0) h += `<text x="${12 + c * 27.5}" y="${200 + r * 30}" fill="#5b6878" style="font:8.5px var(--mono)">${g * 128 + e}</text>`; }
      h += `<text x="10" y="318" fill="#8b97a6" style="font:12px var(--sans)">the thread's registers</text>`;
      ["LDG.E.128 R8 ← data1[alu0]", "LDG.E.128 R12 ← data2[alu0]", "FADD ×4: R8..R11 += R12..R15", "STG.E.128 data0[alu0] ← R8"].forEach((s, i) =>
        h += `<g class="reg" data-r="${i}" opacity=".35"><rect x="${10 + i * 222}" y="326" width="214" height="34" rx="6" fill="#10161e" stroke="${[C.u, C.u, C.a, C.st][i]}"/><text x="${117 + i * 222}" y="347" text-anchor="middle" fill="#dfe6ee" style="font:10.5px var(--mono)">${s}</text></g>`);
      svg.innerHTML = h;
      svg.querySelectorAll(".thr").forEach((n) => n.addEventListener("click", () => { if (busy) return; t = +n.dataset.t; gt.value = t; update(); }));
    }
    function math() {
      const a = (g << 7) + (t << 2);
      root.querySelector(".gd-math").innerHTML =
        `gidx0 = blockIdx.x = <span class="hl">${g}</span>        lidx0 = threadIdx.x = <span class="hl">${t}</span>\n` +
        `alu0  = (${g}&lt;&lt;7) + (${t}&lt;&lt;2) = ${g << 7} + ${t << 2} = <span class="hl">${a}</span>          SASS: LEA R0 = (${g}&lt;&lt;5)+${t} = ${(g << 5) + t};  SHF R6 = R0&lt;&lt;2 = ${a}\n` +
        `floats  ${a} … ${a + 3}  →  row ${Math.floor(a / 1024)}, cols ${a % 1024} … ${(a % 1024) + 3} of the 1024×1024 tensor\n` +
        `bytes   data1 + ${a * 4} … +${a * 4 + 15}   (IMAD.WIDE R2 = R6*4 + c[0x0][0x168]; one 16-byte LDG.E.128)`;
    }
    function update() { root.querySelector(".gbv").textContent = g; root.querySelector(".gtv").textContent = t; build(); math(); }
    gb.oninput = () => { if (!busy) { g = +gb.value; update(); } };
    gt.oninput = () => { if (!busy) { t = +gt.value; update(); } };
    const wait = (ms) => new Promise((r) => setTimeout(r, ms));
    root.querySelector(".grun").onclick = async () => {
      if (busy) return; busy = true; const ph = root.querySelector(".gphase");
      const cells = [...svg.querySelectorAll(".cell")], regs = [...svg.querySelectorAll(".reg")], thr = [...svg.querySelectorAll(".thr rect")];
      cells.forEach((c) => c.setAttribute("fill", "#151b23"));
      thr.forEach((r) => r.setAttribute("fill", C.l));
      const phases = [["LDG.E.128 data1", C.u, 0], ["LDG.E.128 data2", C.u, 1], ["FADD ×4 (in registers)", C.a, 2], ["STG.E.128 data0", C.st, 3]];
      for (const [label, col, ri] of phases) {
        ph.textContent = label; regs.forEach((r, i) => r.setAttribute("opacity", i === ri ? 1 : .35));
        if (ri === 2) { cells.forEach((c) => c.setAttribute("fill", "#34d39955")); await wait(900); continue; }
        // all 32 lanes issue together; draw them in quick succession so the eye can follow lane -> 4 cells
        for (let lane = 0; lane < 32; lane++) { for (let k = 0; k < 4; k++) cells[lane * 4 + k].setAttribute("fill", col); if (lane % 2) await wait(28); }
        await wait(500);
      }
      ph.textContent = "done: 512 contiguous bytes per access, 4 full 128-byte lines"; regs.forEach((r) => r.setAttribute("opacity", 1));
      await wait(1600); busy = false; update(); ph.textContent = "idle";
    };
    update();
  })();

  // ---- 4. SASS side by side, grouped by role ----
  (function sasscmp() {
    const root = document.getElementById("sasscmp"); if (!root) return;
    const tk = th.triton_kernels.find((k) => k.metadata && k.metadata.num_warps === 4) || th.triton_kernels[0];
    const lines = (t) => noNops(t).split("\n").filter((l) => /\/\*[0-9a-f]{4}\*\//.test(l)).map((l) => l.trim().replace(/\*\/\s+/, "*/  "));
    const L = lines(K.sass.text), R = lines(tk.sass.text);
    const roles = [["ids", "thread / block id", /S2R/, "#c084fc"], ["addr", "index & address math", /LEA|SHF|IMAD|LOP3|MOV|ULDC/, "#a78bfa"],
      ["ld", "global loads", /LDG/, "#60a5fa"], ["add", "the adds", /FADD/, "#34d399"], ["st", "global stores", /STG/, "#f87171"], ["end", "exit", /EXIT|BRA/, "#64748b"]];
    const role = (l) => roles.find((r) => r[2].test(l)) || roles[5];
    const col = (ls) => ls.map((l) => { const r = role(l); return `<div class="sl" data-r="${r[0]}" style="border-left:3px solid ${r[3]};padding:1px 8px;font:12px/1.6 var(--mono);white-space:pre;transition:opacity .3s,background .3s">${B.highlight(l.replace(/\s+;/, " ;"), "sass")}</div>`; }).join("");
    const cnt = (ls, re) => ls.filter((l) => re.test(l)).length;
    root.innerHTML = `<div class="widget-head"><span class="title">SASS: tinygrad vs Triton (4-warp config), colored by role</span>
      <div class="tabs">${roles.map((r) => `<button class="btn" data-role="${r[0]}" style="border-color:${r[3]}">${r[1]} <span class="mono">${cnt(L, r[2])}·${cnt(R, r[2])}</span></button>`).join("")}<button class="btn" data-role="all">all</button></div>
      <button class="btn cyc">▶ cycle</button></div>
      <div class="widget-body sass-pair"><div><div class="chip">tinygrad E_8192_32_4 · ${L.length} instr · ${K.regs} regs · 1 float4 / thread</div><div style="background:#0a0e14;margin-top:6px;padding:6px 0;overflow:auto">${col(L)}</div></div>
      <div><div class="chip">${B.esc(tk.name)} · ${R.length} instr · ${(tk.res_usage.match(/REG:(\d+)/) || [])[1]} regs · 2 float4 / thread</div><div style="background:#0a0e14;margin-top:6px;padding:6px 0;overflow:auto">${col(R)}</div></div></div>
      <div class="jr-cap scap">Buttons show the count per role as tinygrad·Triton. Click one to highlight it in both listings.</div>`;
    const notes = { ids: "Both read blockIdx.x and threadIdx.x with S2R (special-register reads, which take a few cycles, so they are issued early).",
      addr: "tinygrad: LEA + SHF. Triton: SHF + LOP3 (mask to 0..508) + LEA by 10 (×1024 = XBLOCK). Both build 64-bit addresses with IMAD.WIDE index*4 + param pointer from constant bank 0.",
      ld: "128-bit loads in both. Triton issues 4 (two float4s per input, the second at +0x800 bytes = 512 floats later); tinygrad issues 2.",
      add: "FADD per lane: 4 in tinygrad, 8 in Triton.", st: "One STG.E.128 per float4.", end: "EXIT, then the BRA-to-self that pads every kernel." };
    const set = (r) => { root.querySelectorAll(".sl").forEach((d) => { const on = r === "all" || d.dataset.r === r; d.style.opacity = on ? 1 : .22; d.style.background = on && r !== "all" ? "rgba(124,196,255,.10)" : "none"; });
      root.querySelector(".scap").textContent = notes[r] || "Buttons show the count per role as tinygrad·Triton."; };
    let timer = null, i = 0;
    root.querySelectorAll("[data-role]").forEach((b) => (b.onclick = () => { if (timer) { clearInterval(timer); timer = null; root.querySelector(".cyc").textContent = "▶ cycle"; } set(b.dataset.role); }));
    root.querySelector(".cyc").onclick = (e) => { if (timer) { clearInterval(timer); timer = null; e.target.textContent = "▶ cycle"; set("all"); return; }
      e.target.textContent = "❚❚ stop"; timer = setInterval(() => { set(roles[i % roles.length][0]); i++; }, 2200); };
    set("all");
  })();
});
