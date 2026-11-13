/* Chapter 11: DEBUG level explorer, annotated DEBUG=2 line, SASS control-code decoder. */
BOOK.ready(function (B) {
  const add = B.get("add/default"), K = add.kernels[0];
  const noNops = (s) => s.split("\n").filter((l) => !/NOP;|^\s*\.{3,}/.test(l) && l.trim()).join("\n");

  // ---- 1. DEBUG levels ----
  (function levels() {
    const root = document.getElementById("dbglevels"); if (!root) return;
    const optsLine = `E_8192_32_4               opts: (${K.applied_opts.join(", ")})`;
    const ptxNum = K.ptx.text.split("\n").map((l, i) => `${String(i + 1).padStart(3)} ${l}`).join("\n");
    // [level, provenance, lang, text, note]
    const blocks = [
      [1, "null", "text", "opened device NULL from pid:56545\ncompiling E_8192_32_4                             : 100%|█| 1/1", "device.py:35 and the compile progress bar (realize.py:277). On the L4 it says 'opened device CUDA'. No 'scheduled' line: it only prints for >1 kernel at this level."],
      [2, "l4", "text", B.get("jit/default").calls[0].log.trim(), "One line per kernel run (realize.py:92). From the jit capture: the a+b capture stored only the parsed time, 21.57 µs."],
      [3, "null", "text", "scheduled     1 kernels in    25.59 ms | CACHE MISS 8dcc01c3 |      49 uops in cache | null_cuda.py:6\n" + optsLine, "schedule/__init__.py:131 and codegen/__init__.py:422. The opts are the same as in the L4 capture."],
      [4, "null", "text", "more upcast axis : [(0, 3, 0, 4)]", "heuristic.py:134: the upcast loop's candidate (num_strides, sum_strides, axis, amount)."],
      [4, "l4", "c", K.source, "The rendered CUDA C (codegen/__init__.py:450)."],
      [5, "l4", "uop", K.ast.text, "pyrender of the kernel AST, printed before codegen (codegen/__init__.py:291)."],
      [5, "l4", "ptx", ptxNum, "On CUDA: the PTX handed to cuModuleLoadData, with line numbers (ops_cuda.py:40). tinygrad also colors it."],
      [6, "null", "text", "(nothing new: identical to DEBUG=5 at this commit)", "No DEBUG >= 6 check in codegen. For uops use tinygrad.uop.render.print_uops."],
      [7, "l4", "sass", noNops(K.sass.text), "ptxas + nvdisasm via cuda_disassemble (compiler_cuda.py:38). Shown here in cuobjdump's format from the capture; nvdisasm prints the same instructions with section headers."],
    ];
    root.innerHTML = `<div class="widget-head"><span class="title">DEBUG=<b class="lv">1</b> python add.py</span>
      <input type="range" min="1" max="7" value="1" class="lvr" style="width:200px"><div class="dbg-levels">${[1, 2, 3, 4, 5, 6, 7].map((l) => `<button class="btn" data-l="${l}">${l}</button>`).join("")}</div>
      <button class="btn lplay">▶ climb</button></div><pre class="code dbg-out"></pre>`;
    const out = root.querySelector(".dbg-out"), rng = root.querySelector(".lvr");
    let level = 0, timer = null;
    function set(l) {
      const prev = level; level = l; rng.value = l; root.querySelector(".lv").textContent = l;
      root.querySelectorAll("[data-l]").forEach((b) => b.classList.toggle("on", +b.dataset.l === l));
      out.innerHTML = blocks.filter((b) => b[0] <= l).map((b) => `<span class="blk ${b[1]}" style="${b[0] <= prev ? "animation:none" : ""}"><span class="tag">DEBUG≥${b[0]} · ${b[1] === "l4" ? "L4 capture" : "NULL device run"}</span> <span style="color:var(--muted);font-family:var(--sans);font-size:12px">${B.esc(b[4])}</span>\n${B.highlight(b[3], b[2])}</span>`).join("");
      const last = out.querySelectorAll(".blk"); if (l > prev && last.length) out.scrollTop = last[last.length - 1].offsetTop - 40;
    }
    rng.oninput = () => set(+rng.value);
    root.querySelectorAll("[data-l]").forEach((b) => (b.onclick = () => set(+b.dataset.l)));
    root.querySelector(".lplay").onclick = (e) => {
      if (timer) { clearInterval(timer); timer = null; e.target.textContent = "▶ climb"; return; }
      e.target.textContent = "❚❚ stop"; set(1);
      timer = setInterval(() => { if (level >= 7) { clearInterval(timer); timer = null; e.target.textContent = "▶ climb"; } else set(level + 1); }, 1800);
    };
    set(1);
  })();

  // ---- 2. annotated DEBUG=2 line ----
  (function d2() {
    const root = document.getElementById("d2line"); if (!root) return;
    const line = B.get("jit/default").calls[0].log.trim();
    const m = line.replace(/(r_\S+)\s{3,}/, "$1   ").match(/^(\*\*\*) (CUDA\s*) (\s*\d+) (r_\S+)(\s+)(arg\s+\d+) (mem\s+[\d.]+ GB) (tm\s+[\d.]+us)(\/\s*[\d.]+ms) \((\s*\d+ GFLOPS)(\s+\d+)\|(\d+\s+GB\/s)\)$/);
    if (!m) { root.innerHTML = `<div class="d2">${B.esc(line)}</div>`; return; }
    const notes = [
      ["***", "Marks a kernel execution. Green the first time this kernel runs, magenta under the JIT (realize.py:86)."],
      ["device", "Device name, padded to 7 characters."],
      ["kernel count", "GlobalCounters.kernel_count: a running count of kernels launched since the counters were reset."],
      ["name", "The kernel name: r_ = has a reduction; then global, local, upcast, reduce and unroll sizes (chapter 1). Colored by axis type in a terminal."],
      ["", ""],
      ["arg", "Number of buffers passed to the kernel: output first, then inputs."],
      ["mem", "Device memory currently allocated by tinygrad, in GB (not this kernel's traffic)."],
      ["tm", "This kernel's time. On CUDA: cuEventElapsedTime between events recorded around the launch (ops_cuda.py:26). Yellow above 10 ms."],
      ["cumulative", "Running total of kernel time, in ms."],
      ["GFLOPS", "Estimated ops / time. The estimate is computed from the uops (Estimates.ops)."],
      ["mem GB/s", "Estimates.mem / time: unique bytes the kernel must read and write (each buffer once)."],
      ["lds GB/s", "Estimates.lds / time: every load and store the code issues. Much larger than mem here because the kernel rereads its inputs many times."],
    ];
    root.innerHTML = `<div class="d2">${m.slice(1).map((s, i) => notes[i][0] ? `<span data-i="${i}">${B.esc(s)}</span>` : B.esc(s)).reduce((a, s, i) => a + (i === 9 ? " (" : i === 11 ? "|" : i && ![4, 5, 8, 10].includes(i) ? " " : "") + s, "")})</div><div class="d2-note">Hover a field.</div>`;
    const note = root.querySelector(".d2-note"), spans = [...root.querySelectorAll("[data-i]")];
    spans.forEach((s) => { const f = () => { spans.forEach((x) => x.classList.toggle("on", x === s)); const n = notes[+s.dataset.i]; note.innerHTML = `<b>${B.esc(n[0])}</b>: ${B.esc(n[1])}`; };
      s.addEventListener("mouseenter", f); s.addEventListener("click", f); });
    let k = 0; const auto = setInterval(() => { if (root.matches(":hover")) return clearInterval(auto); spans[k % spans.length].dispatchEvent(new Event("mouseenter")); k++; if (k > spans.length * 2) clearInterval(auto); }, 2200);
  })();

  // ---- 3. SASS control-code decoder ----
  (function sassdec() {
    const root = document.getElementById("sassdec"); if (!root) return;
    // encoding words by address, from captures/add/default.json (cuobjdump -sass); the loaded data keeps only the instruction text
    const ENC = {"0000":["0x00000a0000017a02","0x000fe40000000f00"],"0010":["0x0000000000007919","0x000e220000002500"],"0020":["0x0000000400077802","0x000fe20000000f00"],"0030":["0x0000460000047ab9","0x000fe40000000a00"],"0040":["0x0000000000037919","0x000e240000002100"],"0050":["0x0000000300007211","0x001fc800078e28ff"],"0060":["0x0000000200067819","0x000fca00000006ff"],"0070":["0x00005a0006027625","0x000fc800078e0207"],"0080":["0x00005c0006047625","0x0c0fe200078e0207"],"0090":["0x0000000402087981","0x000ea8000c1e1d00"],"00a0":["0x00000004040c7981","0x000ea2000c1e1d00"],"00b0":["0x0000580006067625","0x000fe200078e0207"],"00c0":["0x0000000f0b0b7221","0x004fe20000000000"],"00d0":["0x0000000e0a0a7221","0x000fe20000000000"],"00e0":["0x0000000d09097221","0x000fe20000000000"],"00f0":["0x0000000c08087221","0x000fca0000000000"],"0100":["0x0000000806007986","0x000fe2000c101d04"],"0110":["0x000000000000794d","0x000fea0003800000"],"0120":["0xfffffff000007947","0x000fc0000383ffff"]};
    // [addr, instruction, low 64 bits, high 64 bits]: rows parsed from the loaded SASS, NOPs omitted
    const S = [...K.sass.text.matchAll(/\/\*([0-9a-f]{4})\*\/\s+(.+?)\s*$/gm)].filter((m) => !/^NOP\b/.test(m[2])).map((m) => [m[1], m[2], ...(ENC[m[1]] || ["?", "0x0"])]);
    const live = S.length === Object.keys(ENC).length && S.every((s) => ENC[s[0]]) && S[9][1] === "LDG.E.128 R8, [R2.64] ;";   // sanity check against the loaded capture
    const OP = {
      MOV: "Move. MOV R1, c[0x0][0x28] loads the stack pointer from the driver's constant bank; MOV R7, 0x4 materializes sizeof(float) for the IMAD.WIDEs.",
      S2R: "Special register to register. SR_CTAID.X = blockIdx.x (gidx0), SR_TID.X = threadIdx.x (lidx0). Variable latency, so it signals a scoreboard barrier when done.",
      ULDC: "Uniform load from constant memory into uniform registers UR4:UR5, one copy per warp. This 64-bit value is the memory descriptor that .E global accesses use implicitly.",
      LEA: "Load effective address: R0 = (R0 << 5) + R3 = (gidx0 << 5) + lidx0. ptxas folded tinygrad's (gidx0<<7)+(lidx0<<2) into this plus one shift.",
      SHF: "Funnel shift left: R6 = R0 << 2 = alu0, the float index of this thread's float4.",
      IMAD: "IMAD.WIDE: 32x32 multiply plus a 64-bit addend: address = alu0 * 4 + pointer, where the pointer is a kernel parameter in constant bank 0 (0x160 data0, 0x168 data1, 0x170 data2). Writes a register pair.",
      LDG: "Global load, .E = 64-bit address in R2:R3 (written R2.64), .128 = 16 bytes into R8..R11 (or R12..R15): the float4. Sets write barrier 2.",
      FADD: "fp32 add, one lane of the float4. The first FADD waits on barrier 2: both loads must have arrived.",
      STG: "Global store of R8..R11 (16 bytes) to data0 + alu0*4.",
      EXIT: "Thread exits.", BRA: "Branch to itself. Never executed; it marks the end of the code, followed by alignment NOPs.",
    };
    root.innerHTML = `<div class="widget-head"><span class="title">E_8192_32_4: click an instruction to decode it</span><span class="chip ${live ? "ok" : "bad"}">${live ? "matches loaded capture" : "capture mismatch"}</span><button class="btn sxplay">▶ walk through</button></div>
      <div class="sx"><div class="sx-list">${S.map((s, i) => `<div data-i="${i}"><span style="color:#6e7b8b">/*${s[0]}*/</span>  ${B.highlight(s[1], "sass")}</div>`).join("")}</div><div class="sx-info"></div></div>`;
    const rows = [...root.querySelectorAll(".sx-list div")], info = root.querySelector(".sx-info");
    const dec = (hi) => { const h = BigInt(hi); const f = (sh, w) => Number((h >> BigInt(sh)) & BigInt((1 << w) - 1));
      return { stall: f(41, 4), yld: f(45, 1), wb: f(46, 3), rb: f(49, 3), wait: f(52, 6), reuse: f(58, 4) }; };
    const D = S.map((s) => dec(s[3]));
    function show(i) {
      rows.forEach((r, j) => r.classList.toggle("on", j === i));
      const d = D[i], op = S[i][1].split(/[ .;]/)[0];
      // dependents: later instructions whose wait mask includes the barrier this one sets
      rows.forEach((r, j) => r.classList.toggle("dep", d.wb !== 7 && j > i && (D[j].wait >> d.wb) & 1));
      const waits = [0, 1, 2, 3, 4, 5].filter((b) => (d.wait >> b) & 1);
      info.innerHTML = `<h4>${B.esc(S[i][1])}</h4><div style="color:#bcc6d1">${B.esc(OP[op] || "")}</div>
        <div style="font:11.5px var(--mono);color:var(--muted);margin-top:8px">encoding ${S[i][2]} ${S[i][3]}</div>
        <div class="bits"><div>stall<b>${d.stall}</b></div><div>yield bit<b>${d.yld}</b></div><div>write bar<b>${d.wb === 7 ? "–" : d.wb}</b></div><div>read bar<b>${d.rb === 7 ? "–" : d.rb}</b></div><div>wait on<b>${waits.length ? waits.join(",") : "–"}</b></div><div>reuse<b>${d.reuse ? "0b" + d.reuse.toString(2) : "–"}</b></div></div>
        <div style="font-size:13px;color:#bcc6d1">Bits of the high word: stall [41:44] cycles before the next instruction may issue · yield [45] (reported as an inverted hint) · write barrier [46:48] and read barrier [49:51], 7 = none · wait mask [52:57] · operand reuse flags [58:61].
        ${d.wb !== 7 ? `<br><b style="color:#fbbf24">Sets barrier ${d.wb}</b>: highlighted rows wait for it.` : ""}${waits.length ? `<br><b style="color:#7cc4ff">Waits for barrier ${waits.join(",")}</b> before issuing.` : ""}</div>`;
    }
    rows.forEach((r) => r.addEventListener("click", () => show(+r.dataset.i)));
    let t = null;
    root.querySelector(".sxplay").onclick = (e) => { if (t) { clearInterval(t); t = null; e.target.textContent = "▶ walk through"; return; }
      e.target.textContent = "❚❚ stop"; let i = 0; show(0); t = setInterval(() => { i++; if (i >= S.length) { clearInterval(t); t = null; e.target.textContent = "▶ walk through"; return; } show(i); }, 2000); };
    show(9);
  })();
});
