/* Preface: the two-lane pipeline overview. Every artifact shown comes from captures add/default and add/torch. */
BOOK.ready(function (B) {
  const root = document.getElementById("overview");
  if (!root) return;
  const tg = B.get("add/default"), th = B.get("add/torch"), k = tg.kernels[0], tk = th.triton_kernels[0];
  const passIr = (name) => { const p = tg.passes.find((x) => x.name === name && x.ir); return p ? p.ir.text : ""; };
  const noAnsi = (s) => s.replace(/\\x1b\[[0-9;]*m/g, "");
  const noNops = (s) => s.split("\n").filter((l) => !/NOP;|^\s*\.{3,}/.test(l)).join("\n").trim();
  const noLoc = (s) => s.split("\n").filter((l) => !/^#loc/.test(l)).map((l) => l.replace(/\s+loc\((#loc\d*|"[^"]*"\(#loc\d*\))\)/g, "")).join("\n");
  const out = th.inductor_debug.output_code.text;
  const tsrc = (() => { const i = out.indexOf("@triton.jit"); return i < 0 ? "" : out.slice(i, out.indexOf("'''", i)).trim(); })();
  const tcall = (() => { const i = out.indexOf("    def call(self, args):"); return i < 0 ? "" : out.slice(i, out.indexOf("return (buf0", i) + 16); })();
  const head = (s, n) => { const l = s.split("\n"); return l.length > n ? l.slice(0, n).join("\n") + `\n… (${l.length - n} more lines)` : s; };
  const launch = `cuLaunchKernel(E_8192_32_4,
    gridDim  = (${k.global_size.join(", ")}),     # blocks
    blockDim = (${k.local_size.join(", ")}),         # threads per block
    sharedMem = 0, stream = default,
    extra = [pointer to struct { data0, data1, data2 }])   # 3 x 8-byte pointers

# measured on the L4 (best of 5, tinygrad DEBUG=2 timing): ${k.time_us} us
# ptxas: ${k.regs} registers per thread`;

  const lanes = [
    { title: "tinygrad", x: 18, color: "var(--accent)", nodes: [
      ["Tensor code", "c = a + b", "tensor", "python", "from tinygrad import Tensor\na = Tensor(np_a).realize()     # 1024x1024 float32 on the L4\nb = Tensor(np_b).realize()\nc = a + b                        # nothing runs: builds a UOp\nc.realize()                      # schedule, compile, launch",
        "Python only records what you asked for."],
      ["Lazy UOp graph", "Tensor.uop", "tensor", "uop", tg.tensor_graph.text, "The tensor graph, printed by pyrender: two BUFFERs, RESHAPEd to 1024x1024, and an ADD. Broadcasting and shapes live on the nodes."],
      ["Rangeify", "schedule", "schedule", "uop", passIr("apply rangeify"), "After the 'apply rangeify' pass: two RANGE loop variables, INDEX into flat PARAM buffers, and a STORE closed by END. RESHAPE is gone, turned into r0*1024+r1."],
      ["Kernel AST", "SINK + KernelInfo", "schedule", "uop", k.ast.text, "The AST handed to codegen for kernel 0. The name is still 'test'; codegen will name it."],
      ["Codegen passes", "opts, expander, gpudims, …", "codegen", "uop", noAnsi(passIr("late decompositions")), "After 'late decompositions': the optimizer applied UPCAST 4 and LOCAL 32, the 4 upcast lanes became one float4 SHRINK load, the loops became gidx0/lidx0, and x*128 became x<<7."],
      ["Linearize", "flat uop list", "codegen", "text", head(k.uops.text, 47), "The DAG toposorted into a list. Columns: index, op, dtype, sources, arg."],
      ["CUDA C", "renderer/cstyle.py", "render", "c", k.source, "The CUDA renderer prints one C statement per uop that needs a name."],
      ["PTX", "NVRTC", "compile", "ptx", noLoc(k.ptx.text), "NVRTC compiles the C to PTX, NVIDIA's virtual ISA. Note ld.global.v4.f32: the float4 load survived."],
      ["SASS", "driver ptxas JIT", "compile", "sass", noNops(k.sass.text), "The driver JITs PTX to SASS inside cuModuleLoadData. Shown here: the same PTX through ptxas -arch=sm_89 and cuobjdump -sass (NOPs dropped)."],
      ["Launch", "cuLaunchKernel", "run", "python", launch, "8192 blocks x 32 threads x 4 floats = 1,048,576 elements."],
    ] },
    { title: "PyTorch torch.compile", x: 222, color: "#ee4c2c", nodes: [
      ["Python function", "torch.compile(fn)", "tensor", "python", "import torch\nfn = lambda X: X[\"a\"] + X[\"b\"]\ncompiled = torch.compile(fn)\nout = compiled(X)        # first call: trace + compile + run", "Nothing is traced until the first call."],
      ["Dynamo → AOTAutograd", "ATen FX graph", "tensor", "python", th.inductor_debug.fx_graph.text, "Dynamo captured the bytecode as an FX graph; AOTAutograd lowered it to ATen: one aten.add.Tensor."],
      ["Inductor IR", "loop body, post-fusion", "schedule", "text", head(th.inductor_debug.ir_post_fusion.text, 40), "Inductor's scheduler node after fusion: one loop over 1048576 elements, load, load, add, store. Indexing was flattened to c0."],
      ["Triton source", "output_code.py", "codegen", "python", tsrc, "Inductor emitted a Triton kernel. Each program handles XBLOCK elements; XBLOCK is chosen by autotuning."],
      ["TTIR", "Triton MLIR", "codegen", "mlir", noLoc(tk.ttir.text), "Triton IR: whole-tensor ops (tensor<1024xf32>), no threads yet. loc() annotations removed for readability."],
      ["TTGIR", "+ thread layout", "codegen", "mlir", noLoc(tk.ttgir.text), "TritonGPU IR adds #blocked: sizePerThread = [4], 32 threads per warp, 4 warps. That layout is what makes each thread load a float4."],
      ["LLVM IR", "NVPTX target", "render", "llvm", head(tk.llir.text, 60), "Lowered to LLVM with inline-asm vector loads (ld.global.v4.b32)."],
      ["PTX", "LLVM NVPTX backend", "compile", "ptx", head(noLoc(tk.ptx.text).split("\n").filter((l) => !/^\s*\.loc|^\s*\/\/|^\s*\$L__/.test(l)).join("\n"), 80), "PTX from LLVM (.version 8.8). Debug lines removed."],
      ["SASS", "ptxas → cubin", "compile", "sass", noNops(tk.sass.text), "Triton runs ptxas itself and caches a cubin. This config (4 warps) has each thread load two float4s from each input."],
      ["Launch", "call() in output code", "run", "python", tcall, "The wrapper allocates the output and launches the kernel with the CUDA driver on the current stream."],
    ] },
    { title: "eager", x: 426, color: "#ee4c2c", thin: true, nodes: [
      ["a + b", "ATen dispatch", "tensor", "python", "out = X[\"a\"] + X[\"b\"]     # eager: dispatch to at::add", "The dispatcher picks the CUDA implementation of add for float tensors."],
      ["prebuilt kernel", "libtorch_cuda.so", "run", "text", th.eager_kernels.map((e) => `${B.fmtUs(e.us)}  ${e.name}`).join("\n"), "No compilation at run time: this kernel template was compiled into PyTorch, for every supported architecture, when PyTorch was built. Device time from torch.profiler."],
    ] },
  ];
  const W = 186, H = 44, GAP = 12, TOP = 34;
  let svg = `<svg viewBox="0 0 ${426 + 120 + 12} ${TOP + 10 * (H + GAP) + 6}">`;
  lanes.forEach((ln, li) => {
    const w = ln.thin ? 120 : W;
    svg += `<text class="ov-lane-title" x="${ln.x}" y="20" style="fill:${ln.color}">${ln.title}</text>`;
    ln.nodes.forEach((n, i) => {
      const y = ln.thin ? TOP + (i === 0 ? 0 : 9) * (H + GAP) : TOP + i * (H + GAP);
      if (i) { const py = ln.thin ? TOP + H : TOP + (i - 1) * (H + GAP) + H; svg += `<path class="ov-link" d="M${ln.x + w / 2} ${py + 2}V${y - 2}" ${ln.thin ? 'stroke-dasharray="4 5"' : ""}/>`; }
      svg += `<g class="ov-node" data-l="${li}" data-i="${i}"><rect x="${ln.x}" y="${y}" width="${w}" height="${H}" rx="8" style="stroke:var(--s-${n[2]})"/>` +
        `<text x="${ln.x + 10}" y="${y + 18}">${B.esc(n[0])}</text><text class="sub" x="${ln.x + 10}" y="${y + 33}">${B.esc(n[1])}</text></g>`;
    });
  });
  svg += `<circle class="ov-pk0" r="5" fill="var(--accent)"/><circle class="ov-pk1" r="5" fill="#ee4c2c"/></svg>`;
  root.querySelector(".ov-svg").innerHTML = svg;
  const headEl = root.querySelector(".ov-head"), pre = root.querySelector("pre.code");
  const nodes = [...root.querySelectorAll(".ov-node")];
  // play order alternates between the lanes so the two compilers advance together
  const order = [];
  for (let i = 0; i < 10; i++) { order.push([0, i]); order.push([1, i]); if (i === 0) order.push([2, 0]); if (i === 9) order.push([2, 1]); }
  let cur = 0, timer = null, shown = null;
  function packet(li, i) {
    const c = root.querySelector(".ov-pk" + li); if (!c || li > 1) return;
    const ln = lanes[li], y = TOP + i * (H + GAP) + H / 2;
    c.setAttribute("cx", ln.x + W - 12); c.style.transition = "cy .6s ease"; c.setAttribute("cy", y);
  }
  function show(li, i) {
    shown = [li, i];
    nodes.forEach((g) => { const L = +g.dataset.l, I = +g.dataset.i; g.classList.toggle("on", L === li && I === i); g.classList.toggle("done", L === li && I < i); });
    const n = lanes[li].nodes[i];
    headEl.innerHTML = `<span class="chip" style="background:var(--s-${n[2]});color:#081018">${B.esc(lanes[li].title)}</span> <b>${B.esc(n[0])}</b><br><span style="color:#bcc6d1">${B.esc(n[5])}</span>`;
    pre.classList.add("fade");
    setTimeout(() => { pre.innerHTML = B.codeHTML(n[4], n[3]); pre.scrollTop = 0; pre.classList.remove("fade"); }, 180);
    packet(li, i);
  }
  const step = (d) => { cur = (cur + d + order.length) % order.length; show(...order[cur]); };
  nodes.forEach((g) => g.addEventListener("click", () => { stop(); const p = [+g.dataset.l, +g.dataset.i]; cur = order.findIndex((o) => o[0] === p[0] && o[1] === p[1]); show(...p); }));
  const playBtn = root.querySelector(".ov-play");
  function stop() { if (timer) { clearInterval(timer); timer = null; playBtn.textContent = "▶ play"; } }
  playBtn.onclick = () => { if (timer) return stop(); playBtn.textContent = "❚❚ pause"; timer = setInterval(() => step(1), 2600); };
  root.querySelector(".ov-prev").onclick = () => { stop(); step(-1); };
  root.querySelector(".ov-next").onclick = () => { stop(); step(1); };
  packet(1, 0); show(0, 0);
  B.whenVisible(root, () => { if (!timer) playBtn.onclick(); });
});
