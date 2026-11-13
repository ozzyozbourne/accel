/* Chapter 5: reductions.
 * - .kname[data-keys]  : decodes captured kernel names using the colours tinygrad itself printed (axis types)
 * - #redtree           : one block of a GROUPTOP reduction, step by step (sizes read from the captured kernels)
 * - #split             : the three sum_all kernels on a 16M-element input, stage by stage
 */
BOOK.ready((B) => {
  const esc = B.esc;

  // ---------- kernel-name decoder ----------
  // tinygrad's KernelInfo.name carries ANSI colours, one per axis type (postrange.py get_optimized_ast). The passes keep that
  // raw string; decode the colours instead of guessing.
  const COL = { 34: ["global", "#60a5fa", "blockIdx"], 36: ["local", "#22d3ee", "threadIdx"], 91: ["group reduce", "#f87171", "threadIdx + shared mem"],
    31: ["reduce", "#fb923c", "for loop"], 33: ["upcast", "#fbbf24", "registers / vector"], 35: ["unroll", "#e879f9", "unrolled reduce"] };
  function rawNames(d) {
    const out = {};
    for (const p of d.passes) {
      if (!p.ir || p.kernel == null || out[p.kernel]) continue;
      const m = /KernelInfo\(name='([^']*)'/.exec(p.ir.text);
      if (m && m[1].includes("x1b")) out[p.kernel] = m[1];
    }
    return out;
  }
  document.querySelectorAll(".kname[data-keys]").forEach((n) => {
    const keys = n.dataset.keys.split(",");
    let rows = "";
    keys.forEach((key) => {
      const d = B.get(key), raws = rawNames(d);
      d.kernels.forEach((k, ki) => {
        const raw = raws[ki]; if (!raw) return;
        const parts = [...raw.matchAll(/\\x1b\[(\d+)m(\d+)/g)].map((m) => [+m[1], m[2]]);
        const chips = parts.map(([c, v]) => { const [t, col] = COL[c] || ["?", "#888"]; return `<span class="kn-ax" style="border-color:${col};color:${col}"><b>${v}</b><i>${t}</i></span>`; }).join("");
        rows += `<tr><td class="mono">${esc(key)}${d.kernels.length > 1 ? " #" + ki : ""}</td><td><span class="kn-k">${esc(k.name[0])}</span>${chips}</td>
          <td class="mono" style="font-size:12px">${esc((k.applied_opts || []).map((o) => o.replace("Opt(op=OptOps.", "").replace(/\)$/, "")).join(", "))}</td>
          <td class="mono">${k.global_size.join("×")} / ${k.local_size.join("×")}</td></tr>`;
      });
    });
    n.classList.add("widget");
    n.innerHTML = `<div class="widget-head"><span class="title">Kernel names, decoded from the colours tinygrad printed</span>
      ${Object.values(COL).map(([t, c, h]) => `<span style="font-size:12px;color:${c}">■ ${t} <span style="color:var(--muted)">(${h})</span></span>`).join("")}</div>
      <div class="widget-body" style="overflow-x:auto"><table class="kn-t"><tr><th>capture</th><th>name, one box per axis</th><th>applied opts</th><th>grid / block</th></tr>${rows}</table></div>
      <div class="code-note">The prefix is <code>r</code> when the kernel has a REDUCE and <code>E</code> otherwise. Colours come from the raw <code>KernelInfo.name</code> in the captured passes.</div>`;
  });

  // ---------- 1. one block reducing (GROUPTOP) ----------
  (function redTree() {
    const root = document.getElementById("redtree");
    if (!root) return;
    const KS = {
      rows: { key: "sum_rows/default", k: 0, label: "sum_rows · r_1024_16_64", op: "+" },
      cols: { key: "max_cols/default", k: 0, label: "max_cols · r_1024_16_64", op: "max" },
      s1: { key: "sum_all/default", k: 1, label: "sum_all #1 · r_256_16", op: "+" },
      s2: { key: "sum_all/default", k: 2, label: "sum_all #2 · r_16_16", op: "+" },
    };
    let which = "rows", mode = "tg", step = 0, timer = null;
    // pseudo-random illustrative inputs
    const rnd = (i) => { const x = Math.sin(i * 12.9898 + 78.233) * 43758.5453; return Math.round((x - Math.floor(x) - 0.5) * 20) / 10; };
    function spec() {
      const c = KS[which], k = B.get(c.key).kernels[c.k];
      const T = +k.local_size[0], m = /for \(int Ridx0 = 0; Ridx0 < (\d+);/.exec(k.source), L = m ? +m[1] : 1;
      const m2 = /for \(int (Ridx1\d\d) = 0; \1 < (\d+);/.exec(k.source);
      return { ...c, kern: k, T, L, G: m2 ? +m2[2] : T, ridx: m2 ? m2[1] : "Ridx10x" };
    }
    root.classList.add("widget");
    root.innerHTML = `<div class="widget-head" style="gap: 8px 10px"><span class="title">One block of a GROUPTOP reduction</span>
      <select class="btn ksel">${Object.entries(KS).map(([k, v]) => `<option value="${k}">${esc(v.label)}</option>`).join("")}</select>
      <div class="tabs"><button class="btn on" data-mode="tg">tinygrad: smem, one reader</button><button class="btn" data-mode="tree">contrast: butterfly (SHFL.BFLY)</button></div>
      <button class="btn prev">◀</button><button class="btn play">▶ play</button><button class="btn next">▶</button><button class="btn reset">reset</button></div>
      <div class="widget-body"><div class="rt-phase"></div><div class="rt-svg"></div><pre class="code rt-src"></pre></div>
      <div class="code-note">Thread count, loop trip counts and the source lines come from the captured kernel. The numbers in the boxes are made-up example data. In <em>tinygrad</em> mode every thread runs the second loop in the CUDA C, but in the SASS ptxas makes the threads with lidx0 ≠ 0 <code>EXIT</code> right after the barrier, because only lidx0 == 0 stores. The tree mode shows how a warp-shuffle reduction (what Triton's <code>tl.sum</code> lowers to) does the same job in log₂ steps.</div>`;
    const svgBox = root.querySelector(".rt-svg");
    function plan() {
      const s = spec(), steps = [];
      if (s.L > 1 || which !== "s1") for (let r = 0; r < s.L; r++) steps.push({ ph: "loop", r });
      else steps.push({ ph: "load" });
      if (mode === "tg") steps.push({ ph: "sts" }, { ph: "bar" });
      if (mode === "tg") { for (let g = 0; g < s.G; g++) steps.push({ ph: "final", g }); }
      else { for (let w = s.T / 2; w >= 1; w /= 2) steps.push({ ph: "tree", w }); }
      steps.push({ ph: "store" });
      return { s, steps };
    }
    function accs(s) { // per-thread partials after phase 1, from the example data
      const vals = [];
      for (let t = 0; t < s.T; t++) { let a = s.op === "max" ? -Infinity : 0; for (let r = 0; r < s.L; r++) { const v = rnd(t * 997 + r); a = s.op === "max" ? Math.max(a, v) : a + v; } vals.push(a); }
      return vals;
    }
    const f = (v) => (v === -Infinity ? "-inf" : (Math.round(v * 10) / 10).toFixed(1));
    function render() {
      const { s, steps } = plan(); step = Math.max(0, Math.min(step, steps.length - 1));
      const st = steps[step], idx = steps.indexOf(st), W = 900, cw = 50, x0 = (W - s.T * cw) / 2, part = accs(s);
      const ident = s.op === "max" ? -Infinity : 0, comb = (a, b) => (s.op === "max" ? Math.max(a, b) : a + b);
      // phase-1 progress
      const loopDone = steps.findIndex((x) => x.ph !== "loop" && x.ph !== "load"), rDone = st.ph === "loop" ? st.r + 1 : (st.ph === "load" || idx >= loopDone) ? s.L : 0;
      const cur = []; for (let t = 0; t < s.T; t++) { let a = ident; for (let r = 0; r < rDone; r++) { const v = rnd(t * 997 + r); a = comb(a, v); } cur.push(a); }
      const smemOn = idx >= loopDone, barOn = idx >= loopDone + 1;
      let svg = `<svg viewBox="0 0 ${W} 400" style="width:100%;display:block">`;
      // input strips
      const sh = Math.max(24, Math.min(120, s.L * 2 + 4));
      svg += `<text class="rt-l" x="10" y="22">global memory: ${s.L} value${s.L > 1 ? "s" : ""} per thread ${s.L > 1 ? "(Ridx0 loop)" : "(no loop)"}</text>`;
      for (let t = 0; t < s.T; t++) {
        const x = x0 + t * cw;
        for (let r = 0; r < s.L; r++) { const h = (sh - 4) / s.L; svg += `<rect x="${x + 8}" y="${30 + r * h}" width="${cw - 16}" height="${Math.max(0.8, h - (s.L > 32 ? 0 : 1))}" class="rt-in ${r < rDone ? "done" : ""} ${st.ph === "loop" && r === st.r ? "now" : ""}"/>`; }
      }
      // accumulators (registers)
      const ay = 58 + sh;
      svg += `<text class="rt-l" x="${x0}" y="${ay - 8}">registers: one accumulator per thread (float buf0[1])</text>`;
      for (let t = 0; t < s.T; t++) {
        const x = x0 + t * cw, alive = mode === "tg" ? !(barOn && idx > loopDone + 1 && t !== 0) : true;
        let v = cur[t];
        if (mode === "tree" && st.ph === "tree" || (mode === "tree" && st.ph === "store")) {
          // butterfly: after k steps each thread holds combination of its 2^k neighbourhood
          let vals = part.slice(); for (const x2 of steps.slice(0, idx + 1).filter((q) => q.ph === "tree")) vals = vals.map((a, i) => comb(a, vals[i ^ x2.w]));
          v = vals[t];
        }
        svg += `<g class="rt-acc ${alive ? "" : "dead"}"><rect x="${x + 3}" y="${ay}" width="${cw - 6}" height="28" rx="5"/><text x="${x + cw / 2}" y="${ay + 18}">${f(v)}</text>
          <text class="rt-tid" x="${x + cw / 2}" y="${ay + 42}">t${t}</text></g>`;
      }
      if (mode === "tg") {
        // shared memory
        const my = ay + 80;
        svg += `<text class="rt-l" x="${x0}" y="${my - 8}">shared memory: __shared__ float buf[${s.T}]</text>`;
        for (let t = 0; t < s.T; t++) {
          const x = x0 + t * cw, reading = st.ph === "final" && st.g === t, read = st.ph === "final" ? t < st.g : st.ph === "store";
          svg += `<g class="rt-sm ${smemOn ? "on" : ""} ${reading ? "now" : ""} ${read ? "used" : ""}"><rect x="${x + 3}" y="${my}" width="${cw - 6}" height="28" rx="3"/><text x="${x + cw / 2}" y="${my + 18}">${smemOn ? f(part[t]) : ""}</text></g>`;
          if (st.ph === "sts") svg += `<path class="rt-arrow" d="M${x + cw / 2} ${ay + 30} L${x + cw / 2} ${my - 2}"/>`;
        }
        svg += `<line class="rt-bar ${barOn ? "on" : ""}" x1="${x0 - 10}" x2="${x0 + s.T * cw + 10}" y1="${my + 44}" y2="${my + 44}"/>
          <text class="rt-l" x="${x0 + s.T * cw - 110}" y="${my + 62}" style="fill:${barOn ? "#f87171" : "#55606e"}">__syncthreads()</text>`;
        // thread 0 final accumulator
        const fy = my + 80, g = st.ph === "final" ? st.g + 1 : st.ph === "store" ? s.G : 0;
        let acc = ident; for (let i = 0; i < g; i++) acc = comb(acc, part[i]);
        svg += `<text class="rt-l" x="${x0}" y="${fy - 8}">thread 0: second loop over the shared slots</text><g class="rt-acc fin ${g ? "on" : ""}"><rect x="${x0 + 3}" y="${fy}" width="${cw * 2}" height="28" rx="5"/><text x="${x0 + cw + 3}" y="${fy + 18}">${g ? f(acc) : f(ident)}</text></g>`;
        if (st.ph === "final") svg += `<path class="rt-arrow" d="M${x0 + st.g * cw + cw / 2} ${my + 30} L${x0 + cw} ${fy - 2}"/>`;
        svg += `<g class="rt-out ${st.ph === "store" ? "on" : ""}"><rect x="${x0 + cw * 2 + 30}" y="${fy}" width="${cw * 4}" height="28" rx="5"/><text x="${x0 + cw * 4 + 30}" y="${fy + 18}">${st.ph === "store" ? "out[gidx0] = " + f(acc) : "if (lidx0==0) store"}</text></g>`;
      } else {
        const ty = ay + 90;
        if (st.ph === "tree") for (let t = 0; t < s.T; t++) { const p = t ^ st.w, x = x0 + t * cw + cw / 2, xp = x0 + p * cw + cw / 2;
          svg += `<path class="rt-arrow tree" d="M${xp} ${ay + 30} Q ${(x + xp) / 2} ${ty + 40 + Math.log2(st.w) * 18} ${x} ${ay + 32}"/>`; }
        svg += `<text class="rt-l" x="10" y="${ty + 120}">${st.ph === "tree" ? `shfl.bfly xor ${st.w}: each lane combines with lane ^ ${st.w} (no shared memory, no barrier inside a warp)` : ""}</text>`;
      }
      svg += "</svg>"; svgBox.innerHTML = svg;
      const PH = { loop: `Phase 1: serial accumulation, Ridx0 = ${st.r}. Each thread adds its own value; ${s.T} threads work in parallel.`,
        load: "Phase 1: r_256_16 has no first loop (GROUPTOP took the whole 16-wide reduce axis), so each thread does one load.",
        sts: "Each thread writes its partial into shared memory: *(buf+lidx0) = acc  (STS).",
        bar: "Barrier: no thread may read buf until every thread has written its slot (BAR.SYNC).",
        final: `Phase 2: lidx0 == 0 reads slot ${st.g} (${s.ridx} loop, LDS.128 in SASS) and accumulates it.`,
        tree: `Butterfly step, offset ${st.w}: after this step every lane holds the ${s.op === "max" ? "max" : "sum"} of a group of ${s.T / st.w} lanes.`,
        store: mode === "tg" ? "Only lidx0 == 0 stores (the gpudims mask, rendered as if (lidx0==0))." : "Every lane now holds the full result; one lane stores it." };
      root.querySelector(".rt-phase").innerHTML = `<span class="chip">step ${step + 1} / ${steps.length}</span> <span class="chip">${esc(s.kern.name)}: ${s.T} threads, first loop ${s.L}, second loop ${s.G}</span> ${esc(PH[st.ph])}`;
      const lines = s.kern.source.split("\n");
      const want = { loop: /Ridx0|val0|buf0\+0\) = \(\(\*\(buf0|alu1 =/, load: /val0 =/, sts: /\*\(buf\d\+lidx0\)/, bar: /__syncthreads/, final: /Ridx1\d\d|buf\d\+Ridx1|alu7/, tree: /__shared__/, store: /lidx0==0|data0_/ }[st.ph];
      const first = lines.findIndex((l) => /__shared__/.test(l));
      root.querySelector(".rt-src").innerHTML = BOOK.codeHTML(lines.slice(Math.max(0, first - 2)).join("\n"), "c", lines.slice(Math.max(0, first - 2)).map((l) => (want.test(l) ? "hl" : "")));
    }
    function stop() { if (timer) { clearTimeout(timer); timer = null; root.querySelector(".play").textContent = "▶ play"; } }
    root.querySelector(".play").onclick = () => {
      if (timer) return stop();
      const n = plan().steps.length; if (step >= n - 1) step = 0;
      root.querySelector(".play").textContent = "❚❚ pause";
      const tick = () => { const ss = plan().steps; if (!timer) return; if (step >= ss.length - 1) return stop(); step++; render(); timer = setTimeout(tick, ss[step].ph === "loop" ? 110 : 650); };
      timer = setTimeout(tick, 300);
    };
    root.querySelector(".next").onclick = () => { stop(); step++; render(); };
    root.querySelector(".prev").onclick = () => { stop(); step--; render(); };
    root.querySelector(".reset").onclick = () => { stop(); step = 0; render(); };
    root.querySelector(".ksel").onchange = (e) => { stop(); which = e.target.value; step = 0; render(); };
    root.querySelectorAll("[data-mode]").forEach((b) => (b.onclick = () => { stop(); mode = b.dataset.mode; step = 0; root.querySelectorAll("[data-mode]").forEach((x) => x.classList.toggle("on", x === b)); render(); }));
    render();
  })();

  // ---------- 2. the three sum_all kernels ----------
  (function split() {
    const root = document.getElementById("split");
    if (!root) return;
    const d = B.get("sum_all/default"), K = d.kernels;
    const STAGES = [
      { t: "Input: 4096 × 4096 floats (64 MB). Ratio 16,777,216 / 1 ≥ 32768, so split_reduceop fires (twice)." },
      { k: 0, t: `Kernel 0 ${K[0].name}: ${K[0].global_size[0]} blocks × ${K[0].local_size[0]} threads. Each thread owns 4 whole rows (UPCAST 4) and walks them with 1024 float4 loads (Ridx0 × UNROLL 4). Result: 4096 row sums.` },
      { k: 1, t: `Kernel 1 ${K[1].name}: ${K[1].global_size[0]} blocks × ${K[1].local_size[0]} threads. Each block adds 16 consecutive row sums through shared memory → 256 partials.` },
      { k: 2, t: `Kernel 2 ${K[2].name}: 1 block × ${K[2].local_size[0]} threads. Each thread adds 16 partials, then thread 0 adds the 16 results → 1 float.` },
    ];
    let stage = 0, prog = 0, raf = null;
    root.classList.add("widget");
    root.innerHTML = `<div class="widget-head"><span class="title">a.sum() on 4096×4096: three kernels</span>
      <div class="tabs">${STAGES.map((_, i) => `<button class="btn" data-s="${i}">${i === 0 ? "input" : "kernel " + (i - 1)}</button>`).join("")}</div>
      <button class="btn play">▶ play all</button></div>
      <div class="widget-body"><div class="sp-txt"></div><div class="sp-svg"></div></div>
      <div class="code-note">Kernel names, grid/block sizes and times are from the capture. Bottom strip: the L4's 58 SMs and how many of them the launch can keep busy (a block runs on one SM).</div>`;
    const box = root.querySelector(".sp-svg"), HUE = (i, n) => `hsl(${(i * 360) / n},70%,60%)`;
    function draw() {
      root.querySelectorAll("[data-s]").forEach((b) => b.classList.toggle("on", +b.dataset.s === stage));
      const st = STAGES[stage], W = 940, M = 256;
      let s = `<svg viewBox="0 0 ${W} 400" style="width:100%;display:block">`;
      // input matrix with 32 block bands
      s += `<text class="sp-l" x="20" y="20">input 4096×4096</text><rect x="20" y="30" width="${M}" height="${M}" class="sp-mat"/>`;
      if (stage >= 1) for (let b = 0; b < 32; b++) {
        const w = stage === 1 ? M * prog : M;
        s += `<rect x="20" y="${30 + b * 8}" width="${w}" height="7.4" style="fill:${HUE(b, 32)};opacity:.75"/>`;
      }
      if (stage === 1) s += `<line x1="${20 + M * prog}" x2="${20 + M * prog}" y1="30" y2="${30 + M}" class="sp-sweep"/>`;
      // partials 64x64
      const px = 330;
      s += `<text class="sp-l" x="${px}" y="20">4096 row sums</text><rect x="${px}" y="30" width="${M}" height="${M}" class="sp-mat"/>`;
      if (stage >= 1) { const filled = stage === 1 ? Math.floor(prog * 1024) * 4 : 4096;
        if (stage === 1 && prog >= 1 || stage > 1) for (let r = 0; r < 64; r++) s += `<rect x="${px}" y="${30 + r * 4}" width="${M}" height="3.6" style="fill:${HUE(Math.floor(r / 2), 32)};opacity:${stage > 1 ? 0.35 : 0.8}"/>`;
        else s += `<text class="sp-l" x="${px + M / 2}" y="${30 + M / 2}" style="fill:#8b97a6;text-anchor:middle">accumulating in registers…</text><text class="sp-l" x="${px + M / 2}" y="${48 + M / 2}" style="fill:#8b97a6;text-anchor:middle">${Math.round(prog * 1024)} / 1024 float4 steps</text>`;
        if (filled < 0) s += ""; }
      if (stage === 2) { // highlight groups of 16 consecutive partials
        const g = Math.floor(prog * 256);
        for (let i = 0; i < Math.min(g, 256); i++) s += `<rect x="${px + (i % 4) * 64}" y="${30 + Math.floor(i / 4) * 4}" width="62" height="3.6" style="fill:${HUE(i, 256)}"/>`;
      }
      // 256 partials 16x16
      const qx = 640, cs = 12;
      s += `<text class="sp-l" x="${qx}" y="20">256 partials</text>`;
      for (let i = 0; i < 256; i++) {
        const on = stage > 2 || (stage === 2 && i < prog * 256), x = qx + (i % 16) * cs, y = 30 + Math.floor(i / 16) * cs;
        s += `<rect x="${x}" y="${y}" width="${cs - 1.5}" height="${cs - 1.5}" rx="2" style="fill:${on ? (stage === 3 ? HUE(Math.floor(i / 16), 16) : HUE(i, 256)) : "#10161e"};stroke:#2a3441"/>`;
      }
      // final
      const fx = 860;
      s += `<text class="sp-l" x="${fx - 20}" y="20">result</text><rect x="${fx}" y="110" width="40" height="40" rx="6" class="sp-final ${stage === 3 && prog >= 1 ? "on" : ""}"/>`;
      if (stage === 3 && prog >= 1) s += `<text x="${fx + 20}" y="170" class="sp-l" style="text-anchor:middle">1 float</text>`;
      // arrows
      s += `<path class="sp-ar" d="M${20 + M + 8} 158 H${px - 8}"/><path class="sp-ar" d="M${px + M + 8} 158 H${qx - 8}"/><path class="sp-ar" d="M${qx + 16 * cs + 6} 130 H${fx - 6}"/>`;
      // SM strip
      const used = stage === 0 ? 0 : Math.min(58, +K[st.k].global_size[0]);
      s += `<text class="sp-l" x="20" y="318">SMs busy: ${used} / 58${stage === 1 ? " (32 blocks of 32 threads: 1024 threads on the whole GPU)" : stage === 2 ? " (256 blocks of 16 threads)" : stage === 3 ? " (one block)" : ""}</text>`;
      for (let i = 0; i < 58; i++) s += `<rect x="${20 + i * 15.6}" y="330" width="13" height="22" rx="2" style="fill:${i < used ? "var(--s-run)" : "#1c2430"}"/>`;
      if (stage >= 1) { const k = K[st.k]; s += `<text class="sp-l" x="20" y="384">${esc(k.name)} · ${BOOK.fmtUs(k.time_us)} on the L4 · ${k.regs ?? "?"} registers · opts ${esc((k.applied_opts || []).map((o) => o.replace("Opt(op=OptOps.", "").replace(/\)$/, "")).join(", "))}</text>`; }
      s += "</svg>"; box.innerHTML = s;
      root.querySelector(".sp-txt").textContent = st.t;
    }
    function animate(onDone) {
      cancelAnimationFrame(raf); prog = 0; const t0 = performance.now(), dur = stage === 1 ? 3200 : 1600;
      const f = (t) => { prog = Math.min(1, (t - t0) / dur); draw(); if (prog < 1) raf = requestAnimationFrame(f); else if (onDone) onDone(); };
      raf = requestAnimationFrame(f);
    }
    root.querySelectorAll("[data-s]").forEach((b) => (b.onclick = () => { stage = +b.dataset.s; if (stage === 0) { prog = 1; draw(); } else animate(); }));
    root.querySelector(".play").onclick = () => {
      stage = 0; prog = 1; draw();
      const next = () => { if (stage >= 3) return; stage++; animate(() => setTimeout(next, 700)); };
      setTimeout(next, 700);
    };
    prog = 1; draw();
  })();
});
