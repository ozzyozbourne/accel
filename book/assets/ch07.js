/* Chapter 7: tensor cores. Custom widgets:
 *   #frag        m16n8k16 fragment layout (PTX ISA), lane by lane, then the mma
 *   #frag-check  the same layout recovered from the captured CUDA source's loads
 *   #tc-trace    the TC opt and the hand-coded opts after it, axis by axis
 *   #tc-cost     math instructions one thread issues per 1024 multiply-adds (from sass_mix)
 *   #tc-tflops   achieved TFLOP/s vs the L4's approximate fp16 tensor-core peak
 */
BOOK.ready(() => {
  const esc = BOOK.esc;
  // PTX ISA, "Matrix Fragments for mma.m16n8k16 with floating point type" (.f16 A/B, .f32 C/D)
  // groupID = lane >> 2, threadID_in_group = lane % 4
  const A = (lane, i) => [(lane >> 2) + 8 * ((i >> 1) & 1), 2 * (lane & 3) + (i & 1) + 8 * (i >> 2)];  // [row, col] of a_i
  const B = (lane, i) => [2 * (lane & 3) + (i & 1) + 8 * (i >> 1), lane >> 2];                          // [k, n] of b_i
  const C = (lane, i) => [(lane >> 2) + 8 * (i >> 1), 2 * (lane & 3) + (i & 1)];                          // [row, col] of c_i
  const laneColor = (l, on = true) => on ? `hsl(${(l >> 2) * 45} 70% ${34 + (l & 3) * 11}%)` : "#1e2733";

  // ---- #frag ----
  const fr = document.getElementById("frag");
  if (fr) {
    const cs = 19, ax = 34, bx = ax + 16 * cs + 56, cx = bx + 8 * cs + 56, top = 46;
    const owner = { A: [], B: [], C: [] };
    for (let l = 0; l < 32; l++) {
      for (let i = 0; i < 8; i++) { const [r, c] = A(l, i); owner.A[r * 16 + c] = [l, i]; }
      for (let i = 0; i < 4; i++) { const [k, n] = B(l, i); owner.B[k * 8 + n] = [l, i]; }
      for (let i = 0; i < 4; i++) { const [r, c] = C(l, i); owner.C[r * 8 + c] = [l, i]; }
    }
    let sel = 0, shown = 32, cshown = 1, timer = null, mode = "lane";
    fr.classList.add("widget");
    fr.innerHTML = `<div class="widget-head"><span class="title">mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32: who holds what</span>
      <button class="btn play">▶ play: load fragments, then mma</button>
      <label class="fr-lab">lane <input type="range" min="0" max="31" value="0" class="ln"> <b class="lnv">0</b></label>
      <div class="tabs"><button class="btn on" data-m="lane">colour by lane</button><button class="btn" data-m="reg">colour by register</button></div></div>
      <div class="widget-body"><svg class="fr-svg" viewBox="0 0 ${cx + 8 * cs + 20} ${top + 16 * cs + 30}"></svg><div class="fr-info"></div></div>`;
    const svg = fr.querySelector("svg"), info = fr.querySelector(".fr-info");
    const REGC = ["#60a5fa", "#93c5fd", "#f472b6", "#f9a8d4", "#34d399", "#86efac", "#fbbf24", "#fde68a"];
    function cell(x, y, l, i, letter, vis) {
      const isSel = l === sel, fill = !vis ? "#151b23" : mode === "lane" ? laneColor(l) : REGC[letter === "a" ? i : i * 2];
      return `<g class="fr-c ${isSel ? "sel" : ""}" data-l="${l}"><rect x="${x}" y="${y}" width="${cs - 1.5}" height="${cs - 1.5}" rx="2" fill="${fill}" opacity="${vis ? (isSel || sel < 0 ? 1 : 0.55) : 1}"/>` +
        (vis ? `<text x="${x + cs / 2 - 0.7}" y="${y + cs / 2 + 3.5}">${letter}${i}</text>` : "") + `</g>`;
    }
    function draw() {
      let h = `<text class="fr-t" x="${ax}" y="18">A: 16×16 f16 (row-major, M×K)</text><text class="fr-t" x="${bx}" y="18">B: 16×8 f16 (K×N, "col")</text><text class="fr-t" x="${cx}" y="18">C/D: 16×8 f32 (M×N)</text>`;
      h += `<text class="fr-op" x="${bx - 28}" y="${top + 8 * cs}">×</text><text class="fr-op" x="${cx - 28}" y="${top + 8 * cs}">→</text>`;
      for (let r = 0; r < 16; r++) h += `<text class="fr-ix" x="${ax - 6}" y="${top + r * cs + cs / 2 + 4}">${r}</text>`;
      for (let c = 0; c < 16; c++) h += `<text class="fr-ix" x="${ax + c * cs + cs / 2}" y="${top - 5}" text-anchor="middle">${c}</text>`;
      for (let r = 0; r < 16; r++) for (let c = 0; c < 16; c++) { const [l, i] = owner.A[r * 16 + c]; h += cell(ax + c * cs, top + r * cs, l, i, "a", l < shown); }
      for (let k = 0; k < 16; k++) for (let n = 0; n < 8; n++) { const [l, i] = owner.B[k * 8 + n]; h += cell(bx + n * cs, top + k * cs, l, i, "b", l < shown); }
      for (let r = 0; r < 16; r++) for (let c = 0; c < 8; c++) { const [l, i] = owner.C[r * 8 + c]; h += cell(cx + c * cs, top + r * cs, l, i, "c", cshown && (r * 8 + c) < cshown * 128); }
      if (sel >= 0 && shown > sel) {  // the dot products lane sel's c0 depends on
        const [r, c] = C(sel, 0);
        h += `<rect class="fr-hl" x="${ax - 1}" y="${top + r * cs - 1}" width="${16 * cs + 1}" height="${cs + 1}"/><rect class="fr-hl" x="${bx + c * cs - 1}" y="${top - 1}" width="${cs + 1}" height="${16 * cs + 1}"/>`;
      }
      svg.innerHTML = h;
      const g = sel >> 2, t = sel & 3, p = (f, l, i) => f(l, i).join(",");
      info.innerHTML = `<div><b>lane ${sel}</b>: groupID = lane&gt;&gt;2 = <b>${g}</b>, threadID_in_group = lane%4 = <b>${t}</b>. Outlined in white: the row of A and the column of B that its c0 = C[${C(sel, 0)}] is the dot product of. That row and column are spread over up to 8 lanes, which is why the instruction needs the whole warp.</div>
        <table class="fr-t2"><tr><th>32-bit register</th><th>holds</th><th>elements</th></tr>
        ${[0, 1, 2, 3].map((j) => `<tr><td><code>a_pk[${j}]</code></td><td>a${2 * j}, a${2 * j + 1}</td><td>A[${p(A, sel, 2 * j)}], A[${p(A, sel, 2 * j + 1)}]</td></tr>`).join("")}
        ${[0, 1].map((j) => `<tr><td><code>b_pk[${j}]</code></td><td>b${2 * j}, b${2 * j + 1}</td><td>B[${p(B, sel, 2 * j)}], B[${p(B, sel, 2 * j + 1)}]</td></tr>`).join("")}
        ${[0, 1, 2, 3].map((j) => `<tr><td><code>c_pk[${j}]</code></td><td>c${j} (f32)</td><td>C[${p(C, sel, j)}]</td></tr>`).join("")}</table>`;
    }
    const setSel = (l) => { sel = l; fr.querySelector(".ln").value = l; fr.querySelector(".lnv").textContent = l; draw(); };
    fr.querySelector(".ln").oninput = (e) => setSel(+e.target.value);
    svg.onmouseover = (e) => { const g = e.target.closest(".fr-c"); if (g && +g.dataset.l !== sel && !timer) setSel(+g.dataset.l); };
    fr.querySelectorAll("[data-m]").forEach((b) => (b.onclick = () => { mode = b.dataset.m; fr.querySelectorAll("[data-m]").forEach((x) => x.classList.toggle("on", x === b)); draw(); }));
    function play() {
      if (timer) { clearInterval(timer); timer = null; fr.querySelector(".play").textContent = "▶ play: load fragments, then mma"; return; }
      shown = 0; cshown = 0; fr.querySelector(".play").textContent = "❚❚ pause";
      timer = setInterval(() => {
        if (shown < 32) { sel = shown; shown++; fr.querySelector(".ln").value = sel; fr.querySelector(".lnv").textContent = sel; }
        else if (cshown < 1) cshown += 0.125;
        else { clearInterval(timer); timer = null; fr.querySelector(".play").textContent = "▶ play: load fragments, then mma"; }
        draw();
      }, 110);
    }
    fr.querySelector(".play").onclick = play;
    draw();
    BOOK.whenVisible(fr, () => { if (!timer) play(); });
  }

  // ---- #frag-check: recover the layout from the captured CUDA source ----
  const fc = document.getElementById("frag-check");
  if (fc) {
    const src = BOOK.get("matmul_fp16/default").kernels[0].source;
    const off = {}, W = 2048;
    for (const m of src.matchAll(/half2 (val\d+) = \(\*\(\(half2\*\)\(\(data1_4194304\+(?:\(alu71\+(\d+)\)|alu71)\)\)\)\);/g)) off[m[1]] = { buf: "A", o: +(m[2] || 0) };
    for (const m of src.matchAll(/half (val\d+) = \(\*\(data2_4194304\+(?:\(alu70\+(\d+)\)|alu70)\)\);/g)) off[m[1]] = { buf: "B", o: +(m[2] || 0) };
    const rows = [];
    let allOk = true;
    for (const m of src.matchAll(/half8 (cast\d+) = make_half8\(([^)]*)\);/g)) {
      const els = m[2].split(",").map((s) => { const [v, f] = s.trim().split("."); const o = off[v].o + (f === "y" ? 1 : 0); return [Math.floor(o / W), o % W, s.trim()]; });
      const r0 = Math.min(...els.map((e) => e[0])), c0 = Math.min(...els.map((e) => e[1]));
      const rel = els.map(([r, c]) => [r - r0, c - c0]), ok = rel.every(([r, c], i) => r === A(0, i)[0] && c === A(0, i)[1]);
      allOk = allOk && ok;
      rows.push(`<tr><td><code>${m[1]}</code> (A, rows +${r0})</td><td class="mono">${rel.map(([r, c], i) => `a${i}:(${r},${c})`).join(" ")}</td><td>${ok ? "✓" : "✗"}</td></tr>`);
    }
    for (const m of src.matchAll(/half4 (cast\d+) = make_half4\(([^)]*)\);/g)) {
      const els = m[2].split(",").map((s) => { const o = off[s.trim()].o; return [Math.floor(o / W), o % W]; });
      const k0 = Math.min(...els.map((e) => e[0])), n0 = Math.min(...els.map((e) => e[1]));
      const rel = els.map(([k, n]) => [k - k0, n - n0]), ok = rel.every(([k, n], i) => k === B(0, i)[0] && n === B(0, i)[1]);
      allOk = allOk && ok;
      rows.push(`<tr><td><code>${m[1]}</code> (B, cols +${n0})</td><td class="mono">${rel.map(([k, n], i) => `b${i}:(${k},${n})`).join(" ")}</td><td>${ok ? "✓" : "✗"}</td></tr>`);
    }
    fc.innerHTML = `<table class="fc"><tr><th>operand built in the source</th><th>(row, col) of each element relative to the tile, lane 0</th><th>= PTX layout?</th></tr>${rows.join("")}</table>
      <div class="code-note">Computed in your browser from the captured <code>r_32_16_32_4_2_2_4_4_128_2</code> source: each <code>make_half8</code>/<code>make_half4</code> element is traced back to its load offset (2048 halves per row). PTX ISA expects a0..a7 at (0,0) (0,1) (8,0) (8,1) (0,8) (0,9) (8,8) (8,9) and b0..b3 at k = 0, 1, 8, 9 of column 0. <b>${allOk && rows.length === 8 ? "All 8 operands match." : "MISMATCH or parse failure."}</b></div>`;
  }

  // ---- #tc-trace (from a CPU experiment; final name checked against the L4 capture) ----
  const tt = document.getElementById("tc-trace");
  if (tt) {
    const COL = { g: "#60a5fa", w: "#a5f3fc", l: "#22d3ee", u: "#facc15", R: "#f87171", r: "#e879f9" };
    const S = [
      ["after convert_loop_to_global", "g0:2048:i g1:2048:j R0:2048:k"],
      ["TC(axis=0, arg=(-1, 0, 1))", "g0:128:i g1:256:j w0:32:lane u0:2:j u1:2:i R0:128:k r0:2:k"],
      ["UPCAST(axis=0, arg=4)", "g0:32:i g1:256:j w0:32:lane u0:2:j u1:2:i u2:4:i R0:128:k r0:2:k"],
      ["UPCAST(axis=1, arg=4)", "g0:32:i g1:64:j w0:32:lane u0:2:j u1:2:i u2:4:i u3:4:j R0:128:k r0:2:k"],
      ["LOCAL(axis=1, arg=4)", "g0:32:i g1:16:j w0:32:lane l0:4:j u0:2:j u1:2:i u2:4:i u3:4:j R0:128:k r0:2:k"]];
    const cap = BOOK.get("matmul_fp16/default").kernels[0];
    const rows = S.map(([op, ax]) => {
      const a = ax.split(" ").map((x) => x.split(":"));
      return `<tr><td class="mono">${esc(op)}</td><td>${a.map(([n, s, o]) => `<span class="axchip" style="border-color:${COL[n[0]]}"><b style="color:${COL[n[0]]}">${n}</b> ${s} <em>${o}</em></span>`).join("")}</td></tr>`;
    });
    const last = S[S.length - 1][1].split(" ").map((x) => x.split(":")[1]), name = "r_" + last.join("_");
    tt.innerHTML = `<table class="nd">${rows.join("")}</table><div class="code-note">CPU experiment (sm_89 renderer, no GPU; <code>.amp/in/scratchpad/ch0607/opt_steps.py</code>): <code>Scheduler.shape_str()</code> after each opt, with the original index each axis came from. Resulting name <code>${name}</code> ${name === cap.name ? `<span class="chip ok">= L4 capture</span>` : `<span class="chip bad">≠ capture ${esc(cap.name)}</span>`}</div>`;
  }

  // ---- #tc-cost ----
  const tc = document.getElementById("tc-cost");
  if (tc) {
    const mt = BOOK.get("matmul_fp16/default").kernels[0], mn = BOOK.get("matmul_fp16/no_tc").kernels[0];
    const mixT = mt.sass_mix, mixN = mn.sass_mix;
    // per listing: no_tc loop body = 2 iterations (ptxas unrolled by 2) × 16 accumulators × 4 k = 128 MACs per thread
    // tc loop body = 16 HMMA × (16·8·16 = 2048 MACs per warp) / 32 lanes = 1024 MACs per thread
    const nN = (mixN.HMUL2 || 0) + (mixN.HADD2 || 0) + (mixN.FADD || 0), perN = nN * 1024 / 128, perT = (mixT.HMMA || 0) * 1024 / 1024;
    tc.classList.add("widget");
    tc.innerHTML = `<div class="widget-head"><span class="title">Math instructions one thread issues for 1024 multiply-adds</span><button class="btn replay">↻ replay</button></div>
      <div class="widget-body"><div class="tcc-row"><div class="tcc-lab"><b>TC=0</b> ${esc(mn.name)}<br><span class="muted">HMUL2 + HADD2.F32 + FADD</span><br><b class="tcc-n" data-n="${perN}">0</b></div><canvas class="tcc-cv" data-n="${perN}" data-col="#34d399"></canvas></div>
      <div class="tcc-row"><div class="tcc-lab"><b>TC</b> ${esc(mt.name)}<br><span class="muted">HMMA.16816.F32</span><br><b class="tcc-n" data-n="${perT}">0</b></div><canvas class="tcc-cv" data-n="${perT}" data-col="#f472b6"></canvas></div>
      <div class="code-note">One square = one instruction in a thread's instruction stream (the warp issues it once for all 32 lanes). From the captured SASS: the TC=0 loop body (ptxas unrolled it twice; it covers 8 values of k for 16 accumulators = 128 multiply-adds per thread) has ${mixN.HMUL2} HMUL2, ${mixN.HADD2} HADD2.F32 and ${mixN.FADD} FADD; the TC loop body has ${mixT.HMMA} HMMA, each doing 16·8·16 = 2048 multiply-adds for the warp, 64 per lane. Scaled to 1024 multiply-adds: ${perN} vs ${perT}.</div></div>`;
    function run() {
      tc.querySelectorAll(".tcc-cv").forEach((cv) => {
        const n = +cv.dataset.n, s = 7, cols = 80, rowsN = Math.ceil(n / cols); cv.width = cols * s; cv.height = Math.max(1, rowsN) * s;
        const g = cv.getContext("2d"), lab = cv.parentNode.querySelector(".tcc-n"); let i = 0;
        g.clearRect(0, 0, cv.width, cv.height);
        const per = Math.max(1, Math.ceil(n / 90));
        (function frame() {
          for (let j = 0; j < per && i < n; j++, i++) { g.fillStyle = cv.dataset.col; g.fillRect((i % cols) * s, Math.floor(i / cols) * s, s - 1.5, s - 1.5); }
          lab.textContent = i; if (i < n) requestAnimationFrame(frame);
        })();
      });
    }
    tc.querySelector(".replay").onclick = run;
    BOOK.whenVisible(tc, run);
  }

  // ---- #tc-tflops ----
  const tf = document.getElementById("tc-tflops");
  if (tf) {
    const FL = 2 * 2048 ** 3, PEAK = 121;
    const g = BOOK.get("matmul_fp16/torch").eager_kernels.find((e) => /gemm/.test(e.name));
    const rows = [["tinygrad TC=0 (FP32 accumulate on CUDA cores)", BOOK.get("matmul_fp16/no_tc").kernels[0].time_us], ["tinygrad default (tensor cores)", BOOK.get("matmul_fp16/default").kernels[0].time_us], ["cuBLAS gemm kernel only", g.us]];
    tf.classList.add("widget", "bars");
    tf.innerHTML = `<div class="widget-head"><span class="title">TFLOP/s = 2·2048³ / time, against ~121 TFLOP/s dense fp16 tensor-core peak</span></div><div class="widget-body">${rows.map(([l, us], i) => {
      const v = FL / (us * 1e-6) / 1e12;
      return `<div class="row"><span>${esc(l)}</span><div class="track"><div class="fill" data-w="${100 * v / PEAK}" style="background:${i === 2 ? "#ee4c2c" : i ? "#f472b6" : "var(--s-codegen)"}"></div></div><span class="val">${v.toFixed(1)} · ${(100 * v / PEAK).toFixed(0)}%</span></div>`; }).join("")}
      <div class="code-note">Full width = 121 TFLOP/s, half of the 242 TFLOPS "with sparsity" in NVIDIA's L4 datasheet. Treat it as an approximate upper bound. The cuBLAS row counts only <code>${esc(g.name)}</code> (${BOOK.fmtUs(g.us)}), not the fp16→fp32 conversion kernel that follows it.</div></div>`;
    BOOK.whenVisible(tf, () => tf.querySelectorAll(".fill").forEach((f) => (f.style.width = Math.max(0.5, f.dataset.w) + "%")));
  }
});
