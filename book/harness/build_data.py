"""Turn the raw captures (book/captures/EXAMPLE/VARIANT.json, from run_all.py on the L4) into the files the
book loads: book/data/EXAMPLE__VARIANT.js, each calling BOOK.register("EXAMPLE/VARIANT", {...}).
Script tags work from file:// where fetch() of JSON does not, so the book opens straight from disk.

Also: strips the hex encoding lines from SASS, and adds per kernel `sass_mix` (instruction counts by
opcode) and `regs` (registers per thread from ptxas -v).
Usage: python build_data.py   (paths are relative to this file)"""
import collections, glob, json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
BOOK = os.path.dirname(HERE)
SASS_LINE = re.compile(r"^\s*/\*[0-9a-f]{4}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_]*)((?:\.[A-Z0-9_]+)*)")


def clean_sass(text: str) -> str:
  out = []
  for line in text.splitlines():
    if re.match(r"^\s*/\* 0x[0-9a-f]{16} \*/\s*$", line): continue  # second encoding word on its own line
    out.append(re.sub(r"\s*/\* 0x[0-9a-f]{16} \*/\s*$", "", line).rstrip())
  return "\n".join(out)


def sass_mix(text: str) -> dict:
  c = collections.Counter()
  for line in text.splitlines():
    m = SASS_LINE.match(line)
    if m: c[m.group(1)] += 1
  return dict(c.most_common())


def fix_kernel(k: dict):
  if "sass" in k:
    k["sass"]["text"] = clean_sass(k["sass"]["text"])
    k["sass_mix"] = sass_mix(k["sass"]["text"])
  m = re.search(r"Used (\d+) registers", k.get("ptxas_info", "") or k.get("res_usage", "") or "")
  if m: k["regs"] = int(m.group(1))
  m = re.search(r"(\d+) bytes smem", k.get("ptxas_info", "") or "")
  if m: k["smem"] = int(m.group(1))


def main():
  os.makedirs(f"{BOOK}/data", exist_ok=True)
  index = {}
  for f in sorted(glob.glob(f"{BOOK}/captures/*/*.json")):
    ex, var = f.split("/")[-2], os.path.basename(f)[:-5]
    d = json.load(open(f))
    for k in d.get("kernels", []): fix_kernel(k)
    for k in d.get("triton_kernels", []): fix_kernel(k)
    for p in d.get("passes", []):  # tinygrad's pyrender can't print graphs holding a CALL (split kernels and later)
      if (p.get("ir") or {}).get("text", "").startswith("<pyrender failed: call can't be pyrendered>"):
        p["ir"]["text"] = (f"# (no IR dump: after this pass the graph holds CALL nodes, which pyrender can't print;\n"
                           f"#  {p['nodes']} UOps, changed={p['changed']}. The kernels themselves appear in the codegen stage.)")
        p["ir"]["lines"] = 2
    key = f"{ex}/{var}"
    with open(f"{BOOK}/data/{ex}__{var}.js", "w") as out:
      out.write(f"BOOK.register({json.dumps(key)}, {json.dumps(d, separators=(',', ':'))});\n")
    entry = {"chapter": d.get("chapter")}
    if "kernels" in d:
      entry.update(kernels=[k.get("name") for k in d["kernels"]], time_us=round(sum(k.get("time_us") or 0 for k in d["kernels"]), 2),
                   ok=d["check"]["ok"], passes=len(d["passes"]))
    if "eager_kernels" in d:
      entry.update(eager=[k["name"][:60] for k in d["eager_kernels"]], triton=[k["name"] for k in d["triton_kernels"]])
    index[key] = entry
  with open(f"{BOOK}/data/index.js", "w") as out:
    out.write(f"BOOK.register('index', {json.dumps(index, indent=1)});\n")
  print(f"wrote {len(index)} data files")


if __name__ == "__main__":
  main()
