"""Prints reward and length per logged training step from train.log, plus progress."""
import ast
import re
import sys

log = open(sys.argv[1] if len(sys.argv) > 1 else "/content/logs/train.log", errors="replace").read()
for i, m in enumerate(re.finditer(r"\{'loss'.*?\}", log), 1):
    d = ast.literal_eval(m.group(0))
    print(f"step {i:3d}  reward {float(d['reward']):.3f}  len {float(d['completions/mean_length']):5.0f}  "
          f"clipped {float(d['completions/clipped_ratio']):.2f}  zero-std groups {float(d['frac_reward_zero_std']):.2f}")
bars = re.findall(r"\d+/\d+ \[[\d:]+<[\d:?]+, *[\d.]+s/it\]", log)
print(bars[-1] if bars else "no progress bar yet")
