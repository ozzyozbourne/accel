# Runs on the Colab VM: writes vector_add.cu, compiles it with nvcc, runs it.
# The SRC placeholder is replaced with vector_add.cu before sending, e.g.:
#   python3 -c "import sys;print(open('run_on_colab.py').read().replace('@@'+'SOURCE@@',open('vector_add.cu').read()))" > /tmp/job.py
#   colab run --gpu T4 /tmp/job.py
import subprocess, sys

SRC = r'''@@SOURCE@@'''


# The Colab kernel only shows Python-level output, so capture child output and print it.
def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(f"$ {cmd}\n{r.stdout}{r.stderr}", flush=True)
    return r.returncode


open("/tmp/vector_add.cu", "w").write(SRC)
sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv")
sh("nvcc --version | tail -2")
if sh("nvcc -O2 -o /tmp/vector_add /tmp/vector_add.cu") or sh("/tmp/vector_add"):
    sys.exit(1)
