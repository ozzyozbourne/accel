#!/usr/bin/env bash
# Runs on the Colab VM: install tinygrad, download Qwen2.5-1.5B-Instruct, train with GRPO in tinygrad,
# then score the adapter with the same vLLM eval as the TRL run (../gsm8k_grpo/eval.py).
# Usage: (setsid nohup bash run.sh <steps> <eval_n> > /content/logs_tg/run.log 2>&1 < /dev/null &)
set -euxo pipefail
for s in TERM HUP INT; do trap "echo run.sh got SIG$s" $s; done
STEPS=${1:-15}
EVAL_N=${2:-500}
cd "$(dirname "$0")"
mkdir -p /content/logs_tg
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
pip install -q "tinygrad==0.14.0" tokenizers datasets safetensors huggingface_hub
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', local_dir='/content/qwen15', allow_patterns=['*.json', '*.safetensors'])"
# tinygrad compiles kernels on every vCPU; twice that starved Colab's runtime agent and the VM was
# reclaimed. Keep CPU 0 free for Colab and run training at lower priority.
N=$(nproc)
DEV=${TG_DEV:-CUDA} PARALLEL=$((N > 3 ? N - 2 : 1)) taskset -c 1-$((N - 1)) nice -n 10 python3 train.py --model /content/qwen15 --steps "$STEPS" --out /content/tg_grpo > /content/logs_tg/train.log 2>&1
echo TRAIN-DONE
if [ "$EVAL_N" -gt 0 ]; then
  pip install -q uv && uv pip install --system -q "vllm==0.32.0" 2>&1 | tail -3
  cd ../gsm8k_grpo
  python3 eval.py --adapter /content/tg_grpo/final --n "$EVAL_N" --out /content/logs_tg/eval_trained.jsonl > /content/logs_tg/eval_trained.log 2>&1
  grep RESULT /content/logs_tg/eval_trained.log
fi
echo RUN-DONE
