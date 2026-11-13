#!/usr/bin/env bash
# Runs on the Colab VM: install, base-model eval, GRPO training, trained-model eval.
# Usage: nohup bash run.sh <max_steps> <eval_n> > /content/logs/run.log 2>&1 &
set -euxo pipefail
STEPS=${1:-15}
EVAL_N=${2:-500}
cd "$(dirname "$0")"
mkdir -p /content/logs
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
pip install -q uv
uv pip install --system -q "trl[vllm,peft]==1.17.1" 2>&1 | tail -3
# Colab preinstalls torchao 0.10; PEFT refuses to import with it, and nothing here uses it.
pip uninstall -q -y torchao || true
python3 -c "import torch, vllm, trl, transformers, peft; print('torch', torch.__version__, 'vllm', vllm.__version__, 'trl', trl.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__)"
[ -f /content/logs/eval_base.done ] || { python3 eval.py --n "$EVAL_N" --out /content/logs/eval_base.jsonl > /content/logs/eval_base.log 2>&1 && touch /content/logs/eval_base.done; }
grep RESULT /content/logs/eval_base.log
python3 train.py --max_steps "$STEPS" --output_dir /content/grpo_gsm8k > /content/logs/train.log 2>&1
python3 eval.py --adapter /content/grpo_gsm8k/final --n "$EVAL_N" --out /content/logs/eval_trained.jsonl > /content/logs/eval_trained.log 2>&1
grep RESULT /content/logs/eval_base.log /content/logs/eval_trained.log
echo RUN-DONE
