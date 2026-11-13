#!/usr/bin/env bash
# Diagnostics for jobs that vanish on Colab: every 20 s, log the training processes, memory and GPU.
while true; do
  echo "$(date +%T) $(pgrep -fc '[t]rain.py --model') train procs, mem used $(free -m | awk '/Mem/{print $3}') MB, gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
  sleep 20
done
