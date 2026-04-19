#!/bin/bash
# Full 5-model in-process ensemble (excludes subprocess models chexagent/llava_med).
# High latency — use --n 20 for smoke-testing.
# Run from project root: bash eval/scripts/full_ensemble_mac.sh

PATHMMU_DATA="${PATHMMU_DATA:-../../PathMMU/data}"
N="${N:-0}"   # set N=20 for a quick test: N=20 bash eval/scripts/full_ensemble_mac.sh

uv run eval/main.py \
  --models biomedclip conch musk medgemma vit_alzheimer \
  --exp_name full_ensemble_pathmmu_mac \
  --data_path "${PATHMMU_DATA}" \
  --categories pdtt clstt att edutt \
  ${N:+--n $N}
