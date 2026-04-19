#!/bin/bash
# MUSK evaluation on PathMMU pathology subsets.
# Run from project root: bash eval/scripts/musk_mac.sh

PATHMMU_DATA="${PATHMMU_DATA:-../../PathMMU/data}"

uv run eval/main.py \
  --model musk \
  --exp_name musk_pathmmu_mac \
  --data_path "${PATHMMU_DATA}" \
  --categories pdtt clstt att edutt
