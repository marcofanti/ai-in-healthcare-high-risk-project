#!/bin/bash
# MedGemma evaluation on PathMMU test_tiny subsets.
# Run from project root: bash eval/scripts/medgemma_mac.sh

PATHMMU_DATA="${PATHMMU_DATA:-../../PathMMU/data}"

uv run eval/main.py \
  --model medgemma \
  --exp_name medgemma_pathmmu_mac \
  --data_path "${PATHMMU_DATA}" \
  --categories pdtt clstt att edutt
