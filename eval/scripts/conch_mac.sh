#!/bin/bash
# CONCH evaluation on PathMMU pathology subsets (best fit: PathCLS, PubMed).
# Run from project root: bash eval/scripts/conch_mac.sh

PATHMMU_DATA="${PATHMMU_DATA:-../../PathMMU/data}"

uv run eval/main.py \
  --model conch \
  --exp_name conch_pathmmu_mac \
  --data_path "${PATHMMU_DATA}" \
  --categories pdtt clstt att edutt
