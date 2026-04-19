#!/bin/bash
# BiomedCLIP evaluation on PathMMU test_tiny subsets.
# Run from project root: bash eval/scripts/biomedclip_mac.sh

PATHMMU_DATA="${PATHMMU_DATA:-../../PathMMU/data}"

uv run eval/main.py \
  --model biomedclip \
  --exp_name biomedclip_pathmmu_mac \
  --data_path "${PATHMMU_DATA}" \
  --categories pdtt clstt att edutt
