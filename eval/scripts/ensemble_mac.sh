#!/bin/bash
# Default ensemble evaluation on PathMMU test_tiny subsets.
# Mirrors the app's default model suggestions for histopathology:
#   biomedclip + conch + medgemma
# Run from project root: bash eval/scripts/ensemble_mac.sh

PATHMMU_DATA="${PATHMMU_DATA:-../../PathMMU/data}"

uv run eval/main.py \
  --models biomedclip conch medgemma \
  --exp_name ensemble_pathmmu_mac \
  --data_path "${PATHMMU_DATA}" \
  --categories pdtt clstt att edutt
