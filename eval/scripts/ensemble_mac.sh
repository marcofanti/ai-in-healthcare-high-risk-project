#!/bin/bash
# Modality-aware ensemble evaluation on PathMMU test_tiny subsets.
# No --models flag: uses the same top-2 default models the UI suggests
# per category based on MODALITY_MODEL_MAPPING.
#   pdtt / clstt / att / edutt → 2D Histopathology → conch, musk
#   sptt                       → Unknown           → biomedclip, llava_med
# Run from project root: bash eval/scripts/ensemble_mac.sh

PATHMMU_DATA="${PATHMMU_DATA:-../../PathMMU/data}"

uv run eval/main.py \
  --exp_name ui_default_pathmmu_mac \
  --data_path "${PATHMMU_DATA}" \
  --categories pdtt clstt att edutt
