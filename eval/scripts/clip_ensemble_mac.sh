#!/bin/bash
# CLIP-only ensemble: biomedclip + conch + musk
# Fast — no LLM inference per model, only for synthesis.
# Run from project root: bash eval/scripts/clip_ensemble_mac.sh

PATHMMU_DATA="${PATHMMU_DATA:-../../PathMMU/data}"

uv run eval/main.py \
  --models biomedclip conch musk \
  --exp_name clip_ensemble_pathmmu_mac \
  --data_path "${PATHMMU_DATA}" \
  --categories pdtt clstt att edutt
