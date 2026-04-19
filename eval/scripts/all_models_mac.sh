#!/bin/bash
# Run all in-process models sequentially on PathMMU test_tiny subsets.
# Skips chexagent and llava_med (require isolated venvs — run separately).
# Run from project root: bash eval/scripts/all_models_mac.sh

PATHMMU_DATA="${PATHMMU_DATA:-../../PathMMU/data}"
CATEGORIES="pdtt clstt att edutt"

for MODEL in biomedclip conch musk medgemma vit_alzheimer; do
  echo ""
  echo "=========================================="
  echo "  Running: ${MODEL}"
  echo "=========================================="
  uv run eval/main.py \
    --model "${MODEL}" \
    --exp_name "${MODEL}_pathmmu_mac" \
    --data_path "${PATHMMU_DATA}" \
    --categories ${CATEGORIES}
done

echo ""
echo "All models complete. Results in eval/outputs/"
