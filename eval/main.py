#!/usr/bin/env python3
"""
eval/main.py — PathMMU benchmark evaluation for MedHuggingGPT models.

Mirrors the structure of PathMMU/eval/main.py so outputs are compatible
with PathMMU's print_results.py.

Usage (run from project root):
    uv run eval/main.py --model biomedclip --categories pdtt clstt att edutt
    uv run eval/main.py --model medgemma   --categories pdtt --n 20  # quick smoke-test
    uv run eval/main.py --model conch      --categories clstt --exp_name conch_pathcls
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PATHMMU_ROOT = PROJECT_ROOT.parent / "PathMMU"
PATHMMU_EVAL = PATHMMU_ROOT / "eval"
PATHMMU_DATA = PATHMMU_ROOT / "data"
OUTPUT_BASE  = Path(__file__).parent / "outputs"

for _p in (str(PROJECT_ROOT), str(PATHMMU_EVAL)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.data_utils import (  # PathMMU's data_utils
    CAT_SHORT2LONG,
    construct_prompt,
    get_pathmmu_data,
    save_json,
)
from utils.eval_utils import evaluate  # PathMMU's eval_utils
from eval.adapters import ALL_MODELS, predict


# ---------------------------------------------------------------------------
# Config loader (mirrors PathMMU's load_yaml)
# ---------------------------------------------------------------------------

def _load_config(config_path: Path) -> dict:
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate MedHuggingGPT models on PathMMU benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", required=True,
        choices=sorted(ALL_MODELS),
        help="Model to evaluate",
    )
    parser.add_argument(
        "--categories", nargs="+", required=True,
        help=f"PathMMU category short-codes. Available: {list(CAT_SHORT2LONG.keys())}",
    )
    parser.add_argument(
        "--data_path", default=str(PATHMMU_DATA),
        help=f"Path to PathMMU data root (default: {PATHMMU_DATA})",
    )
    parser.add_argument(
        "--exp_name", default="",
        help="Experiment name used as output subdirectory (default: <model>_pathmmu)",
    )
    parser.add_argument(
        "--n", type=int, default=0,
        help="Limit samples per category for quick testing (0 = all)",
    )
    parser.add_argument(
        "--config_path",
        default=str(Path(__file__).parent / "configs" / "default.yaml"),
        help="YAML config with multi_choice_example_format",
    )
    args = parser.parse_args()

    exp_name = args.exp_name or f"{args.model}_pathmmu"
    config   = _load_config(Path(args.config_path))

    print(f"\nModel:      {args.model}")
    print(f"Experiment: {exp_name}")
    print(f"Data path:  {args.data_path}")
    print(f"Categories: {args.categories}\n")

    all_results: dict[str, dict] = {}

    for cat_short in args.categories:
        if cat_short not in CAT_SHORT2LONG:
            print(f"[WARN] Unknown category '{cat_short}', skipping.")
            continue

        category = CAT_SHORT2LONG[cat_short]
        print("#" * 60)
        print(f"  Category: {category}")
        print("#" * 60)

        raw_samples = get_pathmmu_data(args.data_path, category)
        if args.n > 0:
            raw_samples = raw_samples[: args.n]
            print(f"  [--n {args.n}] Using {len(raw_samples)} samples.")

        # Build prompt-enriched samples
        samples = []
        for s in raw_samples:
            try:
                samples.append(construct_prompt(s, config))
            except Exception as exc:
                print(f"  [WARN] construct_prompt failed for No={s['No']}: {exc}")

        # Run model
        out_samples = []
        for sample in tqdm(samples, desc=f"{args.model} / {category}"):
            try:
                response, pred_ans = predict(args.model, sample)
            except Exception as exc:
                print(f"  [ERROR] predict failed for No={sample['No']}: {exc}")
                response = "error"
                pred_ans = list(sample["index2ans"].values())[0]

            out_samples.append({
                "No":        sample["No"],
                "img_path":  sample["img_path"],
                "question":  sample["question"],
                "gt_content": sample["gt_content"],
                "response":  response,
                "answer":    sample["answer"],
                "pred_ans":  pred_ans,
                "all_choices": sample["all_choices"],
                "index2ans": sample["index2ans"],
                "prompt":    sample["final_input_prompt"],
            })

        # Evaluate
        if out_samples:
            judge_dict, metric_dict = evaluate(out_samples)
            for s in out_samples:
                s["judge"] = judge_dict.get(s["No"], "Unknown")
            metric_dict["num_example"] = len(out_samples)
        else:
            judge_dict, metric_dict = {}, {"acc": 0.0, "num_example": 0}

        acc_pct = metric_dict["acc"] * 100
        print(f"  Accuracy: {acc_pct:.1f}%  ({metric_dict['num_example']} samples)\n")
        all_results[category] = metric_dict

        # Save per-category outputs (matches PathMMU format)
        out_dir = OUTPUT_BASE / exp_name / category
        out_dir.mkdir(parents=True, exist_ok=True)
        save_json(str(out_dir / "output.json"), out_samples)
        save_json(str(out_dir / "result.json"), metric_dict)
        print(f"  Saved → {out_dir}")

    # Print summary
    if all_results:
        print("\n" + "=" * 60)
        print(f"  SUMMARY — {exp_name}")
        print("=" * 60)
        total_correct = sum(
            r["acc"] * r["num_example"] for r in all_results.values()
        )
        total_samples = sum(r["num_example"] for r in all_results.values())
        for cat, metrics in all_results.items():
            print(f"  {cat:<35} {metrics['acc']*100:5.1f}%  (n={metrics['num_example']})")
        if total_samples:
            print(f"  {'OVERALL':<35} {total_correct/total_samples*100:5.1f}%  (n={total_samples})")
        print("=" * 60)

        summary_path = OUTPUT_BASE / exp_name / "summary.json"
        save_json(str(summary_path), {
            "exp_name": exp_name,
            "model": args.model,
            "categories": all_results,
            "overall_acc": total_correct / total_samples if total_samples else 0.0,
            "total_samples": total_samples,
        })
        print(f"\n  Summary saved → {summary_path}")


if __name__ == "__main__":
    main()
