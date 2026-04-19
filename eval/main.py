#!/usr/bin/env python3
"""
eval/main.py — PathMMU benchmark evaluation for MedHuggingGPT ensemble.

Mirrors the system's actual runtime flow:
  1. Load image + multi-choice question
  2. Run all selected models on the image (same as Executor node)
  3. Pass all model outputs to the LLM synthesizer (same as Synthesizer node)
     — synthesizer is asked to pick the answer letter, not write a clinical report
  4. Score the synthesized answer against PathMMU ground truth

Usage (run from project root):
    # Default ensemble (biomedclip + conch + medgemma), all test_tiny categories
    bash eval/scripts/ensemble_mac.sh

    # Custom ensemble
    uv run eval/main.py --models biomedclip conch --categories pdtt clstt att edutt

    # Quick 20-sample smoke-test
    uv run eval/main.py --models medgemma --categories pdtt --n 20
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

from utils.data_utils import (
    CAT_SHORT2LONG,
    construct_prompt,
    get_pathmmu_data,
    save_json,
)
from utils.eval_utils import evaluate
from eval.adapters import ALL_MODELS, predict
from eval.parse import parse_prediction

# ---------------------------------------------------------------------------
# PathMMU category → MedHuggingGPT modality mapping
# Mirrors the modality detection the UI would apply if these images were loaded
# via datasets_config.json. Used to look up the default model suggestions.
# ---------------------------------------------------------------------------
PATHMMU_CAT_MODALITY: dict[str, str] = {
    "PubMed_test_tiny":       "2D Histopathology",
    "PubMed_test":            "2D Histopathology",
    "PubMed_val":             "2D Histopathology",
    "PathCLS_test_tiny":      "2D Histopathology",
    "PathCLS_test":           "2D Histopathology",
    "PathCLS_val":            "2D Histopathology",
    "Atlas_test_tiny":        "2D Histopathology",
    "Atlas_test":             "2D Histopathology",
    "Atlas_val":              "2D Histopathology",
    "EduContent_test_tiny":   "2D Histopathology",
    "EduContent_test":        "2D Histopathology",
    "EduContent_val":         "2D Histopathology",
    "SocialPath_test_tiny":   "Unknown",
    "SocialPath_test":        "Unknown",
    "SocialPath_val":         "Unknown",
}

# Same mapping as app.py MODALITY_MODEL_MAPPING — default top-2 per modality
MODALITY_MODEL_MAPPING: dict[str, list[str]] = {
    "Legacy MRI":       ["vit_alzheimer", "biomedclip", "medgemma", "llava_med"],
    "2D Histopathology":["conch", "musk", "biomedclip", "llava_med"],
    "3D Volumetric":    ["biomedclip", "medgemma", "llava_med", "chexagent"],
    "Hyperspectral":    ["biomedclip", "musk"],
    "CT Image":         ["chexagent", "biomedclip", "medgemma", "llava_med"],
    "DICOM CT":         ["chexagent", "biomedclip", "medgemma", "llava_med"],
    "NIfTI":            ["biomedclip", "medgemma", "llava_med"],
    "WSI Pathology":    ["conch", "musk", "biomedclip"],
    "HSI Pathology":    ["biomedclip", "musk"],
    "Unknown":          ["biomedclip", "llava_med", "medgemma"],
}

def default_models_for_category(category: str, n: int = 2) -> list[str]:
    """Return the top-n UI-suggested models for a PathMMU category."""
    modality = PATHMMU_CAT_MODALITY.get(category, "Unknown")
    return MODALITY_MODEL_MAPPING.get(modality, MODALITY_MODEL_MAPPING["Unknown"])[:n]


# ---------------------------------------------------------------------------
# LLM setup (same as agent/graph.py)
# ---------------------------------------------------------------------------

def _build_llm():
    provider = os.getenv("LLM_PROVIDER", "google").lower()
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    from langchain_google_genai import ChatGoogleGenerativeAI
    if not os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
    return ChatGoogleGenerativeAI(model=os.getenv("GOOGLE_MODEL_NAME", "gemini-1.5-flash"))


def _summarize_output(output: dict) -> str:
    """Extract the key prediction string from a model output dict."""
    if "error" in output:
        return f"[error: {output['error']}]"
    if "top1" in output:          # CLIP-style models
        return f"predicted '{output['top1']}' (confidence {output.get('top1_prob', '?')})"
    if "prediction" in output:    # VLM + ViT models
        return output["prediction"]
    return json.dumps(output)


def synthesize_answer(
    question: str,
    index2ans: dict,
    all_choices: list,
    model_outputs: list,
    llm,
) -> tuple[str, str]:
    """
    Ask the LLM to pick the correct answer letter given all model outputs.
    Returns (raw_synthesis_text, pred_ans_text).
    """
    options_text = "\n".join(f"({c}) {index2ans[c]}" for c in all_choices)
    outputs_text = "\n".join(
        f"  - {o.get('model', 'unknown')}: {_summarize_output(o)}"
        for o in model_outputs
    )

    prompt = f"""You are a pathology AI evaluator. Based on the model outputs below, answer the multiple-choice question.

Question: {question}
Options:
{options_text}

Model outputs:
{outputs_text}

Reply with ONLY the answer letter ({"/".join(all_choices)}) on the first line, then one sentence of reasoning."""

    try:
        response = llm.invoke(prompt)
        text = response.content.strip()
    except Exception as exc:
        text = f"error: {exc}"

    pred_ans = parse_prediction(text, all_choices, index2ans)
    return text, pred_ans


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def _load_config(config_path: Path) -> dict:
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate MedHuggingGPT ensemble on PathMMU benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        choices=sorted(ALL_MODELS),
        help=(
            "Models to include in the ensemble. "
            "If omitted, uses the same top-2 defaults the UI suggests per category "
            "based on MODALITY_MODEL_MAPPING."
        ),
    )
    parser.add_argument(
        "--n_models", type=int, default=2,
        help="Number of top-suggested models to use when --models is not set (default: 2)",
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
        help="Experiment name for output subdirectory (default: <models>_pathmmu)",
    )
    parser.add_argument(
        "--n", type=int, default=0,
        help="Limit samples per category for quick testing (0 = all)",
    )
    parser.add_argument(
        "--config_path",
        default=str(Path(__file__).parent / "configs" / "default.yaml"),
    )
    args = parser.parse_args()

    exp_name = args.exp_name or (
        f"{'_'.join(sorted(args.models))}_pathmmu" if args.models
        else f"ui_default_top{args.n_models}_pathmmu"
    )
    config = _load_config(Path(args.config_path))
    llm    = _build_llm()

    print(f"\nExperiment: {exp_name}")
    print(f"Data path:  {args.data_path}")
    print(f"Categories: {args.categories}")
    if args.models:
        print(f"Ensemble:   {args.models} (fixed for all categories)")
    else:
        print(f"Ensemble:   UI default top-{args.n_models} per category (modality-aware)")
    print()

    all_results: dict[str, dict] = {}

    for cat_short in args.categories:
        if cat_short not in CAT_SHORT2LONG:
            print(f"[WARN] Unknown category '{cat_short}', skipping.")
            continue

        category = CAT_SHORT2LONG[cat_short]
        # Resolve models: explicit override, or UI default for this category's modality
        models = args.models or default_models_for_category(category, args.n_models)
        modality = PATHMMU_CAT_MODALITY.get(category, "Unknown")

        print("#" * 60)
        print(f"  Category: {category}  [{modality}]")
        print(f"  Models:   {models}")
        print("#" * 60)

        raw_samples = get_pathmmu_data(args.data_path, category)
        if args.n > 0:
            raw_samples = raw_samples[: args.n]
            print(f"  [--n {args.n}] Using {len(raw_samples)} samples.")

        samples = []
        for s in raw_samples:
            try:
                samples.append(construct_prompt(s, config))
            except Exception as exc:
                print(f"  [WARN] construct_prompt failed for No={s['No']}: {exc}")

        out_samples = []
        for sample in tqdm(samples, desc=f"ensemble / {category}"):
            index2ans   = sample["index2ans"]
            all_choices = sample["all_choices"]

            # --- Step 1: run all models (Executor) ---
            model_outputs = []
            per_model_responses = {}
            for model_name in models:
                try:
                    raw_response, _ = predict(model_name, sample)
                except Exception as exc:
                    raw_response = f"error: {exc}"
                # Re-run to get the full output dict for the synthesizer
                from tools.tool_model_executor import run_model
                from eval.adapters import CLIP_MODELS, VLM_MODELS
                if model_name in CLIP_MODELS:
                    labels = [index2ans[c] for c in all_choices]
                    result = run_model(model_name, sample["img_path"], ", ".join(labels))
                elif model_name in VLM_MODELS:
                    result = run_model(model_name, sample["img_path"], sample["final_input_prompt"])
                else:
                    result = run_model(model_name, sample["img_path"], "")
                model_outputs.append(result)
                per_model_responses[model_name] = raw_response

            # --- Step 2: synthesize (Synthesizer) ---
            synthesis_text, pred_ans = synthesize_answer(
                question=sample["question"],
                index2ans=index2ans,
                all_choices=all_choices,
                model_outputs=model_outputs,
                llm=llm,
            )

            out_samples.append({
                "No":               sample["No"],
                "img_path":         sample["img_path"],
                "question":         sample["question"],
                "gt_content":       sample["gt_content"],
                "response":         synthesis_text,
                "answer":           sample["answer"],
                "pred_ans":         pred_ans,
                "all_choices":      all_choices,
                "index2ans":        index2ans,
                "prompt":           sample["final_input_prompt"],
                "model_outputs":    model_outputs,
                "per_model":        per_model_responses,
            })

        # --- Evaluate ---
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

        # --- Save (PathMMU-compatible schema) ---
        out_dir = OUTPUT_BASE / exp_name / category
        out_dir.mkdir(parents=True, exist_ok=True)
        save_json(str(out_dir / "output.json"), out_samples)
        save_json(str(out_dir / "result.json"), metric_dict)
        print(f"  Saved → {out_dir}")

    # --- Summary ---
    if all_results:
        print("\n" + "=" * 60)
        print(f"  SUMMARY — {exp_name}  [{', '.join(args.models)}]")
        print("=" * 60)
        total_correct = sum(r["acc"] * r["num_example"] for r in all_results.values())
        total_samples = sum(r["num_example"] for r in all_results.values())
        for cat, metrics in all_results.items():
            print(f"  {cat:<35} {metrics['acc']*100:5.1f}%  (n={metrics['num_example']})")
        if total_samples:
            print(f"  {'OVERALL':<35} {total_correct/total_samples*100:5.1f}%  (n={total_samples})")
        print("=" * 60)

        save_json(str(OUTPUT_BASE / exp_name / "summary.json"), {
            "exp_name":        exp_name,
            "models_override": args.models,
            "n_models":        args.n_models,
            "categories":      all_results,
            "overall_acc":     total_correct / total_samples if total_samples else 0.0,
            "total_samples":   total_samples,
        })
        print(f"\n  Summary → {OUTPUT_BASE / exp_name / 'summary.json'}")


if __name__ == "__main__":
    main()
