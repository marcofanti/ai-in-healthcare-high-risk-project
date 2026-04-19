"""
adapters.py — Translate PathMMU multi-choice samples to each model's interface.

CLIP models (biomedclip, conch, musk):
    parse_labels() in model_utils already converts comma-separated text to candidate
    labels for zero-shot classification. We pass the answer options as the prompt.
    The returned `top1` is the predicted answer text.

VLM models (medgemma, chexagent, llava_med):
    Receive the full formatted question + options as a free-text prompt.
    Response is parsed with PathMMU's get_multi_choice_prediction().

vit_alzheimer:
    Domain-specific (Alzheimer MRI only). Included for completeness; accuracy
    on non-brain-MRI categories will be near random.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.tool_model_executor import run_model

CLIP_MODELS  = {"biomedclip", "conch", "musk"}
VLM_MODELS   = {"medgemma", "chexagent", "llava_med"}
OTHER_MODELS = {"vit_alzheimer"}
ALL_MODELS   = CLIP_MODELS | VLM_MODELS | OTHER_MODELS


def predict(model_name: str, sample: dict) -> tuple[str, str]:
    """
    Run *model_name* on the PathMMU sample.

    Returns
    -------
    response : str
        Raw string returned / emitted by the model.
    pred_ans : str
        Parsed answer text (one of the option strings, e.g. "Storiform pattern").
    """
    from eval.parse import parse_prediction   # local import to avoid circular

    image_path  = sample["img_path"]
    index2ans   = sample["index2ans"]   # {'A': 'label text', ...}
    all_choices = sample["all_choices"] # ['A', 'B', 'C', ...]

    # ------------------------------------------------------------------
    # CLIP-style models — pass options as comma-separated candidate labels
    # ------------------------------------------------------------------
    if model_name in CLIP_MODELS:
        labels = [index2ans[c] for c in all_choices]
        prompt = ", ".join(labels)

        result = run_model(model_name, image_path, prompt)
        if "error" in result:
            return result["error"], _random_ans(index2ans)

        response = result.get("top1", "")
        pred_ans = parse_prediction(response, all_choices, index2ans)
        return response, pred_ans

    # ------------------------------------------------------------------
    # VLM models — pass full formatted prompt, parse letter response
    # ------------------------------------------------------------------
    if model_name in VLM_MODELS:
        prompt = sample["final_input_prompt"]

        result = run_model(model_name, image_path, prompt)
        if "error" in result:
            return result["error"], _random_ans(index2ans)

        response = result.get("prediction", result.get("response", ""))
        pred_ans = parse_prediction(response, all_choices, index2ans)
        return response, pred_ans

    # ------------------------------------------------------------------
    # ViT-Alzheimer — domain-specific; map output to closest option text
    # ------------------------------------------------------------------
    if model_name == "vit_alzheimer":
        result = run_model(model_name, image_path, "")
        if "error" in result:
            return result["error"], _random_ans(index2ans)

        response = result.get("prediction", "")
        pred_ans = parse_prediction(response, all_choices, index2ans)
        return response, pred_ans

    raise ValueError(f"Unknown model: '{model_name}'. Available: {sorted(ALL_MODELS)}")


def _random_ans(index2ans: dict) -> str:
    return random.choice(list(index2ans.values()))
