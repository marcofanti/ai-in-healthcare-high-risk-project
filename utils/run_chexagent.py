#!/usr/bin/env python
"""
Subprocess entry-point for CheXagent-2-3b inference.

Called by the MCP server using .venv-chexagent/bin/python.

Usage:
    .venv-chexagent/bin/python utils/run_chexagent.py \
        --image_path /abs/path/to/image.jpg \
        [--prompt "Describe the findings."]

Outputs a single JSON object to stdout:
    {"model": "CheXagent-2-3b", "prediction": "...", "prompt": "...", "image_path": "..."}
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pydicom
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "StanfordAIMI/CheXagent-2-3b"
DEFAULT_PROMPT = "Given the image, identify any diseases and describe the key findings."


def load_image(path: Path) -> Image.Image:
    suffix = path.suffix.lower()
    if suffix == ".dcm":
        ds = pydicom.dcmread(str(path))
        raw = ds.pixel_array.astype(float)
        slope = float(getattr(ds, "RescaleSlope", 1))
        intercept = float(getattr(ds, "RescaleIntercept", 0))
        hu = raw * slope + intercept
        wl, ww = 40.0, 400.0
        lo, hi = wl - ww / 2, wl + ww / 2
        windowed = np.clip((hu - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(windowed).convert("RGB")
    return Image.open(path).convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        json.dump({"error": f"File not found: {image_path}"}, sys.stdout)
        sys.exit(1)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = model.to(dtype).to(device)
    model.eval()

    pil_img = load_image(image_path)

    # CheXagent requires a file path, not a PIL Image — save to temp file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        pil_img.save(tmp.name, format="JPEG")
        tmp_path = tmp.name

    query = tokenizer.from_list_format([
        {"image": tmp_path},
        {"text": args.prompt},
    ])
    conv = [
        {"from": "system", "value": "You are a helpful assistant."},
        {"from": "human", "value": query},
    ]
    input_ids = tokenizer.apply_chat_template(
        conv, add_generation_prompt=True, return_tensors="pt"
    ).to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=256,
            do_sample=False,
        )

    generated = output_ids[0][input_ids.shape[-1]:]
    prediction = tokenizer.decode(generated, skip_special_tokens=True).strip()

    Path(tmp_path).unlink(missing_ok=True)

    result = {
        "model": MODEL_ID,
        "image_path": str(image_path),
        "prompt": args.prompt,
        "prediction": prediction,
    }
    json.dump(result, sys.stdout, indent=2)


if __name__ == "__main__":
    main()
