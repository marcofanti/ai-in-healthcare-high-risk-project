#!/usr/bin/env python
"""
Subprocess entry-point for LLaVA-Med inference.

Called by the MCP server using .venv-llava/bin/python.

Usage:
    .venv-llava/bin/python utils/run_llava_med.py \
        --image_path /abs/path/to/image.jpg \
        [--prompt "Describe the findings."]

Outputs a single JSON object to stdout:
    {"model": "llava-med-v1.5-mistral-7b", "prediction": "...", "prompt": "...", "image_path": "..."}
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pydicom
import torch
from PIL import Image

MODEL_ID = "microsoft/llava-med-v1.5-mistral-7b"
DEFAULT_PROMPT = "Describe the findings in this medical image."


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

    # LLaVA imports — only available in .venv-llava
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.model.builder import load_pretrained_model

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=MODEL_ID,
        model_base=None,
        model_name="llava-med-v1.5-mistral-7b",
        device=device,
    )
    model.eval()

    pil_img = load_image(image_path)

    # Build conversation prompt
    user_text = DEFAULT_IMAGE_TOKEN + "\n" + args.prompt
    conv = copy.deepcopy(conv_templates["mistral_instruct"])
    conv.messages = list(conv.messages)
    conv.append_message(conv.roles[0], user_text)
    conv.append_message(conv.roles[1], None)
    prompt_text = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)

    image_tensor = process_images([pil_img], image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = [t.to(device, dtype=torch.float16) for t in image_tensor]
    else:
        image_tensor = image_tensor.to(device, dtype=torch.float16)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            max_new_tokens=256,
            do_sample=False,
            use_cache=True,
        )

    generated = output_ids[0][input_ids.shape[-1]:]
    prediction = tokenizer.decode(generated, skip_special_tokens=True).strip()

    result = {
        "model": MODEL_ID,
        "image_path": str(image_path),
        "prompt": args.prompt,
        "prediction": prediction,
    }
    json.dump(result, sys.stdout, indent=2)


if __name__ == "__main__":
    main()
