#!/usr/bin/env python
"""
MedHuggingGPT MCP Server

Exposes 7 week2 medical AI models as MCP tools.

In-process (main venv):
  biomedclip   — zero-shot contrastive (any modality)
  conch        — zero-shot contrastive (histopathology)
  musk         — zero-shot contrastive (pathology)
  medgemma     — generative VLM (radiology + clinical Q&A)
  vit_alzheimer— image classifier (brain MRI → Alzheimer stage)

Subprocess (isolated venvs):
  chexagent    — generative VLM (chest X-ray/CT, .venv-chexagent)
  llava_med    — generative VLM (general medical, .venv-llava)

Usage:
    # stdio (for Claude Code / MCP config)
    python mcp_server.py

    # HTTP + SSE  (port 8000 by default)
    python mcp_server.py --transport sse

    # Custom port
    python mcp_server.py --transport sse --port 9000
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mcp.server.fastmcp import FastMCP
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
VENV_CHEXAGENT = PROJECT_ROOT / ".venv-chexagent" / "bin" / "python"
VENV_LLAVA = PROJECT_ROOT / ".venv-llava" / "bin" / "python"
SCRIPT_CHEXAGENT = PROJECT_ROOT / "utils" / "run_chexagent.py"
SCRIPT_LLAVA = PROJECT_ROOT / "utils" / "run_llava_med.py"

HF_TOKEN: str | None = os.environ.get("HF_TOKEN")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ---------------------------------------------------------------------------
# FastMCP instance
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "MedHuggingGPT",
    instructions=(
        "Medical AI inference tools. Each tool accepts an image_path (absolute path to any "
        "medical image: JPEG, PNG, DICOM .dcm, NIfTI .nii/.img, or ENVI .hdr) and an optional "
        "prompt. CLIP-style tools (biomedclip, conch, musk) interpret prompt as a "
        "comma-separated list of candidate labels for zero-shot classification. Generative tools "
        "(medgemma, chexagent, llava_med) interpret prompt as a text question. The classifier "
        "tool (vit_alzheimer) ignores prompt."
    ),
)

# ---------------------------------------------------------------------------
# Shared image loaders
# ---------------------------------------------------------------------------

def _load_image(path: Path) -> Image.Image:
    """Load any supported medical image to a PIL RGB image."""
    suffix = path.suffix.lower()

    if suffix == ".dcm":
        import pydicom
        ds = pydicom.dcmread(str(path))
        raw = ds.pixel_array.astype(float)
        slope = float(getattr(ds, "RescaleSlope", 1))
        intercept = float(getattr(ds, "RescaleIntercept", 0))
        hu = raw * slope + intercept
        wl, ww = 40.0, 400.0
        lo, hi = wl - ww / 2, wl + ww / 2
        windowed = np.clip((hu - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(windowed).convert("RGB")

    if suffix in (".nii", ".gz", ".img", ".hdr") and suffix != ".hdr":
        import nibabel as nib
        vol = np.squeeze(nib.load(str(path)).get_fdata())
        if vol.ndim == 3:
            z = vol.shape[2] // 2
            slc = vol[:, :, z]
        else:
            slc = vol
        lo, hi = slc.min(), slc.max()
        arr = ((slc - lo) / (hi - lo + 1e-9) * 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")

    if suffix == ".hdr":
        import spectral.io.envi as envi
        img = envi.open(str(path))
        wl_vals = np.array([float(w) for w in img.metadata["wavelength"]])
        br = int(np.argmin(np.abs(wl_vals - 650)))
        bg = int(np.argmin(np.abs(wl_vals - 550)))
        bb = int(np.argmin(np.abs(wl_vals - 450)))

        def _norm(band_idx: int) -> np.ndarray:
            arr = np.squeeze(img.read_band(band_idx))
            lo, hi = arr.min(), arr.max()
            return ((arr - lo) / (hi - lo + 1e-9) * 255).astype(np.uint8)

        return Image.fromarray(np.stack([_norm(br), _norm(bg), _norm(bb)], axis=2))

    # Default: standard image formats
    return Image.open(path).convert("RGB")


def _parse_labels(prompt: str, default: list[str]) -> list[str]:
    """Parse comma-separated labels from prompt, or return default if empty."""
    if not prompt or not prompt.strip():
        return default
    return [l.strip() for l in prompt.split(",") if l.strip()]


# ---------------------------------------------------------------------------
# Lazy model caches
# ---------------------------------------------------------------------------
_MODELS: dict[str, Any] = {}


def _get_biomedclip():
    if "biomedclip" not in _MODELS:
        import open_clip
        MODEL_HF = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        model, _, preprocess = open_clip.create_model_and_transforms(MODEL_HF)
        tokenizer = open_clip.get_tokenizer(MODEL_HF)
        model = model.to(DEVICE).eval()
        _MODELS["biomedclip"] = (model, preprocess, tokenizer)
    return _MODELS["biomedclip"]


def _get_conch():
    if "conch" not in _MODELS:
        from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer
        model, preprocess = create_model_from_pretrained(
            "conch_ViT-B-16", "hf_hub:MahmoodLab/conch", hf_auth_token=HF_TOKEN
        )
        tokenizer = get_tokenizer("conch_ViT-B-16")
        model = model.to(DEVICE).eval()
        _MODELS["conch"] = (model, preprocess, tokenizer)
    return _MODELS["conch"]


def _get_musk():
    if "musk" not in _MODELS:
        import sentencepiece as spm
        import torchvision.transforms as T
        from huggingface_hub import hf_hub_download
        from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
        from timm.models import create_model

        import musk.modeling  # noqa: F401 — registers timm model
        from musk import utils as musk_utils

        model = create_model("musk_large_patch16_384")
        musk_utils.load_model_and_may_interpolate(
            ckpt_path="hf_hub:xiangjx/musk",
            model=model,
            model_key="model|module",
            model_prefix="",
        )
        dtype = torch.float32
        model = model.to(device=DEVICE, dtype=dtype).eval()

        tokenizer_path = hf_hub_download(
            repo_id="xiangjx/musk", filename="tokenizer.spm", token=HF_TOKEN
        )

        class _SPTokenizer:
            def __init__(self, spm_path: str) -> None:
                self._sp = spm.SentencePieceProcessor()
                self._sp.Load(spm_path)
                self.bos_token_id = self._sp.bos_id()
                self.eos_token_id = self._sp.eos_id()
                self.pad_token_id = 1

            def encode(self, text: str) -> list[int]:
                ids = self._sp.encode_as_ids(text)
                return [self.bos_token_id] + ids + [self.eos_token_id]

        tokenizer = _SPTokenizer(tokenizer_path)
        transform = T.Compose([
            T.Resize(384, interpolation=3, antialias=True),
            T.CenterCrop((384, 384)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
        ])
        _MODELS["musk"] = (model, tokenizer, transform, dtype, musk_utils)
    return _MODELS["musk"]


def _get_medgemma():
    if "medgemma" not in _MODELS:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        MODEL_ID = "google/medgemma-4b-it"
        dtype = torch.bfloat16
        processor = AutoProcessor.from_pretrained(MODEL_ID, token=HF_TOKEN)
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, torch_dtype=dtype, token=HF_TOKEN
        )
        model = model.to(DEVICE).eval()
        _MODELS["medgemma"] = (model, processor, dtype)
    return _MODELS["medgemma"]


def _get_vit_alzheimer():
    if "vit_alzheimer" not in _MODELS:
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        MODEL_ID = "dhritic99/vit-base-brain-alzheimer-detection"
        processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
        model = model.to(DEVICE).eval()
        CLASS_NAMES = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
        _MODELS["vit_alzheimer"] = (model, processor, CLASS_NAMES)
    return _MODELS["vit_alzheimer"]


# ---------------------------------------------------------------------------
# MCP Tools — CLIP-style (zero-shot classification)
# ---------------------------------------------------------------------------

_BIOMEDCLIP_DEFAULTS = [
    "chest X-ray showing normal lungs",
    "chest CT with lung nodule",
    "brain MRI with white matter lesion",
    "histopathology slide showing tumor cells",
    "normal spinal CT",
]

_CONCH_DEFAULTS = [
    "adenocarcinoma",
    "squamous cell carcinoma",
    "normal tissue",
    "inflammatory infiltrate",
    "necrosis",
]

_MUSK_DEFAULTS = [
    "glioblastoma multiforme",
    "normal brain tissue",
    "low-grade glioma",
    "meningioma",
    "adenocarcinoma",
]


@mcp.tool()
def biomedclip(image_path: str, prompt: str = "") -> dict:
    """
    Zero-shot classification using Microsoft BiomedCLIP
    (trained on 15M PubMed image-caption pairs, all medical modalities).

    Args:
        image_path: Absolute path to the image (JPEG, PNG, DICOM, NIfTI, ENVI .hdr).
        prompt: Comma-separated candidate labels. Leave empty for default labels.

    Returns:
        Top-5 labels with softmax probabilities.
    """
    path = Path(image_path)
    if not path.exists():
        return {"error": f"File not found: {image_path}"}

    model, preprocess, tokenizer = _get_biomedclip()
    import torch.nn.functional as F

    pil_img = _load_image(path)
    labels = _parse_labels(prompt, _BIOMEDCLIP_DEFAULTS)

    img_tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        img_emb = model.encode_image(img_tensor, normalize=True)

    import open_clip
    text_tokens = tokenizer(labels).to(DEVICE)
    with torch.inference_mode():
        txt_embs = model.encode_text(text_tokens, normalize=True)

    probs = (100.0 * img_emb @ txt_embs.T).softmax(dim=-1)[0].cpu().float().numpy()
    top5 = sorted(zip(labels, probs.tolist()), key=lambda x: x[1], reverse=True)[:5]

    return {
        "model": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "image_path": image_path,
        "top1": top5[0][0],
        "top1_prob": round(top5[0][1], 4),
        "top5": [{"label": l, "prob": round(p, 4)} for l, p in top5],
    }


@mcp.tool()
def conch(image_path: str, prompt: str = "") -> dict:
    """
    Zero-shot classification using MahmoodLab CONCH
    (histopathology VLM trained on 1.17M H&E image-caption pairs).
    Requires HF_TOKEN env var with accepted MahmoodLab terms.

    Args:
        image_path: Absolute path to the image.
        prompt: Comma-separated candidate pathology labels. Leave empty for defaults.

    Returns:
        Top-5 labels with softmax probabilities.
    """
    path = Path(image_path)
    if not path.exists():
        return {"error": f"File not found: {image_path}"}

    model, preprocess, tokenizer = _get_conch()
    from conch.open_clip_custom import tokenize as conch_tokenize

    pil_img = _load_image(path)
    labels = _parse_labels(prompt, _CONCH_DEFAULTS)

    img_tensor = preprocess(pil_img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        img_emb = model.encode_image(img_tensor, normalize=True)

    template = "an H&E stained histopathology image of {}"
    text_tokens = conch_tokenize(tokenizer, [template.format(l) for l in labels]).to(DEVICE)
    with torch.inference_mode():
        txt_embs = model.encode_text(text_tokens, normalize=True)

    probs = (100.0 * img_emb @ txt_embs.T).softmax(dim=-1)[0].cpu().float().numpy()
    top5 = sorted(zip(labels, probs.tolist()), key=lambda x: x[1], reverse=True)[:5]

    return {
        "model": "MahmoodLab/conch",
        "image_path": image_path,
        "top1": top5[0][0],
        "top1_prob": round(top5[0][1], 4),
        "top5": [{"label": l, "prob": round(p, 4)} for l, p in top5],
    }


@mcp.tool()
def musk(image_path: str, prompt: str = "") -> dict:
    """
    Zero-shot classification using MUSK (Nature 2025)
    (ViT-Large pathology foundation model trained on paired WSI + pathology reports).

    Args:
        image_path: Absolute path to the image.
        prompt: Comma-separated candidate pathology labels. Leave empty for defaults.

    Returns:
        Top-5 labels with softmax probabilities.
    """
    path = Path(image_path)
    if not path.exists():
        return {"error": f"File not found: {image_path}"}

    model, tokenizer, transform, dtype, musk_utils = _get_musk()
    labels = _parse_labels(prompt, _MUSK_DEFAULTS)
    template = "histopathology image of {}"

    pil_img = _load_image(path)
    img_tensor = transform(pil_img.convert("RGB")).unsqueeze(0).to(DEVICE, dtype=dtype)
    with torch.inference_mode():
        img_emb = model(
            image=img_tensor, with_head=True, out_norm=True, ms_aug=False, return_global=True
        )[0]

    prompts = [template.format(l) for l in labels]
    all_ids, all_pads = [], []
    for t in prompts:
        ids, pad = musk_utils.xlm_tokenizer(t, tokenizer, max_len=100)
        all_ids.append(ids)
        all_pads.append(pad)
    txt_ids = torch.tensor(all_ids, dtype=torch.long).to(DEVICE)
    pad_mask = torch.tensor(all_pads, dtype=torch.long).to(DEVICE)
    with torch.inference_mode():
        txt_embs = model(
            text_description=txt_ids, padding_mask=pad_mask,
            with_head=True, out_norm=True, ms_aug=False, return_global=True
        )[1]

    probs = (100.0 * img_emb.float() @ txt_embs.float().T).softmax(dim=-1)[0].cpu().numpy()
    top5 = sorted(zip(labels, probs.tolist()), key=lambda x: x[1], reverse=True)[:5]

    return {
        "model": "xiangjx/musk",
        "image_path": image_path,
        "top1": top5[0][0],
        "top1_prob": round(top5[0][1], 4),
        "top5": [{"label": l, "prob": round(p, 4)} for l, p in top5],
    }


# ---------------------------------------------------------------------------
# MCP Tools — Generative VLMs (in-process)
# ---------------------------------------------------------------------------

@mcp.tool()
def medgemma(image_path: str, prompt: str = "") -> dict:
    """
    Visual question answering using Google MedGemma-4b-it
    (4B multimodal clinical model fine-tuned on radiology + pathology).
    Requires HF_TOKEN env var with accepted Google terms.

    Args:
        image_path: Absolute path to the image.
        prompt: Clinical question or instruction. Defaults to a general findings description.

    Returns:
        Generated clinical text response.
    """
    path = Path(image_path)
    if not path.exists():
        return {"error": f"File not found: {image_path}"}

    model, processor, dtype = _get_medgemma()
    pil_img = _load_image(path)
    question = prompt.strip() or "Describe the findings in this medical image."

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": pil_img},
            {"type": "text", "text": question},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(DEVICE)

    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)

    generated = output_ids[0][input_len:]
    prediction = processor.decode(generated, skip_special_tokens=True).strip()

    return {
        "model": "google/medgemma-4b-it",
        "image_path": image_path,
        "prompt": question,
        "prediction": prediction,
    }


# ---------------------------------------------------------------------------
# MCP Tools — Classifier (in-process)
# ---------------------------------------------------------------------------

@mcp.tool()
def vit_alzheimer(image_path: str, prompt: str = "") -> dict:
    """
    Alzheimer's severity classification using ViT-base fine-tuned on brain MRI.
    4 classes: MildDemented, ModerateDemented, NonDemented, VeryMildDemented.
    Designed for axial T1 brain MRI slices; returns entropy as OOD confidence signal.

    Args:
        image_path: Absolute path to the image (any format — NIfTI middle slice auto-extracted).
        prompt: Unused. Pass any string or leave empty.

    Returns:
        Predicted class with all 4 class probabilities and entropy.
    """
    import torch.nn.functional as F

    path = Path(image_path)
    if not path.exists():
        return {"error": f"File not found: {image_path}"}

    model, processor, class_names = _get_vit_alzheimer()
    pil_img = _load_image(path)

    inputs = processor(images=pil_img, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1).squeeze().cpu().float().numpy()
    pred_idx = int(np.argmax(probs))
    entropy = float(-np.sum(probs * np.log(probs + 1e-9)))

    return {
        "model": "dhritic99/vit-base-brain-alzheimer-detection",
        "image_path": image_path,
        "prediction": class_names[pred_idx],
        "confidence": round(float(probs[pred_idx]), 4),
        "entropy": round(entropy, 4),
        "entropy_pct_max": round(entropy / np.log(4) * 100, 1),
        "probabilities": {cls: round(float(p), 4) for cls, p in zip(class_names, probs)},
    }


# ---------------------------------------------------------------------------
# MCP Tools — Subprocess (isolated venvs)
# ---------------------------------------------------------------------------

def _run_subprocess(python: Path, script: Path, image_path: str, prompt: str) -> dict:
    """Invoke an isolated-venv inference script and parse its JSON stdout."""
    if not python.exists():
        return {"error": f"Python interpreter not found: {python}"}
    if not script.exists():
        return {"error": f"Script not found: {script}"}

    cmd = [str(python), str(script), "--image_path", image_path]
    if prompt:
        cmd += ["--prompt", prompt]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(PROJECT_ROOT),
        )
    except subprocess.TimeoutExpired:
        return {"error": "Inference timed out after 300s"}
    except Exception as exc:
        return {"error": str(exc)}

    if result.returncode != 0:
        return {
            "error": f"Subprocess exited with code {result.returncode}",
            "stderr": result.stderr[-2000:] if result.stderr else "",
        }

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {
            "error": "Could not parse subprocess output as JSON",
            "raw_stdout": result.stdout[-2000:],
        }


@mcp.tool()
def chexagent(image_path: str, prompt: str = "") -> dict:
    """
    Chest X-ray / radiology VLM using StanfordAIMI CheXagent-2-3b.
    Runs in an isolated venv (.venv-chexagent) via subprocess.
    First call triggers model loading (~30-60s, 6 GB download on first run).

    Args:
        image_path: Absolute path to the image (JPEG, PNG, or DICOM).
        prompt: Clinical question or task. Defaults to disease identification.

    Returns:
        Generated clinical findings text.
    """
    path = Path(image_path)
    if not path.exists():
        return {"error": f"File not found: {image_path}"}

    return _run_subprocess(VENV_CHEXAGENT, SCRIPT_CHEXAGENT, image_path, prompt)


@mcp.tool()
def llava_med(image_path: str, prompt: str = "") -> dict:
    """
    General medical VLM using Microsoft LLaVA-Med v1.5 Mistral-7B.
    Runs in an isolated venv (.venv-llava) via subprocess.
    First call triggers model loading (~60-120s, large download on first run).

    Args:
        image_path: Absolute path to the image (any standard format).
        prompt: Clinical question or instruction. Defaults to describing findings.

    Returns:
        Generated medical description text.
    """
    path = Path(image_path)
    if not path.exists():
        return {"error": f"File not found: {image_path}"}

    return _run_subprocess(VENV_LLAVA, SCRIPT_LLAVA, image_path, prompt)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MedHuggingGPT MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="MCP transport (default: stdio)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for sse/streamable-http transport (default: 8000)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        # FastMCP reads port from FASTMCP_PORT env var
        os.environ.setdefault("FASTMCP_PORT", str(args.port))
        mcp.run(transport=args.transport)
