import os
from pathlib import Path
from typing import Any, List, Tuple
import numpy as np
import torch
from PIL import Image

HF_TOKEN: str | None = os.environ.get("HF_TOKEN")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Shared image loaders
# ---------------------------------------------------------------------------

def load_image(path: Path) -> Image.Image:
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


def parse_labels(prompt: str, default: List[str]) -> List[str]:
    """Parse comma-separated labels from prompt, or return default if empty."""
    if not prompt or not prompt.strip():
        return default
    return [l.strip() for l in prompt.split(",") if l.strip()]


# ---------------------------------------------------------------------------
# Lazy model caches
# ---------------------------------------------------------------------------
_MODELS: dict[str, Any] = {}


def get_biomedclip():
    if "biomedclip" not in _MODELS:
        import open_clip
        MODEL_HF = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        model, _, preprocess = open_clip.create_model_and_transforms(MODEL_HF)
        tokenizer = open_clip.get_tokenizer(MODEL_HF)
        model = model.to(DEVICE).eval()
        _MODELS["biomedclip"] = (model, preprocess, tokenizer)
    return _MODELS["biomedclip"]


def get_conch():
    if "conch" not in _MODELS:
        from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer
        model, preprocess = create_model_from_pretrained(
            "conch_ViT-B-16", "hf_hub:MahmoodLab/conch", hf_auth_token=HF_TOKEN
        )
        tokenizer = get_tokenizer("conch_ViT-B-16")
        model = model.to(DEVICE).eval()
        _MODELS["conch"] = (model, preprocess, tokenizer)
    return _MODELS["conch"]


def get_musk():
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


def get_medgemma():
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


def get_vit_alzheimer():
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
# Inference Logic
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

def run_biomedclip(image_path: str, prompt: str = "") -> dict:
    model, preprocess, tokenizer = get_biomedclip()
    pil_img = load_image(Path(image_path))
    labels = parse_labels(prompt, _BIOMEDCLIP_DEFAULTS)

    img_tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        img_emb = model.encode_image(img_tensor, normalize=True)

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

def run_conch(image_path: str, prompt: str = "") -> dict:
    model, preprocess, tokenizer = get_conch()
    from conch.open_clip_custom import tokenize as conch_tokenize
    pil_img = load_image(Path(image_path))
    labels = parse_labels(prompt, _CONCH_DEFAULTS)

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

def run_musk(image_path: str, prompt: str = "") -> dict:
    model, tokenizer, transform, dtype, musk_utils = get_musk()
    labels = parse_labels(prompt, _MUSK_DEFAULTS)
    template = "histopathology image of {}"
    pil_img = load_image(Path(image_path))
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

def run_medgemma(image_path: str, prompt: str = "") -> dict:
    model, processor, dtype = get_medgemma()
    pil_img = load_image(Path(image_path))
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

def run_vit_alzheimer(image_path: str, prompt: str = "") -> dict:
    import torch.nn.functional as F
    model, processor, class_names = get_vit_alzheimer()
    pil_img = load_image(Path(image_path))
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
