"""
Authoritative capability catalog for every model available in the ensemble.
Consumed by the query validator so the LLM can judge whether a user's query
can reasonably be answered by the user's current model selection.

Entries are grounded in each model's official model card / publication.
Changes here directly affect validation behavior.
"""

MODEL_CAPABILITIES: dict[str, dict] = {
    "biomedclip": {
        "full_name": "BiomedCLIP (Microsoft) — PubMedBERT + ViT-B",
        "specialty": "General biomedical vision-language foundation model (CLIP-style, pretrained on PMC-15M)",
        "modalities": [
            "radiology (X-ray, CT, MRI)",
            "histopathology",
            "microscopy",
            "dermatology and general biomedical figures",
        ],
        "tasks": [
            "zero-shot image classification against a label set",
            "cross-modal image-to-text and text-to-image retrieval",
            "biomedical visual question answering (with extra head)",
        ],
        "limitations": [
            "Pretrained on scientific figures from PubMed Central — academic distribution, not clinical imaging",
            "Low input resolution (224-336) — underperforms on high-resolution pathology",
            "Not a report generator; produces embeddings / similarity scores, not narratives",
            "English-only; research use, not cleared for clinical deployment",
        ],
    },
    "conch": {
        "full_name": "CONCH (Mahmood Lab, Harvard/MGB) — Nature Medicine 2024",
        "specialty": "Vision-language foundation model for computational pathology (CONtrastive learning from Captions for Histopathology)",
        "modalities": [
            "2D histopathology (H&E)",
            "whole-slide images (WSI)",
            "non-H&E stains (IHC, special stains)",
        ],
        "tasks": [
            "tissue and cancer subtype classification",
            "text-to-image and image-to-text retrieval",
            "image captioning",
            "tumor and tissue segmentation",
            "rare-disease identification",
            "adapts to 30+ clinical pathology tasks",
        ],
        "limitations": [
            "Pathology only — not suitable for radiology (CT, MRI, X-ray)",
            "Research use; not clinically validated for deployment",
            "Gigapixel WSIs require external patch-level orchestration",
        ],
    },
    "musk": {
        "full_name": "MUSK (Li Lab, Stanford) — Nature 2025",
        "specialty": "Multimodal vision-language foundation model for precision oncology in pathology",
        "modalities": [
            "histopathology images (trained on 50M images across 33 cancer types)",
            "paired pathology image-text",
        ],
        "tasks": [
            "cancer prognosis and outcome prediction",
            "molecular biomarker prediction",
            "immunotherapy response prediction",
            "image-to-text retrieval",
            "pathology visual question answering",
        ],
        "limitations": [
            "Trained exclusively on pathology — NOT validated for radiology (CT/MRI/X-ray)",
            "Despite the 'multimodal' name, MUSK does NOT process hyperspectral / spectral data — flag as a weak fit for any HSI query",
            "Clinical utility requires further validation before real-world adoption",
        ],
    },
    "medgemma": {
        "full_name": "MedGemma (Google, Gemma 3-based) — 4B / 27B multimodal",
        "specialty": "General-purpose medical vision-language model with SigLIP encoder trained on de-identified medical data",
        "modalities": [
            "radiology: chest X-ray, CT (incl. 3D volumes), MRI (incl. 3D volumes)",
            "histopathology (incl. WSI multi-patch interpretation)",
            "ophthalmology",
            "dermatology",
        ],
        "tasks": [
            "medical report generation",
            "medical visual question answering",
            "multi-step clinical reasoning",
            "anatomical localization via bounding boxes (on CXR)",
            "longitudinal imaging comparison (current vs. prior scans)",
            "EHR / clinical text interpretation",
        ],
        "limitations": [
            "Generalist — may underperform specialists (CheXagent on CXR, CONCH on pathology, ViT-Alzheimer on brain-MRI AD)",
            "Requires clinician review for any clinical decision",
        ],
    },
    "vit_alzheimer": {
        "full_name": "ViT-Alzheimer (Vision Transformer fine-tuned for AD classification)",
        "specialty": "Brain MRI classifier for Alzheimer's disease and pre-AD stages",
        "modalities": [
            "brain MRI (typically T1-weighted)",
        ],
        "tasks": [
            "Alzheimer's stage classification — commonly CN / MCI / AD; some variants predict 5 stages (CN, EMCI, LMCI, MCI, AD)",
        ],
        "limitations": [
            "Brain MRI ONLY — will not produce meaningful output on CT, X-ray, pathology, or non-brain anatomy",
            "Narrowly scoped to Alzheimer's — cannot detect tumors, strokes, vascular disease, or other pathologies",
            "Classification only — no free-text report, no segmentation, no reasoning",
        ],
    },
    "chexagent": {
        "full_name": "CheXagent (Stanford AIMI) — 8B parameters, instruction-tuned foundation model",
        "specialty": "Foundation model for chest X-ray interpretation (trained on CheXinstruct, 28 public CXR datasets)",
        "modalities": [
            "chest X-ray (CXR)",
        ],
        "tasks": [
            "findings and full radiology report generation",
            "binary and multi-class disease classification",
            "disease identification (pneumonia, pleural effusion, pneumothorax, nodule, etc.)",
            "view classification and view matching",
            "phrase grounding and abnormality localization",
            "object detection (chest tubes, rib fractures, foreign objects)",
            "temporal image classification across time",
            "findings summarization",
        ],
        "limitations": [
            "Chest X-ray domain ONLY — not suitable for brain, abdomen, musculoskeletal, pathology, or 3D CT volumes",
            "Validated on CheXbench; performance outside chest CXR is unverified",
        ],
    },
    "llava_med": {
        "full_name": "LLaVA-Med (Microsoft) — 7B multimodal, LLaVA-derived and instruction-tuned on biomedicine",
        "specialty": "General medical visual instruction-tuned VLM for open-ended biomedical conversation",
        "modalities": [
            "radiology (CXR, CT, MRI — evaluated on VQA-RAD)",
            "pathology (evaluated on PathVQA)",
            "general biomedical images",
        ],
        "tasks": [
            "open-ended medical visual question answering",
            "conversational reasoning over a medical image",
            "descriptive analysis of a single image",
        ],
        "limitations": [
            "Explicitly research-only — NOT for clinical decision making (stated in model card)",
            "Inherits LLaVA's hallucination and bias tendencies",
            "Training data from PubMed Central figures may over-represent extreme / positive cases",
            "Not a classifier; not a segmenter — narrative VQA output only",
        ],
    },
}


def capability_catalog_markdown(model_names: list[str] | None = None) -> str:
    """
    Render the catalog (or a subset) as compact markdown suitable for
    embedding in an LLM prompt. Used by the query validator.
    """
    names = model_names if model_names is not None else list(MODEL_CAPABILITIES.keys())
    lines: list[str] = []
    for name in names:
        entry = MODEL_CAPABILITIES.get(name)
        if not entry:
            lines.append(f"- **{name}**: (no capability entry — treat as unknown)")
            continue
        lines.append(f"- **{name}** — {entry['full_name']}")
        lines.append(f"  - Specialty: {entry['specialty']}")
        lines.append(f"  - Modalities: {', '.join(entry['modalities'])}")
        lines.append(f"  - Tasks: {', '.join(entry['tasks'])}")
        lines.append(f"  - Limitations: {'; '.join(entry['limitations'])}")
    return "\n".join(lines)
