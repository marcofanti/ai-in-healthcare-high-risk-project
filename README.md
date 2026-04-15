# MedHuggingGPT — Pan-Medical Multimodal AI Agent

**Project Name:** MedHuggingGPT  
**Phase:** Academic Capstone — 4-Week Sprint  
**Architecture:** Local "Pass-by-Path" | LangGraph Orchestration | MCP Tool Server

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Layout](#2-repository-layout)
3. [Datasets](#3-datasets)
4. [Week 1 — Dataset Probe Suite](#4-week-1--dataset-probe-suite)
5. [Week 2 — Medical AI Model Evaluations](#5-week-2--medical-ai-model-evaluations)
6. [MCP Server — Medical AI Tools](#6-mcp-server--medical-ai-tools)
7. [System Architecture](#7-system-architecture)
8. [4-Week Sprint Plan](#8-4-week-sprint-plan)
9. [Team Composition](#9-team-composition)
10. [Risk Management](#10-risk-management)

---

## 1. Project Overview

MedHuggingGPT is a LangGraph-orchestrated AI agent that routes 5 medical imaging modalities to specialized Python tools, runs Human-in-the-Loop (HITL) safety checks, and synthesizes clinical reports in a Streamlit UI.

**Core design principles:**

- **Local "Pass-by-Path":** The LLM never touches raw data. Tools process files and return paths/JSON back to the agent.
- **No custom model training:** All inference uses pre-trained HuggingFace models.
- **Standard loaders only:** `nibabel`, `pydicom`, `spectral`, `SimpleITK` for all I/O.
- **MCP-exposed tools:** All 7 week2 models are available as MCP tools for agentic AI (Claude Code, etc.).

---

## 2. Repository Layout

```
ai-in-healthcare-high-risk-project/
├── README.md                    # This file
├── CLAUDE.md                    # Claude Code session context
├── requirements.txt             # Python dependencies
├── mcp_server.py                # MCP server — 7 medical AI tools
├── mcp_config.json              # Claude Code MCP config block
├── app.py                       # Streamlit UI entry point
├── main.py                      # CLI entry point
│
├── datasets.md                  # Deep dataset analysis (Claude)
├── datasets_gemini.md           # Dataset analysis (Gemini, cross-reference)
├── datasets_info.md             # Supplementary dataset metadata
│
├── week1/                       # Dataset probe suite (10 scripts + notebook)
│   ├── test_iq_oth.py
│   ├── test_oasis1.py
│   ├── test_pkg_hsi.py
│   ├── test_quilt1m_pubmed.py
│   ├── test_quilt1m_quilt.py
│   ├── test_quilt1m_openpath.py
│   ├── test_quilt1m_laion.py
│   ├── test_spinal_dicom.py
│   ├── test_spinal_nifti.py
│   ├── _quilt1m_common.py       # Shared helper for all Quilt1M tests
│   ├── week1.ipynb              # Combined notebook
│   ├── data/                    # Sample data (one per dataset)
│   └── output/                  # JSON reports + PNG visualisations
│
├── week2/                       # Medical AI model evaluation notebooks
│   ├── test_CheXagent-8b.ipynb
│   ├── test_biomedclip.ipynb
│   ├── test_conch.ipynb
│   ├── test_llava-med.ipynb
│   ├── test_medgemma-4b-it.ipynb
│   ├── test_musk.ipynb
│   ├── test_vit-alzheimer.ipynb
│   └── output/                  # Per-model JSON results with timings
│       ├── chexagent_results.json
│       ├── biomedclip_results.json
│       ├── conch_results.json
│       ├── musk_results.json
│       ├── medgemma_results.json
│       ├── llava_med_results.json
│       └── vit_alzheimer_results.json
│
├── agent/                       # LangGraph agent (in progress)
├── tools/                       # Modality tool functions (in progress)
├── utils/
│   ├── manifest_generator.py
│   ├── run_chexagent.py         # Subprocess entry-point for MCP (isolated venv)
│   └── run_llava_med.py         # Subprocess entry-point for MCP (isolated venv)
│
├── .venv/                       # Main Python venv (transformers 5.x)
├── .venv-chexagent/             # Isolated venv for CheXagent (transformers==4.40.0)
├── .venv-llava/                 # Isolated venv for LLaVA-Med (custom llava package)
└── workspace/                   # Local staging for agent sessions
```

---

## 3. Datasets

All datasets live at:
`/Users/mfanti/Documents/Masters_UniversityOfTexas_Austin/AI_for_HealthCare/FinalProject/Datasets/`

| # | Folder | Modality | Domain | Format | Size |
|---|--------|----------|--------|--------|------|
| 1 | `IQ-OTH:NCCD - Lung Cancer Dataset` | CT (Chest) | Radiology | JPEG/PNG | ~1,097 images |
| 2 | `Oasis1` | MRI (Brain, T1) | Radiology | Analyze 7.5 `.hdr/.img` | ~18 GB |
| 3 | `PKG - HistologyHSI-GB` | Hyperspectral microscopy | Pathology | ENVI BIL | ~582 GB |
| 4 | `Quilt1M` | Optical microscopy (H&E) | Pathology | JPEG/PNG | ~721K files |
| 5 | `Spinal` | Spectral CT (Spine) | Radiology | DICOM + NIfTI | ~304 GB |

Full analysis in [datasets.md](datasets.md).

---

## 4. Week 1 — Dataset Probe Suite

**Objective:** Develop standalone Python scripts that read each dataset format and output JSON summaries + PNG visualisations, completely independent of any LLM or UI.

### 10 Test Programs

| File | Dataset | Key libraries |
|------|---------|---------------|
| `test_iq_oth.py` | IQ-OTH/NCCD Lung Cancer | `SimpleImageWrapper`, `PIL`, numpy stats, matplotlib 3-panel |
| `test_oasis1.py` | OASIS-1 Brain MRI | `nibabel` (Analyze 7.5), `NiftiImageWrapper`, FSL tissue volumes |
| `test_pkg_hsi.py` | PKG HistologyHSI-GB | `spectral.io.envi`, white/dark calibration, spectral signatures |
| `test_quilt1m_pubmed.py` | Quilt1M — PubMed | `SimpleImageWrapper`, `pandas` CSV, UMLS entities |
| `test_quilt1m_quilt.py` | Quilt1M — YouTube | Same |
| `test_quilt1m_openpath.py` | Quilt1M — Twitter/OpenPath | Same |
| `test_quilt1m_laion.py` | Quilt1M — LAION | Same |
| `test_spinal_dicom.py` | Spinal CT — DICOM | `pydicom`, `DicomImageWrapper`, `SimpleITK` 3D series, HU conversion |
| `test_spinal_nifti.py` | Spinal CT — NIfTI masks | `nibabel`, per-vertebra volume (mm³/cm³), co-registration check |

Each program outputs a JSON report and PNG visualisation to `week1/output/`.

### Running the Week 1 Suite

```bash
cd week1/
uv run python test_iq_oth.py
uv run python test_oasis1.py
uv run python test_pkg_hsi.py        # ~30-60s — loads 1.2 GB HSI cube
uv run python test_quilt1m_pubmed.py
uv run python test_quilt1m_quilt.py
uv run python test_quilt1m_openpath.py
uv run python test_quilt1m_laion.py
uv run python test_spinal_dicom.py
uv run python test_spinal_nifti.py

# Or open the combined notebook:
jupyter notebook week1.ipynb
```

---

## 5. Week 2 — Medical AI Model Evaluations

**Objective:** Test 7 pre-trained medical AI models against all 5 week1 datasets. Each notebook records per-sample inference times, model load time, and saves results to a JSON checkpoint.

### Models Evaluated

| Notebook | Model | Type | HF Token | Venv |
|----------|-------|------|----------|------|
| `test_CheXagent-8b.ipynb` | `StanfordAIMI/CheXagent-2-3b` | Generative VLM (chest X-ray) | No | `.venv-chexagent` |
| `test_biomedclip.ipynb` | `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` | CLIP zero-shot | No | `.venv` |
| `test_conch.ipynb` | `MahmoodLab/conch` | CLIP zero-shot (histopathology) | Yes (gated) | `.venv` |
| `test_llava-med.ipynb` | `microsoft/llava-med-v1.5-mistral-7b` | Generative VLM (general medical) | No | `.venv-llava` |
| `test_medgemma-4b-it.ipynb` | `google/medgemma-4b-it` | Generative VLM (clinical) | Yes (gated) | `.venv` |
| `test_musk.ipynb` | `xiangjx/musk` | CLIP zero-shot (pathology, Nature 2025) | No | `.venv` |
| `test_vit-alzheimer.ipynb` | `dhritic99/vit-base-brain-alzheimer-detection` | Image classifier (4-class brain MRI) | No | `.venv` |

### ViT-Alzheimer — Multi-Slice + OOD Evaluation

The ViT notebook runs two evaluation modes:

**In-domain (OASIS-1):** Samples every 15th axial slice across the full MRI volume, classifying each into one of 4 stages. Plots per-class probability trajectory across slice positions.

**Out-of-domain (4 other datasets):** Tests calibration on chest CT, spinal CT, H&E histopathology, and hyperspectral images. A well-calibrated model should produce near-uniform distributions (high entropy ≈ `ln(4) ≈ 1.386`) on OOD inputs.

Classes:

| ID | Class | Clinical meaning |
|----|-------|-----------------|
| 0 | `MildDemented` | Noticeable memory loss |
| 1 | `ModerateDemented` | Needs daily assistance |
| 2 | `NonDemented` | No detectable impairment |
| 3 | `VeryMildDemented` | Earliest detectable stage |

### Timing format (all notebooks)

Every result JSON contains:

```json
{
  "model_load_s": 1.15,
  "total_inference_s": 4.32,
  "results": [
    {
      "name": "...",
      "dataset": "...",
      "prediction": "...",
      "time_s": 0.21,
      "model_load_s": 1.15
    }
  ]
}
```

### Isolated venvs

CheXagent and LLaVA-Med require conflicting versions of `transformers`:

| Venv | Python | Key constraint |
|------|--------|----------------|
| `.venv` | 3.11 | `transformers>=5.0` (main) |
| `.venv-chexagent` | 3.11 | `transformers==4.40.0` (hardcoded assertion in CheXagent model files) |
| `.venv-llava` | 3.11 | Custom `llava` package from microsoft/LLaVA-Med fork |

### Running Week 2 Notebooks

Open each notebook in Jupyter using its designated kernel:

```bash
# main venv notebooks (ai-healthcare kernel)
jupyter notebook week2/test_biomedclip.ipynb
jupyter notebook week2/test_conch.ipynb
jupyter notebook week2/test_musk.ipynb
jupyter notebook week2/test_medgemma-4b-it.ipynb
jupyter notebook week2/test_vit-alzheimer.ipynb

# isolated venv notebooks
jupyter notebook week2/test_CheXagent-8b.ipynb   # kernel: chexagent
jupyter notebook week2/test_llava-med.ipynb       # kernel: llava-med
```

---

## 6. MCP Server — Medical AI Tools

All 7 week2 models are exposed as [MCP (Model Context Protocol)](https://modelcontextprotocol.io) tools, allowing Claude Code and other agentic AI systems to invoke them on arbitrary images.

### Architecture

```
mcp_server.py
├── In-process (main venv .venv/bin/python)
│   ├── biomedclip   — lazy model load on first call
│   ├── conch        — lazy model load on first call
│   ├── musk         — lazy model load on first call
│   ├── medgemma     — lazy model load on first call
│   └── vit_alzheimer— lazy model load on first call
│
└── Subprocess (isolated venvs)
    ├── chexagent  → .venv-chexagent/bin/python utils/run_chexagent.py
    └── llava_med  → .venv-llava/bin/python utils/run_llava_med.py
```

Models load lazily (on first tool call) and are cached in-process for subsequent calls. CheXagent and LLaVA-Med are invoked via subprocess to respect their isolated venv requirements — no separate MCP server needed.

### Tool Reference

| Tool | Model | Input `prompt` | Returns |
|------|-------|----------------|---------|
| `biomedclip` | Microsoft BiomedCLIP | comma-sep candidate labels (or empty for defaults) | top-5 labels + softmax probs |
| `conch` | MahmoodLab CONCH | comma-sep candidate labels | top-5 labels + softmax probs |
| `musk` | xiangjx/MUSK | comma-sep candidate labels | top-5 labels + softmax probs |
| `medgemma` | google/medgemma-4b-it | clinical question (text) | generated answer string |
| `vit_alzheimer` | dhritic99/vit-base | ignored | predicted class + all 4 probs + entropy |
| `chexagent` | StanfordAIMI/CheXagent-2-3b | clinical question (text) | generated findings string |
| `llava_med` | microsoft/llava-med-v1.5 | clinical question (text) | generated description string |

All tools accept an `image_path` (absolute path) in any of these formats:

| Extension | Format | Auto-handling |
|-----------|--------|---------------|
| `.jpg`, `.png`, etc. | Standard image | Direct PIL load |
| `.dcm` | DICOM | HU conversion + soft-tissue windowing |
| `.nii`, `.nii.gz`, `.img` | NIfTI / Analyze 7.5 | Middle axial slice extracted |
| `.hdr` | ENVI hyperspectral | Pseudo-RGB via 650/550/450 nm bands |

### Startup

```bash
# stdio transport (for Claude Code — default)
.venv/bin/python mcp_server.py

# SSE / HTTP transport (port 8000)
.venv/bin/python mcp_server.py --transport sse

# Custom port
.venv/bin/python mcp_server.py --transport sse --port 9000
```

### Claude Code Integration

Add to your project's `.mcp.json` or merge [mcp_config.json](mcp_config.json) into `~/.claude.json`:

```json
{
  "mcpServers": {
    "medhugging": {
      "type": "stdio",
      "command": "/path/to/.venv/bin/python",
      "args": ["/path/to/mcp_server.py"],
      "env": {
        "HF_TOKEN": "${HF_TOKEN}"
      }
    }
  }
}
```

Or via the CLI:

```bash
claude mcp add medhugging \
  /path/to/.venv/bin/python \
  /path/to/mcp_server.py
```

Once connected, you can ask Claude:

> "Use biomedclip to classify `/path/to/image.jpg` for chest pathology"  
> "Run vit_alzheimer on this brain MRI slice: `/path/to/slice.nii`"  
> "Ask medgemma to describe the findings in `/path/to/xray.dcm`"

### Environment Variables

| Variable | Required by | Purpose |
|----------|-------------|---------|
| `HF_TOKEN` | CONCH, MedGemma | HuggingFace token (gated models — accept terms at hf.co) |

### Example Tool Responses

**CLIP-style** (`biomedclip`, `conch`, `musk`):
```json
{
  "model": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
  "image_path": "/data/xray.jpg",
  "top1": "chest CT with lung nodule",
  "top1_prob": 0.6821,
  "top5": [
    {"label": "chest CT with lung nodule", "prob": 0.6821},
    {"label": "normal lung parenchyma",    "prob": 0.1403},
    ...
  ]
}
```

**Generative** (`medgemma`, `chexagent`, `llava_med`):
```json
{
  "model": "google/medgemma-4b-it",
  "image_path": "/data/xray.jpg",
  "prompt": "Describe the findings in this medical image.",
  "prediction": "The chest X-ray demonstrates..."
}
```

**Classifier** (`vit_alzheimer`):
```json
{
  "model": "dhritic99/vit-base-brain-alzheimer-detection",
  "image_path": "/data/mri_slice.jpg",
  "prediction": "NonDemented",
  "confidence": 0.8912,
  "entropy": 0.3201,
  "entropy_pct_max": 23.1,
  "probabilities": {
    "MildDemented": 0.0421,
    "ModerateDemented": 0.0312,
    "NonDemented": 0.8912,
    "VeryMildDemented": 0.0355
  }
}
```

---

## 7. System Architecture

### 7.1 Local Storage & Manifest Extraction

- **Local Staging Workspace:** All data ingestion and processing in `./workspace/session_*/`. No cloud storage.
- **Manifest Generator:** Background script generates a lightweight JSON manifest from the workspace directory. The LLM plans from metadata only.
- **Pass-by-Path Execution:** Tools receive absolute file paths, process data, and return output paths + JSON.

### 7.2 LangGraph Orchestration

Four primary states:

1. **Planner Node:** Ingests JSON manifest + user query → sequential execution plan
2. **HITL Gateway:** `interrupt_before` pause for human approval. State via `MemorySaver()` (RAM)
3. **Executor Node:** Iterates the approved plan, calls modality tools
4. **Synthesizer Node:** Aggregates tool outputs → cohesive clinical report

### 7.3 UI

**Streamlit** — native Python, local file system access, fast state rendering.

### 7.4 Key Technical Notes

**Analyze 7.5 vs NIfTI:** OASIS-1 uses `.hdr/.img` pairs. `nibabel.load()` handles both formats transparently.

**DICOM — single slice vs 3D volume:** `DicomImageWrapper` reads one `.dcm` at a time. For 3D reconstruction, use `SimpleITK.ImageSeriesReader`.

**HSI calibration:** Always apply `reflectance = (raw − dark) / (white − dark)` before band statistics.

**HU conversion:** `HU = pixel_value × RescaleSlope + RescaleIntercept`. Typical ranges: air ≈ −1000, fat ≈ −100, water = 0, soft tissue ≈ 40, bone ≈ 400+.

**transformers version conflicts:** CheXagent requires `transformers==4.40.0` (hardcoded assertion). LLaVA-Med requires a forked `llava` package. Both run in isolated venvs; the MCP server delegates to them via subprocess.

---

## 8. 4-Week Sprint Plan

### Week 1 ✅ — Dataset Probe Suite

**Completed:** 10 standalone test scripts covering all 5 datasets. Each produces a JSON report and PNG visualisation. Combined `week1.ipynb` notebook. Sample data copied to `week1/data/`.

### Week 2 ✅ — Medical AI Model Evaluations

**Completed:** 7 model evaluation notebooks with full timing instrumentation, all 5 datasets, and MCP server exposing all models as tools.

### Week 3 — LangGraph Integration

- **Objective:** Connect model tools to the LangGraph agent.
- **Deliverables:**
  - `tools/` — modality tool functions wrapping week2 model APIs
  - LangGraph `StateGraph` with Planner → HITL → Executor → Synthesizer
  - Remove mock data; connect Streamlit UI to live backend
  - `MemorySaver()` checkpointer for HITL state persistence

### Week 4 — Pan-Medical Validation & Demo

- **Objective:** Stress-test the full pipeline, freeze codebase, record demo.
- **Deliverables:**
  - Mixed workspace test: all 5 modalities in one session
  - Synthesizer node → cohesive clinical report in Streamlit
  - Final documentation and demo video

---

## 9. Team Composition

| Member | Role | Responsibilities |
|--------|------|-----------------|
| 1 | AI / Agent Architect | LangGraph state design, node/edge wiring, system prompts, JSON tool schema |
| 2 | Data & Modality Engineer | Dataset loaders, modality tool functions, model evaluation notebooks |
| 3 | System Integrator & UI | Streamlit app, manifest extractor, HITL workflow, report rendering |

---

## 10. Risk Management

| Risk | Impact | Mitigation |
|------|--------|------------|
| Hardware / compute bottlenecks | High | Academic goal is routing architecture, not accuracy. Use pre-computed results if inference > 60s. |
| Scope creep in pre-processing | High | Strict "off-the-shelf" policy — only established library boilerplate. |
| transformers version conflicts | Medium | Isolated venvs per model; MCP subprocess delegation pattern. |
| Gated HuggingFace models | Low | `HF_TOKEN` env var; fallback to non-gated alternatives where possible. |
| Large dataset I/O | Medium | Pass-by-path architecture; samples only (not full datasets) in `week1/data/`. |
