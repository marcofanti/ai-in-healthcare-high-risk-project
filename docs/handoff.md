# MedHuggingGPT — Handoff Document

**Date:** 2026-04-19  
**Branch:** `ui-changes` (not yet merged to `main`)  
**Status:** Week 3 complete, Week 4 pending

---

## 1. Current State

The system is a fully functional Interactive Ensemble agent. All core Week 3 deliverables are complete and working end-to-end on local hardware.

### What is built and working
| Component | File(s) | Status |
|---|---|---|
| LangGraph agent (Reviewer → Executor → Synthesizer) | `agent/graph.py`, `agent/state.py` | ✅ Working |
| 7 medical AI models (in-process + subprocess) | `tools/tool_model_executor.py`, `utils/model_utils.py` | ✅ Working |
| Streamlit UI with dataset config picker | `app.py` | ✅ Working |
| Dataset config management | `datasets_config.json` | ✅ Working |
| Dataset config CLI utility | `manage_datasets.py` | ✅ Working |
| Model registry validation (3-layer) | `agent/graph.py`, `tools/tool_model_executor.py` | ✅ Working |
| Unit + end-to-end test suite | `tests/test_manage_datasets.py`, `tests/test_model_executor.py` | ✅ 17/17 passing |
| PathMMU benchmark eval harness | `eval/main.py`, `eval/adapters.py`, `eval/scripts/` | ✅ Built |

### What is NOT yet built
- Week 4: HSI (`tool_hsi_calibrator`) — `.raw`/`.hdr` ENVI calibration via `spectral`
- Pan-medical stress test (all 4 modalities in one session)
- HITL interrupt at graph level (`interrupt_before=["Executor"]`) — currently gated in UI only

---

## 2. Architecture Summary

```
User (Streamlit) → datasets_config.json → file selection → "Visualize" button
  → Ensemble config (models + prompt) → "Run Ensemble Analysis"
    → LangGraph: Reviewer → Executor → Synthesizer
      → run_model() × N → consensus report (Gemini)
```

**Key design decisions:**
- `datasets_config.json` is the single source of truth for available datasets. `modality` field (not file extension) drives routing.
- LLM never touches raw medical data — pass-by-path only.
- `MemorySaver()` checkpointer is RAM-based (session is lost on browser close — see PRD backlog).
- Model names are validated at 3 layers before execution to prevent runtime crashes.

---

## 3. UI Flow (current, post ui-changes branch)

1. Sidebar: checkbox per dataset → file selectbox populates from config
2. "Visualize" button must be clicked explicitly to load image preview
3. Changing file selection immediately clears the current visualization
4. Section 3 (Ensemble Config) hidden until Visualize is clicked
5. Results shown in two tabs: Synthesized Report / Model Evidence (Audit Trail)

---

## 4. `manage_datasets.py` CLI

```bash
# Add a dataset (LLM identifies modality, fuzzy-matches existing keys)
uv run manage_datasets.py add ./workspace/mock_oasis --n 10

# Add without prompts (CI / automation)
uv run manage_datasets.py add ./workspace/mock_oasis --yes

# Preview without writing
uv run manage_datasets.py add ./workspace/mock_oasis --dry-run

# Remove a dataset entry
uv run manage_datasets.py remove OASIS_MRI --yes
```

LLM identification escalates: path → file listing → sample metadata. Asks user only on low confidence (skipped with `--yes`). Merges into existing entries without duplicating files.

---

## 5. Open PRD Backlog

Items explicitly flagged as PRDs during development — not yet scoped or scheduled.

| # | Title | Description |
|---|---|---|
| PRD-001 | Session Recovery | `MemorySaver()` is RAM-only; browser close loses all state. Needs persistent checkpointer (SQLite or file-based). |
| PRD-002 | Cross-Machine Path Portability | `datasets_config.json` stores absolute paths. Breaks on different machine or moved workspace. |
| PRD-003 | Synthetic Data Labeling | When inference falls back to `time.sleep(3)` + pre-computed mock, the clinical report must clearly label the output as synthetic/mock — not real analysis. |

---

## 6. PathMMU Benchmark Evaluation

Evaluates the MedHuggingGPT ensemble against the PathMMU pathology Q&A benchmark using the same dataset and ground truth. Mirrors the app's runtime flow exactly: Executor runs each model, Synthesizer picks the answer letter, score against ground truth.

**Model selection is modality-aware** — the same `MODALITY_MODEL_MAPPING` the UI uses to pre-select models is applied per category. No `--models` flag = same defaults a user would see in the UI.

| Category | Modality | Default models (top-2) |
|---|---|---|
| `pdtt` PubMed tiny | 2D Histopathology | `conch`, `musk` |
| `clstt` PathCLS tiny | 2D Histopathology | `conch`, `musk` |
| `att` Atlas tiny | 2D Histopathology | `conch`, `musk` |
| `edutt` EduContent tiny | 2D Histopathology | `conch`, `musk` |
| `sptt` SocialPath tiny | Unknown | `biomedclip`, `llava_med` |

```bash
# Modality-aware defaults (mirrors UI) — recommended
bash eval/scripts/ensemble_mac.sh

# CLIP-only ensemble (fastest — no per-model LLM)
bash eval/scripts/clip_ensemble_mac.sh

# All 5 in-process models
bash eval/scripts/full_ensemble_mac.sh

# Quick 20-sample smoke-test
uv run eval/main.py --categories pdtt --n 20

# Explicit model override (same as user changing the UI multiselect)
uv run eval/main.py --models biomedclip medgemma --categories pdtt clstt

# Override PathMMU data location
PATHMMU_DATA=/path/to/PathMMU/data bash eval/scripts/ensemble_mac.sh
```

**Flow per sample:**
1. Load PathMMU image + question + answer options
2. Executor: run each model — CLIP models receive options as comma-separated labels; VLMs receive the full formatted prompt
3. Synthesizer: LLM asked to pick the answer letter (A/B/C/D) given all model outputs
4. Score against PathMMU ground truth

**Output schema** (matches PathMMU — compatible with `print_results.py`):
- `eval/outputs/<exp_name>/<category>/output.json` — per-sample results including per-model outputs
- `eval/outputs/<exp_name>/<category>/result.json` — `{"acc": float, "num_example": int}`
- `eval/outputs/<exp_name>/summary.json` — overall accuracy across all categories

---

## 7. Running the App

```bash
# Install dependencies
uv sync

# Set API key
export GEMINI_API_KEY=your_key_here

# Run Streamlit app
uv run streamlit run app.py

# Run tests (unit only, no LLM required)
uv run python -m pytest tests/ -k "not identify and not cmd_add and not cmd_remove" -v

# Run all tests including end-to-end (requires API key)
uv run python -m pytest tests/ -v
```

---

## 8. Week 4 Remaining Work

| Task | Owner | Notes |
|---|---|---|
| `tool_hsi_calibrator.py` | Data & Modality Engineer | Use `spectral.envi.open`, dark/white reference calibration, output pseudo-RGB PNG |
| Add HSI to `datasets_config.json` | System Integrator | Use `manage_datasets.py add` on HSI workspace |
| Pan-medical stress test | All | Single session with all 4 modalities simultaneously |
| Merge `ui-changes` → `main` | AI/Agent Architect | Review open PRDs before merge |
| Demo video recording | All | Final deliverable |

---

## 9. Key Files Reference

| What | Where |
|---|---|
| App entry point | `app.py` |
| LangGraph graph | `agent/graph.py` |
| Agent state schema | `agent/state.py` |
| Model dispatcher | `tools/tool_model_executor.py` |
| Dataset config | `datasets_config.json` |
| Dataset CLI | `manage_datasets.py` |
| Test suite | `tests/` |
| PathMMU eval harness | `eval/main.py` |
| PathMMU eval adapters | `eval/adapters.py` |
| PathMMU eval scripts | `eval/scripts/*.sh` |
| Design spec | `docs/superpowers/specs/2026-04-15-medhugginggpt-interactive-ensemble-design.md` |
| Quickstart | `QUICKSTART.md` |
