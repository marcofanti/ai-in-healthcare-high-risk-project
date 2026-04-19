# Changelog

All notable changes to MedHuggingGPT are documented here, organized by sprint week.

---

## [ui-changes] — 2026-04-19 (current branch, not yet merged)

### Added
- `datasets_config.json` — single source of truth for available datasets; each entry declares `modality` and `files[]` (path + type); `modality` field drives routing, resolving `.hdr` ambiguity between Legacy MRI and HSI datasets
- `manage_datasets.py` — CLI utility (`uv run manage_datasets.py add/remove`) for managing the dataset config; uses 3-step escalating LLM identification (path → file listing → sample metadata); fuzzy key matching against existing entries; `--yes`, `--dry-run` flags
- `tests/test_manage_datasets.py` — 16 unit tests (pure functions) + 5 end-to-end tests with real LLM (skipped without API key); written before implementation (TDD)
- `docs/handoff.md` — current state snapshot for team handoff
- `docs/DECISIONS.md` — architecture decision records (ADRs)
- `docs/CHANGELOG.md` — this file
- `docs/prds/PRD-001-session-recovery.md` — session persistence on browser close
- `docs/prds/PRD-002-path-portability.md` — cross-machine path portability
- `docs/prds/PRD-003-synthetic-data-labeling.md` — labeling mock/pre-computed outputs in reports

### Changed
- `app.py` — replaced sidebar directory scan with `datasets_config.json`-driven dataset checkboxes + file selectbox; added "Visualize" button gating image preview and Section 3 (Ensemble Config); modality comes from config, not extension heuristics
- `agent/graph.py` — `reviewer_node` now validates all model names against `SUPPORTED_MODELS` before building the execution manifest; returns clean error state on invalid names
- `tools/tool_model_executor.py` — `run_model` returns `{"error": ..., "model": ...}` dict for unknown model names instead of raising `ValueError`
- `tests/test_model_executor.py` — updated test to match new `run_model` error-dict behavior
- `.gitignore` — added `docs/` and `eval/outputs/` to gitignore

### Added (eval harness)
- `eval/main.py` — PathMMU benchmark evaluation harness; mirrors actual app flow: Executor runs each model, Synthesizer picks the answer letter, result scored against ground truth. `--models` is optional — when omitted, uses the same `MODALITY_MODEL_MAPPING` top-2 defaults the UI pre-selects per category. `--n_models` controls how many suggestions to use (default: 2). Outputs `output.json` + `result.json` per category and `summary.json` overall in PathMMU-compatible schema.
- `eval/adapters.py` — model-specific adapters: CLIP models (biomedclip, conch, musk) receive answer options as comma-separated labels via `parse_labels()`; VLMs (medgemma, chexagent, llava_med) receive the full formatted prompt; response parsed with `get_multi_choice_prediction()`
- `eval/parse.py` — path-safe import wrapper for PathMMU's `get_multi_choice_prediction`
- `eval/configs/default.yaml` — `multi_choice_example_format` for `construct_prompt`
- `eval/scripts/ensemble_mac.sh` — modality-aware default eval (no `--models`; mirrors UI defaults)
- `eval/scripts/clip_ensemble_mac.sh` — CLIP-only ensemble (biomedclip + conch + musk; fastest)
- `eval/scripts/full_ensemble_mac.sh` — all 5 in-process models

---

## [main] — Week 3 (2026-04-15)

### Added
- `tools/tool_model_executor.py` — unified wrapper for all 7 medical AI models; in-process dispatch for BiomedCLIP, CONCH, MUSK, MedGemma, ViT-Alzheimer; subprocess isolation for CheXagent and LLaVA-Med via dedicated venvs
- `agent/graph.py` — full LangGraph `StateGraph` with three nodes: Reviewer (validation + manifest), Executor (model dispatch), Synthesizer (LLM consensus); `MemorySaver` checkpointer
- `agent/state.py` — `AgentState` TypedDict schema
- `app.py` — Streamlit interactive ensemble UI; sidebar directory scan + manifest; modality-aware model suggestions; ensemble multiselect; clinical query text area; two-tab result display (Synthesized Report / Audit Trail)
- `utils/viz_utils.py` — `create_medical_viz()` and `get_image_metadata()` for modality-aware image preview
- `QUICKSTART.md` — setup and run instructions

### Changed
- `utils/manifest_generator.py` — extension-based modality heuristics for directory scanning

---

## [main] — Week 2 (2026-04-07 approx.)

### Added
- `mcp_server.py` — FastMCP server exposing 7 medical AI models as MCP tools (stdio/SSE transport)
- `utils/model_utils.py` — inference implementations for BiomedCLIP, CONCH, MUSK, MedGemma, ViT-Alzheimer; lazy model caching; unified `load_image()` supporting DICOM, NIfTI, standard formats
- `utils/run_chexagent.py`, `utils/run_llava_med.py` — subprocess entry points for isolated venv models
- Multi-model testing across Week 1 image types; timing benchmarks documented

---

## [main] — Week 1 (2026-03-31 approx.)

### Added
- LangGraph multimodality parsing tools and router logic (initial version)
- `tools/tool_oasis_parser.py` — OASIS Legacy MRI parser (`nibabel` + `xml.etree`)
- `tools/tool_dicom_parser.py`, `tool_hsi_parser.py`, `tool_quilt_parser.py`, `tool_iq_oth_parser.py` — modality parsers
- `download_data.py` — automated Google Drive data fetching (skip-if-exists logic)
- `pyproject.toml` — dependency lockfile with all medical imaging libraries
- `workspace/mock_oasis/` — sample `.img`, `.hdr`, `.xml` files for local testing
- Tests for all Week 1 image types
- `.gitignore` — excludes datasets, venvs, credentials, AI tooling files
