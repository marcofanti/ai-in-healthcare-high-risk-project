# Architecture Decision Records

Decisions made during design and development of MedHuggingGPT. Each entry records what was decided, why, and what alternatives were rejected — so the team does not re-litigate settled questions.

---

## ADR-001: Local-Only Architecture (No Cloud Storage)

**Date:** 2026-04-13  
**Status:** Accepted

**Decision:** All data ingestion and processing runs entirely on local hardware. Cloud storage is out of scope.

**Reason:** Academic capstone with a 4-week timeline. Eliminating cloud infrastructure removes authentication overhead, network latency, and cost, allowing the team to focus on the AI routing architecture — which is the primary deliverable.

**Rejected alternative:** AWS S3 or Google Cloud Storage for dataset staging. Rejected due to setup time and scope.

---

## ADR-002: Pass-by-Path Instead of Pass-by-Value

**Date:** 2026-04-13  
**Status:** Accepted

**Decision:** The LLM never receives raw medical image data. Tools are invoked with absolute local file paths. Results (PNG masks, JSON summaries) are written to the workspace and returned as new paths.

**Reason:** Medical imaging files (3D MRI volumes, HSI cubes) are orders of magnitude too large to fit in an LLM context window. Pass-by-path also preserves data locality and avoids encoding/decoding overhead.

**Rejected alternative:** Base64-encoding image slices into the LLM prompt. Rejected — impractical for 3D data and introduces security concerns.

---

## ADR-003: Streamlit over React/Next.js

**Date:** 2026-04-13  
**Status:** Accepted

**Decision:** The UI is built exclusively in Streamlit.

**Reason:** Native Python integration eliminates a separate frontend build pipeline. Streamlit's session state and widget model maps directly to LangGraph's thread-based checkpointing. Given a 4-week timeline and a Python-only team, the productivity gain outweighs the UI flexibility tradeoff.

**Rejected alternative:** React/Next.js with a FastAPI backend. Rejected — development overhead unjustified for an academic prototype.

---

## ADR-004: MemorySaver for HITL State Persistence

**Date:** 2026-04-15  
**Status:** Accepted (with known limitation — see PRD-001)

**Decision:** LangGraph's `MemorySaver()` (RAM-based) is used as the checkpointer for the MVP.

**Reason:** Eliminates the need for database configuration. Sufficient for local, single-session demos within a 4-week timeline.

**Known limitation:** State is lost on browser close or server restart. PRD-001 tracks migration to `SqliteSaver` for a persistent solution.

**Rejected alternative:** `SqliteSaver` — correct long-term choice, deferred to post-Week 4.

---

## ADR-005: datasets_config.json as Source of Truth for Dataset Routing

**Date:** 2026-04-19  
**Status:** Accepted

**Decision:** Available datasets and their modalities are declared in `datasets_config.json`. The `modality` field — not file extension heuristics — drives model routing and UI display.

**Reason:** File extension alone is ambiguous. Both Legacy MRI and HSI datasets use `.hdr` files. A config-driven approach makes modality explicit, eliminates heuristic failures, and enables the UI to show datasets without scanning directories at runtime.

**Rejected alternative:** Live directory scanning with extension-based modality detection (previous approach in `utils/manifest_generator.py`). Rejected — `.hdr` collision broke routing; scanning on every UI render added latency.

---

## ADR-006: Three-Layer Model Registry Validation

**Date:** 2026-04-19  
**Status:** Accepted

**Decision:** Model name validation is enforced at three independent layers:
1. UI `st.multiselect` — limits selectable options to `SUPPORTED_MODELS`
2. `reviewer_node` — validates names before building the execution manifest
3. `run_model()` — returns a structured error dict for any name that bypasses the above

**Reason:** Any single layer can be bypassed (UI can be called programmatically; LLM could hallucinate a tool name in future agentic extensions). Defense-in-depth ensures the executor never crashes the graph regardless of entry point.

**Rejected alternative:** Validate only at `run_model()`. Rejected — a crash inside the executor node corrupts LangGraph state and prevents the Synthesizer from generating even a partial report.

---

## ADR-007: "Visualize" Button Gates Image Preview and Ensemble Config

**Date:** 2026-04-19  
**Status:** Accepted

**Decision:** Selecting a file in the sidebar does not trigger visualization automatically. The user must click "Visualize" explicitly. Sections 2 (image preview) and 3 (Ensemble Configuration) are hidden until the button is clicked. Changing the file selection immediately clears the current visualization.

**Reason:** The previous behavior triggered expensive visualization calls on every widget interaction, including during the default-selection render on page load. The explicit gate prevents accidental ensemble runs against an unconfirmed file and avoids unnecessary compute on the default item.

**Rejected alternative:** Auto-visualize on selection. Rejected — caused immediate LLM/model calls on page load before the user had confirmed their intent.
