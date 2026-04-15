# Design Spec: MedHuggingGPT Interactive Ensemble (v1.1)

**Date:** 2026-04-15  
**Topic:** Interactive Medical AI Agent with Multi-Model Ensemble and Clinical Reporting  
**Status:** Approved  

---

## 1. Overview
MedHuggingGPT is an interactive, LangGraph-orchestrated AI agent designed for medical imaging analysis across five modalities (CT, MRI, HSI, Histopathology, Spectral CT). This design (v1.1) pivots from a purely autonomous agent to a human-interactive ensemble system where the user drives model selection and clinical questioning through a dynamic web UI.

## 2. Goals & Success Criteria
- **Modality-Aware Ingestion:** Automatically detect modality and suggest specialized "Expert" models.
- **Multi-Model Ensemble:** Allow users to run multiple models simultaneously for cross-verification.
- **LLM-Driven Consensus:** Use Gemini 3.1 Pro to synthesize potentially conflicting model outputs into a cohesive clinical report.
- **Safety & Transparency:** Implement a "Dual-Gate" HITL (Human-in-the-Loop) strategy where users approve complex plans and review final reports with clear audit trails.

## 3. Architecture

### 3.1. Frontend: Dynamic Streamlit UI
- **File/Folder Selection:** Browser-based selection of Week 1 datasets.
- **Domain Intelligence:** Automatically identifies modality (e.g., Radiology, Pathology) using Week 1 probe logic.
- **Dynamic Query Panel:**
  - **Expert Panel:** Recommends 1-3 models (e.g., CheXagent for CXR, CONCH for Pathology) based on modality.
  - **Smart Prompts:** Pre-filled suggested questions (e.g., "Are there signs of Alzheimer's?" for MRI).
  - **Free-Text Input:** Full flexibility for user-defined clinical questions.
- **Conditional Approval Gate:** A modal appears for confirmation ONLY if >3 models are selected or the modality is high-latency (e.g., 582GB HSI cubes).

### 3.2. Orchestration: LangGraph State Machine
The agent uses a `StateGraph` with a centralized `AgentState`:

1.  **Planner (Reviewer Node):** 
    - **Model:** Gemini 3.1 Pro.
    - **Logic:** Validates user selections against modality compatibility; generates a formal execution manifest.
2.  **Executor Node:**
    - **Logic:** Dispatches parallel inference jobs.
    - **Tooling:** Wraps Week 2 model inference logic into `tools/tool_model_executor.py`.
    - **Subprocesses:** Isolated venvs for CheXagent (`.venv-chexagent`) and LLaVA-Med (`.venv-llava`).
3.  **Synthesizer (Consensus Node):**
    - **Model:** Gemini 3.1 Pro.
    - **Logic:** Analyzes all model JSON outputs.
    - **Conflict Resolution:** Identifies "Consensus Findings" and flags "Model Discrepancies" (e.g., Nodule detected by Model A but missed by Model B).

### 3.3. Data Flow: Pass-by-Path
The LLM never touches raw medical data.
- **Input:** Absolute file paths to medical images/volumes.
- **Processing:** Local tools (SimpleITK, nibabel, pydicom) handle I/O.
- **Output:** JSON results + path to generated visualizations (PNG).

## 4. Components & Tooling

### 4.1. New Tools
- **`tools/tool_model_executor.py`**: Unified wrapper for all 7 models.
  - `run_model(model_name: str, image_path: str, prompt: str) -> Dict[str, Any]`

### 4.2. Agent State (`agent/state.py`)
```python
class AgentState(TypedDict):
    user_manual_selections: Dict[str, Any] # Models, questions, files
    execution_manifest: List[Dict]          # Approved plan
    model_outputs: List[Dict]               # Raw JSON from tools
    clinical_report: str                    # Final synthesized markdown
    status: str                             # Planner -> Executing -> Synthesizing
```

## 5. Reporting Format
The **Clinical Report** will be structured as:
- **[MODALITY & ENSEMBLE INFO]**
- **### CONSENSUS FINDINGS:** Points agreed upon by all models.
- **### MODEL DISCREPANCIES / FLAGS:** Highlighting contradictions or low-confidence results.
- **### CLINICAL INTERPRETATION:** A final synthesis by Gemini 3.1 Pro with recommendations for human review.

## 6. Testing & Validation
- **Unit Tests:** `tests/test_model_executor.py` for each of the 7 model wrappers.
- **Integration Tests:** Mixed-modality workspace test (all 5 modalities in one session).
- **Safety Check:** Verify the Synthesizer correctly flags conflicting model outputs.

## 7. Risks & Mitigations
- **Latency:** High-volume HSI or MRI data may slow down the UI. 
  - *Mitigation:* Conditional HITL gate with estimated timings.
- **Model Bias:** Ensemble models may all share a common bias. 
  - *Mitigation:* Explicit "Consensus" vs "Discrepancy" reporting to encourage human critical review.
