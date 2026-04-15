# MedHuggingGPT Interactive Ensemble Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the MedHuggingGPT Interactive Ensemble (v1.1) including a dynamic Streamlit UI, a modality-aware expert selection system, and a multi-model consensus report.

**Architecture:** The agent uses a LangGraph `StateGraph`. The UI allows users to select files, which are automatically identified by modality. Users can then select an ensemble of models and provide a clinical prompt. The agent executes these models via a unified tool wrapper and synthesizes a final report using Gemini 3.1 Pro.

**Tech Stack:** `streamlit`, `langgraph`, `langchain-google-genai`, `transformers`, `torch`, `pydicom`, `nibabel`, `spectral`.

---

### Task 1: Model Executor Tool Wrapper

**Files:**
- Create: `tools/tool_model_executor.py`
- Test: `tests/test_model_executor.py`

- [ ] **Step 1: Write the failing test for `run_model`**

```python
import pytest
from tools.tool_model_executor import run_model

def test_run_model_unsupported():
    with pytest.raises(ValueError, match="Unsupported model"):
        run_model("non_existent_model", "path/to/image.jpg", "test prompt")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_model_executor.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `tools/tool_model_executor.py`**
Wrap the `mcp_server.py` logic.

```python
import os
import subprocess
import json
from pathlib import Path
from typing import Any, Dict
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENV_CHEXAGENT = PROJECT_ROOT / ".venv-chexagent" / "bin" / "python"
VENV_LLAVA = PROJECT_ROOT / ".venv-llava" / "bin" / "python"
SCRIPT_CHEXAGENT = PROJECT_ROOT / "utils" / "run_chexagent.py"
SCRIPT_LLAVA = PROJECT_ROOT / "utils" / "run_llava_med.py"

def run_model(model_name: str, image_path: str, prompt: str) -> Dict[str, Any]:
    """Unified wrapper for all 7 medical AI models."""
    if model_name in ["chexagent", "llava_med"]:
        venv_path = VENV_CHEXAGENT if model_name == "chexagent" else VENV_LLAVA
        script_path = SCRIPT_CHEXAGENT if model_name == "chexagent" else SCRIPT_LLAVA
        
        cmd = [str(venv_path), str(script_path), image_path, prompt]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return {"error": result.stderr}
        return json.loads(result.stdout)
    
    # Placeholder for in-process models (biomedclip, conch, musk, medgemma, vit_alzheimer)
    # In a real implementation, we would import their loaders from mcp_server.py
    # For now, we'll return a mock for testing the orchestration
    return {"model": model_name, "image_path": image_path, "prediction": f"Mock result for {model_name}"}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_model_executor.py -v`
Expected: PASS (if we handle the ValueError correctly)

- [ ] **Step 5: Commit**

```bash
git add tools/tool_model_executor.py tests/test_model_executor.py
git commit -m "feat: add model executor tool wrapper"
```

---

### Task 2: Agent State Update

**Files:**
- Modify: `agent/state.py`

- [ ] **Step 1: Update `AgentState` schema**

```python
from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict):
    """
    LangGraph state schema for MedHuggingGPT Interactive Ensemble.
    """
    user_manual_selections: Dict[str, Any] # {'file_path': '...', 'models': [...], 'prompt': '...'}
    execution_manifest: List[Dict[str, Any]] # List of {model, path, prompt}
    model_outputs: List[Dict[str, Any]]      # Raw JSON from tools
    clinical_report: str                    # Final synthesized markdown
    status: str                             # Current status
    images_to_render: List[str]             # UI helper
```

- [ ] **Step 2: Commit**

```bash
git add agent/state.py
git commit -m "chore: update agent state schema for interactive ensemble"
```

---

### Task 3: LangGraph Node Implementation

**Files:**
- Modify: `agent/graph.py`

- [ ] **Step 1: Implement `reviewer_node`**
Validates selections and prepares manifest.

- [ ] **Step 2: Implement `executor_node`**
Calls `run_model` for each item in manifest.

- [ ] **Step 3: Implement `synthesizer_node`**
Uses Gemini 3.1 Pro for consensus report.

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from agent.state import AgentState
from tools.tool_model_executor import run_model

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro") # Or gemini-3.1-pro if available in SDK

def reviewer_node(state: AgentState):
    selections = state["user_manual_selections"]
    manifest = []
    for model in selections["models"]:
        manifest.append({
            "model": model,
            "image_path": selections["file_path"],
            "prompt": selections["prompt"]
        })
    return {"execution_manifest": manifest, "status": "planning_complete"}

def executor_node(state: AgentState):
    manifest = state["execution_manifest"]
    outputs = []
    for step in manifest:
        result = run_model(step["model"], step["image_path"], step["prompt"])
        outputs.append(result)
    return {"model_outputs": outputs, "status": "executed"}

def synthesizer_node(state: AgentState):
    outputs = state["model_outputs"]
    # Prompt logic to Gemini 3.1 Pro for consensus
    prompt = f"Synthesize a clinical report from these model outputs: {json.dumps(outputs)}"
    report = llm.invoke(prompt).content
    return {"clinical_report": report, "status": "completed"}
```

- [ ] **Step 4: Update `create_agent_graph`**
Wire the new nodes.

- [ ] **Step 5: Commit**

```bash
git add agent/graph.py
git commit -m "feat: implement interactive ensemble nodes"
```

---

### Task 4: Streamlit UI Integration

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Implement Modality-Aware UI**
Add logic to detect modality from selected file and suggest models.

- [ ] **Step 2: Add Prompt & Model Selection**
Allow users to pick models and enter prompts.

- [ ] **Step 3: Add Conditional Approval Gate**
Show confirmation if >3 models are selected.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat: integrate interactive ensemble UI in Streamlit"
```
