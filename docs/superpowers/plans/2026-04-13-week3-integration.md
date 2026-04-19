# MedHuggingGPT Week 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect Week 2 medical AI models to the LangGraph agent, implement LLM-driven dynamic routing in the Planner, and enhance clinical report synthesis.

**Architecture:** The agent uses a LangGraph `StateGraph`. The **Planner** uses an LLM to map user queries and file manifests to specific modality tools (structural or AI-based). **Model Tools** wrap the existing Week 2 inference logic. The **Synthesizer** uses an LLM to create a final report from all tool outputs.

**Tech Stack:** `langgraph`, `langchain-openai`, `transformers`, `torch`, `pydicom`, `nibabel`, `spectral`.

---

### Task 1: Create Model Tool Wrappers

**Files:**
- Create: `tools/tool_model_executor.py`
- Test: `tests/test_model_executor.py`

- [ ] **Step 1: Implement `tools/tool_model_executor.py`**
Wrap the functions from `mcp_server.py` and the subprocess calls for CheXagent/LLaVA-Med into a unified interface for the LangGraph agent.

- [ ] **Step 2: Commit**
`git add tools/tool_model_executor.py && git commit -m "feat: add model executor tool wrapper"`

---

### Task 2: Update Agent State for LLM Integration

**Files:**
- Modify: `agent/state.py`

- [ ] **Step 1: Add LLM-related fields to `AgentState`**
Ensure the state can hold the necessary context for the LLM.

- [ ] **Step 2: Commit**
`git add agent/state.py && git commit -m "chore: update agent state schema"`

---

### Task 3: Implement LLM-Driven Planner Node

**Files:**
- Modify: `agent/graph.py`

- [ ] **Step 1: Update `planner_node` to use `ChatOpenAI`**
Replace the hardcoded mapping with a dynamic LLM-based planner.

- [ ] **Step 2: Commit**
`git add agent/graph.py && git commit -m "feat: implement LLM-driven planner node"`

---

### Task 4: Enhance Executor and Synthesizer Nodes

**Files:**
- Modify: `agent/graph.py`

- [ ] **Step 1: Update `executor_node` to handle AI models**
Update the dispatch logic to include the Week 2 models.

- [ ] **Step 2: Update `synthesizer_node` to use LLM**
Use an LLM to write a professional clinical report based on the `execution_steps`.

- [ ] **Step 3: Commit**
`git add agent/graph.py && git commit -m "feat: enhance executor and synthesizer nodes"`

---

### Task 5: End-to-End Validation

- [ ] **Step 1: Run `main.py` with the new graph**
- [ ] **Step 2: Verify the generated report is clinical-grade and accurate**
