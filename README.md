# Project Implementation Plan & Architecture Document

**Project Name:** MedHuggingGPT (Pan-Medical Multimodal AI Agent)  
**Phase:** Minimum Viable Product (MVP) Development  
**Duration:** 4 Weeks  
**Team Size:** 3 Members  

---

## 1. Executive Summary
This document outlines the architecture, development methodology, and agile sprint schedule for MedHuggingGPT, an intelligent AI agent designed to orchestrate and process complex, multimodal medical datasets. 

Developed as an academic capstone prototype, the project is strictly scoped to meet a rapid four-week delivery timeline. To eliminate the overhead of enterprise cloud infrastructure and network latency, the system employs a highly optimized **Local "Pass-by-Path"** architecture. Furthermore, to eliminate the bottleneck of custom data pipeline engineering, the project strictly mandates the use of standard Python libraries (`nibabel`, `MONAI`, `spectral`) for all image pre-processing. The project relies on a LangGraph-orchestrated Large Language Model (LLM) to route diverse medical modalities, enforce Human-in-the-Loop (HITL) safety checks, and synthesize clinical reports.

## 2. System Architecture
Given the massive file sizes associated with medical data (e.g., 3D volumetric scans, Hyperspectral cubes), an LLM cannot process raw files directly within its context window. The system resolves this via a pointer-based metadata handoff strategy.

### 2.1 Local Storage & Manifest Extraction
*   **Local Staging Workspace:** All data ingestion and processing occur within a designated local directory structure (e.g., `./workspace/session_001/`). Cloud storage is completely scoped out to ensure rapid local testing and data privacy.
*   **Manifest Generator:** Prior to the LLM's invocation, a background script recursively scans the uploaded directory and generates a lightweight JSON Manifest (e.g., `{"file_path": "./workspace/OAS1_MR1.img", "type": "Legacy MRI", "size": "15MB"}`). The LLM formulates its execution plan based purely on this metadata.
*   **Pass-by-Path Execution:** External analysis tools are invoked by passing the absolute local file path. The tool processes the data, saves the resulting output (e.g., a `.png` segmentation mask or `.json` summary) back to the workspace, and returns the newly generated file path to the Agent.

### 2.2 LangGraph Orchestration & UI
The core decision engine is built using LangGraph, consisting of four primary states:
1.  **Planner Node:** Ingests the JSON Manifest and user query, outputting a sequential JSON execution plan.
2.  **HITL Gateway:** Utilizes LangGraph’s built-in `interrupt_before` functionality to pause the graph. State persistence is managed locally via `MemorySaver()` (RAM checkpointer) to bypass complex database configurations. 
3.  **Executor Node:** Iterates through the approved plan, executing the corresponding Python modality tools.
4.  **Synthesizer Node:** Aggregates intermediate tool outputs (JSON texts, image paths) and drafts the final cohesive clinical report.

The User Interface is built exclusively in **Streamlit**, providing native Python integration, localized file system management, and rapid state rendering without the overhead of web-framework (React/Next.js) development.

## 3. Development Methodology & Decoupled Testing
To ensure the three-person team can maintain parallel velocity without blocking dependencies, the project relies on strict version control and a decoupled testing strategy until full system integration in Week 3.

### 3.1 Git Repository Strategy
*   **Centralized Monorepo:** Managed via GitHub/GitLab.
*   **Branching Workflow:** Direct commits to `main` are restricted. Development occurs on isolated feature branches aligned with team domains (e.g., `feature/langgraph-engine`, `feature/modality-tools`, `feature/streamlit-ui`).
*   **Integration:** Pull Requests (PRs) must be reviewed by at least one other team member before merging.

### 3.2 Decoupled Testing & Mocking (Weeks 1 & 2)
Team members will test their components in total isolation using mocked interfaces to simulate pending upstream/downstream components.

*   **Agent Logic Testing (LangGraph Developer):** 
    *   *Protocol:* The developer tests the routing logic without waiting for real data pipelines. They will supply a **hardcoded JSON manifest** (simulating an OASIS dataset) to the LLM. When the Planner devises a route, the Executor triggers a **dummy Python tool**. This tool bypasses actual processing and immediately returns a **fixed local path to a pre-processed dummy image** and a mock JSON text. Finally, a **fixed clinical query** is evaluated by the Synthesizer node to draft the report based on the mock data.
*   **Data Pipeline Testing (Modality Engineer):** 
    *   *Protocol:* Tools are developed as standalone Python scripts. Testing is conducted via local command-line execution (e.g., `python test_nifti.py ./sample.nii`), ensuring the script successfully reads standard inputs and writes standard outputs (JSON/PNG) completely independent of the LLM or UI.
*   **UI Testing (System Integrator):** 
    *   *Protocol:* The Streamlit dashboard is built using hardcoded state dictionaries to simulate the Agent's "Proposed Plan." This allows the engineer to finalize the layout and test the HITL "Approve/Reject" state transitions before connecting the live LangGraph backend.

## 4. Implementation Schedule (4-Week Agile Sprint)
The project introduces one new medical dataset modality per week, relying strictly on standard library boilerplate to accelerate pre-processing.

### Week 1: Core Engine, UI Setup, & OASIS (Legacy MRI)
*   **Objective:** Establish the foundational architecture, repository structure, and process the first multi-file dataset.
*   **Deliverables:**
    *   Initialize Git repository and basic Streamlit UI capable of reading a local directory path.
    *   Build the base LangGraph `StateGraph` utilizing the hardcoded mock testing strategy.
    *   **Dataset Integration (OASIS):** Develop `tool_oasis_parser`. Utilize the standard Python `xml.etree.ElementTree` to extract demographics and `nibabel.load()` to parse `.img/.hdr` metadata, outputting a JSON summary.

### Week 2: 2D Histopathology & Human-in-the-Loop (HITL)
*   **Objective:** Introduce multimodality to test conditional routing and implement the safety approval gateway.
*   **Deliverables:**
    *   **Dataset Integration (Pathology):** Add 2D Histopathology (`.jpg` / `.tif`). Develop `tool_pathology_analyzer` utilizing standard `PIL`/`cv2` libraries and an off-the-shelf Hugging Face vision model (via `transformers`) for basic tissue classification.
    *   Implement LangGraph's `MemorySaver()` checkpointer and establish the HITL gateway (`interrupt_before=["Executor"]`).
    *   Update Streamlit UI to halt, display the agent's proposed execution plan, and capture user authorization.

### Week 3: 3D Volumetric (NIfTI) & Full System Integration
*   **Objective:** Handle modern 3D arrays, merge all decoupled branches, and finalize report generation.
*   **Deliverables:**
    *   **Dataset Integration (Volumetric):** Add 3D Volumetric scans (`.nii` / `.nii.gz`). Develop `tool_volumetric_segmenter` utilizing **MONAI** standard transforms (`LoadImaged`) to process the 3D array into a 2D mask slice in minimal lines of code.
    *   **System Convergence:** Remove all mock data objects. Connect the live Streamlit UI to the live LangGraph Engine and the real Python processing tools.
    *   Develop the **Synthesizer Node** to format intermediate text and image outputs into a cohesive clinical report rendered in the UI.

### Week 4: Hyperspectral Imaging (HSI) & Pan-Medical Validation
*   **Objective:** Integrate the final complex modality, execute stress tests, and freeze the codebase.
*   **Deliverables:**
    *   **Dataset Integration (HSI):** Add HSI (`.raw` / `.hdr`). Develop `tool_hsi_calibrator` using exact code snippets from the `spectral` library documentation (`envi.open`) to mathematically apply dark/white reference calibrations and extract a pseudo-RGB slice.
    *   **The Pan-Medical Stress Test:** Execute system validation using a single "mixed" workspace directory containing all four dataset types simultaneously. Verify the agent successfully routes each file to its respective tool in a unified session.
    *   Finalize code documentation and record the class demonstration video.

## 5. Team Composition & Resource Allocation
To ensure timely delivery, responsibilities are strictly siloed.

*   **Member 1: AI / Agent Architect**
    *   **Responsibilities:** Designs the LangGraph state object, configures nodes and edges, writes the system prompts, and enforces the JSON schema for tool calling. Manages the mocked LangGraph testing pipeline during Weeks 1–2.
*   **Member 2: Data & Modality Engineer**
    *   **Responsibilities:** Acts as the data pipeline specialist. Researches library documentation (`nibabel`, `MONAI`, `spectral`) to adapt standard boilerplate code into isolated, executable Python tool functions. *Constraint: Focuses entirely on data pipelining; no custom model training is permitted.*
*   **Member 3: System Integrator & UI Engineer**
    *   **Responsibilities:** Manages the Streamlit application framework. Handles local directory staging, writes the manifest extraction script, implements the visual HITL approval workflow, and renders the final markdown/image reports.

## 6. Risk Management & Scope Control
*   **Hardware / Compute Bottlenecks:** Deep learning inference on local hardware (especially 3D MONAI models) may cause system timeouts and hinder development flow.
    *   **Mitigation:** The primary academic deliverable is the *AI Routing Architecture*, not model accuracy. If heavy inference takes longer than 60 seconds, the Modality Engineer will configure the tool to use a `time.sleep(3)` delay and return a pre-computed local `.png` mask to unblock testing.
*   **Scope Creep in Pre-Processing:** Attempting to write custom data loaders for highly specialized medical formats will jeopardize the timeline.
    *   **Mitigation:** Strict adherence to an "off-the-shelf" policy. All modality processing must rely exclusively on existing boilerplate scripts provided by established Python libraries.
