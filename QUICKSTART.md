# MedHuggingGPT Quickstart Guide

MedHuggingGPT is a pan-medical multimodal AI agent designed to orchestrate and process complex medical datasets.

---

## 🚀 1. Prerequisites

Before starting, ensure you have the following:

- **Python 3.11 or higher**
- **Environment Variables:**
  - `OPENAI_API_KEY`: Required for the LangGraph agent logic.
  - `HF_TOKEN`: (Optional) Required for downloading gated models like `MahmoodLab/conch` or `google/medgemma-4b-it`.
- **System Dependencies:**
  - `gdown`: For downloading the medical datasets from Google Drive.

---

## 📦 2. Installation

### Clone the Repository
```bash
git clone <repository_url>
cd ai-in-healthcare-high-risk-project-built-by-gemini
```

### Set Up Your Environment
The project uses `uv` for fast, reliable dependency management.

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies and create a virtual environment (.venv)
uv sync
```

Alternatively, using standard `pip`:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **Note on TissueLab-SDK:** The project expects `TissueLab-SDK` to be located at `../TissueLab-SDK`. If you do not have this locally, `uv sync` may fail. You can modify `pyproject.toml` or `requirements.txt` to point to the correct location or remove the dependency if you are only running parts of the system.

---

## 💾 3. Data Setup

Download the required medical datasets (MRI, CT, Pathology, HSI) automatically:

```bash
uv run download_data.py
```
This will download the datasets into the `week1/data` directory (approx. 3-5 GB).

---

## 🖥️ 4. Running the UI (Recommended)

The primary way to interact with MedHuggingGPT is through the Streamlit interactive ensemble dashboard.

```bash
# Set your API Key first
export OPENAI_API_KEY="your-api-key-here"

uv run streamlit run app.py
```

### UI Features:
1. **Ingestion:** Scan local directories for medical datasets.
2. **Ensemble Configuration:** Select a file, pick multiple AI "Expert" models, and provide a clinical prompt.
3. **Consensus Analysis:** The agent orchestrates the ensemble and synthesizes a final clinical report.

---

## 🔌 5. Running the MCP Server (Optional)

MedHuggingGPT can also be run as a **Model Context Protocol (MCP)** server:

```bash
uv run mcp_server.py
```

### Available Tools:
- `biomedclip`: Zero-shot classification for any medical modality.
- `conch`: Zero-shot histopathology classification.
- `medgemma`: Generative VLM for radiology and clinical Q&A.
- `vit_alzheimer`: Brain MRI Alzheimer's severity classification.
- `chexagent`: Generative VLM for Chest X-rays.
- `llava_med`: Generative VLM for general medical images.

---

## 📂 6. Project Structure

- `agent/`: LangGraph orchestration logic (`graph.py`, `state.py`).
- `tools/`: Modality-specific parsing tools (`tool_oasis_parser.py`, etc.).
- `mcp_server.py`: The FastMCP server implementation.
- `download_data.py`: Script to sync datasets from Google Drive.
- `app.py`: Streamlit UI entry point.
- `week1/data/`: Default directory where datasets are stored.

---

## 🛡️ 7. Safety & HITL
The system is designed with a **Human-in-the-Loop (HITL)** safety gate. By default, the agent will propose an execution plan and pause for user authorization before processing any medical data. This ensures clinicians maintain full control over the diagnostic pipeline.

---

## 🛠️ 8. Troubleshooting

- **Missing Data:** If `uv run main.py` fails with "File not found", ensure you have run `uv run download_data.py` successfully.
- **Dependency Issues:** If `uv sync` fails on `TissueLab-SDK`, comment out the line in `pyproject.toml` or `requirements.txt` and try again.
- **Model Download Failures:** For models in `mcp_server.py` (like `CONCH`), ensure your `HF_TOKEN` has been granted access to the specific gated repository on Hugging Face.
- **CUDA/MPS Memory Errors:** If running the MCP server on a GPU with limited memory, try running only one model at a time. The server uses MPS (Apple Silicon) or CPU by default.
