#!/usr/bin/env python
"""
MedHuggingGPT MCP Server
Exposes 7 medical AI models as MCP tools.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from utils.model_utils import (
    run_biomedclip, run_conch, run_musk, 
    run_medgemma, run_vit_alzheimer
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
VENV_CHEXAGENT = PROJECT_ROOT / ".venv-chexagent" / "bin" / "python"
VENV_LLAVA = PROJECT_ROOT / ".venv-llava" / "bin" / "python"
SCRIPT_CHEXAGENT = PROJECT_ROOT / "utils" / "run_chexagent.py"
SCRIPT_LLAVA = PROJECT_ROOT / "utils" / "run_llava_med.py"

# ---------------------------------------------------------------------------
# FastMCP instance
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "MedHuggingGPT",
    instructions=(
        "Medical AI inference tools. Each tool accepts an image_path and an optional prompt."
    ),
)

# ---------------------------------------------------------------------------
# MCP Tools — In-process
# ---------------------------------------------------------------------------

@mcp.tool()
def biomedclip(image_path: str, prompt: str = "") -> dict:
    """Zero-shot classification using Microsoft BiomedCLIP."""
    return run_biomedclip(image_path, prompt)

@mcp.tool()
def conch(image_path: str, prompt: str = "") -> dict:
    """Zero-shot classification using MahmoodLab CONCH."""
    return run_conch(image_path, prompt)

@mcp.tool()
def musk(image_path: str, prompt: str = "") -> dict:
    """Zero-shot classification using MUSK."""
    return run_musk(image_path, prompt)

@mcp.tool()
def medgemma(image_path: str, prompt: str = "") -> dict:
    """Visual question answering using Google MedGemma."""
    return run_medgemma(image_path, prompt)

@mcp.tool()
def vit_alzheimer(image_path: str, prompt: str = "") -> dict:
    """Alzheimer's severity classification using ViT-base."""
    return run_vit_alzheimer(image_path, prompt)

# ---------------------------------------------------------------------------
# MCP Tools — Subprocess
# ---------------------------------------------------------------------------

def _run_subprocess(python: Path, script: Path, image_path: str, prompt: str) -> dict:
    if not python.exists():
        return {"error": f"Python interpreter not found: {python}"}
    
    cmd = [str(python), str(script), "--image_path", image_path]
    if prompt:
        cmd += ["--prompt", prompt]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            return {"error": result.stderr or "Subprocess failed"}
        return json.loads(result.stdout)
    except Exception as exc:
        return {"error": str(exc)}

@mcp.tool()
def chexagent(image_path: str, prompt: str = "") -> dict:
    """Chest X-ray / radiology VLM using StanfordAIMI CheXagent."""
    return _run_subprocess(VENV_CHEXAGENT, SCRIPT_CHEXAGENT, image_path, prompt)

@mcp.tool()
def llava_med(image_path: str, prompt: str = "") -> dict:
    """General medical VLM using Microsoft LLaVA-Med."""
    return _run_subprocess(VENV_LLAVA, SCRIPT_LLAVA, image_path, prompt)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MedHuggingGPT MCP Server")
    parser.add_argument("--transport", choices=["stdio", "sse"], default="stdio")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        os.environ.setdefault("FASTMCP_PORT", str(args.port))
        mcp.run(transport=args.transport)
