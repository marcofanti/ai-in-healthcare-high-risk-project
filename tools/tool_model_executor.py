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

# Supported models list
SUPPORTED_MODELS = [
    "biomedclip", "conch", "musk", "medgemma", 
    "vit_alzheimer", "chexagent", "llava_med"
]

def run_model(model_name: str, image_path: str, prompt: str) -> Dict[str, Any]:
    """Unified wrapper for all 7 medical AI models."""
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {model_name}")

    if model_name in ["chexagent", "llava_med"]:
        venv_path = VENV_CHEXAGENT if model_name == "chexagent" else VENV_LLAVA
        script_path = SCRIPT_CHEXAGENT if model_name == "chexagent" else SCRIPT_LLAVA
        
        # Ensure image_path is absolute for subprocess
        abs_image_path = os.path.abspath(image_path)
        
        cmd = [str(venv_path), str(script_path), abs_image_path, prompt]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return {"error": result.stderr}
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            return {"error": f"Failed to decode output from {model_name}", "raw_output": result.stdout}
    
    # Placeholder for in-process models (biomedclip, conch, musk, medgemma, vit_alzheimer)
    # For now, return a mock result with status indicating it needs proper integration
    return {
        "model": model_name, 
        "image_path": image_path, 
        "prediction": f"Mock result for {model_name} (Ensemble integration in progress)",
        "status": "mock_output"
    }
