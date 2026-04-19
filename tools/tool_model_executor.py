import os
import subprocess
import json
from pathlib import Path
from typing import Any, Dict

# Supported models list
SUPPORTED_MODELS = [
    "biomedclip", "conch", "musk", "medgemma", 
    "vit_alzheimer", "chexagent", "llava_med"
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENV_CHEXAGENT = PROJECT_ROOT / ".venv-chexagent" / "bin" / "python"
VENV_LLAVA = PROJECT_ROOT / ".venv-llava" / "bin" / "python"
SCRIPT_CHEXAGENT = PROJECT_ROOT / "utils" / "run_chexagent.py"
SCRIPT_LLAVA = PROJECT_ROOT / "utils" / "run_llava_med.py"

def run_model(model_name: str, image_path: str, prompt: str) -> Dict[str, Any]:
    """Unified wrapper for all 7 medical AI models."""
    if model_name not in SUPPORTED_MODELS:
        return {"error": f"Unsupported model: '{model_name}'. Valid models: {SUPPORTED_MODELS}", "model": model_name}

    # Subprocess models (isolated venvs)
    if model_name in ["chexagent", "llava_med"]:
        venv_path = VENV_CHEXAGENT if model_name == "chexagent" else VENV_LLAVA
        script_path = SCRIPT_CHEXAGENT if model_name == "chexagent" else SCRIPT_LLAVA
        
        abs_image_path = os.path.abspath(image_path)
        cmd = [str(venv_path), str(script_path), "--image_path", abs_image_path]
        if prompt:
            cmd += ["--prompt", prompt]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                return {"error": result.stderr or f"Subprocess exited with code {result.returncode}"}
            return json.loads(result.stdout)
        except subprocess.TimeoutExpired:
            return {"error": f"Model {model_name} timed out after 300s"}
        except Exception as e:
            return {"error": str(e)}

    # In-process models
    from utils.model_utils import (
        run_biomedclip, run_conch, run_musk, 
        run_medgemma, run_vit_alzheimer
    )
    
    dispatch = {
        "biomedclip": run_biomedclip,
        "conch": run_conch,
        "musk": run_musk,
        "medgemma": run_medgemma,
        "vit_alzheimer": run_vit_alzheimer
    }
    
    try:
        return dispatch[model_name](image_path, prompt)
    except Exception as e:
        return {"error": f"In-process execution failed for {model_name}: {str(e)}"}
