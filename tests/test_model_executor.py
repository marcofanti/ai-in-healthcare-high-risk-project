import pytest
import sys
import os

# Ensure tools are importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.tool_model_executor import run_model

def test_run_model_unsupported():
    with pytest.raises(ValueError, match="Unsupported model"):
        run_model("non_existent_model", "path/to/image.jpg", "test prompt")
