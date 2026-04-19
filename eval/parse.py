"""
parse.py — Re-export PathMMU's get_multi_choice_prediction so the rest of the
eval package can import it without embedding PathMMU's path manipulation everywhere.
"""

import sys
from pathlib import Path

_PATHMMU_EVAL = Path(__file__).resolve().parent.parent.parent / "PathMMU" / "eval"
if str(_PATHMMU_EVAL) not in sys.path:
    sys.path.insert(0, str(_PATHMMU_EVAL))

from utils.eval_utils import get_multi_choice_prediction as parse_prediction  # noqa: F401
