"""
Test program: Quilt1M — Quilt (YouTube) subset
Sample     : dTr3MNl1FxE_image_c54e9a8d-9348-456a-9645-3b8921eb0b79.jpg
Source     : Frame extraction from YouTube pathology educational video
Caption    : Nephrogenic systemic fibrosis — dermatopathology / soft tissue / breast

Loads the sample image, extracts pixel statistics, retrieves full CSV metadata
(caption, UMLS entities, pathology sub-specialty, magnification), renders a
diagnostic visualization, and writes a JSON report.
"""

from pathlib import Path
from _quilt1m_common import run_test

SUBSET           = "quilt"
IMAGE_FILENAME   = "dTr3MNl1FxE_image_c54e9a8d-9348-456a-9645-3b8921eb0b79.jpg"
DATA_DIR         = Path(__file__).parent / "data" / "Quilt1M_quilt"
OUTPUT_DIR       = Path(__file__).parent / "output" / "Quilt1M_quilt"

if __name__ == "__main__":
    run_test(SUBSET, IMAGE_FILENAME, DATA_DIR, OUTPUT_DIR)
