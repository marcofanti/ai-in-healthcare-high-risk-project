"""
Test program: Quilt1M — OpenPath (Twitter/social media) subset
Sample     : 994701482116173824_0.jpg
Source     : Twitter/social media pathology post (OpenPath collection)
Caption    : Peritoneal biopsy — DailyDx challenge (UMichPath / SurgPath)

Loads the sample image, extracts pixel statistics, retrieves full CSV metadata
(caption, UMLS entities, pathology sub-specialty, magnification), renders a
diagnostic visualization, and writes a JSON report.
"""

from pathlib import Path
from _quilt1m_common import run_test

SUBSET           = "openpath"
IMAGE_FILENAME   = "994701482116173824_0.jpg"
DATA_DIR         = Path(__file__).parent / "data" / "Quilt1M_openpath"
OUTPUT_DIR       = Path(__file__).parent / "output" / "Quilt1M_openpath"

if __name__ == "__main__":
    run_test(SUBSET, IMAGE_FILENAME, DATA_DIR, OUTPUT_DIR)
