"""
Test program: Quilt1M — LAION (web-scraped) subset
Sample     : 00004000040081.jpg
Source     : LAION large-scale web-scraped image-text dataset
Caption    : Cytosolic Sulfotransferase 1A1 / SULT1A1 antibody

Loads the sample image, extracts pixel statistics, retrieves full CSV metadata
(caption, UMLS entities, pathology sub-specialty, magnification), renders a
diagnostic visualization, and writes a JSON report.
"""

from pathlib import Path
from _quilt1m_common import run_test

SUBSET           = "laion"
IMAGE_FILENAME   = "00004000040081.jpg"
DATA_DIR         = Path(__file__).parent / "data" / "Quilt1M_laion"
OUTPUT_DIR       = Path(__file__).parent / "output" / "Quilt1M_laion"

if __name__ == "__main__":
    run_test(SUBSET, IMAGE_FILENAME, DATA_DIR, OUTPUT_DIR)
