"""
Test program: Quilt1M — PubMed subset
Sample     : c901a42b-0ab9-45d9-809d-dd646effcf9c_1.jpg
Source     : PubMed open-access figure (academic paper illustration)
Caption    : IDH1 immunocytochemistry in MG63 and U2OS osteosarcoma cell lines

Loads the sample image, extracts pixel statistics, retrieves full CSV metadata
(caption, UMLS entities, pathology sub-specialty, magnification), renders a
diagnostic visualization, and writes a JSON report.
"""

from pathlib import Path
from _quilt1m_common import run_test

SUBSET           = "pubmed"
IMAGE_FILENAME   = "c901a42b-0ab9-45d9-809d-dd646effcf9c_1.jpg"
DATA_DIR         = Path(__file__).parent / "data" / "Quilt1M_pubmed"
OUTPUT_DIR       = Path(__file__).parent / "output" / "Quilt1M_pubmed"

if __name__ == "__main__":
    run_test(SUBSET, IMAGE_FILENAME, DATA_DIR, OUTPUT_DIR)
