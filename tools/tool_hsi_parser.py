import sys
import os
from pathlib import Path

try:
    import spectral
except ImportError:
    spectral = None

def parse_hsi(target_file: str) -> dict:
    """Parses ENVI hyperspectral (.hdr/.raw) histology datasets."""
    if not spectral:
        return {"error": "spectral module not installed"}
        
    try:
        # Load the ENVI header
        img = spectral.envi.open(target_file)
        return {
            "spatial_dimensions": [img.nrows, img.ncols],
            "spectral_bands": img.nbands,
            "metadata_keys": list(img.metadata.keys()),
            "note": "Successfully ingested massive dimensional HSI tensor format."
        }
    except Exception as e:
        return {"error": f"HSI ENVI parsing failed: {e}"}
