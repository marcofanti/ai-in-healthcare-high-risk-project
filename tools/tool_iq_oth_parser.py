import os
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None

def parse_ct_jpeg(target_file: str) -> dict:
    """Parses standard X-Ray / CT jpegs using robust Pillow mappings."""
    if not Image:
        return {"error": "Pillow not installed"}
        
    try:
        with Image.open(target_file) as img:
            return {
                "dimensions": list(img.size),
                "format": img.format,
                "color_mode": img.mode,
                "note": "Standard 2D snapshot parsed."
            }
    except Exception as e:
        return {"error": f"CT Image parsing failed: {e}"}
