import os
from pathlib import Path

try:
    from tissuelab_sdk.wrapper import SimpleImageWrapper
except ImportError:
    SimpleImageWrapper = None

def parse_wsi(target_file: str) -> dict:
    """Parses WSI/Pathology images using TissueLab-SDK."""
    if not SimpleImageWrapper:
        from PIL import Image
        try:
            with Image.open(target_file) as img:
                return {
                    "dimensions": list(img.size),
                    "note": "Standard Pillow fallback (TissueLab-SDK unavailable)."
                }
        except Exception as e:
            return {"error": f"WSI generic parsing failed: {e}"}
            
    try:
        wrapper = SimpleImageWrapper(str(target_file))
        dims = wrapper.dimensions
        levels = getattr(wrapper, "level_count", 1)
        properties = getattr(wrapper, "properties", {})
        wrapper.close()
        
        return {
            "dimensions": list(dims),
            "level_count": levels,
            "vendor_properties_count": len(properties),
            "note": "Successfully parsed robust WSI coordinates through TissueLab SDK."
        }
    except Exception as e:
        return {"error": f"WSI pipeline extraction failed: {e}"}
