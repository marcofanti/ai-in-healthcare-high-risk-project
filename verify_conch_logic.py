import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import spectral.io.envi as envi
from conch.open_clip_custom import tokenize, get_tokenizer

def test_hsi_extraction():
    print("Testing HSI extraction logic...")
    hdr_path = Path("week1/data/PKG_HistologyHSI_GB/P1/ROI_01_C01_T/raw.hdr")
    if not hdr_path.exists():
        print(f"FAILED: {hdr_path} not found")
        return False
    
    img = envi.open(str(hdr_path))
    md = img.metadata
    wavelengths = [float(w) for w in md["wavelength"]]
    wl_arr = np.array(wavelengths)
    band_r = int(np.argmin(np.abs(wl_arr - 650)))
    band_g = int(np.argmin(np.abs(wl_arr - 550)))
    band_b = int(np.argmin(np.abs(wl_arr - 450)))
    
    # Test partial load to save memory/time
    cube = img.read_bands([band_r, band_g, band_b])
    
    def norm(arr2d):
        lo, hi = arr2d.min(), arr2d.max()
        return ((arr2d - lo) / (hi - lo + 1e-9) * 255).astype(np.uint8)
    
    r_ch = norm(cube[:, :, 0])
    g_ch = norm(cube[:, :, 1])
    b_ch = norm(cube[:, :, 2])
    
    res_img = Image.fromarray(np.stack([r_ch, g_ch, b_ch], axis=2))
    print(f"SUCCESS: HSI extracted. Image size: {res_img.size}")
    return True

def test_tokenizer():
    print("\nTesting Tokenizer logic...")
    try:
        tokenizer = get_tokenizer()
        labels = ["cancer", "normal"]
        tokens = tokenize(tokenizer, labels)
        print(f"SUCCESS: Tokenized. Shape: {tokens.shape}")
        return True
    except Exception as e:
        print(f"FAILED: Tokenizer error: {e}")
        return False

def test_paths():
    print("\nTesting File Paths...")
    paths = [
        "week1/data/Quilt1M_quilt/dTr3MNl1FxE_image_c54e9a8d-9348-456a-9645-3b8921eb0b79.jpg",
        "week1/data/PKG_HistologyHSI_GB/P1/ROI_01_C01_T/raw.hdr"
    ]
    for p in paths:
        if Path(p).exists():
            print(f"SUCCESS: Found {p}")
        else:
            print(f"FAILED: Missing {p}")

if __name__ == "__main__":
    hsi = test_hsi_extraction()
    tok = test_tokenizer()
    test_paths()
