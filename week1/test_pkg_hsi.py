"""
Test program: PKG – HistologyHSI-GB (Hyperspectral Histology of Glioblastoma)
Modality   : Hyperspectral Imaging (HSI) — H&E-stained histology slides
Domain     : Pathology (Brain tumor — GBM)
Format     : ENVI BIL (headerless binary + .hdr) + PNG preview
Tools      : spectral (Python Spectral Imagery), SimpleImageWrapper (TissueLab-SDK),
             numpy, matplotlib

Sample     : P1 / ROI_01_C01_T  (Patient 1, ROI 1, Tumor tissue)

What this program extracts:
  - Full ENVI header metadata (bands, wavelengths, sensor, interleave)
  - Raw hyperspectral cube shape: 800 × 1004 × 826 bands (400–1000 nm)
  - White/dark calibration references (reflectance normalisation)
  - Calibrated reflectance cube (pixel-level normalisation)
  - Spectral signature at multiple representative pixels
  - Per-band statistics (sample 20 bands across visible + NIR)
  - RGB preview via SimpleImageWrapper
  - False-colour band image (single band as grayscale)
  - Spectral mean plot
  - JSON + PNG report output
"""

import json
import sys
from pathlib import Path

import numpy as np

try:
    import spectral
    import spectral.io.envi as envi
    HAS_SPECTRAL = True
except ImportError:
    print("[ERROR] spectral library is required: pip install spectral")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not installed — skipping visualizations")

try:
    from tissuelab_sdk.wrapper import SimpleImageWrapper
    HAS_SDK = True
except ImportError:
    HAS_SDK = False
    print("[WARN] TissueLab-SDK not available — skipping SDK demo")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PATIENT_ID = "P1"
ROI_ID     = "ROI_01_C01_T"   # Tumor ROI
LABEL      = "Tumor"

DATA_DIR   = Path(__file__).parent / "data" / "PKG_HistologyHSI_GB" / PATIENT_ID / ROI_ID
OUTPUT_DIR = Path(__file__).parent / "output" / "PKG_HistologyHSI_GB"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_HDR         = DATA_DIR / "raw.hdr"
RAW_BIN         = DATA_DIR / "raw"
WHITE_HDR       = DATA_DIR / "whiteReference.hdr"
WHITE_BIN       = DATA_DIR / "whiteReference"
DARK_HDR        = DATA_DIR / "darkReference.hdr"
DARK_BIN        = DATA_DIR / "darkReference"
RGB_PREVIEW     = DATA_DIR / "rgb.png"
PATIENT_PREVIEW = DATA_DIR.parent / f"{PATIENT_ID}.png"

# Wavelength range label
WL_RANGE = "400–1000 nm (826 bands)"

# ---------------------------------------------------------------------------
# Header parsing helper
# ---------------------------------------------------------------------------
def parse_envi_header(hdr_path: Path) -> dict:
    """Load ENVI header and return a serialisable metadata dict."""
    img = envi.open(str(hdr_path))
    md  = img.metadata
    # Extract wavelengths as floats
    wavelengths = None
    if "wavelength" in md:
        try:
            wavelengths = [float(w) for w in md["wavelength"]]
        except (ValueError, TypeError):
            wavelengths = md["wavelength"]

    result = {
        "lines":           int(md.get("lines", 0)),
        "samples":         int(md.get("samples", 0)),
        "bands":           int(md.get("bands", 0)),
        "interleave":      md.get("interleave", ""),
        "data_type":       md.get("data_type", ""),
        "byte_order":      md.get("byte_order", ""),
        "header_offset":   md.get("header offset", ""),
        "sensor_type":     md.get("sensor type", ""),
        "wavelength_units":md.get("wavelength units", ""),
        "wavelength_min":  round(wavelengths[0], 2) if wavelengths else None,
        "wavelength_max":  round(wavelengths[-1], 2) if wavelengths else None,
        "num_wavelengths": len(wavelengths) if wavelengths else None,
        "wavelength_sample_10": (
            [round(w, 2) for w in wavelengths[::len(wavelengths)//10]]
            if wavelengths and len(wavelengths) >= 10 else wavelengths
        ),
        "full_metadata": {k: str(v)[:200] for k, v in md.items()},
    }
    return result, img, wavelengths


# ---------------------------------------------------------------------------
# Calibration (reflectance = (raw - dark) / (white - dark))
# ---------------------------------------------------------------------------
def load_cube(hdr_path: Path) -> np.ndarray:
    """Load an ENVI file as a float32 numpy array (lines × samples × bands)."""
    img = envi.open(str(hdr_path))
    return img.load().astype(np.float32)


def calibrate(raw: np.ndarray, white: np.ndarray, dark: np.ndarray) -> np.ndarray:
    """
    Convert raw DN values to relative reflectance.
    white and dark are (1 × samples × bands) reference frames.
    """
    denom = white - dark
    # Avoid division by zero
    denom = np.where(denom == 0, 1e-6, denom)
    reflectance = (raw - dark) / denom
    return np.clip(reflectance, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Spectral signature extraction
# ---------------------------------------------------------------------------
def spectral_signature(cube: np.ndarray, row: int, col: int, wavelengths: list) -> dict:
    """Extract and describe the spectral signature at a pixel."""
    sig = cube[row, col, :].tolist()
    wls = wavelengths or list(range(len(sig)))
    peak_idx = int(np.argmax(sig))
    return {
        "pixel_rc":   [row, col],
        "signature":  [round(float(v), 6) for v in sig],
        "peak_wl_nm": round(wls[peak_idx], 2),
        "peak_value": round(float(sig[peak_idx]), 6),
        "mean":       round(float(np.mean(sig)), 6),
        "std":        round(float(np.std(sig)), 6),
    }


def band_statistics(cube: np.ndarray, wavelengths: list, n_sample: int = 20) -> list:
    """Sample n bands evenly across the spectral dimension and compute stats."""
    n_bands = cube.shape[2]
    indices = np.linspace(0, n_bands - 1, n_sample, dtype=int)
    stats = []
    for idx in indices:
        band = cube[:, :, idx].flatten().astype(float)
        wl   = round(float(wavelengths[idx]), 2) if wavelengths else int(idx)
        stats.append({
            "band_index":    int(idx),
            "wavelength_nm": wl,
            "min":           round(float(band.min()), 6),
            "max":           round(float(band.max()), 6),
            "mean":          round(float(band.mean()), 6),
            "std":           round(float(band.std()), 6),
        })
    return stats


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def make_visualization(
    raw_cube: np.ndarray,
    cal_cube: np.ndarray,
    wavelengths: list,
    rgb_path: Path,
    sdk_info: dict | None,
) -> None:
    if not HAS_MPL:
        return

    n_bands = raw_cube.shape[2]
    # Pick representative bands: ~450nm (blue), ~550nm (green), ~700nm (red-edge), ~800nm (NIR)
    if wavelengths:
        wl_arr = np.array(wavelengths)
        band_blue  = int(np.argmin(np.abs(wl_arr - 450)))
        band_green = int(np.argmin(np.abs(wl_arr - 550)))
        band_red   = int(np.argmin(np.abs(wl_arr - 650)))
        band_nir   = int(np.argmin(np.abs(wl_arr - 800)))
    else:
        band_blue, band_green, band_red, band_nir = 0, n_bands//4, n_bands//2, 3*n_bands//4

    def norm(arr2d: np.ndarray) -> np.ndarray:
        lo, hi = arr2d.min(), arr2d.max()
        return ((arr2d - lo) / (hi - lo + 1e-9) * 255).astype(np.uint8)

    # Build pseudo-RGB from calibrated cube
    r_ch = norm(cal_cube[:, :, band_red])
    g_ch = norm(cal_cube[:, :, band_green])
    b_ch = norm(cal_cube[:, :, band_blue])
    pseudo_rgb = np.stack([r_ch, g_ch, b_ch], axis=2)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 1. SDK / PIL RGB preview
    ax = axes[0, 0]
    if rgb_path.exists():
        from PIL import Image as PILImage
        preview = np.array(PILImage.open(str(rgb_path)).convert("RGB"))
        ax.imshow(preview)
        ax.set_title(f"RGB Preview (rgb.png)\n{PATIENT_ID}/{ROI_ID}", fontsize=10)
    else:
        ax.text(0.5, 0.5, "RGB preview\nnot available", ha="center", va="center",
                transform=ax.transAxes)
    ax.axis("off")

    # 2. Pseudo-RGB from calibrated reflectance
    axes[0, 1].imshow(pseudo_rgb)
    axes[0, 1].set_title(
        f"Pseudo-RGB (calibrated reflectance)\n"
        f"B={wavelengths[band_blue]:.0f}nm  G={wavelengths[band_green]:.0f}nm  "
        f"R={wavelengths[band_red]:.0f}nm",
        fontsize=10
    )
    axes[0, 1].axis("off")

    # 3. NIR band (single band grayscale)
    axes[0, 2].imshow(cal_cube[:, :, band_nir], cmap="viridis")
    axes[0, 2].set_title(
        f"NIR band — {wavelengths[band_nir]:.0f} nm\n(calibrated reflectance)", fontsize=10)
    axes[0, 2].axis("off")

    # 4. Spectral mean ± std across whole image
    ax4 = axes[1, 0]
    spatial_mean = cal_cube.reshape(-1, n_bands).mean(axis=0)
    spatial_std  = cal_cube.reshape(-1, n_bands).std(axis=0)
    wl_x = wavelengths if wavelengths else list(range(n_bands))
    ax4.fill_between(wl_x,
                     spatial_mean - spatial_std,
                     spatial_mean + spatial_std,
                     alpha=0.3, color="steelblue", label="±1 std")
    ax4.plot(wl_x, spatial_mean, color="steelblue", linewidth=1.5, label="mean")
    ax4.set_xlabel("Wavelength (nm)", fontsize=9)
    ax4.set_ylabel("Reflectance", fontsize=9)
    ax4.set_title("Mean spectral signature — full image", fontsize=10)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # 5. Three pixel signatures: center, corner, edge
    ax5 = axes[1, 1]
    h, w = cal_cube.shape[:2]
    pixels = [
        ("Center",     h // 2, w // 2, "tomato"),
        ("Top-left",   h // 8, w // 8, "seagreen"),
        ("Mid-right",  h // 2, 7*w//8, "mediumpurple"),
    ]
    for name, r, c, color in pixels:
        sig = cal_cube[r, c, :]
        ax5.plot(wl_x, sig, label=f"{name} ({r},{c})", color=color, linewidth=1.2)
    ax5.set_xlabel("Wavelength (nm)", fontsize=9)
    ax5.set_ylabel("Reflectance", fontsize=9)
    ax5.set_title("Pixel spectral signatures", fontsize=10)
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # 6. Band variance map (which bands show the most spatial variance)
    ax6 = axes[1, 2]
    band_variance = cal_cube.reshape(-1, n_bands).var(axis=0)
    ax6.plot(wl_x, band_variance, color="darkorange", linewidth=1.2)
    ax6.set_xlabel("Wavelength (nm)", fontsize=9)
    ax6.set_ylabel("Spatial variance", fontsize=9)
    ax6.set_title("Per-band spatial variance\n(discriminative spectral regions)", fontsize=10)
    ax6.grid(True, alpha=0.3)

    fig.suptitle(
        f"PKG HistologyHSI-GB — {PATIENT_ID}/{ROI_ID} [{LABEL}]\n"
        "Hyperspectral Histology of Glioblastoma (H&E, 20×, 826 bands 400–1000 nm)",
        fontsize=13, fontweight="bold"
    )

    out_path = OUTPUT_DIR / f"{PATIENT_ID}_{ROI_ID}_analysis.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [VIZ] Saved → {out_path}")


# ---------------------------------------------------------------------------
# SDK demo
# ---------------------------------------------------------------------------
def sdk_rgb_demo(png_path: Path) -> dict | None:
    if not HAS_SDK or not png_path.exists():
        return None
    print(f"\n  [SDK] Loading RGB preview via SimpleImageWrapper")
    wrapper = SimpleImageWrapper(str(png_path))
    w, h = wrapper.dimensions
    region = wrapper.read_region((0, 0), 0, (w, h), as_array=True)
    info = {
        "wrapper_class":  type(wrapper).__name__,
        "dimensions_wh":  list(wrapper.dimensions),
        "properties":     dict(wrapper.properties),
        "region_shape":   list(region.shape),
    }
    print(f"    Dimensions : {info['dimensions_wh']}")
    print(f"    Mode       : {wrapper.properties.get('mode', '?')}")
    wrapper.close()
    return info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("\n" + "="*60)
    print("  PKG HistologyHSI-GB — Hyperspectral Histology Probe")
    print(f"  Sample   : {PATIENT_ID}/{ROI_ID}  [{LABEL}]")
    print("  Modality : Hyperspectral Imaging (HSI)")
    print("  Format   : ENVI BIL  (raw + raw.hdr)")
    print("  Domain   : Pathology (Glioblastoma)")
    print("="*60)

    report: dict = {
        "dataset":   "PKG HistologyHSI-GB",
        "patient":   PATIENT_ID,
        "roi":       ROI_ID,
        "label":     LABEL,
        "modality":  "Hyperspectral Imaging (HSI)",
        "format":    "ENVI BIL",
        "domain":    "Pathology (Brain — Glioblastoma)",
        "clinical_context": (
            "13-patient hyperspectral histology dataset of GBM tissue. "
            f"ROI {ROI_ID} is a Tumor (_T) region from Patient 1. "
            "826 spectral bands (400–1000 nm), spatial resolution 800×1004, "
            "captured at 20× magnification with HEADWALL Hyperspec III sensor."
        ),
    }

    # -- ENVI Header --------------------------------------------------------
    print("\n--- ENVI Header (raw cube) ---")
    if not RAW_HDR.exists():
        print(f"  [ERROR] raw.hdr not found: {RAW_HDR}")
        sys.exit(1)

    hdr_info, raw_img, wavelengths = parse_envi_header(RAW_HDR)
    report["envi_header"] = hdr_info
    print(f"  Lines × Samples × Bands : {hdr_info['lines']} × {hdr_info['samples']} × {hdr_info['bands']}")
    print(f"  Interleave              : {hdr_info['interleave']}")
    print(f"  Wavelength range        : {hdr_info['wavelength_min']} – {hdr_info['wavelength_max']} nm")
    print(f"  Sensor                  : {hdr_info.get('sensor_type', 'N/A')}")

    # -- White reference header
    if WHITE_HDR.exists():
        white_hdr_info, _, _ = parse_envi_header(WHITE_HDR)
        report["white_reference_header"] = white_hdr_info
        print(f"  White ref shape         : {white_hdr_info['lines']} × {white_hdr_info['samples']} × {white_hdr_info['bands']}")
    if DARK_HDR.exists():
        dark_hdr_info, _, _ = parse_envi_header(DARK_HDR)
        report["dark_reference_header"] = dark_hdr_info

    # -- Load cubes ---------------------------------------------------------
    print("\n--- Loading Hyperspectral Cubes ---")
    print("  Loading raw cube ... (this may take a moment for a 1.2 GB file)")
    raw_cube = load_cube(RAW_HDR)
    print(f"  Raw cube loaded  : shape={raw_cube.shape}  dtype={raw_cube.dtype}")
    report["raw_cube_shape"] = list(raw_cube.shape)
    report["raw_cube_dtype"] = str(raw_cube.dtype)

    white_cube = load_cube(WHITE_HDR) if WHITE_HDR.exists() else None
    dark_cube  = load_cube(DARK_HDR)  if DARK_HDR.exists()  else None

    # -- Calibration --------------------------------------------------------
    cal_cube = raw_cube
    if white_cube is not None and dark_cube is not None:
        print("  Applying white/dark calibration ...")
        # References are (lines=1, samples, bands) — broadcast over lines
        white_ref = white_cube.mean(axis=0, keepdims=True)
        dark_ref  = dark_cube.mean(axis=0, keepdims=True)
        cal_cube  = calibrate(raw_cube, white_ref, dark_ref)
        print(f"  Calibrated cube  : min={cal_cube.min():.4f}  max={cal_cube.max():.4f}")
        report["calibration"] = "white/dark reflectance normalisation applied"
    else:
        print("  [WARN] Missing white/dark references — using raw values")
        report["calibration"] = "none (missing references)"

    # -- Pixel statistics ---------------------------------------------------
    print("\n--- Pixel Statistics ---")
    h, w, b = cal_cube.shape
    flat = cal_cube.reshape(-1, b)
    print(f"  Full cube stats  : mean={flat.mean():.4f}  std={flat.std():.4f}  "
          f"min={flat.min():.4f}  max={flat.max():.4f}")
    report["cube_stats"] = {
        "shape": list(cal_cube.shape),
        "mean":  round(float(flat.mean()), 6),
        "std":   round(float(flat.std()), 6),
        "min":   round(float(flat.min()), 6),
        "max":   round(float(flat.max()), 6),
    }

    # -- Band statistics (sampled) ------------------------------------------
    print("\n--- Per-Band Statistics (20 sampled bands) ---")
    bstats = band_statistics(cal_cube, wavelengths, n_sample=20)
    report["band_statistics_sampled"] = bstats
    print(f"  {'Band':>5}  {'WL(nm)':>8}  {'Mean':>8}  {'Std':>8}")
    for bs in bstats:
        print(f"  {bs['band_index']:>5}  {bs['wavelength_nm']:>8.1f}  "
              f"{bs['mean']:>8.5f}  {bs['std']:>8.5f}")

    # -- Spectral signatures ------------------------------------------------
    print("\n--- Spectral Signatures at Representative Pixels ---")
    pixels = [("center", h // 2, w // 2), ("top_left", h // 8, w // 8),
              ("mid_right", h // 2, 7 * w // 8)]
    sigs = {}
    for name, r, c in pixels:
        sig = spectral_signature(cal_cube, r, c, wavelengths)
        sigs[name] = sig
        print(f"  Pixel [{name:10s}] ({r},{c})  peak={sig['peak_wl_nm']}nm  "
              f"mean={sig['mean']:.5f}")
    report["spectral_signatures"] = sigs

    # -- SDK demo -----------------------------------------------------------
    sdk_info = sdk_rgb_demo(RGB_PREVIEW)
    if sdk_info:
        report["sdk_simple_wrapper_rgb"] = sdk_info

    # -- Visualization ------------------------------------------------------
    make_visualization(raw_cube, cal_cube, wavelengths, RGB_PREVIEW, sdk_info)

    # -- Save report --------------------------------------------------------
    report_path = OUTPUT_DIR / f"{PATIENT_ID}_{ROI_ID}_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[REPORT] Saved → {report_path}")
    print("\n[DONE]")


if __name__ == "__main__":
    main()
