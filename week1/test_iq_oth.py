"""
Test program: IQ-OTH/NCCD Lung Cancer Dataset
Modality   : CT (Chest) — JPEG/PNG slices
Domain     : Radiology
Tools      : SimpleImageWrapper (TissueLab-SDK), PIL, numpy, matplotlib

Loads one sample from each of the three classes (Benign, Malignant, Normal),
extracts pixel-level statistics, renders a diagnostic visualization, and writes
a structured JSON report.
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not installed — skipping visualizations")

try:
    from tissuelab_sdk.wrapper import SimpleImageWrapper
    HAS_SDK = True
except ImportError:
    HAS_SDK = False
    print("[WARN] TissueLab-SDK not installed — falling back to PIL only")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data" / "IQ-OTH_NCCD"
OUTPUT_DIR = Path(__file__).parent / "output" / "IQ-OTH_NCCD"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES: dict[str, Path] = {
    "Benign":    DATA_DIR / "Benign"    / "Benign_case_1.jpg",
    "Malignant": DATA_DIR / "Malignant" / "Malignant_case_1.jpg",
    "Normal":    DATA_DIR / "Normal"    / "Normal_case_1.jpg",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def pixel_stats(arr: np.ndarray) -> dict:
    """Compute comprehensive pixel statistics from a numpy array."""
    flat = arr.flatten().astype(float)
    counts, edges = np.histogram(flat, bins=10)
    return {
        "shape":        list(arr.shape),
        "dtype":        str(arr.dtype),
        "min":          float(flat.min()),
        "max":          float(flat.max()),
        "mean":         round(float(flat.mean()), 4),
        "std":          round(float(flat.std()), 4),
        "median":       round(float(np.median(flat)), 4),
        "percentile_5": round(float(np.percentile(flat, 5)), 4),
        "percentile_95":round(float(np.percentile(flat, 95)), 4),
        "histogram_counts": counts.tolist(),
        "histogram_edges":  [round(e, 2) for e in edges.tolist()],
    }


def load_with_sdk(path: Path) -> dict:
    """Load image via SimpleImageWrapper and extract wrapper-level metadata."""
    wrapper = SimpleImageWrapper(str(path))
    info = {
        "wrapper_class":      type(wrapper).__name__,
        "dimensions_wh":      list(wrapper.dimensions),
        "level_count":        wrapper.level_count,
        "level_dimensions":   [list(d) for d in wrapper.level_dimensions],
        "properties":         dict(wrapper.properties),
    }
    # Read full region as array
    w, h = wrapper.dimensions
    region_arr = wrapper.read_region((0, 0), 0, (w, h), as_array=True)
    info["region_shape"] = list(region_arr.shape)
    wrapper.close()
    return info, region_arr


def load_with_pil(path: Path) -> tuple[dict, np.ndarray]:
    """Load image directly with PIL and extract format metadata."""
    img = Image.open(str(path))
    info = {
        "format":   img.format,
        "mode":     img.mode,
        "size_wh":  list(img.size),
        "info_tags": {k: str(v) for k, v in img.info.items()},
    }
    # Convert to RGB for consistent array shape
    rgb = img.convert("RGB")
    arr = np.array(rgb)
    img.close()
    return info, arr


# ---------------------------------------------------------------------------
# Per-sample analysis
# ---------------------------------------------------------------------------
def analyse_sample(label: str, path: Path) -> dict:
    print(f"\n{'='*60}")
    print(f"  CLASS: {label}")
    print(f"  FILE : {path.name}")
    print(f"{'='*60}")

    if not path.exists():
        print(f"  [ERROR] File not found: {path}")
        return {"label": label, "error": "file not found"}

    result: dict = {"label": label, "file": str(path)}

    # --- PIL load ---
    pil_info, arr = load_with_pil(path)
    result["pil"] = pil_info
    print(f"  PIL format  : {pil_info['format']}")
    print(f"  PIL mode    : {pil_info['mode']}")
    print(f"  Dimensions  : {pil_info['size_wh'][0]}W × {pil_info['size_wh'][1]}H")

    # --- SDK load ---
    if HAS_SDK:
        sdk_info, sdk_arr = load_with_sdk(path)
        result["sdk"] = sdk_info
        print(f"  SDK wrapper : {sdk_info['wrapper_class']}")
        print(f"  SDK dims    : {sdk_info['dimensions_wh']}")
        print(f"  Level count : {sdk_info['level_count']}")
        arr = sdk_arr  # prefer SDK-loaded array

    # --- Pixel statistics ---
    stats = pixel_stats(arr)
    result["pixel_stats"] = stats
    print(f"  Shape       : {stats['shape']}")
    print(f"  Dtype       : {stats['dtype']}")
    print(f"  Intensity   : min={stats['min']:.1f}  max={stats['max']:.1f}  "
          f"mean={stats['mean']:.1f}  std={stats['std']:.1f}")
    print(f"  Percentiles : p5={stats['percentile_5']}  p95={stats['percentile_95']}")

    # Per-channel stats for RGB
    if arr.ndim == 3 and arr.shape[2] == 3:
        channels = {}
        for i, ch in enumerate(["R", "G", "B"]):
            ch_arr = arr[:, :, i]
            channels[ch] = {
                "mean": round(float(ch_arr.mean()), 4),
                "std":  round(float(ch_arr.std()), 4),
            }
        result["channel_stats"] = channels
        print(f"  Channels    : R_mean={channels['R']['mean']}  "
              f"G_mean={channels['G']['mean']}  B_mean={channels['B']['mean']}")

    return result, arr


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def make_visualization(samples: list[tuple[str, np.ndarray]]) -> None:
    if not HAS_MPL:
        return

    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    for col, (label, arr) in enumerate(samples):
        # Image panel
        ax_img = fig.add_subplot(gs[0, col])
        ax_img.imshow(arr)
        ax_img.set_title(f"{label}", fontsize=13, fontweight="bold")
        ax_img.axis("off")

        # Histogram panel
        ax_hist = fig.add_subplot(gs[1, col])
        gray = np.mean(arr, axis=2) if arr.ndim == 3 else arr
        ax_hist.hist(gray.flatten(), bins=50, color="steelblue", edgecolor="none", alpha=0.8)
        ax_hist.set_xlabel("Pixel intensity", fontsize=9)
        ax_hist.set_ylabel("Count", fontsize=9)
        ax_hist.set_title(f"{label} — intensity histogram", fontsize=10)
        ax_hist.grid(True, alpha=0.3)

    fig.suptitle("IQ-OTH/NCCD Lung Cancer Dataset — Sample Analysis\n"
                 "CT chest slices: Benign · Malignant · Normal",
                 fontsize=14, fontweight="bold")

    out_path = OUTPUT_DIR / "IQ-OTH_NCCD_sample_analysis.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [VIZ] Saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("\n" + "="*60)
    print("  IQ-OTH/NCCD Lung Cancer Dataset — Sample Probe")
    print("  Modality : Chest CT (JPEG slices)")
    print("  Domain   : Radiology")
    print("="*60)

    report: dict = {
        "dataset": "IQ-OTH/NCCD Lung Cancer Dataset",
        "modality": "CT (Chest)",
        "domain": "Radiology",
        "format": "JPEG",
        "classes": ["Benign", "Malignant", "Normal"],
        "clinical_context": (
            "2D CT chest slices labelled by Iraqi oncologists/radiologists. "
            "Benign=120 images/15 patients, Malignant=561/40, Normal=416/55. "
            "Original DICOM from SOMATOM Siemens; converted to JPEG at 1mm slice thickness."
        ),
        "samples": [],
    }

    arrays_for_viz: list[tuple[str, np.ndarray]] = []

    for label, path in SAMPLES.items():
        result, arr = analyse_sample(label, path)
        report["samples"].append(result)
        arrays_for_viz.append((label, arr))

    # Visualization
    make_visualization(arrays_for_viz)

    # Save JSON report
    report_path = OUTPUT_DIR / "IQ-OTH_NCCD_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[REPORT] Saved → {report_path}")
    print("\n[DONE]")


if __name__ == "__main__":
    main()
