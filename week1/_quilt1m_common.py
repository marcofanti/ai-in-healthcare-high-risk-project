"""
Shared utilities for Quilt1M test programs.
Not intended to be run directly.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from tissuelab_sdk.wrapper import SimpleImageWrapper
    HAS_SDK = True
except ImportError:
    HAS_SDK = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "week1" / "data" / "quilt_1M_lookup.csv"

# ---------------------------------------------------------------------------
# Metadata lookup
# ---------------------------------------------------------------------------
def lookup_csv_row(image_filename: str) -> dict:
    """Find the CSV row matching image_filename. Returns {} if not found."""
    if not HAS_PANDAS:
        return {"error": "pandas not installed"}
    if not CSV_PATH.exists():
        return {"error": f"CSV not found: {CSV_PATH}"}

    df = pd.read_csv(CSV_PATH, low_memory=False)
    matches = df[df["image_path"] == image_filename]
    if matches.empty:
        return {"warning": "no CSV row found for this image"}
    row = matches.iloc[0].to_dict()
    return {k: (None if (isinstance(v, float) and np.isnan(v)) else v)
            for k, v in row.items()}


# ---------------------------------------------------------------------------
# Image analysis
# ---------------------------------------------------------------------------
def pixel_stats(arr: np.ndarray) -> dict:
    flat = arr.flatten().astype(float)
    counts, edges = np.histogram(flat, bins=10)
    return {
        "shape":         list(arr.shape),
        "dtype":         str(arr.dtype),
        "min":           float(flat.min()),
        "max":           float(flat.max()),
        "mean":          round(float(flat.mean()), 4),
        "std":           round(float(flat.std()), 4),
        "median":        round(float(np.median(flat)), 4),
        "percentile_5":  round(float(np.percentile(flat, 5)), 4),
        "percentile_95": round(float(np.percentile(flat, 95)), 4),
        "histogram_counts": counts.tolist(),
        "histogram_edges":  [round(e, 2) for e in edges.tolist()],
    }


def load_image(path: Path) -> tuple[dict, np.ndarray]:
    """Load image with PIL; return (pil_info dict, RGB numpy array)."""
    img = PILImage.open(str(path))
    info = {
        "format":   img.format,
        "mode":     img.mode,
        "size_wh":  list(img.size),
        "info_tags": {k: str(v)[:100] for k, v in img.info.items()},
    }
    arr = np.array(img.convert("RGB"))
    img.close()
    return info, arr


def sdk_load(path: Path) -> dict | None:
    if not HAS_SDK:
        return None
    wrapper = SimpleImageWrapper(str(path))
    w, h = wrapper.dimensions
    region = wrapper.read_region((0, 0), 0, (w, h), as_array=True)
    info = {
        "wrapper_class":    type(wrapper).__name__,
        "dimensions_wh":    list(wrapper.dimensions),
        "level_count":      wrapper.level_count,
        "properties":       dict(wrapper.properties),
        "region_shape":     list(region.shape),
    }
    wrapper.close()
    return info


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def make_visualization(
    arr: np.ndarray,
    subset: str,
    image_filename: str,
    meta: dict,
    output_dir: Path,
) -> None:
    if not HAS_MPL:
        return

    caption = str(meta.get("caption", ""))[:120]
    pathology = str(meta.get("pathology", ""))
    magnification = meta.get("magnification", "")
    umls_ids = str(meta.get("med_umls_ids", ""))[:120]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Image
    axes[0].imshow(arr)
    axes[0].set_title(f"Quilt1M [{subset}]\n{image_filename}", fontsize=10)
    axes[0].axis("off")

    # Histogram
    gray = np.mean(arr, axis=2)
    axes[1].hist(gray.flatten(), bins=50, color="teal", edgecolor="none", alpha=0.8)
    axes[1].set_xlabel("Pixel intensity", fontsize=9)
    axes[1].set_ylabel("Count", fontsize=9)
    axes[1].set_title("Intensity histogram (grayscale mean)", fontsize=10)
    axes[1].grid(True, alpha=0.3)

    meta_text = (
        f"Subset: {subset}  |  Split: {meta.get('split','?')}  |  "
        f"Magnification: {magnification}\n"
        f"Pathology: {pathology[:80]}\n"
        f"Caption: {caption}\n"
        f"UMLS IDs: {umls_ids}"
    )
    fig.text(0.01, -0.05, meta_text, fontsize=7.5, wrap=True,
             verticalalignment="bottom", family="monospace")

    fig.suptitle(f"Quilt1M Dataset — {subset.upper()} subset sample",
                 fontsize=13, fontweight="bold")

    out_path = output_dir / f"Quilt1M_{subset}_{Path(image_filename).stem}_analysis.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [VIZ] Saved → {out_path}")


# ---------------------------------------------------------------------------
# Run one subset test (called from each test_quilt1m_*.py)
# ---------------------------------------------------------------------------
def run_test(
    subset: str,
    image_filename: str,
    data_dir: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = data_dir / image_filename

    print("\n" + "="*60)
    print(f"  Quilt1M Dataset — {subset.upper()} subset")
    print(f"  Sample : {image_filename}")
    print("  Modality : Optical microscopy (H&E histopathology)")
    print("  Domain   : Pathology")
    print("="*60)

    report: dict = {
        "dataset":  "Quilt1M",
        "subset":   subset,
        "image":    image_filename,
        "modality": "Optical microscopy (H&E histopathology)",
        "domain":   "Pathology",
        "clinical_context": (
            "1M+ image-text pairs for histopathology. "
            f"This sample comes from the '{subset}' source: "
            + {
                "pubmed":    "PubMed open-access figure images with academic captions.",
                "quilt":     "Frame extractions from YouTube pathology educational videos.",
                "openpath":  "Twitter/social media pathology posts (OpenPath).",
                "laion":     "LAION web-scraped histopathology images.",
            }.get(subset, "")
        ),
    }

    if not image_path.exists():
        print(f"  [ERROR] Image not found: {image_path}")
        return

    # PIL load
    pil_info, arr = load_image(image_path)
    report["pil"] = pil_info
    print(f"  Format     : {pil_info['format']}")
    print(f"  Mode       : {pil_info['mode']}")
    print(f"  Size       : {pil_info['size_wh'][0]}W × {pil_info['size_wh'][1]}H px")

    # SDK load
    sdk_info = sdk_load(image_path)
    if sdk_info:
        report["sdk"] = sdk_info
        print(f"  SDK wrapper: {sdk_info['wrapper_class']}  dims={sdk_info['dimensions_wh']}")

    # Pixel stats
    stats = pixel_stats(arr)
    report["pixel_stats"] = stats
    print(f"  Intensity  : min={stats['min']:.1f}  max={stats['max']:.1f}  "
          f"mean={stats['mean']:.1f}  std={stats['std']:.1f}")

    # Per-channel
    channels = {}
    for i, ch in enumerate(["R", "G", "B"]):
        channels[ch] = {
            "mean": round(float(arr[:, :, i].mean()), 4),
            "std":  round(float(arr[:, :, i].std()), 4),
        }
    report["channel_stats"] = channels
    print(f"  Channels   : R={channels['R']['mean']:.1f}  "
          f"G={channels['G']['mean']:.1f}  B={channels['B']['mean']:.1f}")

    # CSV metadata
    print("\n--- CSV Metadata ---")
    meta = lookup_csv_row(image_filename)
    report["csv_metadata"] = meta
    for k, v in meta.items():
        if v is not None and k not in ("image_path",):
            print(f"  {k:<20}: {str(v)[:100]}")

    # Visualization
    make_visualization(arr, subset, image_filename, meta, output_dir)

    # Save report
    stem = Path(image_filename).stem
    report_path = output_dir / f"Quilt1M_{subset}_{stem}_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[REPORT] Saved → {report_path}")
    print("\n[DONE]")
