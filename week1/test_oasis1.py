"""
Test program: OASIS-1 Cross-Sectional Brain MRI Dataset
Modality   : MRI (T1-weighted MPRAGE, 3D structural brain)
Domain     : Radiology / Neuroscience
Format     : Analyze 7.5 (.hdr/.img pairs)
Tools      : nibabel (direct), NiftiImageWrapper (TissueLab-SDK), matplotlib, numpy

Subject    : OAS1_0001_MR1  (74-year-old female, CDR=0, MMSE=29 — cognitively normal)

What this program extracts:
  - Full subject metadata (age, sex, CDR, MMSE, brain volumes) from .txt file
  - Analyze 7.5 header info (voxel size, orientation, data type, field of view)
  - 3D volume statistics (intensity range, mean, std, tissue contrast)
  - Three-plane visualization (sagittal / coronal / axial) of the processed MRI
  - FSL tissue segmentation mask — per-label voxel counts (GM/WM/CSF)
  - NiftiImageWrapper region read (SDK interface demo)
  - JSON + PNG report output
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    print("[ERROR] nibabel is required: pip install nibabel")
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
    from tissuelab_sdk.wrapper import NiftiImageWrapper
    HAS_SDK = True
except ImportError:
    HAS_SDK = False
    print("[WARN] TissueLab-SDK not installed — skipping NiftiImageWrapper demo")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SUBJECT_ID  = "OAS1_0001_MR1"
DATA_DIR    = Path(__file__).parent / "data" / "Oasis1" / SUBJECT_ID
OUTPUT_DIR  = Path(__file__).parent / "output" / "Oasis1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_HDR         = DATA_DIR / "RAW" / f"{SUBJECT_ID}_mpr-1_anon.hdr"
PROCESSED_HDR   = DATA_DIR / "PROCESSED" / "MPRAGE" / "T88_111" / \
                  f"{SUBJECT_ID}_mpr_n4_anon_111_t88_masked_gfc.hdr"
FSEG_HDR        = DATA_DIR / "FSL_SEG" / \
                  f"{SUBJECT_ID}_mpr_n4_anon_111_t88_masked_gfc_fseg.hdr"
FSEG_TXT        = DATA_DIR / "FSL_SEG" / \
                  f"{SUBJECT_ID}_mpr_n4_anon_111_t88_masked_gfc_fseg.txt"
METADATA_TXT    = DATA_DIR / f"{SUBJECT_ID}.txt"

# FSL tissue labels (standard FAST 3-class segmentation)
FSEG_LABELS = {0: "Background", 1: "CSF", 2: "Gray Matter", 3: "White Matter"}

# ---------------------------------------------------------------------------
# Metadata parsing
# ---------------------------------------------------------------------------
def parse_metadata_txt(path: Path) -> dict:
    """Parse OASIS per-subject metadata text file into a dict."""
    if not path.exists():
        return {"error": f"metadata file not found: {path}"}

    meta: dict = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("SCAN") or line.startswith("TYPE") or \
               line.startswith("Vox") or line.startswith("Rect") or \
               line.startswith("Orient") or line.startswith("TR") or \
               line.startswith("TE") or line.startswith("TI") or \
               line.startswith("Flip") or line.startswith("mpr-"):
                continue
            if ":" in line:
                key, _, val = line.partition(":")
                meta[key.strip()] = val.strip()
    return meta


def parse_fseg_txt(path: Path) -> dict:
    """Parse FSL segmentation stats text file (tissue volumes)."""
    if not path.exists():
        return {}
    stats: dict = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                stats[parts[0]] = parts[1]
    return stats

# ---------------------------------------------------------------------------
# Volume statistics
# ---------------------------------------------------------------------------
def volume_stats(data: np.ndarray, label: str) -> dict:
    """Compute statistics on a 3D volume."""
    flat = data.flatten()
    nonzero = flat[flat > 0]
    print(f"\n  [{label}]")
    print(f"    Shape        : {data.shape}")
    print(f"    Voxel count  : {flat.size:,}")
    print(f"    Non-zero     : {len(nonzero):,}  ({100*len(nonzero)/flat.size:.1f}%)")
    print(f"    Intensity    : min={flat.min():.1f}  max={flat.max():.1f}  "
          f"mean={flat.mean():.2f}  std={flat.std():.2f}")
    if len(nonzero):
        print(f"    Brain only   : mean={nonzero.mean():.2f}  std={nonzero.std():.2f}")

    return {
        "shape":           list(data.shape),
        "voxel_count":     int(flat.size),
        "nonzero_voxels":  int(len(nonzero)),
        "min":             float(flat.min()),
        "max":             float(flat.max()),
        "mean":            round(float(flat.mean()), 4),
        "std":             round(float(flat.std()), 4),
        "brain_mean":      round(float(nonzero.mean()), 4) if len(nonzero) else None,
        "brain_std":       round(float(nonzero.std()), 4) if len(nonzero) else None,
    }


def analyze_header(img: nib.analyze.AnalyzeImage) -> dict:
    """Extract key header fields from an Analyze 7.5 / NIfTI header."""
    hdr = img.header
    zooms = hdr.get_zooms()
    return {
        "format":           type(img).__name__,
        "data_shape":       list(hdr.get_data_shape()),
        "voxel_size_mm":    [round(float(z), 4) for z in zooms],
        "data_dtype":       str(hdr.get_data_dtype()),
        "slope_intercept":  [float(hdr.get_slope_inter()[0] or 1.0),
                             float(hdr.get_slope_inter()[1] or 0.0)],
        "affine":           img.affine.tolist(),
        "header_str":       str(hdr),
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def make_visualization(processed_data: np.ndarray, fseg_data: np.ndarray | None) -> None:
    if not HAS_MPL:
        return

    # Pick middle slices in each orientation
    processed_data = np.squeeze(processed_data)
    sx, sy, sz = processed_data.shape
    sag_slice = processed_data[sx // 2, :, :]
    cor_slice = processed_data[:, sy // 2, :]
    axi_slice = processed_data[:, :, sz // 2]

    rows = 2 if fseg_data is not None else 1
    fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))
    if rows == 1:
        axes = axes[np.newaxis, :]

    for ax, sl, title in zip(axes[0],
                              [sag_slice, cor_slice, axi_slice],
                              ["Sagittal (mid)", "Coronal (mid)", "Axial (mid)"]):
        ax.imshow(np.rot90(sl), cmap="gray", interpolation="lanczos")
        ax.set_title(f"T88 Processed MRI — {title}", fontsize=11)
        ax.axis("off")

    if fseg_data is not None:
        fseg_data = np.squeeze(fseg_data)
        fsx, fsy, fsz = fseg_data.shape
        f_sag = fseg_data[fsx // 2, :, :]
        f_cor = fseg_data[:, fsy // 2, :]
        f_axi = fseg_data[:, :, fsz // 2]
        cmap = plt.cm.get_cmap("Set1", 4)
        for ax, sl, title in zip(axes[1],
                                  [f_sag, f_cor, f_axi],
                                  ["Sagittal", "Coronal", "Axial"]):
            ax.imshow(np.rot90(sl), cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
            ax.set_title(f"FSL Segmentation — {title}", fontsize=11)
            ax.axis("off")

        # Add colorbar legend
        import matplotlib.patches as mpatches
        colors = [cmap(i) for i in range(4)]
        patches = [mpatches.Patch(color=colors[i], label=FSEG_LABELS[i]) for i in range(4)]
        fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=10,
                   bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f"OASIS-1 Subject: {SUBJECT_ID}\n"
                 "T1-weighted MPRAGE brain MRI (T88 atlas space)",
                 fontsize=13, fontweight="bold")

    out_path = OUTPUT_DIR / f"{SUBJECT_ID}_visualization.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [VIZ] Saved → {out_path}")


# ---------------------------------------------------------------------------
# SDK demo
# ---------------------------------------------------------------------------
def sdk_region_demo(hdr_path: Path) -> dict | None:
    if not HAS_SDK:
        return None
    print(f"\n  [SDK] Loading via NiftiImageWrapper: {hdr_path.name}")
    try:
        wrapper = NiftiImageWrapper(str(hdr_path))
        w, h = wrapper.dimensions
        region = wrapper.read_region((0, 0), 0, (w, h), as_array=True)
        info = {
            "wrapper_class":   type(wrapper).__name__,
            "dimensions_wh":   list(wrapper.dimensions),
            "level_count":     wrapper.level_count,
            "level_dimensions": [list(d) for d in wrapper.level_dimensions],
            "region_shape":    list(region.shape),
            "properties":      dict(wrapper.properties),
        }
        print(f"    Dimensions : {info['dimensions_wh']}")
        print(f"    Levels     : {info['level_count']}")
        print(f"    Region arr : {info['region_shape']}")
        wrapper.close()
        return info
    except Exception as e:
        print(f"    [WARN] NiftiImageWrapper failed: {e}")
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# FSL segmentation analysis
# ---------------------------------------------------------------------------
def analyse_segmentation(fseg_path: Path) -> dict:
    """Load FSL tissue segmentation and compute per-label volume statistics."""
    if not fseg_path.exists():
        return {"error": f"FSL segmentation not found: {fseg_path}"}

    img = nib.load(str(fseg_path))
    data = np.asarray(img.get_fdata(), dtype=int)
    zooms = img.header.get_zooms()
    voxel_vol_mm3 = float(np.prod(zooms[:3]))

    label_stats: dict = {}
    for label_id, label_name in FSEG_LABELS.items():
        mask = data == label_id
        voxels = int(mask.sum())
        vol_cm3 = round(voxels * voxel_vol_mm3 / 1000.0, 4)
        label_stats[label_name] = {"voxels": voxels, "volume_cm3": vol_cm3}
        print(f"    {label_name:<15}: {voxels:>8,} voxels  ({vol_cm3:.2f} cm³)")

    return {
        "voxel_size_mm": [round(float(z), 4) for z in zooms[:3]],
        "voxel_vol_mm3": round(voxel_vol_mm3, 4),
        "tissue_volumes": label_stats,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("\n" + "="*60)
    print("  OASIS-1 Brain MRI Dataset — Sample Probe")
    print(f"  Subject  : {SUBJECT_ID}")
    print("  Modality : T1-weighted MPRAGE (3D structural brain MRI)")
    print("  Format   : Analyze 7.5 (.hdr/.img)")
    print("  Domain   : Radiology / Neuroscience")
    print("="*60)

    report: dict = {
        "dataset":   "OASIS-1 Cross-Sectional MRI",
        "subject":   SUBJECT_ID,
        "modality":  "T1-weighted MPRAGE (3D structural brain MRI)",
        "format":    "Analyze 7.5 (.hdr/.img)",
        "domain":    "Radiology / Neuroscience",
        "clinical_context": (
            "436-subject cross-sectional brain MRI study (Washington University). "
            "CDR 0=normal, 0.5=very mild dementia, 1=mild, 2=moderate. "
            "OAS1_0001_MR1: Female, 74y, CDR=0, MMSE=29 — cognitively normal."
        ),
    }

    # -- Metadata -----------------------------------------------------------
    print("\n--- Subject Metadata ---")
    meta = parse_metadata_txt(METADATA_TXT)
    report["subject_metadata"] = meta
    for k, v in meta.items():
        print(f"  {k:<12}: {v}")

    # -- RAW volume ---------------------------------------------------------
    print("\n--- RAW Acquire Volume (mpr-1) ---")
    if RAW_HDR.exists():
        raw_img = nib.load(str(RAW_HDR))
        raw_data = raw_img.get_fdata()
        raw_hdr_info = analyze_header(raw_img)
        raw_stats = volume_stats(raw_data, "RAW mpr-1")
        report["raw_volume"] = {"header": raw_hdr_info, "stats": raw_stats}
    else:
        print(f"  [WARN] RAW HDR not found: {RAW_HDR}")

    # -- Processed T88 volume -----------------------------------------------
    print("\n--- Processed T88 Atlas Volume ---")
    proc_data = None
    if PROCESSED_HDR.exists():
        proc_img = nib.load(str(PROCESSED_HDR))
        proc_data = proc_img.get_fdata()
        proc_hdr_info = analyze_header(proc_img)
        proc_stats = volume_stats(proc_data, "T88 Masked GFC")
        report["processed_volume"] = {"header": proc_hdr_info, "stats": proc_stats}

        # SDK demo on processed volume
        sdk_info = sdk_region_demo(PROCESSED_HDR)
        if sdk_info:
            report["sdk_nifti_wrapper"] = sdk_info
    else:
        print(f"  [WARN] Processed HDR not found: {PROCESSED_HDR}")

    # -- FSL Segmentation ---------------------------------------------------
    print("\n--- FSL Tissue Segmentation (GM / WM / CSF) ---")
    fseg_data = None
    if FSEG_HDR.exists():
        fseg_img = nib.load(str(FSEG_HDR))
        fseg_data = np.asarray(fseg_img.get_fdata(), dtype=int)
        seg_stats = analyse_segmentation(FSEG_HDR)
        report["fsl_segmentation"] = seg_stats

        # Also parse the FSL stats text if present
        fseg_txt_stats = parse_fseg_txt(FSEG_TXT)
        if fseg_txt_stats:
            report["fsl_segmentation"]["fseg_txt"] = fseg_txt_stats
    else:
        print(f"  [WARN] FSL segmentation not found: {FSEG_HDR}")

    # -- Visualization ------------------------------------------------------
    if proc_data is not None:
        make_visualization(proc_data, fseg_data)
    elif RAW_HDR.exists():
        make_visualization(raw_data, fseg_data)

    # -- Save report --------------------------------------------------------
    report_path = OUTPUT_DIR / f"{SUBJECT_ID}_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[REPORT] Saved → {report_path}")
    print("\n[DONE]")


if __name__ == "__main__":
    main()
