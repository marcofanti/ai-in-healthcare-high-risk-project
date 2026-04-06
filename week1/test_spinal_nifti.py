"""
Test program: Spinal Multiple Myeloma — NIfTI Segmentation Masks
Modality   : Spectral CT derived segmentation (spine anatomy + lesion masks)
Domain     : Radiology / Oncology (Multiple Myeloma)
Format     : NIfTI (.nii) — uncompressed
Tools      : NiftiImageWrapper (TissueLab-SDK), nibabel (full header + volume),
             numpy, matplotlib, pandas (metadata.csv)

Sample     : Myel_001
  - Myel_001_spine_segmentation.nii   — vertebra-level spine anatomy mask
  - Myel_001_lesions_segmentation.nii — lesion (tumour) mask

What this program extracts:
  - NIfTI header metadata (voxel dimensions, orientation, data type, affine)
  - Per-label voxel counts and volume (cm³) for each mask
  - Unique label values (vertebra IDs for spine mask, lesion IDs for lesion mask)
  - NiftiImageWrapper region read (SDK interface demo — sagittal mid-slice)
  - Three-plane visualizations: spine mask + lesion mask overlay
  - Spatial co-registration check (same FOV between masks)
  - metadata.csv row for Myel_001
  - JSON + PNG report output
"""

import json
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
    import matplotlib.colors as mcolors
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not installed — skipping visualizations")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from tissuelab_sdk.wrapper import NiftiImageWrapper
    HAS_SDK = True
except ImportError:
    HAS_SDK = False
    print("[WARN] TissueLab-SDK not installed — skipping NiftiImageWrapper demo")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PATIENT_ID   = "Myel_001"
DATA_DIR     = Path(__file__).parent / "data" / "Spinal_NIfTI" / PATIENT_ID
METADATA_CSV = Path(__file__).parent / "data" / "Spinal_NIfTI" / "metadata.csv"
OUTPUT_DIR   = Path(__file__).parent / "output" / "Spinal_NIfTI"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPINE_NII   = DATA_DIR / f"{PATIENT_ID}_spine_segmentation.nii"
LESION_NII  = DATA_DIR / f"{PATIENT_ID}_lesions_segmentation.nii"

# ---------------------------------------------------------------------------
# Header info
# ---------------------------------------------------------------------------
def nifti_header_info(img: nib.Nifti1Image, label: str) -> dict:
    hdr   = img.header
    zooms = hdr.get_zooms()
    shape = hdr.get_data_shape()
    info  = {
        "label":            label,
        "nifti_class":      type(img).__name__,
        "shape":            list(shape),
        "voxel_size_mm":    [round(float(z), 4) for z in zooms[:3]],
        "voxel_vol_mm3":    round(float(np.prod(zooms[:3])), 4),
        "data_dtype":       str(hdr.get_data_dtype()),
        "affine":           img.affine.tolist(),
        "qform_code":       int(hdr.get("qform_code", 0)),
        "sform_code":       int(hdr.get("sform_code", 0)),
        "description":      str(hdr.get("descrip", b"")).strip(),
    }
    print(f"\n  [{label}]")
    print(f"    Shape          : {info['shape']}")
    print(f"    Voxel size     : {info['voxel_size_mm']} mm")
    print(f"    Voxel volume   : {info['voxel_vol_mm3']:.4f} mm³")
    print(f"    Data dtype     : {info['data_dtype']}")
    return info


# ---------------------------------------------------------------------------
# Label analysis
# ---------------------------------------------------------------------------
def label_analysis(data: np.ndarray, mask_name: str, voxel_vol_mm3: float) -> dict:
    """Compute per-label statistics (voxel count, volume) for a segmentation mask."""
    unique_labels = sorted(np.unique(data).astype(int).tolist())
    label_stats: dict = {}
    print(f"\n  [{mask_name}] — unique labels: {unique_labels}")

    for lbl in unique_labels:
        voxels  = int((data == lbl).sum())
        vol_mm3 = round(voxels * voxel_vol_mm3, 2)
        vol_cm3 = round(vol_mm3 / 1000.0, 4)
        label_stats[str(lbl)] = {
            "voxels":   voxels,
            "vol_mm3":  vol_mm3,
            "vol_cm3":  vol_cm3,
        }
        if lbl != 0:  # skip background from printout
            tag = "Lesion" if mask_name == "Lesion" else f"Vertebra {lbl}"
            print(f"    Label {lbl:>3} ({tag:<16}): {voxels:>8,} voxels  "
                  f"({vol_mm3:>10.1f} mm³ / {vol_cm3:.4f} cm³)")

    bg = label_stats.get("0", {})
    fg_voxels = sum(v["voxels"] for k, v in label_stats.items() if k != "0")
    total     = int(data.size)
    print(f"    Background voxels : {bg.get('voxels', 0):>8,}")
    print(f"    Foreground voxels : {fg_voxels:>8,}  ({100*fg_voxels/total:.2f}% of volume)")

    return {
        "unique_labels":    unique_labels,
        "n_foreground_labels": len(unique_labels) - (1 if 0 in unique_labels else 0),
        "foreground_voxels": fg_voxels,
        "total_voxels":     total,
        "foreground_pct":   round(100 * fg_voxels / total, 4),
        "per_label":        label_stats,
    }


# ---------------------------------------------------------------------------
# SDK demo
# ---------------------------------------------------------------------------
def sdk_demo(nii_path: Path, label: str) -> dict | None:
    if not HAS_SDK:
        return None
    print(f"\n  [SDK] NiftiImageWrapper — {label}")
    try:
        wrapper = NiftiImageWrapper(str(nii_path))
        w, h    = wrapper.dimensions
        region  = wrapper.read_region((0, 0), 0, (w, h), as_array=True)
        info = {
            "wrapper_class":    type(wrapper).__name__,
            "dimensions_wh":    list(wrapper.dimensions),
            "level_count":      wrapper.level_count,
            "level_dimensions": [list(d) for d in wrapper.level_dimensions],
            "region_shape":     list(region.shape),
            "properties":       dict(wrapper.properties),
        }
        print(f"    Dimensions  : {info['dimensions_wh']}")
        print(f"    Levels      : {info['level_count']}")
        print(f"    Region arr  : {info['region_shape']}")
        wrapper.close()
        return info
    except Exception as e:
        print(f"    [WARN] Error: {e}")
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def make_visualization(
    spine_data: np.ndarray,
    lesion_data: np.ndarray,
    spine_vox: list[float],
) -> None:
    if not HAS_MPL:
        return

    sx, sy, sz = spine_data.shape

    # Mid-slices
    sag_s = spine_data[sx // 2, :, :]
    cor_s = spine_data[:, sy // 2, :]
    axi_s = spine_data[:, :, sz // 2]

    sag_l = lesion_data[sx // 2, :, :]
    cor_l = lesion_data[:, sy // 2, :]
    axi_l = lesion_data[:, :, sz // 2]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Spine mask colormap (one color per vertebra label)
    n_spine_labels = int(spine_data.max()) + 1
    spine_cmap = plt.cm.get_cmap("tab20", max(n_spine_labels, 2))

    for col, (sag, cor, axi, title_suffix) in enumerate([
        (sag_s, cor_s, axi_s, "Spine mask"),
        (sag_l, cor_l, axi_l, "Lesion mask"),
    ]):
        pass  # layout handled below
    # rows = Spine/Lesion, cols = Sag/Cor/Axi
    planes_spine  = [sag_s, cor_s, axi_s]
    planes_lesion = [sag_l, cor_l, axi_l]
    plane_names   = ["Sagittal (mid)", "Coronal (mid)", "Axial (mid)"]

    for col in range(3):
        # Row 0 — spine
        ax = axes[0, col]
        im = ax.imshow(np.rot90(planes_spine[col]),
                       cmap=spine_cmap, vmin=0, vmax=n_spine_labels,
                       interpolation="nearest")
        ax.set_title(f"Spine mask — {plane_names[col]}", fontsize=10)
        ax.axis("off")

        # Row 1 — lesion
        ax = axes[1, col]
        ax.imshow(np.rot90(planes_spine[col]),
                  cmap="gray_r", vmin=0, vmax=n_spine_labels,
                  interpolation="nearest", alpha=0.4)
        ax.imshow(np.rot90(planes_lesion[col]),
                  cmap="hot", vmin=0, vmax=max(1, int(lesion_data.max())),
                  interpolation="nearest", alpha=0.9)
        ax.set_title(f"Lesion mask overlay — {plane_names[col]}", fontsize=10)
        ax.axis("off")

    fig.suptitle(
        f"Spinal Multiple Myeloma — {PATIENT_ID}\n"
        "NIfTI Segmentation Masks: Spine Anatomy + Lesion",
        fontsize=13, fontweight="bold"
    )

    out_path = OUTPUT_DIR / f"{PATIENT_ID}_NIfTI_segmentation.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [VIZ] Saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("\n" + "="*60)
    print("  Spinal Multiple Myeloma — NIfTI Segmentation Probe")
    print(f"  Patient  : {PATIENT_ID}")
    print("  Masks    : spine_segmentation + lesions_segmentation")
    print("  Modality : Spectral CT derived masks (NIfTI)")
    print("  Domain   : Radiology / Oncology")
    print("="*60)

    report: dict = {
        "dataset":   "Spinal-Multiple-Myeloma-SEG",
        "patient":   PATIENT_ID,
        "modality":  "NIfTI Segmentation Masks (Spectral CT derived)",
        "domain":    "Radiology / Oncology",
        "clinical_context": (
            "Expert-refined nnU-Net v2 segmentations. "
            "Spine mask labels = vertebra IDs (C1–L5 etc.). "
            "Lesion mask labels = Multiple Myeloma lesion foci. "
            "Source CT: Philips IQon DECT, spine + lesion annotations for 67 patients."
        ),
    }

    # -- Spine mask ---------------------------------------------------------
    print("\n--- Spine Segmentation Mask ---")
    if not SPINE_NII.exists():
        print(f"  [ERROR] Not found: {SPINE_NII}")
        sys.exit(1)

    spine_img  = nib.load(str(SPINE_NII))
    spine_hdr  = nifti_header_info(spine_img, "Spine mask")
    spine_data = np.asarray(spine_img.get_fdata(), dtype=int)
    spine_vox  = spine_hdr["voxel_size_mm"]
    spine_vv   = spine_hdr["voxel_vol_mm3"]

    spine_labels = label_analysis(spine_data, "Spine", spine_vv)
    report["spine_mask"] = {"header": spine_hdr, "label_analysis": spine_labels}

    # SDK demo — spine
    sdk_spine = sdk_demo(SPINE_NII, "Spine mask")
    if sdk_spine:
        report["sdk_nifti_wrapper_spine"] = sdk_spine

    # -- Lesion mask --------------------------------------------------------
    print("\n--- Lesion Segmentation Mask ---")
    if not LESION_NII.exists():
        print(f"  [ERROR] Not found: {LESION_NII}")
        sys.exit(1)

    lesion_img  = nib.load(str(LESION_NII))
    lesion_hdr  = nifti_header_info(lesion_img, "Lesion mask")
    lesion_data = np.asarray(lesion_img.get_fdata(), dtype=int)
    lesion_vv   = lesion_hdr["voxel_vol_mm3"]

    lesion_labels = label_analysis(lesion_data, "Lesion", lesion_vv)
    report["lesion_mask"] = {"header": lesion_hdr, "label_analysis": lesion_labels}

    # SDK demo — lesion
    sdk_lesion = sdk_demo(LESION_NII, "Lesion mask")
    if sdk_lesion:
        report["sdk_nifti_wrapper_lesion"] = sdk_lesion

    # -- Co-registration check ----------------------------------------------
    print("\n--- Co-registration Check ---")
    same_shape  = spine_data.shape == lesion_data.shape
    same_affine = np.allclose(spine_img.affine, lesion_img.affine, atol=1e-3)
    print(f"  Same shape   : {same_shape}  {spine_data.shape} vs {lesion_data.shape}")
    print(f"  Same affine  : {same_affine}")
    report["coregistration"] = {
        "same_shape":  same_shape,
        "same_affine": same_affine,
    }

    # -- Metadata CSV -------------------------------------------------------
    if HAS_PANDAS and METADATA_CSV.exists():
        df  = pd.read_csv(METADATA_CSV)
        rows = df[df["Subject ID"] == PATIENT_ID].to_dict(orient="records")
        print(f"\n--- metadata.csv: {len(rows)} series rows for {PATIENT_ID} ---")
        for row in rows[:3]:
            print(f"  {row.get('Series Description','?'):<35}  "
                  f"images={row.get('Number of Images','?')}")
        report["metadata_csv_rows"] = rows

    # -- Visualization ------------------------------------------------------
    make_visualization(spine_data, lesion_data, spine_vox)

    # -- Save report --------------------------------------------------------
    report_path = OUTPUT_DIR / f"{PATIENT_ID}_NIfTI_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[REPORT] Saved → {report_path}")
    print("\n[DONE]")


if __name__ == "__main__":
    main()
