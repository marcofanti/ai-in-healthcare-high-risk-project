"""
Test program: Spinal Multiple Myeloma — DICOM (Spectral CT)
Modality   : Spectral/Dual-Energy CT (Spine)
Domain     : Radiology (Oncology — Multiple Myeloma)
Format     : DICOM (.dcm) — MonoE 80 keV reconstruction, Myel_001
Tools      : DicomImageWrapper (TissueLab-SDK), pydicom (full tag dump),
             SimpleITK (3D series reconstruction), pandas (metadata.csv),
             matplotlib, numpy

What this program extracts:
  - Full DICOM tag dump of the first slice (pydicom)
  - Key clinical DICOM tags (patient, scanner, acquisition, reconstruction)
  - DicomImageWrapper region read (SDK interface demo)
  - SimpleITK 3D volume reconstruction from the series (shape, voxel spacing, HU range)
  - Three-plane volume visualisation (axial / coronal / sagittal)
  - Intensity statistics and HU histogram
  - metadata.csv row for Myel_001
  - JSON + PNG report output

Note: Only 20 DICOM slices are copied to week1/data for portability.
      The full 1095-slice series lives in the original Spinal dataset.
"""

import json
import sys
from pathlib import Path

import numpy as np

try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    print("[ERROR] pydicom is required: pip install pydicom")
    sys.exit(1)

try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False
    print("[WARN] SimpleITK not installed — 3D reconstruction unavailable")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("[WARN] pandas not installed — metadata.csv lookup unavailable")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not available — skipping visualizations")

try:
    from tissuelab_sdk.wrapper import DicomImageWrapper
    HAS_SDK = True
except ImportError:
    HAS_SDK = False
    print("[WARN] TissueLab-SDK not installed — skipping DicomImageWrapper demo")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PATIENT_ID   = "Myel_001"
SERIES_NAME  = "MonoE_80keVHU"

DATA_DIR     = Path(__file__).parent / "data" / "Spinal_DICOM"
SERIES_DIR   = DATA_DIR / PATIENT_ID / SERIES_NAME
METADATA_CSV = DATA_DIR / "metadata.csv"
OUTPUT_DIR   = Path(__file__).parent / "output" / "Spinal_DICOM"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Full-series path (for SimpleITK — if the full copy exists)
FULL_SERIES_DIR = Path(
    "/Volumes/ExternalOwc/AI_For_Healthcare/Final_Project/Datasets"
    "/Spinal/manifest-1774389300184/Spinal-Multiple-Myeloma-SEG"
    f"/Myel_001/01-06-2015-8006-NA-94443/20781.000000-MonoE 80keVHU-77135"
)

# ---------------------------------------------------------------------------
# DICOM tag extraction
# ---------------------------------------------------------------------------
CLINICAL_TAGS = {
    "PatientID":                "0010,0020",
    "StudyDate":                "0008,0020",
    "Modality":                 "0008,0060",
    "SeriesDescription":        "0008,103e",
    "Manufacturer":             "0008,0070",
    "ManufacturerModelName":    "0008,1090",
    "KVP":                      "0018,0060",
    "SliceThickness":           "0018,0050",
    "PixelSpacing":             "0028,0030",
    "Rows":                     "0028,0010",
    "Columns":                  "0028,0011",
    "BitsAllocated":            "0028,0100",
    "RescaleIntercept":         "0028,1052",
    "RescaleSlope":             "0028,1053",
    "ImagePositionPatient":     "0020,0032",
    "ImageOrientationPatient":  "0020,0037",
    "SliceLocation":            "0020,1041",
    "InstanceNumber":           "0020,0013",
    "PhotometricInterpretation":"0028,0004",
    "WindowCenter":             "0028,1050",
    "WindowWidth":              "0028,1051",
}


def extract_tag_value(ds: pydicom.Dataset, tag_name: str) -> str:
    """Safely extract a DICOM tag value by keyword name."""
    try:
        val = getattr(ds, tag_name, None)
        if val is None:
            return "N/A"
        if hasattr(val, "__iter__") and not isinstance(val, str):
            return str(list(val))
        return str(val)
    except Exception:
        return "N/A"


def dump_all_tags(ds: pydicom.Dataset) -> dict:
    """Return a dict of all non-pixel-data DICOM elements."""
    tags = {}
    for elem in ds:
        if elem.keyword == "PixelData":
            continue
        try:
            tags[elem.keyword or str(elem.tag)] = str(elem.value)[:200]
        except Exception:
            tags[str(elem.tag)] = "<unreadable>"
    return tags


def to_hounsfield(pixel_array: np.ndarray, ds: pydicom.Dataset) -> np.ndarray:
    """Apply RescaleSlope and RescaleIntercept to get HU values."""
    slope     = float(getattr(ds, "RescaleSlope",     1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    return pixel_array.astype(float) * slope + intercept


# ---------------------------------------------------------------------------
# Metadata CSV
# ---------------------------------------------------------------------------
def lookup_metadata_csv(csv_path: Path, subject_id: str) -> list[dict]:
    if not HAS_PANDAS or not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    matches = df[df["Subject ID"] == subject_id]
    return matches.to_dict(orient="records")


# ---------------------------------------------------------------------------
# SimpleITK 3D reconstruction
# ---------------------------------------------------------------------------
def sitk_load_series(series_dir: Path) -> dict | None:
    if not HAS_SITK:
        return None
    if not series_dir.exists():
        print(f"  [WARN] Series directory not found for SimpleITK: {series_dir}")
        return None

    print(f"\n  [SimpleITK] Reading series from: {series_dir}")
    reader = sitk.ImageSeriesReader()
    dcm_names = reader.GetGDCMSeriesFileNames(str(series_dir))
    if not dcm_names:
        print("  [WARN] No DICOM files found in series directory")
        return None

    print(f"  Found {len(dcm_names)} DICOM files")
    reader.SetFileNames(dcm_names)
    image = reader.Execute()

    size    = image.GetSize()          # (x, y, z)
    spacing = image.GetSpacing()       # mm per voxel (x, y, z)
    origin  = image.GetOrigin()
    direction = image.GetDirection()

    arr = sitk.GetArrayFromImage(image)  # shape: (z, y, x)
    print(f"  Volume shape (z,y,x) : {arr.shape}")
    print(f"  Voxel spacing (mm)   : {[round(s, 4) for s in spacing]}")
    print(f"  HU range             : {arr.min():.1f} – {arr.max():.1f}")

    return {
        "sitk_size_xyz":    list(size),
        "volume_shape_zyx": list(arr.shape),
        "voxel_spacing_mm": [round(float(s), 4) for s in spacing],
        "origin_mm":        [round(float(o), 4) for o in origin],
        "n_files_loaded":   len(dcm_names),
        "hu_min":           float(arr.min()),
        "hu_max":           float(arr.max()),
        "hu_mean":          round(float(arr.mean()), 2),
        "hu_std":           round(float(arr.std()), 2),
        "array":            arr,   # not serialised; used for viz
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def make_visualization(
    first_slice_hu: np.ndarray,
    volume_info: dict | None,
    all_hu: list[np.ndarray],
) -> None:
    if not HAS_MPL:
        return

    n_panels = 3 if volume_info else 2
    fig, axes = plt.subplots(1, n_panels + 1, figsize=(5 * (n_panels + 1), 5))

    # Panel 0: First DICOM slice (windowed for soft tissue)
    wc, ww = 40, 400  # typical soft tissue window
    vmin, vmax = wc - ww / 2, wc + ww / 2
    axes[0].imshow(first_slice_hu, cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")
    axes[0].set_title(f"Slice 1 — MonoE 80 keV\n(windowed W={ww}/L={wc})", fontsize=10)
    axes[0].axis("off")

    if volume_info and "array" in volume_info:
        vol = volume_info["array"]
        nz, ny, nx = vol.shape
        # Axial (mid)
        axes[1].imshow(vol[nz // 2, :, :], cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")
        axes[1].set_title(f"Axial mid-slice\n({nz} total slices)", fontsize=10)
        axes[1].axis("off")
        # Coronal (mid)
        axes[2].imshow(vol[:, ny // 2, :], cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")
        axes[2].set_title("Coronal mid-slice", fontsize=10)
        axes[2].axis("off")

    # Last panel: HU histogram
    ax_hist = axes[-1]
    if all_hu:
        hu_flat = np.concatenate([s.flatten() for s in all_hu])
        ax_hist.hist(hu_flat, bins=80, color="steelblue", edgecolor="none", alpha=0.8)
        ax_hist.set_xlabel("HU value", fontsize=9)
        ax_hist.set_ylabel("Voxel count", fontsize=9)
        ax_hist.set_title("HU distribution\n(all loaded slices)", fontsize=10)
        ax_hist.axvline(-100, color="red",   linestyle="--", alpha=0.7, label="~Fat")
        ax_hist.axvline(  40, color="green", linestyle="--", alpha=0.7, label="~Soft tissue")
        ax_hist.axvline( 400, color="orange",linestyle="--", alpha=0.7, label="~Bone")
        ax_hist.legend(fontsize=7)
        ax_hist.grid(True, alpha=0.3)

    fig.suptitle(
        f"Spinal Multiple Myeloma — {PATIENT_ID}\n"
        "Spectral CT (MonoE 80 keV) — Spine + Lesion imaging",
        fontsize=13, fontweight="bold"
    )
    out_path = OUTPUT_DIR / f"{PATIENT_ID}_DICOM_analysis.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [VIZ] Saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("\n" + "="*60)
    print("  Spinal Multiple Myeloma — DICOM Probe")
    print(f"  Patient  : {PATIENT_ID}")
    print(f"  Series   : {SERIES_NAME}")
    print("  Modality : Spectral CT (Dual-Energy) — Spine")
    print("  Domain   : Radiology / Oncology")
    print("="*60)

    report: dict = {
        "dataset":   "Spinal-Multiple-Myeloma-SEG",
        "patient":   PATIENT_ID,
        "series":    SERIES_NAME,
        "modality":  "Spectral CT (Dual-Energy) — Spine",
        "domain":    "Radiology / Oncology",
        "clinical_context": (
            "67-patient DECT spine dataset (Philips IQon). Multiple Myeloma lesion "
            "segmentation task. MonoE 80 keV is a standard soft-tissue equivalent "
            "reconstruction. Lesion and spine NIfTI masks available separately."
        ),
    }

    # -- Find DICOM slices --------------------------------------------------
    dcm_files = sorted(SERIES_DIR.glob("*.dcm"))
    if not dcm_files:
        print(f"  [ERROR] No .dcm files found in {SERIES_DIR}")
        sys.exit(1)

    print(f"\n  Found {len(dcm_files)} DICOM slice(s) in local copy ({SERIES_DIR.name})")
    report["local_slice_count"] = len(dcm_files)

    # -- Load first slice with pydicom --------------------------------------
    first_dcm = dcm_files[0]
    print(f"\n--- pydicom: First Slice ({first_dcm.name}) ---")
    ds = pydicom.dcmread(str(first_dcm))

    all_tags = dump_all_tags(ds)
    report["all_dicom_tags_slice1"] = all_tags

    print("\n  Key Clinical DICOM Tags:")
    clinical = {}
    for name in CLINICAL_TAGS:
        val = extract_tag_value(ds, name)
        clinical[name] = val
        print(f"    {name:<30}: {val}")
    report["clinical_tags"] = clinical

    # HU conversion for the first slice
    pixel_arr = ds.pixel_array
    hu_slice  = to_hounsfield(pixel_arr, ds)
    print(f"\n  Pixel array shape : {pixel_arr.shape}")
    print(f"  HU range          : {hu_slice.min():.1f} – {hu_slice.max():.1f}")
    print(f"  HU mean / std     : {hu_slice.mean():.2f} / {hu_slice.std():.2f}")
    report["first_slice_stats"] = {
        "pixel_shape": list(pixel_arr.shape),
        "hu_min":      round(float(hu_slice.min()), 2),
        "hu_max":      round(float(hu_slice.max()), 2),
        "hu_mean":     round(float(hu_slice.mean()), 2),
        "hu_std":      round(float(hu_slice.std()), 2),
    }

    # -- SDK DicomImageWrapper demo -----------------------------------------
    all_hu_slices = [hu_slice]
    if HAS_SDK:
        print(f"\n  [SDK] Loading via DicomImageWrapper: {first_dcm.name}")
        try:
            wrapper = DicomImageWrapper(str(first_dcm))
            w, h    = wrapper.dimensions
            region  = wrapper.read_region((0, 0), 0, (w, h), as_array=True)
            sdk_info = {
                "wrapper_class":     type(wrapper).__name__,
                "dimensions_wh":     list(wrapper.dimensions),
                "properties":        dict(wrapper.properties),
                "region_shape":      list(region.shape),
            }
            report["sdk_dicom_wrapper"] = sdk_info
            print(f"    Dimensions : {sdk_info['dimensions_wh']}")
            print(f"    Properties : {sdk_info['properties']}")
            wrapper.close()
        except Exception as e:
            print(f"    [WARN] DicomImageWrapper error: {e}")
            report["sdk_dicom_wrapper"] = {"error": str(e)}

    # -- Load all local slices for histogram --------------------------------
    print(f"\n--- Loading all {len(dcm_files)} local DICOM slices ---")
    for dcm_path in dcm_files[1:]:
        ds_i = pydicom.dcmread(str(dcm_path))
        hu_i = to_hounsfield(ds_i.pixel_array, ds_i)
        all_hu_slices.append(hu_i)

    all_hu_arr = np.stack(all_hu_slices, axis=0)
    print(f"  Stacked local volume : {all_hu_arr.shape}  "
          f"HU={all_hu_arr.min():.0f}–{all_hu_arr.max():.0f}")
    report["local_volume_stats"] = {
        "shape":    list(all_hu_arr.shape),
        "hu_min":   float(all_hu_arr.min()),
        "hu_max":   float(all_hu_arr.max()),
        "hu_mean":  round(float(all_hu_arr.mean()), 2),
        "hu_std":   round(float(all_hu_arr.std()), 2),
    }

    # -- SimpleITK full series (uses original path) -------------------------
    volume_info = None
    if HAS_SITK:
        # Prefer full series if available; fall back to local copy
        series_for_sitk = FULL_SERIES_DIR if FULL_SERIES_DIR.exists() else SERIES_DIR
        volume_info = sitk_load_series(series_for_sitk)
        if volume_info:
            arr = volume_info.pop("array", None)    # remove non-serialisable
            report["sitk_3d_volume"] = volume_info
            volume_info["array"] = arr              # restore for viz

    # -- Metadata CSV -------------------------------------------------------
    print("\n--- metadata.csv ---")
    csv_rows = lookup_metadata_csv(METADATA_CSV, PATIENT_ID)
    if csv_rows:
        print(f"  Found {len(csv_rows)} series for {PATIENT_ID}")
        for row in csv_rows[:3]:
            print(f"    {row.get('Series Description','?'):<30}  "
                  f"images={row.get('Number of Images','?')}")
        report["metadata_csv_rows"] = csv_rows
    else:
        print("  No metadata rows found")

    # -- Visualization ------------------------------------------------------
    make_visualization(hu_slice, volume_info, all_hu_slices)

    # -- Save report --------------------------------------------------------
    report_path = OUTPUT_DIR / f"{PATIENT_ID}_DICOM_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[REPORT] Saved → {report_path}")
    print("\n[DONE]")


if __name__ == "__main__":
    main()
