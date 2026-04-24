import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import io
from PIL import Image

# Resilient imports for medical formats
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

try:
    import spectral.io.envi as envi
    HAS_SPECTRAL = True
except ImportError:
    HAS_SPECTRAL = False

try:
    import tiffslide
    HAS_TIFFSLIDE = True
except ImportError:
    HAS_TIFFSLIDE = False

try:
    import openslide
    HAS_OPENSLIDE = True
except ImportError:
    HAS_OPENSLIDE = False

# .mrxs requires openslide; other WSI formats use tiffslide
MRXS_SUFFIXES = {".mrxs"}
WSI_SUFFIXES = {".svs", ".ndpi", ".scn", ".vms", ".vmu", ".bif"}

def get_image_metadata(path: str, modality: str) -> Dict[str, Any]:
    """Extract technical metadata for overlay."""
    file_path = Path(path)
    if not file_path.exists():
        return {"error": "File not found"}
    
    meta = {
        "File Name": file_path.name,
        "Size": f"{os.path.getsize(file_path) / (1024*1024):.2f} MB",
        "Modality": modality
    }
    
    suffix = file_path.suffix.lower()
    
    try:
        if suffix == ".dcm" and HAS_PYDICOM:
            ds = pydicom.dcmread(str(file_path))
            meta["Dimensions"] = f"{ds.Rows}x{ds.Columns}"
            meta["Patient ID"] = getattr(ds, "PatientID", "N/A")
            meta["Study Date"] = getattr(ds, "StudyDate", "N/A")
            meta["Modality Tag"] = getattr(ds, "Modality", "N/A")
            meta["Manufacturer"] = getattr(ds, "Manufacturer", "N/A")
        
        elif suffix in (".nii", ".gz", ".img", ".hdr") and suffix != ".hdr" and HAS_NIBABEL:
            img = nib.load(str(file_path))
            header = img.header
            meta["Shape (XYZ)"] = list(img.shape)
            meta["Voxel Size"] = [round(float(x), 3) for x in header.get_zooms()]
            meta["Data Type"] = str(header.get_data_dtype())
        
        elif suffix == ".hdr" and HAS_SPECTRAL:
            img = envi.open(str(file_path))
            meta["Shape (WHB)"] = [img.shape[0], img.shape[1], img.shape[2]]
            meta["Bands"] = img.shape[2]
            meta["Interleave"] = img.metadata.get("interleave", "N/A")
            meta["Data Type"] = img.metadata.get("data type", "N/A")

        elif suffix in MRXS_SUFFIXES and HAS_OPENSLIDE:
            slide = openslide.OpenSlide(str(file_path))
            w, h = slide.dimensions
            meta["Dimensions"] = f"{w}x{h} px"
            meta["Pyramid Levels"] = slide.level_count
            meta["Objective Power"] = slide.properties.get("openslide.objective-power", "N/A")
            mpp_x = slide.properties.get("openslide.mpp-x", None)
            if mpp_x:
                meta["MPP (µm/px)"] = f"{float(mpp_x):.4f}"
            meta["Vendor"] = slide.properties.get("openslide.vendor", "N/A")
            slide.close()

        elif suffix in WSI_SUFFIXES and HAS_TIFFSLIDE:
            slide = tiffslide.TiffSlide(str(file_path))
            w, h = slide.dimensions
            meta["Dimensions"] = f"{w}x{h} px"
            meta["Pyramid Levels"] = slide.level_count
            meta["Objective Power"] = slide.properties.get("tiffslide.objective-power", "N/A")
            mpp_x = slide.properties.get("tiffslide.mpp-x", None)
            if mpp_x:
                meta["MPP (µm/px)"] = f"{float(mpp_x):.4f}"
            meta["Vendor"] = slide.properties.get("tiffslide.vendor", "N/A")
            slide.close()

        else:
            with Image.open(file_path) as img:
                meta["Dimensions"] = f"{img.width}x{img.height}"
                meta["Format"] = img.format
                meta["Mode"] = img.mode
                
    except Exception as e:
        meta["Metadata Error"] = str(e)
        
    return meta

def create_medical_viz(path: str, modality: str) -> Optional[io.BytesIO]:
    """Create a high-quality visualization matching week2/week1 tests."""
    if not HAS_MATPLOTLIB:
        return None

    file_path = Path(path)
    if not file_path.exists():
        return None
    
    suffix = file_path.suffix.lower()
    
    try:
        plt.style.use('dark_background')
        
        # 3D Orthogonal View (MRI / CT / NIfTI)
        if suffix in (".nii", ".gz", ".img", ".hdr", ".dcm") and suffix != ".hdr":
            if suffix == ".dcm" and HAS_PYDICOM:
                ds = pydicom.dcmread(str(file_path))
                vol = ds.pixel_array.astype(float)
                slope = float(getattr(ds, "RescaleSlope", 1))
                intercept = float(getattr(ds, "RescaleIntercept", 0))
                vol = vol * slope + intercept
                if vol.ndim == 2:
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.imshow(vol, cmap='bone')
                    ax.axis('off')
                    ax.set_title(f"DICOM Slice: {file_path.name}")
                else:
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    z, y, x = vol.shape[0]//2, vol.shape[1]//2, vol.shape[2]//2
                    axes[0].imshow(vol[z, :, :], cmap='bone')
                    axes[0].set_title("Axial")
                    axes[1].imshow(vol[:, y, :], cmap='bone', aspect=vol.shape[0]/vol.shape[2])
                    axes[1].set_title("Coronal")
                    axes[2].imshow(vol[:, :, x], cmap='bone', aspect=vol.shape[0]/vol.shape[1])
                    axes[2].set_title("Sagittal")
                    for ax in axes: ax.axis('off')
            elif HAS_NIBABEL:
                img = nib.load(str(file_path))
                vol = np.squeeze(img.get_fdata())
                if vol.ndim == 3:
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    x, y, z = vol.shape[0]//2, vol.shape[1]//2, vol.shape[2]//2
                    axes[0].imshow(vol[:, :, z].T, cmap='gray', origin='lower')
                    axes[0].set_title("Axial")
                    axes[1].imshow(vol[:, y, :].T, cmap='gray', origin='lower', aspect=img.header.get_zooms()[2]/img.header.get_zooms()[0])
                    axes[1].set_title("Coronal")
                    axes[2].imshow(vol[x, :, :].T, cmap='gray', origin='lower', aspect=img.header.get_zooms()[2]/img.header.get_zooms()[1])
                    axes[2].set_title("Sagittal")
                    for ax in axes: ax.axis('off')
                else:
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.imshow(vol.T, cmap='gray', origin='lower')
                    ax.axis('off')
                    ax.set_title(f"2D Slice: {file_path.name}")
            else:
                return None
                    
        # Hyperspectral (HSI) Pseudo-RGB
        elif suffix == ".hdr" and HAS_SPECTRAL:
            img = envi.open(str(file_path))
            wl_vals = np.array([float(w) for w in img.metadata["wavelength"]])
            br = int(np.argmin(np.abs(wl_vals - 650)))
            bg = int(np.argmin(np.abs(wl_vals - 550)))
            bb = int(np.argmin(np.abs(wl_vals - 450)))
            
            def _norm(band_idx):
                arr = np.squeeze(img.read_band(band_idx))
                return (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)

            rgb = np.stack([_norm(br), _norm(bg), _norm(bb)], axis=2)
            fig, axes = plt.subplots(1, 2, figsize=(15, 7))
            axes[0].imshow(rgb)
            axes[0].set_title("Pseudo-RGB (650, 550, 450 nm)")
            bnir = int(np.argmin(np.abs(wl_vals - 800)))
            axes[1].imshow(_norm(bnir), cmap='viridis')
            axes[1].set_title("NIR Band (800 nm)")
            for ax in axes: ax.axis('off')
            
        # Whole Slide Image — MRXS pyramid overview (openslide)
        elif suffix in MRXS_SUFFIXES and HAS_OPENSLIDE:
            TILE = 512
            slide = openslide.OpenSlide(str(file_path))
            n_levels = min(slide.level_count, 8)
            mpp = slide.properties.get("openslide.mpp-x", None)
            mpp_str = f"MPP {float(mpp):.4f} µm/px" if mpp else ""

            cols = min(n_levels, 4)
            rows = (n_levels + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4.5))
            axes = np.array(axes).flatten()

            for i in range(n_levels):
                w, h = slide.level_dimensions[i]
                cx = max(0, w // 2 - TILE // 2)
                cy = max(0, h // 2 - TILE // 2)
                ds = slide.level_downsamples[i]
                # read_region takes level-0 coordinates and returns RGBA
                tile = slide.read_region(
                    (int(cx * ds), int(cy * ds)), i, (TILE, TILE)
                ).convert("RGB")
                axes[i].imshow(tile)
                axes[i].set_title(
                    f"Level {i}\n{w:,}×{h:,}\n({ds:.0f}×)", fontsize=8
                )
                axes[i].axis("off")

            for j in range(n_levels, len(axes)):
                axes[j].set_visible(False)

            slide.close()
            plt.suptitle(
                f"{file_path.stem} — pyramid levels  {mpp_str}", fontsize=10
            )

        # Whole Slide Image — SVS, NDPI, SCN … (tiffslide)
        elif suffix in WSI_SUFFIXES and HAS_TIFFSLIDE:
            slide = tiffslide.TiffSlide(str(file_path))
            thumb = slide.get_thumbnail((1024, 1024))
            w, h = slide.dimensions
            obj_power = slide.properties.get("tiffslide.objective-power", "?")
            mpp = slide.properties.get("tiffslide.mpp-x", None)
            mpp_str = f"{float(mpp):.4f} µm/px" if mpp else "N/A"
            slide.close()

            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(thumb)
            ax.axis('off')
            ax.set_title(
                f"WSI Thumbnail — {file_path.name}\n"
                f"Full res: {w}×{h} px  |  {obj_power}×  |  MPP: {mpp_str}",
                fontsize=10,
            )

        # Standard 2D Image
        else:
            pil_img = Image.open(file_path).convert("RGB")
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(pil_img)
            ax.axis('off')
            ax.set_title(f"Image Preview: {file_path.name}")

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close(fig)
        buf.seek(0)
        return buf
        
    except Exception as e:
        print(f"Visualization Error: {e}")
        return None
