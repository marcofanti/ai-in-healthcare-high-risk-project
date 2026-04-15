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
