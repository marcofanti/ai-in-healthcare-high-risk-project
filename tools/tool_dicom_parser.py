import os
from pathlib import Path
import json

try:
    import pydicom
except ImportError:
    pydicom = None

try:
    import nibabel as nib
except ImportError:
    nib = None

def parse_dicom(target_dir: str) -> dict:
    """Parses standard DICOM directories and returns metadata."""
    if not pydicom:
        return {"error": "pydicom not installed"}
    
    dir_path = Path(target_dir)
    dcm_files = sorted(dir_path.rglob("*.dcm"))
    if not dcm_files:
        return {"error": f"No .dcm files found in {target_dir}"}
        
    first_dcm = dcm_files[0]
    
    try:
        ds = pydicom.dcmread(str(first_dcm))
        return {
            "slice_count": len(dcm_files),
            "study_date": str(getattr(ds, "StudyDate", "Unknown")),
            "modality": str(getattr(ds, "Modality", "Unknown")),
            "rows": getattr(ds, "Rows", "Unknown"),
            "columns": getattr(ds, "Columns", "Unknown"),
            "patient_id": str(getattr(ds, "PatientID", "Unknown")),
            "note": "Successfully parsed DICOM spatial dimensions and patient info."
        }
    except Exception as e:
        return {"error": f"DICOM parsing failed: {e}"}

def parse_nifti(target_file: str) -> dict:
    """Parses standard NIfTI (.nii) files and returns metadata."""
    if not nib:
        return {"error": "nibabel not installed"}
        
    try:
        img = nib.load(target_file)
        header = img.header
        return {
            "voxel_shape": [int(x) for x in img.shape],
            "voxel_zooms": [float(x) for x in header.get_zooms()],
            "data_type": str(header.get_data_dtype()),
            "note": "Successfully parsed NIfTI robust 3D/4D dimensions."
        }
    except Exception as e:
        return {"error": f"NIfTI parsing failed: {e}"}
