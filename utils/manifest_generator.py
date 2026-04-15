import os
import json

def generate_manifest(directory_path: str) -> str:
    """
    Recursively scans the local directory and generates a lightweight JSON manifest.
    Example output format:
    [
        {"file_path": "./workspace/OAS1_MR1.img", "type": "Legacy MRI", "size": "15MB"},
        ...
    ]
    """
    manifest = []
    
    if not os.path.exists(directory_path):
        return json.dumps(manifest)

    for root, _, files in os.walk(directory_path):
        for file in files:
            # Skip hidden files
            if file.startswith('.'):
                continue
            
            file_path = os.path.join(root, file)
            size_bytes = os.path.getsize(file_path)
            
            # Simple conversion to MB for readability, minimum 1MB text if size > 0
            size_mb = size_bytes / (1024 * 1024)
            size_str = f"{size_mb:.2f}MB" if size_mb >= 0.01 else f"{size_bytes}B"
            
            # Simple heuristic for type based on extension
            file_type = "Unknown"
            if file.endswith('.img') or file.endswith('.hdr'):
                file_type = "Legacy MRI"
            elif file.endswith('.jpg') or file.endswith('.tif'):
                file_type = "2D Histopathology"
            elif file.endswith('.nii') or file.endswith('.nii.gz'):
                file_type = "3D Volumetric"
            elif file.endswith('.raw') or file.endswith('.hdr') and 'hsi' in file.lower():
                file_type = "Hyperspectral"
            elif file.endswith('.xml'):
                file_type = "Metadata XML"
                
            manifest.append({
                "file_path": file_path,
                "type": file_type,
                "size": size_str
            })
            
    return json.dumps(manifest, indent=2)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(generate_manifest(sys.argv[1]))
    else:
        print("Usage: python manifest_generator.py <directory_path>")
