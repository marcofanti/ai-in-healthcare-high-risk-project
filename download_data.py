import os
import sys

try:
    import gdown
except ImportError:
    print("Error: 'gdown' is fundamentally required. Run 'pip install gdown'")
    sys.exit(1)

def main():
    # Google Drive Folder ID provided by the project manager 
    FOLDER_ID = "17Zk6Zs_rGqOgV0pcL2MTQ4e_ArZ_zqpC"
    OUTPUT_DIR = "week1/data"

    print(f"Downloading massive dataset folder from Google Drive (ID: {FOLDER_ID})...")
    
    # Ensure the target base directory exists natively
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Downloading the exact Drive Folder natively!
    # quiet=False prints a beautiful progress bar for every file mapping.
    gdown.download_folder(id=FOLDER_ID, output=OUTPUT_DIR, quiet=False, use_cookies=False)

    print("\nData synchronization successfully completed!")

if __name__ == "__main__":
    main()
