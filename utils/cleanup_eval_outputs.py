import json
import os
import shutil
from pathlib import Path

# Mapping of old 4th-level folder names to new ones
FOLDER_RENAMES = {
    "Atlas_test": "Atlas_All",
    "Atlas_test_tiny": "Atlas_Tiny",
    "EduContent_test": "EduContent_All",
    "EduContent_test_tiny": "EduContent_Tiny",
    "PathCLS_test": "PathCLS_All",
    "PathCLS_test_tiny": "PathCLS_Tiny",
    "PubMed_test_tiny": "PubMed_Tiny",
}

def cleanup_outputs():
    outputs_base = Path("eval/outputs")
    if not outputs_base.exists():
        print(f"[ERROR] Directory {outputs_base} not found.")
        return

    # 1. Rename folders
    print("--- Renaming Folders ---")
    for parent_dir in outputs_base.iterdir():
        if not parent_dir.is_dir():
            continue
            
        for old_name, new_name in FOLDER_RENAMES.items():
            old_path = parent_dir / old_name
            new_path = parent_dir / new_name
            
            if old_path.exists() and old_path.is_dir():
                # If target exists, merge or skip? For cleanup, we move.
                if new_path.exists():
                    print(f"[WARN] Target {new_path} already exists. Skipping rename for {old_path}.")
                else:
                    print(f"Renaming: {old_path} -> {new_path}")
                    shutil.move(str(old_path), str(new_path))

    # 2. Update JSON files
    print("\n--- Updating summary.json files ---")
    summary_files = list(outputs_base.glob("**/summary.json"))
    for f in summary_files:
        try:
            with open(f, 'r') as jf:
                data = json.load(jf)
            
            modified = False
            folder_name = f.parent.name # e.g. Atlas_All
            
            # Update exp_name
            if data.get("exp_name") != folder_name:
                print(f"Updating exp_name in {f}: {data.get('exp_name')} -> {folder_name}")
                data["exp_name"] = folder_name
                modified = True
            
            # Update category keys in categories object
            # We look for any keys that match the 'old' names and update them to the folder name
            categories = data.get("categories", {})
            new_categories = {}
            for k, v in categories.items():
                if k in FOLDER_RENAMES:
                    new_key = FOLDER_RENAMES[k]
                    print(f"Updating category key in {f}: {k} -> {new_key}")
                    new_categories[new_key] = v
                    modified = True
                else:
                    new_categories[k] = v
            data["categories"] = new_categories

            if modified:
                with open(f, 'w') as jf:
                    json.dump(data, jf, indent=4)
                print(f"Saved: {f}")

        except Exception as e:
            print(f"Error processing {f}: {e}")

if __name__ == "__main__":
    cleanup_outputs()
