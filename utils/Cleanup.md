# Evaluation Output Cleanup

This document describes the `utils/cleanup_eval_outputs.py` script, which automates the standardization of evaluation output folders and metadata.

## Purpose
The `eval/main.py` script sometimes generates folder names and JSON keys that use a "test" or "test_tiny" suffix (e.g., `Atlas_test`). To maintain a consistent structure for report generation, this script renames those folders and updates their internal JSON metadata to use "All" and "Tiny" suffixes instead.

## Usage
Run the script from the project root using `uv`:

```bash
uv run utils/cleanup_eval_outputs.py
```

## Transformations

### 1. Folder Renaming
The script traverses `eval/outputs/` and renames the 4th-level subdirectories:
- `Atlas_test` -> `Atlas_All`
- `Atlas_test_tiny` -> `Atlas_Tiny`
- `EduContent_test` -> `EduContent_All`
- `EduContent_test_tiny` -> `EduContent_Tiny`
- `PathCLS_test` -> `PathCLS_All`
- `PathCLS_test_tiny` -> `PathCLS_Tiny`
- `PubMed_test_tiny` -> `PubMed_Tiny`

### 2. JSON Metadata Updates
For each `summary.json` file found:
- **exp_name**: Updated to match the name of the folder containing the file (e.g., `"exp_name": "Atlas_All"`).
- **Categories**: The keys inside the `"categories"` object are renamed to match the new folder structure (e.g., `"Atlas_test"` becomes `"Atlas_All"`).

## Why use this?
If you run new evaluation experiments that default to the old naming convention, you should run this script before running `utils/generate_table.py` to ensure all data is correctly aggregated into the final comparison tables.
