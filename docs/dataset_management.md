# Dataset Configuration Management

`manage_datasets.py` is the CLI tool for maintaining `datasets_config.json` — the registry that maps dataset keys to their local file paths and medical imaging modalities.

---

## `datasets_config.json` Schema

```json
{
  "DATASET_KEY": {
    "modality": "<modality string>",
    "files": [
      { "path": "/absolute/path/to/file.dcm", "type": "dcm" },
      ...
    ]
  }
}
```

| Field | Type | Description |
|---|---|---|
| `DATASET_KEY` | string | `UPPER_SNAKE_CASE` identifier assigned by LLM or user |
| `modality` | string | One of the [supported modalities](#supported-modalities) |
| `files[].path` | string | Absolute path to a sampled file from the dataset directory |
| `files[].type` | string | File extension without the leading dot (e.g. `dcm`, `nii.gz`) |

### Supported Modalities

| Value | Description |
|---|---|
| `Legacy MRI` | Analyze 7.5 `.hdr`/`.img` pairs (e.g. OASIS-1) |
| `2D Histopathology` | Flat H&E or IHC microscopy images |
| `3D Volumetric` | Generic 3D volumetric data |
| `Hyperspectral` | ENVI BIL hyperspectral cubes |
| `CT Image` | Generic CT images (JPEG/PNG reconstructions) |
| `DICOM CT` | DICOM series CT |
| `NIfTI` | NIfTI (`.nii`, `.nii.gz`) volumes |
| `WSI Pathology` | Whole-slide images (`.svs`, `.ndpi`, `.tiff`) |
| `HSI Pathology` | Hyperspectral histology (ENVI BIL) |
| `Unknown` | Could not be determined |

### Recognized File Extensions

`.img` `.hdr` `.jpg` `.jpeg` `.tif` `.tiff` `.nii` `.nii.gz` `.raw` `.dcm` `.png` `.svs` `.ndpi`

---

## Commands

### `add` — Register a new dataset

```bash
uv run manage_datasets.py add <path> [--n N] [--yes] [--dry-run]
```

**What it does:**

1. Recursively scans `<path>` for recognized medical imaging files.
2. Randomly samples up to `--n` files (default 10).
3. Calls the LLM (Google Gemini by default) in up to 3 escalating rounds to identify the dataset key and modality:
   - Round 1: directory path only
   - Round 2: path + file listing (if confidence is `low`)
   - Round 3: path + listing + sample metadata from text-readable files (if still `low`)
4. If the proposed key is similar (≥ 60% string match) to an existing key, offers to reuse it.
5. If confidence is `low` and `--yes` is not set, prompts for manual correction.
6. Merges the new files into the config (no duplicates by path).
7. Writes `datasets_config.json`.

**Options:**

| Flag | Default | Description |
|---|---|---|
| `<path>` | required | Path to the dataset directory to scan |
| `--n N` | `10` | Maximum number of files to sample |
| `--yes` | off | Auto-accept LLM proposals without any prompts |
| `--dry-run` | off | Preview the change without writing to disk |

**Examples:**

```bash
# Add with interactive confirmation
uv run manage_datasets.py add '/path/to/Datasets/Spinal'

# Add non-interactively (CI-friendly)
uv run manage_datasets.py add '/path/to/Datasets/Oasis1/disc1/OAS1_0016_MR1' --yes

# Preview only
uv run manage_datasets.py add '/path/to/Datasets/PKG - HistologyHSI-GB/P1' --dry-run

# Sample more files for better LLM context
uv run manage_datasets.py add '/path/to/Datasets/Quilt1M/quilt_1m' --n 25
```

---

### `remove` — Delete a dataset entry

```bash
uv run manage_datasets.py remove <key> [--yes] [--dry-run]
```

**What it does:**

Removes the entry for `<key>` from `datasets_config.json` and saves the file.

**Options:**

| Flag | Default | Description |
|---|---|---|
| `<key>` | required | The `UPPER_SNAKE_CASE` key to remove |
| `--yes` | off | Skip the confirmation prompt |
| `--dry-run` | off | Preview removal without writing to disk |

**Examples:**

```bash
# Interactive removal
uv run manage_datasets.py remove QUILT_1M

# Non-interactive
uv run manage_datasets.py remove QUILT_1M --yes

# Preview only
uv run manage_datasets.py remove OASIS1_MRI --dry-run
```

---

## LLM Configuration

The tool uses `LLM_PROVIDER` (default `google`) to select the backend. Configure via environment variables (`.env` file or shell exports).

### Google Gemini (default)

```env
LLM_PROVIDER=google
GOOGLE_API_KEY=your-key-here        # or GEMINI_API_KEY — both are accepted
GOOGLE_MODEL_NAME=gemini-1.5-flash  # optional, this is the default
```

### Ollama (local)

```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3                      # optional, default: llama3
OLLAMA_BASE_URL=http://localhost:11434   # optional, this is the default
```

---

## Current Registry

The five datasets currently registered in `datasets_config.json`:

| Key | Modality | Dataset |
|---|---|---|
| `IQ_OTHNCCD_LUNG_CANCER` | NIfTI | IQ-OTH/NCCD Lung Cancer (JPEG CT images) |
| `OASIS1_MRI` | Legacy MRI | OASIS-1 Brain MRI (Analyze 7.5 `.hdr`/`.img`) |
| `HISTOLOGY_HSI_GB` | HSI Pathology | PKG HistologyHSI-GB (ENVI BIL hyperspectral) |
| `QUILT_1M` | Unknown | Quilt-1M optical microscopy (JPEG/PNG) |
| `SPINAL_MYELOMA_SEG` | DICOM CT | Spinal Multiple Myeloma SEG (DICOM + NIfTI) |

---

## How the App Uses This Config

`app.py` loads `datasets_config.json` at startup via `load_datasets_config()`. The sidebar "Dataset Selection" panel lists all registered keys, and selecting one populates the file picker with the sampled paths stored in `files[]`.

The agent receives the selected file path and modality string, which routes it to the appropriate specialized tool in `tools/`.
