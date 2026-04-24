#!/usr/bin/env python3
"""
manage_datasets.py — Add and remove datasets from datasets_config.json.

Usage:
    uv run manage_datasets.py add <path> [--n 10] [--yes] [--dry-run]
    uv run manage_datasets.py remove <key> [--yes] [--dry-run]
"""

import argparse
import json
import os
import random
import sys
from difflib import SequenceMatcher
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASETS_CONFIG_PATH = Path(__file__).parent / "datasets_config.json"

# Extensions that count as recognized medical imaging files
RECOGNIZED_EXTENSIONS = {
    ".img", ".hdr", ".jpg", ".jpeg",
    ".tif", ".tiff", ".nii", ".raw",
    ".dcm", ".png", ".svs", ".ndpi", ".mrxs",
}
DOUBLE_EXTENSIONS = {".nii.gz"}  # must be checked before single-ext lookup

EXTENSION_TYPE: dict[str, str] = {
    ".img": "img", ".hdr": "hdr",
    ".jpg": "jpg", ".jpeg": "jpeg",
    ".tif": "tif", ".tiff": "tiff",
    ".nii": "nii", ".nii.gz": "nii.gz",
    ".raw": "raw", ".dcm": "dcm",
    ".png": "png", ".svs": "svs", ".ndpi": "ndpi", ".mrxs": "mrxs",
}

MODALITIES = [
    "Legacy MRI",
    "2D Histopathology",
    "3D Volumetric",
    "Hyperspectral",
    "CT Image",
    "DICOM CT",
    "NIfTI",
    "WSI Pathology",
    "Histopathology",
    "HSI Pathology",
    "Unknown",
]

KEY_MATCH_THRESHOLD = 0.6

# Extensions readable as text (used for sample metadata)
TEXT_EXTENSIONS = {".hdr", ".xml", ".txt", ".json"}


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------

def _is_complete_mrxs(mrxs_path: Path) -> bool:
    """Return True if the MRXS companion data directory exists and is fully downloaded."""
    data_dir = mrxs_path.with_suffix("")
    if not data_dir.is_dir():
        return False
    return any(data_dir.glob("*.dat")) and not any(data_dir.glob("*.partial"))


def scan_files(directory: Path, max_n: int = 10) -> list[dict]:
    """
    Recursively scan *directory* and return up to *max_n* randomly sampled
    recognized medical imaging files as ``{"path": str, "type": str}`` dicts.
    Returns all found files when fewer than *max_n* exist.

    MRXS slides: only the ``.mrxs`` entry-point file is registered (the
    companion ``.dat`` directory is skipped).  Incomplete slides (missing
    companion dir or still-downloading ``.partial`` files) are excluded.
    """
    found: list[dict] = []
    for root, _, files in os.walk(directory):
        for fname in files:
            fpath = Path(root) / fname
            # Double-extension check must come first
            if fname.lower().endswith(".nii.gz"):
                found.append({"path": str(fpath), "type": "nii.gz"})
                continue
            ext = fpath.suffix.lower()
            if ext not in RECOGNIZED_EXTENSIONS:
                continue
            if ext == ".mrxs":
                if _is_complete_mrxs(fpath):
                    found.append({"path": str(fpath), "type": "mrxs"})
                # skip incomplete / still-downloading slides silently
            else:
                found.append({"path": str(fpath), "type": EXTENSION_TYPE.get(ext, ext.lstrip("."))})

    if len(found) <= max_n:
        return found
    return random.sample(found, max_n)


def find_matching_key(proposed: str, existing_keys: list[str]) -> str | None:
    """
    Return the closest existing key if its similarity score to *proposed* is
    at or above ``KEY_MATCH_THRESHOLD`` (case-insensitive). Returns ``None``
    when no key is close enough or *existing_keys* is empty.
    """
    if not existing_keys:
        return None
    best = max(
        existing_keys,
        key=lambda k: SequenceMatcher(None, proposed.lower(), k.lower()).ratio(),
    )
    score = SequenceMatcher(None, proposed.lower(), best.lower()).ratio()
    return best if score >= KEY_MATCH_THRESHOLD else None


def merge_into_config(config: dict, key: str, modality: str, new_files: list[dict]) -> dict:
    """
    Return a *new* config dict with *new_files* merged into *config[key]*.
    Creates the entry if it does not exist. Files already present (by path)
    are skipped — no duplicates are introduced. The original dict is not mutated.
    """
    result = {k: dict(v) for k, v in config.items()}  # shallow copy of top level
    if key not in result:
        result[key] = {"modality": modality, "files": []}

    existing_paths = {f["path"] for f in result[key].get("files", [])}
    to_add = [f for f in new_files if f["path"] not in existing_paths]
    result[key]["files"] = list(result[key].get("files", [])) + to_add
    return result


def get_sample_metadata(files: list[dict], max_files: int = 3) -> str:
    """Read the first 300 chars of up to *max_files* text-readable files."""
    samples: list[str] = []
    text_files = [f for f in files if Path(f["path"]).suffix.lower() in TEXT_EXTENSIONS]
    for f in text_files[:max_files]:
        try:
            with open(f["path"], "r", errors="ignore") as fh:
                content = fh.read(300).strip()
            samples.append(f"[{f['path']}]:\n{content}")
        except OSError:
            pass
    return "\n\n".join(samples) if samples else "No readable metadata found."


# ---------------------------------------------------------------------------
# LLM identification
# ---------------------------------------------------------------------------

def _build_llm():
    provider = os.getenv("LLM_PROVIDER", "google").lower()
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    from langchain_google_genai import ChatGoogleGenerativeAI
    if not os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
    return ChatGoogleGenerativeAI(model=os.getenv("GOOGLE_MODEL_NAME", "gemini-1.5-flash"))


def _call_llm(
    llm,
    directory: Path,
    files: list[dict],
    include_listing: bool,
    include_metadata: bool,
) -> dict:
    listing_block = (
        "File listing:\n" + "\n".join(f["path"] for f in files)
        if include_listing else ""
    )
    metadata_block = (
        "Sample metadata:\n" + get_sample_metadata(files)
        if include_metadata else ""
    )

    prompt = f"""You are a medical imaging dataset classifier.

Given the following information about a directory, identify:
1. A concise dataset key in UPPER_SNAKE_CASE (e.g. "OASIS_MRI", "TCGA_PATHOLOGY", "HSI_SKIN_001")
2. The medical imaging modality — choose exactly one from this list: {MODALITIES}
3. Your confidence: "high", "medium", or "low"

Directory path: {directory}
{listing_block}
{metadata_block}

Respond with JSON only, no markdown fences:
{{"key": "DATASET_KEY", "modality": "modality string", "confidence": "high|medium|low", "reasoning": "brief explanation"}}"""

    response = llm.invoke(prompt)
    text = response.content.strip()
    # Strip markdown code fences if the model wraps its response
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else parts[0]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import re
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            return json.loads(m.group())
        raise


def identify_dataset(directory: Path, files: list[dict], llm=None) -> dict:
    """
    Escalating 3-step LLM identification:
      1. Directory path alone
      2. Path + file listing (if step 1 returns low confidence)
      3. Path + listing + sample metadata (if step 2 still returns low confidence)

    Returns ``{"key", "modality", "confidence", "reasoning"}``.
    """
    if llm is None:
        llm = _build_llm()

    result = _call_llm(llm, directory, files, include_listing=False, include_metadata=False)
    if result.get("confidence") != "low":
        return result

    result = _call_llm(llm, directory, files, include_listing=True, include_metadata=False)
    if result.get("confidence") != "low":
        return result

    return _call_llm(llm, directory, files, include_listing=True, include_metadata=True)


# ---------------------------------------------------------------------------
# Config I/O
# ---------------------------------------------------------------------------

def load_config() -> dict:
    if DATASETS_CONFIG_PATH.exists():
        with open(DATASETS_CONFIG_PATH) as f:
            content = f.read().strip()
            if content:
                return json.loads(content)
    return {}


def save_config(config: dict) -> None:
    with open(DATASETS_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Written to {DATASETS_CONFIG_PATH}")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_add(path: str, n: int, yes: bool, dry_run: bool) -> None:
    directory = Path(path).resolve()
    if not directory.is_dir():
        print(f"Error: {directory} is not a directory.", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {directory} for recognized medical files...")
    files = scan_files(directory, max_n=n)
    if not files:
        print("No recognized medical files found. Nothing to add.")
        return
    print(f"Found {len(files)} file(s).")

    config = load_config()

    print("Identifying dataset with LLM...")
    identification = identify_dataset(directory, files)
    proposed_key = identification["key"]
    modality = identification["modality"]
    confidence = identification["confidence"]

    print(f"\nLLM proposal:")
    print(f"  Key:        {proposed_key}")
    print(f"  Modality:   {modality}")
    print(f"  Confidence: {confidence}")
    print(f"  Reasoning:  {identification.get('reasoning', '')}")

    # Check for a close existing key and offer to reuse it
    matched_key = find_matching_key(proposed_key, list(config.keys()))
    if matched_key and matched_key != proposed_key:
        print(f"\nClose match found in config: '{matched_key}'")
        if yes:
            proposed_key = matched_key
            print(f"  --yes: using existing key '{matched_key}'")
        else:
            answer = input(f"  Use existing key '{matched_key}' instead? [Y/n]: ").strip().lower()
            if answer in ("", "y", "yes"):
                proposed_key = matched_key

    # Ask user to correct when LLM is uncertain (and --yes was not passed)
    if confidence == "low" and not yes:
        print("\nLow confidence — please confirm or correct the proposal:")
        new_key = input(f"  Dataset key [{proposed_key}]: ").strip()
        if new_key:
            proposed_key = new_key
        new_modality = input(f"  Modality [{modality}]: ").strip()
        if new_modality:
            modality = new_modality

    new_config = merge_into_config(config, proposed_key, modality, files)
    added_count = len(new_config[proposed_key]["files"]) - len(
        config.get(proposed_key, {}).get("files", [])
    )

    if dry_run:
        print(f"\n[dry-run] Would add {added_count} file(s) to '{proposed_key}' (modality: {modality})")
        print(json.dumps(new_config[proposed_key], indent=2))
        return

    save_config(new_config)
    print(f"\nAdded {added_count} file(s) to '{proposed_key}' (modality: {modality})")


def cmd_remove(key: str, yes: bool, dry_run: bool) -> None:
    config = load_config()
    if key not in config:
        print(f"Key '{key}' not found. Available: {list(config.keys())}", file=sys.stderr)
        sys.exit(1)

    file_count = len(config[key].get("files", []))

    if not yes:
        answer = input(f"Remove '{key}' ({file_count} file(s))? [y/N]: ").strip().lower()
        if answer not in ("y", "yes"):
            print("Aborted.")
            return

    if dry_run:
        print(f"[dry-run] Would remove '{key}' ({file_count} file(s)) from config.")
        return

    del config[key]
    save_config(config)
    print(f"Removed '{key}' from config.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manage datasets_config.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    add_p = sub.add_parser("add", help="Add a dataset directory to config")
    add_p.add_argument("path", help="Path to the dataset directory")
    add_p.add_argument("--n", type=int, default=10, metavar="N",
                       help="Max files to include (default: 10)")
    add_p.add_argument("--yes", action="store_true",
                       help="Auto-accept LLM proposals without prompting")
    add_p.add_argument("--dry-run", action="store_true",
                       help="Preview changes without writing to config")

    rm_p = sub.add_parser("remove", help="Remove a dataset entry from config")
    rm_p.add_argument("key", help="Dataset key to remove")
    rm_p.add_argument("--yes", action="store_true",
                       help="Skip confirmation prompt")
    rm_p.add_argument("--dry-run", action="store_true",
                       help="Preview removal without writing to config")

    args = parser.parse_args()

    if args.command == "add":
        cmd_add(args.path, args.n, args.yes, args.dry_run)
    elif args.command == "remove":
        cmd_remove(args.key, args.yes, args.dry_run)


if __name__ == "__main__":
    main()
