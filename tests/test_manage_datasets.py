import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from manage_datasets import (
    cmd_add,
    find_matching_key,
    identify_dataset,
    merge_into_config,
    scan_files,
)

MOCK_OASIS_PATH = Path(__file__).parent.parent / "workspace" / "mock_oasis"
HAS_LLM = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
requires_llm = pytest.mark.skipif(not HAS_LLM, reason="No LLM API key configured")


# ---------------------------------------------------------------------------
# scan_files
# ---------------------------------------------------------------------------

def test_scan_files_filters_unrecognized(tmp_path):
    (tmp_path / "image.tif").write_bytes(b"fake")
    (tmp_path / "notes.txt").write_bytes(b"noise")
    (tmp_path / "data.dcm").write_bytes(b"fake")

    result = scan_files(tmp_path, max_n=10)
    names = {Path(f["path"]).name for f in result}
    assert "image.tif" in names
    assert "data.dcm" in names
    assert "notes.txt" not in names


def test_scan_files_respects_max_n(tmp_path):
    for i in range(20):
        (tmp_path / f"scan_{i:02d}.dcm").write_bytes(b"fake")

    result = scan_files(tmp_path, max_n=5)
    assert len(result) == 5


def test_scan_files_returns_all_when_fewer_than_max(tmp_path):
    for i in range(3):
        (tmp_path / f"scan_{i}.tif").write_bytes(b"fake")

    result = scan_files(tmp_path, max_n=10)
    assert len(result) == 3


def test_scan_files_empty_dir(tmp_path):
    assert scan_files(tmp_path, max_n=10) == []


def test_scan_files_detects_nii_gz(tmp_path):
    (tmp_path / "brain.nii.gz").write_bytes(b"fake")

    result = scan_files(tmp_path, max_n=10)
    assert len(result) == 1
    assert result[0]["type"] == "nii.gz"


def test_scan_files_recursive(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "deep.tif").write_bytes(b"fake")
    (tmp_path / "top.dcm").write_bytes(b"fake")

    result = scan_files(tmp_path, max_n=10)
    assert len(result) == 2


def test_scan_files_result_contains_path_and_type(tmp_path):
    (tmp_path / "img.tif").write_bytes(b"fake")

    result = scan_files(tmp_path, max_n=10)
    assert "path" in result[0]
    assert "type" in result[0]
    assert result[0]["type"] == "tif"


# ---------------------------------------------------------------------------
# find_matching_key
# ---------------------------------------------------------------------------

def test_find_matching_key_exact():
    assert find_matching_key("OASIS_MRI", ["OASIS_MRI", "TCGA"]) == "OASIS_MRI"


def test_find_matching_key_close():
    result = find_matching_key("OASIS_MRI_V1", ["OASIS_MRI", "TCGA_PATHOLOGY"])
    assert result == "OASIS_MRI"


def test_find_matching_key_no_match():
    result = find_matching_key("HSI_SKIN", ["TCGA_PATHOLOGY", "DICOM_CHEST"])
    assert result is None


def test_find_matching_key_empty_existing():
    assert find_matching_key("OASIS_MRI", []) is None


def test_find_matching_key_case_insensitive():
    result = find_matching_key("oasis_mri", ["OASIS_MRI"])
    assert result == "OASIS_MRI"


# ---------------------------------------------------------------------------
# merge_into_config
# ---------------------------------------------------------------------------

def test_merge_creates_new_entry():
    config = {}
    files = [{"path": "/data/a.tif", "type": "tif"}]
    result = merge_into_config(config, "TCGA", "2D Histopathology", files)
    assert "TCGA" in result
    assert result["TCGA"]["modality"] == "2D Histopathology"
    assert len(result["TCGA"]["files"]) == 1


def test_merge_adds_new_files_only():
    config = {
        "OASIS_MRI": {
            "modality": "Legacy MRI",
            "files": [{"path": "/data/a.img", "type": "img"}],
        }
    }
    new_files = [
        {"path": "/data/a.img", "type": "img"},  # duplicate — skip
        {"path": "/data/b.hdr", "type": "hdr"},  # new — add
    ]
    result = merge_into_config(config, "OASIS_MRI", "Legacy MRI", new_files)
    paths = [f["path"] for f in result["OASIS_MRI"]["files"]]
    assert paths.count("/data/a.img") == 1
    assert "/data/b.hdr" in paths


def test_merge_does_not_mutate_original():
    config = {"OASIS_MRI": {"modality": "Legacy MRI", "files": []}}
    merge_into_config(config, "OASIS_MRI", "Legacy MRI", [{"path": "/x.img", "type": "img"}])
    assert config["OASIS_MRI"]["files"] == []


def test_merge_all_duplicates_adds_nothing():
    config = {
        "DS": {
            "modality": "Legacy MRI",
            "files": [{"path": "/a.img", "type": "img"}],
        }
    }
    result = merge_into_config(config, "DS", "Legacy MRI", [{"path": "/a.img", "type": "img"}])
    assert len(result["DS"]["files"]) == 1


# ---------------------------------------------------------------------------
# identify_dataset — end-to-end with real LLM
# ---------------------------------------------------------------------------

@requires_llm
def test_identify_dataset_mock_oasis():
    files = scan_files(MOCK_OASIS_PATH, max_n=10)
    assert files, "mock_oasis has no recognized files"

    result = identify_dataset(MOCK_OASIS_PATH, files)

    assert "key" in result
    assert "modality" in result
    assert result["confidence"] in ("high", "medium", "low")
    # LLM should recognize MRI or OASIS from the path and .img/.hdr files
    key_lower = result["key"].lower()
    modality_lower = result["modality"].lower()
    assert "mri" in key_lower or "oasis" in key_lower or "mri" in modality_lower


@requires_llm
def test_identify_dataset_returns_valid_modality(tmp_path):
    from manage_datasets import MODALITIES
    (tmp_path / "slide.tif").write_bytes(b"fake")
    (tmp_path / "slide2.jpg").write_bytes(b"fake")

    files = scan_files(tmp_path, max_n=10)
    result = identify_dataset(tmp_path, files)

    assert result["modality"] in MODALITIES


# ---------------------------------------------------------------------------
# cmd_add — end-to-end dry-run with real LLM
# ---------------------------------------------------------------------------

@requires_llm
def test_cmd_add_dry_run_does_not_write(tmp_path, monkeypatch, capsys):
    import manage_datasets

    temp_config = tmp_path / "datasets_config.json"
    temp_config.write_text("{}")
    monkeypatch.setattr(manage_datasets, "DATASETS_CONFIG_PATH", temp_config)

    cmd_add(str(MOCK_OASIS_PATH), n=10, yes=True, dry_run=True)

    captured = capsys.readouterr()
    assert "[dry-run]" in captured.out
    assert json.loads(temp_config.read_text()) == {}


@requires_llm
def test_cmd_add_writes_to_config(tmp_path, monkeypatch):
    import manage_datasets

    temp_config = tmp_path / "datasets_config.json"
    temp_config.write_text("{}")
    monkeypatch.setattr(manage_datasets, "DATASETS_CONFIG_PATH", temp_config)

    cmd_add(str(MOCK_OASIS_PATH), n=10, yes=True, dry_run=False)

    result = json.loads(temp_config.read_text())
    assert len(result) == 1
    key = list(result.keys())[0]
    assert len(result[key]["files"]) > 0
    assert "modality" in result[key]


@requires_llm
def test_cmd_add_merge_does_not_duplicate(tmp_path, monkeypatch):
    import manage_datasets

    temp_config = tmp_path / "datasets_config.json"
    temp_config.write_text("{}")
    monkeypatch.setattr(manage_datasets, "DATASETS_CONFIG_PATH", temp_config)

    # Add twice
    cmd_add(str(MOCK_OASIS_PATH), n=10, yes=True, dry_run=False)
    cmd_add(str(MOCK_OASIS_PATH), n=10, yes=True, dry_run=False)

    result = json.loads(temp_config.read_text())
    key = list(result.keys())[0]
    paths = [f["path"] for f in result[key]["files"]]
    assert len(paths) == len(set(paths)), "Duplicate files found after second add"
