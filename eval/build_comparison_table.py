#!/usr/bin/env python3
"""
eval/build_comparison_table.py

Generates two HTML tables:
  Table 1 — Your results: rows = models + judge, cols = dataset×split + weighted Overall (All splits only)
  Table 2 — PathMMU reference: rows = paper models, cols = same datasets/splits you have run + Overall

Usage:
    uv run eval/build_comparison_table.py
    uv run eval/build_comparison_table.py --out eval/comparison_table.html
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_LABELS = {
    "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224": "BiomedCLIP",
    "MahmoodLab/conch": "CONCH",
    "xiangjx/musk": "MUSK",
}

DATASET_ORDER = ["PubMed", "Atlas", "EduContent", "PathCLS"]
SPLIT_ORDER = ["Tiny", "All"]
JUDGE_LABEL = "Judge (LLM synthesizer)"

# Maps (dataset, split) → reference JSON key
_REF_KEY = {
    ("PubMed", "Tiny"): "pubmed_tiny",
    ("PubMed", "All"): "pubmed_all",
    ("EduContent", "Tiny"): "educontent_tiny",
    ("EduContent", "All"): "educontent_all",
    ("Atlas", "Tiny"): "atlas_tiny",
    ("Atlas", "All"): "atlas_all",
    ("PathCLS", "Tiny"): "pathcls_tiny",
    ("PathCLS", "All"): "pathcls_all",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_results(outputs_dir: Path) -> dict:
    """nested dict: results[dataset][split] = {judge_acc, per_model, n}"""
    results: dict = defaultdict(dict)
    for summary_path in sorted(outputs_dir.glob("*/summary.json")):
        data = json.loads(summary_path.read_text())
        exp = data["exp_name"]
        for split in SPLIT_ORDER:
            if exp.endswith(f"_{split}"):
                dataset = exp[: -(len(split) + 1)]
                break
        else:
            dataset, split = exp, "?"
        results[dataset][split] = {
            "judge_acc": data["judge_acc"],
            "per_model": data["per_model_acc"],
            "n": data["total_samples"],
        }
    return results


def load_reference(ref_path: Path) -> dict:
    return json.loads(ref_path.read_text())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def pct(value: float) -> str:
    return f"{value:.1f}"


def weighted_overall(results: dict, columns: list[tuple[str, str]]) -> dict[str, float | None]:
    """
    Compute weighted-average overall accuracy (All splits only) per model key
    and for judge. Returns {model_key: pct_float, "judge": pct_float}.
    """
    all_cols = [(ds, sp) for ds, sp in columns if sp == "All"]
    if not all_cols:
        return {}

    # Gather model keys
    model_keys: list[str] = []
    for ds, sp in all_cols:
        for k in results[ds][sp]["per_model"]:
            if k not in model_keys:
                model_keys.append(k)

    totals: dict[str, float] = defaultdict(float)
    n_totals: dict[str, int] = defaultdict(int)
    judge_total = 0.0
    judge_n = 0

    for ds, sp in all_cols:
        cell = results[ds][sp]
        n = cell["n"]
        judge_total += cell["judge_acc"] * n
        judge_n += n
        for mk in model_keys:
            v = cell["per_model"].get(mk)
            if v is not None:
                totals[mk] += v * n
                n_totals[mk] += n

    out: dict[str, float | None] = {}
    for mk in model_keys:
        out[mk] = (totals[mk] / n_totals[mk] * 100) if n_totals[mk] else None
    out["judge"] = (judge_total / judge_n * 100) if judge_n else None
    return out


def best_in_col(values: list[float | None]) -> float | None:
    cleaned = [v for v in values if v is not None]
    return max(cleaned) if cleaned else None


def cell_td(value: float | None, best: float | None, *, pct_already: bool = False) -> str:
    if value is None:
        return "<td>—</td>"
    s = f"{value:.1f}" if pct_already else pct(value * 100)
    if best is not None and abs((value if pct_already else value * 100) - best) < 0.05:
        return f"<td><strong>{s}</strong></td>"
    return f"<td>{s}</td>"


# ---------------------------------------------------------------------------
# Table 1 — Your results
# ---------------------------------------------------------------------------
def build_table1(results: dict) -> str:
    model_keys: list[str] = []
    for dataset in results.values():
        for split in dataset.values():
            for k in split["per_model"]:
                if k not in model_keys:
                    model_keys.append(k)

    columns: list[tuple[str, str]] = [
        (ds, sp)
        for ds in DATASET_ORDER
        for sp in SPLIT_ORDER
        if ds in results and sp in results[ds]
    ]

    overall = weighted_overall(results, columns)
    has_overall = bool(overall)

    # Compute column bests (including overall col)
    def col_vals(ds: str, sp: str) -> list[float]:
        cell = results[ds][sp]
        vals = [cell["per_model"][mk] * 100 for mk in model_keys if mk in cell["per_model"]]
        vals.append(cell["judge_acc"] * 100)
        return vals

    col_best = {(ds, sp): best_in_col(col_vals(ds, sp)) for ds, sp in columns}
    overall_vals = list(overall.values()) if has_overall else []
    overall_best = best_in_col([v for v in overall_vals if v is not None])

    # --- rows ---
    rows = []
    for mk in model_keys:
        label = MODEL_LABELS.get(mk, mk)
        cells = "".join(
            cell_td(results[ds][sp]["per_model"].get(mk), col_best[(ds, sp)])
            for ds, sp in columns
        )
        ov_td = (
            cell_td(overall.get(mk), overall_best, pct_already=True) if has_overall else ""
        )
        rows.append(f"<tr><td>{label}</td>{ov_td}{cells}</tr>")

    rows.append(
        f'<tr class="table-light fw-semibold">'
        f'<td colspan="{1 + (1 if has_overall else 0) + len(columns)}">LLM Synthesizer</td></tr>'
    )

    judge_cells = "".join(
        cell_td(results[ds][sp]["judge_acc"], col_best[(ds, sp)]) for ds, sp in columns
    )
    ov_judge = (
        cell_td(overall.get("judge"), overall_best, pct_already=True) if has_overall else ""
    )
    rows.append(f"<tr><td>{JUDGE_LABEL}</td>{ov_judge}{judge_cells}</tr>")

    n_cells = "".join(
        f'<td class="text-muted small">n={results[ds][sp]["n"]}</td>' for ds, sp in columns
    )
    n_overall = (
        f'<td class="text-muted small">weighted</td>' if has_overall else ""
    )
    rows.append(f'<tr class="text-muted small"><td>Samples</td>{n_overall}{n_cells}</tr>')

    # --- headers ---
    ds_groups: dict[str, list[str]] = defaultdict(list)
    for ds, sp in columns:
        ds_groups[ds].append(sp)

    h1 = '<th rowspan="2" class="align-middle">Model</th>'
    if has_overall:
        h1 += '<th rowspan="2" class="align-middle border-start">Overall<br><small>(All, wtd)</small></th>'
    for ds in DATASET_ORDER:
        splits = ds_groups.get(ds, [])
        if not splits:
            continue
        h1 += f'<th colspan="{len(splits)}" class="text-center border-start">{ds}</th>'

    h2 = ""
    for ds, sp in columns:
        border = ' class="border-start"' if sp == SPLIT_ORDER[0] else ""
        h2 += f"<th{border}>{sp}</th>"

    return f"""
  <h2>Table 1 — MedHuggingGPT Results</h2>
  <p class="text-muted mb-3">Accuracy (%) on PathMMU. <strong>Bold</strong> = best per column.</p>
  <div class="table-responsive">
    <table class="table table-bordered table-hover align-middle">
      <caption>Ensemble: BiomedCLIP + CONCH + MUSK &nbsp;|&nbsp; Judge: Gemini</caption>
      <thead>
        <tr>{h1}</tr>
        <tr>{h2}</tr>
      </thead>
      <tbody>{"".join(rows)}</tbody>
    </table>
  </div>"""


# ---------------------------------------------------------------------------
# Table 2 — PathMMU reference
# ---------------------------------------------------------------------------
def build_table2(results: dict, ref: dict) -> str:
    # Only show columns the user has actually run
    columns: list[tuple[str, str]] = [
        (ds, sp)
        for ds in DATASET_ORDER
        for sp in SPLIT_ORDER
        if ds in results and sp in results[ds]
    ]
    has_overall = any(sp == "All" for _, sp in columns)

    ref_n = ref["n"]

    # Flatten all models for best-per-column calculation
    all_models = [m for g in ref["groups"] for m in g["models"]]

    def ref_val(model: dict, ds: str, sp: str) -> float | None:
        key = _REF_KEY.get((ds, sp))
        return model.get(key) if key else None

    def ref_overall(model: dict) -> float | None:
        if not has_overall:
            return None
        total, n_total = 0.0, 0
        for ds, sp in columns:
            if sp != "All":
                continue
            key = _REF_KEY.get((ds, sp))
            if not key:
                continue
            v = model.get(key)
            n = ref_n.get(key, 0)
            if v is not None and n:
                total += v * n
                n_total += n
        return total / n_total if n_total else None

    # Collect bests per column
    col_all_vals: dict[tuple, list[float]] = {col: [] for col in columns}
    overall_vals_ref: list[float] = []
    for m in all_models:
        for col in columns:
            v = ref_val(m, *col)
            if v is not None:
                col_all_vals[col].append(v)
        ov = ref_overall(m)
        if ov is not None:
            overall_vals_ref.append(ov)

    col_best_ref = {col: best_in_col(col_all_vals[col]) for col in columns}
    overall_best_ref = best_in_col(overall_vals_ref)

    def cell_ref(value: float | None, best: float | None) -> str:
        if value is None:
            return "<td>—</td>"
        s = f"{value:.1f}"
        if best is not None and abs(value - best) < 0.05:
            return f"<td><strong>{s}</strong></td>"
        return f"<td>{s}</td>"

    # --- rows ---
    rows = []
    for group in ref["groups"]:
        if group.get("separator_above"):
            rows.append(
                f'<tr><td colspan="{1 + (1 if has_overall else 0) + len(columns)}"'
                f' style="border-top:2px solid #6c757d;padding:0"></td></tr>'
            )
        if group.get("label"):
            rows.append(
                f'<tr class="table-secondary fw-semibold">'
                f'<td colspan="{1 + (1 if has_overall else 0) + len(columns)}">'
                f'{group["label"]}</td></tr>'
            )
        for m in group["models"]:
            ov = ref_overall(m)
            ov_td = cell_ref(ov, overall_best_ref) if has_overall else ""
            cells = "".join(cell_ref(ref_val(m, ds, sp), col_best_ref[(ds, sp)]) for ds, sp in columns)
            rows.append(f'<tr><td>{m["name"]}</td>{ov_td}{cells}</tr>')

    # n row
    n_cells = "".join(
        f'<td class="text-muted small">n={ref_n.get(_REF_KEY.get((ds, sp), ""), "?")}</td>'
        for ds, sp in columns
    )
    n_overall_ref = '<td class="text-muted small">weighted</td>' if has_overall else ""
    rows.append(f'<tr class="text-muted small"><td>Samples (paper)</td>{n_overall_ref}{n_cells}</tr>')

    # --- headers ---
    ds_groups: dict[str, list[str]] = defaultdict(list)
    for ds, sp in columns:
        ds_groups[ds].append(sp)

    h1 = '<th rowspan="2" class="align-middle">Model</th>'
    if has_overall:
        h1 += '<th rowspan="2" class="align-middle border-start">Overall<br><small>(All, wtd)</small></th>'
    for ds in DATASET_ORDER:
        splits = ds_groups.get(ds, [])
        if not splits:
            continue
        h1 += f'<th colspan="{len(splits)}" class="text-center border-start">{ds}</th>'

    h2 = ""
    for ds, sp in columns:
        border = ' class="border-start"' if sp == SPLIT_ORDER[0] else ""
        h2 += f"<th{border}>{sp}</th>"

    return f"""
  <h2 class="mt-5">Table 2 — PathMMU Reference (paper)</h2>
  <p class="text-muted mb-3">
    Same columns as Table 1. <strong>Bold</strong> = best per column.
    Overall is weighted by paper n values across matched All splits.
  </p>
  <div class="table-responsive">
    <table class="table table-bordered table-hover align-middle">
      <thead>
        <tr>{h1}</tr>
        <tr>{h2}</tr>
      </thead>
      <tbody>{"".join(rows)}</tbody>
    </table>
  </div>"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", default=Path(__file__).parent / "outputs", type=Path)
    parser.add_argument("--ref", default=Path(__file__).parent / "pathmmu_reference.json", type=Path)
    parser.add_argument("--out", default=Path(__file__).parent / "comparison_table.html", type=Path)
    args = parser.parse_args()

    results = load_results(args.outputs)
    ref = load_reference(args.ref)

    t1 = build_table1(results)
    t2 = build_table2(results, ref)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>PathMMU Comparison Tables</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body {{ font-family: 'Georgia', serif; background: #f8f9fa; padding: 2rem; }}
    h2 {{ font-weight: bold; margin-bottom: 0.25rem; }}
    table {{ font-size: 0.91rem; }}
    thead th {{ background: #212529; color: #fff; text-align: center; vertical-align: middle; white-space: nowrap; }}
    tbody td {{ text-align: center; vertical-align: middle; }}
    tbody td:first-child {{ text-align: left; font-style: italic; white-space: nowrap; }}
    .border-start {{ border-left: 2px solid #adb5bd !important; }}
    caption {{ caption-side: top; font-size: 0.85rem; color: #6c757d; margin-bottom: 0.5rem; }}
  </style>
</head>
<body>
  <div class="container-fluid">
    {t1}
    <hr class="my-5">
    {t2}
  </div>
</body>
</html>"""

    args.out.write_text(html)
    print(f"Written → {args.out}")


if __name__ == "__main__":
    main()
