import json
import os
from pathlib import Path
import pandas as pd
from tabulate import tabulate

MODEL_MAP = {
    "MahmoodLab/conch": "CONCH",
    "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224": "BiomedCLIP",
    "xiangjx/musk": "musk"
}

REFERENCE_JSON = "eval/pathmmu_reference.json"

def parse_category(folder_name):
    """Parses folder name like 'Atlas_All' or 'Atlas_Tiny' into (Dataset, Split)."""
    if "_" not in folder_name:
        return folder_name, "All"
    parts = folder_name.split("_")
    dataset = parts[0]
    split = parts[1]
    return dataset, split

def calculate_overall(row_dict, sorted_datasets):
    """Calculates overall accuracy as an average of 'All' splits for the datasets."""
    values = []
    for ds in sorted_datasets:
        val = row_dict.get(ds, {}).get("All", None)
        if val is not None:
            values.append(val)
    return sum(values) / len(values) if values else None

def generate_table():
    outputs_path = Path("eval/outputs")
    summary_files = list(outputs_path.glob("**/summary.json"))
    
    data = {}
    datasets = set()
    splits = ["Tiny", "All"]
    judge_data = {}

    for f in summary_files:
        try:
            with open(f, 'r') as jf:
                summary = json.load(jf)
            dataset, split = parse_category(f.parent.name)
            datasets.add(dataset)
            
            per_model = summary.get("per_model_acc", {})
            for raw_name, acc in per_model.items():
                model_name = MODEL_MAP.get(raw_name, raw_name.split('/')[-1])
                if model_name not in data: data[model_name] = {}
                if dataset not in data[model_name]: data[model_name][dataset] = {}
                data[model_name][dataset][split] = acc * 100

            judge_acc = summary.get("judge_acc", 0)
            if "Judge (Ensemble)" not in judge_data: judge_data["Judge (Ensemble)"] = {}
            if dataset not in judge_data["Judge (Ensemble)"]: judge_data["Judge (Ensemble)"][dataset] = {}
            judge_data["Judge (Ensemble)"][dataset][split] = judge_acc * 100
        except Exception as e:
            print(f"Error reading {f}: {e}")

    sorted_datasets = sorted(list(datasets))
    columns = []
    for ds in sorted_datasets:
        for sp in splits: columns.append((ds, sp))
    columns.append(("OVERALL", "All"))
    
    col_index = pd.MultiIndex.from_tuples(columns, names=['Dataset', 'Split'])
    
    # Prepare Table 1 Rows
    all_models = sorted(list(data.keys()))
    row_data = []
    for model in all_models:
        row = []
        model_row_dict = data.get(model, {})
        for ds, sp in columns:
            if ds == "OVERALL":
                val = calculate_overall(model_row_dict, sorted_datasets)
            else:
                val = model_row_dict.get(ds, {}).get(sp, None)
            row.append(f"{val:.1f}" if val is not None else "-")
        row_data.append(row)
    
    judge_row = []
    judge_row_dict = judge_data.get("Judge (Ensemble)", {})
    for ds, sp in columns:
        if ds == "OVERALL":
            val = calculate_overall(judge_row_dict, sorted_datasets)
        else:
            val = judge_row_dict.get(ds, {}).get(sp, None)
        judge_row.append(f"{val:.1f}" if val is not None else "-")
    row_data.append(judge_row)
    
    df1 = pd.DataFrame(row_data, index=all_models + ["Judge (Ensemble)"], columns=col_index)

    # --- Table 2 (Reference) ---
    with open(REFERENCE_JSON, 'r') as rf:
        ref_json = json.load(rf)

    ref_rows = []
    ref_labels = []
    
    for group in ref_json["groups"]:
        if group["separator_above"]:
            ref_rows.append(["---"] * len(columns))
            ref_labels.append("---")
            
        for m in group["models"]:
            row = []
            # Create a nested dict for calculate_overall
            m_dict = {}
            for ds in sorted_datasets:
                m_dict[ds] = {
                    "Tiny": m.get(f"{ds.lower()}_tiny"),
                    "All": m.get(f"{ds.lower()}_all")
                }
            
            for ds, sp in columns:
                if ds == "OVERALL":
                    val = calculate_overall(m_dict, sorted_datasets)
                else:
                    val = m_dict.get(ds, {}).get(sp)
                row.append(f"{val:.1f}" if val is not None else "-")
            ref_rows.append(row)
            ref_labels.append(m["name"])

    df2 = pd.DataFrame(ref_rows, index=ref_labels, columns=col_index)

    print("\n--- TABLE 1: CURRENT EVALUATION RESULTS (%) ---")
    flat_cols = [f"{ds}\n({sp})" for ds, sp in columns]
    print(tabulate(df1, headers=flat_cols, tablefmt="grid"))
    
    print("\n\n--- TABLE 2: PATHMMU REFERENCE BENCHMARKS (%) ---")
    print(tabulate(df2, headers=flat_cols, tablefmt="grid"))
    
    with open("eval/results_table.md", "w") as f:
        f.write("# Evaluation Results Comparison\n\n")
        f.write("## Table 1: Current Results\n\n")
        f.write(df1.to_markdown() + "\n\n")
        f.write("## Table 2: Reference Benchmarks\n\n")
        f.write(df2.to_markdown())
    print(f"\nMarkdown report saved to: eval/results_table.md")

if __name__ == "__main__":
    generate_table()
