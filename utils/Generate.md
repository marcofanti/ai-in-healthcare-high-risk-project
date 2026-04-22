# Results Table Generation

This document describes the `utils/generate_table.py` script, which aggregates evaluation results into a formatted comparison table.

## Purpose
The script creates two comparison tables:
1.  **Table 1: Current Results** - Shows the performance of your local model configurations (CONCH, BiomedCLIP, musk) and the LLM Ensemble (Judge).
2.  **Table 2: Reference Benchmarks** - Shows the performance of various models from the PathMMU benchmark paper (including Baselines like Expert performance and API models like GPT-4V).

## Usage
Run the script from the project root using `uv`:

```bash
uv run utils/generate_table.py
```

## Data Sources
-   **Local Results**: Read from `eval/outputs/*/summary.json`. The script automatically parses the directory names to determine the Dataset (e.g., Atlas, PubMed) and the Split (Tiny, All).
-   **Reference Data**: Read from `eval/pathmmu_reference.json`, which contains hardcoded values from the PathMMU paper.

## Table Structure
-   **Rows**: Individual models and the Ensemble/Judge.
-   **Columns**: Datasets (Atlas, EduContent, PathCLS, PubMed).
-   **Sub-columns**: Splits (Tiny, All).
-   **Overall Column**: Calculates the average accuracy across all "All" splits available for that row.

## Outputs
-   **Terminal**: Prints the formatted tables using the `tabulate` library.
-   **Markdown**: Saves a copy of the tables to `eval/results_table.md`.
