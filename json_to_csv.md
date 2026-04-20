# JSON to CSV Dataset Generator

This utility script converts the agentic evaluation outputs (JSON format) from the `ai-in-healthcare-high-risk-project` into structured `.CSV` files. This is designed to facilitate downstream statistical analysis and dataset preparation for pathology-based Vision-Language Model (VLM) research.

## Purpose
The script extracts ground truth answers, individual model predictions (with associated probabilities), ensemble probability sums, and the final decision made by the LLM Judge.

## Requirements
- **Python 3.10+**
- Standard libraries: `json`, `csv`, `argparse`, `os`
- No external dependencies are required for the script itself, but `uv` is recommended for execution to ensure a clean environment.

## Data Structure in CSV
The generated CSV contains the following columns for each test case:

1.  **Metadata**: `No`, `Question`.
2.  **Ground Truth**: `Right Answer` (as a Letter: A, B, C, or D).
3.  **Model Performance (Dynamic)**: For every model found in the JSON (e.g., `conch`, `musk`):
    - `{model}_Correct`: `T` or `F` based on whether the model's top prediction matched the ground truth.
    - `{model}_Prob_{A-D}`: The specific probability assigned by that model to each of the four choices.
4.  **Ensemble Logic**: 
    - `Sum_Prob_{A-D}`: The sum of probabilities for each choice across all tested models.
5.  **Judge Decision**:
    - `Judge Answer`: The final letter choice (A, B, C, or D) decided by the LLM Judge.

## Usage with `uv`

Using `uv run` is the fastest way to execute the script without manually managing a virtual environment.

### Run on a single file:
```bash
uv run utils/json_to_csv.py eval/outputs/ui_default_top2_pathmmu/PubMed_test_tiny/output.json
```

The script will generate an `output.csv` in the **same directory** as the input JSON file.

## Logic Details
- **Dynamic Model Handling**: The script scans the entire JSON file first to identify all unique models. This ensures that even if different entries used different models, the CSV header remains consistent.
- **Probability Mapping**: If a model's `top5` output is missing a specific choice (A, B, or C), the probability is defaulted to `0.0`.
- **Letter Extraction**: The script uses a robust extraction logic to find choice letters from various formats (e.g., "A", "A)", "(A)", or "Answer: A").
