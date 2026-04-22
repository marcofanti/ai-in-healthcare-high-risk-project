# Evaluation — MedHuggingGPT on PathMMU

End-to-end evaluation of the MedHuggingGPT ensemble against the [PathMMU](https://github.com/PathMMU-Benchmark/PathMMU) benchmark. The pipeline mirrors the actual runtime flow of the app: images are fed to each selected model, outputs are passed to the LLM synthesizer, and the synthesized answer is scored against PathMMU ground truth.

---

## Directory layout

```
eval/
├── main.py                    # CLI entry point
├── adapters.py                # Model wrappers (CLIP, VLM, other)
├── parse.py                   # Answer extraction from LLM output
├── configs/
│   └── default.yaml           # LLM judge config (provider, model)
├── scripts/                   # Pre-built experiment shell scripts
│   ├── ensemble_mac.sh        # UI-default top-2 models, all tiny categories
│   ├── clip_ensemble_mac.sh   # BiomedCLIP+CONCH+MUSK on PubMed test
│   ├── clip_ensemble_mac2.sh  # BiomedCLIP+CONCH+MUSK on Atlas test
│   ├── clip_ensemble_mac3.sh  # BiomedCLIP+CONCH+MUSK on PathCLS test
│   ├── clip_ensemble_mac4.sh  # BiomedCLIP+CONCH+MUSK on EduContent test
│   └── full_ensemble_mac.sh   # 5-model in-process ensemble, all tiny categories
├── outputs/                   # One subdirectory per experiment (auto-created)
│   └── <exp_name>/
│       ├── summary.json       # Aggregated accuracies across all categories
│       └── <Category>/
│           ├── output.json    # Per-sample predictions and raw model outputs
│           ├── output.csv     # Same, tabular
│           ├── output_viz.html# Per-sample visual browser
│           └── result.json    # Category-level accuracy breakdown
├── pathmmu_reference.json     # Reference results from PathMMU paper (editable)
├── build_comparison_table.py  # Generates comparison_table.html from outputs + reference
├── comparison_table.html      # Generated output — Table 1 (yours) + Table 2 (paper)
├── all-summary.html           # Dashboard view of all experiments
└── Eval.md                    # This file
```

---

## Running evaluations

All commands are run from the **project root**.

### Quick smoke test (n=1)

```bash
uv run eval/main.py \
  --models biomedclip conch musk \
  --exp_name my_test \
  --data_path "../PathMMU/data" \
  --categories pdt \
  --n 1
```

### Pre-built scripts

```bash
# UI-default top-2 models, all Tiny categories
bash eval/scripts/ensemble_mac.sh

# CLIP-only ensemble on PubMed (full test set)
bash eval/scripts/clip_ensemble_mac.sh

# CLIP-only on Atlas / PathCLS / EduContent (full test sets)
bash eval/scripts/clip_ensemble_mac2.sh   # Atlas
bash eval/scripts/clip_ensemble_mac3.sh   # PathCLS
bash eval/scripts/clip_ensemble_mac4.sh   # EduContent

# 5-model in-process ensemble (slow — set N=20 for testing)
N=20 bash eval/scripts/full_ensemble_mac.sh
```

### Custom run

```bash
uv run eval/main.py \
  --models <model1> <model2> ...   # see Models section
  --exp_name <name>                # output goes to eval/outputs/<name>/
  --data_path "../PathMMU/data"
  --categories <code1> <code2> ... # see Categories section
  --n <int>                        # 0 = all samples (default)
```

Override the PathMMU data path via env var:

```bash
PATHMMU_DATA=/absolute/path/to/PathMMU/data bash eval/scripts/clip_ensemble_mac.sh
```

---

## Available models

| Flag value | Full name | Type |
|---|---|---|
| `biomedclip` | BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 | CLIP |
| `conch` | CONCH (MahmoodLab) | CLIP |
| `musk` | MUSK (xiangjx) | CLIP |
| `medgemma` | MedGemma | VLM |
| `vit_alzheimer` | ViT-Alzheimer | VLM |
| `chexagent` | CheXagent | subprocess |
| `llava_med` | LLaVA-Med | subprocess |

CLIP models run in-process and are fast. `chexagent` and `llava_med` spawn subprocesses and are significantly slower.

When `--models` is omitted, the harness uses the same top-2 defaults the UI applies per modality:

| Modality | Default models |
|---|---|
| 2D Histopathology | `conch`, `musk` |
| Unknown | `biomedclip`, `llava_med` |

---

## Category short codes

Pass one or more to `--categories`:

| Code | Full category | Split |
|---|---|---|
| `pdtt` | PubMed_test_tiny | Tiny |
| `pdt` | PubMed_test | All |
| `pdd` | PubMed_val | Val |
| `clstt` | PathCLS_test_tiny | Tiny |
| `clst` | PathCLS_test | All |
| `clsd` | PathCLS_val | Val |
| `att` | Atlas_test_tiny | Tiny |
| `at` | Atlas_test | All |
| `ad` | Atlas_val | Val |
| `edutt` | EduContent_test_tiny | Tiny |
| `edut` | EduContent_test | All |
| `edud` | EduContent_val | Val |
| `sptt` | SocialPath_test_tiny | Tiny |
| `spt` | SocialPath_test | All |
| `spd` | SocialPath_val | Val |

---

## Output structure

Each experiment writes to `eval/outputs/<exp_name>/`:

### `summary.json`

Top-level aggregated result for the experiment:

```json
{
  "exp_name": "PubMed_Tiny",
  "models_override": ["biomedclip", "conch", "musk"],
  "categories": {
    "PubMed_Tiny": {
      "acc": 0.416,
      "num_example": 281,
      "detailed": {
        "judge_acc": 0.416,
        "max_prob_acc": 0.284,
        "per_model": {
          "microsoft/BiomedCLIP-...": 0.302,
          "MahmoodLab/conch": 0.281,
          "xiangjx/musk": 0.256
        }
      }
    }
  },
  "overall_acc": 0.416,
  "judge_acc": 0.416,
  "max_prob_acc": 0.284,
  "per_model_acc": { "...": 0.302, "...": 0.281, "...": 0.256 },
  "total_samples": 281
}
```

Key fields:

| Field | Description |
|---|---|
| `judge_acc` | Accuracy of the LLM synthesizer answer (primary metric) |
| `max_prob_acc` | Accuracy of the highest-confidence CLIP model (no LLM) |
| `per_model_acc` | Per-model accuracy before synthesis |

### `<Category>/output.json`

List of per-sample dicts with `question`, `choices`, `answer`, `prediction`, `correct`, and raw model outputs.

### `<Category>/output_viz.html`

Visual per-sample browser. Open in any browser.

---

## Generating the comparison table

```bash
uv run eval/build_comparison_table.py
# → writes eval/comparison_table.html
```

Options:

```bash
uv run eval/build_comparison_table.py \
  --outputs eval/outputs \          # directory to scan for summary.json files
  --ref eval/pathmmu_reference.json \  # paper reference data
  --out eval/comparison_table.html
```

The script auto-detects which dataset × split combinations you have run and builds matching columns for both tables. Running new experiments and re-running the script is all that's needed to update the table.

### Table 1 — Your results

- Rows: each individual model + Judge (LLM synthesizer)
- Columns: dataset × split (Tiny / All) for every experiment in `outputs/` + weighted Overall (All splits only, weighted by sample count)
- Bold = best per column

### Table 2 — PathMMU paper reference

- Same column structure as Table 1 (only datasets/splits you have run)
- Overall = weighted by the paper's n values across matched All splits
- Groups: Baselines → open-source LMMs → (separator) → API models

---

## Reference data

`eval/pathmmu_reference.json` stores the paper numbers. Edit it directly to fix misreads or add new models. Structure:

```json
{
  "n": {
    "test_all": 9677,
    "pubmed_tiny": 281, "pubmed_all": 3068,
    "educontent_tiny": 255, "educontent_all": 1938,
    "atlas_tiny": 208, "atlas_all": 1007,
    "pathcls_tiny": 177, "pathcls_all": 1809
  },
  "groups": [
    {
      "label": "Baselines",
      "separator_above": false,
      "models": [
        {
          "name": "Random Choice",
          "test_all": 23.7,
          "pubmed_tiny": 22.1, "pubmed_all": 25.1,
          ...
        }
      ]
    }
  ]
}
```

Keys follow the pattern `<dataset>_<split>` (lowercase, matching the exp_name parsing in `main.py`). `null` means the paper did not report that value. `test_all` is the Test Overall ALL column (n=9677) used as the reference Overall.

---

## Current results

Experiments run as of 2026-04-22. Ensemble: BiomedCLIP + CONCH + MUSK. Judge: Gemini.

| | Overall (All, wtd) | PubMed Tiny | Atlas Tiny | Atlas All | EduContent Tiny | EduContent All | PathCLS Tiny | PathCLS All |
|---|---|---|---|---|---|---|---|---|
| BiomedCLIP | 26.5% | 30.2% | 20.0% | 37.5% | 28.6% | 28.3% | 15.7% | 13.8% |
| CONCH | 25.9% | 28.1% | 46.7% | 31.2% | 27.8% | 30.1% | 14.0% | 16.4% |
| MUSK | 21.1% | 25.6% | 40.0% | 20.8% | 17.6% | 28.2% | 8.3% | 14.3% |
| **Judge** | **44.9%** | **41.6%** | **60.0%** | **45.8%** | **44.7%** | **40.3%** | **51.2%** | **48.6%** |

PubMed All not yet run. The Judge consistently outperforms any individual model by 15–20 pp.

**Paper reference (same columns, GPT-4V top model):**

| | Overall (All, wtd) | PubMed All | Atlas All | EduContent All | PathCLS All |
|---|---|---|---|---|---|
| Random Choice | ~22% | 25.1% | 19.7% | 25.6% | 16.3% |
| GPT-4V | ~49% | 53.5% | 52.8% | 53.6% | 33.8% |
| **Judge (ours)** | **44.9%** | — | **45.8%** | **40.3%** | **48.6%** |

The ensemble Judge sits between GPT-4V and the open-source LMMs on the All splits, and above all open-source baselines.
