# PRD-003: Synthetic / Mock Output Labeling in Clinical Reports

**Status:** Open  
**Priority:** High (safety / academic integrity)  
**Owner:** Unassigned  
**Target Sprint:** Week 4

---

## Problem Statement

The risk mitigation strategy documented in the README states: if heavy inference (e.g. MONAI 3D segmentation) exceeds 60 seconds, the modality tool returns a `time.sleep(3)` delay and a pre-computed mock output instead of real analysis results.

Currently, this mock output flows into the Synthesizer node and is rendered as a clinical report with no indication that the underlying data is synthetic. A reader of the report cannot distinguish:

- Real model inference on the submitted image
- A pre-computed result from a different image (or fabricated entirely)

In a high-risk AI healthcare context, presenting synthetic outputs as real clinical findings — even in an academic prototype — undermines the audit trail and misrepresents the system's actual capabilities.

---

## Goals

- Any clinical report that incorporates mock/pre-computed model output is clearly and visibly labeled as containing synthetic data.
- The label is machine-readable (in the JSON output) and human-readable (in the rendered report).
- The Synthesizer node is aware of which outputs are synthetic and includes this in its consensus analysis.

## Non-Goals

- Eliminating mock outputs — they remain a valid latency mitigation strategy.
- Validating whether a model's output is clinically accurate (out of scope for MVP).

---

## Proposed Solution

**Step 1 — Tag mock outputs at the tool level:**
Each tool function that returns a mock result must include a `"synthetic": true` field in its JSON output dict.

```python
# In tool_volumetric_segmenter.py (example)
if elapsed > 60:
    return {
        "model": "monai_segmenter",
        "synthetic": True,
        "synthetic_reason": "inference_timeout",
        "prediction": precomputed_result,
    }
```

**Step 2 — Propagate through AgentState:**
The `model_outputs` list already carries the full tool JSON. No state schema change needed — `synthetic` travels with the output.

**Step 3 — Synthesizer prompt awareness:**
Update the synthesis prompt in `synthesizer_node` to check for synthetic outputs and include a mandatory disclosure:

```
If any model output contains "synthetic": true, you MUST begin the report with a
prominent warning block stating which models returned synthetic/pre-computed results
and that those findings should not be used for clinical interpretation.
```

**Step 4 — UI badge:**
In the Model Evidence (Audit Trail) tab, show a "SYNTHETIC" badge on any expander whose output contains `"synthetic": true`.

---

## Acceptance Criteria

- [ ] Any tool returning a mock/pre-computed result sets `"synthetic": true` in its output dict.
- [ ] The Synthesizer report includes a visible warning section when any synthetic output is present.
- [ ] The warning names the specific model(s) that returned synthetic data.
- [ ] The Audit Trail tab in the UI visually distinguishes synthetic outputs from real ones.
- [ ] A report with zero synthetic outputs shows no warning (no false positives).
- [ ] Unit test: a `model_outputs` list containing one synthetic entry produces a report with the warning; a list with no synthetic entries does not.

---

## Open Questions

- Should the presence of any synthetic output prevent the "Run Ensemble Analysis" button from being re-labeled as "Completed" — e.g. show "Completed (partial)" instead?
- Should `synthetic_reason` be shown to the user in the UI, or kept internal to the audit log only?
