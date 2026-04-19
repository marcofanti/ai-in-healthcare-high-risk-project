# PRD-002: Cross-Machine Path Portability

**Status:** Open  
**Priority:** High (demo risk)  
**Owner:** Unassigned  
**Target Sprint:** Week 4 (before demo)

---

## Problem Statement

`datasets_config.json` stores absolute file paths (e.g. `/Users/mfanti/Masters_.../workspace/mock_oasis/sample.img`). The config is committed to git. When any other team member clones the repo, or when the demo is run on a different machine, every path in the config is broken — the UI shows no files and the analysis cannot run.

This is a **demo-day risk**: the demo machine will almost certainly have a different username and directory structure.

---

## Goals

- `datasets_config.json` works on any machine that has the repo checked out.
- Team members can add datasets to their local config without creating merge conflicts from absolute paths.
- The `manage_datasets.py add` command produces portable paths by default.

## Non-Goals

- Network or cloud path support.
- Automatic dataset downloading or syncing across machines.

---

## Proposed Solution

**Option A — Relative paths (recommended for MVP):**
Store paths relative to the project root. The app resolves them at runtime using `Path(__file__).parent / stored_path`.

```json
{
  "OASIS_MRI": {
    "modality": "Legacy MRI",
    "files": [
      {"path": "workspace/mock_oasis/sample.img", "type": "img"}
    ]
  }
}
```

`manage_datasets.py` converts absolute paths to relative on write. The app resolves to absolute on read. No user-visible change.

**Option B — Environment variable prefix:**
Store paths as `${WORKSPACE_ROOT}/mock_oasis/sample.img`. The app expands the variable at runtime. More flexible but requires every user to set the variable.

**Option C — Per-user local config override:**
`datasets_config.json` is a template committed to git. Each user maintains a `datasets_config.local.json` (gitignored) that overrides it. Most flexible but adds setup friction.

**Recommendation:** Option A for the demo. Option C if the team wants to maintain different datasets per developer long-term.

---

## Acceptance Criteria

- [ ] Clone the repo on a fresh machine, run the app — at least the `workspace/mock_oasis` dataset is visible without editing any config file.
- [ ] `manage_datasets.py add` stores relative paths by default.
- [ ] Absolute paths passed to `add` are automatically converted to relative if they fall within the project directory.
- [ ] Paths outside the project root produce a warning and are stored as-is (with a note in the UI).
- [ ] No regression: existing analyses with resolved paths continue to work.

---

## Open Questions

- Should paths outside the project root be blocked entirely, or allowed with a warning?
- Does the `datasets_config.json` in git serve as a shared "team template" of known datasets, or is it machine-specific? This determines whether Option A or C is the right long-term choice.
