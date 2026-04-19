# PRD-001: Session Recovery on Browser Close

**Status:** Open  
**Priority:** Medium  
**Owner:** Unassigned  
**Target Sprint:** Post-Week 4 (post-demo)

---

## Problem Statement

The LangGraph agent uses `MemorySaver()` as its checkpointer, which stores graph state exclusively in RAM. If the user closes the browser tab, refreshes the page, or the Streamlit server restarts mid-analysis, the entire session is lost — including in-progress model outputs, the HITL approval state, and any synthesized report that was not manually copied.

For a clinical tool, this creates two risks:
1. **Completed analysis is silently lost** — the user has no way to retrieve a report they were viewing.
2. **The HITL approval state is unrecoverable** — if the user approved a plan and the server died before synthesis completed, there is no way to resume or audit what was approved.

---

## Goals

- A user who closes and reopens the browser can resume or retrieve their last session.
- A completed clinical report survives a server restart.
- The approved HITL execution plan is persisted for audit purposes.

## Non-Goals

- Multi-user session management.
- Cloud storage or remote persistence (out of scope for MVP — local only).
- Automatic re-execution of a failed analysis.

---

## Proposed Solution

Replace `MemorySaver()` with a file-based or SQLite checkpointer provided by LangGraph.

**Option A — SQLite checkpointer (recommended):**
LangGraph ships `SqliteSaver` which writes state to a local `.db` file. Minimal code change:

```python
from langgraph.checkpoint.sqlite import SqliteSaver

memory = SqliteSaver.from_conn_string("./workspace/checkpoints.db")
app = workflow.compile(checkpointer=memory)
```

Session IDs (`thread_id`) already exist in the app. On page load, Streamlit can check for an existing thread and offer "Resume last session."

**Option B — JSON file checkpointer:**
Serialize `AgentState` to a JSON file keyed by `thread_id` after each node completes. Simpler but requires custom implementation and does not benefit from LangGraph's built-in replay.

**Recommendation:** Option A. `SqliteSaver` is already in the LangGraph dependency tree and requires fewer than 5 lines of change.

---

## Acceptance Criteria

- [ ] Closing and reopening the browser within the same local session restores the last known agent state.
- [ ] A completed clinical report is visible after a Streamlit server restart.
- [ ] The `thread_id` persists across page loads (stored in browser via `st.query_params` or a cookie-equivalent).
- [ ] The approved HITL execution manifest is queryable from the checkpoint store.
- [ ] No regression in the happy-path flow (new analysis, full run, results displayed).

---

## Open Questions

- Should old checkpoints be automatically purged after N days, or left to the user?
- Should the UI show a "Resume" prompt on load, or silently restore state?
