# MFG-Hedge v1 Project Rules

These instructions apply to the entire `mfg_hedge_v1` project.

## Agent skills

### Issue tracker

Issues and specs use Local Markdown under `.scratch/<feature-slug>/`. See `docs/agents/issue-tracker.md`.

### Domain docs

This is a single-context project. Read `CONTEXT.md` and the relevant ADRs under `docs/adr/` before changing domain behavior. See `docs/agents/domain.md`.

## Working agreement

### Before changing anything

1. Read `CONTEXT.md`, the relevant ADRs, and the active spec/ticket.
2. Create or claim exactly one ticket under `.scratch/<feature-slug>/issues/`.
3. Confirm the ticket has a bounded scope and executable acceptance criteria.
4. Inspect the working tree and preserve unrelated user changes.

### Repository hygiene

- Keep all work inside this project directory.
- Do not place temporary scripts, logs, plots, datasets, or experiment results in the project root.
- Keep temporary planning and work history under `.scratch/<feature-slug>/`.
- Keep reusable code in `src/mfg_hedge/`, tests in `tests/`, and versioned configurations in `configs/`.
- Write generated experiment output only under `artifacts/<run-id>/`; never overwrite a previous run.
- Do not commit `.venv`, caches, editable-install metadata, or generated artifacts.
- Do not add a third-party dependency without explicit user approval and an ADR explaining why the standard library is insufficient.

### Implementation discipline

- Deliver one vertical slice at a time. Do not implement adjacent roadmap items unless the active spec includes them.
- For behavior changes, follow red-green-refactor: add a failing test, implement the smallest change, then clean up while tests remain green.
- Keep calibration/solver models separate from the event simulator used for evaluation.
- Preserve common random numbers across policies. A policy must not change the generated arrival, token-class, service-time, or failure trace.
- Do not silently change metric definitions, configuration semantics, random seeds, or experimental assumptions.
- Use the domain terms defined in `CONTEXT.md`; do not call same-Expert replica protection Expert rerouting.
- Record a durable architectural or scientific decision in `docs/adr/` before relying on it.

### Required verification

At minimum, every code update must run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s .\tests -v
.\.venv\Scripts\python.exe -m mfg_hedge check --config .\configs\v1_minimal.json
```

Run any additional focused test named by the active ticket. Never report a command as passing unless it was actually executed in the current update.

### Required update format

Append every work-session update to the active ticket's `## Progress log` using `docs/agents/update-format.md`. Use the same headings in the final user-facing report.

