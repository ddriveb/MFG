# Design the minimal MFG-Hedge vs No-Hedge paired comparison

Type: task
Status: resolved
Blocked by: none

## Goal

Produce the complete design for the paired-comparison feature: `../spec.md`, implementation tickets 01-04 (blocking chain 01 -> 02 -> 03 -> 04), and Proposed ADR-0005/0006. No Python code, no experiments, no artifacts this round.

## Scope

- Fix every design decision enumerated in the feature request: A/B trace identity, unified hedge-capable engine with No-Hedge regression, attempt-2 random keys, Backup pairing, three action semantics, winner/cancellation statuses, Hedge-Replay interaction cases, hedge delay `tau0`, quota rounding, calibration separation, minimal MFG solver, the degenerate-defaults analysis, paired metrics, and the comparison output layout.
- Leave clearly-marked confirmation items for the user (listed in spec section 17).

## Acceptance criteria

- `../spec.md` fixes all decisions; nothing is left to implementer improvisation.
- Tickets 01-04 exist with bounded scopes and executable acceptance criteria; 02 blocked by 01, 03 blocked by 02, 04 blocked by 03.
- ADR-0005 and ADR-0006 exist with Status: Proposed.
- No production code, config, test, or artifact changes; the full suite and `check` still pass.

## Progress log

### Update: 2026-09-04 — Design the paired-comparison feature

Status: completed

#### Goal

Deliver the complete design for the minimal MFG-Hedge vs No-Hedge paired comparison: spec, four implementation tickets with the 01 -> 02 -> 03 -> 04 blocking chain, and Proposed ADR-0005/0006. No code, no experiments, no artifacts.

#### Changed

- `../spec.md`: fixed all enumerated decisions — A/B arm definitions and trace identity; unified hedge-capable engine with the No-Hedge Token-for-Token regression; attempt-2 CRN (`hedge_service_times`, label `service:{replica}:attempt2`); fixed Backup pairing with feasible-set projection (`{Normal}` in F, suppressions counted); Normal/Delayed/Immediate semantics; conservative winner/cancellation with five explicit attempt statuses; the five-case Hedge-Replay case law; `tau0 = exp(mu + sigma * NormalDist().inv_cdf(0.9))` from arrival; deterministic quota (W = 25.0, largest remainder, tie-break N<D<I, budget demotion Immediate→Delayed→Normal by descending token_id); calibration separation (independent `calibration:` namespace, 400 Tokens per cell, utilization grid {0.5, 0.7, 0.9, 1.1, 1.3}, linear interpolation with clamping); the minimal solver (soft BR, damping, price update, max_iterations 200, residuals 1e-6, explicit non-convergence); the degenerate-defaults analysis (load 0.7 ⇒ D/F budgets zero; recommended 0.6 via a new config); paired metrics incl. arrival cohorts, migration matrix, storm bins (width 1.0, half-open); and the four-file artifact layout with comparison deltas.
- Tickets 01-04 created open with the required blocking chain.
- `docs/adr/0005-hedge-copy-lifecycle-and-random-keys.md` and `docs/adr/0006-finite-action-mfg-and-quota-semantics.md` created with Status: Proposed.
- No production code, tests, configs, or artifacts touched.

#### Verification

- Read before designing: AGENTS.md, CONTEXT.md, issue-tracker.md, update-format.md, the design review (standing in for the absent `pasted-text.txt`), the common-state spec and tickets 00-06, ADR-0001..0004, all current source modules, `configs/v1_common_state.json`, and both schema-2 artifacts.
- `.\.venv\Scripts\python.exe -m unittest discover -s .\tests -v` — `Ran 152 tests ... OK` after the design work (proves zero code change).
- `.\.venv\Scripts\python.exe -m mfg_hedge check --config .\configs\v1_minimal.json` — exit 0.

#### Artifacts

- `.scratch/mfg-hedge-paired-comparison/spec.md`, `issues/00..04-*.md`
- `docs/adr/0005-hedge-copy-lifecycle-and-random-keys.md`, `docs/adr/0006-finite-action-mfg-and-quota-semantics.md` (both Proposed)

#### Decisions and risks

- The degenerate-defaults problem is surfaced, not hidden: at load 0.7 the D/F Hedge budgets are zero and arm B collapses to arm A there; the spec recommends the 0.6 mechanism config and keeps 0.7 as a stress test (confirmation item 1).
- Nine parameters are flagged for user confirmation (spec section 17) before ticket 01 starts.
- `pasted-text.txt` does not exist in the workspace; the design review document and the request text were used instead.

#### Next

User confirms ADR-0005/0006 and the nine spec section-17 items; then claim ticket 01.

## Answer

All acceptance criteria pass: spec fixes every enumerated decision; tickets 01-04 open with the correct blocking chain; ADR-0005/0006 Proposed; 152/152 tests and `check` green with zero code/config/artifact changes.

## Comments

- 2026-09-04: the feature request's reading list mentions `pasted-text.txt`; no such file exists in the workspace. The original research content was taken from `docs/design_review_zh.md` plus the request text itself. Flagged to the user in the final report.
