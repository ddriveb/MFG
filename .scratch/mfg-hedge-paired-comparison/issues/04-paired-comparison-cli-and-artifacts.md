# Paired comparison CLI, metrics, and artifacts

Type: task
Status: resolved
Blocked by: 15

## Goal

Deliver the user-runnable paired comparison of `../spec.md` sections 13-14: one command producing both arms on one shared trace plus the comparison artifact.

## Scope

- `configs/v1_paired_comparison.json`: self-contained config (base keys + timeline + `healthy_offered_load = 0.5` + `control_window = 25.0` + storm bin width 1.0 + solver parameters + `replay_penalty_regular/urgent`), own schema version, strict loader with single-read hashing (per ADR-0003 conventions). Loads 0.6/0.7 are stress diagnostics only.
- CLI `compare-mfg-hedge-vs-no-hedge --config ... --tokens 1000 [--run-id] [--artifacts-root]`: one config read, one shared trace, arm A then arm B, trace-identity verification between arms (arrivals, classes, attempt-0/1/2 draws), then one unique artifact directory with `manifest.json`, `no_hedge_summary.json`, `mfg_hedge_summary.json`, `comparison.json`.
- Paired metrics per spec section 13: arrival-phase cohorts, completion-phase diagnostics, migration matrix, winner-kind counts, `policy_requested_action_counts` / `applied_action_counts` / `quota_suppressed` / `executor_hedge_suppressed` per window, class, and global (with the section-9 identities asserted), Hedge launch and cancelled counts, work accounting, storm metrics over fixed 1.0 bins (peak/mean is `null` when an arm has zero Hedge launches; sustained overload duration is the longest run of consecutive overloaded bins; the final partial bin normalizes by its actual width and is excluded from peak-rate comparisons).
- Planned-vs-realized reporting per spec section 14: per-window planned expected Hedge work vs budget, realized Hedge work, realized excess/violation, and suppressed counts split into `quota_suppressed` and `executor_hedge_suppressed`, reported separately and never merged.
- Transactional artifact commit: the four files are staged in a temporary sibling directory and moved into place only when all four are written; any failure removes the partial directory; an existing run directory is never overwritten.
- `comparison.json`: A/B absolutes, B−A deltas, relative change (null when A = 0), config identity, trace identity, MFG policy/price/rho/work rates/residuals/convergence per state, tau0, action counts, planned vs realized Hedge work, invariants.

## Acceptance criteria

- Spec section 18 scenarios 19-20 pass, including a hand-computed comparison delta on a manual trace.
- Determinism: two identical invocations produce byte-identical artifacts except `run_id`.
- All artifact protections unchanged (run-id confinement, no overwrite, atomic writes, cleanup).
- In the 1000-Token controlled run the B arm shows nonzero Hedge activity in H/D and zero in F; all invariants hold in both arms.
- `python -m unittest discover -s tests -v` green; `check` exit 0; no third-party dependency; no existing artifact modified.

## Progress log

### Update: 2026-09-04 — Paired comparison CLI, metrics, and artifacts

Status: completed

#### Goal

Deliver the user-runnable paired comparison of `../spec.md` sections 13-15: one command producing both arms on one shared trace plus the transactional four-file artifact.

#### Changed

- `configs/v1_paired_comparison.json` (new): self-contained schema-3 config — all 21 base keys with `healthy_offered_load = 0.5`, timeline 100/200/220, `control_window = 25.0`, `storm_bin_width = 1.0`, `replay_penalty_regular = 1.0`, `replay_penalty_urgent = 5.0`.
- `src/mfg_hedge/config.py`: `PairedExperimentConfig` (strict missing/unknown keys, bool/NaN/Infinity rejection, fixed slice dimensions, timeline validation) with `base_config()`/`timeline()`, plus single-read `load_paired_config_with_sha256`.
- `src/mfg_hedge/artifacts.py`: `write_run_directory` — staged sibling directory, per-file atomic writes, single rename commit, full cleanup on failure, no overwrite, same run-id confinement.
- `src/mfg_hedge/paired_metrics.py`: `build_arm_summary` (overall/per-arrival-phase/per-completion-phase latency, migration matrix, cumulative queue delay, replay rate, action counts global/per-class/per-window, hedge lifecycle counters, winner kinds, full work accounting, per-Replica utilization, live queue snapshots, storm metrics with fixed bin rules, planned-vs-realized per window, invariants) and `build_comparison` (absolutes, B−A deltas, relative change with null when A = 0).
- `src/mfg_hedge/paired.py`: `verify_trace_identity` (per-stream, fail-loud), `trace_identity_fingerprint`, `compute_tau0`, and `run_paired_comparison` (coverage gate, official solver gate, quota, both arms on one shared trace, result-vs-trace alignment guards).
- `src/mfg_hedge/hedge_simulation.py`: additive `hedge_suppressed_events` diagnostic (time, token) so executor suppression is attributable per window; engine semantics unchanged.
- `src/mfg_hedge/__main__.py`: `compare-mfg-hedge-vs-no-hedge` command — single config read, production 70-cell table, official gate before any artifact, transactional commit.
- `src/mfg_hedge/__init__.py`, `pyproject.toml`: new public API exports; version 0.10.0 -> 0.11.0.
- Tests: `tests/test_paired_config.py` (6), `tests/test_paired_metrics.py` (9 incl. the hand-computed scenario-20 comparison), `tests/test_paired_cli.py` (8 incl. transactional failure and per-window F-phase zero-hedge assertions).

#### Verification

- Red (before implementation): `Ran 265 tests ... FAILED (errors=3)` — `ImportError: cannot import name 'PairedExperimentConfig'`, `write_run_directory`, `mfg_hedge.paired`/`paired_metrics` missing.
- Green: `Ran 286 tests ... OK`; `.\.venv\Scripts\python.exe -m mfg_hedge check --config .\configs\v1_minimal.json` — exit 0.
- Two real 1000-Token paired runs (production calibration, no mocks): run ids `...-20260904T163623526509Z` and `...-20260904T163839703148Z`; `no_hedge_summary.json`, `mfg_hedge_summary.json`, `comparison.json` byte-identical across runs; `manifest.json` identical except `run_id`.
- Controlled-run acceptance: both arms share one trace (fingerprints equal, alignment guards pass); both complete all 1000 Tokens; arm A zero Hedge activity; arm B hedges in H/D (126 applied) and zero applied Hedge in F; quota (27) and executor (0) suppression reported separately; all invariants true in both arms; solver gate H/D converged, F forced_normal.
- Hand-computed manual trace (scenario 20): all deltas match exactly, including `relative = null` when A = 0; scenario 19 identity/perturbation tests pass.

#### Artifacts

- `artifacts/single_expert_mechanism_check-paired-mfg-vs-no-hedge-t1000-20260904T163623526509Z/{manifest,no_hedge_summary,mfg_hedge_summary,comparison}.json`
- `artifacts/single_expert_mechanism_check-paired-mfg-vs-no-hedge-t1000-20260904T163839703148Z/{...}` (determinism evidence pair; both gitignored)

Real A/B outcome (1000 Tokens, seed 20260901, load 0.5): mean latency 3.4268 -> 3.0966 (-9.6%); P95 21.17 -> 9.06 (-57.2%); P99 29.37 -> 16.17 (-44.9%); P50 1.16 -> 2.23 (+93.2%); replay rate 1.6% -> 0.5% (-68.8%); Hedge launches 0 -> 828 (quota-suppressed 27, executor-suppressed 0); winners A {primary 984, replay 16} vs B {primary 640, hedge 355, replay 5}; total work +57.3% (amplification 1.0043 -> 1.5802); drain duration 1.13 -> 2.50; sustained overload 7.0 in both arms; storm peak launch rate 4.0/bin with peak/mean 4.80 (bounded, no runaway).

#### Decisions and risks

- The tail-latency and replay improvements cost +57% executed work and a worse median — the honest latency/work trade-off, reported as measured; no metric was dropped or rerun for aesthetics.
- Arm B hedges also occur in the R phase (the H policy is reused there by design); F remains exactly zero.
- Peak/mean ratio for arm A is null by rule (zero launches).
- Quota suppressions (27) are planned-budget projections; realized Hedge work exceeded the planned expectation in some windows because realized draws differ from expected costs — both are reported separately per the planned/realized contract.

#### Next

Feature complete through ticket 04. Possible follow-ups (user's call): multi-seed campaign, stress runs at 0.6/0.7, or the baseline-ablation spec.

## Answer

All acceptance criteria pass: schema-3 config, paired metrics with per-window/class/global auditing, transactional four-file artifacts, trace identity verification, hand-computed deltas, determinism across two real runs, and the controlled 1000-Token run meeting every gate. No forbidden scope; no existing artifact modified.

## Comments

### Update: 2026-09-05 — Linear-time storm aggregation correction

Status: completed

#### Goal

Remove a formal-scale performance defect exposed by the 100k-Token campaign
without changing the ticket-04 storm metric definition.

#### Changed

- `paired_metrics._storm_metrics` now pre-bins Primary arrival work and Hedge
  launch work once, then evaluates bins, instead of rescanning every Token and
  attempt for every bin.

#### Verification

- Red scan-count fixture: 10,001 Token iterations for a 10,000.5 horizon.
- Green: at most one Token iteration; hand-computed storm values unchanged;
  final full suite `295 tests ... OK`.

#### Artifacts

- No ticket-04 artifact was overwritten. The correction enabled the separate
  load-0.5 campaign artifact recorded in its own ticket.

#### Decisions and risks

- This is a complexity correction only: bin width, half-open boundaries,
  demand definition, capacity lookup, partial-bin handling, and all reported
  fields are unchanged.

#### Next

None.
