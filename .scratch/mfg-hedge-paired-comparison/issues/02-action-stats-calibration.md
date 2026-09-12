# ActionStats calibration

Type: task
Status: resolved
Blocked by: 01

## Goal

Build the coarse `ActionStatsEstimator` of `../spec.md` section 11: the solver-side calibration table, fully separated from evaluation.

## Scope

- Representative-token / tagged-probe calibration through the ticket-01 engine: a cell's rho is the background total work rate (background Primary + Replay) over state capacity faced by the probe, excluding the probe's own Primary/Hedge/Replay work. This is a finite-action mean-field approximation closing queueing cost through aggregate rho only.
- Frozen, independent `calibration:`-namespace background traces per cell; within-cell CRN — the N/D/I probes share one background trace and one set of probe attempt-0/1/2 draws; no reads of evaluation traces or A/B artifacts.
- Real H→D→F→R transitions in every episode: D-cohort probes genuinely cross the D→F boundary; in-flight work at the boundary fails, Replays, or is Hedge-protected per engine semantics; F-state cells contain Normal only.
- Deterministic arrival-rate adjustment: bisection on the background arrival rate over bounds [0.05, 3.0], at most 40 iterations, fixed per-cell seeds, fail-fast on non-convergence; acceptance `abs(achieved_rho - target_rho) <= 0.02`.
- Per-cell records: `target_rho`, `achieved_rho`, `rho_error`, `background_arrival_rate`, `sample_count` (>= 400 probe cohort samples, accumulated across episodes) plus `ActionStats(mean_latency, replay_probability, deadline_miss_probability = 0.0, expected_hedge_work, expected_replay_work)`; grid {0.5, 0.7, 0.9, 1.1, 1.3} with linear interpolation, endpoint clamping, and missing-point errors; byte-stable deterministic serialization.
- Deterministic acceptance fixture: at least one D-state cell where Normal and some Hedge action differ in `q` or `expected_replay_work`; calibration fails fast otherwise. Never reverse-fit from evaluation results.

## Acceptance criteria

- Spec section 18 scenario 16 passes: a D-cohort cell where Normal differs from a Hedge action in `q` or `expected_replay_work`; every populated cell reports `sample_count >= 400`; the acceptance fixture fails fast when the contrast vanishes.
- Rho contract: every populated cell records target/achieved rho, rho_error <= 0.02, background arrival rate; bisection respects the fixed seeds, bounds, and 40-iteration cap and fails fast on non-convergence; within-cell CRN is proven by reusing one background and one probe draw set across N/D/I.
- Determinism: two calibration runs produce byte-identical tables; namespace separation test proves calibration draws never perturb evaluation streams at the same base seed.
- Interpolation/clamping/missing-point rules covered by unit tests.
- `python -m unittest discover -s tests -v` green; no evaluator or CLI changes; no third-party dependency.

## Progress log

### Update: 2026-09-04 — ActionStats calibration

Status: completed

#### Goal

Implement the representative-token / tagged-probe ActionStats calibration per spec section 11 and ADR-0006 (now Accepted), fully separated from the final A/B evaluation. No MFG solver, price iteration, quota, paired CLI/config, comparison metrics, or experiments.

#### Changed

- `docs/adr/0006-...`: Status -> Accepted (confirmed by the user on 2026-09-04); decision content untouched.
- New `src/mfg_hedge/calibration.py`: `CalibrationCell` / `CalibrationTable` / `CalibrationEstimate` (strict key uniqueness, full float/bool validation, linear interpolation with clamp flags, missing-point errors, deterministic byte-stable `to_json_bytes`); `ProbeSample` + `aggregate_probe_samples` (executed_work-based, zero-fill, deadline_miss fixed 0.0); public `background_work_rate` pinning the spec's rho formula (arrival-state attribution, attempt-0+1 executed work only, window-duration denominator); `find_background_rate` deterministic bisection ([0.05, 3.0], <= 40 iterations, tolerance 0.02, first-in-tolerance midpoint wins, bracket pre-check, fail-fast non-convergence); `calibrate_cohort` (frozen calibration-namespace episodes, one tagged probe per engine run, N/D/I variants share the identical trace and differ only in action, F cells Normal-only); `build_calibration_table` (production entry enforcing >= 400 samples per cell + contrast check); `validate_contrast` fail-fast.
- `src/mfg_hedge/__init__.py`: exported the calibration API; version 0.4.0 -> 0.5.0 (`pyproject.toml` synced) — new public API, no existing behavior changed, per the ADR-0002 versioning practice.
- New `tests/test_calibration.py` (20 tests), written before implementation.

#### Verification

- Red (before implementation): `Ran 186 tests ... FAILED (errors=1)` — `ModuleNotFoundError: No module named 'mfg_hedge.calibration'`.
- Green: `.\.venv\Scripts\python.exe -m unittest discover -s .\tests -v` — `Ran 205 tests ... OK` (185 + 20 new). Two test-side expectation fixes along the way: bisection returns the first midpoint within tolerance (not the exact root), and `places=2` was too strict for a tolerance-based result; both corrected to match the documented stop rule.
- `.\.venv\Scripts\python.exe -m mfg_hedge check --config .\configs\v1_minimal.json` — exit 0.
- Recorded probes (D cohort, target rho 0.7, Regular, timeline 10/20/22, 400 samples per action, 8 episodes):
  1. Real 400-sample D cell: all three actions at sample_count=400.
  2. Contrast: Normal q=0.1325 / replay_work=0.1185; Delayed q=0.0625 / 0.0585; Immediate q=0.0000 / 0.0000 — Normal vs Immediate clearly separated.
  3. Byte-identical rerun of the cohort serialization: True.
  4. Bisection: target 0.7, achieved 0.690209, error 0.009791 (<= 0.02), background arrival rate 0.775977.
  5. Evaluation namespace untouched: `generate_workload_with_hedge` identical before/after calibration.
- rho formula pinned by name and hand-computed test: `test_background_rho_counts_executed_work_by_arrival_state_per_window` (D window work 6.0/10/1.5 = 0.4; F arrival attributed to F even when executed in R).
- No hedge-engine semantics were modified; no artifacts written; no third-party dependency.

#### Artifacts

None generated. Source: `src/mfg_hedge/calibration.py` (new), `src/mfg_hedge/__init__.py`, `pyproject.toml`. Tests: `tests/test_calibration.py` (new).

#### Decisions and risks

- Calibration timeline is a scaled-down (10/20/22) mechanism-verification timeline; probe positions are evenly spaced per episode; episode count fixed at 8; all recorded in code constants for later tuning.
- rho measurement uses probe-free background episodes from the same labels; probe runs insert one tagged probe (dense re-indexing), so background draws beyond the probe slot shift — a documented aggregate-level mean-field caveat, while within-sample N/D/I CRN is exact.
- Version 0.5.0 because `CalibrationTable`/`CalibrationCell`/`calibrate_cohort`/`build_calibration_table` are public exports.
- A full 70-cell production table is intentionally not exercised in unit tests (performance); layering plus one real D cell covers the path.

#### Next

Claim ticket 03 (`03-mfg-solver-and-quota.md`): the load fixed point, price iteration, and the two-phase quota allocator over this calibration table.

## Answer

All acceptance criteria pass: calibration is fault-aware (D cohorts cross D→F with real failure/Replay/Hedge semantics), probe-isolated, CRN-exact within cells, deterministic to the byte, with the rho formula pinned and >= 400 samples per production cell. 205/205 tests green; `check` exit 0; ADR-0006 Accepted; ticket 02 resolved; tickets 03/04 remain open; no solver/price/quota/CLI/experiment work.

## Comments

- 2026-09-04 (ticket 08): two calibration defects were corrected after this ticket resolved — (1) probe insertion renumbered/redrew background Tokens; replaced by designated-probe semantics (existing in-window Token, stable probe draw keys, no parity/draw perturbation); (2) the F window was too short for the rho bisection to meet tolerance; rho measurement now extends episodes until >= 400 in-window background cohort samples, with per-(state, rho) caching and full bisection logging. See `.scratch/mfg-hedge-paired-comparison/issues/08-harden-calibration-crn-and-production-path.md`.
- 2026-09-04 (ticket 09): two further corrections — the default calibration timeline is now the real 100/200/220 evaluation timeline (the earlier 10/20/22 scaling silently changed dimensionless queue load and failure exposure), and probe selection is now a stable uniform pick over the in-window arrival cohort instead of the midpoint-biased nearest-token rule; background generation is time-horizon based. See `.scratch/mfg-hedge-paired-comparison/issues/09-align-calibration-timeline-and-arrival-cohorts.md`.

