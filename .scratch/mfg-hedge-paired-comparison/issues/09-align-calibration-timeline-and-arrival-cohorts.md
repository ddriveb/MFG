# Align calibration timeline and arrival cohorts

Type: task
Status: resolved
Blocked by: none

## Goal

Fix the calibration timeline mismatch (10/20/22 vs the real 100/200/220) and the probe arrival-cohort bias (midpoint selection), switch background generation to time-horizon form, and add episode audit fields. Tickets 02/08 stay resolved; history untouched.

## Scope

1. Default calibration timeline becomes 100/200/220 (the paired-evaluation timeline); no partial scaling of rates/service/tau0.
2. Background generation is time-horizon based: unit-rate prefix scaled by rate, generate while arrival < recovered_start, guard zero-token and runaway cases.
3. Probe selection is a stable SHA-256-seeded uniform pick over in-window arrivals; quartile acceptance on relative positions.
4. Rho contract unchanged (tolerance 0.02, bounds [0.05, 3.0], 40 iterations); bisection endpoints join the best-candidate comparison.
5. Cell audit fields: rho_measurement_episode_count, rho_background_cohort_count, probe_successful_episode_count, probe_generation_attempt_count, probe_skipped_episode_count with the attempts = successful + skipped identity; production requires successful == sample_count.
6. Version 0.6.0 -> 0.7.0 (calibration schema/provenance and default timeline change; no released artifact compatibility at stake).

## Acceptance criteria

- Red evidence: midpoint bias quartile failure, timeline mismatch, high-rate horizon starvation on the old fixed-count generator.
- After the fix: quartile acceptance passes on 400-sample cohorts; real-timeline table builds twice byte-identically with max rho_error <= 0.02.
- Full suite and `check` pass; no solver/price/quota/CLI/config/experiment work; no artifacts written.

## Progress log

### Update: 2026-09-04 — Align calibration timeline and arrival cohorts

Status: completed

#### Goal

Align calibration with the real evaluation timeline (100/200/220), replace midpoint-biased probe selection with a stable uniform pick over the in-window arrival cohort, switch background generation to time-horizon form, and add episode audit fields.

#### Changed

- `src/mfg_hedge/calibration.py`: `CALIBRATION_TIMELINE` is now (100, 200, 220); `build_calibration_background(label, rate, config, horizon)` generates from a unit-rate prefix until the next arrival reaches the horizon (guards: illegal rate/horizon, runaway cap, zero-Token episodes allowed but unusable downstream); `prepare_probe_trace` picks uniformly via `Random(seed(probe_label:pick))` over in-window arrivals; `find_background_rate` accepts endpoints meeting tolerance (low checked first) and includes endpoints in the audit log; `CalibrationCell` carries `rho_measurement_episode_count`, `rho_background_cohort_count`, `probe_successful_episode_count`, `probe_generation_attempt_count`, `probe_skipped_episode_count`; production validation adds cohort floor, successful == sample_count, and the attempts identity; serialization and provenance updated (`episode_horizon: recovered_start`).
- `tests/test_calibration.py`: new TimelineAlignmentTests, HorizonGenerationTests, ProbeQuartileTests, BisectionEndpointTests; updated helpers and real-cell tests to the real timeline and new audit fields.
- Version 0.6.0 -> 0.7.0: the calibration schema/provenance and default timeline changed; no released experiment artifacts depend on the old schema.
- Ticket 02: Comments pointer appended; history untouched.

#### Verification

- Red (before the fix): `Ran 223 tests ... FAILED (failures=3, errors=21)` — timeline mismatch `(10.0, 20.0, 22.0) != (100.0, 200.0, 220.0)`; quartile test found only 9 usable probes on the real timeline (midpoint rule + fixed-count starvation); endpoint-acceptance test showed bounds never win.
- Green: `.\.venv\Scripts\python.exe -m unittest discover -s .\tests -v` — `Ran 223 tests ... OK` (~18 s).
- `.\.venv\Scripts\python.exe -m mfg_hedge check --config .\configs\v1_minimal.json` — exit 0.
- Probe quartiles over 400 samples per state (real timeline, rate 1.0): H [113, 101, 100, 86] mean 0.468; D [113, 96, 105, 86] mean 0.476; F [113, 96, 90, 101] mean 0.484 — every quartile >= 86 (>> 10%), no midpoint clustering.
- Real production table (twice): 70 cells; 119.3 s and 119.5 s; byte-identical; max rho_error 0.016535 (<= 0.02); min sample_count 400; min rho cohort 400; probe_successful == sample_count everywhere; attempts identity holds; total probe skips 0; D/0.7/R contrast: Normal q=0.0850 / replay_work=0.0809, Delayed 0.0175 / 0.0134, Immediate 0.0 / 0.0.
- No artifacts written; no solver/price/quota/CLI/config/experiment work.

#### Artifacts

None generated. Source: `src/mfg_hedge/calibration.py`, `src/mfg_hedge/__init__.py`, `pyproject.toml`. Tests: `tests/test_calibration.py`.

#### Decisions and risks

- Calibration now shares the evaluation's physical timeline; any future scaled-down calibration timeline must scale all times and rates together and prove statistical equivalence first (not done here).
- Probe selection uses a SHA-256-seeded `randrange`, not modulo arithmetic on the digest.
- Full-table build is ~2 minutes; remains a verification step, not a unit test.
- Version 0.7.0 rationale: public calibration schema/provenance and the default timeline changed; Healthy/Common-State/Hedge behavior unchanged.

#### Next

Wait for the user's ticket 03 instruction (MFG solver and quota). Tickets 03/04 remain open.

## Answer

All acceptance criteria pass: real 100/200/220 timeline everywhere; time-horizon background generation; uniform cohort-representative probes with passing quartiles; rho contract intact (0.02, [0.05, 3.0], 40 iterations, endpoints audited); 70-cell table built twice byte-identically in ~119 s each; D contrast present; 223/223 tests green; `check` exit 0; ticket 02 carries the pointer; tickets 03/04 open; no forbidden scope.

## Comments

