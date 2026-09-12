# Run predeclared load 0.6/0.7 diagnostics

Type: task
Status: resolved
Blocked by: none

## Goal

Implement and execute the bounded diagnostics in `../spec.md`.

## Progress log

### Update: 2026-09-05 — Predeclared 0.6/0.7 stress diagnostics

Status: completed

#### Goal

Determine whether the frozen solver admits official paired policies at loads
0.6 and 0.7 without relaxing ADR-0008 or pooling stress results with load 0.5.

#### Changed

- Added `src/mfg_hedge/stress.py` and `diagnose-mfg-hedge-loads`.
- Added explicit official-gate diagnostics and tests for converged,
  infeasible, grid-clamped, and exact F all-Normal requirements.
- Version 0.12.0 -> 0.13.0 and documented all paired commands in `README.md`.

#### Verification

- Red: `ModuleNotFoundError: mfg_hedge.stress`.
- Green: three focused tests passed; final `295 tests ... OK`; environment
  `check` exited 0.
- Real load 0.6: H converged, D `infeasible` with
  `boundary_minimum_exceeds_capacity`, F forced-Normal; official gate failed.
- Real load 0.7: H converged, D/F `nonconverged` with `grid_clamped`; official
  gate failed. This ordering matches ADR-0008's mandatory price-zero/grid gate.

#### Artifacts

- `artifacts/single_expert_mechanism_check-load-stress-06-07-20260905-night/`
  (`manifest.json`, `load_0p6.json`, `load_0p7.json`).

#### Decisions and risks

- No paired simulation was run at either stress load; doing so after the
  official gate failed would create scientifically invalid A/B results.
- These diagnostics do not alter or qualify the load-0.5 campaign artifact.

#### Next

Write a baseline-ablation spec for disabling persistent Healthy-policy Hedge
after recovery and/or comparing against a conventional threshold Hedge policy.

## Answer

Both predeclared stress loads correctly fail closed. Load 0.6 has a certified
Degraded-state capacity infeasibility; load 0.7 exits at the mandatory
calibration-grid boundary. No misleading paired result was produced.
