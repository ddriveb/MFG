# Price persistent incremental and wasted work

Type: task
Status: resolved
Blocked by: none

## Goal

Implement the user-directed persistent runtime-cost correction in `../spec.md`
without rewriting the completed zero-cost experiment history.

## Scope

- Record and implement ADR-0009.
- Extend ActionStats calibration with expected wasted work.
- Add schema-4 paired configuration with explicit work/waste coefficients while
  retaining schema-3 zero-cost compatibility.
- Add both costs to the finite-action objective and all solver provenance.
- Run the full verification suite, real 70-cell gate, and one deterministic
  1000-Token same-trace paired comparison in a fresh artifact directory.

## Acceptance criteria

- All six spec acceptance items pass under red-green-refactor.
- No existing artifact is overwritten or modified.
- The update reports policy, Hedge activity, latency, Replay, and executed-work
  deltas rather than claiming success solely from fewer Hedges.

## Progress log

- Read the completed paired-comparison design, ADRs, solver, calibration,
  configuration, CLI, metrics, campaign, and stress paths before changing the
  objective.
- Added the feature spec and accepted ADR-0009 before implementation.  The
  schema-3 zero-cost experiment remains reproducible; schema 4 opts into the
  corrected objective with explicit positive coefficients.
- Red: the focused contract suite ran 23 tests and failed with 1 failure and
  14 errors.  The failures were the missing wasted-work statistic, missing
  persistent-cost solver parameters, and missing schema-4 configuration path.
- Green/refactor: implemented the calibration, objective, configuration,
  provenance, CLI, paired/stress wiring, documentation, and version changes;
  corrected one test fixture that was outside the calibration grid and added
  the missing solver-parameter provenance exposed by an end-to-end test.
- Final verification: `.venv\Scripts\python.exe -m unittest discover -s tests -v`
  ran 303 tests successfully; `.venv\Scripts\python.exe -m mfg_hedge check
  --config configs\v1_minimal.json` exited 0 with `status: ok`.
- Built the real 70-cell calibration table under the 100/200/220 timeline.
  It serialized as calibration schema 2, passed the official H/D/F solver
  gate, and produced byte-identical solver output on repeat evaluation.
- Ran a deterministic 1,000-token same-trace paired check and an unmocked
  100,000-token seed-20260901 paired CLI run.  Each run committed exactly the
  four required files into a fresh directory; no prior artifact was changed.

## Answer

### Update: 2026-09-05 — Price persistent incremental and wasted work

Status: completed

#### Goal

Correct the finite-action MFG objective so incremental execution is never free
when the capacity shadow price is zero, and so executed non-winning attempts
receive a separate waste penalty.  Preserve all completed schema-3 history and
run a fresh same-trace comparison under the corrected schema-4 model.

#### Changed

- `src/mfg_hedge/domain.py` and `src/mfg_hedge/calibration.py` now carry,
  validate, aggregate, interpolate, and serialize `expected_wasted_work`.
  Wasted work is the executed work of terminal attempts whose attempt id is not
  the token's winner; cancelled/invalidated queued attempts contribute zero.
- `src/mfg_hedge/mfg_solver.py` now evaluates
  `latency + replay_penalty * replay_probability +
  (price + incremental_work_cost) * incremental_work +
  wasted_work_cost * expected_wasted_work`.  Both coefficients and the final
  action statistics are serialized in solver provenance.
- `src/mfg_hedge/config.py` accepts legacy schema 3 only with the historical
  implicit zero costs, while schema 4 requires finite positive
  `incremental_work_cost` and `wasted_work_cost` values.
- Added `configs/v2_paired_runtime_cost.json` with the predeclared mechanism
  defaults `incremental_work_cost=1.0` and `wasted_work_cost=1.0`; these values
  were fixed before observing the new experiment results.
- Wired the parameters through paired runs, stress diagnostics, the CLI
  comparison payload, and the manifest.  Calibration serialization moved to
  schema 2 and package version moved from 0.13.0 to 0.14.0.
- Added the feature spec, ADR-0009, tests, and README/CONTEXT updates.  Existing
  schema-3 configurations and artifacts remain unchanged.

#### Verification

- Red: focused suite `Ran 23 tests ... FAILED (failures=1, errors=14)` before
  implementation.
- Final suite: `Ran 303 tests ... OK`; configuration check exited 0 with
  `status: ok`.
- Real 70-cell calibration and solver gate:
  - H: `converged/price_zero_slack`, price 0, rho 0.513936, predicted Hedge work
    rate 0.027873.
  - D: `converged/boundary_root`, price 34.559503, rho 0.9, capacity gap about
    -1.1e-8.
  - F: forced Normal with the expected structural overload diagnosis.
- 1,000-token same-trace A/B: mean latency -40.34%, P50 +0.05%, P95 -65.10%,
  P99 -43.29%, Replay rate -68.75%, and execution amplification +2.16%; 86
  Hedges were launched.  This short controlled trace is a mechanism check, not
  a statistical conclusion.
- 100,000-token seed-20260901 same-trace A/B: mean +0.27%, P50 +1.56%, P95
  +0.67%, P99 -1.60%, Replay rate -68.75%, and execution amplification +2.83%.
  D/F/R P99 changed by -55.15%/-36.08%/-0.91%; 5,304 Hedges were launched.
  All paired invariants passed.
- Against the old zero-cost MFG arm on the identical long trace, launches fell
  from 90,930 to 5,304, amplification from 1.6635 to 1.0284, mean latency from
  2.6902 to 1.2940, and P99 from 10.1232 to 3.9967.
- Repeated real solves were byte-identical.  The two fresh artifact directories
  each contain exactly `manifest.json`, `comparison.json`,
  `no_hedge_summary.json`, and `mfg_hedge_summary.json`.

#### Artifacts

- `artifacts/single_expert_mechanism_check-runtime-cost-cw1-cww1-t1000-20260905/`
- `artifacts/single_expert_mechanism_check-runtime-cost-cw1-cww1-t100000-seed20260901-20260905/`

#### Decisions and risks

- `incremental_work_cost` and `wasted_work_cost` deliberately overlap: the
  first prices all added resource use; the second adds a penalty when executed
  work does not win.  This is recorded in ADR-0009.
- The correction removes pathological, effectively free hedging, but the
  predeclared 1.0/1.0 coefficients are not claimed optimal.  On the long trace,
  tail latency improves while mean/P50/P95 remain slightly worse than
  No-Hedge; this tradeoff is reported rather than tuned away after inspection.
- The single long seed establishes mechanism behavior, not uncertainty bounds.
  Formal claims require a preregistered coefficient sensitivity and multi-seed
  campaign.

#### Next

Design and run a predeclared sensitivity grid for the two runtime-cost
coefficients across multiple paired seeds.  Select/report the Pareto tradeoff
between tail-risk reduction and added work without tuning on the final test
seeds.

## Comments
