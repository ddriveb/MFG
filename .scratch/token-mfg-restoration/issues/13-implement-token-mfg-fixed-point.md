Type: implementation and experiment

Status: resolved

Blocked by: none

## Goal

Implement and run the first damped finite-population Token response iteration:

```text
pi(k) -> Q(. | b, pi(k)) -> BR(k) -> pi(k+1)
```

The result is a population-policy fixed-point diagnostic only.  It must not be
called Nash, MFG, or a verified best response.

## Frozen T4-v1 contract

- Environment remains `theta_token_load0p7_v1`: the T3A timeline, service,
  fault, slowdown `2.0`, hedge delay `2.0`, Reservation, dispatcher, and full
  drain are unchanged.
- Initial population policy is exactly online NIIN.  Subsequent policies use
  the exact function mixture
  `pi(k+1) = (1-0.2) pi(k) + 0.2 BR(k)`, retaining NIIN as the base function
  and retaining every previous BR component.  The current policy is called
  online for every Token; no action trajectory is replayed.
- `bin_schema_v1`, its nine T3A-retained bins, S_v2 anchor assignment, runtime
  ADR-0009 scorer, N/D/I action set, and 32 complete-panel floor are fixed.
- Each iteration uses a fresh namespace
  `token-mfg-restoration:t4:fixed-point:v1:iteration:<k>`, macro seed
  `20260913`, and at most `2048` episode slots.  Each slot consumes one
  target-selection run and, when a target exists, four complete baseline/N/D/I
  physical runs.  The per-iteration ceiling is `10,240`; ten rounds have a
  declared ceiling of `102,400` calls.  No retry, supplement, bin merge, or
  result-driven sample change is allowed.
- Q rows are pooled by `(iteration, bin_id, requested_action)` and report
  mean, SE, effective n, and `completed`/`insufficient`.  A bin with any
  action below 32 panels makes that iteration `statistics_insufficient` and
  prevents a fixed-point claim.
- Tie breaking is `N < D < I`.  A completed Q table produces a deterministic
  one-hot BR table for all nine bins.  Unknown bins remain on NIIN and are not
  silently reclassified.
- Residuals are recorded as: maximum policy L1 change over observed target
  observations, maximum current-policy mass not assigned to the current BR,
  and L1 distance between successive selected-target-bin occupancy vectors
  (including `missing_target` and `other_bin` categories).  Tolerances are
  `0.01` for each residual and require two comparable iterations.
- A repeated non-consecutive BR table is `cycle` with period; no update is
  hidden.  If no fixed point/cycle/insufficient result occurs by ten rounds,
  the status is `not_converged_10`.
- Fresh artifact run ID:
  `token-t4-fixed-point-20260908-r1/`.  Required files are manifest, metric
  contract, iteration rows, Q rows, and summary.  Writes are transactional and
  never overwrite prior artifacts.
- All claim flags for best response, regret, Nash, and MFG are false.

## Acceptance criteria

1. Red tests first cover exact function damping, online fresh policy state,
   deterministic BR/ties, Q floor and fail-closed status, same-trace CRN
   branches, occupancy/residual accounting, cycle/nonconvergence states,
   exact call budget, complete iteration rows, and artifact no-overwrite.
2. The real run uses the frozen ten-round ceiling and records every completed
   iteration until a permitted stop reason occurs; no formal campaign follows
   automatically.
3. The artifact records source/environment/protocol fingerprints, complete
   policy and BR rows, Q rows, occupancy/residual fields, stop reason, and
   actual calls.
4. Focused tests, one full suite, and one config check pass.

## Progress log

### Update: 2026-09-08 — Claimed T4 fixed-point diagnostic

Status: partial

#### Goal

Start the first damped population-response iteration from NIIN without
mistaking the single-Token T3B response for a population equilibrium.

#### Changed

- Created and claimed this single T4 implementation/experiment ticket.
- Frozen `eta=0.2`, ten-round maximum, 2048 episode slots per round, 32-panel
  floor, 102,400-call declared ceiling, residual definitions, and fail-closed
  statuses above.
- No production code or T4 artifact has been changed yet.

#### Verification

- Read repository rules, T3B/T3C tickets and artifacts, online engine,
  continuation/bin schema, deviation evaluator, and attribution metrics.
- Red: `.venv\\Scripts\\python.exe -m unittest tests.test_token_mfg_fixed_point -v`
  ran 6 tests and all 6 failed with `ModuleNotFoundError` because the T4
  module had not yet been created.

#### Artifacts

- None.  T3B and T3C artifacts remain unchanged.

#### Decisions and risks

- This is an empirical finite-population damped response diagnostic, not a
  closed conditional-law MFG forward model.  The claim boundary remains
  explicit.
- Damping is implemented as a mixture of policy functions, not as a frozen
  action list and not as a manual Hedge adjustment.

#### Next

Write the failing T4 tests, implement the smallest response iterator, then run
the frozen diagnostic.

### Update: 2026-09-08 — Core T4 slice and engine smoke

Status: partial

#### Goal

Turn the T4 contract into an online, auditable implementation and verify its
first complete physical panel before the frozen run.

#### Changed

- Added `src/mfg_hedge/token_mfg_fixed_point.py` with immutable NIIN-plus-BR
  function mixtures, causal target capture, deterministic N/D/I BR ties,
  Q-panel floor handling, residual/cycle helpers, exact call ledger, and
  transactional artifact output.
- Added `tests/test_token_mfg_fixed_point.py` covering the first T4 contract
  slice.
- Corrected the unknown-bin fallback so historical BR components preserve the
  observation's NIIN base action instead of silently forcing N.
- Started the frozen `token-t4-fixed-point-20260908-r1` run; it has not yet
  committed an artifact directory.

#### Verification

- Real Red: focused suite initially ran 6 tests and all 6 failed with
  `ModuleNotFoundError` before the new module existed.
- Focused Green: `.venv\\Scripts\\python.exe -m unittest
  tests.test_token_mfg_fixed_point -v` ran 6/6.
- Real-engine smoke: 16 episode slots produced 13 complete panels and
  `68` attempted/reserved calls (`16 + 13*4`); missing and other-bin slots
  were retained and no retry occurred.
- The full frozen run is currently executing with the declared 2048-slot
  protocol; final suite and config checks remain pending.

#### Artifacts

- No T4 artifact committed yet; the transactional writer has not published a
  partial directory.

#### Decisions and risks

- Unknown retained-support bins keep the exact NIIN base function, including
  NIIN hedges for observations outside the retained T3A support.
- The run remains a diagnostic and cannot produce a Nash/MFG claim.

#### Next

Complete the frozen run, inspect its committed rows and call ledger, then run
the required full suite and config check.

### Update: 2026-09-08 — First frozen-run failure and correction

Status: partial

#### Goal

Run the full frozen T4 protocol and verify that completed Q rows enter the
response state machine.

#### Changed

- Corrected the main loop to branch on `_run_iteration`'s explicit `q_status`
  instead of an absent `status` field when the Q table is complete.
- No artifact was published by the failed attempt because the writer is
  transactional; the requested run directory remains available for the
  corrected rerun.

#### Verification

- The first frozen invocation ran until the Q iteration completed, then
  failed closed with `KeyError: 'status'` at the state-machine handoff.
- No candidate, fixed-point, or partial artifact was emitted.
- Focused suite after the correction: 6/6 passed; module compilation passed.

#### Artifacts

- None from the failed invocation.

#### Decisions and risks

- This was an execution-layer state-field defect, not a change to the frozen
  policy, physical engine, CRN, panel floor, or stopping semantics.

#### Next

Rerun the same frozen artifact ID after the correction, then inspect its
complete rows before final verification.

### Update: 2026-09-08 — Artifact boundary correction

Status: partial

#### Goal

Commit the completed frozen iteration without exposing in-memory observation
objects to the JSON artifact writer.

#### Changed

- Added a canonical JSON conversion at the transactional artifact boundary;
  internal `TokenObservation` values remain available for exact residual
  calculation but are serialized as data fields.

#### Verification

- The second frozen invocation completed its physical iteration but failed
  closed during artifact serialization with `TypeError: Object of type
  TokenObservation is not JSON serializable`.
- No T4 artifact directory was published.
- Focused suite after the correction: 6/6 passed; module compilation passed.

#### Artifacts

- None; both failed invocations left no committed run directory.

#### Decisions and risks

- This is a durable-output boundary fix only; the physical traces, policy
  calls, CRN branches, Q rows, and stopping rules are unchanged.

#### Next

Rerun the same frozen ID once more, then inspect the committed artifact before
running the full verification suite.

### Update: 2026-09-08 — T4 diagnostic completed

Status: completed

#### Goal

Execute the frozen damped finite-population response diagnostic from NIIN and
retain every policy/Q/occupancy row through its permitted stop reason.

#### Changed

- Completed `token_mfg_fixed_point.py` with the online NIIN-plus-historical-BR
  function mixture, exact `eta=0.2` update, causal target selection, complete
  baseline/N/D/I CRN reruns, per-bin Q summaries, residuals, cycle detection,
  fail-closed Q floor, and transactional output.
- Added focused tests for function damping, fresh episode state, N/D/I tie
  order, Q-floor failure, cycle classification, budget settlement, and
  non-overwriting artifacts.
- Kept the module internal; package version remains `0.24.0` and no public API
  or historical engine/config behavior changed.

#### Verification

- Real Red: the initial focused run was `6/6` failures with
  `ModuleNotFoundError` before the module existed.
- Two subsequent frozen-run attempts failed closed after substantial physical
  work: first at the Q-to-state handoff (`KeyError: 'status'`), then at JSON
  serialization of in-memory `TokenObservation`. Neither published an
  artifact; both defects were corrected without changing the scientific
  contract.
- Final focused suite: `.venv\\Scripts\\python.exe -m unittest
  tests.test_token_mfg_fixed_point -v` — `6/6` passed.
- Final full suite: `.venv\\Scripts\\python.exe -m unittest discover -s tests
  -v` — `578/578` passed in 164.849 seconds.
- Config check: `.venv\\Scripts\\python.exe -m mfg_hedge check --config
  configs\\v1_minimal.json` — `status: ok`; the existing 1.4 transient
  single-domain load warning remains diagnostic.
- Artifact checks: `policy_iterations.json` has 3 complete iteration rows;
  `q_rows.json` has `81 = 3 x 9 x 3` rows; all 27 rows per iteration are
  `completed`, with minimum action-panel n of 60, 45, and 32.
- Actual scheduler accounting is `27,560` attempted and `27,560` reserved,
  below the declared `102,400` ten-round ceiling.
- BR fingerprints are A, B, A, so the permitted result is `cycle` with period
  2. Residuals remain above the `0.01` fixed-point tolerances; no fixed-point
  claim was made.
- All artifact claim flags (`best_response`, `regret`, `nash`, `mfg`) are
  `false`.

#### Artifacts

- [T4 diagnostic artifact](D:/project/mfg_hedge_v1/artifacts/token-t4-fixed-point-20260908-r1/)
  with `manifest.json`, `metric_contract.json`, `policy_iterations.json`,
  `q_rows.json`, and `summary.json`.
- No formal Fit/Validation/candidate/Nash/MFG artifact was generated.

#### Decisions and risks

- The result is a finite-population damped response cycle under the fixed
  `theta_token_load0p7_v1` environment and T4 policy protocol, not a Nash or
  MFG result.
- The run demonstrates that re-estimating Q under a changing population is
  necessary; it does not establish convergence, regret bounds, or a closed
  conditional-law mean field.
- No price update, predictor, Soft BR, or population forward model was added.
- The two failed execution attempts remain process-history evidence only and
  were not silently merged into the successful artifact or call count.

#### Next

Design the next bounded T4 convergence/diagnostic slice from the retained
period-2 artifact; do not treat this cycle as an equilibrium and do not start
price feedback or MFG certification automatically.

## Answer

Status: resolved

T4-v1 was implemented, executed, verified, and stopped at the frozen
population-response diagnostic boundary.  The result is a period-2 cycle, not
a fixed point or equilibrium claim.
