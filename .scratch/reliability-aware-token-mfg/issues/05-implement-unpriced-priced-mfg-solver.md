# Implement unpriced and priced Token-MFG response solver

Type: implementation
Status: resolved
Blocked by: none

## Goal

Implement the diagnostic forward/response loop for the two separately named
models `unpriced_mfg` and `priced_mfg` once finite physical, population, and
tagged-deviation contracts are available.

## Scope

- Freeze response type (pure or logit), temperature, damping, convergence
  tolerances, supported-state rule, sample libraries, namespaces, seeds,
  fingerprints, and call budgets before solving.
- Use causal `m` and `x`; preserve simultaneous action semantics.
- Keep the congestion charge observable, pre-registered, and separate from
  social transfer accounting.
- Detect fixed points, cycles, nonconvergence, physical failure, and
  statistics insufficiency without relabeling them.
- Do not add Hedge, placement control, HJB/FPK, or a market-clearing claim.

## Frozen bounded implementation-smoke protocol

This ticket freezes an isolated diagnostic response solver.  It is not a
formal qualification or experiment campaign.

### Models and private costs

The only model IDs are `unpriced_mfg` and `priced_mfg`.  Both use the same
pathwise No-Hedge physical components and separate model records.  Cost units
are normalized time/work units:

`C = latency + alpha_class * deadline_miss + gamma_class * replay_count`
`    + c_lost * lost_work + congestion_charge`.

The frozen coefficients are `alpha_regular=1.0`, `alpha_urgent=5.0`,
`gamma_regular=4.0`, `gamma_urgent=6.0`, and `c_lost=1.0`.  These are
implementation-protocol values for this solver and are not copied from a
holdout result or another cost model.  Social transfers are not included in
the private physical-cost components.

`unpriced_mfg` has zero charge.  `priced_mfg` uses the observable charge
`price_per_queue_unit * observable_queue_load_index`, with
`price_per_queue_unit=0.25` cost units per normalized queue-load index per
tagged Token.  The queue-load index is supplied by the causal observable
state/bucket and must be finite and non-negative.  No future service draw,
remaining work, future fault, or hidden queue state may enter the charge.

### Response/update rule

The frozen response is entropy-regularized Logit response, not a pure BR:
`probability(a|s) proportional to exp(-beta * Q(s,a))`, with `beta=2.0`
(temperature `tau=0.5`) and damping `eta=0.2`.  Updates are simultaneous.
The supported-state set is exactly the complete positive-occupancy state set
returned by the forward provider.  Missing or unsupported states are
fail-closed; there is no silent fallback or state filling.

The maximum is `10` iterations.  The convergence tolerance is `0.01` for
the weighted policy/share residual and the active population residual.  Each
iteration preserves complete forward data, fresh mean-field continuation
data, and fresh finite-K deviation checks.  Policy, population, share, and
finite-K diagnostics are separate fields.

### Samples, namespaces, and calls

The bounded smoke uses `16` forward episodes, `16` continuation episodes and
`16` finite-K deviation episodes, with a minimum complete panel size of `2`
for every supported state/action cell.  Forward data uses namespace
`reliability-aware-token-mfg:v1:mfg-forward-smoke` and macro seed `20260914`.
Continuation reuses the accepted ticket-04 namespace and seed
`reliability-aware-token-mfg:v1:mean-field-continuation` / `20260912`; finite
deviation reuses `reliability-aware-token-mfg:v1:finite-k-deviation` /
`20260913`.  Model ID and iteration are branch-key components.  Matched
unpriced/priced branches use the same exogenous CRN identity but never pool
observations or statistics.  The fixed call accounting is:

`forward=16`, `continuation=16*(1+8)=144`, `finite-K=16*(1+8)=144` per
model/iteration, hence at most `2*10*(16+144+144)=6080` scheduler calls.
The parent solver owns this budget; a failed provider call is counted, is not
retried, and makes the result fail-closed.

### Inputs, records, and terminal statuses

The solver accepts immutable forward snapshots and fresh provider outputs; it
never accepts a realized future action list.  Every iteration record stores
the full canonical policy rows, `mu/nu/eta/x` fingerprints, Q fingerprint,
call accounting, and stopping reason.  The output claim boundary is only
`bounded_mfg_implementation_smoke` and must assert no BR, regret, Nash, or
MFG qualification claim.

The only terminal statuses are `fixed_point`, `cycle`, `not_converged`,
`statistics_insufficient`, and `physical_failed`.  A repeated complete
composite state `(policy,population,share)` is a cycle.  Incomplete/underfloor
statistics and provider/invariant exceptions are never converted into a
successful status.

## Progress log

### Update: 2026-09-10 — bounded response solver implementation

Status: partial

#### Goal

Implement the isolated bounded response loop for `unpriced_mfg` and
`priced_mfg`, with frozen Logit response, causal population fingerprints,
finite-K diagnostics, and parent-owned call accounting.  No qualification or
holdout campaign is in scope.

#### Changed

- Added `src/mfg_hedge/token_mfg_response_solver.py` with immutable cost,
  forward snapshot, policy, Q-row, deviation-diagnostic, iteration, model
  result, and batch result contracts.
- Frozen private-cost coefficients and the observable priced queue-load
  charge declared above; priced and unpriced models remain separate.
- Added stable Logit response (`beta=2.0`, temperature `0.5`) and simultaneous
  damping (`eta=0.2`), plus policy/population/share residuals.
- Added composite-state cycle detection and the frozen terminal statuses.
- Added parent-side monotone call reservation/settlement with no retry.
- Added `tests/test_token_mfg_response_solver.py` with Red coverage for missing
  module, then Green coverage for costs, response updates, both models, cycle,
  statistics insufficiency, ordinary provider exceptions, and call budget.

#### Verification

- Real Red: `.venv\\Scripts\\python.exe -m unittest tests.test_token_mfg_response_solver -v`
  failed before implementation with `ModuleNotFoundError` for
  `mfg_hedge.token_mfg_response_solver`.
- Focused Green: the same command passed `9/9` tests.
- Affected suites passed `56/56`: population estimator, simultaneous routing,
  tagged response, and historical T4/T4b/T4c diagnostics.
- `git diff --check`: passed.
- `py_compile` for the new module and focused test: passed.

#### Artifacts

None.  No experiment, qualification, holdout, or campaign artifact was
generated.

#### Decisions and risks

The solver is an internal isolated API, so package version and historical
scientific behavior are unchanged.  Provider exceptions are captured only at
the ordinary `Exception` boundary; `KeyboardInterrupt` and `SystemExit` are
not intercepted.  The implementation smoke still does not establish BR,
regret, Nash, or MFG claims.

#### Next

Run the full test suite and the required configuration check, then resolve
this ticket only if both pass.

### Update: 2026-09-10 — verification and resolution

Status: completed

#### Goal

Complete the bounded implementation smoke for the frozen unpriced/priced
Token-MFG response layer without starting scale qualification or holdout.

#### Changed

- Kept the solver as an internal isolated module; no package export, version
  bump, historical engine change, configuration change, or artifact write.
- Ordinary provider/simulation-boundary exceptions now terminate the affected
  model as `physical_failed` with exception type, message, iteration, and
  parent-accounted calls; `KeyboardInterrupt` and `SystemExit` remain
  uncaught.
- Incomplete forward/Q/deviation panels terminate as
  `statistics_insufficient`; complete repeated composite states remain
  `cycle`; residual convergence is reported as `fixed_point`, otherwise the
  frozen iteration limit reports `not_converged`.
- The complete iteration rows retain policy rows, scored Q rows, separate
  `mu/nu/eta/x` fingerprints, environment and Q fingerprints, residuals,
  finite-K diagnostic fields, call accounting, and stop reason.

#### Verification

- Focused solver suite: `9/9` passed with the exact command
  `.venv\\Scripts\\python.exe -m unittest tests.test_token_mfg_response_solver -v`.
- Affected suites: `56/56` passed, covering the population estimator,
  simultaneous finite routing, tagged continuation/deviation layer, and
  historical T4/T4b/T4c diagnostics.
- Full suite: `771/771` passed in `352.456s` with
  `.venv\\Scripts\\python.exe -m unittest discover -s tests -v`.
- Config check passed with exit code `0` and JSON `status: ok` using
  `.venv\\Scripts\\python.exe -m mfg_hedge check --config configs/v1_minimal.json`.
- `git diff --check` and `py_compile` for the new module/test passed.
- No formal solver calls, K-scale qualification, holdout, or experiment
  campaign was run; the `6080` value remains only the frozen maximum budget.

#### Artifacts

None.  No MFG candidate, Nash result, qualification, holdout, or experiment
artifact was generated.

#### Decisions and risks

Ticket 05 delivers only `bounded_mfg_implementation_smoke`.  The Logit
response is a regularized diagnostic with `beta=2.0`, temperature `0.5`, and
`eta=0.2`; it is not a pure BR or an equilibrium certificate.  The priced
charge remains an observable queue-load term and is not a market-clearing
price.  Ticket 06 remains responsible for K-scale qualification and
independent holdout under a separately frozen protocol.

#### Next

Ticket 06 may be reviewed for its qualification protocol; do not start a
formal campaign from this implementation ticket.

## Answer

The frozen bounded unpriced/priced response solver and its diagnostics are
implemented and verified.  No formal MFG conclusion is available from this
ticket.

## Acceptance criteria

- Forward and response data are independent as declared.
- Every iteration preserves complete physical reruns and finite-deviation
  checks for supported cells.
- Pure BR, regularized response, finite-N regret, Nash, and MFG residual have
  separate fields and statuses.
- A successful run is labelled only within the accepted claim boundary.
- No holdout result changes model, price, grid, or stopping parameters.

## Next dependency

Ticket 06 may start only after both solver variants and their diagnostics are
resolved.
