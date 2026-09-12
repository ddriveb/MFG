# Implement simultaneous finite Token routing engine

Type: implementation
Status: resolved
Blocked by: none

## Goal

Implement the first finite physical slice in which all Tokens in a micro-batch
decide from one immutable pre-batch public state and aggregate trend, then
commit assignments atomically to the eight-Replica queues.

## Scope

- Freeze exact physical parameters, estimator values, bucket schema, CRN keys,
  namespace, sample identities, and call budget before trace generation.
- Add a new isolated engine or adapters; do not alter historical engines or
  ADR-0035 outputs.
- Preserve corrected failure-first, Replay, FCFS, work-ledger, complete-drain,
  and causal observation semantics.
- Represent `mu_t` (active Tokens), `nu_t` (Replica states), `eta_t`
  (new/Replay decision cohort), and `x_t` (cohort action shares) separately;
  queued/running Tokens must not be counted again in `eta_t`.
- Use the frozen scaling notation `N_K` for finite batch Token count, keep the
  batch release rate fixed, and require `K` to be a multiple of four.
- Add deterministic hand fixtures and information-structure tests.
- Keep the engine No-Hedge and single-copy.

## Acceptance criteria

- Same-batch observation and pre-batch state fingerprints are identical.
- `F_t^0` contains only exogenous common fault history, while public queue,
  placement, batch, and published-trend facts are tracked under `G_t`.
- No Token can read a same-batch realized action or a future event.
- Queue mutation occurs only after the full action batch commits.
- Only the `eta_t` cohort contributes to predicted routing shares.
- CRN traces and all physical invariants match the declared protocol.
- `K=8` has an explicit field-level regression against the accepted finite
  routing engine wherever semantics are intended to match.
- Focused and full tests plus config check pass.

## Next dependency

Ticket 03 may start only after the simultaneous physical contract and its
tests are resolved.

## Progress log

### Update: 2026-09-10 — simultaneous finite routing implementation

Status: completed

#### Goal

Implement and verify the isolated first physical slice of ADR-0037: each
same-time micro-batch observes one immutable pre-batch public state, makes all
single-Replica decisions online, commits them atomically, and then runs the
complete finite single-copy routing physics. Population estimation, best
response, prices, and MFG solving were not run.

#### Changed

- Added `src/mfg_hedge/simultaneous_token_routing.py` with frozen public
  observations, population decision context, online policy protocol, batch
  audit, strict input/action validation, and complete single-copy failure,
  Replay, queue, dispatch, and drain handling.
- Kept `src/mfg_hedge/reliability_aware_routing.py` and its historical
  `simulate_routing` path unchanged; the new engine is isolated and not
  exported from `mfg_hedge.__init__`.
- Added deterministic tests covering same-batch state sharing, hidden
  same-batch actions, atomic commit, Replay-plus-arrival cohorts, K=8
  field-level regression, determinism, and invalid policy actions.
- Corrected batch audit identity to preserve decision-cohort order while
  requiring unique Token IDs; this keeps assignment rows aligned with the
  online decision order.

#### Verification

- Real Red was recorded before implementation: focused collection failed with
  `ModuleNotFoundError: No module named 'mfg_hedge.simultaneous_token_routing'`.
  The first implementation pass also exposed a syntax error, followed by two
  test assertion defects (a survivor starts later, and t=0 calls were mixed
  with t=1 calls); these were corrected without changing historical behavior.
- Focused suite: `tests.test_simultaneous_token_routing`, **7/7 passed**.
- Affected suites: reliability-aware routing **12/12**, Lazarus fairness
  **3/3**, routing experiment **4/4**, baseline adapters **15/15**; total
  **41/41 passed**.
- Full suite: `unittest discover -s tests -v`, **739/739 passed**.
- Config check: `python -m mfg_hedge check --config configs/v1_minimal.json`,
  exit 0 with `status: ok`.
- The K=8 single-copy regression matched the historical result field by
  field for trace/token fingerprints, Token results, starts, metrics, and
  invariant counters. Same-batch calls shared one public-state fingerprint;
  queue mutation occurred only after all decisions; Replay and same-time
  arrival formed one cohort.
- No formal experiment, population estimator, MFG solver, or scheduler call
  campaign was run. No experiment artifact was generated and package version
  remains `0.28.0` because the new module is not a public package export.

#### Artifacts

- New source: `src/mfg_hedge/simultaneous_token_routing.py`.
- New focused tests: `tests/test_simultaneous_token_routing.py`.
- No Fit, Validation, campaign, population, MFG, or experiment artifact was
  created or overwritten.

#### Decisions and risks

- ADR-0037 is now marked accepted after the user's explicit confirmation.
- The implementation exposes only finite simultaneous physical routing. It
  does not claim a population law, equilibrium, best response, price, or MFG
  result.
- `published_realized_share` is retained as the prior published trend in the
  causal context; it is not an estimator or a future action list.
- Historical scientific behavior remains unchanged; future population work
  must proceed through ticket 03 and separately implement its estimator and
  consistency contract.

#### Next

Ticket 03 may be unblocked and claimed only when work on the population
estimator is explicitly authorized. This ticket does not authorize a formal
MFG experiment.
