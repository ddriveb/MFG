# Implement Token private payoff and finite pathwise deviations

Type: implementation
Status: resolved
Blocked by: none

## Scope

Implement only the accepted ADR-0021 T2 boundary in the restored Token-player
branch:

- `src/mfg_hedge/token_payoff.py`;
- `src/mfg_hedge/token_deviations.py`;
- strict validation and provenance for `token_initial_price_v0`,
  `token_runtime_adr0009_v1`, and `token_extended_reservation_v1`;
- complete loser settlement and pathwise cost/gain records;
- one requested-action intervention for one target Token, with all other
  Tokens run through their fixed policy functions;
- same-CRN complete physical reruns and the 18-class regression matrix.

The implementation must stop at finite pathwise cost/deviation evidence. It
must not implement or report best response, regret, Nash, MFG, continuation
prediction, price feedback, forward/HJB/FPK, scaling, or a formal campaign.

## Acceptance criteria

- The three cost models have independent parameter schemas, price bases/units,
  model IDs, and provenance fingerprints. The withdrawn
  `token_original_runtime_v1` ID is rejected.
- Cost scoring is strict, immutable, complete-only, and uses actual winner,
  Replay, Primary/Hedge/Replay executed work, loser waste, Reservation charge,
  timer/suppression, latency, and final settlement time.
- Baseline and each candidate call a fresh policy factory, validate the base
  `choose()` at the target once before overriding only the requested action,
  and rerun the complete episode with identical immutable CRN inputs.
- Other Tokens retain policy functions and fresh private state, not cached
  action trajectories or observations. Candidate order cannot leak state.
- Results contain signed pathwise cost/gain rows ordered N/D/I and explicit
  `claims_best_response=False`, `claims_regret=False`, `claims_nash=False`,
  and `claims_mfg=False`; no BR/regret/equilibrium fields are emitted.
- The 18 ADR-0021 regression scenarios pass, including loser/drain,
  Reservation denial, timer voiding, nonzero external quote, all three model
  identities, target no-op, state isolation, CRN, tie/order, incomplete-run,
  and strict bad-input cases.
- Red tests precede implementation; focused T2/T1/legacy tests, full suite,
  and config check pass. No formal artifact, worker pool, or campaign runs.

## Progress log

### Update: 2026-09-06 — Claim accepted T2 implementation boundary

Status: partial

#### Goal

Implement the accepted ADR-0021 finite Token private-cost and pathwise
requested-action deviation layer as a standalone T2 slice.

#### Changed

- Created and claimed this independent T2 implementation ticket after explicit
  user acceptance of ADR-0021.
- Reserved the two production modules, their strict three-model provenance,
  complete settlement, same-CRN reruns, and 18-class test matrix for this
  ticket only.
- No production implementation, formal campaign, or new experiment artifact
  has been started in this claim step.

#### Verification

- Confirmed ADR-0021 now has status `Accepted (confirmed on 2026-09-06)`.
- Confirmed tickets 01–04 are resolved and no other ticket is claimed.
- Confirmed the accepted boundary stops before predictor, price feedback,
  best response, regret, forward, scaling, Nash, MFG, and formal campaign.

#### Artifacts

None.

#### Decisions and risks

The implementation will preserve T1's causal online entry, Reservation,
event ordering, CRN streams, policy reset contract, and historical static
path. Pathwise gains are evidence for one realized Token intervention only;
they will not be aggregated into an expected-regret or equilibrium claim.

#### Next

Write the 18-class failing regression tests, then implement strict payoff
records and complete single-request deviation reruns.

### Update: 2026-09-07 — Complete T2 finite pathwise payoff and deviations

Status: completed

#### Goal

Implement the accepted ADR-0021 T2 slice: three independent Token private-cost
models, strict completed-run settlement, and one-target requested-action
counterfactual reruns on the identical CRN trace. Stop at finite pathwise
cost/gain evidence.

#### Changed

- Added `src/mfg_hedge/token_payoff.py` with immutable
  `token_initial_price_v0`, `token_runtime_adr0009_v1`, and
  `token_extended_reservation_v1` parameter types, `ExternalQuote`, strict
  model/basis validation, provenance fingerprints, and `TokenPayoff` records.
- Scoring uses actual latency, Replay count, Primary/Hedge/Replay executed
  work, losing work, Reservation charge, timer/executor suppression, winner,
  and complete loser settlement. Queued cancelled work remains zero; failed
  running work remains billable physical work.
- Added `src/mfg_hedge/token_deviations.py` with fresh policy-factory branches,
  quoted causal observations, one target requested-action override after the
  base `choose()` call, complete T1 reruns, CRN fingerprint checks, canonical
  N/D/I rows, no-retry failure accounting, and pathwise-only claim flags.
- Added `tests/test_token_payoff.py`, `tests/test_token_deviations.py`, and
  `tests/token_t2_support.py` covering the 18 ADR-0021 regression scenarios.
- Kept new modules internal; no package export or version bump. Historical T1,
  static Mapping, event-engine, Reservation, workload, solver, and
  Shared-Backup behavior were not changed.

#### Verification

- Real Red before implementation:
  `.venv\\Scripts\\python.exe -m unittest tests.test_token_payoff tests.test_token_deviations -v`
  failed at module loading with `ModuleNotFoundError` for both new production
  modules.
- T2 focused suite:
  `.venv\\Scripts\\python.exe -m unittest tests.test_token_payoff tests.test_token_deviations -v` —
  `Ran 18 tests`, `OK`.
- Affected regression suites:
  `.venv\\Scripts\\python.exe -m unittest tests.test_token_payoff tests.test_token_deviations tests.test_token_online tests.test_hedge_simulation tests.test_attribution_episode tests.test_transient_control tests.test_workload -v` —
  `Ran 126 tests`, `OK`.
- Full suite:
  `.venv\\Scripts\\python.exe -m unittest discover -s tests -v` —
  `Ran 535 tests in 153.126s`, `OK`.
- Configuration:
  `.venv\\Scripts\\python.exe -m mfg_hedge check --config
  configs\\v1_minimal.json` — `status: ok`; the pre-existing diagnostic
  `single-domain failure ... 1.400 > 0.900` remains unchanged.
- Verified same-action intervention reproduces the complete baseline result
  exactly; target policy is still called once before each requested-action
  override; other Tokens run their policy functions again; candidate order is
  canonical; attempted calls stop without retry on injected failure.
- Verified pathwise output label is exactly
  `finite_token_pathwise_deviation` and all BR/regret/Nash/MFG claim flags are
  false. No Python worker remains and no new formal artifact was generated.

#### Artifacts

Source and tests only:
`src/mfg_hedge/token_payoff.py`, `src/mfg_hedge/token_deviations.py`,
`tests/test_token_payoff.py`, `tests/test_token_deviations.py`, and
`tests/token_t2_support.py`. No experiment, predictor, validation, BR,
regret, Nash, MFG, or campaign artifact was created.

#### Decisions and risks

- The three cost models retain independent IDs, parameter schemas, price
  bases, and provenance even when numeric costs coincide. The withdrawn
  `token_original_runtime_v1` ID is rejected.
- External quotes are immutable and episode-constant; they are delivered to
  policy observations and scorer under the same model-specific basis. No
  endogenous price feedback was added.
- This implementation produces only signed pathwise cost/gain rows for one
  finite Token intervention. It does not calculate expected BR, regret, Nash,
  MFG, continuation prediction, or any equilibrium statistic.
- Package version remains 0.24.0 because the new modules are not exported as
  public package API.

#### Next

等待用户审查 T2；下一切片才可另行设计 continuation predictor 或其他后续层，不能由本 ticket 自动进入。

## Answer

ADR-0021 T2 is implemented and verified at the finite pathwise deviation
boundary. Ticket 05 is resolved; no predictor, expectation-based BR, regret,
price feedback, Nash, MFG, or formal experiment was started.
