# Implement online Token entry and Reservation

Type: implementation
Status: resolved
Blocked by: 01

## Scope

Implement only T1 of the accepted ADR-0020 boundary: causal
`TokenObservation`, one-shot online N/D/I decisions, the non-refundable
Decimal-audited Reservation ledger, and a compatible online entry on the
original one-Expert/two-Replica A/B engine. Preserve the legacy static action
mapping entry point and all historical modules and artifacts.

This ticket does not implement Token payoff, deviations, continuation
prediction, posted prices, best response, forward/HJB/FPK, K-executor scaling,
Nash/MFG claims, Fit/Validation, Stage 4b, or any formal experiment.

## Acceptance criteria

- ADR-0020 is Accepted for the Token identity and T1 physical contract only;
  deferred payoff/price/solver/forward/scaling proposals remain unaccepted.
- The online observation is immutable and causal, with no trace, future,
  service draw, true remaining work, or future outcome exposure.
- Each Token requests exactly one N/D/I action at arrival. Delayed timers and
  Replay never call the policy a second time. Stateful sources reset per
  Expert/episode/counterfactual run; legacy stateless/static sources remain
  compatible and private policy randomness cannot perturb CRN streams.
- Reservation uses the existing transient Decimal/window/cap semantics:
  immediate charge, no refund, requested/applied/suppression fields separated,
  denied D/I applied as N, and normal Replay remains possible.
- The online engine observes before dispatch and preserves the existing
  fault/completion/timer/arrival ordering, full drain, loser lifecycle and
  static mapping results exactly.
- NIIN/static-plan old and online-adapter paths are compared per Token and
  per attempt, including reservation and physical audit fields.
- Red tests precede implementation; focused affected suites, full suite, and
  config check pass. No formal experiment artifact or worker remains.

## Progress log

### Update: 2026-09-06 — Claim T1 online Token entry

Status: partial

#### Goal

Restore the original Token-player physical entry and reservation boundary as
the sole claimed Token ticket, while leaving the independent Expert branch
untouched.

#### Changed

- Returned Shared-Backup ticket 15 to `open` and appended its deferral record;
  preserved `Blocked by: 14` and all prior history.
- Tightened ADR-0020 to `Status: Accepted (Token-player restoration boundary
  and T1 only; confirmed on 2026-09-06)`.
- Created and claimed this ticket with `Blocked by: 01`.
- No production implementation, formal experiment, or new artifact has been
  started in this claim step.

#### Verification

- Read the governance documents, Token spec/audit/ticket, ADR-0020, relevant
  ADRs, Shared-Backup ticket 15, and the actual historical engine/planner
  entry points before code changes.
- Confirmed ticket 02 did not already exist.
- Confirmed no other top-level ticket is `claimed` after the handoff.

#### Artifacts

None.

#### Decisions and risks

ADR-0020 accepts only Token identity, causal one-shot arrival decisions,
two-Replica physical reuse, and Reservation/T1 boundaries. Payoff, price,
deviation, solver, forward, scaling, and all equilibrium labels remain
deferred. The existing Shared-Backup source and artifacts are outside this
ticket and remain unchanged.

#### Next

Add the required failing online-entry tests, then implement the smallest
compatible adapter and Reservation slice without changing historical results.

### Update: 2026-09-06 — Complete T1 online Token entry and Reservation

Status: completed

#### Goal

Implement only the accepted ADR-0020 T1 boundary: causal online Token
decisions, one-Expert/two-Replica physical entry, and non-refundable Decimal
Reservation accounting. No payoff, deviation, solver, forward model, MFG
campaign, or formal experiment was authorized or run.

#### Changed

- Added the internal `src/mfg_hedge/token_online.py` module with immutable
  `TokenObservation`, strict N/D/I validation, online `TokenActionSource`,
  static compatibility adapters, stable private policy randomness, immutable
  admission records, and the per-episode Decimal `ReservationLedger`.
- Added the online entry point to the historical Token engine in
  `src/mfg_hedge/hedge_simulation.py`. Arrival observation is created before
  enqueue/dispatch; `choose()` is called once per Token; Delayed timers and
  Replay never call the policy again; all existing event ordering, loser
  lifecycle, Replay, drain, and static Mapping behavior remain intact.
- Kept requested action, applied action, reservation suppression, executor
  suppression, timer state, charge, and no-refund behavior separately
  auditable. Stateful sources receive a fresh episode state, while stateless
  and legacy Mapping sources remain compatible.
- Added `tests/test_token_online.py` with 21 causal-observation, action,
  reservation, lifecycle, CRN, reset, and legacy-equivalence tests.
- ADR-0020 is accepted only for the Token identity/T1 physical boundary;
  Shared-Backup ticket 15 remains `open` and blocked by 14. No public package
  API was exported, so the version remains 0.24.0.

#### Verification

- Real Red before implementation:
  `.venv\\Scripts\\python.exe -m unittest tests.test_token_online -v`
  failed with `ImportError` / `ModuleNotFoundError: No module named
  'mfg_hedge.token_online'`.
- Focused suite:
  `.venv\\Scripts\\python.exe -m unittest tests.test_token_online -v` —
  `Ran 21 tests`, `OK`.
- Affected suites — all passed: `test_hedge_simulation` 33,
  `test_common_state_simulation` 26, `test_transient_control` 19,
  `test_quota` 17, `test_workload` 20, `test_game_workload` 15, and
  `test_shared_backup` 14; total 144/144.
- Full suite:
  `.venv\\Scripts\\python.exe -m unittest discover -s tests -v` —
  `Ran 516 tests in 156.284s`, `OK`.
- Configuration:
  `.venv\\Scripts\\python.exe -m mfg_hedge check --config
  configs\\v1_minimal.json` — `status: ok`; the existing diagnostic
  `single-domain failure ... 1.400 > 0.900` remains a reported transient
  warning, not a new failure.
- The NIIN/static-plan adapter and online path matched the legacy simulation
  dataclass, per-Token decisions, charges, and applied physics in the
  equivalence fixture. Hidden future fault/arrival/service data stayed out of
  observations; arrival ordering, one-shot decisions, delayed timer voiding,
  Replay after denied protection, budget suppression, state reset, and stable
  policy randomness were covered.
- Post-verification checks found no Python processes and no formal token,
  Stage4b, or pilot artifact directory under `artifacts`.

#### Artifacts

Source, tests, ADR, and ticket updates only:
`src/mfg_hedge/token_online.py`,
`src/mfg_hedge/hedge_simulation.py`, `tests/test_token_online.py`,
`docs/adr/0020-restore-token-player-model.md`, and the two ticket logs.
No Fit/Validation, pilot, candidate, Nash, MFG, or other formal experiment
artifact was generated.

#### Decisions and risks

- This is a restoration of the finite Token online entry contract, not a new
  game result. The historical static Mapping entry and independent
  Shared-Backup branch were preserved.
- The online ledger uses the existing transient Decimal reservation semantics:
  immediate non-refundable charge, half-open relative windows, cap-based
  admission, and no implicit D-to-I or I-to-D downgrade. The causal window
  preview does not inspect future failure boundaries.
- The default compatibility cap of 2.8125 is exercised only as a finite
  reference ledger value; it is not a payoff, tariff, price, or equilibrium
  claim.
- No version bump was made because the new entry remains internal and was not
  exported from `mfg_hedge.__init__`.

#### Next

等待用户审查 T1；下一切片才是单 Token payoff 与完整单边偏离。未开始 predictor、price、best response、forward 或 MFG。
