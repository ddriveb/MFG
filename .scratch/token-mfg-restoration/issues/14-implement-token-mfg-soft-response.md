Type: implementation and experiment

Status: resolved

Blocked by: none

## Goal

Implement and run the T4b zero-price soft-response population diagnostic:

```text
fixed episode library -> Q/action-gap/paired-SE -> SoftBR -> damped probability policy
```

This ticket must stop at a finite-population soft-policy diagnostic.  It must
not claim Nash, MFG, regret, or introduce price feedback.

## Frozen T4b contract

- Proposed ADR-0024 is the governing design; it remains Proposed during this
  implementation unless separately accepted.
- Environment and physical engine remain `theta_token_load0p7_v1` and the T4
  runtime ADR-0009 scorer.  Nine T3A-retained bins, N/D/I, reservation,
  slowdown `2.0`, hedge delay `2.0`, full drain, and 32-panel floors remain
  unchanged.
- Generate one immutable 2,048-episode library under namespace
  `token-mfg-restoration:t4b:soft-fixed-point:v1:library`, macro seed
  `20260914`. Reuse the same episode fingerprints in every round; create fresh
  endogenous state for every selection/baseline/counterfactual run.
- Initial policy is online NIIN. Use `beta=1.0`, `eta=0.2`, zero price, and a
  maximum of 20 rounds. Per-round and total declared scheduler-call ceilings
  are 10,240 and 204,800.
- Keep all per-bin N/D/I probabilities, Q mean/SE/n/status, action gap, paired
  action-difference SE, occupancy, policy-probability residual, and population
  occupancy residual. Repeated hard BR fingerprints are diagnostic only.
- Allowed terminal statuses are `soft_fixed_point`, `not_converged_20`, and
  `statistics_insufficient`. All claim flags remain false.

## Acceptance criteria

1. Red tests first cover numerically stable SoftBR, probability normalization,
   exact damping, fresh online policy state, fixed-library reuse and CRN,
   paired gap/SE, unknown-bin NIIN fallback, residual stopping, call budget,
   and transactional artifact behavior.
2. The implementation never replays a previous action list and never changes
   the physical engine or price.
3. The real bounded run records every round until a permitted status and
   stores the full probability/Q/gap/occupancy rows.
4. Focused suite, full suite, and config check pass.

## Progress log

### Update: 2026-09-08 — Claimed T4b soft-response diagnostic

Status: partial

#### Goal

Replace T4 hard argmin oscillation with a zero-price soft response and fixed
episode library while preserving full online physical reruns.

#### Changed

- Created and claimed this single T4b implementation ticket.
- Added Proposed ADR-0024 with the frozen beta, eta, library, paired-gap, and
  claim-boundary semantics.
- No production code, tests, or T4b artifact has been changed yet.

#### Verification

- Read repository rules, T4 ticket/artifact, token online/deviation/oracle
  modules, and the existing T4 implementation.
- Red: `.venv\\Scripts\\python.exe -m unittest
  tests.test_token_mfg_soft_fixed_point -v` ran 7 tests and all 7 failed with
  `ModuleNotFoundError` because the T4b module did not yet exist.

### Update: 2026-09-08 — Soft-response core implementation

Status: partial

#### Goal

Implement the numerically stable soft response and its immutable online policy
function before running the physical library loop.

#### Changed

- Added `src/mfg_hedge/token_mfg_soft_fixed_point.py` with stable SoftBR,
  immutable NIIN-plus-soft-response mixtures, fresh online episode state,
  paired action-gap/SE calculation, fixed-library fingerprint validation,
  residual stopping, call accounting, and transactional artifact output.
- Added `tests/test_token_mfg_soft_fixed_point.py` with seven focused contract
  tests.

#### Verification

- Real Red: focused suite ran 7/7 failures with `ModuleNotFoundError` before
  the module existed.
- Focused Green: `.venv\\Scripts\\python.exe -m unittest
  tests.test_token_mfg_soft_fixed_point -v` ran 7/7 passed.
- Module compilation passed with `.venv\\Scripts\\python.exe -m py_compile
  src\\mfg_hedge\\token_mfg_soft_fixed_point.py`.
- Full physical smoke is pending.

#### Artifacts

- None. T4-v1 artifact remains unchanged.

#### Decisions and risks

- `beta=1.0` is frozen in Proposed ADR-0024; hard-BR repeats are diagnostic
  only and cannot stop the soft run as a cycle.

#### Next

Run the small fixed-library physical smoke, then run the frozen T4b diagnostic
only if CRN reuse and residual accounting pass.

### Update: 2026-09-08 — Fixed-library physical smoke

Status: partial

#### Goal

Verify that T4b reuses one immutable episode library while each round reruns
the complete online physical panel with fresh endogenous state.

#### Changed

- No additional production changes; the smoke exercised the new runner with a
  reduced test-only episode count and bin support.

#### Verification

- A 16-slot smoke over all nine bins correctly failed closed with
  `statistics_insufficient` because several bins had no complete panel.
- A second 16-slot smoke over the six observed bins ran two rounds, used 136
  calls, and reported one identical library fingerprint across both rounds.
- The two rounds retained probability residuals `0.39978357778965795` and
  `0.32387346207831297`, and population residual `0.125`; no false convergence
  was reported.

#### Artifacts

- None. These were in-memory smoke runs; no T4b artifact was written.

#### Decisions and risks

- The physical smoke confirms fixed-library reuse and full rerun semantics,
  but it is not the frozen 2,048-episode statistical diagnostic.

#### Next

Run the frozen T4b diagnostic with all nine T3A-retained bins and the declared
20-round ceiling.

### Update: 2026-09-08 — Memory-bound correction before frozen run

Status: partial

#### Goal

Prevent the multi-round result accumulator from retaining full observation
objects after residuals have been computed.

#### Changed

- The runner now consumes the in-memory observation list immediately after
  computing each round's residual inputs and retains only
  `observed_target_count` in the durable iteration row.
- Zero paired-SE gaps now serialize as `null` rather than non-finite JSON.
- The in-progress full run was stopped before artifact commit; no T4b result
  was published.

#### Verification

- Before the correction, the full fixed-library process reached about 507 MiB
  RSS while retaining the first round's raw observation objects; it was
  stopped fail-closed before publication.
- Focused suite after the correction: 7/7 passed.
- Module compilation passed.
- Earlier 16-slot fixed-library smoke remained valid: two rounds, 136 calls,
  one library fingerprint.

#### Artifacts

- None from the stopped run; the requested T4b run directory does not exist.

#### Decisions and risks

- This correction changes only result retention, not the fixed library, online
  decisions, physical reruns, Q values, or residual definitions.

#### Next

Rerun the frozen T4b diagnostic after the bounded-memory correction.

### Update: 2026-09-08 — T4b diagnostic completed

Status: completed

#### Goal

Run the zero-price soft-response population loop on one fixed external episode
library and distinguish probability convergence from hard-BR repetition.

#### Changed

- Completed the T4b runner with numerically stable `SoftBR`, per-bin N/D/I
  probability functions, exact function damping, fixed-library reuse, paired
  action-gap/SE rows, residual stopping, and fail-closed panel floors.
- Kept all physical evaluation online: every round created fresh engine and
  policy state, and no prior action list was replayed.
- Kept the implementation internal; package version remains `0.24.0`.

#### Verification

- Real Red: focused suite initially ran 7/7 failures with
  `ModuleNotFoundError` before the T4b module existed.
- Focused Green: `.venv\\Scripts\\python.exe -m unittest
  tests.test_token_mfg_soft_fixed_point -v` — `7/7` passed.
- Full suite: `.venv\\Scripts\\python.exe -m unittest discover -s tests -v`
  — `585/585` passed in 162.030 seconds.
- Config: `.venv\\Scripts\\python.exe -m mfg_hedge check --config
  configs\\v1_minimal.json` — `status: ok`; the existing 1.4 transient
  single-domain load warning remains diagnostic.
- T4b artifact contains 2,048 library fingerprints and one identical library
  fingerprint across all four recorded rounds.
- Three rounds had complete 27-row Q tables with minimum action-panel n of
  78, 57, and 40.  Round 4 retained all rows but reached minimum n=21, so
  the final status is `statistics_insufficient`.
- Calls: `37,072` attempted and `37,072` reserved, below the declared
  `204,800` ceiling.
- Probability residuals for complete rounds were
  `0.396191457626772`, `0.31262840147221704`, and `0.24168238238140505`.
  Population occupancy residuals were `0.1025390625` and `0.0615234375`
  where comparable; neither met `0.01`.
- All 27 complete bin-round gap rows include paired SE.  Six gaps were at or
  below their paired SE and three had zero paired SE; hard-BR repeats were
  retained only diagnostically and did not terminate the soft run as a cycle.
- Peak observed process RSS was approximately 581 MiB during the fixed-library
  run.  Raw observations are discarded after residual calculation, so RSS did
  not grow linearly with the number of rounds.
- All claim flags (`best_response`, `regret`, `nash`, `mfg`) are `false`.

#### Artifacts

- [T4b diagnostic artifact](D:/project/mfg_hedge_v1/artifacts/token-t4b-soft-fixed-point-20260908-r1/)
  with manifest, metric contract, fixed library fingerprints, per-round
  policy/gap/occupancy rows, Q rows, and summary.
- No price-feedback, predictor, MFG, Nash, or formal campaign artifact was
  generated.

#### Decisions and risks

- Zero-price SoftBR reduced hard-table overreaction but did not establish a
  soft fixed point before the per-bin statistical floor failed.
- The result is `statistics_insufficient`, not `not_converged_20`, because the
  fourth round lost the required complete panel support.
- The fixed library improves cross-round comparability but does not turn the
  finite sample into an exact conditional law.
- ADR-0024 remains `Proposed`; no price mechanism was added or accepted.

#### Next

Review whether the declared fixed-library memory/cost and per-bin floor are
acceptable before any separate T4c design; do not add price feedback or call
the T4b result an equilibrium.

### Update: 2026-09-08 — Final verification after immutability correction

Status: completed

#### Goal

Close T4b after rerunning the required verification following the final
`SoftPolicyState` immutability correction.  This update does not rerun or
overwrite the completed diagnostic artifact.

#### Changed

- Made `SoftPolicyState` strictly immutable and retained the existing online,
  fixed-library, fail-closed execution semantics.
- The correction affects only policy-state mutability; it does not alter the
  episode library, physical results, Q rows, residuals, call accounting, or
  artifact contents.

#### Verification

- Real Red: the initial T4b focused run had `7/7` real
  `ModuleNotFoundError` failures before the implementation existed.
- Focused suite after the final correction:
  `.venv\\Scripts\\python.exe -m unittest
  tests.test_token_mfg_soft_fixed_point -v` — `7/7` passed.
- Full suite after the final correction:
  `.venv\\Scripts\\python.exe -m unittest discover -s tests -v` —
  `585/585` passed in `159.469` seconds.
- Config check:
  `.venv\\Scripts\\python.exe -m mfg_hedge check --config
  configs\\v1_minimal.json` — `status: ok`; the existing 1.4 transient
  single-domain load warning remains diagnostic.
- Frozen diagnostic result remains `statistics_insufficient`: four recorded
  rounds, complete-round minimum panel sizes `78`, `57`, and `40`, final
  retained-round minimum `21`, and `37,072/204,800` attempted/reserved calls.
- Complete-round policy residuals remain
  `0.396191457626772`, `0.31262840147221704`, and
  `0.24168238238140505`; comparable occupancy residuals remain
  `0.1025390625` and `0.0615234375`.  Neither convergence criterion met
  `0.01`.
- The fixed 2,048-episode library fingerprint is reused in every recorded
  round.  Six of 27 complete bin-round gaps are at or below paired SE and
  three have zero paired SE; hard-BR repeats remain diagnostic only.
- Peak observed process RSS remains approximately `581 MiB`; all claim flags
  remain false.

#### Artifacts

- [T4b diagnostic artifact](D:/project/mfg_hedge_v1/artifacts/token-t4b-soft-fixed-point-20260908-r1/)
  remains the sole T4b run artifact; no artifact was overwritten.
- No price-feedback, predictor, Nash, MFG, or formal campaign artifact was
  generated.

#### Decisions and risks

- Ticket is resolved with a fail-closed `statistics_insufficient` diagnostic,
  not a soft fixed point or equilibrium result.
- ADR-0024 remains `Proposed`; package version remains `0.24.0`.
- No campaign, price feedback, BR claim, regret claim, Nash claim, or MFG
  conclusion was started or produced.

#### Next

Any T4c or price-feedback work requires a separate design and ticket; do not
reinterpret this insufficient finite-population diagnostic as an equilibrium.

## Answer

Status: resolved

T4b was implemented and executed under the frozen zero-price soft-response
protocol.  It stopped fail-closed with `statistics_insufficient`; no soft
fixed point, Nash, or MFG result was claimed.

#### Artifacts

- [T4b diagnostic artifact](D:/project/mfg_hedge_v1/artifacts/token-t4b-soft-fixed-point-20260908-r1/)
  remains unchanged and is marked `statistics_insufficient`.

#### Decisions and risks

- ADR-0024 is intentionally Proposed, not Accepted.
- T4b is a finite-population soft-policy diagnostic, not an equilibrium or
  conditional-law MFG implementation.

#### Next

Use the recorded timing, memory, residual, and panel-floor evidence to design
any later T4c or price-feedback work separately; no equilibrium claim is
licensed by this ticket.
