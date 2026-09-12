# Risk exposure horizon correction for reliability-aware routing

Type: bugfix
Status: resolved
Blocked by: none

## Scope

Correct only the causal exposure horizon used by the reliability-aware routing
score. The horizon must include the newly arriving Token's known mean service
requirement in addition to currently observable queued/running work:

`exposure_horizon = estimated_work + SERVICE_MEAN / current_speed`.

This ticket must not change the five fixed baseline policies, workload/fault
generation, CRN keys, estimator priors, event ordering, topology, or any prior
artifact. The corrected three-round evaluation, if authorized by this ticket's
acceptance criteria, must use fresh identities and a fresh artifact directory.

## Acceptance criteria

- A Red test demonstrates that with empty queues and different observed hazard,
  risk-aware routing selects the lower-risk available Replica after accounting
  for the new Token's service horizon.
- The implementation uses the current observable speed and `SERVICE_MEAN`; it
  never reads the Token's true service draw or future fault state.
- Existing five fixed policies produce byte/field-identical results on the
  frozen smoke traces before versus after the correction.
- Corrected development, robustness, and holdout runs use fresh namespaces,
  macro seeds, trace fingerprints, and artifact directories; prior
  `development_candidate`, `candidate_qualified_for_holdout`, and
  `holdout_not_improving` artifacts remain unchanged.
- All panels are complete, CRN fingerprints agree within episode, and any
  physical/invariant failure is fail-closed.
- No MFG, Nash, optimality, or final negative claim is made from the prior
  artifact or from this correction alone.

## Progress log

### Update: 2026-09-10 — Exposure-horizon correction ticket opened

Status: partial

#### Goal

Open and claim the narrow scientific correction identified in the review:
include the new Token's mean service time in the causal failure-exposure
horizon before any corrected evaluation.

#### Changed

- Created and claimed ticket 05.
- Confirmed ticket 04 is resolved and no other ticket is claimed.
- No production code, configuration, or historical artifact was changed in
  this opening update.

#### Verification

- Read `AGENTS.md`, `CONTEXT.md`, the local issue tracker and update format,
  ADR-0035, the routing spec, the resolved implementation/evaluation tickets,
  and the current routing implementation/tests.
- Confirmed the current implementation uses
  `horizon = max(estimated_work, 1e-12)` at the reported defect location.

#### Artifacts

- `.scratch/reliability-aware-multi-replica-routing/issues/05-fix-risk-exposure-horizon.md`
- Existing three-round artifacts are retained and are not modified.

#### Decisions and risks

The previous holdout result remains valid only for the prior implementation
fingerprint and must not be presented as the final ADR-0035 result. This ticket
does not reopen the routing design; it corrects a missing known service-horizon
term and requires a fresh evaluation.

#### Next

Add the failing empty-queue heterogeneous-hazard test, then implement the
smallest causal horizon correction.

### Update: 2026-09-10 — Red test reproduced

Status: partial

#### Goal

Expose the missing new-Token service horizon before changing production code.

#### Changed

- Added a focused test using empty queues and a replica with an observed prior
  failure/recovery history.
- The test checks both the lower-risk routing choice and the finite-horizon
  failure-risk value for the newly arriving Token.

#### Verification

- Real Red reproduced with:
  `.venv\\Scripts\\python.exe -m unittest tests.test_reliability_aware_routing.ReliabilityAwareRoutingTests.test_risk_aware_router_exposes_new_token_service_horizon -v`
- Failure under the old implementation: observed risk
  `2.098321516541546e-14` versus the required one-unit-horizon value
  `0.02073440484579392`.

#### Artifacts

- No experiment artifact was created or modified.

#### Decisions and risks

The routing-choice assertion alone did not expose the defect because the old
epsilon horizon still preserved a tiny hazard ordering. The explicit risk
magnitude assertion is therefore required to make the contract violation
genuinely Red without changing the test's causal information boundary.

#### Next

Implement only the current-speed-aware exposure horizon, then run the focused
regression and fixed-baseline equivalence checks.

### Update: 2026-09-10 — Baseline compatibility boundary found

Status: partial

#### Goal

Verify that the causal horizon correction does not silently rewrite the five
frozen comparator policies.

#### Changed

- Added an exact result comparison against an emulated pre-correction risk
  projection for `uniform_rr`, `jsq`, `loew`, `reliability_only`, and
  `lazarus_algorithm1`.

#### Verification

- The new baseline-equivalence test initially failed after applying the
  horizon globally: `reliability_only` changed its routing and full result
  rows, while the other policies remained unaffected.
- This is a real shared-observation compatibility conflict, not a tolerance
  issue; the frozen comparator must retain its historical risk projection.

#### Artifacts

- No experiment artifact was created or modified.

#### Decisions and risks

The corrected horizon will be selected explicitly only for the
`risk_aware_jsq` candidate scoring path. The five fixed baselines will retain
their old risk projection, while the candidate receives the corrected
new-Token service horizon. This preserves the frozen comparison contract
without changing physical execution, workload, CRN, or event semantics.

#### Next

Add the explicit observation-mode boundary and rerun the Red tests plus exact
baseline equivalence.

### Update: 2026-09-10 — Corrected evaluation identities frozen

Status: partial

#### Goal

Define fresh, non-overlapping identities for the corrected development,
robustness, and holdout rerun.

#### Changed

- Frozen corrected development R1 as namespace
  `reliability-aware-routing:v1:corrected:development:r1`, macro seed
  `20260913`.
- Frozen corrected robustness R2 as namespace
  `reliability-aware-routing:v1:corrected:development:r2`, macro seed
  `20260914`.
- Frozen corrected holdout as namespace
  `reliability-aware-routing:v1:corrected:holdout:r1`, macro seed `20260915`.
- Frozen fresh artifact root:
  `artifacts/reliability-aware-routing-corrected-evaluation-20260910/`.
- Added an internal runner entry point that supplies these identities while
  leaving the historical runner defaults unchanged.

#### Verification

- Corrected routing and experiment focused suites pass after the runner
  change; configuration check remains `status: ok`.
- The new protocol values are passed to trace generation and written into each
  corrected round protocol; no old namespace or seed is reused.

#### Artifacts

- No corrected evaluation artifact has been generated yet.
- All prior evaluation artifacts remain untouched.

#### Decisions and risks

The corrected rerun retains the historical R1/R2/R3 sample shapes, candidate
grid, selection and holdout call counts. A fresh source fingerprint will be
recorded after the causal correction. The holdout result remains a finite
routing result, not an MFG or equilibrium claim.

#### Next

Run the required full suite and config check, then execute the corrected
three-round evaluation only after those gates pass.

### Update: 2026-09-10 — Corrected evaluation complete

Status: completed

#### Goal

Complete the causal risk-exposure correction and rerun development,
robustness, and holdout under fresh identities without changing the finite
routing protocol or historical artifacts.

#### Changed

- Corrected the candidate `risk_aware_jsq` exposure horizon to include the
  known arriving Token service mean divided by the current available Replica
  speed.
- Kept the five fixed comparator policies on their historical risk projection
  so their frozen comparison rows remain unchanged.
- Added an internal fresh-identity three-round runner. No public package API
  was added and no version bump was made.

#### Verification

- Real Red: old formula returned risk
  `2.098321516541546e-14` instead of the required
  `0.02073440484579392` for the one-unit service horizon.
- Focused routing/Lazarus/experiment suites: `19/19` passed.
- Full suite: `732/732` passed.
- Config check: `status: ok`.
- `git diff --check`: passed.
- Fixed-baseline equivalence: all five policies matched field-for-field on
  all 32 frozen smoke traces.
- Corrected R1: `14,336` calls, `0` failures,
  `development_candidate`, selected `risk_aware_jsq_w50_g8_12`.
- Corrected R2: `6,144` calls, `0` failures,
  `candidate_qualified_for_holdout`.
- Corrected R3: `24,576` calls, `0` failures,
  `holdout_not_improving`; all `4,096` episode trace groups had a complete,
  single shared trace fingerprint across arms.
- Corrected formal calls: `45,056`; no retry, deleted episode, or supplemental
  draw.
- Isolated preflight: `24` calls, `3.5659074642093427` calls/s,
  `6.730404599918984` seconds, formal identity not consumed.
- Runner RSS probe returned `null`; no unsupported RSS compliance claim was
  made.

#### Artifacts

- Corrected evaluation:
  `artifacts/reliability-aware-routing-corrected-evaluation-20260910/`
- Corrected report:
  `.scratch/reliability-aware-multi-replica-routing/corrected-three-round-evaluation-report-20260910.md`
- Prior artifacts remain untouched.

#### Decisions and risks

The corrected finite holdout still does not significantly improve LOEW on the
primary overall CVaR95 comparison: point `-0.0208616941`, simultaneous 95% CI
`[-0.0729876375, 0.0312642492]`. It is therefore recorded as
`holdout_not_improving` for this corrected implementation. This supersedes the
old result only for scientific interpretation of the corrected implementation;
the old artifact itself remains immutable. No MFG, Nash, optimality, or final
negative claim is made.

#### Next

Ticket 05 is resolved. Any future algorithmic change or MFG work requires a
separate scoped ticket; this correction does not authorize solver or MFG work.
