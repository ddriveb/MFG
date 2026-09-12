# Correct K-scale risk composition and action CRN identity

Type: correction
Status: resolved
Blocked by: none

## Goal

Correct the two qualification-readiness contracts that affect scale trends and
paired action confidence intervals, without changing routing policy semantics,
physical dynamics, sample protocols, or historical artifacts.

## Scope

- Keep S1/S2 high-risk replica composition at 25% for every supported
  `Topology(K)` by applying the existing risk classes to `replica_id % 8`.
- Keep K=8 risk assignments and deterministic outputs unchanged.
- Make policy action keys episode-specific and include namespace, macro seed,
  scenario, episode index, batch ID, token ID, and action/bucket identity.
- Keep action keys free of model ID, solver policy iteration, price, queue
  outcome, and realized service information so fixed episodes remain paired
  across policies and solver iterations.
- Preserve all fixed-dispatcher and historical output behavior.
- Do not run K-scale qualification, development, holdout, or any formal
  experiment.

## Acceptance criteria

1. S1 and S2 risk membership uses the modulo-8 type mapping for K=8, 16, 32,
   and 64; K=8 is a deterministic regression.
2. Policy action keys differ across episode/scenario identity, repeat exactly
   for the same episode, and are unchanged when only policy iteration/model
   or price metadata changes.
3. Population-response branch keys include the full action identity and
   scenario/episode identity but exclude policy iteration and model/price
   metadata from the random-key payload.
4. New Red tests reproduce the pre-fix high-risk ratio and CRN defects; the
   focused suite passes after the smallest implementation change.
5. Focused, affected, full-suite, and config checks pass; no qualification
   artifact is created.

## Non-goals

- No changes to scheduler ordering, estimator mathematics, solver stopping,
  topology scaling, policy definitions, sample sizes, or call budgets.
- No qualification run and no new ADR.

## Progress log

### Update: 2026-09-10 — correction ticket claimed

Status: partial

#### Goal

Claim the sole active correction ticket and establish the bounded CRN and
K-scale fixes before qualification.

#### Changed

- Created and claimed ticket 09.
- Blocked ticket 06 on this correction ticket.
- No production code or experiment artifact changed.

#### Verification

- Confirmed ticket 08 is resolved and no other ticket is claimed.
- Read the repository rules, update format, current ticket 06, and relevant
  K-scale/action-key implementation paths.

#### Artifacts

None.

#### Decisions and risks

The correction remains isolated from the qualification protocol and preserves
all existing historical artifacts and experiment identities.

#### Next

Add the failing tests before modifying production code.

### Update: 2026-09-10 — Red reproduced

Status: partial

#### Goal

Demonstrate the K-scale risk-composition and action-CRN defects with focused
tests before touching production code.

#### Changed

- Added `tests/test_k_scale_action_crn_correction.py` covering fixed-fraction
  S1/S2 risk classes, episode/scenario action-key identity, iteration-invariant
  population branch keys, bucket identity, and K=8 determinism.

#### Verification

- `.venv\\Scripts\\python.exe -m unittest tests.test_k_scale_action_crn_correction -v`
  failed as intended: 2 failures and 1 error in 4 tests.
- The failures reproduced high-risk count `2 != 4` at K=16, identical action
  keys for distinct episodes, and `_branch_key` rejecting the required
  scenario argument while retaining the old iteration contract.

#### Artifacts

None.

#### Decisions and risks

The Red test confirms the correction is necessary and bounded. No production
code, qualification call, or experiment artifact was changed.

#### Next

Implement only the modulo-8 risk mapping and stable action-key identity, then
rerun this focused suite.

### Update: 2026-09-10 — correction verified and resolved

Status: completed

#### Goal

Complete the bounded K-scale risk and action-CRN correction while preserving
K=8 science and keeping qualification disabled.

#### Changed

- Changed S1/S2 high-risk membership to the repeating modulo-8 type map, so
  `{0,1}` and `{2,3}` remain 25% classes for K=8, 16, 32, and 64.
- Changed the simultaneous online policy key to include namespace, macro seed,
  scenario, episode index, batch, Token, and base action/bucket identity.
- Added the scenario to immutable `ForwardEpisode` provenance and passed it to
  population-response branch keys.
- Removed policy iteration from the branch-key payload while retaining the
  validated compatibility argument; branch keys now include scenario, episode,
  batch, Token, and bucket identity.
- Updated finite-K target-call detection for the new stable key format.
- Added a trace/episode scenario consistency check before finite-K execution.
- Added focused regression coverage in
  `tests/test_k_scale_action_crn_correction.py`.

#### Verification

- Red: `.venv\\Scripts\\python.exe -m unittest tests.test_k_scale_action_crn_correction -v`
  failed as intended with 2 failures and 1 error in 4 tests, reproducing the
  K=16 high-risk count defect, episode-colliding action keys, and the old
  branch-key iteration/scenario contract.
- Green focused suite: 4/4 passed.
- Affected suites: 63/63 passed in 213.885 seconds, covering reliability,
  simultaneous routing, population estimation, population response, solver
  readiness, and the correction tests.
- Full suite: 783/783 passed in 420.747 seconds.
- `.venv\\Scripts\\python.exe -m mfg_hedge check --config configs\\v1_minimal.json`:
  exit 0, `status: ok`; the existing informational single-domain load finding
  remains unchanged.
- `git diff --check`: exit 0; only pre-existing LF/CRLF warnings were emitted.
- Existing K=8 fixed-baseline and simultaneous physics regressions passed
  without qualification calls or changed historical outputs.

#### Artifacts

None. No development, holdout, qualification, or formal experiment was run;
no artifact was created or overwritten.

#### Decisions and risks

The action key now pairs policies and solver iterations on the same exogenous
episode while separating independent episodes/scenarios. The correction does
not change scheduler semantics, policy choices, solver mathematics, sample
counts, namespaces, seeds, or call budgets. Ticket 06 may resume its own
review/claim process, but this correction does not authorize qualification.

#### Next

Restore ticket 06 to `open` with no blocker; leave it unclaimed pending its
separate qualification start decision.
