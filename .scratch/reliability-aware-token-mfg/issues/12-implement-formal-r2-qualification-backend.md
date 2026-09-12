# Implement formal r2 qualification backend

Type: implementation
Status: resolved
Blocked by: none

## Goal

Implement the real dynamic qualification backend permitted by accepted
ADR-0038, without running r2 qualification or holdout in this ticket.

## Frozen scope

- Use `QualificationPlan.r2()` and only fresh r2 qualification identities.
- Implement both `unpriced_mfg` and `priced_mfg` model rows.
- Execute K=8 from `uniform`, `simultaneous_loew_projection`, and
  `corrected_risk_aware_projection`.
- Run at most six response iterations per model/start and one final-policy
  confirmation using the same policy/environment/Q tuple.
- After the three K=8 starts agree and pass all gates, continue only in order
  K=16 -> K=32 -> K=64.
- Submit every dynamic phase atomically with serialized PolicyState,
  ForwardEpisode identities, output fingerprints, and exact call accounting.
- Stop immediately on physical failure, CRN/fingerprint mismatch, unsupported
  panel, statistics insufficiency, residual failure, multiple candidates, or
  unresolved checkpoint state.
- Holdout code path remains unreachable until K=64 qualification passes every
  frozen gate.

## Non-goals

Do not change routing physics, estimator mathematics, action support, sample
floor, namespace contract, call budget, prices, stopping tolerances, or
holdout protocol. Do not launch r2, generate candidate artifacts, or make an
MFG/Nash claim.

## Acceptance

- Red tests cover dynamic phase dependency, both model identities, three K=8
  starts, six-iteration stop, final confirmation, K-scale continuation,
  fail-fast gates, and holdout gating.
- Bounded fixtures prove one/eight-worker and interruption byte identity.
- A formal backend dry-run can build the exact frozen work shape without
  dispatching qualification identities.
- Focused, affected, full suite, config check, and diff check pass.

## Progress log

### Update: 2026-09-11 — formal r2 backend completed without campaign launch

#### Goal

Complete the real dynamic r2 qualification backend and leave formal
qualification/holdout unstarted.

#### Changed

- ADR-0038 is accepted with fresh r2 qualification namespaces and seeds; the
  contaminated r1 identity remains permanently retired.
- Added `FormalQualificationBackend` and `FormalQualificationRunner` for the
  two frozen models, K=8 three-start order, six-iteration dynamic phases,
  final-policy confirmation, and ordered K=16 -> K=32 -> K=64 continuation.
- Dynamic phases are checkpointed only after their predecessor commits;
  serialized policy, episode identities, plan/input/source fingerprints and
  exact call counts are carried across the boundary.
- Finite-K output now carries a state/action grouped simultaneous UCB using
  baseline-minus-deviation gain and paired standard error.  Incomplete
  support is explicitly `statistics_insufficient` and cannot produce a
  candidate.
- Final confirmation carries the same finite-K UCB status and rejects stale
  policy/environment/Q tuples.  Holdout remains reachable only through the
  explicit qualification-pass gate.

#### Verification

- Historical Red remains recorded above: the concrete backend and formal
  runner were initially absent; the first import-focused tests failed before
  implementation.
- Current focused r2 correction suite: `6/6` passed.
- Affected qualification/backend/solver suites: `34/34` passed.
- Full suite: `811/811` passed in `285.177s`.
- Config check: exit `0`, `status: ok`.
- `py_compile`: passed for the formal backend, qualification backend and
  response solver.
- `git diff --check`: passed; only pre-existing unrelated line-ending
  warnings under `rl_algorithms` remain.
- Non-formal 32-episode fixture exercised both models and the real physics;
  each bounded run used `2,464` scheduler calls and ended
  `not_converged_6`/`not_converged`, so it is not a qualification result.
- The formal finite-K fixture emitted a complete simultaneous-UCB payload.
- No r2 qualification or holdout call was dispatched; no r2 checkpoint or
  experiment artifact was created.  The only checkpoint remains the retired
  r1 journal.

#### Artifacts

No qualification, candidate, holdout, Nash or MFG artifact was generated.

#### Decisions and risks

- The backend is now the only permitted route to the frozen r2 runner; no
  synthetic provider is accepted.
- `32/160/160` is treated as the exact selected-target panel shape for the
  bounded fixture and remains below the frozen `32/288/288` upper bounds; the
  runner never truncates or pads panels.
- Passing implementation tests is not a qualification result and does not
  establish a regularized MFG candidate.

#### Next

Ticket 06 is restored to `open / Blocked by: none`, remains unclaimed, and may
only start after its separate timing/preflight gate and fresh r2 checkpoint
creation.  Holdout remains inaccessible until all K=64 qualification gates
pass.
