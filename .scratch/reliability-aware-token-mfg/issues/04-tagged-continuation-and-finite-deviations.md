# Implement tagged continuation and finite-N Token deviations

Type: implementation
Status: resolved
Blocked by: none

## Goal

Estimate supported state/action continuation costs against a declared
population environment and evaluate a tagged Token's finite-system unilateral
deviations without freezing other Tokens' trajectories.

## Scope

- Freeze continuation protocol, policy factories, action bucket support, CRN
  keys, independent episode libraries, sample floors, fingerprints, and call
  budget.
- Hold `(m, x)` fixed only as the declared mean-field environment for the
  tagged response calculation.
- Re-run complete finite queues, health, Replay, failure, and drain dynamics;
  other Tokens continue to call their policy functions online.
- Report pure and regularized response diagnostics separately.
- Preserve fail-closed behavior and do not select an action or claim regret,
  Nash, or MFG.

## Frozen implementation-validation protocol

This ticket implements two separate outputs.  They are never pooled,
substituted, or given a shared claim label.

### `mean_field_continuation`

- The tagged Token receives a complete immutable transient environment path
  `(mu, nu, eta, x)` from a forward episode and treats it as exogenous.
- The tagged action cannot change that path, its aggregate shares, or any
  other Token's environment state.  The output is conditional path cost
  `Q(s, a | mu, nu, x)` only; no action selection is performed.
- The continuation evaluator receives one target per episode at most.  Target
  selection is future-blind and must be completed before any action branch.

### `finite_k_deviation`

- Only the tagged requested action is overridden.  Every other Token creates
  and calls its policy function online in every branch.
- Every branch reruns the complete finite queue, health, failure, Replay, and
  drain physics from the same immutable trace.  The tagged action may change
  later public states, cohorts, other actions, and costs.
- The output contains pathwise candidate-minus-baseline differences only.  It
  has no best-action, BR, regret, Nash, or MFG field or claim.

### Shared frozen execution contract

- An action is a `ReplicaStateActionBucket`, not a concrete Replica ID.  A
  bucket contains health, domain, queue/work, service-age, individual and
  failure-risk levels, plus Token-relative locality.  DOWN or empty buckets
  are unsupported.  If multiple eligible Replicas match, selection is stable
  uniform using the canonical SHA-256 action key; the chosen ID is audit-only.
- Mean-field namespace/seed:
  `reliability-aware-token-mfg:v1:mean-field-continuation`, `20260912`.
  Finite-K namespace/seed:
  `reliability-aware-token-mfg:v1:finite-k-deviation`, `20260913`.
  These namespaces are disjoint from ticket 03 and from each other.
- Branch keys are canonical and include namespace, episode identity, target
  batch/token, action-bucket fingerprint, and policy iteration.  Branches use
  the same workload/fault/service CRN within an episode and create fresh
  endogenous state.
- The implementation-validation panel has 16 episodes per output, at least
  2 complete paired panels per supported action bucket, and at most 8 action
  buckets per target.  Its maximum is `16 * (1 + 8)` calls per output,
  `288` calls total; a failed call is retained and never retried or replaced.
- Missing target, unsupported bucket, policy/physics failure, CRN mismatch,
  incomplete drain, non-finite cost, duplicate identity, or insufficient
  action support is fail-closed.  No formal development/holdout experiment is
  authorized by this ticket.

## Acceptance criteria

- Same-episode action branches share the causal prefix and CRN while creating
  fresh endogenous state.
- No future service draw, fault, action list, or hidden remaining work leaks
  into an observation.
- Finite-N deviation changes can propagate to other Tokens' delays.
- Unsupported cells and failed panels are nonnumeric and retained in audit.
- Tagged results carry distinct finite-deviation and mean-field provenance.

## Next dependency

Ticket 05 may start only after tagged continuation and finite deviations are
resolved.

## Progress log

### Update: 2026-09-10 — population estimator correction cleared

Ticket 07 corrected the upstream active-population representation and
per-Token conditional-share calibration.  This ticket is open again and is
not claimed; its continuation and finite-deviation scope remains unchanged.

### Update: 2026-09-10 — response estimands and execution contract frozen

Ticket 04 is now claimed.  The mean-field and finite-K outputs are separate
record types with separate labels and fingerprints.  The frozen namespaces,
seeds, action-bucket semantics, fresh paired branches, one-target rule,
sample floor, and 288-call implementation-validation ceiling are recorded
above.  This ticket still excludes prices, BR selection, regret, and MFG
solving.

### Update: 2026-09-10 — implementation started with real Red

Added the focused response test module before production implementation.
The first run was a real Red at test collection:
ModuleNotFoundError: No module named mfg_hedge.token_population_response.
The new isolated implementation is
src/mfg_hedge/token_population_response.py.  It keeps
mean_field_continuation exogenous and path-fingerprint checked, and keeps
finite_k_deviation as a fresh complete finite routing rerun with only the
tagged requested action overridden.  The implementation remains internal and
does not change the package version or historical APIs.

The seven focused response tests are now green.  They cover immutable
mean-field paths, action-bucket selection, paired finite-K routing, online
reaction by other Tokens, same-action reproduction, fresh policy creation,
call accounting, trace mismatch, and fail-closed unsupported inputs.

### Update: 2026-09-10 — implementation verified and resolved

Implemented the isolated internal module
src/mfg_hedge/token_population_response.py.

Changed:

- mean_field_continuation accepts a complete immutable
  (mu, nu, eta, x) ForwardEpisode path, keeps it exogenous, performs
  stable SHA-256 uniform selection inside observable Replica-state buckets,
  and reports bucket-level conditional cost means and standard errors only.
- finite_k_deviation creates a fresh policy and fresh endogenous routing
  engine for every baseline/candidate branch, overrides only the tagged
  requested action, and leaves every other Token online in every rerun.
- Branches carry distinct frozen namespaces, macro seeds, episode/target/
  bucket/iteration keys, trace fingerprints, policy fingerprints, and
  complete-drain checks.
- Failure, unsupported bucket, duplicate identity, prefix mismatch,
  non-finite cost, and insufficient panel support fail closed.  Baseline
  failure accounts for its attempted call.
- Results contain no action-selection, BR, regret, Nash, or MFG claim.
  Package exports, version, historical engines, configuration, and artifacts
  were not changed.

Verification:

- Real Red: initial focused collection failed with ModuleNotFoundError for
  the new module before production implementation.
- Focused response suite: 8/8 passed.
- Affected response/population/simultaneous/reliability suites: 42/42 passed.
- Full suite: 762/762 passed in 344.833 seconds.
- Config check: exit 0, status: ok.
- Same-action finite-K branch reproduced the baseline physical fingerprint.
- Finite-K paired reruns changed later online assignments when the tagged
  route changed, demonstrating finite endogenous reaction.
- Mean-field paths remained byte-stable while different episode paths were
  correctly allowed to have different fingerprints.
- No formal development/holdout calls or response experiment artifact was
  run or generated.

The implementation-validation portion of ticket 04 is complete.  Later
continuation sampling, BR selection, prices, and MFG solver work remain
outside this ticket.
