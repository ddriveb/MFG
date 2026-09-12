# Correct population empirical state and share calibration

Type: correction
Status: resolved
Blocked by: none

## Goal

Correct the population-estimator implementation from ticket 03 without
rewriting its resolved history.  The correction must represent the active
Token empirical measure `mu_t` and estimate conditional policy/assignment
shares at the Token-cell level, while preserving the accepted ADR-0037
causal, transient, simultaneous-routing semantics.

## Scope

- Add an immutable, canonical active-Token state row to each decision-boundary
  audit.  It must contain only arrived, nonterminal, observable state; future
  arrivals, service draws, hidden remaining work, and future events remain
  excluded.
- Make `MuState` hold the complete active-Token empirical measure, with
  canonical rows and class-count compatibility fields derived from those rows.
  The decision cohort remains `eta_t`; queued/running Tokens are in `mu_t` but
  are not repeated in `eta_t`.
- Record one immutable predicted action distribution and one actual assignment
  for every decision-cohort Token.  The batch `x_t` and realized share remain
  aggregates of those per-Token rows, not substitutes for them.
- Calibrate per `(cell, episode)`: average the Token rows belonging to that
  cell within each episode, then give each episode one equal weight.  Do not
  pseudo-replicate an episode by its Token or batch count.
- Preserve ticket 03's protocol identities, CRN, source fingerprints, fresh
  endogenous engine state, and fail-closed validation.  Do not enter
  continuation, best response, prices, or an MFG solver.

## Acceptance criteria

- A mixed Regular/Urgent decision cohort has distinct per-Token state and
  conditional predicted/realized rows where the policy distinguishes them.
- `MuState` is a true active empirical measure and is causally bounded at each
  decision time.
- Per-cell predictions and assignments are not copied from a whole-batch
  share; calibration is episode-weighted and deterministic.
- Legacy fixed physical routing behavior and ticket 03's resolved artifact
  history are unchanged.
- Focused, affected, full-suite, and configuration checks pass.

## Progress log

### Update: 2026-09-10 — correction claimed

#### Goal

Repair the two implementation gaps identified after ticket 03: aggregate
class counts are not an empirical `mu_t`, and batch-level shares cannot
identify `pi(a | s, nu, x)`.

#### Changed

- Ticket 04 is blocked by this correction.  Ticket 03 remains resolved with
  its historical progress log unchanged.
- No continuation, BR, price, or MFG code is in scope.

#### Verification

- Red tests will cover a mixed Regular/Urgent cohort, true active-state rows,
  per-Token predictions/assignments, and equal episode weighting.

#### Artifacts

- No experiment artifact is created by this correction.

#### Decisions and risks

- The correction changes estimator representation and audit exposure only; it
  does not alter the finite routing physics or historical artifacts.

#### Next

- Implement the minimum engine audit exposure and population calibration
  correction, then run the required verification suites before resolving this
  ticket.

### Update: 2026-09-10 — correction resolved

#### Goal

Correct ticket 03's population representation and conditional-share
calibration without changing the finite routing physics or entering later MFG
stages.

#### Changed

- Added immutable `ActiveTokenState` rows to simultaneous decision-boundary
  audits.  Rows are canonical by `(arrival_time, token_id)` and distinguish
  `waiting`, `queued`, `running`, and `decision`; only arrived, nonterminal
  Tokens are included.
- Replaced aggregate-only `MuState` storage with the complete active empirical
  measure.  `active_token_count` and `class_counts` remain derived compatibility
  properties.
- Added immutable `TokenShareRecord` rows containing each cohort Token's
  predicted distribution, cell key, and post-commit assignment.  Batch `x_t`
  is now the aggregate of these predictions.
- Reworked calibration to average Token rows within each `(cell, episode)`
  first, then average episodes equally.  Token multiplicity no longer creates
  pseudo-replication, and mixed Regular/Urgent cells retain distinct
  conditional values.
- Ticket 03's resolved history, physical routing behavior, version, and
  historical artifacts were left unchanged.  Ticket 04 was unblocked after
  this correction; no later-stage implementation was started.

#### Verification

- Real Red: the new correction tests initially produced **1 error and 1
  failure in 13 tests**: `MuState` had no active-token rows, and mixed-cell
  calibration returned the copied batch share instead of the Token-conditional
  share.
- Focused population estimator suite: **15/15 passed**.  Simultaneous routing
  suite: **7/7 passed**.
- Affected game/workload/expert/deviation/solver suites: **44/44 passed**.
- Full suite: `.venv\\Scripts\\python.exe -m unittest discover -s tests -v`,
  **754/754 passed** in 343.619 seconds.
- Config check: `.venv\\Scripts\\python.exe -m mfg_hedge check --config
  configs/v1_minimal.json`, exit 0 with `status: ok`.
- Verified future arrivals are excluded, running/new decision states remain
  distinct, per-Token predictions and assignments are retained, mixed-class
  cells remain conditional, and unequal Token multiplicity still gives each
  episode one calibration weight.

#### Artifacts

- No experiment, continuation, BR, MFG, or candidate artifact was generated.

#### Decisions and risks

- The corrected `MuState` is an empirical active-Token measure, not merely a
  pair of class counters.  `eta_t` remains the decision cohort and is not used
  to replace `mu_t`.
- Conditional calibration requires the new online
  `predict_token_share(observation, context)` policy method; it never reuses a
  pre-generated action list or future state.
- No package version bump was made because the corrected modules are not
  exported from `mfg_hedge.__init__`; historical scientific results are
  unaffected.

#### Next

- Ticket 04 may proceed from its own scope.  This ticket is resolved; do not
  start continuation, BR, prices, or MFG work as part of this correction.

### Update: 2026-09-10 — post-validation consistency rerun

The final batch-record self-consistency checks were added after the first
full-suite run and then reverified.  The complete suite passed **754/754** in
349.965 seconds, and `mfg_hedge check --config configs/v1_minimal.json`
returned exit 0 with `status: ok`.  No ticket status or scientific scope
changed during this rerun.
