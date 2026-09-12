# ADR-0038: Qualification r2 after qualification-namespace contamination

Status: Accepted (confirmed on 2026-09-11)

## Context

The unique r1 qualification namespace was used by an uncommitted 32-episode
panel-cardinality probe. The probe did not write the r1 checkpoint, but it did
execute real routing physics against qualification identities. Those
identities are therefore not reusable as a formal campaign input.

The r1 run and checkpoint remain retained for audit and are permanently
abandoned. This ADR does not reinterpret the probe as a qualification result
and does not change the mathematical model, sample floors, or total call
budget.

## Decision

Create a new qualification run only after the panel backend correction and an
independent dry run pass.

### r2 identities

- Run ID: `reliability-aware-token-mfg-qualification-20260911-r2`
- Forward namespace: `reliability-aware-token-mfg:v1:r2:qualification:forward`
- Forward macro seed: `20260921`
- Continuation namespace: `reliability-aware-token-mfg:v1:r2:qualification:continuation`
- Continuation macro seed: `20260922`
- Finite-K namespace: `reliability-aware-token-mfg:v1:r2:qualification:finite-k`
- Finite-K macro seed: `20260923`

The frozen holdout namespace and identities remain unused until r2
qualification passes. No r1 trace, output, rate row, or checkpoint record may
be reused in r2.

### Panel cardinality

Each episode selects at most one future-blind target using the already frozen
target-selection rule. A missing target is fail-closed and consumes no formal
replacement episode.

- Forward: exactly one complete physical run per episode, `32` calls.
- Mean-field continuation: one baseline plus every supported action for the
  selected target; at most `32*9 = 288` calls.
- Finite-K deviation: one baseline plus every supported action for the same
  selected target; at most `32*9 = 288` calls.

Actual calls are the exact number of selected targets and supported actions.
The runner must not truncate excess targets, fill short panels with duplicate
calls, or borrow a target from another episode.

### Dry run gate

Before r2 qualification, use only the independent non-formal namespace:

- `reliability-aware-token-mfg:v1:r2:dry-run`
- macro seed `20260926`
- 32 episodes with a separate dry-run source/protocol fingerprint

The dry run must verify target uniqueness, action-support accounting, sample
floor handling, CRN identity, committed-boundary reuse, and one/eight-worker
determinism. It produces no qualification or holdout artifact.

## Consequences

- r1 cannot be resumed and is not a failed qualification result; it is an
  abandoned contaminated run.
- r2 has fresh qualification identities and a fresh run/checkpoint.
- Holdout remains unavailable until the complete r2 qualification gates pass.
- The qualification budget remains `51,072`, holdout remains `8,064`, and the
  total frozen ceiling remains `59,168`.

## Scope boundary

This ADR only repairs provenance and panel construction. It does not alter
the routing physics, population estimator, response model, finite-K estimand,
prices, sample floors, convergence gates, or holdout comparisons.
