# ADR-0025: Token-MFG T4c support-weighted soft-response diagnostic

Status: Accepted (confirmed on 2026-09-08)

Date: 2026-09-08

## Context

The T4b zero-price Soft-BR diagnostic showed decreasing policy residuals,
but stopped fail-closed because a bin that was initially populated became
rare under the updated population policy.  Requiring every originally
retained bin to satisfy the panel floor in every round confounds statistical
support with the fixed-point question.

T4c keeps the zero-price physical and statistical protocol unchanged while
making support explicit.  It is still a finite-population diagnostic and does
not establish a Nash equilibrium or a complete MFG.

## Decision

Use the following Proposed T4c contract:

- Keep `beta=1.0`, `eta=0.2`, at most 20 rounds, the T4 physical protocol,
  the nine retained T3A bins, N/D/I, complete drain, and the existing paired
  panel scorer.
- Build one immutable library of 4,096 `EpisodeIdentity` records under
  namespace `token-mfg-restoration:t4c:soft-fixed-point:v1:library` and macro
  seed `20260915`.  The library stores identities and its canonical
  fingerprint only; it does not retain complete workload traces.
- Reconstruct each episode deterministically from its identity when a round
  needs it.  Reconstruction must produce the same trace fingerprint for the
  same identity in every round.  Every physical branch still creates fresh
  endogenous policy and engine state.
- Use the following support classification independently for every retained
  `(bin_id, round)`:
  - `inactive`: zero selected targets reached the bin;
  - `unresolved`: positive bin occupancy but fewer than 32 complete paired
    panels;
  - `active`: at least 32 complete paired panels.
  No positive occupancy may be silently classified as inactive.
- Only active bins participate in Soft-BR and policy updates.  Inactive bins
  do not block a round.  Any unresolved bin stops the run fail-closed with
  `statistics_insufficient`.
- Report the current occupancy distribution every round.  The policy residual
  is both the current-population weighted residual
  `sum_b m_k(b) * L1(pi_next(b), pi_k(b))` over retained-bin support and the
  maximum active-bin residual.  The weighted residual must not hide the
  maximum active-bin residual.
- Convergence requires two comparable rounds for which both weighted policy
  residual and maximum active-bin residual are at most `0.01`, and the
  retained-bin occupancy residual is at most `0.01`.  Hard-BR fingerprint
  repetition remains diagnostic only.
- A formal T4c execution records, for every round, active/inactive/unresolved
  bins, occupancy mass, weighted and maximum-active residuals, N/D/I
  probability changes, panel counts, scheduler calls, and latency, Replay,
  Hedge, work, and Protection-Storm metrics.
- The only allowed formal terminal statuses are
  `supported_soft_fixed_point`, `statistics_insufficient`,
  `not_converged_20`, and `physical_failed`.  A supported fixed point still
  means only a zero-price finite-Token supported soft-fixed-point candidate;
  it is not a formal MFG equilibrium.
- Per-round calls are at most `4,096 * 5 = 20,480`; the declared 20-round
  ceiling is `409,600`.  There is no retry, supplementation, bin merge, or
  result-driven sample change.
- Artifact output uses a fresh run directory and includes identity-library
  fingerprint, reconstruction fingerprints, support classifications,
  occupancy, Q/gap/paired-SE rows, policy rows, residuals, and call ledger.

## Consequences

The gate follows the current population support without declaring a rare
state solved.  A positive but undersampled state remains an explicit
`unresolved` failure.  Identity-only storage bounds memory, but repeated
deterministic reconstruction adds CPU work and must be measured.

## Claim boundary

T4c artifacts must set `claims_best_response`, `claims_regret`,
`claims_nash`, and `claims_mfg` to `false`.  Price feedback, predictor
training, and any equilibrium qualification remain outside this ADR.
