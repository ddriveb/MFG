# ADR-0039: Formal Token-MFG scale handoff and qualification gates

Status: Accepted (confirmed on 2026-09-11)

Date: 2026-09-11

## Context

The r2 backend connects the real physical layers, but a pre-launch audit found
that the formal runner can be constructed without the accepted timing report,
expects caller-supplied K=8 and larger-K PolicyState objects, and returns
`qualification_pass` without a fully defined cross-K policy distance or
normalized finite-K UCB.  Constructing those caller inputs from the formal
libraries would consume uncheckpointed qualification identities.  In addition,
the checkpoint path does not yet reject a dispatch that would exceed the global
51,072-call qualification ceiling.

These are execution-contract gaps, not changes to the accepted MFG equations or
routing physics.  Starting r2 before closing them could produce an
unreproducible initialization or an invalid scale claim.

## Decision

If accepted, the formal runner uses the following deterministic contracts.

1. The constructor requires a complete `QualificationTimingReport` for the r2
   plan and calls `plan.require_formal_start` before creating or reading a formal
   checkpoint.
2. K=8 starts are generated inside the first checkpointed forward boundary:
   - `uniform`: uniform over currently UP Replicas;
   - `simultaneous_loew_projection`: choose the UP Replica with minimum
     `(estimated_work, queue_depth, replica_id)`;
   - `corrected_risk_aware_projection`: choose the UP Replica minimizing
     `estimated_work + gamma_k * estimated_failure_risk`, with frozen
     `gamma_Regular=8`, `gamma_Urgent=12`, then queue depth and Replica ID.
   The resulting per-state predicted shares are materialized as the first
   `PolicyState` and committed with the forward output.  No caller may provide a
   scientific start policy.
3. After K passes confirmation, its policy is converted to token-class/action-
   bucket marginals weighted by the confirmed snapshot occupancy.  The next K
   uses these marginals, restricted and renormalized to its currently available
   action buckets, as its checkpointed initial routing policy.  If no prior
   action bucket is supported, that observation falls back to uniform over UP
   Replicas and the fallback count is reported.
4. The K=32/K=64 policy distance is the maximum token-class L1 distance between
   the same occupancy-weighted action-bucket marginals.  It must be at most
   `0.05`.
5. For every finite-K state/action row, normalized gain UCB is
   `(gain_mean + critical_value * paired_SE) / mean_baseline_private_cost`.
   The denominator must be finite and strictly positive.  The reported scale
   statistic is the maximum row value; K=64 must be at most `0.02`, and the
   sequence must be non-increasing from K=8 through K=64.  Missing support or an
   invalid denominator is `statistics_insufficient`.
6. Before each reservation, the parent verifies that existing reserved calls
   plus the new reservation do not exceed `qualification_calls=51,072`.
   Holdout retains its separate 8,064-call ceiling and remains unreachable until
   all qualification gates pass.

## Consequences

- All formal initialization and scale handoff become reproducible checkpointed
  state rather than caller authority.
- The scale comparison is defined on action-bucket marginals shared across K,
  avoiding invalid direct comparison of topology-specific state hashes.
- The proposal does not alter r2 namespaces, macro seeds, samples, physics,
  prices, response temperature, damping, or residual thresholds.
- r2 cannot start until this ADR is explicitly accepted and correction ticket
  13 is implemented and verified.
