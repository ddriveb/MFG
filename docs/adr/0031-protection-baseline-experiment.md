# ADR-0031: Protection-baseline experiment protocol

Status: Accepted (user authorized the experiment on 2026-09-08)

## Context

The existing three-arm result compares No-Hedge, NIIN, and the requested-price
candidate. Credible request-hedging baselines require P95 delayed hedging and
LÆDGE, but LÆDGE cannot be represented as a one-shot N/D/I action because it
launches work whenever a replica becomes idle.

## Decision

1. Run one paired protection panel on the existing one-Expert/two-Replica
   `theta_token_load0p7_v1` topology and the exact 1,024 evaluation episode
   identities used by the three-arm holdout.
2. Compare Failover-only, P95 delayed hedging, generalized LÆDGE, exact NIIN,
   and the sealed requested-price candidate. Preserve the fixed Gate.
3. Calibrate the P95 timer from No-Hedge end-to-end latency of Healthy-arrival
   Tokens on 256 independent episodes under a disjoint namespace; never mix
   D/F tail observations into the timer or read evaluation outcomes.
4. P95 and LÆDGE retain their native unbudgeted launch semantics. NIIN and the
   requested-price arm retain the existing Reservation projector. Resource
   fairness is assessed explicitly with total/wasted work and storm metrics,
   not by silently changing an algorithm.
5. LÆDGE owns an isolated work-conserving two-Replica executor. At arrival it
   uses up to two idle live replicas; on release it serves the oldest unserved
   request before hedging the oldest singly served request. Winner completion
   cancels its other running copy immediately, as in the LÆDGE kernel. Common
   failure remains fault-first; a Token with no surviving copy is replayed on
   the surviving Replica. Arrival, service, Hedge, Replay, and fault CRN are
   identical across arms.
6. Report pooled arm metrics and paired episode-cluster bootstrap differences
   against NIIN. This is a finite-system baseline experiment, not a BR, Nash,
   or MFG claim.

## Consequences

The experiment can rank protection behavior on one shared topology while
showing its resource cost. Routing-after-placement remains a separate panel
because it needs batched multi-Expert/GPU physics.

## Amendment: 2026-09-08

The first r1 execution interpreted the calibration cohort as all phases and
produced a delay of 35.02. Post-run audit found that this makes the comparator
depend rather than a healthy historical P95 baseline. Clause 3 now explicitly
freezes Healthy-arrival observations; r1 is retained but superseded by r2 for
the P95 comparison.
