# ADR-0014: Scaled eight-Expert, dual-Backup development experiment

Status: Accepted (user confirmed the proposed 8 Expert / 3 Replica / ~150k Token scale on 2026-09-05)

## Context

The accepted v1 engine models one Logical Expert and two Active-Active Replicas.
The user requested a larger experiment in Token, Expert, and Backup dimensions.
Changing those values inside an old schema would silently change dispatch,
failure, replay, random-key, quota, and cancellation semantics. This ADR creates
an isolated run family and keeps every historical artifact and API immutable.

## Decision

1. Model eight Logical Experts. The Top-1 Gate result is an exogenous uniform
   `expert_id` from an independent stable stream and never changes after arrival.
   Expert queues are independent conditional on the deterministic common state.
2. Every Expert has three fixed Replicas in domains A/B/C: Replica 0 is the
   normal Primary; Replicas 1 and 2 are healthy Backups. Common D/F affects all
   Replica-0 queues simultaneously. H/D/R arrivals use Primary 0; F arrivals
   are dispatched deterministically to `1 + global_token_id mod 2`. Replay uses
   the same healthy-destination rule. Per-Expert offered load is 0.45, so the
   aggregate arrival rate is 3.6 and D Primary utilization is nominally 0.9.
3. `backup_count=1` launches one deterministic Backup copy on
   `1 + global_token_id mod 2`; `backup_count=2` launches both Backup copies.
   Attempt IDs are Primary=0, Replay=1, first Hedge=2, second Hedge=3. Attempt-3
   uses a new SHA-256-derived stream and cannot perturb attempts 0/1/2, arrivals,
   classes, or Expert assignment. First completion wins; queued losers cancel;
   running losers complete; failure/timer ordering remains ADR-0005. A live
   Hedge prevents Replay after Primary failure.
4. Reservation is isolated per Expert and per 25-time D window. Each Expert
   receives the same `0.25 * 0.45 * 25 = 2.8125` reserved-work cap. Charge is
   one mean-work unit per requested Backup copy: 1 for single and 2 for dual.
   Therefore single/dual arms share the same total prospective work envelope;
   dual protection is not granted a free second copy. FIFO, no refund, and
   applied-action physics follow ADR-0012.
5. Freeze 128 fit episodes in namespace `scaled-control:v1:fit`, macro-seed
   20260905, timeline 100/200/220, arrivals before 320, slowdown 2. Expected
   aggregate sample size is approximately 147,456 Tokens; the realized Poisson
   count is reported. Compare exact No-Hedge, the previously frozen NIIN rule
   with one Backup, and all 81 rules with two Backups. The two-Backup selected
   rule uses the same J/work/lexical ranking and H/R/work filters as ADR-0013.
6. Transactionally write a new complete artifact containing configuration,
   topology, stream contract, 128 trace fingerprints, No-Hedge, frozen
   single-Backup NIIN, all dual-Backup rule rows, and comparisons. Never overwrite.

## Claim boundary

This is a larger development experiment, not qualification or holdout. Expert
assignment is uniform and fixed-primary topology is a new controlled scenario,
not a claim about a production MoE router. Common failure correlation is maximal
across Experts. The selected dual rule is fit on the same data used to report it.
No result is called MFG, globally optimal, statistically significant, or robust
to unknown faults. The old one-Expert NIIN result remains a different topology.
