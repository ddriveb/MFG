# Scaled multi-Expert, multi-Backup development experiment

Status: frozen by Accepted ADR-0014.

## Fixed dimensions

- 8 Experts, uniform immutable Gate assignment.
- 3 Replicas/Expert: fixed Primary A plus Backup B/C.
- 128 independent episodes; aggregate rate 3.6; expected ~147,456 Tokens.
- H[0,100), D[100,200), F[200,220), R[220,320), then drain.
- Per-Expert offered load .45; slowdown 2; mean service 1, CV .5.
- Same class ratio .8/.2, delay quantile .9 and objective as ADR-0012.

## Arms

1. `no_hedge`: NNNN, zero Backup copies.
2. `frozen_single_backup_NIIN`: the one-Expert development winner, frozen before
   this trace is generated; one Backup per admitted request.
3. `dual_backup_81_search`: all 81 rules; two Backups per admitted request; choose
   sample-best feasible with ADR-0013 filters and tie-break.

The shared per-Expert reserved-work cap makes backup-count arms comparable in
prospective budget, not necessarily equal in realized work.

## Required output

Every arm reports J components, D/F fractional CVaR95, phase/class latency and
Replay, total/wasted work, H/R safety ratios, request/admit/suppress counts,
Hedge launches/winners and invariants. The artifact includes every dual rule and
trace fingerprint. Any incomplete Token, duplicate attempt, trace mismatch,
invalid stream, missing rule, or nonfinite statistic aborts before commit.
