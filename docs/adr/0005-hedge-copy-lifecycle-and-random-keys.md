# ADR-0005: Hedge Copy Lifecycle and Random Keys

Status: Accepted (confirmed by the user on 2026-09-04)

Date: 2026-09-04

Builds on: ADR-0002, ADR-0003, ADR-0004

## Context

The paired-comparison feature adds Hedge Backup copies to the failure-capable engine. Without fixed lifecycle and randomness rules, Hedge copies would be improvised: ambiguous completion semantics, order-dependent random streams, and silent no-ops would destroy comparability between the No-Hedge and MFG-Hedge arms and break common random numbers.

## Decision

1. Copy kinds and attempt IDs: Primary = 0, Replay = 1, Hedge Backup = 2. Hedge draws live in an optional `WorkloadTrace.hedge_service_times[replica][token_id]` field generated from stream labels `service:{replica}:attempt2`; adding them never perturbs arrival, class, attempt-0, or attempt-1 streams, and No-Hedge paths never read them. Existing artifacts remain valid.
2. Delayed Hedge starts its timer at Token arrival (`arrival_time + tau0`); Immediate Hedge creates both copies at arrival, Primary enqueued before Hedge. The same-time order is fixed: `DOMAIN_STATE_CHANGE` (with its failure/replay substeps) → `COMPLETE` → `HEDGE_TIMER` → `TOKEN_ARRIVAL` → dispatch/start, so a completion voids a pending timer and a failure creates the Replay and voids the timer before it can fire.
3. Conservative cancellation is the v1 semantics: the first valid completion wins; a queued loser is cancelled with zero executed work; a running loser completes as `completed_loser`, its result is discarded, and its work counts as consumed. Ideal (preemptive) cancellation is deferred to a separate ablation. Loser work enters `total_executed_work`; the winner alone determines the Token's terminal outcome. Attempt statuses are `completed_winner`, `completed_loser`, `cancelled_queued`, `failed_running`, `invalidated_queued`.
4. Hedge–Replay interaction: a Primary lost with no surviving copy gets one Replay; a Primary lost with a Hedge Backup queued/running gets none; a Primary lost before the timer fires gets a Replay and the timer is voided (never both for the same loss); a Hedge Backup on the later-Failed domain is lost without Replay; a Token with a winner never gains further copies. Per Token: at most one Replay, at most one Hedge Backup, exactly one winner and one completion.
5. If the Backup domain is Failed at decision time, the action is infeasible: the policy/quota layer restricts the feasible set to `{Normal}` and the executor never launches-and-pretends; suppressions are counted.

## Consequences

- The hedge-capable engine in No-Hedge mode must regress Token-for-Token against the accepted Common State + No Hedge engine.
- Every copy's service requirement is bound by stable key `(token_id, replica_id, attempt_id)`; no consumption-order randomness exists anywhere.
- Rejecting any point requires superseding this ADR before tickets 01-04 are implemented.
