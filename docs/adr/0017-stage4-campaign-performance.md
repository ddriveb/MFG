# ADR-0017: Stage 4 campaign execution performance boundary

Status: Accepted (confirmed by the user on 2026-09-06)

Date: 2026-09-05

## Context

The frozen Stage 4 solver is numerically bounded but a real N=8 nested trace
contains substantially more Tokens than the unit timing smoke. A single
reference scheduler call currently takes about 0.5 seconds, so the frozen
921,088-call upper bound is not an acceptable single-process campaign plan.
The execution layer must improve throughput without changing the finite
physics, policy observation boundary, sample split, rule bank, or accounting.

Implementation note: this acceptance authorizes performance-layer work only;
it does not authorize Fit, Validation, or any formal campaign execution.

## Decision

1. Keep the current full-result simulator as the reference path. Add an
   immutable `PreparedTrace` compiled once per trace. It owns only validated,
   deterministic input metadata and cached fingerprints/precomputed event
   projections; it must not cache policy actions, queue state, speed,
   Replay, completion, budget, or any endogenous result.
2. Add an exact tagged-evaluation mode. Every Expert and every copy still
   participates in the same event loop and shared pool, but the returned
   scorer payload retains only the tagged Expert's Token/attempt/action data
   plus the invariant values required by its scorer. Full mode remains the
   reference default. Tagged output must score byte-for-byte equal to the
   corresponding Expert projection of a full reference result.
3. Campaign work is grouped by `(deviating_expert, candidate_rule)`. One
   worker processes that pair's complete fixed scenario list in deterministic
   order, reducing Windows process IPC. Workers never choose rules, merge
   rows, or own the global call counter.
4. The parent process reserves and settles the fixed scheduler-call quota for
   each dispatched task. A failed task is not automatically retried; the
   campaign records a partial failure and stops before claiming completeness.
   This prevents retries from duplicating or losing call-count accounting.
5. The parent merges returned rows in fixed order: Expert ID, canonical bank
   order `N<D<S<X>`, then scenario key. Artifact writes occur only once,
   transactionally, into a fresh non-overwriting run directory after the
   complete result is assembled.
6. Initial parallel execution is capped at 8 worker processes and 8
   in-flight tasks, leaving logical CPU headroom on the 12-processor
   development host. The campaign must record worker count, task granularity,
   memory cap, failure policy, and deterministic merge policy. A worker may
   use at most 512 MiB resident memory; the parent must not intentionally
   exceed a 4 GiB campaign working-set cap.
7. Before campaign launch, rerun real N=8 timing on NNNN/NSSN/XXXX and freeze
   the observed worst-case per-call runtime and projected wall-clock gate in
   the campaign ticket. Do not start fit or validation if the gate is not
   met; changing the gate requires a new decision and does not authorize
   changing samples or call budget.

## Invariance gate

For hand traces covering N/D/S/X, Replay burst, failure-boundary tie,
same-time completion tie, and running loser behavior, optimized and reference
outputs must match field-by-field for Token results, attempt statuses and
work, action/budget ledgers, queue audits, pool intervals, invariant counters,
fingerprints, and scorer values. A tagged result must match the tagged
Expert's fields from the full result. Historical engines and configurations
must remain unchanged.

## Consequences

Prepared inputs and tagged result construction can reduce repeated overhead,
while the physics state remains per-call and endogenous. Parallelism improves
throughput but adds worker lifecycle, memory, and failure boundaries; no
parallel result may be treated as complete unless the parent settles every
declared scheduler call. This ADR does not authorize a campaign, a solver
claim, a Nash/MFG claim, or an adaptive sample extension.
