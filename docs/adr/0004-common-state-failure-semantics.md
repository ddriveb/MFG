# ADR-0004: Common-State Failure Semantics, Event Order, and Replay Keys

Status: Accepted (confirmed by the user on 2026-09-03)

Date: 2026-09-03

Builds on: ADR-0001, ADR-0002, ADR-0003

Revised after design review (`.scratch/common-state-no-hedge/issues/04-correct-design-review-findings.md`): per-Replica feasibility, State/Phase separation, Replay rate denominator, drain reporting, Replay ordering at the failure instant, and work-amplification definitions. The user confirmed this ADR on 2026-09-03.

## Context

The next slice adds a deterministic failure process on failure domain A while all protection actions remain Normal. The Healthy slice's static FCFS recursion cannot express in-flight slowdown, mid-run failure, Replay insertion, recovery, or stale completions. Without fixed decisions, implementers would have to improvise boundary semantics, same-time event order, dispatch after recovery, Replay randomness, and capacity accounting — each of which silently changes scientific results.

## Decision

1. The Common State timeline is three ordered instants producing half-open intervals `[0, d)`, `[d, f)`, `[f, r)`, `[r, +inf)` for H, D, F, and Recovered-as-H. A boundary instant belongs to the interval it opens. `CommonState` keeps only `H`/`D`/`F`; a separate pure `phase_at(t)` returns the metrics Phase `H`/`D`/`F`/`R`, and all grouping fields are named `completion_phase` (never `completion_state`).
2. A service draw is a normalized work requirement. Processing speed is 1.0 when Healthy and `1 / degraded_slowdown` for domain A while Degraded. Executions integrate work piecewise across state boundaries; work already executed is banked, and remaining work continues at the new speed. Entering F discards uncompleted work immediately and records consumed work as wasted.
3. Same-time events follow a fixed order: domain state change; invalidation of domain-A running and queued executions; Replay creation and enqueue; effective completions; Token arrivals; dispatch/start. A completion exactly at the failure boundary therefore fails; an arrival exactly at recovery sees Healthy. Replay order at the failure instant is fixed: the running Token first, then queued Tokens in original FCFS order, all enqueued on Replica 1 ahead of same-time arrivals.
4. Dispatch is a pure function `1 if state == F else token_id % 2` with no cursor state; recovery needs no reset rule. A lost Primary yields at most one Replay on the surviving Replica, consuming the independent stable draw `(token_id, replica_id, attempt_id = 1)` stored in an optional `WorkloadTrace.replay_service_times` field that leaves attempt-0 sampling untouched.
5. Capacity feasibility is judged per Replica because queues are per Replica: the system reports `capacity_violation` for a window iff any traffic-carrying Replica has arrival/speed ratio > 1 (defaults: D and F violate, H and R do not); aggregate ratios are descriptive context only.
6. Work amplification is measured against `nominal_primary_work` (the sum of full attempt-0 samples on each Token's assigned Primary Replica): `extra_execution_ratio = executed_replay_work / nominal_primary_work` and `execution_amplification = total_executed_work / nominal_primary_work`. `replay_rate = replayed_tokens / generated_tokens`, with `completed_tokens == generated_tokens` held as an invariant rather than a denominator.
7. Runs terminate only when every Token has a terminal outcome; summaries record `last_arrival_time`, `drain_end_time`, and `drain_duration`, and every rate names its observation interval. End-to-end throughput includes the drain period and is never read as fixed-window capacity.
8. The scenario's `summary.json` uses `schema_version: 2` (a superset adding fault metrics); the Healthy scenario keeps schema 1 unchanged.

## Consequences

- Event engines and metrics can be implemented and tested against exact hand-computed expectations.
- Policies compared later share identical per-Token-copy work requirements for both attempts.
- Degraded-window overload under fixed parity dispatch is reported honestly instead of being masked by aggregate capacity.
- If any point is rejected, this ADR must be superseded before implementation continues; implementation tickets 01-03 must not silently deviate.
