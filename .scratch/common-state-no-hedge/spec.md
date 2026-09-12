# Deterministic Common State + No Hedge Failure Simulation

## Goal

Extend the verified Healthy + No Hedge foundation with one deterministic failure process on failure domain A, while every Token still uses the Normal protection action. The slice validates piecewise work integration under degradation, failure-driven Replay (at most one per Token), recovery, an explicit event engine, fault-aware metrics, and reproducible artifacts.

This spec is the single functional authority for the slice. Every design question below is decided here; nothing is left for implementers to improvise.

## Research boundary (fixed)

- One Logical Expert, two Replicas: Replica 0 in failure domain A, Replica 1 in failure domain B.
- Only domain A degrades or fails. Replica 1 always runs at speed 1.0.
- Gate output is fixed; protection never changes the Logical Expert.
- Every Protection Action is Normal. No Delayed Hedge, no Immediate Hedge, no proactive Backup, no MFG, no prices, no quota projection.
- A Primary lost to failure allows at most one Replay.
- New Tokens never start on a Replica whose domain is confirmed Failed.
- No third-party dependencies.
- The accepted Healthy + No Hedge behavior (tickets under `.scratch/healthy-no-hedge/`) must not change, under the stability contract below.

### Healthy stability contract

- Healthy dispatch, `TokenResult`s, metric definitions, the summary schema, and all scientific values remain unchanged.
- Under the same `simulator_version`, configuration, and seed, two Healthy runs are byte-identical except `run_id`.
- Across `simulator_version`s, provenance metadata such as `simulator_version` may change.
- Version 0.3.0 adds capability and therefore changes the `simulator_version` field of future Healthy summaries, but not their scientific content.
- Existing artifacts are never rewritten.

## 1. Common State timeline and Phase

### Definitions

The Common State describes the condition of failure domain A (the only domain that can degrade or fail):

- `H` = Healthy
- `D` = Degraded
- `F` = Failed
- `R` = Recovered — in engineering terms a re-entry into `H`, not a new enum member

`CommonState` keeps its three members (`H`, `D`, `F`); recovery is modeled as the timeline returning to `H`.

**State and Phase are separate concepts.** `state_at(timeline, t)` returns the engineering `CommonState` (`H`, `D`, `F`); after `recovered_start` it returns `H` again. `phase_at(timeline, t)` returns the metrics Phase `H`, `D`, `F`, or `R`, distinguishing the initial Healthy window from the post-recovery window. Both are pure functions (ticket 01). It is never sufficient to use `state_at` alone where initial-H and R must be told apart.

### Half-open intervals

The timeline is three ordered instants `degraded_start < failed_start < recovered_start`, producing half-open intervals:

```text
[0, degraded_start)            -> H
[degraded_start, failed_start) -> D
[failed_start, recovered_start)-> F
[recovered_start, +inf)        -> state H, phase R
```

A boundary instant belongs to the interval it opens (e.g. `state_at(failed_start) == F`, `phase_at(recovered_start) == R`).

### Timeline values and per-Replica feasibility

Time is normalized service time; no milliseconds or GPU-measured units. With the default arrival rate `lambda = 1.4` Tokens per unit and fixed parity dispatch, each active Replica receives `lambda / 2 = 0.7` while both are alive, and Replica 1 receives the full `lambda = 1.4` while A is Failed.

Queues are isolated per Replica, so feasibility is judged **per Replica**, not by aggregate speed:

| Phase | Interval | Replica 0 arrival/speed/ratio | Replica 1 arrival/speed/ratio | Aggregate ratio (descriptive only) |
|-------|----------|-------------------------------|-------------------------------|------------------------------------|
| H | [0, 100)    | 0.7 / 1.0 / 0.7 | 0.7 / 1.0 / 0.7 | 0.7 |
| D | [100, 200)  | 0.7 / 0.5 / **1.4** | 0.7 / 1.0 / 0.7 | ~0.933 |
| F | [200, 220)  | inactive (N/A)  | 1.4 / 1.0 / **1.4** | 1.4 |
| R | [220, +inf) | 0.7 / 1.0 / 0.7 | 0.7 / 1.0 / 0.7 | 0.7 |

`capacity_violation(window)` is true iff **any** traffic-carrying Replica has ratio > 1 in that window. With the defaults: H = false, D = **true** (parity keeps sending 0.7 to the 0.5-speed Replica 0), F = **true** (Replica 1 absorbs all traffic at ratio 1.4), R = false. The aggregate ratio may be reported as descriptive context only; it must never drive the violation flag.

Recommended defaults: `degraded_start = 100.0`, `failed_start = 200.0`, `recovered_start = 220.0`.

Rationale, not mechanical numbers:

- H and D windows target >= ~100 Tokens each so per-phase percentiles have minimally meaningful samples.
- The F window is deliberately short: Replica 1 runs at ratio 1.4, so its queue grows about 0.4 Tokens per unit plus the Replay burst injected at `failed_start`. Twenty units yields ~28 fresh arrivals plus the burst, enough to observe failure-window behavior while keeping post-recovery drain bounded.
- These values are versioned configuration, not code constants (see Configuration below), and are **mechanism-verification defaults**, not final experimental windows (see Statistical interpretation in Section 7).

### Two trace levels

1. **Manual trace**: tiny hand-built `WorkloadTrace` plus a hand-set timeline (e.g. D at 10, F at 20, R at 30) used only in unit tests to verify state boundaries and event order with exact expected values.
2. **Controlled trace**: produced by the workload generator with the configured timeline. A controlled run is only valid if the last arrival exceeds `recovered_start`; the CLI must fail fast otherwise. At lambda = 1.4, `--tokens >= 400` suffices in expectation (400/1.4 ~= 286 > 220); the documented controlled run uses 1000.

### Configuration

A new versioned file `configs/v1_common_state.json` (its own `schema_version`) carries every `ExperimentConfig` key plus `degraded_start`, `failed_start`, `recovered_start`. `configs/v1_minimal.json` and `ExperimentConfig` are not modified; the Healthy slice is untouched. Timeline validation: `0 <= degraded_start < failed_start < recovered_start`.

## 2. Service requirement and degradation semantics

A service draw is a **normalized work requirement** `W` (work units, configured arithmetic mean 1.0), not a wall-clock duration.

Processing speed (work units per normalized time unit):

- `speed(H) = 1.0` for both Replicas.
- `speed(D) = 1 / degraded_slowdown` on Replica 0 (default `1/2 = 0.5`); Replica 1 stays 1.0.
- `speed(F) = 0` on Replica 0 conceptually: uncompleted work is lost immediately (Section 4).
- Replica 1 is always 1.0 regardless of state.

Executions integrate work piecewise against the state timeline:

- An execution running when a state boundary is crossed keeps its **executed work**; its **remaining work** `W - executed` continues at the new speed; its completion time is recomputed as `now + remaining / speed(new)`.
- Pure functions required (ticket 01): `state_at`, `phase_at`, `speed_at(replica, state)`, `work_executed_between(replica, t0, t1, timeline)`, and `completion_after(replica, start, work, timeline)` returning either the completion instant or a failure outcome.
- Example: Token starts on Replica 0 at t=9 with W=2 and D begins at t=10. It banks 1.0 work unit in [9,10), retains remaining 1.0, and completes at `10 + 1.0/0.5 = 12`.
- Multiplying the whole service time by `degraded_slowdown` once at start is forbidden.

## 3. Same-time event order

Events at the same `event_time` are processed in this fixed priority, with `deterministic_sequence` (a global monotonic insertion counter) as the final tie-breaker:

1. `DOMAIN_STATE_CHANGE`
2. Invalidate in-flight and queued domain-A executions (sub-step of a state change into F)
3. Create and enqueue Replays on Replica 1 (sub-step, same trigger; ordering fixed in Section 4)
4. Effective `COMPLETE`
5. `TOKEN_ARRIVAL`
6. `DISPATCH / START` (dispatch pass after the event batch at that instant)

Rationale:

- Fault-first at the F boundary: a completion scheduled exactly at `failed_start` is ranked after the state change, so that execution fails. Coinciding with a failure must not count as a lucky success.
- Recovery-first at the R boundary: an arrival exactly at `recovered_start` sees Healthy and may be dispatched to Replica 0.
- Replay-before-arrival: Replays are older Tokens; enqueuing them before same-time new arrivals preserves arrival-order fairness on Replica 1.

This order is adopted as fixed; any future change requires a spec amendment with stated reasons.

## 4. Failure and Replay semantics

When domain A enters Failed (at `failed_start`):

- The execution currently running on Replica 0 fails immediately; work it already consumed is recorded as **wasted work**.
- Executions queued on Replica 0 but never started are invalidated immediately (zero consumed work).
- Tokens already completed are unaffected.
- Every lost Token with `replay_count == 0` gets exactly one Replay on Replica 1, enqueued at `failed_start` (event-order step 3). `replay_count` becomes 1.
- **Replay order at the failure instant is fixed**: take Replica 0's currently running Token first, then Replica 0's queued Tokens in their original FCFS order; enqueue the Replays on Replica 1 in exactly this order. New Token arrivals at the same instant join Replica 1's queue after all of these Replays.
- Replay is not a Hedge: it is replacement work after a lost copy, reported separately. Hedge counters stay 0.
- The Replay uses the same Logical Expert and a fresh independent service draw keyed `(token_id, replica_id = 1, attempt_id = 1)`.
- A second Replay is never allowed; in this slice Replica 1 cannot fail, so every Replay completes.

While the state is F:

- The Dispatcher starts every new Token's Primary on Replica 1; no Token is first assigned to known-Failed Replica 0.
- These new Tokens are ordinary Primaries (attempt 0 on Replica 1), not Replays.

At `recovered_start`:

- Replica 0 returns with an empty queue and nothing running. Lost executions do not resurrect.
- New Tokens are again dispatched between both healthy Replicas.

### Deterministic dispatch rule (no cursor)

Dispatch is a pure function with no mutable cursor:

```text
assign(token_id, state) = 1                if state == F
assign(token_id, state) = token_id % 2     otherwise (H, D, and Recovered H)
```

Consequently "the round-robin cursor after recovery" is defined by construction: parity resumes from `token_id % 2`; there is nothing to reset. Implementers must not add cursor state.

## 5. Stable random keys

- Primary draws keep the key `(token_id, replica_id, attempt_id = 0)`; existing sampling results do not change.
- Replay draws use `(token_id, replica_id, attempt_id = 1)`.
- Storage: `WorkloadTrace` gains one optional field `replay_service_times: tuple[tuple[float, ...], ...] | None = None`, shaped like `service_times` and indexed `[replica][token_id]`. The field defaults to `None`, so all existing constructors, tests, and the Healthy simulator remain valid and byte-identical; the Healthy path never reads it.
- Generation: a dedicated generator for this scenario (ticket 01) draws attempt-1 streams from a new stream label distinct from the attempt-0 label `service:{replica}` (e.g. `service:{replica}:attempt1`). Adding Replay sampling must not perturb arrival, Token-class, or attempt-0 draws: a regression test asserts a replay-augmented trace and a plain trace with the same config and seed are identical in arrivals, classes, and attempt-0 service values.
- Order-based consumption of random streams remains forbidden; all access is by stable key.
- Same config + seed must give identical per-Token-copy work requirements across policies.
- Memory: the extension adds `replica_count x token_count` floats (same again as attempt 0; ~1.6 MB per 100k Tokens in Python floats, more as objects) — acceptable for v1 and documented here.
- API compatibility/migration: additive optional field, no trace version field is introduced; acceptance tests for tickets 01-02 must construct both trace shapes.

## 6. Event engine

The static FCFS recursion in `simulate_healthy_no_hedge` is insufficient here because:

- slowdown changes the completion time of an already-running execution (retroactive rescheduling);
- failure kills running and queued executions mid-trace;
- Replay inserts new work into another Replica's queue at the failure instant;
- recovery re-enables a Replica;
- previously scheduled completion events can become stale.

The engine uses an explicit min-heap event queue (`heapq`, stdlib). Queued event types: `DOMAIN_STATE_CHANGE`, `COMPLETE`, `TOKEN_ARRIVAL`. Every event record carries:

```text
event_time              float
event_type              enum
deterministic_sequence  int, global monotonic insertion order
token_id                int | None (None for DOMAIN_STATE_CHANGE)
replica_id              int | None (None for TOKEN_ARRIVAL before dispatch)
attempt_id              int | None
generation              int, per-Replica scheduling generation
```

Heap ordering key: `(event_time, type_rank, deterministic_sequence)` with type ranks `DOMAIN_STATE_CHANGE = 1`, `COMPLETE = 4`, `TOKEN_ARRIVAL = 5`. Steps 2, 3, and 6 of the same-time order are ordered sub-steps inside handlers, not separate heap events.

Stale completions: each Replica owns a `generation` counter. Scheduling or rescheduling a completion on a Replica increments its generation and stamps the new event. Any state change affecting that Replica (slowdown, failure, invalidation) also increments it. A popped `COMPLETE` whose `generation` differs from the Replica's current generation is ignored. A second guard: a Token that already reached a terminal outcome ignores further events.

Termination and drain: the engine runs until every Token has exactly one terminal outcome; it does not stop at the last arrival. In this slice every Token eventually completes (Replay on Replica 1 always succeeds), so `completed_tokens == generated tokens` holds as an invariant. The engine must expose `last_arrival_time` (arrival of the final Token), `drain_end_time` (the last effective completion), and `drain_duration = drain_end_time - last_arrival_time` for reporting.

## 7. Metrics

All existing Healthy metrics remain for this scenario's summary. **Naming rule**: grouping uses the Phase at completion, and the field is named `completion_phase` everywhere (engine results and summary); the name `completion_state` is forbidden to prevent mixing the two concepts.

### Counters and rates

- `completed_tokens`: Tokens with exactly one effective completion. Invariant: `completed_tokens == generated_tokens` (this is an invariant of the slice, not a ratio denominator).
- `failed_primary_executions`: attempt-0 executions destroyed at `failed_start`, split into `failed_running_primary_executions` (started) and `invalidated_queued_primary_executions` (never started).
- `replayed_tokens`: unique Tokens with `replay_count == 1`.
- `replay_executions`: Replay attempts started. In this slice it always equals `replayed_tokens` — Replica 1 never fails and each Replay runs exactly once to completion — but it is counted separately so future slices where the two diverge stay honest.
- `replay_rate = replayed_tokens / generated_tokens`.
- Invariants: `hedge_launches == 0`; every Token completes at most once; `replay_count <= 1`.

### Work accounting

- `nominal_primary_work`: for each Token, the **full** attempt-0 work sample on the Replica its Primary was assigned to (regardless of execution outcome), summed over all Tokens. This is the nominal demand baseline.
- `wasted_work`: work units consumed by Primary executions that later failed (queued-invalidated executions contribute 0).
- `executed_primary_work`: total work units consumed by attempt-0 executions (successful + wasted).
- `executed_replay_work`: total work units consumed by attempt-1 executions.
- `total_executed_work = executed_primary_work + executed_replay_work`.
- `extra_execution_ratio = executed_replay_work / nominal_primary_work`.
- `execution_amplification = total_executed_work / nominal_primary_work`.
- Denominator rule: `generated_tokens >= 1` is enforced by workload validation and every lognormal draw is strictly positive, so `nominal_primary_work > 0` and `generated_tokens > 0` by construction; no zero-division path exists. If a future slice ever relaxes these guarantees, the ratios must report `null` instead of dividing by zero.

### Per-phase statistics

- `completions_by_phase`: completed Token counts grouped by `phase_at(completion_time)` — `H`, `D`, `F`, `R` reported separately (R is engineering-Healthy but a distinct metrics phase).
- `sample_count` per phase, alongside each phase's statistics.
- `latency_p50/p95/p99_by_phase`: completion-latency percentiles per phase, using the existing linear-interpolation definition; phases with `sample_count == 0` report `null`.

### Observed versus unobserved phases

A phase is **observed** iff its window has a positive-length intersection with `[0, drain_end_time]`.

- Observed phase: report `observed=true`, the observation bounds, per-Replica overload ratios, and `capacity_violation` normally.
- Unobserved phase: `observed=false`, `observation_start/end=null`, `capacity_violation=null`, `sample_count=0`, latency percentiles `null`. A phase that never happened must never be reported as "observed with no violation".
- An observed phase with zero completions (sample_count 0) still reports its capacity verdict.

### Token queue delay

A Token's cumulative queue delay sums over its attempts: a started attempt contributes `start_time - enqueue_time`; an invalidated queued attempt contributes `terminal_time - enqueue_time`. Identity:

```text
latency = cumulative queue delay + sum of started attempts' wall-clock running durations
```

With no failure and no Replay this degenerates to the Healthy `start_time - arrival_time`. `mean_queue_delay` uses this cumulative definition.

**Statistical interpretation**: with the default timeline, H and D phases hold roughly 140 samples each; a P99 computed from ~140 samples is a **diagnostic value, not a stable statistical inference**. Formal experiments must use longer windows or multiple seeds (`seed_count`); that decision is part of experiment design, not this slice.

### Per-Replica and window accounting

- Per Replica: `busy_wall_time` (sum of wall-clock busy intervals) and `normalized_work` (work units executed = integral of speed over busy intervals).
- `failed_window_queue_length`: queued (not running) counts per Replica at `failed_start` (after Replay enqueue) and at `recovered_start`.
- `base_overload_ratio` per Replica per window: assigned arrival rate / speed — H/R: 0.7/0.7; D: 1.4 (Replica 0) and 0.7 (Replica 1); F: N/A (Replica 0 inactive) and 1.4 (Replica 1). The aggregate ratio may accompany it as descriptive context only.
- `capacity_violation` per window: true iff any traffic-carrying Replica ratio > 1 — defaults D = true, F = true, H/R = false. This is the base-load infeasibility report required by CONTEXT invariant 7; Hedge-budget violation does not exist in this slice (Hedge work is identically 0) and the two must never be merged.

### Observation interval and throughput denominators

The summary records `last_arrival_time`, `drain_end_time`, and `drain_duration`. `throughput = generated_tokens / drain_end_time` and per-Replica `utilization = busy_wall_time / drain_end_time` are measured over the observation interval `[0, drain_end_time]` and must be labeled as such. This end-to-end throughput **includes the drain period** and must never be interpreted as fixed-window capacity. Any future windowed rate must name its window explicitly.

Deadline-miss metrics are explicitly excluded: no deadline configuration or semantics exist yet.

### Summary schema

The new scenario's `summary.json` uses `schema_version: 2` (a superset adding the fault metrics above plus an echoed timeline). The Healthy scenario keeps `schema_version: 1` byte-for-byte. `scenario` becomes `common_state_no_hedge`; `dispatcher` becomes `round_robin_failure_aware` (the pure function of Section 4); `policy` stays `no_hedge`. Configuration identity follows ADR-0003: single-read bytes, `resolved_config`, `config_sha256`.

## 8. Verification scenarios (mandatory manual deterministic tests)

1. A Token completing inside H is unaffected by later states.
2. A domain-A Token crossing H->D continues its remaining work at speed 0.5 (exact expected completion).
3. A domain-A Token completing before the F boundary is valid.
4. A domain-A Token completing exactly at the F boundary fails (event order).
5. A running domain-A execution fails at F and its Token Replays on B (exact times).
6. A queued domain-A execution is invalidated at F and its Token Replays on B.
7. A Token arriving during F goes straight to B and is not counted as a Replay.
8. A Token arriving after recovery can enter Replica 0 again.
9. The Replay consumes the `(token_id, 1, attempt_id = 1)` draw; attempt-0 samples are unchanged (compare against the plain trace with the same seed).
10. A stale completion event (superseded generation) is ignored.
11. Same seed repeated runs give identical results (engine level; ticket 03 re-verifies at CLI byte level).
12. Every Token completes at most once.
13. Every Token Replays at most once.
14. `hedge_launches == 0` always.
15. Healthy regression: all state-change boundaries are set later than the same trace's **last completion time under the Healthy baseline** (equivalently, a never-trigger fixture), and the event engine's `TokenResult`s are identical to `simulate_healthy_no_hedge`. Boundaries merely beyond the last arrival are **not** sufficient, because in-flight work would still cross them.
16. Replay order at `failed_start`: the running Token's Replay is enqueued first, then queued Tokens' Replays in original FCFS order, all ahead of same-time new arrivals on Replica 1.

## 9. Ticket split

- `01-common-state-and-work-requirements.md`: timeline type and validation, `state_at`, `phase_at`, speed model, piecewise work-integration pure functions, attempt-1 trace extension, pure-function tests. No event loop.
- `02-failure-replay-event-engine.md` (Blocked by: 01): explicit event queue, piecewise execution, failure, Replay, recovery, stale-event handling, manual-trace tests (scenarios 1-10, 12-16).
- `03-fault-metrics-cli-and-artifacts.md` (Blocked by: 02): metrics, CLI `simulate-common-state-no-hedge`, `configs/v1_common_state.json`, summary schema 2, controlled trace, artifacts, determinism verification.

## 10. Out of scope

Delayed/Immediate Hedge, proactive Backup, multiple Experts, MFG, prices, quota projection, deadline misses, plotting, third-party dependencies, and any change to the accepted Healthy + No Hedge behavior.

## 11. Confirmed decisions

Resolved by design review (ticket 04): timeline defaults confirmed as mechanism-verification values; `replay_rate` denominator is `generated_tokens`; engine runs to full termination with explicit drain reporting; `failed_primary_executions` is split into running vs queued sub-counters; capacity feasibility is per-Replica.

Confirmed by the user on 2026-09-03:

1. This scenario's `summary.json` uses `schema_version: 2`; the Healthy scenario keeps schema 1.
2. `configs/v1_common_state.json` is self-contained, duplicating the base configuration keys; no shared-configuration inheritance mechanism.
3. Formal experiments will use longer windows plus multiple seeds; the 100/200/220 timeline is a mechanism-verification default, and formal windows and seed counts are decided after a pilot.
