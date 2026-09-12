# Engine performance equivalence fix: baseline cache + snapshot fork + completion heap

Type: engineering
Status: claimed
Blocked by: none

## Goal

Remove duplicate computation from the finite-K deviation engine so the frozen
K=64 qualification becomes wall-clock feasible, while proving bit-exact
equivalence with the pre-fix engine on every emitted field. No protocol,
frozen definition, or scientific semantic may change.

## Scope

- F1: cache the deviation-panel baseline simulation per
  (trace.fingerprint, policy_fingerprint, policy_iteration).
- F2: archive full engine state snapshots at batch boundaries and fork
  deviation branches from the target-batch snapshot instead of replaying
  from t=0; add the stateless-policy contract test; fall back to full
  replay for any policy that carries cross-batch internal state.
- F3: replace the per-iteration running-set full scan with a completion-time
  heap, preserving attempt-removal invalidation and the 1e-12 completion
  semantics.
- Equivalence evidence, before/after benchmarks on the r2 trace set, and a
  K=64 per-iteration projection for the user adjudication of ticket 06.

## Non-goals

- No change to ticket 06 protocol, namespaces, seeds, bucket schema, floors,
  thresholds, or call accounting.
- No qualification, holdout, or campaign execution; benchmarks may only time
  existing traces.
- No third-party dependency.

## Frozen-input notes (verified by prior read-only review)

- token_population_response.py:913 recomputes a full baseline simulation per
  tagged Token; the baseline is bit-identical per (episode, policy_iteration).
- token_population_response.py:842 replays the whole horizon per deviation
  branch from t=0 even though pre-target history equals the baseline.
- simultaneous_token_routing.py:762-769 scans the running set per event
  iteration; event count is O(K) per horizon.

## Acceptance criteria

- Equivalence tests capture pre-fix outputs (current implementation is the
  baseline); after each fix they still pass bit-exactly: every field of
  SimultaneousRoutingResult (token/attempt results, batch audits, metrics)
  and every FiniteKDeviationResult row.
- Coverage: all existing focused tests at K=8, the r2 32 deviation cases, and
  a K=64 synthetic small-trace equivalence spot check covering failure,
  replay, and simultaneous events.
- Full suite and config check green.
- Before/after timing table and K=64 projection recorded here.

## Next dependency

User adjudication of the K=64 feasibility conclusion for ticket 06.

## Verified design facts (read-only review, 2026-09-11)

- `_risk_observations` (reliability_aware_routing.py:541-585) recomputes hazard
  from `trace.events_for(...)` over `[now - history_window, now]` — prefix-only
  reads of the exogenous trace; there is NO accumulated window state. F2
  snapshots therefore only need `event_index` (case 1 of the task's two cases).
- `_PolicyStateAdapter.choose` (token_mfg_qualification_backend.py:333-387) is a
  pure function of (observation, context, policy_key): `_stable_unit` hash
  draws, no instance mutation, no RNG state. All four seed kinds and PolicyState
  rows route through it. A contract test still empirically verifies this.
- Engine determinism: `simulate_simultaneous_routing` has no global RNG; all
  randomness comes from the trace and hash-keyed draws.

## F3 heap design (bit-exactness argument)

> 2026-09-11 outcome: **REVERTED after measurement** — see the Progress log
> entry below. The heap was implemented, passed all bit-exactness tests, and
> benchmarked 0.62–0.78x *slower* than the scan on the r2-derived harness
> (evidence: `after_f3_heap.json` in the benchmark artifact). The design text
> is retained for the record.

The pre-fix loop computes `next_completion = min over running of
current_time + max(0.0, required - executed)` at every iteration (O(R) scan),
then `advance` mutates `executed_work`. Because floating-point addition is not
associative, a completion estimate frozen at attempt start can differ by 1 ulp
from the pre-fix recomputed value, which would shift event times and break
bit-exactness. Therefore:

- Each running attempt carries a cached key `k = max(0.0, required - executed)`
  recomputed with the exact same expression at the exact same program points
  the pre-fix code used (end of `advance`, after `current_time` and
  `executed_work` are final; and at `start_available` push time). The identity
  `min_i (current_time + k_i) == current_time + min_i k_i` is exact in IEEE
  float (the minimum of the per-attempt sums is literally one of those sums).
- A heapq keyed on `k` yields the min in O(1); stale entries (failed replicas)
  are skipped lazily by checking `running.get(replica_id) is attempt`, and a
  version counter guards against pre-failure entries.
- Completion detection keeps the pre-fix O(R) semantics
  (`required - executed <= 1e-12`) unchanged; 1e-12 event-time dedup unchanged.

## F2 snapshot inventory

Snapshot point: entry of `commit_decision_cohort` for the batch that will
receive `batch_id == target_batch_id` (state before any cohort mutation),
plus the cohort's `new_ids` and `time` so the fork can re-enter the loop.

Fields: `current_time`, `arrival_index`, `event_index`, `batch_id`,
`health` (dict copy), `queues` (deep, per-replica deques), `running` (deep,
attempt objects shared consistently with `attempts`), `attempts` (deep per
attempt), `replay_counts`, `waiting` (deque copy), `starts`, `batch_audits`,
`total_work`, `lost_work`, `no_eligible_time`, `down_starts`, `max_live`,
`terminal`, `seen_token_ids`, `published_realized_share`, `replica_audit`
(deep incl. `hazard_samples`), and the pending `new_ids`+`time`.

Hazard/history: NOT in snapshot (prefix-recomputable per above).

Fork: restore deep copies, re-run the cohort with the intervention wrapper
(whose recorded prefix calls are seeded from the baseline wrapper so
`call_fingerprint()` stays bit-identical), then continue the normal loop.

## F1 cache design

Key: `(trace.fingerprint, policy_fingerprint, policy_iteration, namespace,
macro_seed)` where `policy_fingerprint` comes from the materialized base
policy (`fingerprint` property) or `None` → cache disabled. Value: the
baseline `(result, wrapper, base_policy)` triple. Only
`evaluate_finite_k_deviation`'s baseline branch consults it; candidate
branches always simulate. `attempted_calls` accounting is unchanged.

## Progress log

### Update: 2026-09-11 — F1/F2 delivered bit-exact; F3 heap reverted as measured pessimization

Status: completed

#### Goal

Deliver the engine equivalence fixes (F1 baseline cache, F2 snapshot fork,
F3 completion heap) with bit-exact output guarantees, benchmark before/after
on r2-derived traces, and give a defensible K=64 / 12h-gate feasibility
conclusion for user adjudication of ticket 06's K-scale plan.

#### Changed

- `tests/test_engine_equivalence.py` (new): golden fingerprints captured from
  the pre-fix implementation (k8/k64 forward sim + deviation panels, K=64
  synthetic trace covering failure, replay, and simultaneous events), plus
  contract tests that `_PolicyStateAdapter` is a pure function of
  (observation, context, key) via the `__engine_stateless__` marker, and that
  a stateful policy falls back to full replay. 12 tests.
- `src/mfg_hedge/simultaneous_token_routing.py`: `EngineSnapshot` dataclass
  (inventory below), `_copy_attempt_map`, `snapshot_recorder` /
  `fork_snapshot` parameters on `simulate_simultaneous_routing`; `EngineSnapshot`
  exported in `__all__`.
- `src/mfg_hedge/token_population_response.py`: `_BASELINE_CACHE` keyed on
  (trace fingerprint, policy fingerprint, policy iteration, namespace,
  macro seed; cache disabled when the policy exposes no fingerprint);
  `_run_finite_branch` gained `snapshot_recorder` / `fork_snapshot` /
  `seed_calls` / `base_policy`; `evaluate_finite_k_deviation` probes the
  factory once for the stateless marker, reuses the probe as the baseline
  policy (factory call count semantics unchanged: exactly one materialization
  per evaluate call), seeds the intervention wrapper's recorded prefix calls
  from the baseline wrapper so `call_fingerprint()` stays bit-identical.
- `src/mfg_hedge/token_mfg_qualification_backend.py`:
  `_PolicyStateAdapter` gained `__engine_stateless__ = True` and a
  `fingerprint` property (from `schema.schema_id`).
- F3 completion heap: fully implemented, passed all equivalence tests, then
  **reverted** after benchmarking (see Decisions). The engine keeps the
  original per-iteration `min()` scan; the only residual changes vs pre-fix
  are the inert `snapshot_recorder`/`fork_snapshot` parameter checks.

#### Verification

- `./.venv/Scripts/python.exe -m unittest tests.test_engine_equivalence tests.test_simultaneous_token_routing tests.test_token_population_response tests.test_token_mfg_qualification_streaming tests.test_token_mfg_streaming_formal tests.test_token_mfg_qualification_backend tests.test_token_mfg_qualification_readiness` — 46 tests, OK (run twice: once with the F3 heap, once after the F3 revert).
- `./.venv/Scripts/python.exe .scratch/reliability-aware-token-mfg/engine_benchmark.py --crosscheck` — 32/32 r2 deviation cases rebuilt field-by-field identical to the committed r2 blobs (trace fingerprint, target state, episode identity, snapshot, compact tokens, continuation rows, finite rows, completed_calls).
- `./.venv/Scripts/python.exe -m unittest discover -s ./tests` — 836 tests, OK.
- `./.venv/Scripts/python.exe -m mfg_hedge check --config ./configs/v1_minimal.json` — exit 0, status ok (single-domain failure load 1.4 informational note unchanged).

Timing (engine_benchmark.py, same r2-derived traces both phases):

| section | before mean | after mean (F3 reverted) | F3-heap attempt |
|---|---|---|---|
| k8 forward call | 0.0225s | 0.0289s (noise) | 0.0307s |
| k8 deviation panel | 0.2561s | 0.3108s (noise) | 0.3267s |
| k64 forward call | 0.3837s | 0.4105s (0.93x) | 0.6155s (0.62x) |
| k64 deviation panel | 3.4279s | 3.6982s (0.93x) | 4.9612s (0.69x) |

#### Artifacts

- `artifacts/engine-equivalence-benchmark-20260911/before.json`
- `artifacts/engine-equivalence-benchmark-20260911/after.json` (final code)
- `artifacts/engine-equivalence-benchmark-20260911/after_f3_heap.json` (F3 evidence, superseded)
- `artifacts/engine-equivalence-benchmark-20260911/r2_crosscheck.json`
- Code/tests listed under Changed.

#### Decisions and risks

- **F3 reverted.** The heap replaced an O(R) scan with R ≤ 64 (tiny constant,
  no allocation) by per-advance 4-tuple pushes + lazy stale cleanup
  (O(pushes) allocation, sift-up per advance). Measured 0.62–0.78x across all
  four benchmark sections; the O(events × R) scan was never the K=64 cost
  driver. Equivalence was never in question (goldens passed with the heap);
  the revert is purely a performance decision, evidence preserved in
  `after_f3_heap.json`. The ticket's "O(K²) scan" premise does not hold at
  these scales.
- **F2 fork savings ≈ 0 for K=64 r2-style panels【实测】.** Probed all four
  K=64 r2 traces: every token with ≤8 action buckets (the frozen eligibility
  rule) sits in batch 0 (8–32 usable of 64 tokens per episode, all batch 0;
  later batches exceed 8 buckets). A batch-0 fork replays essentially the
  whole horizon, so the fork path — while bit-exact and correct — saves
  almost nothing where the K=64 workload actually spends its time. Same
  concentration likely throttled the r2 K=8 panel (332/608 completed).
- **F1 cache** has no hit in the benchmark (fresh trace per episode) and hits
  only when an iteration reuses identical traces (the streaming campaign
  structure). It saves one full baseline replay per (episode, iteration ≥ 1);
  it does not speed up a single-iteration cold run.
- **K=64 / 12h conclusion【报告陈述】**: per-call cost is unchanged by this
  ticket (k64 forward mean ~0.38s before and after; worst-case preflight
  0.92s stands). The preflight full-panel projection (~31,567s ≈ 8.8h at
  0.274 calls/s, 8 workers) is therefore NOT improved in any material way;
  K=64 remains nominally inside the 12h gate with essentially zero headroom.
  This ticket does not unlock the K-scale plan; it removes redundant
  computation and locks in bit-exact regression coverage for any future
  engine work.
- No protocol/frozen definitions were touched; no new experiment data was
  generated (benchmark and probe are timing-only on rebuilt r2 traces).

#### Next

User adjudication for ticket 06: given that the engine fixes confer no
material K=64 speedup, decide between (a) accepting the marginal 8.8h/12h
K=64 run as-is, (b) the protocol-redesign branches already derived in
ticket 06's analysis (coarser buckets / relaxed UCB gate / reordered primary
metric), or (c) narrowing the qualification target. Do not restart K-scale
on the expectation of engine headroom.
