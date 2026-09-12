# Fault metrics, CLI, and artifacts

Type: task
Status: resolved
Blocked by: 02

## Goal

Deliver the user-facing surface of `../spec.md`: fault-aware metrics, the controlled trace, a new CLI command, summary schema 2, and reproducible artifacts.

## Scope

- `configs/v1_common_state.json`: every `ExperimentConfig` key plus `degraded_start = 100.0`, `failed_start = 200.0`, `recovered_start = 220.0`, with its own strict loader and validation (spec section 1).
- Metrics exactly as defined in spec section 7: per-phase (`completion_phase`, H/D/F/R) completion counts with `sample_count` and latency percentiles; `replayed_tokens` and `replay_rate = replayed_tokens / generated_tokens` with `replay_executions` counted separately; work accounting (`nominal_primary_work`, `wasted_work`, `executed_primary_work`, `executed_replay_work`, `total_executed_work`, `extra_execution_ratio = executed_replay_work / nominal_primary_work`, `execution_amplification = total_executed_work / nominal_primary_work`); per-Replica `busy_wall_time`/`normalized_work`; `failed_window_queue_length`; per-Replica per-window `base_overload_ratio` with aggregate ratio descriptive-only; `capacity_violation` true iff any traffic-carrying Replica ratio > 1; drain reporting (`last_arrival_time`, `drain_end_time`, `drain_duration`) with explicitly labeled observation intervals. Replay and Hedge stay separate; no deadline metrics.
- CLI `simulate-common-state-no-hedge --config configs/v1_common_state.json --tokens N [--run-id] [--artifacts-root]`, failing fast when the last arrival does not exceed `recovered_start`.
- `summary.json` with `schema_version: 2`, `scenario: common_state_no_hedge`, `dispatcher: round_robin_failure_aware`, echoed timeline, and ADR-0003 configuration identity (single-read bytes, `resolved_config`, `config_sha256`).
- Artifacts reuse `write_summary_json` unchanged (run-id confinement, no overwrite, atomic write, cleanup on failure).

## Acceptance criteria

- Controlled run with `--tokens 1000` writes `artifacts/<run-id>/summary.json` containing every spec section 7 metric with values consistent with the engine result; per-Replica ratios are reported per window (H/R: 0.7/0.7; D: 1.4 on Replica 0, 0.7 on Replica 1; F: Replica 0 N/A, 1.4 on Replica 1) and `capacity_violation` is true in D and F, false in H and R; the summary carries `last_arrival_time`, `drain_end_time`, `drain_duration`, and an explicitly labeled observation interval.
- Metric unit tests with hand-computed expectations, including: a Replay-bearing manual trace; an H-only trace where all fault metrics are zero and `capacity_violation` is false in H and `null` (unobserved) in D/F/R; `replay_rate` uses `generated_tokens` as denominator; `extra_execution_ratio` and `execution_amplification` use `nominal_primary_work`; cumulative Token queue delay sums per-attempt waits (`terminal_time - enqueue_time` for invalidated queued attempts) and degenerates to the Healthy definition without faults; per-phase `sample_count` is emitted; empty phases report `null` percentiles; unobserved phases (no positive-length intersection with `[0, drain_end_time]`) report `observed=false` and `capacity_violation=null` while observed D/F report `true` and observed H/R report `false`.
- Two identical CLI invocations produce byte-identical `summary.json` except `run_id` (spec section 8 scenario 11 at CLI level).
- H-only timeline produces metrics equal to the Healthy baseline where definitions overlap.
- `python -m unittest discover -s tests -v` green; `python -m mfg_hedge check --config configs/v1_minimal.json` exit 0; the Healthy CLI output stays byte-identical; no third-party dependency.

## Progress log

### Update: 2026-09-03 — Fault metrics, CLI, and artifacts

Status: completed

#### Goal

Complete the user-runnable Common State + No Hedge vertical slice: self-contained schema-2 configuration, fault-aware metrics, CLI, summary schema 2, artifacts, and determinism verification. No Hedge/Backup/MFG/price/quota/training/plotting/multi-Expert.

#### Changed

- Metric semantics closure first (spec section 7 + this ticket's acceptance): observed-vs-unobserved phase rule (positive-length intersection with `[0, drain_end_time]`; unobserved => `observed=false`, `capacity_violation=null`, `sample_count=0`, null percentiles) and cumulative Token queue delay (per-attempt waits, invalidated attempts count `terminal_time - enqueue_time`; degenerates to the Healthy definition without faults).
- `configs/v1_common_state.json`: self-contained schema-2 config duplicating all schema-1 base keys plus `degraded_start=100.0`, `failed_start=200.0`, `recovered_start=220.0`. `configs/v1_minimal.json` untouched.
- `src/mfg_hedge/config.py`: `CommonStateExperimentConfig` (strict missing/unknown keys, bool/NaN/Infinity rejection, fixed slice dimensions 1/2/2, base constraints via validated schema-1 projection `base_config()`, timeline via `CommonStateTimeline`), `parse_common_state_config_bytes`, `load_common_state_config_with_sha256` (single read; hash over the same bytes).
- `src/mfg_hedge/common_state_metrics.py`: `build_common_state_summary` with identity block, overall latency/queue-delay statistics, drain fields with explicitly labeled observation interval and `throughput_basis: end_to_end_including_drain`, per-phase blocks (H/D/F/R) with observed flag/bounds/sample_count/percentiles/per-Replica ratios/descriptive aggregate/violation, fault counters, work accounting with strict algebra assertions, per-Replica busy wall time vs normalized work, and the invariants block (false booleans raise).
- `src/mfg_hedge/__main__.py`: `simulate-common-state-no-hedge --config --tokens [--run-id] [--artifacts-root]`; single-read config; fail-fast (exit 2, no run directory) when the last arrival does not exceed `recovered_start`; all config/trace/run-id/artifact errors become exit 2 + stderr without tracebacks; default run id carries `common-state-no-hedge` and the Token count; artifact writing reuses the confined, no-overwrite, atomic, self-cleaning `write_summary_json` unchanged.
- `src/mfg_hedge/__init__.py`: exported the schema-2 config API. Version stays 0.3.0.
- Tests: `tests/test_common_state_config.py` (8), `tests/test_common_state_metrics.py` (11), `tests/test_common_state_cli.py` (6), written before implementation; one hand-computed expectation was corrected during green (`queue_length_at_recovered_start` is [0, 0] on the tiny trace because Replays drain before recovery).

#### Verification

- Red (before implementation): `Ran 131 tests ... FAILED (errors=8)` — ImportErrors for `mfg_hedge.common_state_metrics` / `CommonStateExperimentConfig` and six CLI errors `invalid choice: 'simulate-common-state-no-hedge'`.
- Green/Refactor: `.\.venv\Scripts\python.exe -m unittest discover -s .\tests -v` — `Ran 149 tests ... OK` (123 + 26 new; Healthy CLI and all pre-existing tests untouched and green).
- `.\.venv\Scripts\python.exe -m mfg_hedge check --config .\configs\v1_minimal.json` — exit 0.
- `.\.venv\Scripts\python.exe -m mfg_hedge simulate-common-state-no-hedge --config .\configs\v1_common_state.json --tokens 1000` — exit 0 twice; run ids `...-20260903T113119835328Z` and `...-20260903T113158365367Z`; the two `summary.json` files are byte-identical except the `run_id` line (`diff` clean after removing it).
- Controlled-run key figures (seed 20260901): primary_executions=1000, completed_tokens=1000, hedge_launches=0, replay_executions=replayed_tokens=38, replay_rate=0.038; capacity_violation D=true F=true H=false R=false (all four phases observed, sample counts H=120/D=137/F=23/R=720); work: nominal=1000.771, executed_primary=962.042, wasted=1.106, replay=31.016, total=993.058, extra=0.031, amplification=0.992; drain_end_time >= last_arrival_time; P50 1.808 <= P95 39.349 <= P99 46.449 (F-window queueing tail; ~140-sample phase P99 values are diagnostics, not stable statistics); every boolean invariant true.
- Healthy regression: `simulate-healthy-no-hedge --tokens 1000` still exits 0; its tests and scientific fields unchanged.
- Short-trace guard: `--tokens 50` exits 2 with a clear stderr message and leaves no run directory.

#### Artifacts

- `artifacts/single_expert_mechanism_check-common-state-no-hedge-t1000-20260903T113119835328Z/summary.json`
- `artifacts/single_expert_mechanism_check-common-state-no-hedge-t1000-20260903T113158365367Z/summary.json` (determinism evidence pair; gitignored)

#### Decisions and risks

- Unobserved phases report `capacity_violation=null`, never a vacuous "no violation"; the never-trigger regression therefore shows D/F/R as null rather than false.
- `total_executed_work` is defined as `executed_primary_work + executed_replay_work` and cross-checked against the per-attempt grand sum (1e-9 relative guard); `extra_execution_ratio` and `execution_amplification` divide by `nominal_primary_work`, which is positive by construction.
- Per-phase ratios are deterministic feasibility facts derived from the dispatch rule; only `capacity_violation` is gated on observation.
- F-window P95/P99 reflect the designed overload (Replica 1 ratio 1.4) plus the Replay burst; interpretation guidance lives in spec section 7's statistical-interpretation note.
- No third-party dependency; no existing artifact was modified or overwritten.

#### Next

The Common State + No Hedge feature is complete (tickets 00-05 resolved). Next candidate: specify the Hedge slice (Delayed/Immediate Hedge with `attempt_id > 0` extension) or a multi-seed pilot of this scenario, per user direction.

## Answer

All acceptance criteria pass: schema-2 config with single-read provenance; full metrics incl. per-phase observed/unobserved semantics and exact work algebra; CLI with fail-fast coverage guard and clean error mapping; two real 1000-Token runs byte-identical except `run_id`; 149/149 tests green; Healthy slice unchanged. No Hedge/Backup/MFG/price/quota/training/plotting/multi-Expert implemented.

## Comments

- 2026-09-03 (ticket 06): the invariant guard in `build_common_state_summary` originally scanned only for `value is False`, so a hypothetical nonzero `hedge_launches` would have been accepted. Fixed by an explicit `hedge_launches == 0` check raising `RuntimeError`; valid summaries are structurally and scientifically unchanged. See `.scratch/common-state-no-hedge/issues/06-enforce-hedge-invariant.md`.

