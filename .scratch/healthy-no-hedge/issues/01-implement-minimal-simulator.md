# Implement the minimal Healthy + No Hedge simulator

Type: task
Status: resolved
Blocked by: none

## Goal

Implement and verify the vertical slice defined in `../spec.md`.

## Scope

Only the Healthy + No Hedge behavior and outputs explicitly listed in the spec.

## Acceptance criteria

All acceptance criteria and verification commands in `../spec.md` pass, with evidence recorded below.

## Progress log

### Update: 2026-09-03 — Implement minimal Healthy + No Hedge simulator

Status: completed

#### Goal

Deliver the Healthy + No Hedge vertical slice from `../spec.md`: deterministic workload generation, fixed round-robin dispatch, per-Replica FCFS queueing, summary metrics, and a unique `artifacts/<run-id>/summary.json`, following red-green-refactor.

#### Changed

- Added `src/mfg_hedge/workload.py`: `generate_workload` builds an immutable `WorkloadTrace` before any policy runs; arrival, Token-class, and per-Replica service times use independent `random.Random` streams whose seeds are derived from `base_seed` via SHA-256 labels; `lognormal_parameters` implements the spec's `sigma2 = log(1 + cv**2)`, `mu = log(mean) - sigma2/2`; arrival rate is `healthy_offered_load * replicas_per_expert / healthy_service_mean` (1.4 for the default config).
- Added `src/mfg_hedge/simulation.py`: `RoundRobinDispatcher` (fixed dispatch choice for this slice, documented in its docstring), `NoHedgePolicy` (Normal action only, consumes no randomness), per-Replica FCFS evaluation, `TokenResult` with all spec-required fields, and `InvariantCounters`.
- Added `src/mfg_hedge/metrics.py`: `percentile` with an explicit linear-interpolation definition (fractional rank `(n-1)*p/100` on the sorted sample) and `build_summary`; observation interval is `[0, last completion]` for throughput and utilization.
- Added `src/mfg_hedge/artifacts.py`: `write_summary_json` creates a fresh run directory (raises `FileExistsError` if it exists, so a previous run is never overwritten) and moves the payload into place with `os.replace`.
- Extended `src/mfg_hedge/__main__.py` with the `simulate-healthy-no-hedge --config --tokens [--run-id] [--artifacts-root]` command; default run id carries a UTC timestamp, and `run_id` is the only summary field allowed to differ across identical runs.
- Exported `generate_workload`/`simulate_healthy_no_hedge` and trace/result types from `src/mfg_hedge/__init__.py`.
- Added focused tests: `tests/test_workload.py`, `tests/test_simulation.py`, `tests/test_metrics.py`, `tests/test_cli.py`; silenced CLI stdout inside `tests/test_cli.py` during refactor.

#### Verification

All commands executed in this session (Linux/WSL host invoking the project venv's Windows interpreter, equivalent to the documented `.\.venv\Scripts\python.exe` form):

- `.\.venv\Scripts\python.exe -m unittest discover -s .\tests -v` — Red phase after adding the four new test files and before implementation: `FAILED (errors=6)` with `ModuleNotFoundError: No module named 'mfg_hedge.simulation'` / `'mfg_hedge.workload'` / CLI failures, while the 6 pre-existing tests stayed green. Final run after implementation and refactor: `Ran 38 tests ... OK` (6 pre-existing + 32 new, covering determinism, queue equations, dispatch balance, percentiles, invariants, and CLI artifact creation).
- `.\.venv\Scripts\python.exe -m mfg_hedge check --config .\configs\v1_minimal.json` — exit 0; reported `status: ok`, Python 3.10.11, `single_domain_failure_load: 1.4`, and the known single-domain failure headroom audit finding.
- `.\.venv\Scripts\python.exe -m mfg_hedge simulate-healthy-no-hedge --config .\configs\v1_minimal.json --tokens 1000` — exit 0; wrote `artifacts/single_expert_mechanism_check-healthy-no-hedge-t1000-20260903T043348850119Z/summary.json` with `arrival_rate: 1.4`, `token_count: 1000`, `primary_executions: 1000`, `hedge_launches: 0`, `replay_executions: 0`, replica assignments 500/500, utilizations 0.7024 and 0.6951, and `latency_p50 <= latency_p95 <= latency_p99`.
- Determinism: a second identical `--tokens 1000` invocation (run id `...-20260903T043409294976Z`) produced a `summary.json` whose bytes are identical to the first run except for the `run_id` line (verified by `diff` on the two files with the `run_id` line removed: no differences).
- Overwrite protection: the CLI test reruns the same `--run-id` and asserts a non-zero exit with the first run's bytes untouched.

#### Artifacts

- Source: `src/mfg_hedge/workload.py`, `src/mfg_hedge/simulation.py`, `src/mfg_hedge/metrics.py`, `src/mfg_hedge/artifacts.py`, `src/mfg_hedge/__main__.py`, `src/mfg_hedge/__init__.py`.
- Tests: `tests/test_workload.py`, `tests/test_simulation.py`, `tests/test_metrics.py`, `tests/test_cli.py`.
- Generated: `artifacts/single_expert_mechanism_check-healthy-no-hedge-t1000-20260903T043348850119Z/summary.json` and `artifacts/single_expert_mechanism_check-healthy-no-hedge-t1000-20260903T043409294976Z/summary.json` (gitignored; two runs kept as determinism evidence).

#### Decisions and risks

- Round-robin is the fixed Dispatcher for this slice, as mandated by `../spec.md`; the choice is documented on `RoundRobinDispatcher` and no new ADR was created because the spec already records the decision.
- Per-Replica service-time streams hold one draw per Token per Replica, so any future dispatcher consumes each Replica's stream in assignment order without perturbing common random numbers; memory cost is `replica_count * token_count` floats and is acceptable for v1 token counts.
- Stream seeds derive from `base_seed` via SHA-256 labels instead of `random.Random(tuple)` because string/tuple hashing is process-randomized; this keeps traces reproducible across processes and platforms.
- Throughput and utilization are measured over `[0, last completion]`; this convention is documented in `build_summary` and must not be changed silently.
- `run_id` is the sole summary field excluded from the byte-identical determinism criterion; no timestamps appear in scientific content.
- The config's known finding stands: single-domain failure load 1.4 exceeds the 0.9 target, to be handled by the failure-slice tickets, not this one.

#### Next

None — the ticket's acceptance criteria and verification commands all pass. The next slice (Degraded/Failed with Delayed/Immediate Hedge) needs its own spec and ticket.

## Answer

All acceptance criteria from `../spec.md` pass; evidence is in the progress log above.

- Same configuration and seed give byte-identical `summary.json` except `run_id` (two real CLI runs, `diff` clean after removing the `run_id` line); no timestamps in scientific content.
- Arrival times non-decreasing and service times positive (`tests/test_workload.py`).
- Replica assignment counts differ by at most one (500/500 in the 1000-Token run; `tests/test_simulation.py`).
- `start >= arrival`, `completion == start + service_time`, `latency == completion - arrival` hold for every Token (`tests/test_simulation.py`).
- Completed Token count equals generated count (1000/1000).
- Hedge launches 0, Replay executions 0, Primary executions 1000 (`summary.json` invariants).
- Utilizations 0.7024 and 0.6951, inside `[0, 1]` over the documented `[0, last completion]` interval.
- The 6 pre-existing tests remain green; 32 new tests cover determinism, queue equations, dispatch balance, percentiles, invariants, and CLI artifact creation (38 total, `OK`).
- No third-party dependency added; `pyproject.toml` `dependencies` stays `[]` and only the standard library is imported.

## Comments
