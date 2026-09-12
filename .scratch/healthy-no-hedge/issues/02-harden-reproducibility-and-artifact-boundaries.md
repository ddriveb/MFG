# Harden reproducibility and artifact boundaries

Type: task
Status: resolved
Blocked by: none

## Goal

Harden the Healthy + No Hedge slice from ticket 01 before any fault/Hedge work: stable CRN binding, run-directory confinement, and full configuration identity in `summary.json`.

## Scope

Only the hardening items below; the core simulation approach from `../spec.md` stays.

1. Bind service times to the stable key `(token_id, replica_id, attempt_id)`: in this slice `service_time = trace.service_times[replica][token_id]` (attempt 0, the Primary). No positional consumption counters.
2. Constrain `run_id` to a single safe directory name and verify the resolved run directory stays inside `artifacts_root` before writing.
3. Extend `summary.json` identity: `config_schema_version`, full `resolved_config`, `config_sha256`, `simulator_version`, `scenario`, `dispatcher`, `policy`.
4. Make `NoHedgePolicy` implement the existing `ProtectionPolicy.distribution()` interface; one policy interface only.
5. Validate `replica_count` against the trace's service-time streams with a clear error instead of a raw `IndexError`.
6. On artifact serialization/write failure, remove the partially created run directory (no empty dir or `.tmp` left behind).
7. Capture stderr in CLI tests so expected error lines stay out of test output.

## Acceptance criteria

- Every simulated Token's `service_time` equals `trace.service_times[primary_replica][token_id]`.
- `run_id` values such as `../escaped`, `..`, `a/b`, `a\\b`, or empty are rejected; a resolved-path check confirms the run directory is inside the artifacts root.
- `summary.json` carries the full identity fields above and `config_sha256` matches the config file bytes.
- `NoHedgePolicy.distribution()` returns probability 1 on Normal for every TokenClass; the simulator validates the distribution and rejects non-Normal policies with a clear error.
- Mismatched `replica_count` or undersized service-time streams raise `ValueError` naming the problem.
- A failed `summary.json` write leaves no run directory behind.
- All tests green, the three verification commands from `../spec.md` pass, and two identical CLI runs remain byte-identical except `run_id`.
- No third-party dependency added.

## Progress log

### Update: 2026-09-03 — Harden CRN binding, artifact boundaries, and run identity

Status: completed

#### Goal

Fix the three blockers raised against ticket 01 (positional CRN consumption, `run_id` path escape, partial configuration identity) plus the four smaller items, so the slice is safe to build the fault/Hedge phase on.

#### Changed

- `src/mfg_hedge/simulation.py`: service times are now indexed by the stable key — `service_time = trace.service_times[replica][spec.token_id]` (attempt 0); positional `stream_positions` counters removed. Added `_validate_trace_shape` (clear `ValueError` when the trace's stream count differs from `replica_count` or a stream is shorter than the largest Token id) and `_require_normal_only` (queries the policy once via the existing `ProtectionPolicy.distribution()` interface and rejects non-Normal distributions). `NoHedgePolicy` now implements `distribution()`; the `action()` method is gone, so a single policy interface remains.
- `src/mfg_hedge/workload.py`: `WorkloadTrace` docstring re-specified to the `(token_id, replica_id, attempt=0)` binding; generation behavior unchanged.
- `src/mfg_hedge/artifacts.py`: `run_id` must full-match `[A-Za-z0-9][A-Za-z0-9._-]*` and the resolved run directory's parent must equal the resolved artifacts root; on any failure after directory creation the partial run directory is removed (`shutil.rmtree`), so no empty dir or `.tmp` survives.
- `src/mfg_hedge/metrics.py`: `build_summary` takes `config_sha256` and adds `config_schema_version`, full `resolved_config` (`dataclasses.asdict`), `config_sha256`, `simulator_version`, `scenario="healthy_no_hedge"`, `dispatcher="round_robin"`, `policy="no_hedge"`.
- `src/mfg_hedge/__main__.py`: computes `config_sha256` over the config file bytes and also maps `ValueError` from artifact writing to exit code 2 with a stderr message.
- Version bumped to `0.2.0` in `pyproject.toml` and `src/mfg_hedge/__init__.py` because the CRN binding change alters results for identical config+seed; `simulator_version` now distinguishes the two behaviors.
- `README.md`: removed the stale "simulator not yet implemented" line.
- Added `docs/adr/0002-crn-stable-keys-and-run-identity.md` recording the durable decisions.
- Tests: rewrote `tests/test_simulation.py` (token-indexed manual trace with poisoned filler values, stable-key binding tests, mismatch `ValueError` tests, `NoHedgePolicy` interface tests), rewrote `tests/test_metrics.py` (identity fields), added `tests/test_artifacts.py` (run-id validation, overwrite refusal, failed-write cleanup), updated `tests/test_cli.py` (stderr captured, escape probe).

#### Verification

All commands executed in this session (WSL invoking the project venv's Windows interpreter):

- `.\.venv\Scripts\python.exe -m unittest discover -s .\tests -v` — Red after rewriting/adding tests and before implementation: `FAILED (failures=7, errors=8)`, including `70.0 != 0.5` showing positional vs token-indexed service draws, missing identity fields, and absent `mfg_hedge.artifacts`. Final: `Ran 50 tests ... OK` (38 previous + 12 net new/rewritten), no stray `error:` lines in output.
- `.\.venv\Scripts\python.exe -m mfg_hedge check --config .\configs\v1_minimal.json` — exit 0, `status: ok`, same audit finding as before.
- `.\.venv\Scripts\python.exe -m mfg_hedge simulate-healthy-no-hedge --config .\configs\v1_minimal.json --tokens 1000` — exit 0; run `...-20260903T050256448329Z`; summary shows `simulator_version: 0.2.0`, `config_sha256: e6bd1952…`, `resolved_config` with 21 keys, invariants `primary=1000, hedge=0, replay=0`, utilizations 0.7027/0.7079.
- Determinism re-check: second identical run `...-20260903T050314165257Z`; `diff` of the two `summary.json` files with the `run_id` line removed reports no differences (byte-identical except `run_id`).
- Escape probe: `--run-id "../escaped"` exits 2 with `error: run_id must be a single directory name …`; no directory created inside or outside `artifacts/`.

#### Artifacts

- Source: `src/mfg_hedge/simulation.py`, `src/mfg_hedge/workload.py`, `src/mfg_hedge/artifacts.py`, `src/mfg_hedge/metrics.py`, `src/mfg_hedge/__main__.py`, `src/mfg_hedge/__init__.py`, `pyproject.toml`.
- Docs: `docs/adr/0002-crn-stable-keys-and-run-identity.md`, `README.md`.
- Tests: `tests/test_simulation.py`, `tests/test_metrics.py`, `tests/test_artifacts.py`, `tests/test_cli.py`.
- Generated: `artifacts/single_expert_mechanism_check-healthy-no-hedge-t1000-20260903T050256448329Z/summary.json` and `artifacts/single_expert_mechanism_check-healthy-no-hedge-t1000-20260903T050314165257Z/summary.json` (gitignored determinism evidence).

#### Decisions and risks

- Utilizations for seed 20260901 moved slightly (0.7024/0.6951 → 0.7027/0.7079) because service draws are now bound by Token id; this is the intended behavior change and is why the version moved to 0.2.0. Old run directories remain untouched as history.
- `attempt_id` is fixed at 0 in this slice; the key scheme reserves `attempt > 0` for Hedge/Replay without shifting attempt-0 draws (ADR-0002).
- The simulator queries the policy distribution once against an initial Healthy context because `NoHedgePolicy` is context-independent; state-dependent policies will need per-decision contexts in a later slice.
- `run_id` validation is whitelist-based; names outside `[A-Za-z0-9][A-Za-z0-9._-]*` are rejected even if a platform would allow them.

#### Next

None for this ticket. The Degraded/Failed + Hedge slice can now be specified against ADR-0002's stable key scheme (`attempt_id > 0`) as a new ticket under `.scratch/healthy-no-hedge/` or a new feature directory.

## Answer

All acceptance criteria pass; evidence is in the progress log above.

- Every Token's `service_time == trace.service_times[primary_replica][token_id]` (unit tests on manual and generated traces; no positional counters remain).
- `../escaped`, `..`, `a/b`, `a\\b`, empty, `.`, and space run ids are rejected with `ValueError`; resolved run directory must sit directly inside the artifacts root; real CLI probe exits 2 and writes nothing.
- `summary.json` carries `config_schema_version`, full `resolved_config`, `config_sha256` (matches the file bytes, `e6bd1952…`), `simulator_version`, `scenario`, `dispatcher`, `policy`.
- `NoHedgePolicy.distribution()` returns Normal with probability 1 for both TokenClasses; a non-Normal policy is rejected with a clear `ValueError`; the old `action()` interface is removed.
- `replica_count`/stream mismatches and undersized streams raise `ValueError` naming the problem; no raw `IndexError`.
- A failed write removes the partially created run directory (unit test with an unserializable payload).
- 50/50 tests green; all three verification commands pass; two identical runs byte-identical except `run_id`; no third-party dependency added.

## Comments

