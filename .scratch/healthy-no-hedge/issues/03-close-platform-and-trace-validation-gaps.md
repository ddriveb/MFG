# Close platform path and trace validation gaps

Type: task
Status: resolved
Blocked by: none

## Goal

Close the remaining Windows artifact-name, stable Token-key, and configuration identity race conditions found during review of ticket 02.

## Scope

1. Reject run IDs that are not stable directory names on Windows, including trailing dots and reserved device names, while retaining the existing cross-platform single-component checks.
2. Require every `WorkloadTrace` consumed by the simulator to use unique, dense, non-negative Token IDs exactly equal to `0..token_count-1` in arrival order.
3. Read configuration bytes once in the CLI, parse `ExperimentConfig` from those same bytes, and calculate `config_sha256` from them.
4. Add regression tests and update the durable decision record and patch version.

## Acceptance criteria

- `run.`, `CON`, `con.txt`, `NUL`, `COM1`, and `LPT9` are rejected before any directory is created.
- Existing valid run IDs remain accepted.
- Negative, duplicate, sparse, or out-of-order Token IDs raise a clear `ValueError` before queue evaluation.
- Generated workload traces remain valid and deterministic.
- The simulation CLI opens the configuration path exactly once while producing a hash that matches the parsed bytes.
- All existing tests and new regression tests pass with no third-party dependency.

## Progress log

### Update: 2026-09-03 — Close platform and trace validation gaps

Status: completed

#### Goal

Reject Windows-aliased artifact names, enforce the dense Token-ID contract behind stable service keys, and derive parsed configuration plus SHA-256 from one file read.

#### Changed

- `src/mfg_hedge/artifacts.py`: rejects trailing-dot run IDs and Windows reserved device stems (`CON`, `PRN`, `AUX`, `NUL`, `COM1..9`, `LPT1..9`) while allowing ordinary names that only contain those substrings.
- `src/mfg_hedge/simulation.py`: validates Token IDs are unique, dense, non-negative, and ordered exactly as `0..token_count-1` before indexing service draws.
- `src/mfg_hedge/config.py`: adds `parse_config_bytes` and `load_config_with_sha256`; both the parsed configuration and digest now come from one immutable byte payload.
- `src/mfg_hedge/__main__.py`: uses the combined configuration loader for simulation runs.
- `src/mfg_hedge/__init__.py` and `pyproject.toml`: expose the combined loader and bump the patch version to `0.2.1`.
- `CONTEXT.md` and ADR-0003: record the dense Token-ID, canonical run-name, and single-read provenance contracts.
- Tests add regressions for Windows aliases/device names, valid substring names, negative/duplicate/sparse/out-of-order Token IDs, digest correctness, and one-open CLI behavior.

#### Verification

- Clean Red run after correcting a test-helper placement mistake: `.\.venv\Scripts\python.exe -m unittest tests.test_artifacts tests.test_simulation tests.test_cli -v` — `Ran 29 tests`, `FAILED (failures=6)`. The failures corresponded to the three intended gaps: Windows names, four invalid Token-ID forms, and two configuration opens.
- Focused Green run: `.\.venv\Scripts\python.exe -m unittest tests.test_artifacts tests.test_simulation tests.test_config tests.test_cli -v` — `Ran 33 tests`, `OK`.
- Full suite: `.\.venv\Scripts\python.exe -m unittest discover -s .\tests -v` — `Ran 57 tests`, `OK`.
- Configuration self-check: `.\.venv\Scripts\python.exe -m mfg_hedge check --config .\configs\v1_minimal.json` — exit 0, `status: ok`, expected failure-headroom audit retained.
- Real CLI: `.\.venv\Scripts\python.exe -m mfg_hedge simulate-healthy-no-hedge --config .\configs\v1_minimal.json --tokens 1000` — exit 0; wrote the versioned summary listed below.
- Boundary probe rejected `run.`, `CON`, `con.txt`, `NUL`, `COM1`, and `LPT9`, leaving the temporary artifacts root empty.
- `.\.venv\Scripts\python.exe -m compileall -q .\src .\tests` — passed.

#### Artifacts

- Source: `src/mfg_hedge/artifacts.py`, `src/mfg_hedge/simulation.py`, `src/mfg_hedge/config.py`, `src/mfg_hedge/__main__.py`, `src/mfg_hedge/__init__.py`, `pyproject.toml`.
- Tests: `tests/test_artifacts.py`, `tests/test_simulation.py`, `tests/test_config.py`, `tests/test_cli.py`.
- Docs: `CONTEXT.md`, `docs/adr/0003-canonical-run-names-trace-keys-and-config-bytes.md`.
- Generated: `artifacts/single_expert_mechanism_check-healthy-no-hedge-t1000-20260903T054950567662Z/summary.json` (gitignored).

#### Decisions and risks

- Token IDs are deliberately constrained rather than replacing the compact array-backed trace with a mapping; generated traces already satisfy this contract.
- Run names use a portable subset even when a host filesystem would accept a broader name.
- Version `0.2.1` changes validation/provenance only; valid default workload and queue semantics are unchanged.

#### Next

Define the Degraded/Failed + Hedge work as a new feature spec and ticket; do not extend this resolved Healthy-only ticket.

## Answer

All acceptance criteria pass. Windows-aliased names are rejected before directory creation, invalid Token IDs fail before queue evaluation, configuration parsing and hashing share one file read, all 57 tests pass, and the real 1,000-Token CLI run succeeds under simulator version `0.2.1`.

## Comments
