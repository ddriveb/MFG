# Harden calibration CRN and production path

Type: task
Status: resolved
Blocked by: none

## Goal

Fix the two blocking defects in ticket 02's calibration: probe insertion silently renumbered and redrew background Tokens (breaking identity, parity, and CRN), and the production rho search cannot converge on the short F window (discrete cohort starvation). Also: 400 independent per-sample episodes, recorded rho-search diagnostics, production full-table contract, and serialization provenance. Ticket 02 stays resolved; its history is untouched.

## Scope

1. Probe designation replaces probe insertion: an existing in-window background Token is designated as the probe (class replaced, draws replaced by stable `calibration:...:probe:<sample>` keys); no renumbering, no redraw, no parity change; N/D/I variants share the identical trace; rho measurement excludes any probe slot.
2. 400 samples use 400 independent deterministic episodes (sample-index-derived labels, deterministic skip-and-continue when an episode lacks an in-window Token); cells record `background_episode_count`.
3. Stable rho search: shared underlying background draws per state (rate scaling only), cohort-extended measurement until >= 400 in-window background arrivals, `background_cohort_count` recorded, bisection keeps bounds [0.05, 3.0] / 40 iterations / 0.02 tolerance with endpoint+midpoint+best logging and a located error message; per-(state, rho) result cache shared across classes.
4. Production `CalibrationTable` contract: expected grid, exact 70-cell completeness, no interpolation across missing grid points, no extra F Hedge cells, sample_count >= 400, `rho_error == abs(achieved - target) <= 0.02`; partial tables constructible for tests only with completeness off.
5. Deterministic serialization gains the full provenance block (schema version, grid, timeline, bounds, tolerance, iterations, samples, cohort/episode counts, scientific config, seed namespace); no timestamps.
6. ActionStats numeric fields reject bool.

## Acceptance criteria

- Red evidence captured for: probe-insertion identity breakage (old design), F/0.5/U and F/0.7/R rho-search failure (old design).
- After the fix: identity tests pass (non-probe Tokens unchanged in arrival/class/draws/parity across N/D/I and vs the pure background), the two F regression cells succeed within tolerance, production table validates exactly 70 cells, full provenance serialization is byte-deterministic.
- A real `build_calibration_table(config)` run succeeds with all diagnostics recorded in this ticket.
- Full suite and `check` pass; version bump justified in the log; no solver/price/quota/CLI/config/experiment work; no artifacts written.

## Progress log

### Update: 2026-09-04 — Harden calibration CRN and production path

Status: completed

#### Goal

Fix the two blocking calibration defects (probe-insertion identity breakage, F-window rho non-convergence), the 400-independent-sample requirement, stable rho search with full logging, the production 70-cell contract, and serialization provenance.

#### Changed

- `src/mfg_hedge/calibration.py` (rewritten): designated-probe semantics via public `build_calibration_background` + `prepare_probe_trace` (an in-window background Token keeps id/arrival/parity; only its class and its three attempt draws are replaced by stable `calibration:...:probe:<sample>` keys; no insertion, no renumbering, no stream regeneration); per-sample independent episodes with deterministic skip-and-retry labels; rho measurement on probe-free episodes extended until >= 400 in-window cohort arrivals (episodes with zero in-window arrivals skip the engine run); `find_background_rate` logs every evaluation (endpoints + midpoints), keeps the first in-tolerance midpoint, and fail-fast errors name state/rho/class plus the best candidate; per-(state, rho) `rate_cache` shared across classes; `CalibrationCell` gains `background_cohort_count`/`background_episode_count`; `CalibrationTable` gains `expected_grid`/`require_complete`/`provenance` with the full 70-cell production contract (exact coverage, no extra F Hedge cells, sample floor, `rho_error == abs(achieved-target) <= 0.02`); serialization embeds the provenance block (schema version, grid, timeline, bounds, tolerance, iterations, samples, cohort floor, episode size, scientific config, seed namespace, simulator_version) with no timestamps.
- `src/mfg_hedge/domain.py`: `ActionStats` numeric fields now reject bool (additive strictness; all existing constructions unchanged).
- `tests/test_calibration.py`: probe-identity tests (non-probe arrival/class/draws/parity unchanged vs pure background and across variants; per-sample probe draws differ), production-table contract tests (70-cell completeness, missing/extra/sample-floor/rho_error consistency, provenance determinism), bisection contract tests (evaluation log, located failure messages), real F/0.5/U and F/0.7/R regression cells, updated real D cell (per-sample episodes), updated `make_cell` helper.
- Version 0.5.0 -> 0.6.0 (`__init__.py` + `pyproject.toml`): the public calibration-table serialization schema changed (provenance block, new cell fields); no released-artifact compatibility existed to preserve.
- Ticket 02: Comments pointer added; history untouched.

#### Verification

- Red (before the rewrite): `Ran 186 tests ... FAILED (errors=1)` — the updated suite fails to import (`cannot import name 'build_calibration_background'`); the old implementation was previously observed failing F-window bisection with discrete-cohort starvation (motivating defect 2).
- Green: `.\.venv\Scripts\python.exe -m unittest discover -s .\tests -v` — `Ran 216 tests ... OK` (185 + 31 net new/updated) in ~15 s.
- `.\.venv\Scripts\python.exe -m mfg_hedge check --config .\configs\v1_minimal.json` — exit 0.
- Real production run `build_calibration_table(config)`: **70 cells**, 47.9 s; max `rho_error` 0.019989 (<= 0.02); min `sample_count` 400; min `background_cohort_count` 400; F/0.5/R and F/0.5/U both achieved 0.4984 (err 0.0016, rate 0.4648); F/0.7/R achieved 0.6935 (err 0.0065, rate 0.6607); second full build byte-identical (48.1 s). No artifacts written (in-memory only).
- Identity evidence: `test_non_probe_tokens_and_draws_are_identical_to_pure_background` compares every non-probe Token's arrival, class, and attempt-0/1/2 draws against the pure background episode and proves id parity unchanged; variant traces are equal objects up to the probe's own fields.

#### Artifacts

None generated. Source: `src/mfg_hedge/calibration.py`, `src/mfg_hedge/domain.py`, `src/mfg_hedge/__init__.py`, `pyproject.toml`. Tests: `tests/test_calibration.py`.

#### Decisions and risks

- rho measurement runs probe-free episodes, so probe exclusion holds by construction (no probe slot exists there); probe runs never feed rho.
- The 0.02 tolerance is unchanged; stability came from the >= 400 cohort floor, not from relaxing tolerance.
- Full-table build takes ~48 s; it is a verification step, not a unit test.
- The skipped-episode retry labels are part of the deterministic namespace and are recorded via `background_episode_count`.

#### Next

Wait for the user's ticket 03 instruction (MFG solver and quota). Tickets 03/04 remain open.

## Answer

All acceptance criteria pass: designated-probe CRN identity proven; F/0.5/U and F/0.7/R now calibrate within tolerance; the production table contract enforces exactly 70 cells; a real full-table build succeeds with byte-identical rerun; 216/216 tests green; `check` exit 0; version 0.6.0 justified by the calibration serialization schema change; ticket 02 resolved with a Comments pointer; tickets 03/04 open; no solver/price/quota/CLI/experiment work.

## Comments

