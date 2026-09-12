# Implement the boundary-pinned capacity solver

Type: task
Status: resolved
Blocked by: none

## Goal

Implement user-accepted ADR-0008: retain the single cold-start price-zero diagnostic, replace the binding branch's nested rho fixed point with a boundary-pinned one-dimensional price root, apply the policy-aware infeasibility certificate, and verify the real 70-cell table at first-round load 0.5. Do not start ticket 04 or generate paired-experiment artifacts.

## Scope

- Accept ADR-0008 and make spec section 12 normative.
- Update `mfg_solver.py` only as required for the boundary-pinned H/D branch and load-aware infeasibility certificate; retain ADR-0007 accounting, F semantics, statuses, final checks, and quota contract.
- Add red-first unit coverage for the boundary-pinned execution contract.
- Version the changed scientific solver behavior.
- Point ticket 04 at this ticket and preserve ticket 12's blocked history with a correction pointer.

## Acceptance criteria

- Price zero is the only path using the cold-start policy/rho fixed point; failure never enters the binding branch.
- A binding H/D solve fixes rho at target utilization, interpolates stats once, and searches a monotone signed capacity gap without nested rho iteration.
- Policy-aware minimum incremental work can certify boundary infeasibility; bracket failure remains nonconverged.
- Final ADR-0007 capacity, complementarity, quota-budget, clamp, F, and official-gate contracts remain true.
- Full unit suite and `check` pass.
- A real 70-cell table at load 0.5 yields official H/D solutions, forced-Normal F, deterministic serialization, and quota acceptance. Otherwise stop honestly without starting ticket 04.

## Progress log

### Update: 2026-09-04 — Implement the boundary-pinned capacity solver

Status: completed

#### Goal

Implement accepted ADR-0008, remove nested rho fixed points from the binding price search, add the policy-aware capacity certificate, and pass the real 70-cell/load-0.5 official gate without starting ticket 04.

#### Changed

- `src/mfg_hedge/mfg_solver.py`: the price-zero cold-start policy/rho fixed point now runs exactly once. A successful overloaded result enters a boundary-pinned branch that queries ActionStats once at `rho = target_max_utilization`, computes the exact per-class minimum `(hedge + replay)` infeasibility witness, and evaluates every price through a direct soft best response over the fixed statistics. Signed bracket expansion and bisection no longer invoke `_inner_solve`; a negative boundary price-zero gap after an overloaded unconstrained result is reported as `nonconverged/binding_branch_inconsistent`. Finalization can reuse the pinned statistics and retains ADR-0007 capacity, complementarity, work, residual, and Hedge-budget accounting.
- `src/mfg_hedge/quota.py`: the official gate now recomputes H/D capacity acceptance conditions, checks the final Hedge-work budget, and requires the F solution to be exact forced-Normal with zero price/Hedge work/budget and passing policy/load residuals.
- `tests/test_mfg_solver.py`: added boundary execution-contract tests (single price-zero inner solve, fixed-rho monotonic price response, failed price-zero isolation, finite load-0.5 D root, and inconsistent-branch diagnostic); updated infeasibility expectations to the policy-aware certificate.
- `tests/test_quota.py`: added corrupted complementarity/F-policy gate cases, aligned the F-path fixture with the official forced-Normal contract, and made cross-test imports package-stable.
- `docs/adr/0008-boundary-pinned-capacity-solver.md`: status changed to Accepted (user confirmation 2026-09-04) and the supersession is now effective.
- `.scratch/mfg-hedge-paired-comparison/spec.md`: section 12 is the normative ADR-0007/0008 boundary-pinned contract; ticket and verification mappings now include ticket 15; confirmed solver caps/tolerances and load 0.5 are explicit.
- `src/mfg_hedge/__init__.py` and `pyproject.toml`: version 0.9.0 -> 0.10.0 because the scientific solver behavior and serialized provenance changed.
- Ticket 04 now declares `Blocked by: 15`; ticket 12 keeps its blocked history and contains only a correction pointer to ticket 15.

#### Verification

- Red (before implementation): `.venv\Scripts\python.exe -m unittest tests.test_mfg_solver -v` -> `Ran 18 tests ... FAILED (failures=5)`: the binding branch called `_inner_solve` 23 times, fixed-price totals were non-monotone through the nested rho solve, load-0.5 D stayed nonconverged, and the old primary-only certificate/reason remained.
- Focused Green: `.venv\Scripts\python.exe -m unittest tests.test_mfg_solver tests.test_quota -v` -> `Ran 39 tests ... OK`.
- Full Green: `.venv\Scripts\python.exe -m unittest discover -s tests -v` -> `Ran 262 tests ... OK` in 20.544 s.
- `.venv\Scripts\python.exe -m mfg_hedge check --config configs\v1_minimal.json` -> exit 0, `status: ok` (the pre-existing load-0.7 single-domain audit warning remains descriptive).
- Real production verification, entirely in memory: built all 70 calibration cells at load 0.5 in 139.381 s; two solver runs were byte-identical. H = `converged/price_zero_slack`, price 0, rho 0.803808486, gap -0.192383036, policy residual 7.18e-7, unclamped. D = `converged/capacity_boundary_root`, price 35.680805969, rho 0.9, gap -1.6e-8, policy residual 0, load residual 1.05e-8, Hedge work 0.251495811 <= budget 0.251495826, unclamped. F = `forced_normal/forced_normal_state`, exact all-Normal, rho 1.0, gap +0.1, structural violation true, unclamped.
- The real solution passed quota projection on a deterministic 1000-Token trace: 40 windows, maximum per-window planned budget excess 0.0; 27 runtime quota suppressions were reported rather than hidden.
- No paired experiment was run and no experiment artifact was created.

#### Artifacts

None generated. Source and test changes: `src/mfg_hedge/mfg_solver.py`, `src/mfg_hedge/quota.py`, `src/mfg_hedge/__init__.py`, `pyproject.toml`, `tests/test_mfg_solver.py`, `tests/test_quota.py`. Documentation: ADR-0008, paired-comparison spec, tickets 04/12/15.

#### Decisions and risks

- Load 0.5 passes the real official gate without loosening any tolerance or changing calibration, penalties, CRN, engine semantics, or quota rules. Loads 0.6/0.7 remain stress diagnostics and are not silently treated as first-round official settings.
- H legitimately remains a zero-price slack solution; D is the intended positive-price capacity-bound solution; F remains a disclosed structural exception, never mislabeled converged.
- The 27 quota suppressions are expected realized online projections under finite windows; the planned expected work still obeys every window budget.
- Version 0.10.0 distinguishes the new scientific solver behavior. No earlier experiment artifact was rewritten.

#### Next

Ticket 04 is now unblocked and may be claimed to implement the paired CLI, metrics, transactional artifacts, and the first actual A/B run.

## Answer

ADR-0008 is implemented and accepted. All unit and configuration checks pass, and the real 70-cell/load-0.5 gate produces official H/D solutions plus forced-Normal F with deterministic serialization and quota acceptance. Ticket 15 is resolved; ticket 04 was not started and no experiment artifact was written.

## Comments
