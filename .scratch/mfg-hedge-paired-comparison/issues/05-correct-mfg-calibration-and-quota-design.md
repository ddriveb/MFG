# Correct MFG calibration and quota design

Type: task
Status: resolved
Blocked by: none

## Goal

Apply nine design corrections to the paired-comparison design before any implementation: fault-aware calibration, the full load fixed point, the corrected Hedge budget, a causal online deterministic quota allocator, boundary-forced control windows, planned-vs-realized budget reporting, boundary-order test split, metric/artifact edge rules, and parameter positioning. Documentation only.

## Scope

1. Replace constant-state calibration with H→D→F→R transition episodes in the calibration namespace; >= 400 valid arrival-cohort samples per cell; deterministic acceptance that Normal and some Hedge action differ in q or expected_replay_work in at least one D-state fixture.
2. Iterate the full load fixed point (rho includes Primary + Hedge + Replay work); convergence requires policy, price, and load residuals below 1e-6 within 200 iterations; non-converged policies are diagnostics only.
3. Hedge budget deducts predicted Primary and Replay work rates.
4. Replace offline window rounding with an online per-class deficit allocator with fixed tie-breaks, explicit error bounds, and Urgent budget reservation.
5. Control windows: W = 25.0 maximum, forcibly cut at Common State boundaries; R reuses the H policy immediately.
6. Separate planned expected Hedge work (guaranteed <= budget) from realized Hedge work and realized excess in reporting.
7. Split the timer-vs-failure boundary test into the two directions; fix the event order chain.
8. Metric/artifact edges: peak/mean null without launches; sustained overload = longest consecutive run; partial final bin rule; transactional four-file artifact commit.
9. Record that all listed defaults are untuned mechanism-verification values.

## Acceptance criteria

- spec, ADR-0005, ADR-0006, and tickets 01-04 reflect all nine corrections consistently.
- ADR-0005/0006 remain Proposed; ticket 01 remains open and unclaimed; ticket 00 stays resolved and untouched.
- Full test suite and `check` pass, proving no code/config/artifact change.

## Progress log

### Update: 2026-09-04 — Correct MFG calibration and quota design

Status: completed

#### Goal

Apply the nine design corrections to the paired-comparison design documents before any implementation; documentation only.

#### Changed

- `../spec.md`: (1) calibration now runs fault-aware H→D→F→R episodes in the `calibration:` namespace with >= 400 arrival-cohort samples per populated cell and a fail-fast D-state contrast fixture; (2) the solver iterates the full load fixed point (rho covers Primary+Hedge+Replay work; three residuals < 1e-6; non-converged = diagnostics only); (3) the Hedge budget deducts predicted primary and replay work rates; (4) the quota is a causal online per-class deficit allocator with fixed tie-breaks, a `< 1` per-action error bound, Urgent share reservation (never borrowed, expires at window close), and no offline largest-remainder claim; (5) control windows max width 25.0, forcibly cut at state boundaries (default cuts at 100/200/220, R reuses H policy at 220 immediately); (6) planned-vs-realized Hedge work reporting separated; (7) the timer-vs-failure test split into 5a (Primary=B/Backup=A suppressed) and 5b (Primary=A: Replay, timer voided), with the full same-time order fixed; (8) metric/artifact edge rules (peak/mean null without launches; sustained overload = longest consecutive run; partial final bin normalized by actual width and excluded from peaks; four-file transactional commit); (9) all listed defaults marked as untuned mechanism-verification values.
- `docs/adr/0006-finite-action-mfg-and-quota-semantics.md`: rewritten around calibration separation, the load fixed point, the corrected budget, the online allocator, and planned-vs-realized honesty; Status remains Proposed.
- `docs/adr/0005-hedge-copy-lifecycle-and-random-keys.md`: explicit same-time order chain added; Status remains Proposed.
- Tickets 01-04 updated (boundary-test split, fault-aware calibration scope/acceptance, fixed-point + online-quota scope/acceptance, transactional artifacts and edge rules in ticket 04). Ticket 00 untouched; ticket 01 remains open and unclaimed.

#### Verification

- `.\.venv\Scripts\python.exe -m unittest discover -s .\tests -v` — `Ran 152 tests ... OK` after the edits (no code changed).
- `.\.venv\Scripts\python.exe -m mfg_hedge check --config .\configs\v1_minimal.json` — exit 0, `status: ok`.

#### Artifacts

- `.scratch/mfg-hedge-paired-comparison/spec.md`, `issues/01..05-*.md`
- `docs/adr/0005-hedge-copy-lifecycle-and-random-keys.md`, `docs/adr/0006-finite-action-mfg-and-quota-semantics.md` (both still Proposed)

#### Decisions and risks

- The online deficit allocator sacrifices offline rounding optimality for causality; the `< 1` per-action bound is the stated guarantee.
- Strict Urgent share reservation can leave budget unused when Urgent arrivals are absent; accepted for v1 and documented.
- All listed parameters are untuned mechanism-verification defaults; formal claims need the later multi-seed/sensitivity work.

#### Next

User gives final confirmation on ADR-0005/0006 and spec section 17; then ticket 01 can be claimed for implementation.

## Answer

All nine corrections are applied consistently across spec, both ADRs (still Proposed), and tickets 01-04; ticket 00 untouched; 152/152 tests and `check` pass with zero code/config/artifact changes; ticket 01 not started.

## Comments

