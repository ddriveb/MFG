# Validate transient queue physics with hand-computed scenarios

Type: task
Status: resolved
Blocked by: none

## Goal

Implement the user's requested first verification step: Replay pulses,
queued-Primary cancellation and admission-suppression feedback, with event-level
work conservation against the accepted engine.

## Scope

- Add test-only observation/accounting helpers and deterministic hand fixtures.
- Integrate a minimal test-only reservation fixture with actual engine actions.
- Check live workload, service integration, cancellation/failure losses, actual
  applied outcomes, boundary order and observer noninterference.
- Write a readable verification report; no controller search or holdout run.

## Acceptance criteria

1. Failure creates an explicit Replay-work jump, and equal total B work with
   different enqueue timing gives different backlog/latency.
2. A Backup winner cancels a queued Primary and advances subsequent work;
   a running Primary is not preempted.
3. A denied Hedge executes as Normal and can cause extra Replay on identical
   exogenous draws. Reservation charge is not confused with sampled work.
4. A separate ledger matches live state after each event/dispatch; injected
   accounting defects fail, and observing does not alter the public result.
5. Focused tests, the full suite, and the minimal configuration check pass.

## Progress log

### Update: 2026-09-05 — Validate transient queue physics

Status: completed

#### Goal

Verify Replay pulses, cancellation feedback and actual admission effects against
hand-computed physical expectations and independent event-level accounting.

#### Changed

- Added `tests/test_transient_physics.py`: 13 tests covering the three requested
  mechanisms, reservation distinctions, boundary ordering and defect detection.
- Added `tests/transient_physics_support.py`: non-mutating live-state observer,
  independent busy-time/execution ledger and test-only mean-one admission fixture.
- Added `physics-validation-report.md` with exact hand calculations and limitations.
- No production source/configuration/version or historical artifact changed.

#### Verification

- Red: `.\.venv\Scripts\python.exe -m unittest tests.test_transient_physics -v`
  -> `Ran 1 test ... FAILED (errors=1)`, missing test helper module.
- Green: same command -> `Ran 13 tests in 0.127s ... OK`.
- `.\.venv\Scripts\python.exe -m unittest discover -s .\tests -v`
  -> `Ran 347 tests in 22.361s ... OK` (334 old + 13 new).
- `.\.venv\Scripts\python.exe -m mfg_hedge check --config .\configs\v1_minimal.json`
  -> exit 0, `status: ok`, Python 3.10.11, old headroom finding unchanged.
- Injected missing admission/cancellation/discard ledger entries each fail the
  live-state conservation check. Eight 100-Token mixed-action development traces
  pass and are result-identical with/without instrumentation.

#### Artifacts

Two test files and `.scratch/mfg-hedge-attribution-study/physics-validation-report.md`.
No persistent experiment output or holdout data generated.

#### Decisions and risks

The accepted engine matches these physical fixtures; no production fix was
needed. The test fixture does not constitute production adoption of the new
quota. ADR-0011 remains Proposed as a whole; ticket 03 remains paused. The
observer uses private hooks and checks after handlers, not every internal
failure substep. Floating conservation tolerance is 1e-9, not a statistical
accuracy claim. No policy benefit, global model validity or equilibrium is claimed.

#### Next

Implement the bounded prospective admission/objective slice after reconciling
its design scope; use these fixtures as regression gates.

## Answer

All acceptance criteria pass. Replay work arrives as a jump; cancelled queued
Primary work frees A and advances later work; denied Hedge becomes Normal and
can add Replay/work. Equal total work can have different waiting trajectories.
Detailed numeric evidence is in the linked verification report.

## Comments

The user authorized this bounded validation step. ADR-0011 remains Proposed
as a whole: its objective, 81-rule search and final admission-policy adoption
are not implemented here. The reservation helper is explicitly test-only.
