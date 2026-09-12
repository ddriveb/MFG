# Implement cancellation-semantics ablation hooks

Type: implementation
Status: resolved
Blocked by: none

## Scope

Implement the isolated 2×2 cancellation-semantics hooks described by
ADR-0032. Preserve the default historical NIIN and LÆDGE behavior, CRN
identity, fault-first ordering, complete drain, executed work accounting, and
the existing result contracts. This ticket covers code and focused tests only;
it does not run the formal 2×2 panel, price sweep, or new algorithm.

## Acceptance criteria

1. Conservative and immediate running-loser modes are explicit and validated
   for both the one-shot NIIN engine and idle-release LÆDGE executor.
2. Immediate cancellation retains work executed through the winner time,
   invalidates stale completion events, and frees the replica only after the
   same-time event batch.
3. Conservative mode records a completed loser and preserves the historical
   default behavior byte-for-byte for existing inputs.
4. Queued loser, failure, Replay, drain, and CRN stream semantics remain
   unchanged.
5. Focused tests are written red first and cover the four arms plus the
   cancellation-specific invariants.
6. Full suite and config check pass after the focused suite.

## Progress log

### Update: 2026-09-09 — Claim and freeze implementation slice

Status: claimed

#### Goal

Separate running-loser cancellation from admission/scheduling while keeping
the historical protection engines and artifacts unchanged.

#### Changed

- Created and claimed this unique implementation ticket.
- Added proposed ADR-0032 with the four-arm semantics and claim boundary.
- Closed the stale duplicate P95 ticket; the resolved cohort ticket remains
  authoritative for the completed r2 work.
- No production implementation or formal experiment has started.

#### Verification

- Read `AGENTS.md`, `CONTEXT.md`, the local issue/update instructions,
  replica-routing spec, ADR-0030, ADR-0031, the resolved baseline tickets,
  and the current NIIN/LÆDGE engines before editing.
- No test or campaign command was run in this setup update.

#### Artifacts

None.

#### Decisions and risks

The ADR remains Proposed. Historical defaults and the r2 artifact are
immutable; no price change or formal experiment is authorized by this ticket.

#### Next

Write the cancellation-mode focused tests, observe the expected red failure,
then implement the isolated hooks.

### Update: 2026-09-09 — Implement 2×2 cancellation hooks

Status: completed

#### Goal

Implement and verify the isolated NIIN/LÆDGE × conservative/preemptive
running-loser comparison without starting the formal ablation panel.

#### Changed

- Added `cancel_running_losers` as an explicit opt-in engine mode. The
  historical NIIN/online default remains conservative (`False`); the
  historical LÆDGE default remains its native preemptive behavior (`True`).
- In preemptive mode, a running loser is settled through the winner time,
  retains its executed work, receives `cancelled_running`, increments the
  replica generation, and has its stale completion ignored.
- In conservative mode, a running loser continues to completion and records
  `completed_loser`. Queued losers remain `cancelled_queued` with zero work.
- Added the internal `protection_cancellation_ablation` selector with four
  canonical arm keys: `niin:conservative`, `niin:preemptive`,
  `laedge:conservative`, and `laedge:preemptive`.
- Added focused tests for both engines, stale completion, executed-work
  retention, queued cancellation, failure-first Replay, and unchanged CRN
  stream binding.
- No price, workload, Reservation, metric, historical artifact, or public
  package export was changed. Version was not bumped.

#### Verification

- Red: `.venv\\Scripts\\python.exe -m unittest
  tests.test_protection_cancellation_ablation -v` failed at collection with
  `ModuleNotFoundError` before the new module existed.
- Final focused: the same command ran 5 tests, all `OK`.
- Affected regression suites (`test_hedge_simulation`, `test_token_online`,
  `test_protection_baseline_experiment`, `test_attribution_episode`, and
  `test_attribution_metrics`) ran 92 tests, all `OK`.
- Full: `.venv\\Scripts\\python.exe -m unittest discover -s tests -v`
  returned exit 0 with 660 tests passing.
- Config: `.venv\\Scripts\\python.exe -m mfg_hedge check --config
  configs/v1_minimal.json` returned exit 0 and `status: ok`.
- The final focused tests confirm a degraded primary at speed 0.5 retains
  0.5 executed work when cancelled at the backup winner time; queued losers
  retain 0.0 executed work; and failed primaries still produce exactly one
  Replay under either mode.

#### Artifacts

None. No formal 2×2 campaign, price sweep, or generated experiment artifact
was run.

#### Decisions and risks

ADR-0032 remains Proposed; this ticket only implements the isolated hook and
tests. The 2×2 selector returns engine results but does not define new
metrics or choose an algorithm. The formal panel still needs a separately
accepted sample count, budget, metric table, and artifact namespace.

#### Next

Accept ADR-0032 and create a separate run ticket that freezes and executes the
2×2 paired cancellation ablation; do not alter prices before that comparison.
