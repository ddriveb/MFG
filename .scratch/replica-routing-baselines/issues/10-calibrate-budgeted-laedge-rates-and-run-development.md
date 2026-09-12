# Calibrate Budgeted LÆDGE rates and run development confirmation

Type: implementation
Status: blocked
Blocked by: none

## Scope

Implement the development calibration runner and the guarded holdout runner
for accepted ADR-0033.  This ticket must first run only the frozen development
panel.  It must not consume holdout calls unless the development mapping and
confirmation gates are complete and independently frozen.

The runner uses the isolated kernel from ticket 09 and does not change the
Budgeted LÆDGE scheduler, cancellation semantics, CRN streams, workload,
prices, Reservation parameters, MFG, best response, or Pareto definitions.

## Frozen development execution protocol

- Parent development namespace:
  `replica-routing-baselines:budgeted-laedge:v1:development`.
- Macro seed: `20260910` for all development sub-libraries.  Sub-library
  namespaces are disjoint and are part of each episode identity:
  - `:calibration` — 64 episodes;
  - `:confirmation` — 192 episodes;
  - `:niin-reference` — 256 episodes;
  - `:unconstrained-reference` — 256 episodes.
- Confirmation's 192 `fixed-dispatcher:no-hedge` reference runs use the same
  confirmation episode identities and are paired with the seven Budgeted
  LÆDGE delta runs; they are not an additional episode library.
- Episode indices are dense `0..count-1` within each sub-library.  No episode
  may be retried, supplemented, deleted, or replaced after failure.
- The common episode protocol is ADR-0033's frozen environment:
  `ATTRIBUTION_V1_PROTOCOL`, `c_B=.5` equivalent two-Replica physics,
  slowdown `2.0`, hedge delay `2.0`, complete drain, and the existing
  attempt-key CRN streams.

## Frozen rate calibration contract

The 12 normalized planned-budget rates are, in ascending order:

`{0.00, 0.01, 0.02, 0.04, 0.06, 0.08, 0.12, 0.18, 0.25, 0.35, 0.50, 1.00}`.

Rate `1.00` is the frozen full normalized Backup-capacity endpoint used to
probe the unconstrained neighborhood.  Rates are never chosen from
confirmation or holdout outcomes.

The seven target labels remain:

`delta = {0%, 1%, 3%, 5%, 8%, 12%, 18%}`.

For each target, select the largest rate whose calibration mean total work is
no greater than the target mean NIIN-conservative work multiplied by
`1 + delta`.  No closest-point substitution is allowed.  A mapping may use
linear interpolation only between adjacent grid points and must record the
bracketing rates and interpolation fraction; the interpolated rate is still
checked against the confirmation panel without adjustment.

Calibration is invalid if the rate-to-work curve is non-monotone, the zero or
upper endpoint is missing, the upper endpoint does not reach the frozen
unconstrained neighborhood within the declared gate, or any required arm
fails.  The upper-end coverage gate is a maximum allowed shortfall of `0.25`
percentage points of the NIIN reference mean work relative to the
unconstrained-conservative reference.

Confirmation is 192 episodes and runs the seven mapped Budgeted LÆDGE rates
plus the paired fixed-dispatcher No-Hedge reference.  The confirmation mean
realized delta for each target must not exceed the target by more than
`0.0025` (0.25 percentage points).  It may be below target.  A shortfall
greater than `0.01` is labeled `target_underfilled` and remains descriptive,
not relabeled as exact target attainment.  Any overfill beyond `0.0025`,
non-monotone mapping, missing rate, failed arm, duplicate fingerprint, or
invariant failure makes the development result `calibration_failed` and
blocks holdout.

## Frozen call arithmetic

- Calibration: `64 × 12 = 768` scheduler calls.
- Confirmation Budgeted LÆDGE: `192 × 7 = 1,344` calls.
- Confirmation fixed No-Hedge reference: `192` calls.
- NIIN-conservative reference: `256` calls.
- Unconstrained conservative reference: `256` calls.
- Total development budget: `2,816` calls exactly.
- Holdout remains `1,024 × 11 = 11,264` calls and is unavailable until the
  development result is `confirmed`.

The runner owns call accounting.  Failed calls count, are recorded with the
original exception type and message, and are never retried.  A partial panel
cannot produce a mapping or holdout permission.

## Provenance and artifacts

- Every episode records the existing full trace fingerprint before the first
  arm and checks it after every arm.
- The development artifact records the complete rate curve, NIIN and
  unconstrained reference means, selected mapping, confirmation achieved
  deltas, target status, attempted calls, failures, and all episode rows.
- `source_bundle_fingerprint_v1` covers the sorted byte-content manifest of
  all `src/mfg_hedge/*.py` files plus `configs/v1_minimal.json`.
- `protocol_fingerprint_v1` covers this protocol, ADR-0033, the exact rate
  grid, delta labels, namespaces, seeds, arm identities, tolerances, and
  call arithmetic.
- Development artifact directory must be fresh and transactional.  No
  existing artifact may be overwritten.
- Holdout runner must reject missing, failed, stale, or mismatched development
  mapping fingerprints before making a scheduler call.

## Acceptance criteria

1. Failing tests precede production implementation and cover rate-grid
   validation, exact call split, largest-feasible-rate selection, interpolation
   recording, monotonicity/coverage gates, confirmation tolerances,
   fail-closed behavior, no retry, and holdout refusal before confirmation.
2. Development runner is deterministic and keeps calibration and confirmation
   episode identities disjoint.
3. The formal development protocol reserves at most 2,816 calls and writes
   one fresh artifact; a calibration or reference gate failure stops before
   confirmation, records only the calls actually consumed, and makes no
   holdout call.
4. Focused, affected, full suite, and config checks pass.

## Progress log

### Update: 2026-09-09 — Claimed calibration gate

Status: claimed

#### Goal

Close the Budgeted LÆDGE budget-rate protocol gap by freezing a calibration
grid and implementing a development-only mapping/confirmation gate before
any holdout execution.

#### Changed

- Created and claimed this ticket.
- Frozen the 12-rate grid, split namespaces, macro seed, 64/192/256 sample
  counts, exact `2,816` development calls, confirmation tolerances, and the
  holdout refusal condition.
- No production code or experiment artifact changed yet.

#### Verification

- Read the accepted ADR-0033, ticket 09, issue-tracker/update-format rules,
  existing episode generation, baseline, cancellation, and artifact writers.
- Confirmed ticket 09 is resolved and no other ticket is top-level claimed.
- No development or holdout scheduler call was run in this initial update.

#### Artifacts

- `.scratch/replica-routing-baselines/issues/10-calibrate-budgeted-laedge-rates-and-run-development.md`

#### Decisions and risks

The previous `2,816 = 256 × 11` arithmetic is retained only as a total
budget.  It is no longer interpreted as seven executable Budgeted arms before
their rates are calibrated.  Holdout remains explicitly blocked until
confirmation freezes the mapping.

#### Next

Write the failing contract tests, then implement the deterministic development
and guarded holdout runners.

### Update: 2026-09-09 — Development calibration fail-closed

Status: blocked

#### Goal

Implement the frozen development calibration and confirmation gate before any
Budgeted LÆDGE holdout execution.  Holdout was intentionally not started.

#### Changed

- Added the isolated development/holdout runner in
  `src/mfg_hedge/budgeted_laedge_campaign.py`.
- Added the Budgeted LÆDGE attribution mode without changing historical fixed
  or idle-release output contracts.
- Added focused contract tests for the exact split, rate mapping,
  monotonicity/coverage gates, confirmation tolerances, transactional output,
  and holdout refusal.
- Reused one immutable episode trace across arms and recorded source/protocol
  fingerprints in a fresh development artifact.

#### Verification

- Real Red before implementation: `ModuleNotFoundError` for the new campaign
  module.
- Focused Budgeted LÆDGE and calibration suites: `18/18` passed.
- Affected suites: `67/67` passed.
- Full suite: `685/685` passed.
- Config check: `status: ok`.
- Formal development attempt: `1,280` calls completed (`768` calibration
  plus `256` NIIN and `256` unconstrained references).  The runner correctly
  stopped before confirmation because the frozen upper-coverage gate failed.
- Calibration rate rows were complete for all `64` episodes.  The NIIN mean
  total work was `449.467444386731`; the unconstrained-conservative reference
  was `529.875313981439`; the `1.00` rate reached `527.957375063414`, leaving
  an absolute shortfall of `1.91793891802502`, or `0.426713645666132`
  percentage points relative to NIIN, above the frozen `0.25`-point gate.
- No confirmation calls, mapping, or holdout calls were made.  No retry,
  supplementation, deletion, or replacement occurred.

#### Artifacts

- [campaign runner](D:/project/mfg_hedge_v1/src/mfg_hedge/budgeted_laedge_campaign.py)
- [focused tests](D:/project/mfg_hedge_v1/tests/test_budgeted_laedge_campaign.py)
- [development summary](D:/project/mfg_hedge_v1/artifacts/budgeted-laedge-development-20260909-r1/summary.json)
- [development manifest](D:/project/mfg_hedge_v1/artifacts/budgeted-laedge-development-20260909-r1/manifest.json)
- [development protocol](D:/project/mfg_hedge_v1/artifacts/budgeted-laedge-development-20260909-r1/protocol.json)
- [episode rows](D:/project/mfg_hedge_v1/artifacts/budgeted-laedge-development-20260909-r1/episode_rows.jsonl)

#### Decisions and risks

The failure is the intended fail-closed protocol outcome, not a reason to
select a different rate grid from the observed result.  The confirmation
mapping is not frozen, so no holdout may run.  This is not a Pareto result and
does not support optimality, BR, Nash, or MFG claims.  The package version was
not changed because the runner remains internal.

#### Next

Keep this ticket blocked.  A separate accepted protocol decision is required
before changing the coverage gate or rate grid; otherwise rerun the same
development protocol only after the governance state permits it.  Do not
consume the `11,264` holdout calls while the mapping is unconfirmed.
