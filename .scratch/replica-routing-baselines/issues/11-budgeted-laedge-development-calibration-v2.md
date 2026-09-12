# Budgeted LÆDGE development calibration v2

Type: implementation
Status: resolved
Blocked by: none

## Scope

Run a fresh Budgeted LÆDGE development calibration under the accepted
ADR-0033 semantics.  This ticket corrects only the calibration-grid coverage
endpoint and the experiment identities.  It does not modify the scheduler,
budget ledger, delta targets, confirmation gate, holdout protocol, or any
historical artifact.

Ticket 10 remains `blocked`; its v1 artifact and calibration failure are
preserved and are not reinterpreted or used as v2 observations.

## Frozen v2 protocol

- Development namespace:
  `replica-routing-baselines:budgeted-laedge:v2:development`.
- Development macro seed: `20260912`.
- Holdout namespace:
  `replica-routing-baselines:budgeted-laedge:v2:holdout`.
- Holdout macro seed: `20260913`.
- The v2 episode identities are disjoint from v1; no r1 row, trace, result, or
  rate estimate may be concatenated into v2.
- Calibration rates, in ascending order, are exactly:
  `{0.00, 0.01, 0.02, 0.04, 0.06, 0.08, 0.12, 0.18, 0.25, 0.35, 0.50,
  4.00}`.
- The seven delta labels remain `{0%, 1%, 3%, 5%, 8%, 12%, 18%}`.
- Calibration is `64 × 12 = 768` calls; NIIN and unconstrained conservative
  references are `256 + 256` calls.  If coverage passes, confirmation adds
  `192 × 7 + 192` calls.  The development ceiling remains exactly `2,816`.
- The upper coverage gate remains a maximum `0.25` percentage-point shortfall
  relative to NIIN mean work.  It is not relaxed.
- The confirmation overfill gate remains `0.25` percentage points.  Underfill
  greater than `1` percentage point is `target_underfilled`.
- If the 18% target exceeds the unconstrained conservative attainable mean,
  its selected finite rate is recorded with `saturation_limited=true`, along
  with target, achieved delta, and underfill.  This is not exact 18% attainment.
- Interpolated rates must be executed in confirmation; no assumption of linear
  realized work is permitted.  Confirmation overfill fails closed.
- Holdout is `1,024 × 11 = 11,264` calls and is prohibited in this ticket,
  including after a confirmed v2 development mapping.
- Failed calls are counted and retained; there is no retry, supplementation,
  deletion, replacement, or r1 reuse.

## Acceptance criteria

1. Red tests precede production changes and cover the v2 endpoint, fresh
   identities, exact call ceiling, saturation-limited 18% handling, and
   rejection of v1/r1 concatenation.
2. v2 calibration/reference execution is deterministic and uses fresh episode
   identities.
3. The v2 development artifact is transactional and non-overwriting, and
   reports the complete rate curve, mapping/confirmation status, fingerprints,
   achieved deltas, and actual calls.
4. No holdout call is made by this ticket.
5. Focused, affected, full suite, and config checks pass.

## Progress log

### Update: 2026-09-09 — Claimed v2 calibration coverage correction

Status: claimed

#### Goal

Replace only the v1 calibration upper endpoint `1.00` with `4.00` under new
development identities, then run the fresh development protocol.  Preserve v1
and stop before holdout.

#### Changed

- Created and claimed this ticket.
- No production code or v2 artifact has been changed yet.

#### Verification

- Confirmed ticket 10 remains blocked and its r1 artifact is retained.
- Confirmed the v2 call arithmetic remains capped at `2,816`.

#### Artifacts

No v2 artifact yet.

#### Decisions and risks

The `0.25` percentage-point upper coverage gate, confirmation gates, delta
labels, and scheduler semantics remain frozen.  The v2 endpoint is not a
post-hoc splice into r1.

#### Next

Write v2 failing tests, implement the isolated protocol variant, and run only
the v2 development calibration.

### Update: 2026-09-09 — v2 development confirmed; holdout not started

Status: resolved

#### Goal

Execute the fresh v2 calibration/reference/confirmation protocol with the
`4.00` upper rate endpoint, preserve v1/r1, and stop before holdout.

#### Changed

- Added explicit v1/v2 protocol profiles to the internal campaign runner.
- Added the v2 rate grid, namespace/seed constants, protocol fingerprint,
  transactional executor, and saturation-limited mapping field.
- Added focused v2 tests for endpoint replacement, fresh identities, r1
  rejection, coverage failure, saturation-limited 18%, and negative realized
  underfill.
- Corrected confirmation validation so a negative realized delta is valid
  underfill; only the declared underfill threshold determines its status.
- No scheduler, budget, CRN, ADR-0033, historical artifact, or package
  version semantics were changed.

#### Verification

- Real Red before implementation: import failure for the absent v2 protocol
  symbols.
- The first v2 execution consumed `2,816` calls but was retained as a failed
  implementation run because the old nonnegative achieved-delta check rejected
  a valid `-0.8035393283753134%` underfill.  Its artifact is preserved and was
  not reused.
- Final v2 focused suite: `6/6` passed; combined Budgeted core/campaign/v2
  focused suite: `23/23` passed; affected suites: `73/73` passed.
- Full suite: `691/691` passed.
- Config check: `status: ok`.
- Final v2 development: `2,816/2,816` calls, `confirmed`; no retries,
  supplementation, deletion, or r1 concatenation.
- NIIN mean total work: `450.30240860918315`.
- Unconstrained conservative mean total work: `529.9509894245826`.
- Rate `4.00` calibration mean total work: `528.869068654762`; upper coverage
  therefore passed the unchanged `0.25` percentage-point gate.
- Frozen planned rates were:
  `0.0, 0.029794334898597338, 0.06752794601883591,
  0.09304612180547656, 0.17872563347189496,
  0.2770264334252587, 4.0` for delta labels
  `0%, 1%, 3%, 5%, 8%, 12%, 18%` respectively.
- Confirmation achieved deltas were:
  `-0.008035393283753134, -0.008035393283753134,
  0.015741761949179978, 0.035651132104542294,
  0.07552662484030792, 0.10848466740723994,
  0.1715168326174492`.
  No point overfilled.  The 18% point is `saturation_limited=true` and is
  underfilled by `0.008483167382550783`; it is not exact 18% attainment.

#### Artifacts

- [v2 ticket](D:/project/mfg_hedge_v1/.scratch/replica-routing-baselines/issues/11-budgeted-laedge-development-calibration-v2.md)
- [v2 runner](D:/project/mfg_hedge_v1/src/mfg_hedge/budgeted_laedge_campaign.py)
- [v2 tests](D:/project/mfg_hedge_v1/tests/test_budgeted_laedge_campaign_v2.py)
- [v2-r1 failed summary](D:/project/mfg_hedge_v1/artifacts/budgeted-laedge-development-20260909-v2-r1/summary.json)
- [v2-r2 confirmed summary](D:/project/mfg_hedge_v1/artifacts/budgeted-laedge-development-20260909-v2-r2/summary.json)
- [v2-r2 manifest](D:/project/mfg_hedge_v1/artifacts/budgeted-laedge-development-20260909-v2-r2/manifest.json)
- [v2-r2 protocol](D:/project/mfg_hedge_v1/artifacts/budgeted-laedge-development-20260909-v2-r2/protocol.json)
- [v2-r2 episode rows](D:/project/mfg_hedge_v1/artifacts/budgeted-laedge-development-20260909-v2-r2/episode_rows.jsonl)

The confirmed v2 protocol fingerprint is
`55600734cd72c4e2fa9cf6e267aef0a6b2adf5c27743d8058760bd1be77c28e9` and the
source bundle fingerprint is
`510ff3b8daf8dad95a0cc1b92d9400058d7195564940aad1af089f8e005d7398`.

#### Decisions and risks

Development is confirmed, but this is only a frozen budget-rate mapping and
confirmation result.  It is not a Pareto result and makes no optimality, BR,
Nash, or MFG claim.  The first v2 failed artifact remains auditable and was
not overwritten.  The 18% point is explicitly saturation-limited.

#### Next

Stop here.  Do not start holdout in this ticket.  A separate authorized step
may consume the frozen `11,264` holdout calls using only the confirmed v2
mapping and its matching v2 holdout namespace/seed.
