# Run Stage 4b precision pilot

Type: implementation and experiment
Status: open
Blocked by: none

## Scope

Implement and execute only the accepted Stage 4b precision pilot under
ADR-0019.  The slice owns the isolated pilot library, exact four-profile
physical calls, pooled nonlinear scoring, common-path delete-one Jackknife,
corrected nested bootstrap diagnostic, immutable artifact transaction, and
its preflight/verification.  It does not modify historical engines,
configuration, Fit/Validation protocols, or historical artifacts.

The frozen inputs are namespace
`shared-backup-game:v1:stage4b-precision-pilot`, macro seed `20260907`,
resampling seed `20260908`, `C={16,32,64}`, `P={2,4,8}`, `N=8`, `c_B=0.5`,
slowdown `2.0`, hedge delay `1.5`, zero tariff, cutoff `360`, complete drain,
`epsilon_nash=0.20`, and `pilot_width_target=0.20`.  The maximum library is
generated once as 64 common-fault paths times 8 population replicates.  The
maximum exact scheduler count is `4*64*8=2,048`; nine cells are views over
the same cache.

## Acceptance criteria

- ADR-0019 is accepted with both frozen parameter names and remains otherwise
  scientifically unchanged.
- The isolated implementation has red tests before implementation and green
  focused/full verification afterward.
- The maximum 64x8 scenario library is created once, all four exact profiles
  are evaluated, calls are exactly 2,048 with no retry or hidden call, and
  nine prefix cells reuse the same immutable results.
- Pooled objectives, delete-one common-path removal, corrected nested variance
  decomposition, paired resampling, fail-closed validation, influence/rank/
  marginal diagnostics, and deterministic serialization match ADR-0019.
- Scheduler, pooled scorer, nested resampling, and RSS preflights complete
  before the formal pilot; if exact nested resampling is not feasible, stop
  with a documented blocker and produce no formal pilot result.
- A fresh non-overwriting artifact directory contains `manifest.json`,
  `pilot.json`, `report.json`, and `report.md` only after complete success.
- The result disposition is only a recommended minimum-call `C x P` cell or
  `precision_plan_infeasible`; no qualification, Fit, Validation, candidate,
  Nash, or MFG claim is produced.
- Historical Fit/diagnosis artifact hashes remain unchanged, no worker is
  left running, and the update log reports exact calls, timings, memory,
  preflights, hand checks, and all failure/claim boundaries.

## Progress log

### Update: 2026-09-06 — Claim Stage 4b precision pilot

Status: partial

#### Goal

Establish the sole claimed implementation/experiment ticket for the accepted
Stage 4b precision pilot before changing code or running any pilot call.

#### Changed

- Accepted ADR-0019 with `Status: Accepted (confirmed on 2026-09-06)`.
- Froze `epsilon_nash=0.20` and `pilot_width_target=0.20` as distinct
  parameters with coincident numeric values.
- Created and claimed this ticket; no other ticket is claimed.
- No production source, test, configuration, or artifact was changed in this
  claim step.

#### Verification

- Confirmed ticket 15 was unused before creation.
- Confirmed no other ticket has a top-level `Status: claimed`.
- Required source, ADR, spec, ticket, and test reading completed before code
  changes.
- Red focused tests and implementation are pending.

#### Artifacts

None.

#### Decisions and risks

The pilot remains planning/precision evidence only.  It cannot produce a
candidate, Nash, or MFG claim, and it cannot authorize qualification.  The
existing Stage 4 `fit_no_candidate` and all historical artifacts remain
untouched.

#### Next

Add the required failing `tests/test_precision_pilot.py` coverage, then
implement the smallest isolated pilot slice and run preflights before any
formal artifact write.

### Update: 2026-09-06 — Defer for Token restoration mainline

Status: partial

#### Goal

Preserve the accepted Expert precision-pilot protocol while yielding the
implementation claim to the restored Token-player mainline.

#### Changed

- Returned this ticket's top-level status from `claimed` to `open`.
- Kept `Blocked by: 14` and preserved all earlier history.
- No Shared-Backup source, artifact, configuration, or formal experiment was
  changed or started by this deferral.

#### Verification

- Confirmed the prior work had only pilot implementation/preflight activity;
  no formal 2,048-call pilot, result artifact, candidate, or experiment was
  produced.
- Confirmed ADR-0019 remains Accepted and no other top-level ticket is claimed
  after this handoff.

#### Artifacts

None.

#### Decisions and risks

The Expert precision branch remains an independent V2 protocol. Its preflight
blocker and historical `fit_no_candidate` boundary are retained; this handoff
does not reinterpret or delete them.

#### Next

Wait for a later explicit claim after the Token restoration slice; do not run
the Stage 4b pilot automatically.

### Update: 2026-09-06 — Clear stale resolved dependency

Status: partial

#### Goal

Remove the stale dependency marker after ticket 14 was resolved while keeping
the Stage 4b pilot deferred behind the Token restoration mainline.

#### Changed

- Changed the current ticket header from `Blocked by: 14` to
  `Blocked by: none`.
- Preserved all prior Progress log entries, including the explanation that
  the pilot was deferred for the Token mainline.

#### Verification

- Confirmed ticket 14 is resolved and ticket 15 remains `open`.
- Confirmed no Stage 4b pilot, Fit, Validation, or experiment was launched by
  this cleanup.

#### Artifacts

None.

#### Decisions and risks

This is a workflow-only correction. It does not authorize an automatic claim
or change the frozen Stage 4b statistical or sampling protocol.

#### Next

Keep ticket 15 open until it is explicitly claimed after the Token restoration
mainline and its required review are complete.
