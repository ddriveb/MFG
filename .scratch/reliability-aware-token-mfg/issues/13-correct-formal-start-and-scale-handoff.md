# Correct formal start and K-scale policy handoff

Type: correction
Status: resolved
Blocked by: none

## Goal

Correct the remaining formal-r2 execution defects before any qualification
identity is dispatched: enforce the accepted timing gate at the formal runner,
derive every K=8 start inside checkpointed physics, carry the confirmed policy
to the next K without caller-supplied scientific state, and enforce the frozen
qualification call ceiling.

## Scope

- A formal runner requires the complete accepted r2 timing report.
- K=8 uniform/LOEW/risk-aware seeds are generated inside the first committed
  forward boundary; no uncheckpointed formal trace evaluation is permitted.
- K=16/32/64 initialization is derived deterministically from the preceding
  confirmed policy and forward snapshot, not supplied by a caller.
- Every dispatch is rejected before reservation if it would exceed the frozen
  qualification call ceiling.
- Preserve models, samples, CRN, namespaces, prices, residuals, UCB gates,
  physics, and holdout boundary.

## Acceptance

- Red tests expose missing preflight enforcement, caller-supplied scale state,
  uncheckpointed start construction, and missing global call-ceiling checks.
- Bounded fixtures prove deterministic internal start construction and actual
  prior-K policy handoff.
- Focused, affected, full-suite, config, compile, and diff checks pass.
- No r2 qualification or holdout call is dispatched.

## Progress log

### Update: 2026-09-11 — pre-launch audit exposed undefined formal scale gates

Status: blocked

#### Goal

Audit and start r2 only if the concrete backend could reproduce every formal
initialization, scale handoff, timing gate, and call ceiling without hidden or
uncheckpointed physical work.

#### Changed

- Created Proposed ADR-0039 to define checkpointed K=8 start construction,
  prior-K policy projection, cross-K policy distance, normalized finite-K UCB,
  mandatory timing validation, and the global call-ceiling check.
- Set ticket 06 to `blocked / Blocked by: 13` before creating an r2 checkpoint
  or dispatching a qualification identity.
- No production code, namespace, seed, sample, or artifact was changed.

#### Verification

- Source inspection confirmed `FormalQualificationRunner.__init__` did not
  require a timing report or enforce the 51,072-call ceiling.
- `run_qualification` required caller-supplied policies for every K; the stored
  preceding candidate was not used for K=16/32/64.
- Constructing the three K=8 policies through the current public helper requires
  a forward evaluation outside the checkpoint.
- The runner could return `qualification_pass` without executing the frozen
  normalized-UCB trend and K=32/K=64 policy-distance gates.
- No scheduler call or formal identity was consumed in this audit.

#### Artifacts

- `docs/adr/0039-formal-token-mfg-scale-handoff-and-gates.md`
- `.scratch/reliability-aware-token-mfg/issues/13-correct-formal-start-and-scale-handoff.md`

#### Decisions and risks

The previous 811-test result proves bounded mechanics, not the missing formal
scale semantics.  Implementing an arbitrary projection or normalization after
seeing r2 results would invalidate qualification, so the contract is frozen
before launch and awaits explicit acceptance.

#### Next

Accept or revise ADR-0039; once accepted, implement ticket 13 with Red tests and
leave r2 unstarted until its full acceptance suite passes.

### Update: 2026-09-11 — ADR-0039 accepted

Status: open
Blocked by: none

ADR-0039 is accepted.  Ticket 13 is now ready for implementation, but remains
unclaimed until the correction work begins.  No r2 checkpoint, qualification
call, holdout call, or artifact was created by this status transition.

### Update: 2026-09-11 — implemented formal start and scale handoff gates

Status: completed

#### Goal

Implement Accepted ADR-0039 without dispatching any r2 qualification identity.

#### Changed

- Made a complete r2 timing report mandatory before checkpoint construction.
- Added checkpointable K=8 uniform, simultaneous-LOEW, and corrected risk-aware
  routing seeds and materialized the actual first forward policy.
- Added occupancy-weighted prior-K action-bucket projection, strict normalized
  finite-K UCB accounting, K=32/K=64 policy distance, fallback reporting, and
  the parent-side 51,072-call ceiling.
- Tightened scale-marginal validation and made unsupported scale projection fall
  back uniformly over currently UP Replicas.

#### Verification

- Real Red: requiring the timing gate first produced
  `TypeError: missing required keyword-only argument 'preflight'`; the initial
  scale-gate test then exposed a missing `math` import.
- Focused solver/backend/state-machine suite: 23 tests passed.
- Affected suite: 72 valid selected tests passed; one separately named module
  was a nonexistent test name and was not counted as a product failure.
- `python -m py_compile` passed for all four changed modules.
- `.venv/Scripts/python.exe -m unittest discover -s tests -v` ran 815 tests:
  all passed in 280.870 seconds.
- `.venv/Scripts/python.exe -m mfg_hedge check --config configs/v1_minimal.json`
  returned `status: ok`.
- `git diff --check` returned exit 0.
- Formal r2 qualification and holdout scheduler calls remained zero.

#### Artifacts

No experiment artifact or formal checkpoint was created.

#### Decisions and risks

ADR-0039 is fully implemented.  A separate no-run memory audit measured one
isolated K=64 trace at 36,910,699 serialized bytes; the current 32-trace formal
payload would therefore exceed 1.18 GB before forward-state and process-copy
overhead.  This is an execution-safety defect outside ADR-0039's scientific
semantics and blocks launch until the panel is streamed.

#### Next

Implement ticket 14's per-episode streaming/checkpoint boundary, then resume
ticket 06 without changing any frozen scientific identity.

## Answer

Accepted ADR-0039 is implemented and verified.  Formal r2 remains deliberately
unstarted because the newly measured K=64 payload size requires a separate
streaming execution correction before a safe launch.
