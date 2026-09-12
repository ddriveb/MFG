# Design Stage 4b precision pilot

Type: design
Status: resolved
Blocked by: none

## Scope

Pre-register a precision-planning pilot for the finite Shared-Backup game.
The pilot compares additional independent common-fault paths with additional
independent populations inside each path.  It does not run a pilot, Fit,
Validation, qualification, solver, or MFG experiment and it does not alter
finite-game physics, Stage 4 samples, or historical artifacts.

The design is paired with Proposed ADR-0019.  Its status remains Proposed
until the user confirms it.

## Acceptance criteria

- Freeze a new pilot-only namespace, macro seed, 64-path/8-population maximum
  nested library, rectangular prefix rule, four evaluated profiles, and the
  exact scheduler-call budget.
- Define one normative gain sign, exact pooled nonlinear objective handling,
  delete-one-common-path jackknife, nested diagnostic variance components,
  fixed resampling seeds/counts, and fail-closed conditions.
- Define every per-cell output, marginal C/P comparison, call/time estimate,
  and the data-independent rule for recommending a later qualification
  configuration or returning `precision_plan_infeasible`.
- Keep the later all-Expert/all-255 simultaneous epsilon-regret check and the
  future state-policy extension outside this pilot.
- Correct ticket 13's stale Next pointer only in its Comments section.
- Run the full regression and config check; do not generate artifacts or run
  any formal experiment.

## Progress log

### Update: 2026-09-06 — Design Stage 4b precision pilot

Status: completed

#### Goal

Design, but not execute, a pre-registered precision pilot that identifies
whether common-path depth or nested population replication is the more useful
source of uncertainty reduction for the three highest-risk finite-system
deviations around the NSNS point estimate.

#### Changed

- Created and claimed this ticket as the only active ticket.
- Read `AGENTS.md`, `CONTEXT.md`, `docs/agents/issue-tracker.md`,
  `docs/agents/update-format.md`, ADR-0015 through ADR-0018, the shared-game
  spec, the Stage 4 Fit/campaign ticket, ticket 13, the variance diagnosis,
  the Stage 4 performance report, and the current Stage 4 scoring/orchestration
  implementation before drafting the protocol.
- Added Proposed ADR-0019 with the frozen sample library, four profiles,
  pooled gain/jackknife definitions, two-layer diagnostic resampling, output
  schema, call/time accounting, qualification disposition, and claim
  boundary.
- Added a correction pointer in ticket 13 Comments; its historical Progress
  log was not rewritten.

#### Verification

- Full regression: `.venv\Scripts\python.exe -m unittest discover -s tests -v`
  — 475/475 passed.
- Config check: `.venv\Scripts\python.exe -m mfg_hedge check --config configs\v1_minimal.json`
  — exit 0, `status: ok`, with the existing warning retained.
- No pilot, Fit, Validation, campaign, solver, MFG run, or artifact write was
  performed.

#### Artifacts

None.  Existing Stage 4 and variance-diagnosis artifacts were preserved.

#### Decisions and risks

The ADR is deliberately Proposed.  The pilot's nested bootstrap variance
decomposition is a diagnostic approximation for a nonlinear pooled CVaR
functional; exact common-path delete-one values remain the primary cluster
uncertainty audit.  Pilot outputs cannot be reused as final qualification or
Validation data and cannot support a Nash or MFG claim.

#### Next

Wait for user confirmation of Proposed ADR-0019; confirmation is required
before creating or claiming an implementation/experiment ticket or running
the precision pilot.

### Update: 2026-09-06 — Correct Stage 4b statistical contracts

Status: completed

#### Goal

Correct the two statistical contract ambiguities identified during review
without accepting ADR-0019 or authorizing pilot execution.

#### Changed

- Corrected the nested bootstrap decomposition by subtracting
  `V_within / B_within` from the sample variance of outer inner-means before
  adding the full within component, with `max(0, ...)` truncation.
- Required independent inner population draws for every outer path position,
  including repeated selections of the same common path ID.
- Split the provisional planning and qualification parameters into
  `pilot_width_target = 0.20` and `epsilon_nash_proposed = 0.20`.
  Their equal numeric values are explicitly documented as coincidental; they
  control different decisions.
- Kept ADR-0019 `Proposed`, the 64x8 library, 2,048-call maximum, four
  profiles, pooled objective, Jackknife, data isolation, fail-closed rules,
  and final all-Expert/all-255 qualification boundary unchanged.

#### Verification

- Documentation-only revision; no production code, tests, configuration, or
  historical artifact was modified.
- Full regression: `.venv\Scripts\python.exe -m unittest discover -s tests -v`
  — 475/475 passed.
- Config check: `.venv\Scripts\python.exe -m mfg_hedge check --config
  configs\v1_minimal.json`
  — exit 0, `status: ok`, with the existing warning retained.
- No pilot, Fit, Validation, campaign, solver, or MFG run was performed.

#### Artifacts

None.  No new or overwritten experiment artifact was produced.

#### Decisions and risks

ADR-0019 remains Proposed pending user confirmation.  The corrected nested
bootstrap is still a diagnostic approximation for the nonlinear pooled CVaR
functional.  The pilot width target is not an epsilon-Nash guarantee, and the
pilot cannot certify the precision of all 255 future deviations.

#### Next

Await confirmation of the corrected Proposed ADR-0019.  Only after
confirmation may a separate implementation/experiment ticket be created or
claimed and the precision pilot be run.
