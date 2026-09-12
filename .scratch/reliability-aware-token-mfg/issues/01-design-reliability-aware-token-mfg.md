# Design reliability-aware simultaneous Token MFG

Type: design
Status: resolved
Blocked by: none

## Goal

Define the simultaneous Token-player mean-field extension of the accepted
finite reliability-aware multi-Replica routing baseline.  Freeze the
information structure, common-noise conditioning, population trend, finite
state/action approximation, scaling convention, cost models, claim boundary,
and implementation dependency chain without changing production behavior.

## Scope

- Create the feature specification and Proposed ADR.
- Create implementation tickets 02 through 06 with the dependency chain
  `02 -> 03 -> 04 -> 05 -> 06`.
- Keep all later implementation tickets open and do not claim ticket 02.
- Do not modify Python, configuration, historical artifacts, or experiment
  results.
- Do not run a solver, MFG, qualification, or campaign.

## Acceptance criteria

- The Token decision is simultaneous within a batch and cannot observe other
  Tokens' realized choices.
- `mu_t`, `nu_t`, `eta_t`, predicted `x_t`, and realized shares are separate
  auditable values.
- Common noise is an external fault-history prefix, distinct from the public
  observation filtration.
- The open-system transient equations replace a static population fixed point.
- The K-scaled family is explicit and does not call fixed-K token-count growth
  an MFG limit; batch release rate and `E[B_K]=b*K` are unambiguous.
- Unpriced and priced private/social cost semantics are distinct.
- Mean-field BR and finite-`K` deviation are separate estimands.
- Pure BR, logit response, finite-N regret, Nash, and MFG residual have
  separate labels and claim boundaries.
- The finite corrected routing result is retained as a baseline only.
- ADR numbering does not overwrite the already accepted ADR-0036.

## Answer

The design is complete.  Because ADR-0036 is already occupied by the accepted
three-round finite routing evaluation, the new proposal is recorded as
ADR-0037 rather than overwriting the historical file.  Tickets 02--06 remain
open with the required dependency chain.

## Progress log

### Update: 2026-09-10 — Design reliability-aware simultaneous Token MFG

Status: completed

#### Goal

Specify a decentralized, simultaneous Token MFG extension over the corrected
finite reliability-aware routing engine, with no implementation or experiment.

#### Changed

Added the feature specification, Proposed ADR-0037, and open implementation
tickets for the simultaneous finite engine, population estimator, tagged finite
deviations, unpriced/priced solver, and K-scale qualification/holdout.  The
spec freezes common-noise conditioning, `m_t`/`x_t`/realized-share separation,
causal observation, finite bucket actions, the K={8,16,32,64} scale, private
and social costs, response semantics, baselines, verification layers, and
claim boundary.  ADR-0036 was preserved because it already has a different
accepted meaning.

#### Verification

Read-only checks performed before this design update: governance files,
ADR-0020, ADR-0021, ADR-0022, ADR-0035, ADR-0036, the accepted finite routing
specification and corrected three-round report, plus the existing Token
continuation/deviation/payoff and fixed-point modules.  No production tests,
configuration check, experiment, or solver run was executed.

#### Artifacts

- `.scratch/reliability-aware-token-mfg/spec.md`
- `docs/adr/0037-reliability-aware-token-mean-field-game.md`
- `.scratch/reliability-aware-token-mfg/issues/01-design-reliability-aware-token-mfg.md`
- `.scratch/reliability-aware-token-mfg/issues/02-implement-simultaneous-finite-routing.md`
- `.scratch/reliability-aware-token-mfg/issues/03-population-state-and-share-calibration.md`
- `.scratch/reliability-aware-token-mfg/issues/04-tagged-continuation-and-finite-deviations.md`
- `.scratch/reliability-aware-token-mfg/issues/05-implement-unpriced-priced-mfg-solver.md`
- `.scratch/reliability-aware-token-mfg/issues/06-qualify-scale-and-holdout.md`

#### Decisions and risks

This is a proposed design, not an accepted MFG or a result.  Placement,
physical routing, CRN, and historical artifacts remain unchanged.  Existence,
uniqueness, scaling accuracy, and support coverage are open risks.  The ADR
number conflict is recorded explicitly rather than silently reusing ADR-0036.

#### Next

Wait for explicit acceptance of ADR-0037; only then claim ticket 02 and freeze
its numerical physical protocol before generating any traces.

### Update: 2026-09-10 — Close transient-state and deviation semantics gaps

Status: completed

#### Goal

Revise the proposed MFG contract so that its population state, filtrations,
scaling law, and two deviation estimands match the open transient system.

#### Changed

Separated `mu_t` (active Token states), `nu_t` (Replica states), `eta_t`
(new/Replay decision cohort), and `x_t` (cohort action shares).  Replaced the
static population equation with the transient path and transition map, split
the external common-noise filtration `F_t^0` from the public observation
filtration `G_t`, fixed batch-release versus batch-size scaling and introduced
`N_K`, and separated mean-field BR from finite-`K` unilateral reruns.  Ticket
02 now has `Blocked by: ADR-0037 acceptance`.

#### Verification

Read-only content and whitespace checks were run on the revised documents.
No production code, tests, config check, solver, or experiment was run.

#### Artifacts

- `.scratch/reliability-aware-token-mfg/spec.md`
- `docs/adr/0037-reliability-aware-token-mean-field-game.md`
- `.scratch/reliability-aware-token-mfg/issues/02-implement-simultaneous-finite-routing.md`

#### Decisions and risks

ADR-0037 remains Proposed.  The system is explicitly transient and open;
static `m*` notation is not used as a consistency condition.  Mean-field BR
and finite-`K` deviation remain separate estimands and cannot be substituted.

#### Next

Wait for explicit acceptance of ADR-0037 before claiming ticket 02.

### Update: 2026-09-10 — Verify revised transient contract

Status: completed

#### Goal

Confirm that the proposed documents no longer contain the rejected static
population equations or an unbounded ticket-02 entry point.

#### Changed

Added the explicit transient core transition to both the spec and ADR, and
updated ticket 02's scope and acceptance criteria to use the separated state
objects and filtration names.

#### Verification

The final read-only contract check confirmed required `mu_t`, `nu_t`, `eta_t`,
`x_t`, `Phi_t`, `F_t^0`, `G_t`, `N_K`, and scaling anchors, found no trailing
whitespace, and found no stale `m_t`/static `m*` equations.  Ticket 02 reports
`Blocked by: ADR-0037 acceptance`.  No production tests, config check,
solver, or experiment was run.

#### Artifacts

- `.scratch/reliability-aware-token-mfg/spec.md`
- `docs/adr/0037-reliability-aware-token-mean-field-game.md`
- `.scratch/reliability-aware-token-mfg/issues/02-implement-simultaneous-finite-routing.md`

#### Decisions and risks

ADR-0037 remains Proposed.  The open-system formulation is now the only
specified consistency object; mean-field BR and finite-`K` deviations remain
separate and cannot be interchanged.

#### Next

Wait for explicit ADR-0037 acceptance before claiming ticket 02.

## Comments

- The corrected ADR-0035 three-round report is finite routing evidence only.
