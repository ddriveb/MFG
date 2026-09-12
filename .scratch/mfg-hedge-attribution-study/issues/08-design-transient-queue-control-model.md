# Design a causally closed transient queue-control model

Type: task
Status: resolved
Blocked by: none

## Goal

Respond to the user's mathematical review by specifying a coherent physical
model, objective, admission mechanism, solution concept, and validation order
before production changes.

## Scope

- Write a reviewable model design and Proposed ADR-0011.
- Distinguish exact finite-system control from an unproved mean-field limit.
- Resolve lifecycle-versus-time accounting and requested/applied feedback.
- State concrete falsification fixtures and development-only implementation order.
- Preserve old specifications as history with explicit pending-revision pointers.

## Acceptance criteria

1. Queue state, observations, service/failure/cancellation jumps, and units are explicit.
2. All predicted outcomes derive from applied actions under the same admission mechanism.
3. Tail objective, operating costs, budgets, and realized resource audits are distinct.
4. A bounded reference solution and the additional requirements for an MFG claim are explicit.
5. No Python, test, configuration, historical artifact, or holdout data is changed.

## Progress log

### Update: 2026-09-05 — Design the transient queue-control reference

Status: completed

#### Goal

Specify a coherent mathematical model before resuming algorithm implementation.

#### Changed

- Added `transient-model-design.md`: exact two-queue state and conservation,
  causal observations, applied-action/admission dynamics, explicit failure
  jumps and cancellation, fractional CVaR plus lifecycle costs, resource/safety
  definitions, an 81-rule reference, independent qualification, and 12 fixtures.
- Added Proposed ADR-0011 with the exact prospective supersession boundary.
- Added a pending-redesign pointer to the old spec without rewriting its model.
- Paused ticket 03 with an explicit design-disposition blocker; no ticket-03
  production implementation was written.

#### Verification

- Read ADR-0005/0009/0010, domain and workflow documents and the active scopes;
  cross-checked queue accounting and the two sample CVaR definitions by hand.
- Consulted primary queueing, CVaR and mean-field/control papers linked in the design.
- `.\.venv\Scripts\python.exe -m unittest discover -s .\tests -v`
  -> `Ran 334 tests in 20.261s`, `OK`.
- `.\.venv\Scripts\python.exe -m mfg_hedge check --config .\configs\v1_minimal.json`
  -> exit 0, `status: ok`, Python 3.10.11; historical headroom warning retained.
- `rg` checked proposal/status pointers and confirmed exactly one claimed
  ticket during this design session. The acceptance fixtures are specified,
  not claimed as implemented or experimentally validated.

#### Artifacts

- `.scratch/mfg-hedge-attribution-study/transient-model-design.md`
- `docs/adr/0011-transient-queue-control-before-mean-field.md`
- Existing spec pending pointer and ticket 03 pause record.
- No production/test/config edits and no persistent experiment output.

#### Decisions and risks

ADR-0011 remains Proposed. The 81-rule reference, pooled mean-work reservation,
fractional CVaR and prospective work-ratio gate are material proposed choices.
They must not be presented as already accepted behavior or automatic MFG
evidence. Ticket 08 being resolved means the design is delivered; ticket 03
stays blocked until the design disposition explicitly reconciles its scope.

#### Next

Review/dispose the concrete ADR-0011 proposal, then revise the implementation
scope before touching the solver.

## Answer

Design delivered. Physical dynamics, objective, admission mechanism and solution
claims are explicit and independently reviewable. Existing 334 tests remain
green; mathematical approximation quality and policy benefit await development
validation. No new MFG effectiveness or equilibrium claim is made.

## Comments
