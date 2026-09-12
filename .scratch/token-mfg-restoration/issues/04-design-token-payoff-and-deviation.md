# Design Token private payoff and finite deviation semantics

Type: design
Status: resolved
Blocked by: none

## Goal

Prepare Proposed ADR-0021 for the user's review before implementing T2.
T1 tickets 02/03 are resolved. This ticket produces a bounded design for
`token_payoff.py` and `token_deviations.py`, not their implementation.

## Scope

Decide named initial/runtime/extended private costs, all component units and price
bases, physical settlement, requested/applied action treatment, exactly one
Token override, opponent policy lifecycle, CRN full reruns and pathwise-only
evidence labels. Conditional continuation sampling, expected regret, BR,
Nash/MFG, endogenous price, solver and campaigns remain outside T2.

## Acceptance criteria

1. ADR-0021 is Proposed; ADR-0020's accepted T1 boundary stays unchanged.
2. Preserve three independent model IDs: initial price-only, ADR-0009 runtime,
   and extended reservation. Parameter degeneration cannot replace provenance.
   Every formula, parameter, price unit, raw quantity and overlap is explicit.
3. Specify full running-loser settlement and fail-closed incomplete results,
   rejected D/I->actual N, admitted-but-voided Delayed and Replay charging.
4. Override exactly one requested action; freeze other policy functions and
   private key rules, not observations/admissions/action trajectories.
5. Require fresh run-local policy/ledger state, baseline invocation at the
   target before output override, complete system rerun, stable external CRN
   and explicit exogenous price provenance.
6. Define typed record/API proposals, sign of pairwise pathwise differences,
   calls/deduplication limits, validation failures and minimum regression matrix.
7. Prohibit realized-path argmin/BR/regret/Nash labels; defer conditional
   expectations and regret to independent continuation designs.
8. Verify source anchors, document links, status and source/config/test hashes;
   production modules, historical results and experiment authorization unchanged.
9. Actually run the full unittest suite and config check, and append their
   real results to this ticket. Hash checks do not replace these final checks.

## Proposed decision summary

The concrete decision is in
[ADR-0021](../../../docs/adr/0021-token-private-cost-and-deviation-semantics.md).
Its status remains Proposed until user confirmation.

- Retain `token_initial_price_v0`, `token_runtime_adr0009_v1`, and
  `token_extended_reservation_v1`. Require an explicit model ID, even when
  runtime parameters produce the same number as the initial price-only model.
- Initial charges latency, Replay and an external H/R execution-work quote;
  it has no fixed work or waste penalty. ADR-0009 runtime adds persistent
  incremental-work and waste penalties. Extended adds SLO excess, charges all
  executed work, and uses an admitted-reservation quote. All three retain raw
  work components. Price value alone is insufficient without its unit/basis.
- Settle after every running loser has terminated. Refused D/I follows actual
  N; Replay and failed executed work still cost money. Admitted Delayed retains
  reservation even if its timer is voided.
- One episode, one stable Token key, one requested-action override. Invoke the
  base source once at that arrival before overriding its returned action.
  Every branch gets fresh source/ledger/engine state; other policy functions
  and private key rules remain fixed while their actual actions may change.
- Full same-CRN reruns, canonical action rows and signed pathwise cost/gain
  differences only. No hindsight minimum, BR, regret, Nash or MFG output.
- T2 price is a frozen episode-constant quote, visible to policy and scorer;
  no endogenous feedback. Conditional continuation samples belong to T3+.

## Proposed implementation boundary after ADR acceptance

Create a separate implementation ticket for `token_payoff.py` and
`token_deviations.py`, their tests and only the required compatible adapters.
The intended interfaces and eighteen regression fixtures are in ADR sections8/9.
Legacy T1/engine inputs and results must remain compatible. Price observation
can use a runner-owned immutable observation adapter rather than changing
queue physics. No campaign, CLI, pooled expected-regret estimator, solver,
continuation generator or private CVaR optimizer is part of this slice.

## Design verification plan

Check referenced files and AST source symbols, Proposed/Accepted status
boundaries, strict one-Token/pathwise evidence clauses, and hand arithmetic for
the named formulas. Compare src/tests/configs plus Accepted ADR-0020 against
the design-start SHA256 snapshot. Confirm neither proposed production module
has been created. Additionally execute:

    .venv\Scripts\python.exe -m unittest discover -s tests -v
    .venv\Scripts\python.exe -m mfg_hedge check --config configs\v1_minimal.json

Record actual counts, duration, exit status and config findings. The expected
current baseline is517 tests, not a substitute for observing the result.

## Progress log

### Update: 2026-09-06 — claim T2 design

Status: partial

#### Goal

Prepare the private-cost and finite-deviation decision for review.

#### Changed

Created and claimed this design-only ticket. No production module changed.

#### Verification

Read project rules, CONTEXT, Accepted ADR-0020 T1 boundary, ADR-0009,
current token_online.py, historical cost formula and resolved T1 tickets.

#### Artifacts

This ticket; Proposed ADR-0021 to follow.

#### Decisions and risks

User explicitly requires ADR confirmation before implementation. Completing
the design ticket will not mean the proposed ADR has been accepted.

#### Next

Write exact formulas and pathwise deviation contracts, then verify the design.

### Update: 2026-09-06 — T2 private-cost/deviation proposal completed

Status: completed

#### Goal

Deliver a concrete Proposed ADR for Token private costs and finite pathwise
deviations before implementation, addressing all nine user-requested decisions.

#### Changed

- Added `docs/adr/0021-token-private-cost-and-deviation-semantics.md` with two
  named formulas, units, exogenous constant-price visibility/settlement rules,
  lifecycle completion and admission cases, one-request override semantics,
  policy-state/CRN contracts, pathwise-only outputs and eighteen future tests.
- Expanded this design ticket with decisions and a bounded future T2 interface.
- Updated only the restoration spec's status/pointer paragraph to reflect
  Accepted T1 and separately Proposed T2; ADR-0020 itself was not changed.
- Recorded design-start protected-file hashes and document verification under
  the feature scratch directory. No production implementation was added.

#### Verification

Executed a Python document/AST/hash/arithmetic check through the project venv:

- Three Markdown documents: local links and code-fence structure valid.
- Seven referenced source symbols exist in the inspected source.
- All82 protected src/tests/config files plus the accepted-ADR coverage in
  the snapshot were checked; changed_protected_files=[] (82 total files).
- Three hand-formula fixtures passed: Hedge winner with running Primary loser
  (original4.5, extended6.5), refused Hedge followed by Normal Replay
  (15.5,22), admitted-but-voided Delayed (1,2.5), with declared fixture inputs.
- ADR-0021 remains Proposed; ADR-0020 remains Accepted for T1 only.
- `token_payoff.py` and `token_deviations.py` do not exist.
- No production regression test or formal campaign was run in this design
  turn. Formula verification does not claim T2 execution tests passed.

#### Artifacts

- `docs/adr/0021-token-private-cost-and-deviation-semantics.md`
- `.scratch/token-mfg-restoration/issues/04-design-token-payoff-and-deviation.md`
- `.scratch/token-mfg-restoration/spec.md` (status/pointer only)
- `.scratch/token-mfg-restoration/t2-design-baseline.json`
- `.scratch/token-mfg-restoration/t2-design-verification.json`

#### Decisions and risks

The proposed original model is explicitly the ADR-0009 runtime formula; the
extended model adds SLO/all-work and uses a different price basis. They cannot
be silently mixed. T2 quotes are external constants, not price equilibrium.
Only one Token's requested action is overridden; opponents keep functions,
not trajectories. Pairwise realized gains remain pathwise, never expected
regret. Private cost waits for complete loser settlement. This ticket's
resolved status means design delivery, not ADR acceptance or implementation.

#### Next

Await the user's confirmation of ADR-0021, then create a separate bounded T2
implementation ticket for the two modules and the declared regression matrix.
This sequencing is explicitly requested by the user.

### Update: 2026-09-06 — reopen for initial-model provenance and final verification

Status: partial

#### Goal

Apply the user's two narrow corrections: distinguish the true initial Token
price-only model from ADR-0009, and execute the full final verification.

#### Changed

Reopened ticket04 as claimed, updated current acceptance criteria and model
summary. Earlier Progress log entries remain historical; their use of
"original" for ADR-0009 and document-only completion is superseded by this
review correction. ADR-0021 remains Proposed; no T2 implementation is authorized.

#### Verification

Inspected current ADR/ticket/spec and working tree. Full suite and config check
will be run after the narrow document changes; results are not yet claimed.

#### Artifacts

This ticket and the revised ADR/spec; verification logs to follow.

#### Decisions and risks

The initial model is an independent provenance identity, not an alias inferred
from zero runtime coefficients. All existing deviation/CRN/settlement and
pathwise-only boundaries remain unchanged.

#### Next

Revise the three-model specification and execute both required commands.

### Update: 2026-09-06 — three model identities corrected and full verification completed

Status: completed

#### Goal

Complete exactly the user's two narrow review corrections without changing
the accepted deviation/CRN/settlement/claim boundary or implementing T2.

#### Changed

- Replaced the erroneous proposed `token_original_runtime_v1` identity with
  THREE explicit model IDs in ADR-0021 and the ticket/spec's current summaries:
  `token_initial_price_v0`, `token_runtime_adr0009_v1`, and
  `token_extended_reservation_v1`.
- Initial is exactly L+gamma_k R+p_exec(W_H+W_R), with no fixed redundant-work
  or waste penalty. Runtime is the later ADR-0009 fixed work/waste extension;
  extended reservation keeps its prior formula unchanged.
- Zero c_inc/c_waste may make runtime numerically equal to initial, but never
  change model ID, provenance or research interpretation. The withdrawn ID
  must be rejected rather than retained as an alias in future T2 code.
- Updated model-specific units/parameter types, accounting explanation and
  regression matrix; preserved all requested-action, target-policy-call,
  same-CRN full-rerun, fixed opponent-function, settlement, pathwise-only and
  external constant-price semantics.
- Updated final design-verification requirements and actually executed them.
  Earlier progress entries are preserved as history; this entry corrects their
  ADR-0009-as-original naming and insufficient document-only final verification.

#### Verification

Actually executed after the narrow design changes:

    .venv\Scripts\python.exe -m unittest discover -s tests -v

Result: exit code0; `Ran 517 tests in 155.971s`; `OK` (517/517 passed).
Full output: `.scratch/token-mfg-restoration/t2-review-r1-full-suite.log`.

    .venv\Scripts\python.exe -m mfg_hedge check --config configs\v1_minimal.json

Result: exit code0; `status: ok`; Python3.10.11. The existing minimal-config
finding remains: single-domain failure base load1.400 exceeds target0.900;
this is not a new failure or a change to the restored Token reference.
Full output: `.scratch/token-mfg-restoration/t2-review-r1-config-check.log`.

Additional checks: local document links and three model sections valid;
three hand-formula fixtures include initial/runtime/extended costs
(1.5,4.5,6.5), (13,15.5,22), (1,1,2.5). Equal numeric initial/zero-runtime
costs retain distinct identity receipts. All82 protected source/test/config
and accepted-ADR files match the design-start hashes. Hashes supplement, not
replace, the regression/config commands above. Neither T2 production module
exists. No formal campaign, best response, regret, Nash or MFG run was made.

#### Artifacts

- `docs/adr/0021-token-private-cost-and-deviation-semantics.md`
- `.scratch/token-mfg-restoration/issues/04-design-token-payoff-and-deviation.md`
- `.scratch/token-mfg-restoration/spec.md` (three-model pointer)
- `.scratch/token-mfg-restoration/t2-review-r1-full-suite.log`
- `.scratch/token-mfg-restoration/t2-review-r1-config-check.log`
- `.scratch/token-mfg-restoration/t2-review-r1-design-verification.json`
- `.scratch/token-mfg-restoration/t2-review-r1-final-verification.json`

#### Decisions and risks

ADR-0021 stays Proposed pending the user's explicit acceptance. Resolved means
the narrow design corrections and required final verification are complete;
it does not authorize T2 implementation or reinterpret the initial model.

#### Next

User review/acceptance of revised ADR-0021, followed by a separate bounded T2
implementation ticket. No additional model or execution change in this revision.

## Answer

The requested scheme now preserves initial price-only, later ADR-0009 runtime,
and extended reservation as three independent model/provenance identities.
Both review corrections are complete: all517 regression tests and the config
check actually passed, with full logs and results appended above. ADR-0021
is now Accepted by the user on 2026-09-06; no T2 implementation, conditional
continuation, formal regret or MFG work was performed.

### Update: 2026-09-06 — ADR-0021 accepted and T2 implementation handed off

Status: completed

#### Goal

Record the user's acceptance of the three-model Token private-cost and finite
pathwise-deviation design and hand implementation to a separate ticket.

#### Changed

- ADR-0021 status is now `Accepted (confirmed on 2026-09-06)`.
- Created and claimed [T2 implementation ticket](05-implement-token-payoff-and-deviations.md)
  for `token_payoff.py`, `token_deviations.py`, strict provenance, complete
  loser settlement, single requested-action interventions, same-CRN reruns,
  and the 18-class regression matrix.

#### Verification

- Confirmed no mathematical or process blocker was declared by the user.
- Confirmed no T2 production module, predictor, price-feedback loop,
  best-response/regret evaluator, forward model, MFG run, or formal campaign
  was started during this handoff.
- Confirmed only the new T2 implementation ticket is `claimed`.

#### Artifacts

None beyond the accepted ADR and implementation-ticket records.

#### Decisions and risks

Acceptance is limited to ADR-0021's finite pathwise cost/gain boundary. It
does not turn pathwise gains into BR, regret, Nash, or MFG evidence and does
not authorize continuation prediction or equilibrium experiments.

#### Next

Implement and test T2, then stop at finite pathwise deviation evidence.
