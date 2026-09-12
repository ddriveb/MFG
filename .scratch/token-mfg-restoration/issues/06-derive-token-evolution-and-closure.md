# Derive Token evolution and identify the mean-field closure obligations

Type: design
Status: resolved
Blocked by: none

## Scope

Following the user's 2026-09-07 instruction to begin derivation before
experiments, derive the exact finite Token queue game under ADR-0020 physics
and the `token_initial_price_v0` cost in ADR-0021. Write a separate proposed
mean-field extension and explicitly identify unresolved analytical obligations.
This ticket owns documentation only; ticket05 remains independently claimed
for T2. No production, configuration, campaign or accepted-ADR changes.

## Acceptance criteria

- Define the full physical state, admissible information and one-shot control.
- Define drift, arrival/admission, completion, timer, fault/Replay, FCFS,
  budget reset, settlement and common-noise boundary maps.
- Derive finite forward and tagged backward/entry equations using the same
  physical maps and the initial price-only cost, including running losers.
- Specify the candidate scale family, marked forward balance and FCFS
  selection rule; distinguish conditional closure from an existence or
  convergence theorem. Do not label an undefined service-start measure as
  a completed closure or silently accept the scale proposal.
- Record source-to-equation checks, exact counterexamples to coarse closure,
  and the remaining theory gate before any MFG experiment.
- Incorporate the user's permission to choose a discrete formulation: derive
  the exact event/holding-time kernel and Bellman recursion, and separate
  event discretization from a finite-state approximation and its error.
- Actually run full unittest discovery and the minimal configuration check;
  record observed outcomes. These are engineering regression, not MFG validation.

## Progress log

### Update: 2026-09-07 — Claim theory derivation

Status: partial

#### Goal

Derive the Token-player equations before choosing solvers or experiments.

#### Changed

Created this bounded documentation ticket; no implementation files changed.

#### Verification

Read project rules, CONTEXT, accepted ADR-0020/0021, restoration spec,
ticket05 and the arrival/fault/completion/timer/dispatch engine handlers.
Inspected working tree; unrelated parent-project changes are preserved.

#### Artifacts

This ticket; derivation document to follow.

#### Decisions and risks

Use the initial price-only utility and fixed external quote. Many-server
scaling remains a proposal. A formal forward/backward notation by itself
does not establish closure, equilibrium existence or a finite-system limit.

#### Next

Write the exact finite derivation and the conditional mean-field boundary problem.

### Update: 2026-09-07 — First derivation and event-kernel formulation complete

Status: completed

#### Goal

Make the original Token model's forward evolution and private decision equations
reviewable before experiments, choosing discrete/continuous methods by fitness.

#### Changed

- Added `token-evolution-derivation.md`:35 numbered equations covering state,
  information, admission, all physical maps, finite forward law, initial-cost
  continuation and arrival Bellman minimum, common-fault conditioning, candidate
  marked mean field/FCFS/budget closure and explicit remaining proof obligations.
- Following user steering, selected the exact event/holding-time transition
  kernel and Bellman recursion as the next formulation; finite-state reduction
  remains an explicit approximation, not an automatic Markov closure.
- Added an equation/source review and a deterministic algebra/quadrature check.
- No production files, configurations, accepted ADRs, version or historical
  experiment results changed by this ticket. Ticket05 ownership/status preserved.

#### Verification

- `.venv\Scripts\python.exe -m unittest discover -s tests -v`: exit0,
  **517 tests in158.756s, OK**.
- `.venv\Scripts\python.exe -m mfg_hedge check --config configs\v1_minimal.json`:
  exit0, **status ok**; historical load1.4 versus target.9 finding retained.
- `.venv\Scripts\python.exe .scratch/token-mfg-restoration/theory-r1-algebra-check.py`:
  exit0; event-kernel mass error1.11e-16, equal holding-time formulas, and
  running-Hedge-loser initial-cost identity checked. No simulation calls.
- Reviewed actual engine/source correspondence and information boundaries;
  see `theory-r1-review.md`. Engineering regressions are not MFG validation.

#### Artifacts

- `.scratch/token-mfg-restoration/token-evolution-derivation.md`
- `.scratch/token-mfg-restoration/theory-r1-review.md`
- `.scratch/token-mfg-restoration/theory-r1-algebra-check.py`
- `.scratch/token-mfg-restoration/theory-r1-algebra-check.json`
- `.scratch/token-mfg-restoration/theory-r1-full-suite.log`
- `.scratch/token-mfg-restoration/theory-r1-config-check.log`

#### Decisions and risks

Finite event physics and private-cost equations are specified. The MFG scale
extension remains proposed; forward well-posedness, sufficient numerical
information state, equilibrium existence and finite-K approximation are not
proved. Smooth D waiting may disappear under many-server pooling. Inert timer
logs can matter for arbitrary history-dependent policies, so a live-state-only
field requires an explicit restricted observation/policy class or extra memory.
No Nash, MFG, best-response estimation or experiment artifact was produced.

#### Next

Derive a computable information-state/transition representation and bound its
conditional action-cost error, using the exact event kernel as reference.

## Answer

This bounded first-derivation ticket is resolved. The detailed equations and
theoretical gaps are now reviewable. Resolution does not accept the proposed
scale family or declare the larger MFG theory complete.
