# Close ADR-0008 execution contracts

Type: task
Status: resolved
Blocked by: none

## Goal

Two narrow documentation corrections before ADR-0008 acceptance: (1) define the price=0 slack-path inner solve precisely (init, damping, cap, residual rule, and the no-binding-inference rule); (2) change the remaining load-0.6 references to 0.5 in the spec and ticket 04, noting 0.6/0.7 as stress diagnostics only.

## Scope

- `docs/adr/0008-boundary-pinned-capacity-solver.md`: exact price=0 solve contract.
- `.scratch/mfg-hedge-paired-comparison/spec.md`: two load references.
- `.scratch/mfg-hedge-paired-comparison/issues/04-...md`: load reference.
- ADR-0008 stays Proposed; no code changes.

## Acceptance criteria

- The price=0 contract is unambiguous and matches the ADR-0007 inner-solve discipline.
- No load=0.6 references remain in spec/ticket 04 except as stress diagnostics.
- Full suite and `check` pass unchanged.

## Progress log

### Update: 2026-09-04 — Close ADR-0008 execution contracts

Status: completed

#### Goal

Close the two review gaps before ADR-0008 acceptance: the exact price=0 slack-path contract and the remaining load-0.6 references.

#### Changed

- `docs/adr/0008-boundary-pinned-capacity-solver.md` decision 1: the price=0 solve is now the exact ADR-0007 inner discipline — all-Normal initial policy, `rho = primary_work_rate / capacity`, configured `policy_damping`, cap 2000, final undamped policy/load residuals < 1e-6; failure is `nonconverged` (`inner_iteration_cap`) and never counts as evidence that capacity binds; only a successful price-0 solve with `capacity_gap > 1e-6` enters the rho=0.9 boundary-pinned branch. No continuation or rho damping anywhere.
- `.scratch/mfg-hedge-paired-comparison/spec.md`: three load references now read 0.5 (Option B, confirmation item 1, parameter positioning), with 0.6/0.7 noted as stress diagnostics only.
- `.scratch/mfg-hedge-paired-comparison/issues/04-...md`: config scope now reads `healthy_offered_load = 0.5` with the stress-diagnostic note.

#### Verification

- `grep` audit: no load-0.6 references remain in the spec or ticket 04 outside stress-diagnostic notes.
- `.\.venv\Scripts\python.exe -m unittest discover -s .\tests -v` — `Ran 256 tests ... OK` (zero code change).
- `.\.venv\Scripts\python.exe -m mfg_hedge check --config .\configs\v1_minimal.json` — exit 0.

#### Artifacts

Documentation only: ADR-0008 (decision 1), spec, ticket 04, this ticket.

#### Decisions and risks

- ADR-0008 remains Proposed; the contract is now unambiguous about the slack path (no continuation, no rho damping, no binding inference from a failed price-0 solve).

#### Next

User confirms ADR-0008; then the final solver revision ticket against the real 70-cell table, then ticket 04.

## Answer

Both review findings are closed: the price=0 slack-path contract is now exact, and all load references read 0.5 with 0.6/0.7 as stress diagnostics. 256/256 tests and `check` pass with zero code change.

## Comments

