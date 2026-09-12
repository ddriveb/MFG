# Correct ADR-0007 review findings

Type: task
Status: resolved
Blocked by: none

## Goal

Correct the capacity-feasible solver design before acceptance: make the outer root use a signed capacity gap, state the actual complementarity conditions, handle the structurally overloaded forced-Normal F state without permanently blocking the paired experiment, and make infeasibility, grid-clamp, final-residual, and quota-closure rules executable. Documentation only; no Python production code, tests, configurations, or experiment artifacts.

## Scope

- Revise Proposed ADR-0007 without accepting it or changing production behavior.
- Distinguish signed capacity gap, violation, and slack; define valid complementarity and bisection conditions.
- Reserve mathematical infeasibility for a certified condition rather than failure to find a price bracket.
- Treat F as a forced-Normal transient with an explicit structural-capacity-violation diagnostic and an official-run exception that does not claim an F-state equilibrium.
- Specify deterministic inner initialization, bounded outer search, fail-fast grid-clamp behavior, final self-consistency residuals, and solver-to-quota budget closure.
- Update the spec's pending-revision note so ticket 04 remains blocked until the corrected ADR is accepted and implemented.

## Acceptance criteria

- ADR-0007 remains `Proposed` and contains no claim that D is feasible before implementation verifies it on the real 70-cell table.
- The root-finding function can change sign; complementarity rejects positive price with unused capacity.
- F no longer makes the official A/B gate mathematically impossible, and its structural overload is never described as a capacity-feasible equilibrium.
- A finite price-search cap alone cannot produce `infeasible`; grid-clamped estimates cannot produce an official solution.
- Final policy/load/capacity residuals and the derived Hedge budget are defined from one final state.
- Full unit suite and `check` pass with no code/config/artifact change.

## Progress log

### Update: 2026-09-04 — Correct ADR-0007 review findings

Status: completed

#### Goal

Make the Proposed capacity-feasible solver contract executable and scientifically honest before acceptance, without changing Python behavior or starting ticket 04.

#### Changed

- Rewrote `docs/adr/0007-capacity-feasible-mfg-solver.md` while keeping `Status: Proposed`: total Hedge+Replay resource pricing; signed `capacity_gap`; separate violation/slack; valid feasibility and complementary-slackness checks; cold-start fixed-price inner solve; signed bracket+bisection; explicit caps and stable failure reasons; fail-fast grid-clamp boundary; narrow Primary-only infeasibility certificate; forced-Normal F semantics; final undamped self-consistency residuals; final Hedge-budget derivation and quota closure; corrected official A/B gate.
- Replaced the impossible requirement that F be capacity-feasible. F now solves only its fixed-Normal rho, reports its expected structural overload, and is admitted to an official paired run only as a disclosed shared transient—not as an MFG equilibrium.
- Removed the unverified claim that D is feasible by construction. H/D feasibility is now an implementation result that must be demonstrated on the real 70-cell table after ADR acceptance.
- Updated `.scratch/mfg-hedge-paired-comparison/spec.md` section 12's pending-revision note to summarize the corrected proposed contract and keep ticket 04 blocked pending acceptance and implementation.

#### Verification

- `.\.venv\Scripts\python.exe -m unittest discover -s .\tests -v` — `Ran 251 tests ... OK`.
- `.\.venv\Scripts\python.exe -m mfg_hedge check --config .\configs\v1_minimal.json` — exit 0, `status: ok`.
- `rg` contract audit confirmed ADR-0007 remains Proposed; root finding uses signed `capacity_gap`; F has `forced_normal` and `structural_capacity_violation`; bracket failure remains nonconverged; clamp is disqualifying; final Hedge budget and quota inequality are present; the former “D becomes feasible by construction” claim is absent.

#### Artifacts

Documentation only: `docs/adr/0007-capacity-feasible-mfg-solver.md`, `.scratch/mfg-hedge-paired-comparison/spec.md`, and this ticket. No experiment artifacts.

#### Decisions and risks

- Proposed numerical safeguards are inner cap 2000, bracket expansion cap 60, bisection cap 80, and `1e-6` policy/load/capacity/complementarity tolerances; these are reproducibility safeguards, not tuned scientific inputs.
- `price_step` becomes only the initial bracket scale; legacy `price_damping` remains serialized for compatibility but is unused by the proposed bisection solver.
- V1 calls H/D infeasible only when Primary work alone exceeds target work. A finite bracket cap or policy-dependent Primary+Replay overload is not an infeasibility proof.
- ADR-0007 is not accepted by this documentation correction. Existing ticket-03 behavior remains normative until explicit user acceptance and a separate red-green implementation ticket.

#### Next

User reviews and explicitly accepts or rejects the corrected ADR-0007. If accepted, create a separate solver-implementation correction ticket; ticket 04 remains blocked until that implementation passes real-table verification.

## Answer

ADR-0007's four blocking findings are corrected without changing code: the solver now has a signed root and valid complementarity contract, F is an explicit forced-Normal structural-overload exception, infeasibility and clamp claims are bounded, and the final state closes the quota budget. All 251 tests and `check` pass; ADR-0007 remains Proposed and ticket 04 remains unstarted.

## Comments
