# Correct MFG capacity feasibility

Type: task
Status: resolved
Blocked by: none

## Goal

Diagnose the ticket-03 convergence failure honestly and design the capacity-feasible solver contract (Proposed ADR-0007). Documentation and diagnosis only: no Python production code, tests, configs, or artifacts; ticket 04 not started.

## Scope

- Record and explain the four reproduced phenomena (200-round non-convergence, 5000-round H-only convergence, price_step 0.5/1.0 fake convergence with rho above target, budget masking base overload, Hedge-only pricing).
- Compare design options A (more iterations / bigger steps), B (Hedge-only price + capacity feasibility reporting), C (shadow price on total incremental work).
- Define the new solver contract: total_capacity_residual, non-negative price with complementarity, deterministic inner fixed point at fixed price, deterministic outer root-finding with explicit termination, explicit infeasible status, the converged/infeasible/nonconverged/grid_clamped taxonomy, the official A/B acceptance rule, and final self-consistency recomputation.
- Spec section 12 gets a pending-revision note pointing to ADR-0007; ADR-0006's editorial "remains Proposed" sentence is corrected without touching its accepted decisions.

## Acceptance criteria

- Diagnostic numbers reproduced and recorded with real runs.
- ADR-0007 exists with Status: Proposed and explicitly states it will supersede ADR-0006's solver/price/convergence clauses upon user acceptance.
- Full suite and `check` pass (zero code change).

## Progress log

### Update: 2026-09-04 — Diagnose convergence and design the capacity-feasible solver

Status: completed

#### Goal

Reproduce and explain the ticket-03 convergence failure, compare design options, and produce Proposed ADR-0007 with the capacity-feasible solver contract. Documentation and in-memory diagnosis only; no production code, tests, configs, or artifacts changed; ticket 04 untouched.

#### Changed

- `docs/adr/0007-capacity-feasible-mfg-solver.md` (Proposed): total-incremental-work shadow pricing (`hedge + replay` in J), explicit `total_capacity_residual`, non-negative price with complementarity, deterministic inner policy/rho fixed point at fixed price, deterministic outer bracket+bisection with explicit termination, the `converged`/`infeasible`/`nonconverged`/`grid_clamped` taxonomy, official A/B acceptance rule, final self-consistency recomputation, and separate base-overload reporting. States that upon acceptance it supersedes ADR-0006's solver/price/convergence and Hedge-budget clauses.
- `docs/adr/0006-...`: editorial fix only — the revision note no longer says "Status remains Proposed"; decisions untouched.
- `.scratch/mfg-hedge-paired-comparison/spec.md`: section 12 gains a pending-revision note pointing to ADR-0007; the ticket-03 contract stays normative until then.

#### Verification

- Diagnostics on the real 70-cell calibration table (all in-memory, no artifacts):
  - **Phenomenon 1 reproduced** (load 0.6, default 200): H converged=false (price 0.2267, hedge 0.7171 vs budget 0.6000), D converged=false (price 0.8642, hedge 0.5891 vs budget 0.1500), F converged.
  - **Phenomenon 2 reproduced** (5000 iters, original step): H converged at 1530 iterations (price 0.8894, rho exactly 0.9000, hedge 0.6000 = budget — a clean fixed point); D still non-converged after 5000 (price 21.93, hedge 0.5879 vs budget 0.1496).
  - **Phenomenon 3 reproduced and explained**: big steps alone (0.5/1.0, 200 iters) do NOT converge D (contrary to the first report); big steps plus 5000 iterations mark D converged at 453 iterations with price 52.53, hedge_work ~1e-6, budget 0, rho 1.1040 — the degenerate "converged but capacity-infeasible" fixed point (the reported rho 0.9545 is the same mechanism under slightly different settings: price explodes, hedge demand dies, residuals vanish while total load stays above target). Big steps also make H oscillate (residuals ~2e-02 after 5000 iters).
  - **Phenomena 4/5 confirmed by construction**: `max(0, target*C - primary - replay)` hides base overload; the price multiplies only `expected_hedge_work`.
- `.\.venv\Scripts\python.exe -m unittest discover -s .\tests -v` — `Ran 251 tests ... OK` (zero code change).
- `.\.venv\Scripts\python.exe -m mfg_hedge check --config .\configs\v1_minimal.json` — exit 0.

#### Artifacts

None (diagnostics were in-memory). Documents: ADR-0007 (new, Proposed), ADR-0006 (editorial note), spec section 12 note, this ticket.

#### Decisions and risks

- Recommended option C (total-work shadow price) makes the D fixed point capacity-feasible by construction at load 0.6 and prices Replay work; options A/B were rejected: A hides the semantics behind iteration budgets, B keeps the unpriced-replay loophole.
- The recommended contract was designed against the recorded diagnostics; no parameter was tuned to force a "has Hedge" outcome.
- ADR-0007 remains Proposed; implementation must not start before user acceptance.

#### Next

Wait for the user to accept or reject ADR-0007. Ticket 04 remains open and blocked on the solver-convergence question.

## Answer

All four phenomena reproduced with real runs and explained; option comparison recorded; ADR-0007 (Proposed) defines the capacity-feasible solver contract; spec and ADR-0006 updated without touching accepted decisions; 251/251 tests and `check` pass with zero code change; ticket 04 not started.

## Comments

