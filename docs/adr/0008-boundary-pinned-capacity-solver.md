# ADR-0008: Boundary-Pinned Capacity Solver and First-Round Load 0.5

Status: Accepted (confirmed by the user on 2026-09-04)

Date: 2026-09-04

Builds on: ADR-0001, ADR-0004, ADR-0005, ADR-0006, ADR-0007

This ADR supersedes ADR-0007's nested fixed-point clauses (the inner policy/load iteration in decision 3 and the bracket/bisection mechanics in decision 4) and broadens its infeasibility certificate (decision 6). It supersedes ADR-0006's first-round load choice (0.6 becomes 0.5). ADR-0007's capacity accounting (signed gap, violation, slack, complementarity), status taxonomy, final self-consistency rule, forced-Normal F exception, quota closure, and official gate remain in force. The user accepted this supersession on 2026-09-04.

## Context

The ADR-0007 implementation (ticket 12) is unit-green but failed the real 70-cell gate honestly:

1. In H, the coupled policy/load map is multi-valued and non-contractive near the capacity boundary: bisection midpoints at price 0.875 vs 0.8875 returned totals 1.8301 vs 1.4410, and the inner iteration hit the 2000 cap at price 0.88359375. The signed-gap bisection's monotonicity premise fails inside a nested rho fixed point.
2. In D at load 0.6, even maximal pricing leaves total work at 1.4318 > target 1.35. User-verified on the independently rebuilt table: at rho = 0.9 the minimal incremental work is 0.19356 (Regular) / 0.21409 (Urgent), weighted lower bound 0.19766, giving D a strict feasibility upper load bound of ≈ 0.5636. Load 0.6 is genuinely infeasible; load 0.55 sits too close to the boundary; load 0.5 has clear headroom with meaningful D protection at the capacity root (≈ 36.5% Regular / ≈ 99.98% Urgent Hedge).

Masking the multi-valuedness with continuation or rho damping would hide, not solve, the problem; tuning parameters to force a "has Hedge" outcome is forbidden.

## Decision

1. **Split the solve at the capacity-binding diagnostic.** The price = 0 solve is the ADR-0007 inner discipline exactly: initial policy all-Normal, initial `rho = primary_work_rate / state_capacity`, damped soft best response with the configured `policy_damping`, cap 2000 iterations, and the final undamped policy/load residuals (< 1e-6) as the stop rule. If this solve fails, the state is `nonconverged` (`inner_iteration_cap`) — failing the price-0 solve never counts as evidence that capacity binds. If it succeeds and `capacity_gap <= 1e-6`, accept price 0 (slack case). Only a *successful* price-0 solve with `capacity_gap > 1e-6` enters the boundary-pinned branch: pin `rho = target_max_utilization` (the boundary value), interpolate ActionStats once at that rho, and reduce the problem to a one-dimensional price root — `policy(price) = softmax(-eta * J(price))` at fixed stats, with `total_work_rate(price)` monotone non-increasing in price. Bisect the signed `capacity_gap(price)` (same [price_step doubling ≤ 60] bracket and ≤ 80 iteration caps). No nested rho fixed point ever runs inside the price search.
2. **Policy-aware feasibility certificate.** At the boundary rho, compute the per-class minimal incremental work `min_a (expected_hedge_work + expected_replay_work)[k,a]`; a state is `infeasible` iff `primary_work_rate + arrival_rate * sum_k weight_k * min_a incremental[k,a] > target_work_rate + capacity_tolerance`. This certifies D at load 0.6 as infeasible and reproduces the independently computed 0.5636 upper load bound, while remaining narrower than any heuristic: it is the exact policy-minimum at the boundary.
3. **First-round load is 0.5.** The paired-comparison configuration (ticket 04) uses `healthy_offered_load = 0.5`; loads 0.6/0.7 remain stress cases. At 0.5, the D boundary solve yields meaningful protection (≈ 36.5% Regular, ≈ 99.98% Urgent Hedge) and H keeps clear slack.
4. **Unchanged from ADR-0007**: F stays forced-Normal with disclosed structural overload; the status taxonomy (`converged` / `forced_normal` / `infeasible` / `nonconverged` + orthogonal `grid_clamped`), final self-consistency recomputation, solver-to-quota budget closure, and the official gate (H/D converged + feasible + unclamped; F forced-Normal + unclamped) remain in force. Grid clamp stays disqualifying (the boundary rho = 0.9 is inside the {0.5..1.3} grid, so a binding solve never clamps; a non-binding solve that wanders off-grid still fails fast).

## Consequences

- The price root is one-dimensional and monotone by construction; no multi-valued rho map participates in the root find.
- Infeasibility is proven before searching and matches the independently verified load bound.
- The first-round experiment gains honest D headroom without parameter tuning.
- Whether the boundary-pinned solve passes the real 70-cell gate is an implementation result, verified after acceptance; this ADR does not assume it.
