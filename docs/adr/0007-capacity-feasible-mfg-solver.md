# ADR-0007: Capacity-Feasible MFG Solver

Status: Accepted (confirmed by the user on 2026-09-04)

Date: 2026-09-04

Builds on: ADR-0001, ADR-0004, ADR-0005, ADR-0006

If accepted, this ADR supersedes ADR-0006's solver, price, and convergence clauses (decision 2) and its Hedge-budget clause (decision 3). ADR-0006's calibration separation, quota allocator, counter identities, load choice, and planned-versus-realized distinction remain in force. Supersession takes effect only after explicit user acceptance; until then ticket 04 remains blocked.

## Context

The ticket-03 solver prices only `expected_hedge_work` and updates the price with a fixed-step damped integral rule. Diagnostics on the real 70-cell calibration table at load 0.6 found:

1. H and D do not converge within the original 200 iterations.
2. Extending the original update to 5000 iterations converges H but not D.
3. A larger price step can drive Hedge demand to zero and make the iteration residuals vanish while total load still exceeds the target. This is a fixed point of the old update, not a capacity-feasible equilibrium.
4. `max(0, target*C - primary - replay)` hides overload when the current policy's Primary plus Replay work already exceeds the target.
5. The old price multiplies only Hedge work, so action-dependent Replay work does not receive the capacity signal.

There is also an intentional structural exception. At load 0.6, F has capacity 1.0 and target work rate 0.9, while Primary work rate is 1.2. F permits only Normal. It is therefore an exogenous transient overload shared by both arms, not a state in which this controller can produce a capacity-feasible equilibrium. Requiring F to be capacity-feasible would make the planned paired experiment impossible by construction.

## Decision

1. **Price total controllable incremental work.** For class `k` and action `a`:

   ```text
   incremental_work[k,a] = expected_hedge_work[k,a]
                           + expected_replay_work[k,a]
   J[k,a] = mean_latency[k,a]
            + gamma[k] * replay_probability[k,a]
            + price * incremental_work[k,a]
   ```

   Replay risk and Replay resource use remain distinct: `gamma*q` is the class-specific failure penalty, while `price*expected_replay_work` is the shared capacity signal.

2. **Use a signed capacity equation.** For each state:

   ```text
   target_work_rate   = target_max_utilization * state_capacity
   total_work_rate    = primary_work_rate + hedge_work_rate + replay_work_rate
   capacity_gap       = total_work_rate - target_work_rate
   capacity_violation = max(0, capacity_gap)
   capacity_slack     = max(0, -capacity_gap)
   ```

   A capacity-constrained equilibrium requires, within the stated tolerances:

   ```text
   price >= 0
   capacity_gap <= capacity_tolerance
   abs(price * capacity_gap) <= complementarity_tolerance
   ```

   Thus price zero is allowed below the target, while a positive price requires the capacity equation to be active. `capacity_gap`, not the non-negative `capacity_violation`, is the function used by root finding.

3. **Solve a deterministic inner policy/load fixed point at each fixed price.** Every independently evaluated price starts cold from the same state: all-Normal probabilities and `rho = primary_work_rate / state_capacity`. Warm starts from another price are forbidden because they make the result depend on bracket traversal. At fixed price, interpolate ActionStats at rho, compute the soft best response, apply the existing policy damping, and update rho from Primary+Hedge+Replay work until both conditions hold:

   ```text
   policy_residual < 1e-6
   load_residual   < 1e-6
   ```

   The inner cap is 2000 iterations. Reaching it returns `nonconverged`; tolerances are never relaxed and iterations are never silently extended.

4. **Use deterministic signed bracketing and bisection for H and D.** First solve at price zero. If `capacity_gap <= 1e-6`, accept price zero after the final checks. Otherwise, start the upper price at the configured `price_step` and double it at most 60 times until a converged inner solution has `capacity_gap <= 0`. Bisect the resulting positive/negative signed-gap bracket for at most 80 iterations. A positive-price result is accepted only when all of the following hold:

   ```text
   abs(capacity_gap) <= 1e-6
   abs(price * capacity_gap) <= 1e-6
   policy_residual < 1e-6
   load_residual < 1e-6
   ```

   Failure to obtain a bracket or meet these conditions is `nonconverged` with a specific reason such as `bracket_failed`, `inner_iteration_cap`, or `outer_iteration_cap`. A finite search cap alone is never evidence of mathematical infeasibility. `price_step` is only the initial positive bracket scale under this ADR; `price_damping` is retained in configuration/provenance for schema compatibility but is not consumed by the bisection solver.

5. **Treat grid clamp as a diagnostic boundary.** If any ActionStats lookup used by an inner or outer candidate would clamp outside the calibration grid, stop solving that state immediately. Return `nonconverged` with `grid_clamped=true` and the offending rho; do not continue searching with clamped estimates and do not emit an official solution.

6. **Use a narrow, auditable infeasibility claim.** In v1, `infeasible` is certified only when `primary_work_rate > target_work_rate + capacity_tolerance`, because all Hedge and Replay work is non-negative. `primary + replay` under the current policy is not called base-load infeasibility: Replay is action-dependent and protection may reduce it. If price bracketing fails without this Primary-only certificate, the status is `nonconverged`, not `infeasible`.

7. **Handle F as a forced policy, not a capacity equilibrium.** F runs no price search and fixes both classes to Normal. It still solves the Normal-only rho equation with the same load tolerance and grid rule. A successful F result has:

   ```text
   status = forced_normal
   price = 0
   forced_normal = true
   structural_capacity_violation = (capacity_gap > capacity_tolerance)
   hedge_budget_rate = 0
   ```

   At load 0.6 the structural violation is expected and reported. It is never described as a capacity-feasible MFG equilibrium. Failure of the Normal-only rho solve remains `nonconverged`, and a clamp remains disqualifying.

8. **Expose unambiguous statuses and diagnostics.** Each state reports exactly one `status` from `converged`, `forced_normal`, `infeasible`, or `nonconverged`, plus an orthogonal `grid_clamped` flag and a stable reason code. H/D official policies require `status=converged`; F requires `status=forced_normal`. The output includes price, rho, all work rates, signed capacity gap, violation, slack, policy/load/complementarity residuals, iteration counts for the inner and outer loops, the clamp location when applicable, and the final ActionStats.

9. **Recompute final self-consistency and close the quota contract.** From the final `(probabilities, rho, price)`, query ActionStats once, recompute the undamped soft best response and all work rates, and report:

   ```text
   policy_residual = max(abs(best_response[k,a] - probabilities[k,a]))
   load_residual   = abs(total_work_rate / state_capacity - rho)
   capacity_gap    = total_work_rate - target_work_rate
   complementarity_residual = abs(price * capacity_gap)
   hedge_budget_rate = max(0, target_work_rate
                              - primary_work_rate
                              - replay_work_rate)
   ```

   A converged H/D solution must additionally satisfy `hedge_work_rate <= hedge_budget_rate + 1e-6`. The quota projector consumes this final `hedge_budget_rate`; it still guarantees only planned expected Hedge work, while realized work is reported separately. F always supplies a zero Hedge budget.

10. **Define the official paired-run gate.** An official A/B run requires:

    - H and D: `status=converged`, all final residual checks passing, and `grid_clamped=false`;
    - F: exact all-Normal probabilities, `status=forced_normal`, a converged Normal-only rho calculation, and `grid_clamped=false`; its structural overload is disclosed rather than rejected;
    - R: reuse the accepted H policy.

    Any other result may be persisted only as diagnostics and must not be called an official MFG equilibrium or official A/B result.

11. **Keep scientific inputs fixed.** Load 0.6, the calibration table, replay penalties, CRN streams, event engine, action meanings, quota windows, and planned-versus-realized reporting remain unchanged. The 2000/60/80 caps are deterministic numerical safeguards for this proposed solver, not tuned scientific parameters.

## Consequences

- Capacity feasibility and complementary slackness become independently auditable; a zero Hedge budget can no longer hide total overload.
- Replay-heavy actions receive the same resource-price signal as Hedge-heavy actions while retaining their separate class penalty.
- F remains the intentionally overloaded failure interval used by both arms, without being mislabeled as a capacity-feasible equilibrium or permanently blocking the paired experiment.
- Whether H and D are feasible and converged is an implementation result to be verified against the real 70-cell table after this ADR is accepted; this ADR does not assume that outcome by construction.
- A failed bracket, inner solve, outer solve, or calibration-grid lookup stays a diagnostic rather than being promoted to an equilibrium claim.
