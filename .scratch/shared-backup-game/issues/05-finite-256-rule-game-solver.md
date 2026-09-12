# Finite Pi256 rule-game solver and frozen validation protocol

Type: implementation
Status: resolved
Blocked by: 04

## Scope

Implement only the bounded Stage 4 finite `Pi_256` symmetric pure-strategy
best-response solver and its predeclared fit/validation protocol. The solver
must retain all starts and all negative outcomes. It may report a pure fixed
point candidate, a rule cycle, a 32-round non-convergence, or insufficient
statistical precision, but it must not claim Nash or MFG equilibrium.

The solver is an in-memory development API. It does not implement an
unrestricted feedback policy, mixed equilibrium, conditional particle
mean-field solver, tariff/market equilibrium, centralized planner, N-size
campaign, qualification/holdout gate, CLI, or post-hoc sample extension.

## Frozen protocol

The following values are binding before any fit or validation simulation:

- Primary system: `N=8`, `c_B=0.5`, degraded slowdown `2.0`, hedge delay
  `1.5`, zero tariff, the existing four-action `N/D/S/X` bank, and the
  accepted Stage 1/2 observation and physical contracts.
- Fit scenarios: 16 independent common fault paths in namespace
  `shared-backup-game:v1:stage4-fit`, macro seed `20260905`, common episode
  IDs `0..15`; exactly two independent nested idiosyncratic populations per
  common path, population IDs `0` and `1`. All three starts and every
  candidate deviation use the same immutable fit scenario set.
- Validation scenarios: 32 new common fault paths in namespace
  `shared-backup-game:v1:stage4-validation`, macro seed `20260906`, common
  episode IDs `0..31`; exactly two independent nested populations per path,
  population IDs `0` and `1`. No fit path, population, or trace fingerprint
  may be reused as validation data.
- Starts: exactly `NNNN`, `NSSN`, and `XXXX`, each run independently and
  retained in input order. Each start has at most 32 best-response updates;
  no extra restart is allowed.
- Per update: rebuild the current population environment from the frozen fit
  scenarios, enumerate all 256 canonical rules for the representative
  Expert, select by exact objective, then exact executed work, then canonical
  lexical order `N<D<S<X`. No epsilon tie is permitted.
- Validation: freeze the selected fit candidate before validation, then run
  all 8 Experts against all 256 deviations on the independent validation
  scenarios. Recompute the complete finite shared scheduler, online policy
  calls, budgets, queues, completions, Replay, timers, and peer/social costs.
  Validation cannot change the candidate rule.

## Frozen precision rule

Common fault paths are the independent statistical clusters. Tokens, Experts,
and queues inside one path are not treated as independent observations. For a
paired candidate-minus-incumbent difference, first aggregate each of the two
population traces within its common-path cluster, then compute the ordinary
cluster sample mean and standard error over the fixed paths.

The predeclared normalized precision target is `0.005 * J_NNNN^0` for the
same data split and Expert, where the positive NNNN denominator is computed
from the same immutable scenarios. Fit precision is `sufficient` only when:

1. the denominator is finite and positive;
2. every retained best-response comparison has cluster standard error at or
   below that target; and
3. the selected rule's exact point-estimate gap over the runner-up exceeds
   twice the paired cluster standard error.

Otherwise retain `statistics_insufficient` alongside the dynamics outcome;
do not append paths, populations, starts, or rounds. Validation reports the
same fixed precision diagnostic, but a final simultaneous one-sided 95%
uncertainty bound is a separate required gate and is not implied by this
protocol. Until that bound exists, the only allowed label is `restricted-game
candidate` or `candidate fixed point`.

## Frozen computation budget

- Hard limit: at most `1,000,000` complete `simulate_shared_backup`
  invocations across fit forward runs, all fit deviations, validation
  forward runs, and all validation deviations. The first budget exhaustion
  is retained as a failure/partial outcome; no automatic sample extension is
  allowed.
- Planned upper bound under this protocol is `789,504` fit calls
  (`3 starts * 32 rounds * 32 scenario traces * (1 forward + 256
  deviations)`) plus `131,584` validation calls
  (`32 paths * 2 populations * 8 Experts * (1 baseline + 256 deviations)`),
  leaving a fixed reserve below the hard limit for candidate replay and
  invariant checks.
- Record elapsed runtime and call counts in memory. Do not turn runtime into
  a post-hoc reason to add data.

## Acceptance criteria

1. The exact canonical 256-rule bank is validated before iteration; all three
   starts are retained, and the result distinguishes pure fixed point, rule
   cycle with period and first repeat, 32-round non-convergence, and
   statistical insufficiency without collapsing one into another.
2. Every fit update reruns the current N=8 shared environment and online
   policies from immutable CRN traces. No baseline pool, speed, completion,
   quota, queue, Replay, action list, or opponent record is reused as a
   counterfactual physics result.
3. The selected fit rule is immutable and validation uses independent fault
   fingerprints, local trace fingerprints, and all-Expert/all-256 deviations.
   Validation preserves the candidate even when it fails, cycles, or lacks
   precision.
4. Output is deterministic, timestamp-free, in memory, and includes visited
   rules, objectives, exact work tie-breaks, candidate/fixed-point/cycle
   disposition, precision diagnostics, scenario fingerprints, scheduler-call
   budget, and the explicit non-equilibrium claim boundary.
5. Red-green tests cover all starts, tie-breaks, fixed point, cycle,
   non-convergence, precision insufficiency, fit/validation disjointness,
   all-Expert 256 validation, call-budget exhaustion, CRN identity, and
   deterministic rerun. Existing historical and Stage 1/2/3 suites remain
   green.

## Progress log

### Update: 2026-09-05 — Frozen Stage 4 solver protocol

Status: partial

#### Goal

Create and claim the sole Stage 4 ticket and freeze sample precision,
common-path counts, scenario identities, iteration limits, and computation
budget before any solver or campaign execution.

#### Changed

- Created this ticket as the only active Stage 4 solver ticket, blocked by the
  resolved Stage 3 ticket 04.
- Froze `N=8`, `c_B=.5`, three starts `NNNN/NSSN/XXXX`, 32 rounds per start,
  fit `16*2` and validation `32*2` common-path/population scenarios, and the
  one-million complete-scheduler-call cap.
- Froze common-path cluster precision, the `0.005 * J_NNNN^0` target, the
  runner-up two-standard-error rule, independent validation, and the
  `statistics_insufficient` disposition without adaptive sampling.
- Recorded the same protocol in ADR-0016 before implementation relies on it.

#### Verification

- Read `AGENTS.md`, `CONTEXT.md`, the issue tracker/update format, ADR-0014,
  ADR-0015, the complete shared-backup spec sections for the bounded solver,
  finite deviations, validation and inference, and the resolved Stage 3
  ticket before writing this protocol.
- No solver, campaign, experiment, artifact, or statistical validation
  command was run in this planning update.

#### Artifacts

- [05-finite-256-rule-game-solver.md](05-finite-256-rule-game-solver.md)
- [0016-finite-256-rule-game-solver-protocol.md](../../../docs/adr/0016-finite-256-rule-game-solver-protocol.md)
- No generated experiment artifact.

#### Decisions and risks

- The numbers are frozen before observing Stage 4 outcomes; insufficient
  precision remains a reportable result and cannot trigger post-hoc sample
  growth.
- The solver may produce only a restricted-game candidate/fixed-point label
  until the separate simultaneous uncertainty bound is complete. No Nash,
  MFG, convergence, or general-policy claim is authorized.
- The one-million-call cap is a hard reproducibility boundary; if the actual
  implementation needs a different accounting model, stop and amend this
  ticket/ADR before execution rather than silently changing the budget.

#### Next

Write failing Stage 4 solver tests, then implement the smallest in-memory
solver slice under this frozen protocol.

### Update: 2026-09-05 — Implement bounded Pi256 solver

Status: completed

#### Goal

Implement the frozen Stage 4 in-memory symmetric `Pi_256` solver with three
starts, complete per-iteration audit rows, cycle/non-convergence handling,
paired-cluster precision diagnostics, frozen validation, and a hard scheduler
call counter, without starting the fit/validation campaign.

#### Changed

- Added `src/mfg_hedge/game_solver.py` with immutable, strictly validated
  `SolverRuleRow`, `BankEvaluation`, `IterationRecord`, start/result,
  preflight, timing, and frozen-validation contracts.
- Added exact canonical 256-rule ranking by objective, executed work, then
  `N<D<S<X>` order. Every complete update retains all 256 rows, the current
  environment fingerprint, call-count interval, runner-up, and stop reason.
- Implemented symmetric current-population reruns through the existing
  Stage 3 evaluator, with one representative Expert's all-256 deviations;
  each candidate rebuilds the shared finite environment and online calls.
- Implemented paired common-fault cluster differences. `2*SE` uses the SE of
  the elementwise best-minus-runner-up cluster differences, never two
  independent candidate SEs.
- Implemented pure fixed point, rule cycle with first-repeat/period,
  `not_converged_32`, and hard `call_budget_exhausted` outcomes. Budget
  interruption retains the current and unvisited remaining starts.
- Implemented independent validation fingerprint checks for both trace and
  common-fault paths, all-Expert/all-256 validation reruns, and the explicit
  `simultaneous_bound_pending`/no-Nash claim boundary.
- Exported the isolated solver API and bumped the package version from
  `0.21.0` to `0.22.0`; historical engines, configuration behavior, and
  artifacts were not changed.
- Added `tests/test_game_solver.py` red-green coverage for fixed points,
  cycles, 32-round non-convergence, exact objective/work/canonical tie-breaks,
  paired SE, budget exhaustion, frozen call arithmetic, timing, validation
  coverage/isolation, and deterministic reruns without input mutation.

#### Verification

- Initial real Red: `.venv\\Scripts\\python.exe -m unittest
  tests.test_game_solver -v` failed during collection with
  `ModuleNotFoundError: No module named 'mfg_hedge.game_solver'` before the
  implementation existed.
- A follow-up tie-break test first failed with the test expecting `DNNN`;
  the canonical product order correctly selected `NNND`, so the test was
  corrected to assert the repository's actual `N<D<S<X>` position order.
- Focused Stage 4: `.venv\\Scripts\\python.exe -m unittest
  tests.test_game_solver -v` — 10/10 passed.
- Frozen call-count preflight: fit `789,504`, validation `131,584`, total
  `921,088`, remaining `78,912`, within the one-million budget.
- Timing preflight: one in-memory scheduler call completed in
  `0.0013092000153847039` seconds; no timestamp or artifact was emitted.
- State-machine hand checks: `NNNN`, `NSSN`, and `XXXX` each reached a pure
  fixed point with 256 retained rows; `NNNN↔NSSN` retained cycle first repeat
  `0` and period `2`; a 33-rule chain retained 32 iterations and
  `not_converged_32`; paired candidate deltas `(0,1)` produced paired SE `0`
  while the marginal maximum candidate SE was `0.5`, so precision was
  `statistics_insufficient`; a ten-call budget stopped at exactly 10 calls.
- Frozen validation smoke: 2 Experts × 256 rows = `512` rows and
  `2 × 257 = 514` scheduler calls, complete for all Experts, with disjoint
  trace and fault fingerprints and `simultaneous_bound_pending`.
- Full regression: `.venv\\Scripts\\python.exe -m unittest discover -s
  tests -v` — 443/443 passed.
- Configuration check: `.venv\\Scripts\\python.exe -m mfg_hedge check --config
  configs\\v1_minimal.json` — exit 0, `status: ok`; the existing capacity-risk
  diagnostic remains unchanged.
- `git diff --check` passed for the changed Stage 4 source, tests, exports,
  package metadata, and ticket.

#### Artifacts

- Source: `src/mfg_hedge/game_solver.py`.
- Tests: `tests/test_game_solver.py`.
- Public API/version: `src/mfg_hedge/__init__.py`, `pyproject.toml`.
- No experiment artifact, campaign output, configuration, CLI, or report was
  generated or overwritten.

#### Decisions and risks

- Results remain labeled only as restricted-game candidates/candidate fixed
  points; `claims_nash` and `claims_mfg` remain false. A simultaneous
  uncertainty bound is still a separate future gate.
- No adaptive sample extension, random fault campaign, nested workload
  campaign, unilateral-deviation harness, tariff/market calculation,
  solver/MFG campaign, qualification, or holdout execution was started.
- The full fit/validation wall-clock estimate is intentionally not inferred
  from the small timing smoke; the frozen call-count preflight must be paired
  with a future approved campaign decision.

#### Next

None for this implementation ticket; a later ticket may run the frozen fit
and validation campaign only after an explicit campaign decision.

## Answer

Resolved. The bounded Stage 4 solver and its required in-memory verification
are complete; the fit/validation campaign remains intentionally unrun.
