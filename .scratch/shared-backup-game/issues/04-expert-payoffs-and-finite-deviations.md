# Expert payoffs and finite-system deviations

Type: implementation
Status: resolved
Blocked by: 03

## Scope

Implement only Stage 3 of the Shared-Backup Expert game:

- pooled per-Expert (J_i^0) costs with phase/class cohorts;
- social cost (S_N) and a separately named pooled-across-Experts diagnostic;
- complete finite-system unilateral-deviation reruns with fresh shared physics;
- lower-level candidate evaluation and official complete \\Pi_256 best response /
  restricted regret APIs;
- a small deterministic in-memory development timing smoke.

Preserve Stage 1/2 physics, random keys, workload traces, observation
semantics, historical engines/configurations/artifacts, and all ADR-0015
mathematics. This ticket does not implement equilibrium iteration, cycle search,
conditional particle models, MFG solver, tariffs/prices, bootstrap/intervals,
campaigns, qualification/holdout, CLI/config/artifacts, training, or a
centralized planner.

## Acceptance criteria

1. Per-Expert scorer pools all supplied episodes by arrival phase/class, uses
   pooled numerators/denominators, retains empty episodes, reports mean latency,
   SLO excess, miss rate, Replay rate, exact fractional CVaR95, work/waste,
   phase losses, objective components and complete/incomplete status.
2. Work and waste are cross-checked from immutable attempt records, including
   failed executed work, Replay, Hedge, running losers through drain, and zero
   queued-loser work. Duplicate/missing/invalid ownership and lifecycle rows
   fail fast.
3. Social cost is the unweighted arithmetic mean of complete individual
   objectives; pooled-across-Experts is a distinct diagnostic and is not used as
   the individual or social objective.
4. A deviation run rebuilds the complete N-Expert shared scheduler and calls
   opponent ActionSources online again. It caches only immutable exogenous
   inputs, never baseline speeds, actions, queues, completions, Replay, timers,
   budgets or result records.
5. Lower-level candidate subsets and official all-Expert/all-256 APIs enforce
   incumbent inclusion, exact lexical bank completeness, deterministic sorting,
   tie-breaking by objective/work/lexical rule, nonnegative regret and the
   same-trace NNNN normalization denominator.
6. Results are immutable/deterministic, expose peer/social externalities and
   trace/fault fingerprints, and the timing smoke remains in memory without
   selecting an equilibrium or writing artifacts.

## Progress log

### Update: 2026-09-05 — Claimed scorer and finite-deviation slice

Status: partial

#### Goal

Claim the sole Stage 3 implementation ticket and establish its bounded
red-green-refactor boundary before implementation changes.

#### Changed

- Created and claimed this ticket as the only active ticket for Expert costs,
  social/pooled diagnostics, finite unilateral deviations, Pi256 restricted
  best response/regret, and timing smoke.
- Retained the Stage 1/2 implementation boundaries and the accepted ADR-0015
  mathematics.

#### Verification

- Re-read `AGENTS.md`, `CONTEXT.md`, issue tracker/update format, ADR-0005,
  ADR-0012, ADR-0014, ADR-0015, the complete feature spec, Stage 2 ticket,
  `shared_backup.py`, `game_workload.py`, `transient_control.py`, and the
  existing Stage 1/2 tests before code changes.
- No implementation or test command claimed in this preparatory update.

#### Artifacts

- This ticket only; no generated artifact or experiment output.

#### Decisions and risks

Scoring will reuse the existing exact fractional-tail function rather than
copying a second CVaR implementation. Counterfactuals will rerun the isolated
Stage 1 scheduler from immutable traces and online policy functions. This slice
will report restricted Pi256 regret only; it will not claim Nash/MFG equilibrium
or statistical certainty.

#### Next

Add the required focused scorer and deviation tests before implementing either
new module.

### Update: 2026-09-05 — Completed scorer and finite-deviation slice

Status: completed

#### Goal

Implement only the individual Expert objective, social/pooled diagnostics,
fresh finite-system unilateral deviations, restricted Pi256 best response and
regret, and deterministic timing smoke while preserving Stage 1/2 physics and
all historical behavior.

#### Changed

- Added [expert_game.py](/D:/project/mfg_hedge_v1/src/mfg_hedge/expert_game.py)
  with immutable `ExpertEpisodeRun`, pooled phase/class `CohortMetrics`,
  per-Expert `J_i^0`, equal-Expert social cost, and separately named
  `pooled_across_experts_objective`. The scorer retains empty episodes,
  rejects missing cohorts with `total=None`, uses raw Token/attempt
  numerators, and reuses the existing exact fractional-tail CVaR function.
- Added [game_deviations.py](/D:/project/mfg_hedge_v1/src/mfg_hedge/game_deviations.py)
  with immutable `DeviationScenario`, fresh full-population reruns,
  online opponent calls, CRN trace/fault fingerprints, peer/social deltas,
  lower-level candidate subsets, exact canonical Pi256 validation, restricted
  best response/regret, and an in-memory deterministic timing smoke.
- Added [test_expert_game.py](/D:/project/mfg_hedge_v1/tests/test_expert_game.py)
  and [test_game_deviations.py](/D:/project/mfg_hedge_v1/tests/test_game_deviations.py)
  as the Stage 3 red-green tests. Existing `shared_backup.py` and
  `game_workload.py` behavior was not changed.
- Exported the new isolated public APIs and bumped the package from `0.20.0`
  to `0.21.0`; this is an additive namespace change and does not change
  historical scientific results.

#### Verification

- Real Red before implementation: both required focused commands failed at
  collection with `ModuleNotFoundError` for the not-yet-created
  `mfg_hedge.expert_game` and `mfg_hedge.game_deviations` modules.
- `.venv/Scripts/python.exe -m unittest tests.test_expert_game -v`: **9/9**.
- `.venv/Scripts/python.exe -m unittest tests.test_game_deviations -v`:
  **10/10**.
- `.venv/Scripts/python.exe -m unittest tests.test_shared_backup -v`:
  **14/14**.
- `.venv/Scripts/python.exe -m unittest tests.test_game_workload -v`:
  **15/15**.
- `.venv/Scripts/python.exe -m unittest discover -s tests -v`: **433/433**.
- `.venv/Scripts/python.exe -m mfg_hedge check --config
  configs/v1_minimal.json`: exit 0, `status: ok`.
- Hand-calculated complete scorer trace: phase losses
  `(H,D,F,R)=(0.1,0.2,0.1,0.1)`, phase component `0.125`, D/F fractional
  CVaR term `0.15`, executed work/Token `0.1`, waste/Token `0`, giving
  `J_i^0=0.375` up to floating-point roundoff.
- Raw pooling check: the H/R cohort had 6 Tokens, pooled mean latency
  `0.4333333333333334`, while the two episode means were `0.1` and `0.5`;
  the scorer used the pooled numerator/denominator, not episode-ratio means.
- Fractional-tail check: `n=20` gives `20.0`; `n=21` gives
  `20.952380952380953`, including the exact fractional boundary mass.
- Asymmetric two-Expert check: social mean Expert objective
  `22.8625`, pooled-across-Experts diagnostic `29.5375`; individual D tails
  were `0.2` and `12.0`, while the pooled D tail was `12.0`.
- N=2 fresh deviation rerun at `c_B=.5`: baseline `NNNN` objective
  `0.45`; deviating `SSSS` objective `11.18`; peer objective changed from
  `0.45` to `0.525` (`+0.075`), and peer F Regular/Urgent mean latency each
  changed from `0.2` to `0.3`. The immutable trace and fault fingerprints
  remained identical; opponent policy calls were made again online.
- The same N=2 deviation changed shared head-time integral from `0.8` to
  `20.4`, confirming endogenous pool recomputation rather than a frozen
  baseline speed/rate. All rows carry invariant-counter fingerprints and
  complete scores.
- The inherited Stage 1 pool audits continued to satisfy each interval's
  `executed_BC <= N*c_B*dt` bound, reaching the bound on busy constrained
  intervals; its 3N queue-conservation assertions also remained green in the
  14/14 shared-physics suite.
- Official Pi256 smoke produced `512` rows for two Experts (`256` each),
  canonical `N<D<S<X` order from `NNNN` through `XXXX`, with complete
  candidate-bank validation. The repeated in-memory timing smoke took
  `1.6317792999907397` seconds, returned `row_count=512`,
  `all_complete=True`, `deterministic_repeat=True`, and selected no
  equilibrium.

#### Artifacts

- No experiment was run and no new experiment artifact, CLI output,
  qualification/holdout result, or campaign output was generated.
- No existing artifact was overwritten.
- No solver/MFG implementation was started; the reported regret is only a
  restricted Pi256 point estimate.

#### Decisions and risks

- The scorer uses pooled own-Expert Token cohorts and exact fractional CVaR;
  it never silently drops empty episodes, fills missing cohorts, pools social
  and individual objectives, or substitutes the historical rounded CVaR.
- Every counterfactual reuses only immutable exogenous trace/fault inputs and
  reruns all N-Expert queues, K/speeds, completions, Replay, timers, budget
  projection, and opponent observations. It makes no Nash, equilibrium,
  confidence-interval, or simultaneous-certification claim.
- The action bank and policies are fixed online rules, not a full feedback
  policy class. Runtime is a development timing cost only; no campaign sizing
  or statistical inference is implied.

#### Next

Leave this ticket resolved. Any future work must open a separately scoped
ticket for nested scale validation, random-fault conditional estimation,
campaign precision, or a restricted solver; none is included in this slice.

## Answer

Resolved. The bounded Stage 3 scorer, finite-system unilateral-deviation
harness, canonical Pi256 evaluation, and in-memory timing smoke are complete;
no later solver, campaign, qualification, or artifact work was started.
