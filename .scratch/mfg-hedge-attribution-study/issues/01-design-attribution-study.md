# Design the MFG-Hedge attribution study

Type: task
Status: resolved
Blocked by: none

## Goal

Freeze a falsifiable attribution-study protocol that can distinguish the value
of the Hedge package, a frozen state-aware rule package, mean-field allocation,
quota projection, and the two persistent runtime-cost terms without reusing the
completed pilot results as confirmation evidence.

## Scope

- Write the feature specification only; do not change Python, tests,
  configurations, or experiment artifacts.
- Record the accepted attribution-first scientific and architectural decisions
  in ADR-0010, including their supersession boundary relative to ADR-0006 through
  ADR-0009.
- Define the primary arms, diagnostic ablations, fixed-horizon repeated-episode
  sampling unit, resource-matching contract, metrics, statistical unit,
  development/holdout separation, success gates, and conclusion hierarchy.
- Create bounded downstream implementation and execution tickets with explicit
  dependencies.

## Acceptance criteria

1. The primary comparison includes No-Hedge, a frozen state-aware rule, a
   myopic non-mean-field controller, and the full mean-field controller under
   the same causal quota and predeclared expected-work budgets.
2. The design separates H, D, F, and R control contexts, prohibits uncontrolled
   H/R Softmax leakage in the primary study, and states exactly how any entropy
   regularization may be interpreted.
3. The optimization target contains a calibratable tail/SLO loss and complete
   action-dependent work accounting, with a destination-B constraint, nominal A
   reservation, and actual execution-phase x Replica audits.
4. The evaluation uses independent fixed-horizon fault episodes grouped into
   macro-seeds; Token-level pseudo-replication and an indefinitely dominant R
   phase are forbidden.
5. Primary/secondary metrics, non-inferiority/resource gates, conclusion levels,
   and development versus holdout namespaces are fixed before implementation.
6. Existing schema-3/schema-4 configurations, ADRs, artifacts, and pilot results
   remain immutable historical evidence rather than confirmation data.
7. The standard unit suite and environment check remain green after the
   documentation-only change.

## Progress log

### Update: 2026-09-05 — Freeze the MFG-Hedge attribution study

Status: completed

#### Goal

Freeze a falsifiable, attribution-first protocol that says exactly what MFG is
allowed to improve, separates it from Hedge/rule/quota/cost effects, and protects
a one-shot holdout from the completed pilot evidence.

#### Changed

- Added `.scratch/mfg-hedge-attribution-study/spec.md`: four primary arms
  (No-Hedge, RULE, MYOPIC, conditional mean-field), no-quota and 00/10/01 cost
  diagnostics, persistent incremental/waste work costs, fixed episode/macro-seed
  design, CVaR95 gates, safety/resource gates, and VALID/C0-C3 plus independent
  diagnostic flags.
- Defined the conditional mean-field mechanism as destination-B load closure:
  Normal Replay is included in the all-Normal fixed point `rho_B^N`; A is a
  conservative nominal reservation; A1/A2 receive non-feedback resource-audit
  fixed points; A3 alone closes decisions to the induced B load.
- Froze one arm-independent `g_budget` table, two explicit capacity/quota duals,
  causal two-phase quota semantics, and separate requested-policy, applied-charge,
  and realized phase x Replica resource views.
- Froze 40 macro-seeds x 50 independent H/D/F/R episodes at nominal load 0.45,
  episode-local aggregation for time-series metrics, zero-variance paired-test
  rules, one-shot holdout namespaces, and transactional preservation of complete
  positive and negative results.
- Added and accepted `docs/adr/0010-attribution-first-evaluation-protocol.md`,
  including explicit retain/replace boundaries for ADR-0005 through ADR-0009.
- Added open downstream tickets 02-07 for episode metrics, ActionStats/objective,
  controllers/quota, campaign/CLI, development qualification/freeze, and pure
  one-shot holdout execution.
- No Python source, test, existing configuration, or experiment artifact changed.

#### Verification

- Three independent read-only reviews checked scientific attribution, resource/
  ADR semantics, statistics, ticket dependencies, and workflow boundaries; after
  two correction rounds all reviewers reported no remaining blocker.
- `rg` audits found one claimed ticket only during the work, a complete acyclic
  dependency chain 01 -> 02 -> 03 -> 04 -> 05 -> 06 -> 07, and no stale L0-L6,
  `rho_B=0.45` baseline, primary-only H/F/R, or success-only artifact wording.
- `.\.venv\Scripts\python.exe -m unittest discover -s .\tests -v` —
  `Ran 303 tests in 21.100s`, `OK`.
- `.\.venv\Scripts\python.exe -m mfg_hedge check --config .\configs\v1_minimal.json`
  — exit 0, `status: ok`, Python 3.10.11; the existing single-domain transient
  capacity warning remains expected.

#### Artifacts

- `.scratch/mfg-hedge-attribution-study/spec.md`
- `docs/adr/0010-attribution-first-evaluation-protocol.md`
- `.scratch/mfg-hedge-attribution-study/issues/01-design-attribution-study.md`
- `.scratch/mfg-hedge-attribution-study/issues/02-implement-episode-protocol-and-metrics.md`
- `.scratch/mfg-hedge-attribution-study/issues/03-extend-action-stats-and-tail-objective.md`
- `.scratch/mfg-hedge-attribution-study/issues/04-implement-attribution-controllers.md`
- `.scratch/mfg-hedge-attribution-study/issues/05-implement-budget-matched-attribution-campaign.md`
- `.scratch/mfg-hedge-attribution-study/issues/06-qualify-development-and-freeze-holdout.md`
- `.scratch/mfg-hedge-attribution-study/issues/07-run-holdout-attribution-study.md`
- No experiment output was generated.

#### Decisions and risks

- The primary MFG claim is deliberately narrow: A3 must beat both a fixed RULE
  and an otherwise identical MYOPIC controller; beating No-Hedge alone proves no
  mean-field contribution.
- The confirmatory endpoint is fault-window tail latency. SLO miss/excess remains
  in the controller objective and explanatory metrics but is not overclaimed as
  a co-primary result.
- Load 0.45 places A only at a nominal boundary. Cross-phase carry-in, F/R
  spillover, and realized per-Replica work remain explicit audits, not hidden
  feasibility assumptions.
- The destination-load table is a tagged-probe surrogate, not a full joint queue
  distribution or HJB-FPK MFG. A policy-composition-consistent calibration is a
  later feature only if this attribution study justifies it.
- The planned campaign is intentionally large (expected 576,000 Tokens per arm
  at the main scenario); ticket 06 must qualify runtime and freeze any unavailable
  cost diagnostic as `NOT_EVALUABLE` before holdout.
- Schema-3/schema-4 artifacts and seeds 20260901..20260920 remain burned
  development evidence and cannot enter confirmation.

#### Next

Claim ticket 02 and implement the fixed-horizon episode protocol and metrics;
do not implement controllers or touch the holdout namespace yet.

## Answer

The attribution protocol is frozen and ADR-0010 is Accepted. The design now
isolates the conditional mean-field contribution from Hedge, a frozen rule,
quota, and runtime-cost effects under one predeclared budget, with an executable
development/freeze/holdout path. All 303 existing tests and the environment check
remain green; ticket 02 is the next unblocked task.

## Comments
