# ADR-0036: Three-round finite reliability-aware routing evaluation

Status: Accepted (authorized by the user for the three-round experiment on 2026-09-10)

## Decision

This ADR authorizes three finite-system rounds for the accepted ADR-0035
single-copy, eight-Replica routing object. It does not add Hedge, prices,
placement optimization, training, cross-Expert capacity, MFG/MFC solvers, or
any equilibrium claim.

The statistical unit is an episode cluster. All arms in an episode share the
same immutable workload, failure paths, Token identities, and service CRN.
Old artifacts are never overwritten and no holdout result may change a
parameter or threshold.

## Frozen budget and preflight

- Hard scheduler-call budget: `60,000`.
- Timing preflight: at most `24` isolated calls, excluded from formal panels
  but included in the hard budget.
- Round 1: `256 * 4 * (5 + 9) = 14,336` calls.
- Round 2: at most `14,336` calls. The qualified-candidate robustness branch
  uses four frozen stress cells × 256 episodes × six arms = `6,144` calls;
  the no-candidate one-revision branch may use the full 14,336-call order.
- Round 3: `1024 * 4 * 6 = 24,576` calls.
- Maximum declared total including preflight is `53,272`, below the hard cap.

If the preflight projects more than six hours, the runner may use immutable
trace reuse and episode-level parallelism, but it may not change physics,
sampling, calls, CRN, or statistical estimands.

## Round 1: development selection

Namespace is `reliability-aware-routing:v1:development:r1`, macro seed
`20260910`, with 256 episodes per S0–S3. The fixed arms are
`uniform_rr`, `jsq`, `loew`, `reliability_only`, and `lazarus_algorithm1`.
The candidate grid is the Cartesian product of history windows
`{25,50,100}` and `(gamma_regular,gamma_urgent)` in
`{(2,3),(4,6),(8,12)}`.

Every candidate must pass physical, CRN, and complete-drain invariants. On S0,
candidate overall mean latency and overall CVaR95 may not exceed LOEW by more
than 2%. Among candidates passing that guardrail, select in order: lowest
equal-weight S2/S3 overall CVaR95; if relative difference is below 0.5%, lowest
Urgent deadline-miss rate; then lowest lost work; then smaller window, smaller
gamma, and canonical dictionary order. The result is only
`development_candidate` or `development_no_candidate`.

## Round 2: frozen robustness or one revision

Round 2 uses namespace `reliability-aware-routing:v1:development:r2` and
macro seed `20260911`; it never reads Round 3.

If Round 1 produces a candidate, freeze it and run these four cells with the
same six arms (frozen candidate plus the five fixed baselines):

1. S2 switch at `t=10`, load `0.9 * 5.6`;
2. S2 switch at `t=30`, load `1.1 * 5.6`;
3. S3 domain hazard `0.024`, load `5.6`;
4. S3 domain hazard `0.036`, load `5.6`.

The cell set checks early/late S2 risk switching, both S3 shock perturbations,
and all three prescribed load levels without selecting a favorable cross-cell
combination. Its status is `candidate_qualified_for_holdout` or
`candidate_not_robust`.

If Round 1 has no candidate, use its diagnostics to select at most one
predeclared revision in this order: separate exponentially decayed
Replica/domain history; an interpretable expected-failure-cost score; a
causal running-age conditional residual. The revision may not read future
faults, future arrivals, service draws, or true remaining work. It is tested
against the same five baselines and may end as `development_no_candidate`,
`physical_failed`, or `statistics_insufficient`; it may not be relabeled as a
successful candidate after seeing Round 3.

## Round 3: independent holdout

Namespace is `reliability-aware-routing:v1:holdout:r1`, macro seed `20260912`,
with 1024 fresh episodes per S0–S3. Arms are the frozen candidate (or the
frozen best interpretable strategy for a no-candidate failure confirmation)
plus the five fixed baselines. No parameter, estimator, threshold, or arm may
be changed using holdout results.

The runner uses 4096 fixed-seed paired cluster bootstrap replicates. For the
candidate contrasts against JSQ, LOEW, Reliability-only, and Lazarus, it uses
a shared paired max-t multiplier for simultaneous 95% intervals. The primary
contrast is the equal-weight S2/S3 overall CVaR95; secondary metrics are
Urgent deadline miss, overall mean/P99, Replay rate, lost work, Replica load
concentration/risk exposure, and drain duration.

Success requires: S2/S3 CVaR95 simultaneous CI strictly below zero versus
LOEW and improvement direction versus JSQ; no significant S0 mean/CVaR95
degradation beyond 2%; Replay rate or lost work not worse; and all physical,
CRN, and complete-drain invariants valid. Otherwise the status is
`holdout_not_improving`.

## Artifacts and claim boundary

Each round writes a unique transactional directory containing
`summary.json`, `paired_statistics.json`, `episode_rows.jsonl`, `manifest.json`,
`protocol.json`, and an analysis Markdown file. A partial/failed panel cannot
publish paired statistics as a successful result. These rounds can establish
finite routing evidence only; they cannot claim optimality, best response,
Nash, MFG/MFC, universal fault prediction, or placement superiority.
