# ADR-0016: Frozen finite Pi256 solver and validation protocol

Status: Accepted (confirmed by the user on 2026-09-06)

Date: 2026-09-05

## Context

ADR-0015 and the completed Stage 1/2/3 tickets define an isolated finite
Shared-Backup game, individual objectives, and complete unilateral reruns. The
next bounded step is a 256-rule symmetric best-response iteration. Without a
predeclared sample split and computation boundary, candidate selection could
silently increase data or budget after seeing a cycle, gap, or uncertainty
result.

## Decision

1. Use `N=8`, `c_B=.5`, degraded slowdown `2.0`, hedge delay `1.5`, zero
   tariff, and the existing canonical `N/D/S/X` rule bank.
2. Use exactly three independent fit starts: `NNNN`, `NSSN`, and `XXXX`. Each
   start gets at most 32 updates. Keep fixed points, cycles, and non-converged
   paths as named outcomes; do not infer equilibrium existence or uniqueness.
3. Fit uses 16 common fault paths with two independent nested populations per
   path under namespace `shared-backup-game:v1:stage4-fit` and macro seed
   `20260905`. Validation uses 32 disjoint common paths with two populations
   per path under namespace `shared-backup-game:v1:stage4-validation` and
   macro seed `20260906`. All policies on a split use its immutable CRN
   traces; validation never reuses fit fingerprints.
4. Treat common fault paths as statistical clusters. The fixed normalized
   precision target is `0.005 * J_NNNN^0`; fit precision also requires the
   selected-versus-runner-up paired gap to exceed twice its cluster standard
   error. Failed precision is retained as `statistics_insufficient`; no
   adaptive sample extension is permitted.

Amendment: ADR-0018 supersedes the standalone-cluster scalar interpretation
of this clause.  The fixed samples and thresholds remain unchanged, but the
nonlinear pooled objective now uses exact delete-one-path jackknife inference.
5. Freeze the selected fit candidate before independently recomputing all 8
   Experts' 256 deviations on validation data. Recompute the complete finite
   shared scheduler and online policy actions; do not freeze endogenous pool,
   queues, completions, timers, budgets, or opponent records.
6. Limit the whole run to 1,000,000 complete scheduler invocations. The
   planned upper bound is 789,504 fit calls plus 131,584 validation calls,
   with a fixed reserve for checks and candidate replay. Record call counts
   and runtime in memory only.

## Consequences

The implementation has a deterministic and auditable stopping protocol, but
16/32 common paths are not a general uncertainty theorem. Before a
simultaneous one-sided 95% uncertainty bound is implemented, reports may say
only `restricted-game candidate` or `candidate fixed point`; Pi256 results are
not Nash or MFG certificates. Future larger N, extra common paths, different
precision targets, or a larger policy class require a new decision before
execution. Historical engines, configs, artifacts, and scientific results are
unchanged.
