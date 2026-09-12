# ADR-0018: Pooled nonlinear objective inference by common-path jackknife

Status: Accepted (confirmed by the user on 2026-09-06)

Date: 2026-09-06

## Context

The first frozen Fit iteration found one common-fault path whose two nested
population traces contain no F/U Token for representative Expert 0.  The full
16-path sample is complete, but the Stage 4 adapter attempted to score each
path as a standalone `J_i^0`.  That is incompatible with the specification:
`J_i^0` is a nonlinear pooled functional containing cohort ratios and CVaR,
so an individual common-path cluster need not define it.

Filling the cohort with zero, dropping the fixed path, or adding samples after
observing the failure would change the estimand or the frozen protocol.

## Decision

1. Point objectives and rule ranking use the exact scorer over all declared
   paths and both population replicates.  The full pooled sample must contain
   every required phase/class cohort or the result is incomplete.
2. Precision is based on delete-one-common-path recomputations of the same
   pooled objective.  Deleting a path removes both of its nested populations;
   the common fault path remains the top-level statistical cluster required by
   ADR-0016.  A delete-one sample missing a required cohort makes the precision
   result `statistics_insufficient`; it is never imputed.
3. For a paired contrast `d`, with C common paths and delete-one estimates
   `d_(-c)`, use the cluster-jackknife standard error

       sqrt((C-1)/C * sum_c (d_(-c) - mean(d_(-c)))^2).

   Fit applies the existing precision target and selected/runner-up `2*SE`
   rule to these paired delete-one recomputations.
4. Validation forms paired jackknife pseudo-values

       u_c = C*d_full - (C-1)*d_(-c).

   The frozen 4096-replicate, seed-20260906 simultaneous max-t calculation
   resamples common-path indices jointly across every Expert/rule pseudo-value
   row.  This preserves cross-rule and cross-Expert common-noise dependence.
   The two nested populations remain together inside their common-path
   cluster; they are not treated as independent top-level observations.
5. Stored fields and method labels must say `delete_one`, `jackknife`, or
   `pseudo_value`; the old `cluster_objective` wording must not silently carry
   the new meaning.  Scientific serialization changes require version 0.24.0.
6. The failed launch's 8,224 calls remain disclosed.  A later complete rerun
   may use the unchanged 921,088-call campaign ceiling; cumulative calls across
   both attempts are at most 929,312, still below the one-million hard limit.

This decision supersedes ADR-0016 only where it treated standalone cluster
objectives as scalar observations.  All samples, seeds, actions, physics,
budgets, Fit/Validation separation, call ceilings and claim boundaries remain
unchanged.

## Consequences

A cluster may lack F/U without invalidating the full pooled estimand.  Every
reported delete-one value is still an exact recomputation of the declared
nonlinear objective.  The jackknife/max-t procedure is a numerical inference
method, not a finite-sample theorem, and a missing cohort in the full or any
delete-one sample remains a fail-closed statistical outcome.
