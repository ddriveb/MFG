# Implement the budget-matched multi-arm attribution campaign

Type: task
Status: open
Blocked by: 02, 04

## Goal

Implement the multi-arm, multi-scale, repeated-episode campaign, macro-seed
inference, gate report, and transactional artifact surface without touching the
holdout namespace.

## Scope

- Run all primary arms and diagnostics over common episodes and quota scales.
- Aggregate exactly at macro-seed level; implement log-ratio, paired difference,
  two-sided/one-sided t intervals, Holm gatekeeping, non-inferiority, conclusion
  levels, validity status, independent diagnostic flags, and Pareto rows exactly
  as specified, including episode-local time-series aggregation and zero-variance
  paired-test rules.
- Add the final `run-attribution-study` CLI with explicit development and holdout
  modes plus a transactional artifact bundle containing all hashes, policies,
  dual witnesses, raw macro values, gates, and invalid-run behavior. Holdout mode
  requires a verified ticket-06 freeze bundle; this ticket exercises only
  development/fixture namespaces.

## Acceptance criteria

1. Hand-computed statistical/gate fixtures cover superiority, non-inferiority,
   Holm-adjusted one-sided diagnostic differences, zero/undefined ratios, valid
   negative C0 results, zero-variance differences, and invalid studies.
2. One calibration/policy build is reused without leaking evaluation results;
   every arm/scale shares trace identity within an episode.
3. No post-hoc realized-work matching, seed deletion, partial inference, or
   committed partial directory is possible; a complete VALID negative C0 result
   is committed with the same schema as a positive result.
4. Small development fixtures deterministically write the complete bundle and
   existing directories are never overwritten.
5. Full tests and environment check pass; the final runner refuses holdout mode
   without a matching freeze bundle, and this ticket generates or consumes no
   `attribution:v1:holdout` key.

## Progress log

## Answer

## Comments
