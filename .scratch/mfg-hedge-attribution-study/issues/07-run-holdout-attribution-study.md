# Run the one-shot holdout attribution study

Type: task
Status: open
Blocked by: 06

## Goal

Execute the frozen 40 x 50 holdout exactly once and report the highest supported
claim level without code, seed, threshold, or configuration changes.

## Scope

- Verify every frozen hash, execute all registered primary/diagnostic arms and
  quota scales, and commit one transactional campaign artifact.
- Apply the fixed validity, efficacy, safety/resource, attribution, quota, and
  cost gates.
- Report positive, null, and negative findings with the same completeness.
- Make no source, test, configuration, analysis, threshold, or registry change;
  this ticket is execution and reporting only.

## Acceptance criteria

1. Exactly 40/40 macro-seeds and 50/50 episodes each complete for every arm/scale
   frozen as scheduled, with identical per-episode trace fingerprints; pre-frozen
   `NOT_EVALUABLE` diagnostics are neither generated nor reclassified.
2. No peeking, early stopping, seed exclusion, result-dependent retry, or
   modification occurs after hash verification; the working tree and frozen
   hashes remain identical before and after execution.
3. The aggregate contains raw macro values, intervals, every gate result, full
   Pareto rows, solver/quota/event invariants, separate VALID/INVALID status,
   one C0-C3 conclusion no higher than its lowest passed prerequisite, and the
   independently gated RULE/QUOTA/COST flags.
4. Infrastructure retries reuse the identical key and are audited; any scientific
   defect invalidates all v1 holdout results rather than salvaging selected seeds.
5. Full tests/environment check pass immediately before execution; every complete
   VALID positive or negative campaign is committed, and its fresh directory is
   never overwritten.

## Progress log

## Answer

## Comments
