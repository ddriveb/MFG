# MFG-Hedge paired multi-seed campaign

## Scientific question

At the frozen load-0.5 mechanism-verification setting, does the paired
MFG-Hedge versus No-Hedge trade-off observed in ticket 04 persist across the
predeclared evaluation seeds?

## Frozen inputs

- Consume `tokens_per_run`, `base_seed`, and `seed_count` from the existing
  schema-3 paired configuration.
- Build the production 70-cell calibration table once from the immutable
  configuration and reuse it for every evaluation seed.
- Evaluation seeds are exactly `base_seed + i` for `i in [0, seed_count)`.
- Each seed uses one pre-generated trace shared byte-for-byte by both arms.
- Solver, calibration, quota, event semantics, failure timeline, penalties,
  and metric definitions are unchanged.

## Artifact contract

One fresh campaign directory contains a manifest, one complete record per
seed, and an aggregate report. Existing artifacts are never overwritten.
The campaign is assembled in a sibling staging directory and committed with
one rename only after every seed and every invariant passes.

Each seed record contains both arm summaries, the paired comparison, trace
fingerprints, the effective evaluation seed, and solver diagnostics. The
aggregate contains seed-level paired distributions for mean/P50/P95/P99
latency, Replay rate, total executed work, work amplification, drain duration,
and sustained overload, plus mean delta, sample standard deviation, 95%
Student-t confidence interval, sign counts, and the raw seed values. Metrics
whose relative denominator is zero remain null and are not coerced.

This is a mechanism campaign, not a population claim beyond the configured
seed family. No seed may be removed or rerun because of its result.

## Failure behavior

Configuration, calibration coverage, solver gate, trace identity, metric
invariants, or artifact failures abort loudly. A failed campaign leaves no
committed campaign directory. Stress loads 0.6 and 0.7 are a separate
diagnostic ticket and are never pooled into the load-0.5 aggregate.

## Acceptance criteria

1. A pure unit fixture verifies exact seed enumeration and hand-computed
   paired mean/SD/95% CI/sign counts.
2. A CLI fixture verifies one calibration build, distinct evaluation traces,
   same-trace A/B identity, transactional output, and no overwrite.
3. The full unit suite and environment check pass.
4. A real campaign using `configs/v1_paired_comparison.json` completes all
   configured seeds and writes the aggregate without changing scientific
   inputs.
