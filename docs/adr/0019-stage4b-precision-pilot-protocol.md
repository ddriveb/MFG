# ADR-0019: Stage 4b precision pilot protocol

Status: Accepted (confirmed on 2026-09-06)

Date: 2026-09-06

## Context

The frozen Stage 4 Fit used 16 independent common-fault paths and two nested
population replicates per path.  It reached an NSNS point-estimate fixed point
from all three starts, but the required precision gate failed.  The retained
variance diagnosis found that important uncertainty is concentrated in a
small number of public fault paths and in D/F tail latency, especially the
F-phase CVaR.  Repeating the same 16 paths, or treating nested populations as
independent public samples, would not answer the sample-allocation question.

This ADR defines a small, pilot-only design to compare more top-level common
paths with more independent populations inside a path.  It is a planning
protocol, not a new equilibrium protocol.  The bounded implementation and
pilot execution are authorized only through the separately scoped ticket 15;
this ADR does not alter ADR-0015, ADR-0016, ADR-0017, or ADR-0018.

## Decision

### 1. Scientific boundary and frozen inputs

The pilot uses the following four profiles for every immutable scenario:

| Profile | Expert 0 | Experts 1--7 | Role |
| --- | --- | --- | --- |
| `NSNS/NSNS` | `NSNS` | `NSNS` | baseline |
| `NSNS/NSSS` | `NSSS` | `NSNS` | high-variance competitor |
| `NSNS/NSND` | `NSND` | `NSNS` | low-SE neighbor |
| `NSNS/NNDX` | `NNDX` | `NSNS` | high-SE representative |

The first label names the common incumbent environment and the second label
names the Expert-0 profile in the pilot table.  `NSNS` is only a
point-estimate reference, never a candidate, Nash equilibrium, or MFG
equilibrium.  Experts 1--7 participate fully in every shared physical rerun;
only Expert 0 is scored for the pilot contrast.  This is variance planning,
not full equilibrium validation.

Use the frozen pilot-only namespace and seed:

- namespace: `shared-backup-game:v1:stage4b-precision-pilot`;
- macro seed: `20260907`;
- maximum library: 64 independent common-fault paths, each with population
  IDs `0..7`;
- physical parameters: `N=8`, `c_B=0.5`, degraded slowdown `2.0`, hedge
  delay `1.5`, zero tariff, arrival cutoff `360`, complete drain, and the
  existing causal observation/action/budget semantics;
- no Stage 4 Fit, diagnosis, Validation, or pilot artifact may be reused.

The common path is the top-level independent cluster.  Within a path,
population IDs are independent local Expert arrival/class/service streams
conditional on the same public fault path.  A population replicate is never
counted as a second common path.

The 64-by-8 library is generated once in canonical order `(common_id,
population_id)` with common IDs `0..63` and population IDs `0..7`.  A cell
`(C,P)` selects common IDs `0..C-1` and, within each selected path,
population IDs `0..P-1`.  Thus each cell is a stable rectangular prefix of the
maximum library along both dimensions; no cell regenerates a trace.  All
trace, fault, identity, and library fingerprints are recorded before scoring.
The pilot data are permanently pilot-only.

The analysis grid is exactly:

```
C in {16, 32, 64}
P in {2, 4, 8}
```

### 2. Normative contrast and exact pooled functional

Use one sign convention throughout this ADR and its future output:

```
G(r) = J_0(NSNS, NSNS_-0) - J_0(r, NSNS_-0)
```

Positive `G(r)` means the unilateral deviation is beneficial to Expert 0.
For audit readability a report may display the mechanically derived cost
difference `Delta(r) = -G(r)` beside `G(r)`, but it is not a second estimand or
decision sign; every formula, rank, planning gate, and disposition below uses
`G`.

For each cell, compute `J_0` by running the existing exact scorer over the
pooled raw Token/attempt payloads from all selected `(c,p)` scenarios.  The
pooled functional retains its phase/class ratios, replay terms, work terms,
and fractional empirical CVaR95 exactly.  No per-common-path `J_0` is formed,
averaged, or treated as a scalar observation.

For `r` in `{NSSS, NSND, NNDX}`:

```
G_hat_(C,P)(r) = J_hat_(C,P)(baseline) - J_hat_(C,P)(r)
```

The baseline result for each immutable scenario is reused as an exact cached
result across the three contrasts and all nine prefix views.  Each deviation
profile still runs a fresh complete shared physical scheduler with all eight
Experts, all queues, all actions, all failures/timers, shared speeds,
running losers, and full drain.  Reusing an identical immutable input/result
does not freeze endogenous state or skip a counterfactual rerun.

### 3. Common-path uncertainty and nested population diagnostic

For each cell and rule `r`, let `G_hat^(-c)_(C,P)(r)` be the exact pooled
contrast after deleting common path `c` and all of its `P` population
replicates.  The primary between-common-path uncertainty is the ADR-0018
delete-one jackknife:

```
bar_G^(-.) = (1/C) * sum_c G_hat^(-c)
SE_common_JK(r)
  = sqrt((C-1)/C * sum_c (G_hat^(-c) - bar_G^(-.))^2)
```

The exact diagnostic normal-width is reported as
`2 * 1.959964 * SE_common_JK(r)`.  It is a width for comparison only, not a
finite-sample coverage certificate.

To estimate the separate within-common-path population contribution without
pretending that paths are independent observations, use a fixed nested
bootstrap on raw payloads:

1. For outer replicate `b`, draw `C` common-path IDs with replacement from
   `0..C-1`.  Repeated path selections retain the path as one cluster.
2. For every selected path position, draw `P` population IDs with replacement
   from that path's `0..P-1` population payloads.  Each selected outer
   position performs its own independent inner draw: if the same common path
   ID appears more than once in an outer draw, those positions do not share
   inner indices or inner state.  The inner draws are also independent across
   the `B_within` replicates, subject only to the declared deterministic child
   stream derivation.
3. For every rule, pool the resulting raw payloads and recompute both pooled
   objectives before taking `G`; never average per-population objectives.
4. For the same outer path draw, repeat the inner population resampling
   `B_within=256` times.  Use `B_outer=4096` outer draws.

For outer draw `b`, let `mu_b` be the mean of its 256 inner `G` values and
`s_b^2` their sample variance.  The sample variance of the inner means already
contains `V_within / B_within`, so subtract that Monte Carlo component before
adding the full within-population component.  Report the corrected
bootstrap law-of-total-variance diagnostic:

```
V_between_raw = sample_variance_b(mu_b)
V_within      = mean_b(s_b^2)
V_between     = max(0, V_between_raw - V_within / B_within)
V_total       = V_between + V_within
SE_nested = sqrt(V_total)
```

The same resampling indices are used for all three rules so the contrasts are
paired.  Derive deterministic child streams from the pilot namespace, macro
seed, `(C,P)`, resampling layer, and replicate number; no global Token ID is
used to seed a workload or resampling stream.  Proposed resampling seed is
`20260908`, with `B_outer=4096` and `B_within=256` frozen before execution.

`V_between_raw`, `V_between`, `V_within`/`V_total` and percentile widths from
the same nested draws are diagnostic approximations.  The nonnegative
truncation is part of the frozen numerical definition, not a data-dependent
choice.  These quantities do not replace the exact common-path jackknife and
do not create a qualification bound.  The nonlinearity of pooled ratios and
CVaR means a component decomposition is not claimed to be an additive
population-level variance theorem.

The following are fail-closed:

- the full pooled cell objective is missing any required phase/class cohort;
- any common-path delete-one pooled objective is missing a required cohort;
- any raw payload, trace, fault identity, or fingerprint is missing or
  inconsistent;
- a nested resample cannot form a complete pooled objective;
- a non-finite score, zero/negative denominator, or malformed identity occurs.

An invalid full/delete-one result makes that cell's exact SE and planning
disposition `statistics_insufficient`; it is never imputed or silently
dropped.  Invalid nested resamples are counted and make the diagnostic
decomposition insufficient rather than being filtered as favorable evidence.

### 4. Audited cell output

Each `(C,P)` cell emits one deterministic row per `r` in `{NSSS, NSND,
NNDX}` and a cell summary.  The row contains:

- full pooled `G_hat` (the only normative sign);
- exact common-path delete-one SE and diagnostic 95% width;
- nested `SE_nested`, `V_between`, `V_within`, `V_total`, and percentile width
  when the nested diagnostic is complete;
- phase/class counts `D/R`, `D/U`, `F/R`, `F/U`, plus phase totals;
- effective fractional CVaR95 tail mass `0.05*n` for D and F (and the
  corresponding phase/class masses); the scorer's fractional weighting is
  used when a tail boundary cuts through an observation;
- common-path influence concentration based on
  `q_c=(G_hat^(-c)-bar_G^(-.))^2`, including the largest single-path share and
  the top-two share;
- maximum single-path variance contribution `max(q_c)/sum(q_c)`;
- exact delete-one rank-flip count: the number of omitted paths for which the
  deterministic rank of the three deviations differs from the full-cell
  rank; ties use the canonical rule order `NSSS < NSND < NNDX` only for this
  diagnostic;
- scheduler calls, elapsed wall time, and separate trace-preparation,
  scoring, and resampling time fields if available.

Counts are exogenous sample counts and therefore identical across the four
profiles for a cell; action outcomes, queue outcomes, and objectives are not
assumed identical.  Influence and rank diagnostics are descriptive and never
used to delete common paths or change the grid.

For a cell summary, define

```
SE_max(C,P) = max_r SE_common_JK(C,P,r)
W_max(C,P)  = max_r [2*1.959964*SE_common_JK(C,P,r)]
```

Report the signed marginal reduction for `C:16->32` and `32->64` at each P,
and for `P:2->4` and `4->8` at each C.  Per additional 1,000 scheduler
calls, use the exact prefix call difference:

```
call_count(C,P) = 4*C*P

SE_reduction_per_1000
  = 1000 * (SE_max(old) - SE_max(new))
    / (call_count(new) - call_count(old))
```

Report negative reductions when the realized jackknife is non-monotone; do
not smooth or choose a favorable direction after seeing results.  Report the
same calculation for `W_max`.

### 5. Call budget and timing preflight

The four profiles are the only scheduler profiles.  One baseline result is
cached per scenario, so the exact scheduler count for a cell is:

```
4*C*P
```

The maximum library therefore requires exactly `4*64*8 = 2,048` scheduler
calls.  The nine prefix views are scored from those cached immutable results;
they are not rerun.  Mechanical repetition of all nine cells would be
`4*(16+32+64)*(2+4+8) = 6,272` calls and is not the protocol.

Using the prior prepared-tagged N=8 timing maximum of `0.3082773` seconds as
a mechanical reference, the 2,048-call scheduler floor is approximately
`631.35` seconds (`10.52` minutes) before trace preparation, worker startup,
IPC, scoring, and resampling.  A planning estimate is `11--15` minutes on
the same host; this is not a measured pilot result and must be replaced by a
preflight timing record before execution.  The pilot must record serial and
parallel wall time separately and must not alter the frozen grid based on
timing.

All scenario preparation and four-profile execution must complete before
cell summaries are emitted.  PreparedTrace/result reuse is permitted only
for identical immutable inputs and profile definitions; it must not reuse
actions, queue state, speeds, completions, timers, budget state, or scorer
state across a profile.

### 6. Later qualification disposition and claim boundary

The pilot may recommend a future qualification allocation only by a
predeclared rule: choose the least-call rectangular cell whose three pilot
rules satisfy `W_max <= pilot_width_target` and whose full/delete-one/nested
outputs are complete.  If no cell satisfies that planning condition, emit
`precision_plan_infeasible` and do not extend the sample opportunistically.
The cell is a planning recommendation only; a later qualification must use
fresh data and rerun all required physics.

The two frozen parameters are:

```ini
epsilon_nash = 0.20
pilot_width_target = 0.20
```

The values happen to be numerically equal; they are distinct frozen
parameters with different roles.  `pilot_width_target` selects a planning
cell from the pilot.  `epsilon_nash` is used only for the later formal
qualification bound.  The pilot's three deviations cannot establish that all
255 future deviations will meet the pilot width target.

Later formal qualification must evaluate every Expert and all 255 unilateral
deviations under the incumbent profile and use simultaneous one-sided
inference for:

```
G_i(r) = J_i(NSNS, NSNS_-i) - J_i(r, NSNS_-i)
max_(i,r) UCB_0.95(G_i(r)) <= epsilon_nash
```

The pilot's three deviations, one tagged Expert, and planning widths cannot
replace that check.  A Stage 4 point-estimate fixed point is not an epsilon
Nash claim, and no result here is a Nash or MFG result.

The next strategy-space idea, “protect Regular only when the Urgent reserve
still leaves budget,” is recorded as a future policy extension.  It is not in
the Pi256 pilot, not estimated here, and requires a separate protocol.

## Consequences and risks

The design spends only 2,048 exact physical calls for the complete maximum
pilot library while exposing nine correlated prefix views.  This makes C/P
comparisons efficient but means cells are strongly correlated; their rows
must not be treated as nine independent experiments.  The exact jackknife is
appropriate for the declared top-level cluster but its small-C behavior and
the nested bootstrap decomposition remain diagnostic approximations.  The
pilot can show that 64x8 is insufficient; it cannot authorize more samples,
change CVaR, remove influential paths, relax the historical gate, or reuse
pilot data as qualification evidence.

Historical engines, configurations, artifacts, the 16x2 Fit data, and the
negative `fit_no_candidate` result remain unchanged.

## Implementation boundary

Ticket 15 is the separately scoped implementation/experiment slice.  It must
specify PreparedTrace/result caching, deterministic serialization, fail-closed
validation, no artifact overwrites, and the preflight timing result before
any pilot execution.  No qualification, Validation, Fit, solver, or MFG run
is authorized by this ADR.
