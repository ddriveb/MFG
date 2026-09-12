# Persistent runtime-cost correction

## Question

Does charging redundant execution even below the capacity boundary prevent the
Healthy policy reused in Recovered from treating Hedge work as free, while
retaining fault-window protection where its latency and Replay benefit exceeds
the full runtime cost?

## Cost contract

For Token class `k` and action `a`, the solver objective is

```text
incremental_work[k,a] = expected_hedge_work[k,a] + expected_replay_work[k,a]
J[k,a] = mean_latency[k,a]
       + replay_penalty[k] * replay_probability[k,a]
       + (capacity_price + incremental_work_cost) * incremental_work[k,a]
       + wasted_work_cost * expected_wasted_work[k,a]
```

`expected_wasted_work` is the expected executed work of every terminal attempt
other than the Token's winning attempt. It includes failed running Primaries and
completed running losers; queued cancellations and invalidations contribute zero.
It may overlap incremental work by design: `incremental_work_cost` prices resource
consumption, while `wasted_work_cost` additionally prices work that produces no
winning result.

Capacity accounting, the boundary-pinned shadow price, Replay penalties, CRN,
quota projection, and event semantics remain unchanged.

## Configuration and compatibility

- Schema 3 remains readable and maps both new costs to zero, preserving the old
  scientific behavior and artifacts.
- Schema 4 requires explicit strictly positive `incremental_work_cost` and
  `wasted_work_cost`.
- `configs/v2_paired_runtime_cost.json` uses 1.0 for both costs. These are
  untuned normalized mechanism defaults, not measured cloud prices.
- Calibration serialization advances to schema 2 because ActionStats gains
  `expected_wasted_work`.

## Acceptance

1. Calibration measures wasted work from attempt winners/statuses and preserves
   common random numbers across actions.
2. With price zero, otherwise identical actions with more incremental or wasted
   work have strictly higher cost and lower Softmax probability when the relevant
   configured coefficient is positive.
3. Setting both coefficients to zero reproduces the legacy solver policy.
4. Schema-3 remains readable as the legacy zero-cost configuration; schema-4
   rejects missing, zero, negative, bool, NaN, and infinite costs.
5. The complete 70-cell schema-4 solve passes the official gate and is
   deterministic. A same-trace paired controlled run is written to a new artifact
   directory; no old artifact is modified.
6. Report the measured trade-off honestly. Do not retune the coefficients after
   seeing the result.

## Boundary

No HJB-FPK, training, Gate changes, cross-Expert routing, dynamic replicas,
load sweep, coefficient sweep, or replacement of the accepted capacity/quota
mechanism.
