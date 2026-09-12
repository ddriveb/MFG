# Corrected three-round reliability-aware routing evaluation

Date: 2026-09-10

## Scope

This report covers the narrow correction to the candidate risk exposure
horizon. The corrected `risk_aware_jsq` path uses the causal horizon

`estimated_work + SERVICE_MEAN / current_speed`

for an available Replica. In this fixed-speed engine `current_speed` is 1.0
for an available Replica; unavailable Replicas remain ineligible and do not
participate in routing. The five frozen comparator policies explicitly retain
their historical risk projection so the correction does not rewrite their
comparison behavior.

No MFG, Nash, optimality, or deployment claim is made.

## Red and equivalence verification

The first focused horizon test was genuinely Red under the old formula:
observed risk was `2.098321516541546e-14`, while the required one-unit-horizon
risk was `0.02073440484579392`.

After the correction:

- the empty-queue heterogeneous-hazard candidate test passed;
- the five fixed policies (`uniform_rr`, `jsq`, `loew`, `reliability_only`,
  and `lazarus_algorithm1`) matched the pre-correction reference field for
  field on all 4 x 8 smoke traces;
- routing, Lazarus, and experiment tests: `19/19` passed;
- full suite: `732/732` passed;
- config check: `status: ok`;
- `git diff --check`: passed.

The experiment implementation remains internal; no package version bump was
made.

## Corrected identities and calls

| round | namespace | macro seed | calls | failures | elapsed seconds | status |
|---|---|---:|---:|---:|---:|---|
| R1 development | `reliability-aware-routing:v1:corrected:development:r1` | 20260913 | 14336 | 0 | 2818.7792168000014 | `development_candidate` |
| R2 robustness | `reliability-aware-routing:v1:corrected:development:r2` | 20260914 | 6144 | 0 | 1424.323747800081 | `candidate_qualified_for_holdout` |
| R3 holdout | `reliability-aware-routing:v1:corrected:holdout:r1` | 20260915 | 24576 | 0 | 6019.722014000057 | `holdout_not_improving` |

Formal scheduler calls were `45,056`; no retry, supplemental draw, or deleted
episode occurred. The 24-call isolated preflight measured `3.5659074642093427`
calls/s in `6.730404599918984` seconds, with `formal_identity_consumed=false`.
The runner's Windows RSS probe returned `null`; no unsupported memory value is
reported as a limit claim.

The corrected R1 candidate was frozen as:

`risk_aware_jsq_w50_g8_12`

with `history_window=50.0`, `gamma_regular=8.0`, and `gamma_urgent=12.0`.

## Holdout observations

The R3 panel contained 24,576 rows, 4,096 episode trace groups, six arms, all
complete. Each episode's arm rows shared one trace fingerprint. The selected
candidate's holdout per-episode means were:

| arm | overall mean | overall P95 | overall P99 | overall CVaR95 | D/F proxy: regular miss | D/F proxy: urgent miss | replay rate | executed work | lost work | drain duration |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| corrected candidate | 2.3265540758 | 5.5681193997 | 7.1111979250 | 6.4538398139 | 0.1254358109 | 0.3725851065 | 0.0227081965 | 225.4371923676 | 3.1124872383 | 4.1153536424 |
| LOEW | 2.3329954529 | 5.5767045663 | 7.1557991692 | 6.4747015080 | 0.1266017897 | 0.3712435875 | 0.0231985189 | 225.8348801791 | 3.1387417703 | 4.1238532452 |

The candidate versus LOEW paired differences were:

- overall CVaR95: `-0.0208616941`, simultaneous 95% CI
  `[-0.0729876375, 0.0312642492]`;
- overall mean: `-0.0064413771`, simultaneous 95% CI
  `[-0.0225379860, 0.0096552318]`;
- overall P99: `-0.0446012442`, simultaneous 95% CI
  `[-0.1056351259, 0.0164326376]`;
- replay rate: `-0.0004903224`, simultaneous 95% CI
  `[-0.0006838628, -0.0002967820]`.

The candidate was clearly lower than JSQ on overall CVaR95 (paired point
`-0.0942656088`, simultaneous 95% CI `[-0.1462487428, -0.0422824749]`),
but the primary candidate-versus-LOEW CVaR interval includes zero. Therefore
the corrected holdout remains `holdout_not_improving` under the frozen LOEW
comparison, not a final negative conclusion about the ADR-0035 method beyond
this corrected finite evaluation.

## Artifacts

All corrected files are in the new, non-overwriting directory:

`artifacts/reliability-aware-routing-corrected-evaluation-20260910/`

It contains one complete artifact directory for each round. The prior smoke,
development, robustness, and holdout directories were not modified.

## Decisions and risks

- The correction is isolated to the candidate's causal reliability score;
  frozen comparator rows remain field-identical.
- The candidate selection changed after the correction and was frozen before
  R2 and R3.
- All corrected panels completed with zero failures and no new physical
  invariant failure.
- This is finite-system routing evidence only. It is not an MFG, Nash,
  optimality, or deployment result.
