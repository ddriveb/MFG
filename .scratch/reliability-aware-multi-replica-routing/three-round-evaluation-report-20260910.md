# Three-round finite reliability-aware routing evaluation

Status: complete; finite-system routing evidence only

## Protocol and accounting

- R1 development: namespace `reliability-aware-routing:v1:development:r1`,
  macro seed `20260910`, 4 scenarios × 256 episodes × 14 arms = 14,336
  scheduler calls.
- R2 robustness: namespace `reliability-aware-routing:v1:development:r2`,
  macro seed `20260911`, four frozen cells × 256 episodes × 6 arms = 6,144
  calls.
- R3 holdout: namespace `reliability-aware-routing:v1:holdout:r1`, macro seed
  `20260912`, 4 scenarios × 1,024 episodes × 6 arms = 24,576 calls.
- Timing preflight: 24 isolated calls; 3.8570803862373357 calls/s and
  3.8347946192379347 hours projected for the declared 53,248-call maximum.
- Formal scheduler calls: 45,056; including preflight: 45,080. Bootstrap used
  4,096 fixed-seed resamples and no scheduler calls.
- All formal panels completed with zero failures. RSS reporting was unavailable
  (`peak_rss_bytes: null`) rather than estimated from Python heap metrics.

## Round 1: development

R1 ended `development_candidate` with frozen candidate
`risk_aware_jsq_w100_g4_6` (`history_window=100`, `gamma=(4,6)`). The S2/S3
CVaR95 selection scores for the nine candidates, in canonical arm order, were:

| arm | S2/S3 CVaR95 score |
| --- | ---: |
| w100 g2/3 | 7.5740911855 |
| w100 g4/6 | 7.5218632705 |
| w100 g8/12 | 7.6051845436 |
| w25 g2/3 | 7.6505872750 |
| w25 g4/6 | 7.6705197514 |
| w25 g8/12 | 7.5891519957 |
| w50 g2/3 | 7.5971156503 |
| w50 g4/6 | 7.6203827451 |
| w50 g8/12 | 7.6474583276 |

The candidate passed the frozen S0 guardrail. R1 elapsed time was
2,867.908659799956 seconds; failures were zero.

## Round 2: robustness

The candidate remained frozen. Candidate versus LOEW overall CVaR95 by cell:

| cell | candidate | LOEW | candidate − LOEW |
| --- | ---: | ---: | ---: |
| S2 early, low load | 4.0069877353 | 4.0405363383 | -0.033549 |
| S2 late, high load | 6.4189403483 | 6.3688839359 | +0.050056 |
| S3 shock, low hazard | 8.3220960390 | 8.3908761039 | -0.068780 |
| S3 shock, high hazard | 11.9573865654 | 12.28987440198 | -0.332488 |

The frozen 2% robustness guardrail passed:
`candidate_qualified_for_holdout`. R2 elapsed time was
1,438.2495463000378 seconds; failures were zero.

## Round 3: independent holdout

R3 completed 24,576/24,576 calls with zero failures in
5,359.447802299983 seconds. The candidate was not retuned. The table gives
episode means over the four holdout scenarios; `regular` and `urgent` are the
two token classes in this engine.

| arm | mean | P95 | P99 | CVaR95 | urgent miss | replay rate | executed work | lost work | drain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| candidate w100 g4/6 | 2.305014 | 5.466501 | 6.996070 | 6.338266 | 0.371433 | 0.022493 | 226.437281 | 3.088484 | 4.084744 |
| JSQ | 2.346253 | 5.546190 | 7.090708 | 6.431197 | 0.383575 | 0.023250 | 226.305153 | 3.172210 | 4.165658 |
| LOEW | 2.300646 | 5.424477 | 6.942290 | 6.290070 | 0.369652 | 0.023129 | 226.247378 | 3.152651 | 4.034878 |
| reliability-only | 2.340699 | 5.788364 | 7.519611 | 6.780533 | 0.368686 | 0.022533 | 226.038929 | 3.098887 | 4.467392 |
| Lazarus Algorithm 1 | 2.727099 | 6.650317 | 8.116607 | 7.496023 | 0.430662 | 0.022438 | 226.074446 | 3.068158 | 5.886340 |
| uniform RR | 2.845380 | 7.438258 | 9.295388 | 8.519911 | 0.430590 | 0.022290 | 226.113756 | 3.052210 | 6.858286 |

Stress-scenario CVaR95 for candidate versus LOEW was:

| scenario | candidate | LOEW | difference |
| --- | ---: | ---: | ---: |
| S0 | 4.6737909890 | 4.6470141313 | +0.026777 |
| S1 | 5.2283306314 | 5.2217691520 | +0.006561 |
| S2 | 5.1854291988 | 5.1836822334 | +0.001747 |
| S3 | 9.6013509568 | 9.6942574846 | -0.092907 |

All seven requested holdout contrasts are in
`paired_statistics.json`. The paired max-t results for the primary overall
CVaR95 contrast (candidate minus comparison) were:

| comparison | point | paired SE | simultaneous 95% CI | relative point |
| --- | ---: | ---: | --- | ---: |
| JSQ | -0.0929311312 | 0.0200655839 | [-0.1428161136, -0.0430461488] | -1.4450% |
| LOEW | +0.0481962483 | 0.0200346670 | [-0.0016118716, +0.0980043683] | +0.7662% |
| reliability-only | -0.4422674102 | 0.0241208996 | [-0.5022342997, -0.3823005207] | -6.5226% |
| Lazarus Algorithm 1 | -1.1577575328 | 0.0301939420 | [-1.2328225931, -1.0826924724] | -15.4450% |

Additional paired point estimates (candidate minus comparison) were:

| comparison | mean | P95 | P99 | urgent miss | replay rate | executed work | lost work | drain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| JSQ | -0.041239 | -0.079690 | -0.094638 | -0.012142 | -0.000757 | +0.132127 | -0.083725 | -0.080915 |
| LOEW | +0.004368 | +0.042023 | +0.053780 | +0.001781 | -0.000636 | +0.189902 | -0.064166 | +0.049865 |
| reliability-only | -0.035684 | -0.321863 | -0.523540 | +0.002747 | -0.000040 | +0.398351 | -0.010403 | -0.382648 |
| Lazarus Algorithm 1 | -0.422085 | -1.183817 | -1.120537 | -0.059230 | +0.000056 | +0.362834 | +0.020326 | -1.801596 |

The candidate beats JSQ on the paired primary CVaR95 contrast, but its LOEW
contrast CI includes zero and its point estimate is slightly worse. Therefore
the frozen outcome is `holdout_not_improving`, not a qualified routing
candidate. The candidate's lower lost-work point estimate than LOEW is not
enough to override the primary gate.

The routing engine is single-copy and has no Hedge/Protection-Storm action;
Hedge launches and Protection-Storm are therefore structurally zero/not
applicable for this protocol. Replica audit rows, trace/token fingerprints,
and invariant counters are retained in `episode_rows.jsonl`; the holdout
invariant sums for all arms were zero for `completed_without_winner`,
`down_starts`, `duplicate_live_attempts`, and `future_work_observations`.

## Claim boundary

This is finite-system routing evidence only. It makes no optimality, Nash,
mean-field, or MFG claim. The evidence supports retaining reliability-aware
multi-Replica routing as a finite baseline, while this R1-selected
risk-aware-JSQ configuration is not validated as an improvement over LOEW.

