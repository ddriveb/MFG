# Scaled multi-Expert, multi-Backup development report

Date: 2026-09-05

## Experiment boundary

This is a development experiment, not an independent qualification run.

- 8 Experts with an immutable uniform exogenous Gate.
- 3 Replicas per Expert: one Primary and two Backups.
- 128 deterministic episodes and 147,585 generated Tokens.
- Per-Expert offered load 0.45; aggregate offered load 3.6.
- Shared H/D/F/R timeline: 100/200/220, arrivals before 320, then full drain.
- All arms use the same workload traces and attempt-level CRN streams.
- A fixed prospective Backup-work envelope is used per Expert and D window.

The compared arms are No-Hedge, the previously frozen single-Backup NIIN rule,
and an exhaustive 81-rule dual-Backup development search.

## Main results

| Arm | Objective J | D CVaR95 | F CVaR95 | Executed work / Token | Wasted work / Token |
| --- | ---: | ---: | ---: | ---: | ---: |
| No-Hedge | 20.4399 | 22.7828 | 5.4228 | 1.0034 | 0.0037 |
| Frozen single-Backup NIIN | 15.0702 | 16.7172 | 4.0695 | 1.0251 | 0.0279 |
| Selected dual-Backup NIII | 17.3408 | 19.3740 | 4.6019 | 1.0399 | 0.0475 |

Relative to No-Hedge, the frozen single-Backup arm:

- reduces the objective by 26.27%;
- reduces D CVaR95 by 26.62% and F CVaR95 by 24.96%;
- increases total executed work by 2.16%;
- applies 7,250 Backup copies, produces 6,850 Hedge winners, and reduces Replay
  executions from 4,104 to 2,778.

Relative to No-Hedge, the selected dual-Backup NIII arm:

- reduces the objective by 15.16%;
- reduces D CVaR95 by 14.96% and F CVaR95 by 15.14%;
- increases total executed work by 3.64%;
- protects 3,873 Tokens with 7,746 Backup copies and produces 3,796 Hedge
  winners; Replay executions are 3,307.

H and R safety ratios remain exactly 1.0 in both protected arms because the
selected rules use Normal in the healthy/recovered early-Regular cohort and the
fixed admission boundary does not perturb those baseline cohorts.

## What the additional Backup changed

The selected dual-Backup arm is worse than the frozen single-Backup arm:

- objective: +15.07%;
- D CVaR95: +15.89%;
- F CVaR95: +13.08%;
- total executed work: +1.45%;
- wasted work: +70.45%.

This is not only a different-rule effect. Holding NIIN fixed, dual Backup still
raises the objective from 15.0702 to 17.3926, raises D CVaR95 from 16.7172 to
19.4109, and increases wasted work by 69.35% relative to single Backup.

The mechanism is the prospective work envelope. A single-Backup protection
costs one reservation unit, while dual Backup costs two. Under the same cap,
dual Backup therefore protects about half as many Tokens while creating an
extra losing copy whenever both Backups launch. The additional race diversity
does not compensate for the reduced coverage and extra loser work in this
configuration.

## Search and reproducibility

- All 81 dual-Backup rules were evaluated and all 81 passed the frozen safety
  and resource filters.
- The selected development rule is NIII.
- A second complete in-memory reconstruction produced identical rule results,
  trace fingerprints, comparisons, and selected rule.
- The artifact was committed transactionally under one unique run directory;
  existing run directories are rejected rather than overwritten.

## Interpretation limit

The expanded run shows that the single-Backup mechanism scales to eight
independent Expert queues under a common failure, but it does not establish a
general theorem that one Backup is always optimal. This trace was also used to
select the dual rule, so the dual result is development evidence only. An
independent namespace with frozen arms is required before making a qualification
claim.

