# Run paired cancellation-semantics ablation

Type: experiment
Status: resolved
Blocked by: none

## Scope

Run the frozen ADR-0032 2×2 paired panel on 1,024 newly generated episodes:
`niin:conservative`, `niin:preemptive`, `laedge:conservative`, and
`laedge:preemptive`. Preserve the existing workload, Reservation, prices,
policy definitions, CRN derivation, and historical artifacts. This ticket
covers experiment execution and transactional artifact writing only; it does
not select a policy, tune a price, or make optimality/Nash/MFG claims.

## Frozen protocol

- Namespace, macro seed, episode identity, protocol fingerprint, and all
  episode fingerprints are frozen before execution and are disjoint from
  historical runs.
- The four arms share each episode's arrival, class, attempt-0/1/2 service
  streams, and common failure timeline exactly.
- Scheduler budget is exactly 4,096 calls; failed calls are recorded,
  fail-closed, never retried, never supplemented, and never removed.
- A timing/preflight run is required before the formal panel but cannot change
  the 1,024 episode count.
- Episode is the statistical unit. Paired cluster bootstrap uses 4,096 frozen
  seeds and reports point estimates, absolute/relative differences, and 95%
  confidence intervals.
- Required metrics include overall latency mean/P95/P99, D/F CVaR95, D/F
  deadline miss rate, H/R P99, Replay rate, executed/wasted work, Hedge
  launches and winner rate, Protection-Storm peak, drain duration, and
  completed/invariant/failure counters.
- Required contrasts are within-family cancellation contrasts, cross-family
  contrasts at each cancellation mode, and cancellation × policy-family
  difference-in-differences.
- Verification must show that only running-loser cancellation changes; queued
  loser, failure-first, Replay, CRN, dispatch, and admission behavior remain
  unchanged.
- Preemptive results are labeled oracle/upper-bound semantics when the
  execution platform cannot physically cancel running work.

## Acceptance criteria

1. Red tests exist before the runner implementation and cover exact arm
   membership, disjoint episode identities, CRN equality, call accounting,
   fail-closed behavior, paired statistics, and non-overwriting artifacts.
2. A small timing/preflight completes without altering the frozen protocol.
3. The formal panel attempts exactly 4,096 scheduler calls and writes a fresh
   transactional artifact directory containing `summary.json`,
   `paired_statistics.json`, `episode_rows.jsonl`, `manifest.json`, and
   `protocol.json`.
4. Missing or failed episode panels do not receive zero-fill or replacement;
   formal results are marked incomplete/failed and cannot be presented as a
   clean comparison.
5. Focused, affected, full-suite, and config checks pass before execution;
   the final ticket update reports exact counts, call totals, artifact path,
   all required contrasts, and the claim boundary.

## Progress log

### Update: 2026-09-09 — Claim formal 2×2 panel

Status: claimed

#### Goal

Freeze and execute the accepted ADR-0032 paired cancellation-semantics panel
without changing historical scientific inputs or artifacts.

#### Changed

- Accepted ADR-0032 at the user's direction.
- Created and claimed this independent formal experiment ticket.
- No runner code, tests, timing preflight, or formal episode has run yet.

#### Verification

- Re-read `AGENTS.md`, `CONTEXT.md`, issue/update instructions, ADR-0032,
  the resolved implementation ticket 05, and the current baseline engine and
  artifact-writing modules.
- Confirmed no other ticket is currently `claimed` at the ticket header.

#### Artifacts

None.

#### Decisions and risks

The experiment is descriptive mechanism attribution only. The panel must not
choose a new algorithm or change prices, Reservation, workload parameters,
CRN keys, or policy definitions. Preemptive results may be oracle semantics
depending on runtime support.

#### Next

Write the formal runner's failing tests, then implement the smallest
transactional panel and statistic adapters.

### Update: 2026-09-09 — Execute frozen 2×2 panel

Status: partial

#### Goal

Run the accepted 1,024-episode, four-arm paired cancellation panel after all
focused, affected, full-suite, and config gates passed.

#### Changed

- Executed the required two-episode timing preflight before the formal panel:
  8 calls, `0.8308002999983728` seconds, `9.629269512800692` calls/second.
- Executed all 1,024 episodes across the four canonical arms in the frozen
  namespace `replica-routing-baselines:cancellation-ablation:v1` with macro
  seed `20260931`.
- Consumed exactly 4,096/4,096 formal scheduler calls. No call was retried,
  supplemented, or removed after failure.
- Wrote the required fresh artifact files transactionally.

#### Verification

- Focused runner suite: `tests.test_protection_cancellation_experiment`, 6/6
  passed.
- Affected suites (cancellation runner, cancellation hooks, hedge engine,
  online engine, baseline experiment, attribution episode, attribution
  metrics): 103/103 passed.
- Full suite: `.venv\\Scripts\\python.exe -m unittest discover -s tests -v`
  returned exit 0 with 666/666 tests passing.
- Config: `.venv\\Scripts\\python.exe -m mfg_hedge check --config
  configs/v1_minimal.json` returned exit 0 and `status: ok`.
- Formal panel status is `physical_failed`: 977 complete episodes and 47
  failed episodes. Failure distribution was 40 `laedge:conservative` calls
  and 34 `laedge:preemptive` calls; all 74 recorded failures had
  `RuntimeError: attribution episode invariant failure:
  failed_replica_a_is_empty`. The two NIIN arms had no such failure.
- Because the frozen protocol is fail-closed, `paired_statistics.json` is
  explicitly `status: unavailable`, with reason `physical_failed`; no paired
  point estimate, CI, contrast, or difference-in-differences is reported as a
  valid experiment result.
- The read-only audit confirmed all four arm trace fingerprints were shared
  on successful rows and the formal call count exactly matched the budget.

#### Artifacts

- `artifacts/cancellation-semantics-ablation-20260909-r1/summary.json`
- `artifacts/cancellation-semantics-ablation-20260909-r1/paired_statistics.json`
- `artifacts/cancellation-semantics-ablation-20260909-r1/episode_rows.jsonl`
- `artifacts/cancellation-semantics-ablation-20260909-r1/manifest.json`
- `artifacts/cancellation-semantics-ablation-20260909-r1/protocol.json`

The failed artifact is retained and was not overwritten.

#### Decisions and risks

The panel did not produce a clean mechanism comparison and must not be used
for conclusions. The failure is concentrated in a LÆDGE-specific boundary
audit: idle-release pending work is not compatible with the fixed-dispatcher
`failed_replica_a_is_empty` attribution check. This is an execution/metric
compatibility defect discovered by the formal run, not evidence about
conservative versus preemptive cancellation. No retry is authorized under the
current ticket; a follow-up fix must preserve this failed artifact and rerun
only under a separately reviewed ticket/protocol decision.

#### Next

Create a narrow follow-up fix for the LÆDGE boundary-audit contract, then
obtain explicit direction on whether a new unique run is allowed; do not
interpret or publish the partial panel as a paired result.

### Update: 2026-09-09 — Blocked by ticket 07

Status: blocked

#### Goal

Pause the formal panel while the LÆDGE boundary-audit correction is handled
under a separate narrow ticket.

#### Changed

Ticket 06 was marked `blocked` with `Blocked by: 07`. No experiment artifact,
sample, or protocol field was changed.

#### Verification

The r1 physical-failed result remains fail-closed: `977` complete episodes,
`47` failed episodes, and no paired statistics.

#### Artifacts

The r1 artifact remains retained and immutable.

#### Decisions and risks

Only the boundary-audit correction was in scope for ticket 07; no physical or
statistical workaround was authorized.

#### Next

Resolve ticket 07, restore ticket 06 to `claimed`, and rerun the exact frozen
panel into a new unique r2 directory.

### Update: 2026-09-09 — Unblocked and resumed after ticket 07

Status: claimed

#### Goal

Resume the unchanged ADR-0032 r2 cancellation-semantics panel after the
LÆDGE boundary-audit correction.

#### Changed

Ticket 07 resolved the online boundary reconstruction defect. This ticket is
unblocked without changing the frozen namespace, macro seed, 1,024-episode
sample, four arms, CRN, or 4,096-call budget.

#### Verification

The correction passed focused, affected, full-suite, config, and 47-episode
r1 replay gates. The formal r2 run is authorized by the frozen protocol.

#### Artifacts

The r1 physical-failed artifact is retained. r2 will use a new unique run
directory and transactional commit.

#### Decisions and risks

No scheduling, cancellation, Replay, price, Reservation, workload, CRN, or
statistical protocol changes are made in this resumption.

#### Next

Run the required timing preflight and then the complete 1,024-episode r2
panel. Generate paired statistics only if all four arms succeed for all
episodes.

### Update: 2026-09-09 — Complete frozen r2 panel

Status: resolved

#### Goal

Complete the accepted ADR-0032 2×2 paired cancellation-semantics ablation
after ticket 07 corrected the LÆDGE online boundary audit, without changing
the frozen physical or statistical protocol.

#### Changed

- Ran the required timing preflight: 8 calls, `0.8174643000238575` seconds,
  `9.786360089078535` calls/second.
- Ran the exact frozen r2 panel in the namespace
  `replica-routing-baselines:cancellation-ablation:v1`, macro seed `20260931`,
  with 1,024 episodes and the four canonical arms.
- Consumed exactly `4,096/4,096` calls; no retry, supplement, deletion, or
  replacement occurred.
- Generated paired cluster statistics using 4,096 frozen bootstrap replicates
  at the episode-index cluster level.

#### Verification

- r2 status: `completed`; `1,024/1,024` episodes completed and `0` failed.
- All four arms have 1,024 rows; all 1,024 rows are completed and have one
  shared CRN trace fingerprint per episode.
- The corresponding r1/r2 episode rows have `1,024/1,024` matching trace
  fingerprints and episode indices; r2 changes only the execution outcome
  after the audit correction.
- All `1,024/1,024` rows passed CRN/admission checks and report the frozen mode
  difference `running_loser_cancellation_only`.
- r1 remains `physical_failed` with `977` complete and `47` failed episodes;
  its protocol bytes match r2 and its artifact was not modified.
- The 47-episode replay completed `188/188` calls with `47/47` fingerprints
  matched, `94/94` LÆDGE failed-start A-zero checks, and `188/188`
  attribution/invariant validations.
- Focused: `12/12` passed. Affected: `92/92` passed. Full suite:
  `667/667` passed. Config check: exit 0, `status: ok`.
- Paired statistics are available with `sample_count=1,024` for every metric
  and comparison. Bootstrap namespace/seed/replicates are recorded in
  `paired_statistics.json`; all claim flags remain false.

Aggregate arm results (episode-level means for work/counters and pooled token
latency summaries):

| arm | overall mean | overall P95 | overall P99 | D/F CVaR95 | D/F miss | H/R P99 | replay rate | total work | wasted work | hedge launches | winner rate | storm peak max | drain mean / max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| niin:conservative | 6.654652 | 23.811614 | 27.536486 | 27.230387 | 0.655734 | 23.611900 | 0.034338 | 459656.099 | 1708.837 | 7877 | 0.967881 | 2.0 | 3.836278 / 28.962046 |
| niin:preemptive | 6.560504 | 23.526617 | 27.244179 | 26.929690 | 0.650836 | 23.388239 | 0.033812 | 459109.055 | 1158.272 | 7877 | 0.963057 | 2.0 | 3.778713 / 28.962046 |
| laedge:conservative | 3.718737 | 10.313424 | 12.496672 | 12.879612 | 0.706099 | 8.604355 | 0.001882 | 541313.974 | 102764.647 | 82518 | 0.380462 | 3.0 | 2.458195 / 7.756234 |
| laedge:preemptive | 3.525521 | 10.112810 | 12.267548 | 12.639949 | 0.675955 | 8.411835 | 0.001879 | 514680.974 | 86887.521 | 125755 | 0.488593 | 3.0 | 1.660159 / 9.056222 |

Counters: every arm completed `458,212` tokens and passed all invariants.
Failed-running primary counts were NIIN `1,024`, LÆDGE conservative `948`,
and LÆDGE preemptive `983`; invalidated queued primaries were respectively
`14,817`, `14,578`, `0`, and `0`; stale completion events ignored were
`1,756`, `2,748`, `1,939`, and `127,478` in arm order above.

The five required paired contrasts are oriented as right minus left; every
row contains point, absolute difference, relative change, paired SE, and 95%
CI for all 13 frozen metrics:

| comparison | overall mean | D/F CVaR95 | D/F miss | total work | wasted work | hedge launches | H/R P99 | replay rate | storm peak | drain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NIIN preemptive − conservative | -0.094148 | -0.300697 | -0.004898 | -0.534223 | -0.537662 | 0 | -0.223661 | -0.000526 | 0 | -0.057565 |
| LÆDGE preemptive − conservative | -0.193216 | -0.239663 | -0.030144 | -26.008789 | -15.505005 | +42.223633 | -0.192520 | -0.000003 | +0.367188 | -0.798036 |
| LÆDGE conservative − NIIN conservative | -2.935915 | -14.350775 | +0.050365 | +79.744019 | +98.687314 | +72.891602 | -15.007545 | -0.032456 | +0.631836 | -1.378083 |
| LÆDGE preemptive − NIIN preemptive | -3.034983 | -14.289741 | +0.025119 | +54.269453 | +83.719971 | +115.115234 | -14.976404 | -0.031933 | +0.999023 | -2.118554 |
| cancellation × policy-family DiD | -0.099068 | +0.061034 | -0.025246 | -25.474566 | -14.967343 | +42.223633 | +0.031141 | +0.000523 | +0.367188 | -0.740470 |

Interpretation is mechanism-only: preemptive cancellation modestly reduces
latency, D/F tail, drain, and wasted work in both families, with a much larger
work/waste reduction for LÆDGE because it cancels many running losers. LÆDGE
has substantially lower latency and replay rate than NIIN but uses much more
total and wasted work and launches many more hedges. The DiD records the
difference in cancellation effect between policy families; it is not a policy
selection or optimality result.

#### Artifacts

- [r2 summary](D:/project/mfg_hedge_v1/artifacts/cancellation-semantics-ablation-20260909-r2/summary.json)
- [r2 paired statistics](D:/project/mfg_hedge_v1/artifacts/cancellation-semantics-ablation-20260909-r2/paired_statistics.json)
- [r2 episode rows](D:/project/mfg_hedge_v1/artifacts/cancellation-semantics-ablation-20260909-r2/episode_rows.jsonl)
- [r2 manifest](D:/project/mfg_hedge_v1/artifacts/cancellation-semantics-ablation-20260909-r2/manifest.json)
- [r2 protocol](D:/project/mfg_hedge_v1/artifacts/cancellation-semantics-ablation-20260909-r2/protocol.json)

The r1 artifact remains at
`artifacts/cancellation-semantics-ablation-20260909-r1` and was not
overwritten or deleted.

#### Decisions and risks

This panel is a paired mechanism attribution only. Preemptive arms are
explicitly `oracle_upper_bound_if_runtime_cannot_cancel`; no new algorithm,
price, Reservation, scheduling policy, optimality, Nash, or MFG claim is made.

#### Next

Use the r2 artifact for the planned cancellation-semantic interpretation
only. Any algorithm change or further population experiment requires a new
ticket and protocol decision.
