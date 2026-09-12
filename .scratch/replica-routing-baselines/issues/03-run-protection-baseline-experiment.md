# Run paired protection-baseline experiment

Type: implementation/experiment
Status: resolved
Blocked by: 02

## Scope

Implement the isolated LÆDGE episode backend and the frozen ADR-0031 campaign,
then run 1,024 paired evaluation episodes for Failover-only, P95, LÆDGE, NIIN,
and the requested-price candidate. Do not implement the routing/EPLB/LPLB
panel, change the Gate, retune the candidate, or make Nash/MFG claims.

## Acceptance criteria

1. LÆDGE hand cases prove work-conserving idle release, oldest-first service,
   immediate loser cancellation, fault-first Replay, complete drain, and CRN
   stream binding.
2. P95 calibration uses 256 disjoint No-Hedge episodes and freezes one finite
   positive delay before evaluation.
3. All five arms consume identical evaluation trace/fault identities; no arm
   mutates an episode.
4. The campaign completes 1,024 episodes and exactly 5,120 evaluation calls;
   failed episodes fail closed.
5. Summary reports common latency, D/F CVaR/miss, Replay, Hedge, total/wasted
   work, storm, paired differences versus NIIN, provenance, and false
   BR/regret/Nash/MFG flags.
6. Artifact write is transactional and never overwrites an old run.
7. Focused tests, full suite, and config check pass.

## Progress log

### Update: 2026-09-08 — Run paired protection-baseline experiment

Status: completed

#### Goal

Implement the missing work-conserving LÆDGE execution semantics and run the
five-arm paired protection experiment frozen by ADR-0031.

#### Changed

- Added `src/mfg_hedge/laedge_episode.py`, an isolated two-Replica LÆDGE event
  executor with idle-release scheduling, oldest-unserved priority, immediate
  running-loser cancellation, fault-first Replay on the surviving Replica,
  complete drain, and attempt-keyed CRN binding.
- Added `src/mfg_hedge/protection_baseline_experiment.py` with disjoint P95
  calibration, the 1,024-episode five-arm campaign, common metrics, paired
  episode-cluster bootstrap comparisons against NIIN, and transactional output.
- Added `CANCELLED_RUNNING` to the attempt-status vocabulary because LÆDGE
  cancels a live losing copy immediately; historical engines retain their
  conservative-cancellation behavior.
- Added six focused tests in `tests/test_protection_baseline_experiment.py`.
  Public API version changed 0.27.0 to 0.28.0; dependencies remain empty.
- Added ADR-0031 and updated the feature spec. The routing-after-placement
  panel was not implemented or run.

#### Verification

- Red: `.venv/Scripts/python.exe -m unittest
  tests.test_protection_baseline_experiment -v` failed with
  `ModuleNotFoundError: mfg_hedge.protection_baseline_experiment` before the
  implementation existed.
- Focused final: protection tests ran 6/6 `OK`; protection + kernel + adapter
  suites ran 31/31 `OK`.
- A 100-episode preflight completed 500/500 evaluation calls and exposed no
  failed episodes or physical failures.
- Formal campaign completed 1,024/1,024 paired episodes, exactly 5,120
  evaluation calls, and zero failed episodes. P95 calibration used 256
  disjoint episodes and 114,622 Token latency samples; frozen delay was
  35.02357615340072.
- Full: `.venv/Scripts/python.exe -m unittest discover -s tests -v` ran 655
  tests in 195.935 seconds, all `OK`.
- Config: `.venv/Scripts/python.exe -m mfg_hedge check --config
  configs/v1_minimal.json` returned exit 0 and `status: ok`, retaining the
  existing single-domain load warning.
- NIIN/requested-price paired outcome statistics reproduce the prior holdout
  exactly; total-work point difference differs only by 4.4e-16 from summation
  order.

#### Artifacts

- `artifacts/protection-baselines-20260908-r1/manifest.json`
- `artifacts/protection-baselines-20260908-r1/summary.json`
- `artifacts/protection-baselines-20260908-r1/episode_rows.json`
- `artifacts/protection-baselines-20260908-r1/paired_statistics.json`
- `artifacts/protection-baselines-20260908-r1/metric_contract.json`
- `src/mfg_hedge/laedge_episode.py`
- `src/mfg_hedge/protection_baseline_experiment.py`
- `tests/test_protection_baseline_experiment.py`
- `docs/adr/0031-protection-baseline-experiment.md`

No prior artifact was overwritten.

#### Decisions and risks

- LÆDGE is the latency leader but uses native aggressive duplication: versus
  NIIN its episode-level overall mean falls 47.24% and D/F CVaR95 falls 53.52%,
  while total work rises 12.00%, wasted work rises about 53 times, and D/F
  deadline-miss rate rises 2.89% relative. It is therefore not a free win.
- The requested-price candidate improves NIIN modestly: D/F CVaR95 falls
  0.987%, overall mean falls 1.47%, and Replay rate falls 1.22%, while wasted
  work rises 5.81%. This reproduces the prior conclusion that it misses the
  predeclared 10% tail-improvement target.
- P95 calibrated at 35.02 is too late for this transient: it is worse than
  NIIN on latency and uses more work. Failover-only saves a small amount of
  work but has substantially worse latency and Replay.
- Results are finite-system protection comparisons on one fixed topology, not
  routing-after-placement, hardware runtime, BR, Nash, or MFG evidence.

#### Next

Implement and run the separate routing-after-placement panel with batched
multi-Expert/GPU physics for RR, JSQ, least-work, and LPLB under fixed EPLB.

## Answer

All acceptance criteria are satisfied and the five-arm experiment completed
without failures. The ticket is resolved.

## Comments

- The routing-after-placement panel remains separate and unstarted.
- Post-run review found that r1 mixed D/F latency into P95 calibration. Ticket
  04 records the correction and fresh r2 run; r1 remains immutable audit data.
