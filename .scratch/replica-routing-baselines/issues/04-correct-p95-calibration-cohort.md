# Correct P95 calibration cohort and rerun protection comparison

Type: correction/experiment
Status: resolved
Blocked by: 03

## Scope

Correct P95 calibration to use only Healthy-arrival No-Hedge latencies from
the disjoint library. Preserve r1, rerun the five-arm panel as r2, and verify
all prior behavior.

## Acceptance criteria

1. Calibration sample count equals the Healthy-arrival cohort and excludes D/F/R arrivals.
2. The corrected delay is finite, positive, and frozen before evaluation.
3. r2 completes 1,024 paired episodes, 5,120 calls, and no failures without overwriting r1.
4. Focused, full-suite, and config checks pass.

## Progress log

### Update: 2026-09-09 — Correct P95 cohort and complete r2

Status: completed

#### Goal

Remove failure-window contamination from the P95 timer calibration and rerun
the complete paired protection panel without changing any other arm.

#### Changed

- `calibrate_p95_delay` now selects only Tokens whose arrival is in the Healthy
  phase of each disjoint No-Hedge calibration episode.
- Added a focused regression assertion that the recorded latency sample count
  exactly equals the independently reconstructed Healthy-arrival cohort.
- ADR-0031 and ticket 03 record the correction. Immutable r1 remains available
  as audit evidence; the corrected headline run is r2.

#### Verification

- Red: the strengthened focused test failed because calibration recorded all
  phase arrivals rather than the Healthy-arrival cohort.
- Focused final: `.venv/Scripts/python.exe -m unittest
  tests.test_protection_baseline_experiment -v` ran 6 tests, all `OK`.
- Corrected calibration used 256 independent episodes and 35,577 Healthy-
  arrival latency samples; the frozen P95 delay is 4.2507117440537625.
- Corrected r2 completed 1,024/1,024 episodes, exactly 5,120 evaluation calls,
  zero failures, and all five arm-level common invariants passed.
- Full: `.venv/Scripts/python.exe -m unittest discover -s tests -v` returned
  exit 0 with all 655 tests passing after the correction.
- Config: `.venv/Scripts/python.exe -m mfg_hedge check --config
  configs/v1_minimal.json` returned exit 0 and `status: ok`.
- Artifact audit confirmed five files, all claim flags false, and summary SHA-
  256 `5277f1bee19a99283a7e825da2fa2ec9aaa10132fd29312d8212c6cca27422ac`.

#### Artifacts

- `artifacts/protection-baselines-20260908-r2/summary.json`
- `artifacts/protection-baselines-20260908-r2/paired_statistics.json`
- `artifacts/protection-baselines-20260908-r2/episode_rows.json`
- `artifacts/protection-baselines-20260908-r2/manifest.json`
- `artifacts/protection-baselines-20260908-r2/metric_contract.json`

r1 was not modified or overwritten.

#### Decisions and risks

- Healthy-only calibration makes P95 a credible pre-failure latency baseline.
  Its delay fell from 35.0236 to 4.2507; this materially changes the P95 arm
  and is why r2, not r1, is the headline result.
- P95 remains native and unbudgeted, so its extra work and storm are part of
  the result rather than hidden by the repository's Reservation projector.
- The routing-after-placement panel remains unimplemented and unrun.

#### Next

Implement the separate batched multi-Expert/GPU routing panel.

## Answer

The corrected five-arm protection experiment completed and all acceptance
criteria are satisfied. Ticket 04 is resolved.

## Comments

- The ticket file was created after the correction test had been drafted in
  the same session, rather than before it. This is a workflow-ordering mistake;
  it is recorded here rather than hidden. No concurrent ticket was active and
  no artifact was overwritten.
