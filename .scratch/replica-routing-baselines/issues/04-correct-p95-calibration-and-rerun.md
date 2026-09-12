# Correct P95 calibration cohort and rerun protection panel

Type: correction/experiment
Status: resolved
Blocked by: none

## Scope

Correct the material baseline bias found in post-run audit: calibrate P95 from
independent No-Hedge Healthy-arrival observations rather than mixing D/F tail
latency into the timer. Preserve all evaluation episodes and other arm
semantics, then write a new run directory. Never overwrite r1.

## Acceptance criteria

1. Calibration records and uses only Healthy-arrival Token latency.
2. A test proves non-Healthy calibration observations are excluded.
3. The five-arm campaign reruns 1,024 episodes with zero failures and a fresh
   run-id; NIIN/requested-price and non-P95 arm outcomes remain stable.
4. Focused, full-suite, and config checks pass.

## Progress log

### Update: 2026-09-09 — Close duplicate ticket

Status: completed

#### Goal

Close the stale duplicate ticket without changing the completed P95 correction
or protection-baseline result.

#### Changed

- Marked this duplicate as resolved because the implemented work is recorded
  in `04-correct-p95-calibration-cohort.md`.
- No production code, experiment configuration, or artifact was changed.

#### Verification

- Read-only comparison confirmed the completed correction and r2 evidence are
  recorded in the cohort ticket.
- No experiment or test command was run in this administrative update.

#### Artifacts

None.

#### Decisions and risks

The separate duplicate file remains as historical ticket metadata; the
resolved cohort ticket is the authoritative progress record.

#### Next

Use a new ticket for the isolated cancellation-semantics ablation.

## Comments

- r1 remains immutable audit evidence but is not the headline P95 comparison.
