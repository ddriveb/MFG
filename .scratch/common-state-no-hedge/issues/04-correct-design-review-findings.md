# Correct design review findings

Type: task
Status: resolved
Blocked by: none

## Goal

Apply the eight design-review corrections to `../spec.md`, `docs/adr/0004-common-state-failure-semantics.md`, and tickets 01-03. Documentation only: no Python production code, tests, configuration files, or artifacts are modified; ticket 01 is not claimed.

## Scope

1. Per-Replica capacity feasibility (queues are per-Replica): report per-Replica arrival/speed/ratio per window (H/R: 0.7/1.0/0.7 both; D: Replica 0 0.7/0.5/1.4, Replica 1 0.7; F: Replica 0 inactive, Replica 1 1.4/1.0/1.4); aggregate ratio descriptive-only; system `capacity_violation` = any traffic-carrying Replica ratio > 1, so defaults are D=true, F=true, H/R=false.
2. Separate `CommonState` (H/D/F) from metrics Phase: new pure `phase_at(t)` returning H/D/F/R; naming fixed as `completion_phase`, never `completion_state`.
3. `replay_rate = replayed_tokens / generated_tokens`; `replay_executions` counted separately; `completed_tokens == generated_tokens` remains an invariant, not a denominator.
4. Drain reporting: `last_arrival_time`, `drain_end_time`, `drain_duration`; explicit observation intervals and denominators; end-to-end throughput is not fixed-window capacity.
5. Healthy regression scenario 15: boundaries later than the same trace's Healthy-baseline last completion (or a never-trigger fixture), proving identical `TokenResult`s.
6. Fixed Replay order at `failed_start`: running Token first, then queued Tokens in original FCFS order, enqueued on Replica 1 in that order ahead of same-time arrivals.
7. Work amplification: add `nominal_primary_work`; `extra_execution_ratio = executed_replay_work / nominal_primary_work`; add `execution_amplification = total_executed_work / nominal_primary_work`; state the denominator rule.
8. Statistical interpretation: 100/200/220 stays as mechanism-verification default; per-phase `sample_count`; P99 at ~140 samples flagged as diagnostic only.

## Acceptance criteria

- All eight corrections are reflected consistently in spec, ADR-0004, and the affected tickets (01 for `phase_at`, 02 for the event engine, 03 for metrics/CLI).
- ADR-0004 remains Proposed; tickets 01-03 remain open.
- `python -m unittest discover -s tests -v` and `python -m mfg_hedge check --config configs/v1_minimal.json` still pass, proving no code changed.

## Progress log

### Update: 2026-09-03 — Correct design review findings

Status: completed

#### Goal

Apply the eight design-review corrections to the Common State + No Hedge design documents without touching any production code, tests, configuration, or artifacts.

#### Changed

- `../spec.md`: rewritten with (1) per-Replica feasibility table and `capacity_violation` = any traffic-carrying Replica ratio > 1 (defaults D=true, F=true, H/R=false; aggregate ratio descriptive-only); (2) State/Phase separation with pure `phase_at(t)` and the mandatory field name `completion_phase`; (3) `replay_rate = replayed_tokens / generated_tokens` with `replay_executions` separate and `completed_tokens == generated_tokens` as invariant; (4) drain reporting (`last_arrival_time`, `drain_end_time`, `drain_duration`) and explicitly labeled observation intervals/denominators; (5) scenario 15 corrected to boundaries beyond the Healthy-baseline last completion (or never-trigger fixture); (6) fixed Replay order at `failed_start` (running first, then queued FCFS, ahead of same-time arrivals) as new scenario 16; (7) `nominal_primary_work`, `extra_execution_ratio = executed_replay_work / nominal_primary_work`, `execution_amplification = total_executed_work / nominal_primary_work`, and the positive-denominator rule; (8) per-phase `sample_count` and the P99 diagnostic-only caveat; section 11 reduced to the three still-open questions.
- `docs/adr/0004-common-state-failure-semantics.md`: revised decisions 1, 3, 5, 6, 7 to match; Status stays Proposed.
- Ticket 01: `phase_at` added to scope and boundary tests (phase R vs state H).
- Ticket 02: result type exposes `completion_phase`; acceptance updated to scenarios 1-10 and 12-16 including the fixed Replay order and the corrected regression.
- Ticket 03: metrics scope and acceptance updated (per-Replica ratios, D/F violation true, drain fields, new ratio definitions, per-phase `sample_count`).
- No Python production code, test code, configuration, or artifacts were modified.

#### Verification

- Re-read before editing: `AGENTS.md`, `CONTEXT.md`, `docs/agents/issue-tracker.md`, `docs/agents/update-format.md`, `../spec.md`, tickets 00-03, ADR-0004.
- `.\.venv\Scripts\python.exe -m unittest discover -s .\tests -v` — `Ran 57 tests ... OK` after the documentation edits.
- `.\.venv\Scripts\python.exe -m mfg_hedge check --config .\configs\v1_minimal.json` — exit 0, `status: ok`.

#### Artifacts

- `.scratch/common-state-no-hedge/spec.md`
- `.scratch/common-state-no-hedge/issues/04-correct-design-review-findings.md` (this ticket)
- `.scratch/common-state-no-hedge/issues/01-common-state-and-work-requirements.md`
- `.scratch/common-state-no-hedge/issues/02-failure-replay-event-engine.md`
- `.scratch/common-state-no-hedge/issues/03-fault-metrics-cli-and-artifacts.md`
- `docs/adr/0004-common-state-failure-semantics.md` (still Proposed)

#### Decisions and risks

- Under fixed parity dispatch the Degraded window is genuinely infeasible on Replica 0 (ratio 1.4); the design now reports this instead of hiding it behind aggregate capacity.
- `phase_at` is metrics-only vocabulary; engine state checks keep using `state_at` (CommonState H/D/F).
- ADR-0004 remains Proposed and awaits user confirmation; tickets 01-03 remain open and unclaimed.

#### Next

User reviews the corrected design and ADR-0004; on confirmation, ticket 01 can be claimed for implementation.

## Answer

All eight review findings are corrected consistently across spec, ADR-0004 (still Proposed), and tickets 01-03 (still open). No production code, tests, configuration, or artifacts were modified; 57/57 tests and the `check` command pass after the edits.

## Comments

