# Implement and run the scaled development experiment

Type: implementation and experiment
Status: resolved
Blocked by: none

## Scope

Implement exactly ADR-0014/spec, with a new isolated workload, simulator,
planner, metrics/search, CLI and artifact. Preserve all historical APIs/configs.

## Acceptance criteria

1. Stable independent arrival/class/Expert/attempt-0/1/2/3 streams; same trace
   across all arms; dense global and per-Expert IDs; deterministic fingerprints.
2. Eight independent Expert queue triplets, fixed Primary semantics, common
   failure, balanced F/Replay destination, dual launch, first-winner and
   conservative loser handling; exact event-order fixtures.
3. Per-Expert reservation; charge 1/2 for one/two Backup copies; no refund;
   same work envelope, requested/applied/suppressed identities.
4. No-Hedge regression fixture, one-vs-two Backup lifecycle fixtures, work
   conservation/result invariants, no future draws in planning.
5. Frozen 128-episode boundary and 81 unique dual rules; correct safety filters,
   deterministic selection, complete comparison and no overwrite.
6. Red/Green focused tests, full suite and both environment/config checks pass.
7. Execute the complete experiment without tuning and retain positive or
   negative artifact plus readable report. Generate no qualification/holdout data.

## Progress log

- Claimed the ticket after freezing ADR-0014 and the feature spec. Kept the
  implementation isolated from all historical one-Expert simulators and
  artifacts.
- Red: added the first scaled-generation test, then ran
  `.venv/Scripts/python.exe -m unittest tests.test_scaled_experiment -v`;
  it failed with `ModuleNotFoundError: mfg_hedge.scaled_experiment`.
- Green: implemented the deterministic eight-Expert trace, three-Replica event
  engine, one/two-Backup reservation, search/metrics, CLI and transactional
  artifact. Focused suite ended at 12 tests passing.
- Refactored the rule search to cache audit results. A four-episode 81-rule
  smoke run fell from 27.446 seconds to 9.808 seconds with identical selected
  rule, Token count, feasibility and finite outputs.
- Ran the frozen command
  `.venv/Scripts/python.exe -m mfg_hedge search-scaled-multi-backup --config configs/v1_scaled_multi_backup_development.json --run-id scaled-8e-3r-e128-m20260905-20260905`.
  It completed 128 episodes and 147,585 Tokens, evaluated all 81 dual rules,
  and selected NIII.
- Reconstructed the complete run a second time in memory. Rule rows, trace
  fingerprints, comparisons and selected rule were identical.
- Verified the existing run directory is rejected with `FileExistsError`, so
  the retained artifact cannot be overwritten.
- Full regression:
  `.venv/Scripts/python.exe -m unittest discover -s tests -v` returned
  `Ran 385 tests ... OK`.
- Both checks passed: `check --config configs/v1_minimal.json` returned status
  ok with the pre-existing single-domain headroom warning; the scaled config
  returned status ok with no audit findings.

## Answer

The scaled development run is complete. Frozen single-Backup NIIN reduced the
objective by 26.27%, D CVaR95 by 26.62% and F CVaR95 by 24.96% against
No-Hedge, with 2.16% additional executed work. The sample-best dual-Backup NIII
reduced the same quantities by 15.16%, 14.96% and 15.14%, but used 3.64% more
work than No-Hedge.

Dual Backup was therefore worse than frozen single Backup by 15.07% in the
objective and 70.45% in wasted work. The same-rule NIIN attribution reaches the
same conclusion: under the shared prospective-work envelope, a dual launch
costs twice the reservation, protects fewer Tokens, and creates more loser
work. This is development evidence on the selection trace, not a qualification
claim. The detailed report is
`.scratch/scaled-multi-expert-backup/scaled-experiment-report.md`.

## Comments

- The topology details not expressible by only three counts are frozen in
  ADR-0014 before code: fixed Primary, domains A/B/C, uniform Gate, maximal
  common failure, per-Expert ledgers, and equal prospective work envelope.
