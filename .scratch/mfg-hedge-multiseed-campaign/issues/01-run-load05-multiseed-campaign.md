# Run the frozen load-0.5 paired multi-seed campaign

Type: task
Status: resolved
Blocked by: none

## Goal

Implement and execute the paired multi-seed campaign defined by `../spec.md`.

## Scope

- Add the smallest reusable campaign aggregation and CLI surface.
- Preserve all ticket-04 simulation, solver, calibration, quota, and metric
  behavior.
- Run the frozen schema-3 configuration exactly once as a full campaign.
- Do not implement load stress diagnostics in this ticket.

## Acceptance criteria

- All criteria in `../spec.md` pass.
- No existing artifact is modified.
- The final report records every configured seed, paired confidence intervals,
  resource/runtime evidence, and honest trade-offs.

## Progress log

### Update: 2026-09-05 — Load-0.5 paired 20-seed campaign

Status: completed

#### Goal

Add and execute the frozen load-0.5 paired multi-seed campaign without
changing ticket-04 scientific semantics.

#### Changed

- Added `src/mfg_hedge/campaign.py`: contiguous seed enumeration, one frozen
  production solution, complete per-seed same-trace A/B records, and paired
  Student-t 95% confidence intervals.
- Added `campaign-mfg-hedge-vs-no-hedge` and tests; version 0.11.0 -> 0.12.0.
- Extended `run_paired_comparison` with an explicit evaluation seed and an
  optional pre-gated frozen solution; defaults preserve the single-run API.
- Replaced the storm metric's O(bins * Tokens) rescans with exactly equivalent
  O(bins + Tokens + attempts) pre-binning. The first formal launch was stopped
  before artifact creation after this scaling defect was identified; no
  partial output existed. A scan-count regression test was added before the
  formal relaunch.

#### Verification

- Red: focused campaign tests failed because `mfg_hedge.campaign` and the CLI
  command did not exist; the storm scan-count test observed 10,001 Token scans.
- Green: `291 tests ... OK` after campaign implementation; `292 tests ... OK`
  after the linear storm fix; final suite later reached `295 tests ... OK`.
- `python -m mfg_hedge check --config configs/v1_minimal.json` exited 0.
- Real campaign: all 20 seeds 20260901..20260920, 100,000 Tokens per seed,
  completed with all arm invariants true and 20 distinct trace fingerprints.

#### Artifacts

- `artifacts/single_expert_mechanism_check-paired-campaign-load05-s20-t100000-20260905-night/`
  (`manifest.json`, `aggregate.json`, and `seed_*.json`; 22 files,
  113,512,907 bytes).

#### Decisions and risks

- The 1k mechanism check did not predict the long-horizon result. Across 20
  seeds, D/F arrival-cohort P99 improves, but R reuses the H Hedge policy and
  dominates the 100k horizon: overall mean/P95/P99 latency and work all worsen.
  This negative result is retained without seed selection or retuning.
- The aggregate is seed-level paired inference over the configured seed
  family, not a broader population claim.

#### Next

Run only the predeclared load-0.6/0.7 solver stress diagnostics, then design a
separate baseline ablation around persistent H/R hedging if the user approves.

## Answer

All acceptance criteria pass. The formal 20-seed campaign completed and the
transactional aggregate honestly records both the D/F protection benefit and
the dominant steady-state latency/work cost.
