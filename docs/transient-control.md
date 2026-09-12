# Transient development API (ADR-0012)

This is an admission-and-scoring slice, not a new MFG implementation or experiment
campaign. Old paired CLI and schema behavior remain unchanged. No new CLI is added.

## Minimal usage

Run with the project Python 3.10 environment. This example only returns in-memory
development results; it does not select a policy or write experiment artifacts.

```python
from dataclasses import replace
from mfg_hedge import (
    ATTRIBUTION_V1_PROTOCOL, ProtectionAction as A, ReservationParameters,
    TimeClassRule, generate_episode_trace, load_config,
    simulate_transient_episode, build_transient_objective,
)
from mfg_hedge.paired import compute_tau0

config = replace(load_config("configs/v1_minimal.json"), healthy_offered_load=0.45)
rule = TimeClassRule(A.DELAYED_HEDGE, A.IMMEDIATE_HEDGE,
                     A.DELAYED_HEDGE, A.IMMEDIATE_HEDGE)
runs = []
for episode_index in range(4):
    trace = generate_episode_trace(config, "transient-control:v1:fit", 10,
                                   episode_index, ATTRIBUTION_V1_PROTOCOL)
    runs.append(simulate_transient_episode(
        trace, rule, ReservationParameters(), hedge_delay=compute_tau0(config)))

admission_audit = runs[0].plan.audit()
score = build_transient_objective([result.run for result in runs])
```

## Contracts

- Four fixed actions in early-Regular, early-Urgent, late-Regular, late-Urgent
  order. The D midpoint separates early/late. H/F/R and D Primary-B are exact N.
  No arbitrary callback/queue-feedback controller is supported. The planner sees
  ordered TokenSpec metadata, never a trace with service streams. Its sequential
  prepass is equivalent to online arrival decisions for this restricted rule.
- Defaults: pooled D-window cap `0.25 * 0.45 * 25 = 2.8125`, one reserved mean
  work unit per admitted Hedge, no refunds, no class shares. At most two requests
  fit per default window. Short last windows scale by their actual duration.
  A separate ledger starts for every episode. Empty D windows are omitted from
  the audit with explicit coverage labeling. No equality of realized work is claimed.
- `plan.audit()` reports N/D/I requested/applied counts, quota suppressions,
  reservations and unused balances, globally, per class and per observed window/class.
  Engine requests count applied Hedge only. Engine suppression remains separate
  in `result.run.simulation.hedge_suppressed`.
- `build_transient_objective` first runs existing physical-result invariants,
  then pools raw outcomes. It returns weighted mean/SLO-excess/Replay loss,
  D/F fractional-tail CVaR95 and total/wasted executed work per generated Token.
  Work includes Primary/Replay/Hedge, including running losers through drain.
  Wasted work is additionally penalized, not a second resource ledger.
- Missing any phase/class produces `status=incomplete`, `total=None` and explicit
  missing cohorts. Do not rank incomplete results or drop sparse episodes.
  Duplicate keys and mismatched episode contracts fail. Coefficients are fixed
  untuned mechanism preferences, not measured monetary costs.
- `cvar95_fractional_tail` is deliberately separate from legacy `empirical_cvar95`.
  For 21 observations with largest values 100 and 10, values are respectively
  `100.5 / 1.05` and `55`.
- The runner permits `transient-control:v1:fit` and `transient-control:v1:test`
  only. No qualification/holdout data or automatic retry/seed replacement.
  The API assumes the known scheduled failure clock, not a learned detector.

## Not yet implemented

Independent safety qualification, new confirmatory gates, holdout execution,
reduced forward model and a new MFG algorithm. A lower development score does
not certify safety constraints, statistical significance or mean-field value.

## Frozen 81-rule development experiment

ADR-0013 activates one reproducible fit-only experiment. It consumes 64 exact
episodes, evaluates all 81 rules, applies the H/R and work safety filters, and
writes all rows transactionally:

```powershell
.\.venv\Scripts\python.exe -m mfg_hedge search-transient-rules `
  --config .\configs\v1_transient_development.json `
  --run-id transient-control-81-fit-m20260905-e64-20260905
```

The selected member is descriptive development evidence. The command cannot
generate qualification or holdout data and does not run an MFG algorithm.

## Scaled 8-Expert / dual-Backup experiment

ADR-0014 freezes an isolated fixed-Primary topology and about 150k Tokens:

```powershell
.\.venv\Scripts\python.exe -m mfg_hedge search-scaled-multi-backup `
  --config .\configs\v1_scaled_multi_backup_development.json `
  --run-id scaled-8e-3r-e128-m20260905-20260905
```

It compares No-Hedge, frozen single-Backup NIIN, and an 81-rule dual-Backup
development search. The second Backup consumes an additional reservation unit.
All eight Experts have separate three-Replica queues and separate ledgers. This
is still development fitting under a known common fault, not qualification or MFG.
