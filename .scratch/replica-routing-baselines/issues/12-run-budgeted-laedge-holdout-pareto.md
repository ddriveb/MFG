# Run Budgeted LÆDGE v2 holdout Pareto panel

Type: experiment
Status: resolved
Blocked by: none

## Scope

Run the formal holdout Pareto panel using only the confirmed v2 development
mapping from ticket 11.  This ticket must not recalibrate, change rates,
change priority, replace underfilled points, or read holdout outcomes to tune
the policy.  It produces a finite-system descriptive Pareto result only; it
does not make optimality, BR, Nash, or MFG claims.

## Frozen holdout protocol

- Namespace:
  `replica-routing-baselines:budgeted-laedge:v2:holdout`.
- Macro seed: `20260913`.
- Episode count: `1,024`; each episode runs the 11 unique arms from ADR-0033.
- Call budget: exactly `1,024 × 11 = 11,264`; failures count, with no retry,
  supplementation, deletion, or episode replacement.
- The seven fixed target labels use the confirmed rates from ticket 11:
  `0.000000, 0.029794334898597338, 0.06752794601883591,
  0.09304612180547656, 0.17872563347189496, 0.2770264334252587, 4.0`.
- The 18% point remains `saturation_limited`; target labels are never used as
  achieved work deltas.
- All arms in one episode share its immutable workload, fault, and attempt-key
  CRN trace.  The holdout namespace/seed is disjoint from development.
- The runner performs a separate preflight on a non-formal preflight namespace;
  it does not consume formal episode identities or alter the 11,264 calls.
- Paired cluster bootstrap uses `4,096` frozen replicates and episode index as
  the independent cluster.  It consumes no scheduler calls.
- The artifact must verify the confirmed development mapping fingerprint,
  protocol fingerprint, and the frozen source/physics contract before the first
  formal scheduler call.

## Required comparisons and metrics

Report actual holdout work deltas, not target labels, and include the routing
versus Hedge distinction:

- Budgeted delta=0 versus fixed No-Hedge;
- each positive budget versus delta=0;
- each budget versus NIIN conservative;
- rate 4.0 versus unconstrained conservative;
- unconstrained preemptive oracle versus unconstrained conservative.

Report the frozen latency, D/F, H/R, Replay, work, Hedge, suppression, storm,
drain, and per-window planned/committed/launch-cohort-realized/
execution-time-realized/overshoot/unused-budget fields.  Compute tail gain per
extra realized work and mark dominated/Pareto points using D/F CVaR95, D/F
miss, realized work, and Protection-Storm peak.

## Acceptance criteria

1. Red tests precede production changes and cover fingerprint guards, fixed
   mapping use, preflight identity isolation, exact 11-arm/call accounting,
   paired bootstrap determinism, and fail-closed incomplete panels.
2. A preflight reports calls, calls/s, estimated wall time, and memory before
   formal execution.
3. Formal holdout has exactly 11,264 attempted calls and 1,024 complete
   episodes, or fails closed without publishing statistics/Pareto results.
4. Artifact writing is transactional and non-overwriting.
5. Focused, affected, full suite, and config checks pass.

## Progress log

### Update: 2026-09-09 — Claimed confirmed-mapping holdout

Status: claimed

#### Goal

Use the confirmed v2 mapping once on fresh holdout episodes and report the
finite-system work/latency Pareto panel without any retuning.

#### Changed

- Created and claimed this ticket.
- No holdout scheduler call or artifact has been created yet.

#### Verification

- Confirmed ticket 11 is resolved with a confirmed mapping and ticket 10/r1
  remains preserved separately.
- Confirmed the formal holdout arithmetic is `11,264` calls.

#### Artifacts

No holdout artifact yet.

#### Decisions and risks

The confirmed mapping is read-only.  Preflight is separate from formal episode
identity.  Any fingerprint, CRN, invariant, or completeness failure remains
fail-closed.

#### Next

Write failing holdout contract tests, implement the isolated runner, run the
timing preflight, and then execute the formal holdout only if all guards pass.

### Update: 2026-09-09 — v2 holdout completed

Status: resolved

#### Goal

Run the confirmed v2 mapping on 1,024 fresh holdout episodes with all 11 arms,
paired cluster bootstrap, and no recalibration or holdout-driven tuning.

#### Changed

- Added the isolated confirmed-mapping holdout runner in
  `src/mfg_hedge/budgeted_laedge_holdout.py`.
- Added mapping, protocol, frozen source-bundle, and derived physics
  fingerprint guards before formal calls.
- Added an 8-episode preflight on a separate namespace and Windows RSS
  measurement through the Windows process API.
- Added actual-work delta labels, per-window budget audit summaries, the five
  required mechanism comparisons, deterministic 4,096-replicate paired
  episode-cluster bootstrap, tail-gain-per-work, and dominated/Pareto labels.
- No scheduler semantics, mapping rate, priority, CRN, workload, or historical
  artifact was changed.

#### Verification

- Real Red before implementation: `ModuleNotFoundError` for the new holdout
  module.
- Focused holdout/core/campaign suites: `30/30` passed.
- Full suite: `697/697` passed.
- Config check: `status: ok`.
- Preflight: `88` non-formal calls, `31.65170919684054 calls/s`, estimated
  formal time `355.8733567893505s`, peak RSS `29,958,144` bytes, zero failures.
- Formal panel: `1,024` episodes, exactly `11,264/11,264` calls, zero failed
  arms/episodes, no retry/supplement/deletion, elapsed
  `345.72748140001204s`, throughput `32.58057460282534 calls/s`, peak RSS
  `29,650,944` bytes.
- Mapping fingerprint, protocol fingerprint, source-bundle fingerprint, and
  physics fingerprint all matched the confirmed v2 development artifact.
- Bootstrap used `4,096` fixed paired replicates with fingerprint
  `8ae103be928bd8fe9befe4860ac6ae1faf0c3c43e48ac5d5d42c351bb4c691e4`.

Key holdout means:

| arm | D/F CVaR95 | D/F miss | total work | storm peak |
|---|---:|---:|---:|---:|
| fixed No-Hedge | 34.0836 | 0.6361 | 448.2117 | 0.0000 |
| NIIN conservative | 27.2600 | 0.6539 | 449.0262 | 1.3213 |
| Budgeted δ=0 | 12.6586 | 0.6662 | 448.8682 | 0.0000 |
| Budgeted δ=3% | 12.7344 | 0.6776 | 459.6696 | 1.0000 |
| Budgeted δ=5% | 12.8315 | 0.6862 | 469.1491 | 1.4004 |
| Budgeted δ=8% | 12.9362 | 0.7011 | 486.3749 | 1.7061 |
| Budgeted δ=12% | 12.9583 | 0.7077 | 501.2227 | 1.8594 |
| Budgeted δ=18% / rate 4.0 | 13.0503 | 0.7138 | 528.8797 | 1.9688 |
| unconstrained conservative | 13.0503 | 0.7138 | 528.8797 | 1.9688 |
| unconstrained preemptive oracle | 12.7838 | 0.6785 | 502.7730 | 2.3467 |

Budgeted δ=0 and δ=1 had identical physical results and realized work
`-0.0352%` versus NIIN, confirming that the fractional rate did not cross the
one-unit admission threshold.  The 18% point remained saturation-limited with
realized work delta `17.7837%` versus NIIN.  No budget point overfilled its
target label in the development confirmation; holdout horizontal coordinates
use realized work deltas only.

#### Artifacts

- [holdout ticket](D:/project/mfg_hedge_v1/.scratch/replica-routing-baselines/issues/12-run-budgeted-laedge-holdout-pareto.md)
- [holdout runner](D:/project/mfg_hedge_v1/src/mfg_hedge/budgeted_laedge_holdout.py)
- [holdout tests](D:/project/mfg_hedge_v1/tests/test_budgeted_laedge_holdout.py)
- [summary](D:/project/mfg_hedge_v1/artifacts/budgeted-laedge-holdout-20260909-v2-r1/summary.json)
- [paired statistics](D:/project/mfg_hedge_v1/artifacts/budgeted-laedge-holdout-20260909-v2-r1/paired_statistics.json)
- [episode rows](D:/project/mfg_hedge_v1/artifacts/budgeted-laedge-holdout-20260909-v2-r1/episode_rows.jsonl)
- [manifest](D:/project/mfg_hedge_v1/artifacts/budgeted-laedge-holdout-20260909-v2-r1/manifest.json)
- [protocol](D:/project/mfg_hedge_v1/artifacts/budgeted-laedge-holdout-20260909-v2-r1/protocol.json)

#### Decisions and risks

The panel is a finite-system descriptive Pareto result.  It does not claim
optimality, best response, Nash, or MFG.  The apparent latency gain of
Budgeted δ=0 is primarily dynamic Primary routing; positive Hedge budgets add
work and storm while the D/F miss rate rises in this holdout.  Rate 4.0 matches
the unconstrained conservative result on this panel and is an attainable
saturation approximation, not exact 18% work attainment.

#### Next

Do not retune from this holdout.  Any price-guided or Token-MFG work requires
an independent design/implementation ticket and new protocol authorization.
