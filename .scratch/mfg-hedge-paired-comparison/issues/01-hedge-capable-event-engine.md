# Hedge-capable workload and event engine

Type: task
Status: resolved
Blocked by: none

## Goal

Extend the event engine to hedge-capable form per `../spec.md` sections 2-8: attempt-2 CRN, HEDGE_TIMER, the three actions, winner/loser/cancellation, and failure interactions — without any MFG solver, calibration, or comparison CLI.

## Scope

- `WorkloadTrace` gains optional `hedge_service_times` (attempt 2, stream label `service:{replica}:attempt2`); generator extension must not perturb arrival/class/attempt-0/attempt-1 streams; No-Hedge paths never read attempt 2.
- New engine module (hedge-capable) with the fixed same-time order: `DOMAIN_STATE_CHANGE` (including failure/replay substeps) → `COMPLETE` → `HEDGE_TIMER` → `TOKEN_ARRIVAL` → dispatch/start; a completion voids a pending timer; a failure creates the Replay and voids the timer before it can fire. Dispatch remains the final step per instant; spec section 3-8 semantics including fault-first at F, fixed Replay order, and the timer rules.
- Attempt statuses: `completed_winner`, `completed_loser`, `cancelled_queued`, `failed_running`, `invalidated_queued`.
- Conservative cancellation: queued loser cancelled, running loser completes and is discarded; loser work counted; loser failure never replays.
- Hedge↔Replay case law of spec section 7, including Primary lost before timer ⇒ Replay only, and Backup-domain-Failed ⇒ action suppressed and counted.
- No-Hedge mode through the new engine must match `simulate_common_state_no_hedge` Token-for-Token (same projection fields as the ticket-02 regression).

## Acceptance criteria

- Spec section 18 scenarios 1-15 pass as deterministic unit tests with exact expected values, including the split timer-vs-failure pair: 5a (Primary=B, Backup=A — no launch, counted suppressed) and 5b (Primary=A — Replay created, timer voided, no Hedge).
- CRN regression: arrivals/classes/attempt-0/attempt-1 identical with and without the attempt-2 stream.
- Invariants per Token: at most one Replay, at most one Hedge Backup, exactly one winner and one completion.
- `python -m unittest discover -s tests -v` green; `check` exit 0; no CLI/metrics/config changes; no third-party dependency.

## Progress log

### Update: 2026-09-04 — Hedge-capable event engine

Status: completed

#### Goal

Implement attempt-2 CRN and the hedge-capable deterministic event engine (Normal / Delayed / Immediate) per `../spec.md` sections 2-8 and ADR-0005 (now Accepted). No MFG solver, calibration, quota, paired CLI/config, comparison metrics, or experiments.

#### Changed

- `docs/adr/0005-hedge-copy-lifecycle-and-random-keys.md`: Status -> Accepted (confirmed by the user on 2026-09-04). ADR-0006 remains Proposed (its two open points belong to ticket 02 and were not touched here).
- `src/mfg_hedge/workload.py`: `WorkloadTrace.hedge_service_times` (optional, `[replica][token]`, attempt-2); `generate_workload_with_hedge` built on `generate_workload_with_replay` with stream label `service:{replica}:attempt2`; `validate_hedge_stream_shape` at the engine consumption boundary.
- `src/mfg_hedge/hedge_simulation.py`: `simulate_hedge_common_state(trace, timeline, degraded_slowdown, actions=None, hedge_delay=None, replica_count=2)`; event order `DOMAIN_STATE_CHANGE(1) -> COMPLETE(2) -> HEDGE_TIMER(3) -> TOKEN_ARRIVAL(4) -> dispatch pass`; failure/replay substeps inside the F transition; conservative winner/loser semantics with statuses `completed_winner/completed_loser/cancelled_queued/failed_running/invalidated_queued`; ADR-0005 case law (replay only when no surviving copy; timer voided by completion or replay; backup-domain-Failed => suppressed, never silent no-op); per-Token counters and invariant-safe recording.
- `src/mfg_hedge/__init__.py`: exported the new API; version 0.3.0 -> 0.4.0 (`pyproject.toml` synced) — additive public API, no existing behavior changed, per ADR-0002 versioning practice.
- `tests/test_hedge_simulation.py` (23 tests), written first.
- Bug found by red/green loop: the F handler re-recorded already-cancelled queued attempts as invalidated; fixed by skipping cancelled entries (test evidence: duplicate attempt records before the fix).
- `tests/test_common_state_metrics.py`: the artifact-consistency test now excludes `simulator_version` alongside `run_id`, per the Healthy stability contract (provenance may move across versions; scientific content must not).

#### Verification

- Red (before implementation): `Ran 153 tests ... FAILED (errors=1)` — `ModuleNotFoundError: No module named 'mfg_hedge.hedge_simulation'`.
- Green/Refactor: `.\.venv\Scripts\python.exe -m unittest discover -s .\tests -v` — `Ran 175 tests ... OK` (152 + 23 new).
- `.\.venv\Scripts\python.exe -m mfg_hedge check --config .\configs\v1_minimal.json` — exit 0.
- Scenario mapping (spec section 18): 1 `Scenario01` (Immediate: both copies at arrival, Primary first), 2 `Scenario02` (timer void after early completion), 3 `Scenario03` (timer fires; Hedge wins at 4.5; loser Primary dies at F with 14.0 executed work), 4 `Scenario04` (completion at timer instant wins; timer voided once), 5a `Scenario05...test_5a` (Primary=B: Backup domain Failed -> suppressed, no launch), 5b `test_5b` (Primary=A: Replay created, timer voided, no attempt-2), 6 `Scenario06` (queued loser `cancelled_queued`, executed 0), 7 `Scenario07` (running loser completes, work counted, drain 8.0), 8 `Scenario08` (Primary lost with live Hedge -> no Replay), 9 `Scenario09` (Primary lost before timer -> Replay only), 10 `Scenario10` (Immediate in F suppressed), 11 `Scenario111213...test_exactly_one_winner...` (one winner/completion per Token on a 300-Token mixed-action run), 12 implied by loser assertions in 3/6/7 (winner completion never overwritten), 13 `Scenario13` (stale completion ignored exactly once; timer voided exactly once), 14 `Scenario14...test_attempt2_stream...` (arrival/class/attempt-0/attempt-1 value-identical with and without attempt-2; attempt-2 differs), plus corrupted-NaN attempt-2 stream proving the No-Hedge path never reads it, 15 `Scenario15` (300-Token controlled trace; nine shared scientific fields match `simulate_common_state_no_hedge` Token-for-Token exactly).
- Extra: hedge-stream missing/wrong shape, illegal action values/ids, illegal tau0 (0/-1/NaN/inf/bool/None), non-dense ids, replica_count validation, determinism across runs, input trace immutability, counter algebra (`requested = launches + suppressed + timers_voided`).
- No experiment artifacts generated; no config changes; no third-party dependency.

#### Artifacts

- Source: `src/mfg_hedge/hedge_simulation.py` (new), `src/mfg_hedge/workload.py`, `src/mfg_hedge/__init__.py`, `pyproject.toml`.
- Tests: `tests/test_hedge_simulation.py` (new), `tests/test_common_state_metrics.py` (contract-aligned comparison fix).
- Docs: ADR-0005 status change only.

#### Decisions and risks

- `drain_end_time` is the last attempt terminal instant including losers (resource-truthful); Token `completion_time` remains the winner's.
- Cancelled queued attempts stay in their deque marked cancelled and are skipped at dispatch and at failure handling; this was the one real bug the red/green loop caught.
- Version bumped to 0.4.0 because new public API exists; no published scientific behavior changed (ticket-06 artifact test re-proves content stability across the bump).
- ADR-0006's open points (calibration rho coordinate, requested/applied quota contract) deliberately untouched; they belong to ticket 02.

#### Next

Claim ticket 02 (`02-action-stats-calibration.md`) after the user resolves ADR-0006's two remaining points.

## Answer

All acceptance criteria pass: attempt-2 CRN with zero perturbation to prior streams; hedge-capable engine with fixed same-time order and ADR-0005 case law; No-Hedge mode regresses Token-for-Token against the accepted engine; 175/175 tests green; `check` exit 0; no MFG/calibration/quota/CLI/config/comparison work; no artifacts generated; no third-party dependency.

## Comments

- 2026-09-04 (ticket 06): two boundary corrections landed after this ticket resolved — (1) queue snapshots at `failed_start`/`recovered_start` now count only live (non-cancelled) queued attempts; previously a cancelled tombstone was miscounted; (2) consumable service-stream values (attempt-0/1 always, attempt-2 when hedging) and trace arrival/class fields are validated before the event loop. No legal-run behavior changed. See `.scratch/mfg-hedge-paired-comparison/issues/06-harden-hedge-engine-boundaries.md`.

