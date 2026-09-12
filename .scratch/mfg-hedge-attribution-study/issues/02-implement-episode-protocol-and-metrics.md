# Implement the fixed-horizon episode protocol and metrics

Type: task
Status: resolved
Blocked by: 01

## Goal

Implement reusable independent H/D/F/R episodes, macro-seed grouping, and the
tail/SLO/recovery metrics required by `../spec.md`, without adding attribution
controllers or running holdout data.

## Scope

- Stop arrivals at 320, drain accepted work, and reset queues between episodes.
- Derive stable macro-seed/episode trace namespaces while preserving all existing
  attempt-level CRN and event semantics.
- Add empirical CVaR95, fixed SLO miss/excess, phase x class, boundary workload,
  recovery backlog/clear time, execution-phase x Replica work, and the required
  invariants.
- Preserve every existing single-trace API and artifact.

## Acceptance criteria

1. Hand-computed CVaR, excess, boundary backlog, recovery, and phase-split work
   fixtures pass, including attempts crossing state boundaries.
2. Two episodes never share queues or random keys; all arms within one episode
   share the exact trace.
3. H/D/F/R arrivals stop at 320 and every accepted attempt drains afterward.
4. Macro-seed aggregation pools 50 episodes internally but exposes one statistical
   observation; Token metrics pool Tokens, rates sum raw components, and time-
   series/boundary/recovery metrics are computed per episode then averaged.
   Independent episode clocks are never stacked, and Token/episode counts are
   not reported as inferential n.
5. Existing 303 tests plus the new focused tests and environment check pass.

## Progress log

### Update: 2026-09-05 — Implement fixed-horizon episodes and attribution metrics

Status: completed

#### Goal

Implement the attribution study's reusable fixed-horizon episode protocol and
per-episode/macro-seed metrics without adding controllers, calibration, campaign
execution, or holdout evaluation.

#### Changed

- Added `src/mfg_hedge/attribution_episode.py`: canonical H/D/F/R timeline
  100/200/220 with arrival cutoff 320, exact half-open workload generation,
  full SHA-256 episode seeds and stable namespaces, immutable episode traces,
  F/R boundary snapshots, fresh-engine execution, and complete post-cutoff drain.
- Added `src/mfg_hedge/attribution_metrics.py`: empirical CVaR95, strict SLO
  miss/excess, arrival/completion phase-by-class cohorts and migration, execution
  work by physical phase/Replica/attempt kind, boundary and recovery metrics,
  fixed-window protection-storm metrics, post-cutoff audit fields, and extensive
  fail-fast consistency checks.
- Extended `src/mfg_hedge/hedge_simulation.py` with an internal instrumentation
  path for boundary snapshots while preserving the public function signature,
  return type, event ordering, and existing queue-length semantics.
- Added public exports in `src/mfg_hedge/__init__.py` and raised the package
  version from 0.14.0 to 0.15.0 in `src/mfg_hedge/__init__.py` and
  `pyproject.toml` for the new public attribution API.
- Added 31 focused tests in `tests/test_attribution_episode.py` and
  `tests/test_attribution_metrics.py`.

#### Verification

- Red before implementation:
  `.\.venv\Scripts\python.exe -m unittest tests.test_attribution_episode tests.test_attribution_metrics -v`
  -> `Ran 2 tests ... FAILED (errors=2)` with
  `ModuleNotFoundError: No module named 'mfg_hedge.attribution_episode'`.
- Focused final verification:
  `.\.venv\Scripts\python.exe -m unittest tests.test_attribution_episode tests.test_attribution_metrics -v`
  -> `Ran 31 tests ... OK`.
- Required full verification:
  `.\.venv\Scripts\python.exe -m unittest discover -s .\tests -v`
  -> `Ran 334 tests in 23.732s ... OK` (303 existing + 31 new).
- Required environment check:
  `.\.venv\Scripts\python.exe -m mfg_hedge check --config .\configs\v1_minimal.json`
  -> exit 0, `status: ok`, Python 3.10.11; the pre-existing single-domain
  failure-headroom audit warning remains unchanged.
- Three independent read-only audits found no remaining reproducible blocker;
  additional mixed-action and boundary probes passed the metric invariants.

#### Artifacts

- `src/mfg_hedge/attribution_episode.py`
- `src/mfg_hedge/attribution_metrics.py`
- `src/mfg_hedge/hedge_simulation.py`
- `src/mfg_hedge/__init__.py`
- `pyproject.toml`
- `tests/test_attribution_episode.py`
- `tests/test_attribution_metrics.py`
- No experiment artifacts were generated.

#### Decisions and risks

- Formal episode identity is
  `attribution:v1:holdout:<macro_seed>:episode:<episode_index>`; full SHA-256
  derived seeds and all attempt-0/1/2 streams are fingerprinted.
- Protection-storm headline metrics use the fixed `[0, 320)` interval; delayed
  launches and execution after cutoff are reported separately so drain length
  cannot dilute the headline rate.
- Physical R-phase work includes drain for lifecycle accounting, while fixed-R
  utilization uses only `[220, 320)` and post-cutoff work remains separate.
- Macro aggregation requires exactly 50 independent episodes. Token metrics are
  pooled, raw work/rate components are summed, episode-clock quantities are
  averaged per episode, and the only inferential observation is the macro seed
  (`inferential_n = 1`).
- This ticket deliberately does not implement attribution arms/controllers,
  ActionStats calibration, the 40-macro campaign, or holdout analysis.

#### Next

Claim ticket 03 to implement the frozen attribution arms and controller logic on
top of this episode/metric substrate.

## Answer

Resolved. The fixed-horizon episode protocol and its per-episode/macro-seed
measurement layer are implemented and verified. Existing 303 tests remain green,
31 focused tests were added, and no experiment output was produced.

## Comments
