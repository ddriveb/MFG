# Implement finite reliability-aware multi-Replica routing

Type: implementation
Status: resolved
Blocked by: none

## Scope

Implement the isolated finite-system slice specified by ADR-0035 and
`.scratch/reliability-aware-multi-replica-routing/spec.md`.

## Frozen numerical protocol

The following values are frozen before any trace generation. They are scoped
to the non-inferential mechanism smoke and are not a development or holdout
experiment.

- Topology: one Logical Expert, Replica IDs `0..7`, four domains
  `domain = replica_id % 4`, nominal speed `1.0`.
- Workload: evaluation horizon `[0, 40)`, history-only burn-in `200`,
  deterministic seeded micro-batch stream with target offered load `5.6`
  Tokens/time, batch sizes in `{1, 2, 3, 4}`, Regular probability `0.8`,
  Urgent probability `0.2`, Regular deadline `4.0`, Urgent deadline `2.0`.
- Service: LogNormal potential work with mean `1.0` and CV `0.5`; work is
  keyed by `(namespace, macro_seed, scenario, episode, token_id, replica_id,
  attempt_id)` and is never exposed to a policy.
- Health: individual UP/DOWN paths use exponential failure durations with
  baseline hazard `0.01` and mean repair duration `6.0`; domain shocks use
  hazard `0.005` and mean repair duration `8.0`. S1 sets individual hazards
  of Replicas `0,1` to `0.04`; S2 switches that pair to `2,3` at `t=20`;
  S3 uses domain shock hazard `0.03`. Paths are finite, end UP, and are
  generated before policy evaluation.
- Estimator: trailing history window `W_h=50.0`, burn-in `200.0`,
  `alpha_h=1.0`, `beta_h=100.0`; individual and domain posterior means are
  added. Risk-aware penalties are `gamma_regular=4.0` and
  `gamma_urgent=6.0`.
- Identity: smoke namespace
  `replica-routing-baselines:reliability-aware-multi-replica-routing:v1:smoke`,
  macro seed `20260909`, SHA-256 stream keys as specified by the spec. The
  source fingerprint covers the accepted ADR, feature spec, this ticket, the
  isolated implementation module, and its focused tests.
- Sampling: scenarios `S0,S1,S2,S3`, episodes `0..7`, policies
  `uniform_rr,jsq,loew,reliability_only,risk_aware_jsq,lazarus_algorithm1`;
  exactly `4 * 8 * 6 = 192` complete scheduler calls. Failed calls are not
  retried, removed, or supplemented. The smoke is non-inferential.

These values are a mechanism-smoke contract only. No formal sample size,
confidence interval, policy selection, placement claim, MFG claim, or
Hedge/price result is authorized by this ticket.

Before changing production code, append to this ticket the frozen numerical
protocol: scenario hazards and repair laws, arrival and batch distributions,
deadlines, service law, history window and prior, burn-in, policy coefficients,
namespace, macro seed, source fingerprints, and the exact 192-call smoke
budget.

Then use red-green-refactor to implement:

1. immutable eight-Replica workload, service, and individual/domain fault
   traces with stable CRN identities;
2. the isolated single-copy event engine with repeated reactive Replay and
   complete drain;
3. the six frozen routing policies, including the batch-faithful Lazarus
   Algorithm 1 adaptation;
4. finite metrics, per-Replica audits, invariants, and deterministic
   provenance;
5. the bounded non-inferential 192-call mechanism smoke.

## Acceptance criteria

- Numerical scenario and sampling values are frozen before implementation.
- Red tests cover event order, health eligibility, no hidden-work leakage,
  Replay ordering, correlated failure, estimator causality, and CRN identity.
- All six policies receive identical immutable workload and fault traces.
- Every physical and scientific invariant in the specification is enforced.
- The smoke consumes no more than 192 complete scheduler calls and is labelled
  non-inferential.
- Existing two-Replica engines, configurations, public behavior, and
  historical artifacts remain unchanged.
- No Hedge, price, placement optimizer, formal evaluation, best response,
  Nash, MFG, or MFC solver is introduced.

## Progress log

### Update: 2026-09-09 — Implement finite routing and mechanism smoke

Status: completed

#### Goal

Implement the isolated ADR-0035 eight-Replica, single-copy causal routing
engine, six deterministic policies, physical/invariant audit, and the bounded
non-inferential 192-call mechanism smoke. Do not enter development/holdout,
policy selection, Hedge, price, placement, or MFG work.

#### Changed

- Accepted ADR-0035 and activated this ticket with the frozen numerical
  protocol recorded above; the historical two-Replica engines remain
  untouched.
- Added `src/mfg_hedge/reliability_aware_routing.py` with immutable workload,
  service and individual/domain fault identities, stable SHA-256 CRN streams,
  causal Gamma-Poisson hazard estimates, failure-first event processing,
  queued/running attempt ledgers, repeated reactive Replay, complete drain,
  and one-live-attempt enforcement.
- Implemented `uniform_rr`, `jsq`, `loew`, `reliability_only`,
  `risk_aware_jsq`, and batch-faithful `lazarus_algorithm1`. Policies see
  only current queue/health/history observations; service draws and future
  paths remain hidden.
- Added `tests/test_reliability_aware_routing.py` covering CRN identity,
  topology, fault-first completion, queued displacement, domain failure,
  repeated Replay, hazard-prefix causality, policy completion, metrics, and
  transactional smoke output.
- Added per-class latency/deadline metrics, per-Replica assignment/start/
  completion/failure/displacement/busy/availability audit, and invariant
  counters. No public package export or version bump was made because this is
  an isolated internal feature API.

#### Verification

- Real Red: initial focused run failed before implementation with
  `ModuleNotFoundError: No module named 'mfg_hedge.reliability_aware_routing'`;
  subsequent red runs exposed and fixed the batch-rate/drain, Lazarus
  displaced-work, and queued-attempt-ledger defects.
- Focused: `.venv/Scripts/python.exe -m unittest
  tests.test_reliability_aware_routing -v` — `10/10` passed.
- Affected routing suite: `tests.test_replica_routing_baselines` — `15/15`
  passed; affected protection suite:
  `tests.test_protection_baseline_experiment` — `6/6` passed.
- Full suite: `.venv/Scripts/python.exe -m unittest discover -s tests -v` —
  `723/723` passed.
- Config check: `.venv/Scripts/python.exe -m mfg_hedge check --config
  configs/v1_minimal.json` — exit `0`, `status: ok`.
- Smoke execution: exactly `4 * 8 * 6 = 192` calls completed with no retries,
  no removed/supplemented episodes, 192/192 completed Token sets, zero
  `down_starts`, zero `completed_without_winner`, 32 paired trace identities,
  and 32 paired token identities. Each of the six policies has 32 rows across
  S0–S3; the four scenarios each have eight episodes.
- The artifact audit confirmed every policy row retained its scenario trace
  and token fingerprint. The smoke is labelled
  `non_inferential_mechanism_smoke_only` and contains no confidence interval,
  winner selection, or formal comparison claim.

#### Artifacts

- `src/mfg_hedge/reliability_aware_routing.py`
- `tests/test_reliability_aware_routing.py`
- `artifacts/reliability-aware-multi-replica-routing-smoke-20260909-r1/`
  (`summary.json`, `protocol.json`, `manifest.json`, and
  `episode_rows.jsonl`)

#### Decisions and risks

- The new engine is isolated and does not alter historical scientific
  behavior, configuration, version, or artifacts.
- The 192-call smoke validates deterministic mechanics, causal observation,
  CRN pairing, completion, and invariants only. It does not establish a best
  router, reliability forecast quality, Pareto result, optimality, Nash/MFG,
  or placement conclusion.
- The chosen smoke load and fault rates are frozen in this ticket. Any
  development/holdout evaluation needs a separate accepted protocol and
  independent namespaces.

#### Next

Create a separate experiment design/protocol ticket only after reviewing this
finite smoke; do not reuse its rows as statistical development or holdout
evidence.

## Answer

The finite reliability-aware multi-Replica routing slice and its authorized
non-inferential smoke are complete. The ticket is resolved. No formal routing
experiment, policy selection, Hedge, price, MFG, or MFC solver was run.
