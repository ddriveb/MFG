# Implement population state, buckets, and share calibration

Type: implementation
Status: resolved
Blocked by: none

## Goal

Build the causal conditional population-state representation `m_t`, the
observable Replica-state bucket schema, and the forward estimator for
predicted routing trend `x_t` versus post-commit realized shares.

## Scope

- Freeze bucket boundaries, common-noise conditioning, forward/continuation
  namespaces, macro seeds, episode counts, fingerprints, and call budgets.
- Keep predicted shares, population occupancy, and realized shares separate.
- Validate calibration with independent samples and episode-level accounting.
- Reject unsupported/empty cells, insufficient samples, overlap, or fingerprint
  mismatches fail-closed; do not merge or supplement cells.
- Do not implement best response, prices, or a solver.

## Frozen implementation-validation protocol

This ticket freezes the protocol needed by the population-state implementation
before any trace is consumed.  It is an implementation-validation protocol,
not a statistical experiment or MFG qualification protocol.

- Protocol ID: `reliability-aware-token-mfg:population-v1`.
- Replica topology: one Logical Expert, `K=8` Replicas, four fixed domains;
  the physical parameters and failure/work CRN remain those of the accepted
  ADR-0035 finite routing engine.  No placement, Hedge, or price change is
  permitted.
- Immutable `bin_schema_v1` fields and half-open intervals are:
  - Token class: `Regular`, `Urgent`;
  - Token age: `[0,1)`, `[1,2)`, `[2,4)`, `[4,+inf)`;
  - retry count: `[0,1)`, `[1,2)`, `[2,+inf)`;
  - Replica queue depth: `[0,1)`, `[1,2)`, `[2,4)`, `[4,8)`, `[8,+inf)`;
  - Replica estimated work: `[0,1)`, `[1,2)`, `[2,4)`, `[4,+inf)`;
  - running service age: `idle`, `[0,1)`, `[1,2)`, `[2,4)`, `[4,+inf)`;
  - individual and domain hazard: `[0,0.01)`, `[0.01,0.05)`,
    `[0.05,0.10)`, `[0.10,+inf)`;
  - health: `UP`/`DOWN`; domain ID `0..3`; locality is
    `same_domain`/`other_domain` relative to the Token ingress domain.
  All numeric intervals are left-closed/right-open except the final
  `(+inf)` interval, which is left-closed and unbounded.  NaN, infinity,
  negative values, unknown labels, and boundary ambiguity fail closed.
- Common-noise conditioning is a SHA-256 fingerprint of only health/failure
  events with event time `<= decision_time`; future events are excluded.
  Public queue, placement, current batch, and published trend remain separate
  public-observation fields and are never folded into `F_t^0`.
- Forward namespace: `reliability-aware-token-mfg:v1:forward`, macro seed
  `20260910`, 16 implementation-validation episodes.  Independent share
  calibration namespace: `reliability-aware-token-mfg:v1:share-calibration`,
  macro seed `20260911`, 32 episodes.  Episode identities are canonical
  `(namespace, macro_seed, episode_index)` tuples and the two namespaces are
  disjoint by construction.
- The implementation-validation call ceiling is `16 + 32 = 48` complete
  finite physical runs.  No formal development/holdout sample or campaign is
  authorized by this ticket.  Any future statistical protocol must freeze its
  own counts and budget in a later ticket.
- A calibration cell requires at least 2 complete episode-level observations
  in this hand-validation slice.  Missing/unsupported cells, duplicate or
  overlapping episode identities, trace/protocol fingerprint mismatches, and
  any failed physical run are fail-closed without zero-fill, merge, retry, or
  supplemental sampling.
- Every forward run creates a fresh endogenous engine state.  All action
  shares are canonical 8-Replica tuples; policy-predicted `x_t` and after-
  commit `realized_share_t` are separate records.  Same-episode CRN is the
  immutable input trace only; no action, queue, completion, or future state is
  cached in a trace/library object.

## Acceptance criteria

- State and trend schemas are immutable and causally observable.
- Common fault histories condition state without exposing future events.
- Predicted and realized share residuals are computed under a frozen contract.
- Source/protocol fingerprints and independent library identities are recorded.
- Historical routing behavior remains unchanged.

## Answer

Implemented the isolated finite population-state and routing-share calibration
layer.  The result is an implementation-validation component only; it does
not estimate a best response, set a price, solve a population fixed point, or
run an experiment.

## Next dependency

Ticket 04 may start only after population state and share calibration are
resolved.

## Progress log

### Update: 2026-09-10 — population state and share calibration

Status: completed

#### Goal

Implement ticket 03 as a causal, immutable finite population-state layer over
the accepted simultaneous routing engine.  Keep `mu_t`, `nu_t`, `eta_t`,
predicted `x_t`, and post-commit realized shares separate, and stop before
best response, prices, or MFG solving.

#### Changed

- Added `src/mfg_hedge/population_estimator.py` with immutable `MuState`,
  `NuState`, `EtaState`, `XState`, batch records, episode identities, cell
  estimates, and calibration results.
- Froze `bin_schema_v1` with strict half-open boundaries for Token age/retry,
  Replica queue/work/service age/hazard, health, domain, and locality.
- Added causal common-noise-prefix fingerprints containing only health events
  observed through the decision time; future faults are excluded.
- Added `run_population_forward` and
  `run_population_forward_library`; each episode creates a fresh policy and a
  fresh endogenous routing engine state.  The online policy must provide
  `predict_share`, which is captured before the first action of each cohort.
- Added episode-level `calibrate_population_shares` with canonical share
  validation, occupancy mass, predicted/realized L1 residuals, common-noise
  provenance, duplicate/overlap detection, required-cell checks, and
  fail-closed insufficient-cell handling.
- Corrected the simultaneous engine's active population accounting to use
  only arrived-but-not-completed Tokens, including canonical active class
  counts; future arrivals are not exposed in the population context.
- Added 11 focused tests.  Historical engines and package exports were not
  changed; version remains `0.28.0`.

#### Verification

- Real Red: before implementation,
  `tests.test_population_estimator` failed with
  `ModuleNotFoundError: No module named 'mfg_hedge.population_estimator'`.
- Focused suite:
  `.venv\Scripts\python.exe -m unittest tests.test_population_estimator -v`,
  **11/11 passed**.
- Affected suites: population estimator **11/11**, simultaneous routing
  **7/7**, game workload **15/15**, historical reliability-aware routing
  **12/12**; total **45/45 passed**.
- Full suite:
  `.venv\Scripts\python.exe -m unittest discover -s tests -v`,
  **750/750 passed** in 345.236 seconds.
- Config check:
  `.venv\Scripts\python.exe -m mfg_hedge check --config
  configs/v1_minimal.json`, exit 0 with `status: ok`.
- Verified immutability, half-open boundaries, future-fault prefix isolation,
  fresh policy state per episode, same-batch predicted/realized-share
  separation, deterministic reruns, duplicate identity/trace rejection, and
  required/insufficient cell fail-closed behavior.
- No formal forward/calibration sample library was consumed, no experiment
  calls or artifact were generated, and no BR/regret/Nash/MFG result exists.

#### Artifacts

- [population_estimator.py](D:/project/mfg_hedge_v1/src/mfg_hedge/population_estimator.py)
- [test_population_estimator.py](D:/project/mfg_hedge_v1/tests/test_population_estimator.py)
- No generated experiment or research artifact.

#### Decisions and risks

- The frozen 48-call ceiling, namespaces, seeds, and cell floor are recorded
  as implementation-validation parameters in this ticket; they are not a
  formal statistical development/holdout protocol.
- `predict_share` is a policy-provided pre-commit prediction.  Calibration
  compares it with post-commit realized shares but does not fit, select, or
  update a policy.
- Occupancy mass is the fraction of all decision-cohort Token positions in
  the episode library belonging to a cell; calibration estimates remain
  episode-independent rather than treating individual Tokens as independent
  episodes.
- Existing finite routing behavior remains unchanged except for correcting
  the new simultaneous context's active-token count so it cannot include
  future arrivals.  No historical result or artifact was rewritten.

#### Next

Ticket 04 is the next dependency and must separately freeze and implement
tagged continuation and finite-`N` deviations.  Do not start a formal MFG
experiment from this ticket alone.
