# ADR-0023: T3A occupancy calibration and statistical execution protocol

Status: Accepted (confirmed on 2026-09-07)

Date: 2026-09-07

## Context

The T3A implementation in `token_continuation.py` is mechanically verified,
but its frozen `16` calibration and `16` validation episodes are only an
implementation-validation smoke.  The accepted `bin_schema_v1` has up to 72
target-eligible combinations after fixing the target to D phase, Primary A,
and D arrival-time support:

```text
2 Token classes * 4 phase-age bins * 3 queue-load bins
  * 3 reservation-balance bins = 72
```

Selecting the first eligible D Token concentrates observations near the start
of degradation and does not provide a defensible allocation for conditional
cost estimates.  This ADR defines the occupancy gate and statistical execution
contract.  The user accepted it on 2026-09-07 and subsequently authorized
implementation and execution of both stages.

## Decision

The accepted estimand is explicitly environment- and population-policy
conditional:

```text
Q_m(u | B(O)=b, S_v2; theta_token_load0p7_v1, pi_NIIN_v1)
```

where `theta_token_load0p7_v1` is the complete physical/protocol fingerprint and
`pi_NIIN_v1` is the complete online population-policy fingerprint.  `m` is
one of the three independent ADR-0021 cost models and `u` is one of `N`, `D`,
or `I`.  No exact-observation claim, action selection, BR, regret, Nash, MFG,
or price-feedback claim is added.

### 1. Frozen physical environment `theta_token_load0p7_v1`

Occupancy and validation use exactly the same immutable physical environment;
only their namespace and macro seed differ.  The canonical environment record
contains:

- one logical Expert with exactly two replicas, Replica 0 in domain A and
  Replica 1 in domain B;
- `ExperimentConfig` identity `v1_minimal`, with
  `expert_count=1`, `replicas_per_expert=2`, `failure_domain_count=2`,
  `base_seed=20260901`, `healthy_offered_load=0.7`, and
  `healthy_service_mean=1.0`, giving arrival rate
  `0.7 * (2 / 1.0) = 1.4`.  This environment is deliberately named
  `theta_token_load0p7_v1`;
- the arrival rate `0.9` used by the ADR-0020/T1 historical equivalence
  fixture is a different compatibility environment and is not reused here;
- during D, the surviving physical capacity is `1.5`, so the base offered-load
  ratio is `1.4 / 1.5 ~= 0.9333`; during F, the surviving capacity is `1.0`,
  so the base ratio is `1.4`.  This is a D-near-saturation and F-structurally-
  overloaded environment, not a reproduction of the T1 `0.9` fixture;
- arrivals on `[0,320)`, `arrival_cutoff=320.0`, with independent Token-class
  draws `P(R)=0.8` and `P(U)=0.2`;
- independent lognormal service draws with arithmetic mean `1.0` and
  coefficient of variation `0.5` for every replica and attempt stream;
- deterministic common fault law and half-open phase intervals
  `H=[0,100)`, `D=[100,200)`, `F=[200,220)`, `R=[220,+inf)`; there is no
  stochastic fault sampling in this protocol;
- the existing Dispatcher: before `F`, `primary_replica = token_id % 2`; in
  `F`, the engine records Primary Replica 1.  The Dispatcher never changes
  the logical Expert;
- `degraded_slowdown=2.0`, `hedge_delay=2.0`, complete physical drain of all
  attempts including running losers, Replay and timer outcomes;
- `ReservationParameters(window_width=25.0, budget_rate=0.45, scale=0.25,
  mean_requirement=1.0)`.  The D-relative half-open window cap is
  `0.45 * 0.25 * 25.0 = 2.8125`; admission is non-refundable and stops when
  the observed failure state makes the D action ineligible.  This cap is the
  historical comparator quota; it is not derived from the `1.4` offered-load
  headroom;
- an episode-constant zero external quote, with no endogenous price feedback.

The workload/fault key schema is part of `theta_token_load0p7_v1`: an episode key is
`{namespace}:{macro_seed}:episode:{episode_index}`; its seed is derived from
the fixed config base seed and that complete identity.  The workload uses
separate SHA-256-derived streams labelled `arrival`, `token-class`,
`service:{replica}`, `service:{replica}:attempt1`, and
`service:{replica}:attempt2`.  Action and model IDs are never included in
these exogenous keys.  The deterministic fault timeline is included in the
episode protocol fingerprint rather than sampled from a random stream.

`theta_token_load0p7_v1` is the SHA-256 of canonical JSON containing every item above,
including the full `EpisodeProtocol`, reservation parameters, dispatcher
version, workload stream labels, slowdown, delay, drain contract and quote
basis.  An episode is invalid if its recorded environment fingerprint does
not equal the frozen `theta_token_load0p7_v1` fingerprint.

### 2. Frozen population policy `pi_NIIN_v1`

Both occupancy and validation use the same policy factory and policy
fingerprint.  The only differences are the independent split namespace and
macro seed.  The policy is the named static `NIIN` comparator:

```text
policy_id = token_static_time_class_v1
rule       = (early_regular=N, early_urgent=I,
              late_regular=I, late_urgent=N)
rule_name  = NIIN
late_after = 50.0 phase-time units after D starts
```

It is materialized as a fresh `StaticTimeClassActionSource` by
`policy_factory()` for every episode and every counterfactual branch.  Its
`new_episode()` method must return fresh state.  The policy is deterministic
and has no private random draw; its declared policy seed is `none`.  The
policy-key schema remains the T1 contract
`token:{base_seed}:{token_id}` and is recorded even though this static policy
does not consume randomness.

At each Token arrival it reads only the delivered causal observation's phase,
phase age, Token class and Primary Replica.  It does not read queue state,
Reservation balance, public price, future data, service draws or outcomes.
The online source is still called once for every Token; the requested action
is then passed through the actual ReservationLedger.  Thus the policy's
requested `I` can be projected to applied `N` by the frozen budget, and the
projection is part of the observed physical result.

`pi_NIIN_v1` is the SHA-256 of canonical JSON containing policy ID, exact rule
tuple, `late_after`, `new_episode`/fresh-state contract, randomness mode and
policy-key schema.  Occupancy and validation must reject any policy-factory,
source-manifest or policy-fingerprint mismatch before event execution.

### 3. Frozen cost-model and price provenance

Each model has its own immutable parameter record and price basis.  The
following values are the only values allowed in this protocol:

| model ID | parameters | quote and basis |
| --- | --- | --- |
| `token_initial_price_v0` | `gamma_R=1`, `gamma_U=5`; no fixed work/waste or SLO term | `p_exec=0.0`, `incremental_executed_work` |
| `token_runtime_adr0009_v1` | `gamma_R=1`, `gamma_U=5`, `c_inc=1`, `c_waste=1`; no SLO term | `p_exec=0.0`, `incremental_executed_work` |
| `token_extended_reservation_v1` | `gamma_R=1`, `gamma_U=5`, `alpha_R=1`, `alpha_U=5`, `d_R=3`, `d_U=2`, `c_W=1`, `c_waste=1` | `p_res=0.0`, `admitted_reserved_work` |

The price values are zero but are not omitted: the model record must contain
the numeric quote, basis, model ID, parameter schema and provenance
fingerprint.  Model fingerprints are distinct even if two numeric totals
coincide.  Validation rejects missing, cross-basis, non-finite, or
cross-model parameter records.

### 4. Canonical source bundle

The execution gate uses `token_t3a_source_bundle_v1`, not a partial policy or
scorer hash.  Its roots must include all of the following files:

```text
src/mfg_hedge/hedge_simulation.py
src/mfg_hedge/token_online.py
src/mfg_hedge/transient_control.py
src/mfg_hedge/token_payoff.py
src/mfg_hedge/token_deviations.py
src/mfg_hedge/token_continuation.py
src/mfg_hedge/workload.py
src/mfg_hedge/common_state.py
src/mfg_hedge/attribution_episode.py
<the explicitly named future T3A execution module>
```

The manifest then expands to the complete local-import closure of those roots,
including `config.py`, `domain.py`, `paired.py` and any other local module
loaded by the event engine, scorer, policy factory or execution adapter.  The
canonical source set is the sorted list of every `src/mfg_hedge/**/*.py` file
present in the execution checkout, plus the exact `configs/v1_minimal.json`
bytes.  Each entry records its normalized relative path and SHA-256; the
bundle fingerprint is the SHA-256 of canonical JSON containing the schema ID,
root list and sorted `(path, sha256)` entries.  A future execution module must
exist, be named, and be included before the fingerprint is sealed; an absent,
changed, or unlisted transitive dependency fails closed.  The sealed bundle
fingerprint is carried in every occupancy/validation record and is not a
post-run label.

Every calibration record and every later validation estimate must carry
`theta_fingerprint`, `population_policy_fingerprint`, `bin_schema_fingerprint`,
`target_selection_fingerprint`, namespace, macro seed, protocol ID and the
complete input fingerprint.  Every validation row additionally carries its
`model_id`, model-parameter fingerprint, quote value and quote basis.  These
fields are required provenance, not optional report labels; a mismatch fails
before the event loop and cannot be repaired by relabeling a result.

### 5. Independent occupancy calibration

Before any cost panel is run, generate an independent occupancy library with:

- namespace `token-mfg-restoration:t3a:occupancy:v1`;
- macro seed `20260909`;
- protocol `occupancy_v1`;
- 512 episodes, with `anchor_id = episode_index % 4`;
- no N/D/I branches and no payoff scoring.

The four anchor windows are the existing `bin_schema_v1` phase-age bins:

```text
early-0:    [0,1.5)
early-1:    [1.5,3)
middle:     [3,10)
late:       [10,+inf)
```

The future-blind selector `S_v2` scans arrivals in canonical
`(arrival_time, token_id)` order and selects at most one Token: the first
arrival whose current observation is phase `D`, Primary A, and whose phase age
falls in the episode's assigned anchor window.  The episode index determines
the anchor before generation; the selector cannot inspect later arrivals,
future faults, work, actions, completions, or outcomes.  There is no fallback
to another anchor or later target after the assigned window is exhausted.

Each episode contributes only occupancy metadata: anchor, target status,
`bin_id`, observation/trace fingerprints, and `missing_target` or
`missing_bin`.  Every episode slot is consumed.  Missing or invalid episodes
are not supplemented, and duplicate/overlapping identities fail closed.

### 6. Deterministic occupancy screen and validation allocation

The 512 calibration episodes are used only to compute the occupancy count
`n_occ(b)` for each canonical target bin.  A target bin is retained if and
only if `n_occ(b) >= 16`.  This screen is decided before validation costs are
observed.

For every retained bin and every named cost model separately, the required
floor is 32 complete independent episode panels: the check is performed for
each `(model_id, bin_id)` cell.  A model's failed panel cannot borrow an
observation from another model, and the three N/D/I branches within one model
panel are paired CRN branches, not three independent observations.

Let:

```text
p_min = min_b n_occ(b) / 512
```

over the retained bins.  Before validation episode generation, freeze the
single common validation library size:

```text
E_validation = ceil(2 * 32 / p_min)
```

The occupancy floor implies `E_validation <= 2048`.  This is a pre-registered
engineering allocation rule, not a probability or statistical-power guarantee
that every bin will reach 32 complete panels.  Validation episodes use
the same four deterministic anchor strata, but a separate library and
namespace:

- namespace `token-mfg-restoration:t3a:oracle-validation:v1`;
- macro seed `20260910`;
- protocol `oracle-validation_v1`.

No bin-specific top-up, filtering, merging, or replacement occurs after the
validation library starts.  A retained `(model_id, bin_id)` cell with fewer
than 32 complete panels is `insufficient` with null numeric estimates.  This
makes the allocation a
pre-registered consequence of occupancy only, not a response to observed
costs.

### 7. CRN and accounting

The exogenous episode key is:

```text
token-mfg-restoration:t3a:{split}:v1:{macro_seed}:episode:{episode_index}
```

The anchor is a deterministic projection of the episode index and is not an
additional random-stream key.  Action and model IDs are never added to the
exogenous key.  All N/D/I branches for one episode share its complete
exogenous trace and target prefix; different episodes remain independent.

Because `pi_NIIN_v1` does not read model ID, cost parameters, or price, and all
three quotes are zero, one episode needs exactly one target-selection run plus
four complete physical runs: baseline, requested `N`, requested `D`, and
requested `I`.  Each of those four completed physical results is passed to all
three independent pure scorers.  The model IDs, parameter records, quote
bases, and provenance fingerprints remain separate even though the physical
result is shared.  A failed physical run invalidates all three model scores
for that episode; a scorer-only failure invalidates only its own
`(model_id, bin_id)` cell and never borrows another model's value.  If a later
policy actually reads model or price, that is a different protocol requiring
model-specific reruns and a new ADR.

The frozen worst-case scheduler-call budget is:

```text
occupancy: 512 * 1 = 512
validation: 2048 * (1 + 4) = 10,240
combined: 10,752
```

The validation term counts one target-selection call and four physical
scheduler calls per episode, not four calls per model.  No call is retried.  A
failed branch is retained in audit and fails its whole model panel.

### 8. Disposition and evidence boundary

- `missing_target` consumes the calibration or validation episode slot and is
  not replaced.
- `missing_bin` consumes the slot and is not merged into a neighboring bin.
- Any failed, non-finite, identity-mismatched, or incompletely drained N/D/I
  branch fails the whole model panel; no surviving action contributes a cost.
- A retained `(model_id, bin_id)` cell below the 32-panel floor is
  `insufficient`, not a noisy favorable estimate.
- Calibration output is occupancy metadata only.  If separately authorized,
  validation output may carry only the
  `observable_bin_conditional_cost_oracle` label and its explicit missing
  states.

Neither the occupancy library nor the later oracle validation authorizes
choosing the lowest mean action, expected best response, regret, Nash/MFG
qualification, posted-price pacing, or population feedback.

## Consequences

The old 16/16 smoke remains useful for implementation regression but cannot be
reused as statistical calibration or validation.  The new protocol spends a
small, isolated occupancy budget before committing to the validation library,
and it can fail closed if the target occupancy does not support a bin.  The
allocation is intentionally conservative: it does not guarantee that every
retained bin reaches 32 complete panels because targetless and failed episodes
remain part of the frozen denominator.

## Deferred work

The implementation/execution ticket must verify the sealed occupancy manifest
before generating validation input, then run only the paired observable-bin
cost oracle.  Predictor training,
action selection, posted prices, BR/regret, Nash, conditional mean field and
MFG remain out of scope.
