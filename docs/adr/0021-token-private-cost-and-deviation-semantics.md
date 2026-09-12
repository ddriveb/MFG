# ADR-0021: Token private costs and finite pathwise deviation semantics

Status: Accepted (confirmed on 2026-09-06)

Date: 2026-09-06

## Context and scope

Accepted ADR-0020 covers Token identity and T1 causal online execution only.
T1 and its strict Replica-ID correction are resolved. The user requested a
separate decision for private costs and finite deviations BEFORE implementing
`token_payoff.py` and `token_deviations.py`.

This ADR defines T2: immutable cost records and complete finite counterfactual
reruns for one tagged Token. It does not authorize continuation prediction,
expected-cost estimation, best response, regret, endogenous price, HJB/FPK,
MFG/Nash statements or a formal campaign. A completed design ticket does not
make this ADR Accepted.

## 1. Retain three independently named cost models

All three models are supported, with a REQUIRED model ID and frozen parameters.
Do not choose a model implicitly from whichever fields are present.

### `token_initial_price_v0`

The initial Token-MFG model, explicitly restated by the user on 2026-09-06, is:

    C_initial(i) = L_i + gamma_k R_i + p_exec (W_i^H + W_i^R).

It deliberately has NO fixed redundant-work penalty and NO fixed waste penalty.
Raw work/waste is still recorded for physical audit, but no c_inc/c_waste/c_W
or SLO-excess coefficient belongs to this model's parameter schema.

### `token_runtime_adr0009_v1`

ADR-0009 later added persistent work/waste costs. Its runtime formula, also
visible in `mfg_solver._soft_best_response`, becomes the distinct model:

    C_runtime(i) = L_i + gamma_k R_i
                   + (c_inc + p_exec) (W_i^H + W_i^R)
                   + c_waste W_i^lose.

This model is not called "Original". It does not inherit the historical
scalar-rho approximation, class-share quota, Softmax or solver certificate.
At c_inc=c_waste=0 its numerical cost equals C_initial for the same inputs,
but its model ID remains token_runtime_adr0009_v1. Never infer, alias, relabel,
deduplicate or merge the two provenance identities from parameter values.

### `token_extended_reservation_v1`

The proposed Token-restoration extension is:

    C_extended(i) = L_i + alpha_k (L_i-d_k)+ + gamma_k R_i
                    + c_W W_i^all + c_waste W_i^lose
                    + p_res Q_i^reserved.

It includes all executed Primary work, SLO-overrun magnitude, and prices
admitted Reservation rather than executed incremental work.

Initial and runtime price incremental executed work; extended prices admitted
reservation. The extended model is not obtained merely by setting alpha=0.
Do not pool costs or cost-difference rows across model IDs as if they described
one utility. An experiment isolating only SLO or Primary-work effects should
hold the price basis fixed in a separately named ablation; T2 does not run one.
The previous proposed ID token_original_runtime_v1 is withdrawn, not an alias
for any of these three IDs; reject it in the future implementation.

### Parameter and units contract

| Component | Initial price v0 | Runtime ADR-0009 v1 | Extended reservation v1 | Unit / meaning |
| --- | --- | --- | --- | --- |
| Latency | L | L | L | time, coefficient1 |
| SLO excess | not charged | not charged | alpha_k max(L-d_k,0) | magnitude in time, not miss indicator |
| Replay | gamma_k R | gamma_k R | gamma_k R | event loss; R is0/1 |
| Persistent execution | not charged | c_inc (W_H+W_R) | c_W (W_P+W_H+W_R) | cost / executed work |
| Waste | not charged | c_waste W_lose | c_waste W_lose | extra cost / nonwinner executed work |
| Exogenous price | p_exec (W_H+W_R) | p_exec (W_H+W_R) | p_res Q_reserved | cost / executed work or cost / reserved work |

Proposed named starting parameter presets are gamma_R=1, gamma_U=5,
c_inc=c_W=c_waste=1 where applicable, and extended d_R=3,d_U=2,
alpha_R=1,alpha_U=5. Presets must expand into explicit recorded values.
All coefficients and prices are finite, nonnegative real numbers (not bool);
deadlines are finite positive reals. Zero coefficients are valid explicit
ablations. Unknown model IDs, missing class values, conflicting/inapplicable
coefficient sets or mismatched price bases fail validation.

Class frequencies .8/.2 and phase weights do NOT multiply one Token's entire
private payoff. Pooled CVaR95, system welfare, SLO violation rate and resource
fairness remain separate evaluation metrics. No model silently inserts a
population CVaR into an individual player's cost.

## 2. Raw quantities and physical settlement

For Token i with arrival a_i and winner completion t_i^win:

    L_i = t_i^win-a_i;
    R_i = 1{replay_count_i=1};
    W_P = sum executed work of attempt0;
    W_R = sum executed work of attempt1;
    W_H = sum executed work of attempt2;
    W_all = W_P+W_R+W_H;
    W_lose = sum executed work on attempts != winner_attempt_id;
    Q_reserved = this Token's actual admission.charge;
    t_i^settle = max terminal_time over this Token's attempts.

Use the engine's Replay-created count, not `W_R>0`: a created but unexecuted
Replay is still a Replay event. T2 enforces the current at-most-one contract.
In particular Normal can incur Replay and positive execution/waste costs.

Required work is not executed work. Cancelled/invalidated queued copies cost
zero executed work; a failed running copy costs its already executed work.
No monetary charge is applied to never-executed residual work. Waste deliberately
overlaps execution: a losing Hedge consumes work AND produces no result.
The runtime model charges a losing Primary's waste, though Primary execution
is excluded from its persistent incremental-work and price components. The
initial model records that waste but does not charge it; it also excludes
Primary execution from its execution-price term. All models still wait for
complete physical settlement so raw accounting and future H/R work are final.

The first winner ends latency but does NOT settle cost. A running loser can
continue or later fail. T2 runs the whole episode through complete physical
drain, then scores. Never stop at winner time, arrival cutoff or phase boundary.
Do not substitute the episode drain time for Token latency or settlement time.

The scorer consumes completed online-run records, cross-checking token IDs,
class, admission/applied action, one winner, replay count, terminal statuses,
unique attempt identities, physical component sums, finite nonnegative work,
per-Token versus global attempt records and drain_end>=settlement_time.
Normal admissions have `reservation_admitted=True` in T1: that boolean alone
does NOT imply a paid Hedge. Use charge and applied action.

An unfinished snapshot, missing winner/terminal record, contradictory admission
or incomplete physical accounting has no numeric total. Reject malformed
inputs; if reporting an interrupted run, use status=incomplete and null costs.
Do not report a zero/partial cost or claim arbitrary fabricated final dataclasses
are proven complete merely because they have the correct Python type.

## 3. Admission and executor outcomes determine billing

| Case | Work and Replay | Initial/runtime execution-price term | Extended reservation-price term |
| --- | --- | --- | --- |
| N | actual N lifecycle | p_exec W_R if Replay occurs | 0 |
| D/I refused by Reservation | actual applied N lifecycle, including Replay | p_exec W_R if Replay occurs; no fictitious Hedge work | 0 |
| D admitted, winner before timer | include actual Primary work; no created Hedge work | no Hedge execution price | p_res times admitted charge; no refund |
| D admitted, failure before timer | actual failure/Replay lifecycle | actual incremental executed work | retain admitted charge |
| D/I admitted, running loser remains | include all later loser work/waste | include any executed H/R work | original admission charge only |
| Queued loser cancelled | zero extra executed work for that queued copy | 0 for that copy | no refund of prior admission |

Admission denial does not exempt a Token from latency or Replay costs. It means
score actual N outcomes, not hypothetical admitted D/I outcomes. In a paired
refused-D versus N run the costs agree when physical outcomes agree; do not
assert arbitrary stateful opponent policies must always generate identical
whole-system traces when different request labels are publicly observable.

## 4. Exogenous price in T2

T2 supports an immutable, episode-constant external quote p>=0, defaulting to
an explicitly recorded0. It is fixed before baseline/candidate execution,
identical across every branch, independent of selected action, workload draws,
queue evolution and resulting performance. Record both value and model-specific
basis (`incremental_executed_work` or `admitted_reserved_work`).

The initial/runtime execution quote is locked at Token arrival and applies to
later actual H/R execution; it does not change at a phase boundary. The extended
quote multiplies actual admitted reserved work, even when the timer later voids.
Constant price makes these timing rules unambiguous; schedules/controllers and
phase-dependent settlement prices require a later design.

Current T1 creates `TokenObservation.public_price=0.0`. For a nonzero T2 quote,
a narrow runner-owned action-source adapter must deliver an immutable copy of
that observation with ONLY public_price replaced before invoking the policy.
Use the same adapter in baseline and every candidate; log the delivered price
and observation fingerprint. No future trace or cost outcome reaches choose().
The policy configuration knows the fixed model/price basis. Price must never
be changed only in the scorer while presenting a different value to the policy.
Zero-price T1 behavior remains unchanged; this adapter needs no new physical
price-feedback loop or modification of event/Reservation semantics.

Scoring the same completed physical record under a different price/model is
allowed as an explicitly labeled retrospective payoff calculation. It does
not establish that the record's policy was executed under that different
incentive. Such scoring must not be substituted for a price-conditioned rerun.

## 5. Exactly one requested-action intervention

Stable target identity is (episode identity/fingerprint, token_id), never an
Expert ID, arrival rank selected AFTER seeing outcomes, or the winning copy.
Choose the target and candidate bank before examining counterfactual outcomes.
T2 permits default N/D/I or an explicit nonempty unique subset of those enums.
Ineligible or unaffordable candidates are retained as projected-to-N rows.

Baseline: execute the supplied policy functions causally for every Token.
For a candidate u, restart the entire episode from the same initial state,
but override only this Token's REQUESTED action at its actual arrival:

    recommendation = base_source.choose(priced_observation, same_policy_key)
    requested = u if this_is_target else recommendation
    applied = same_Reservation.admit(requested)
    execute the complete original physical engine through drain.

Call and validate the base policy ONCE even at the target, before replacing
its returned recommendation. This preserves call count/RNG/private-state
updates; skipping choose() would inadvertently perturb a stateful policy at
all later arrivals. The intervention changes the return sent to admission;
it does not retroactively rewrite arbitrary private policy memory. Future
observations contain actual engine outcomes under the intervention.

Other Tokens keep their policy functions, parameterization, reset contract and
private randomness RULES. They do NOT keep cached observations, admissions,
actions, completion times or queue paths. Different observable conditions may
produce different recommendations/actions; this is required finite interaction.

The wrapper has no override for timers, Replay, winner/cancellation, application
of budget, other Tokens, service speeds or Gate. Candidate label is not included
in an exogenous or private-policy random key. A private sampler can map the
same per-Token uniform to a different action when probabilities change.

## 6. Fresh state, CRN and full reruns

Require a caller-declared policy factory/configuration that produces fresh
run-local state per branch. The T2 wrapper's new_episode() materializes one
fresh base source. T1's permissive reuse when new_episode is absent is not a
safe assumption for deviations: T2 must require a factory/reset contract and
reject factories returning an already-used mutable source or invalid choose().
Never share a ledger, mutable policy, deficit counter or event engine across
branches. Do not reset a policy halfway through an episode.

Every branch uses identical arrival times/order, class, Dispatcher metadata,
Primary/Hedge/Replay work draws, fault timeline, delay, slowdown, Reservation
parameters and exogenous quote. Reuse immutable EpisodeTrace and check
episode_trace_fingerprint before/after each run. No regeneration, candidate
dependent seeding, ad hoc service resampling or reuse of cached opponent paths.

Keep the T1 policy key rule (currently `token:{base_seed}:{token_id}`) stable
across branches and include its version plus policy seed/config in provenance.
Its cross-episode independence is not asserted here; independent continuation
namespaces/keys will be designed later. Pairing requires same randomness RULE,
not forcing the same number of arbitrary stateful RNG draws after observations
have legitimately diverged.

Physical prefix through immediately before the target decision must agree;
the target delivered observation, baseline recommendation and policy key must
agree across branches. After intervention, divergence is expected. Re-requesting
the baseline's target action on a full fresh rerun must reproduce the baseline.
The baseline still calls all policies, including before the target.

For one episode and c candidate actions: exactly1+c complete calls, c<=3.
Retain a fresh same-action candidate as an identity check; no baseline/candidate
deduplication. Scoring all three named costs on an already-run record adds no physical
call, but comparisons under different policy incentives need their own runs.
Count calls before invocation. No retries, adaptive candidate expansion,
parallel worker pool or experiment CLI in T2. If a run fails, retain attempted
call count and explicit failure status; no cost for an incomplete branch and
no success or equilibrium label for the overall comparison.

## 7. Output is pathwise evidence, not expected regret

For an exogenous realization omega define:

    delta_cost_i(u;omega) = C_i^u(omega)-C_i^base(omega);
    pathwise_gain_i(u;omega) = C_i^base(omega)-C_i^u(omega).

Negative delta / positive gain means this realized intervention helped the
target. Keep signed values; a harmful intervention is not clipped to zero.
Rows are ordered canonically N,D,I, not by realized total cost.

Evidence label: `finite_token_pathwise_deviation`.
Required booleans: claims_best_response=False, claims_regret=False,
claims_nash=False, claims_mfg=False. Do not return a `best_action`, `best_response`,
`regret`, `epsilon_nash` or `equilibrium` field, even if its value is null.
All candidate costs/differences remain available for inspection.

T2 must NOT calculate E[min_u C_u(omega)] or an average per-path improvement
against the hindsight winner as a BR certificate. It also does not aggregate
multiple episodes into an expected-regret field. A finite price or a favorable
pathwise gain does not create a best response.

Future T3+ will fix information O, obtain independent continuation samples
from its conditional law, estimate Q(u|O)=E[C_u|O], then assess a policy against
min_u Q(u|O) with selection/inference safeguards. Reusing fixed CRN across
actions within a continuation sample is compatible with independence between
samples. Those state-conditioning, sampling and statistical choices are deferred.

## 8. Proposed module/API and records

Names below are contracts to implement only after acceptance; they do not
describe callable code in the present repository.

`token_payoff.py`:

- Validated frozen initial/runtime/extended parameter types and an external quote.
  Model identity participates in parameter/provenance fingerprints even when
  initial and zero-coefficient runtime records have identical numeric costs.
- `score_token_payoff(completed_run, token_id, parameters, quote, ...)`:
  no simulation, no policy call, no cost-model inference; separate raw and
  weighted components and distinguish retrospective scoring metadata.
- `TokenPayoff`: model/schema IDs, parameters and price basis/value; identity;
  requested/applied/charge/timer/suppression; latency, excess, Replay count;
  W_P/W_H/W_R/W_all/W_lose; winner and settlement times; component costs,
  total, completed status and provenance. Raw SLO excess may be null/not-applicable
  in initial/runtime models without a declared diagnostic deadline; its cost is0.

`token_deviations.py`:

- `evaluate_token_pathwise_deviations(episode, policy_factory, target_token_id,
  candidates, parameters, quote, reservation, slowdown, delay, ...)`.
- Runner-owned quoted-observation/single-request override source, fresh per run.
- `TokenPathwiseDeviationRow`: candidate requested action; baseline/candidate
  actual admission; full target payoff; signed component/total differences;
  target prefix observation/key fingerprints; input fingerprint; optional
  changed-other-action and physical externality diagnostics, never a second
  deviation or population optimization score.
- Batch record: baseline plus canonical candidate rows, completed/failed status,
  attempted calls, fixed scope (`one_episode`, `one_token`, `requested_action`),
  policy configuration/hash, cost/quote/config/source fingerprints and the
  negative claim flags above. No formal artifact schema/CLI/campaign is added.

Validation rejects bool/float IDs, absent target, unknown/duplicate actions,
nonfinite/negative coefficients/prices, invalid deadlines, incompatible model
and quote basis, changed traces, leaking/shared source state, missing delay
needed for a valid D comparison, incomplete cost input and identity mismatch.
An impossible action due to legitimate admission is a result, not an input error.

## 9. Minimum T2 regression matrix (red before implementation)

| Fixture | Required result |
| --- | --- |
| Hand-computable N without Replay | Initial/runtime exclude Primary execution; extended includes it |
| Immediate wins, Primary running loser | Latency ends at winner; work/waste and settlement include later Primary |
| Losing Hedge keeps running | Include all post-winner Hedge work and waste |
| Failed Primary then Replay | Replay flag1, executed failed work retained, residual not billed |
| Never-started queued cancellation | Zero executed cost for that attempt; no refund |
| Admitted Delayed timer voids | Reservation retained; no invented Hedge work |
| Refused D and I | Applied N; zero reserved-price fee; actual N Replay costs retained |
| Nonzero quote | Policy and scorer see the same constant quote/basis in all branches |
| Three named models | Exact components/formulas; initial has no fixed work/waste penalties; no crossed price basis; zero-coefficient runtime remains a separate model ID/fingerprint; reject withdrawn ID |
| Target identity/no-op intervention | Same-action fresh rerun exactly reproduces baseline |
| Other online Token reacts | Same policy changes action under changed queue/budget, not frozen output |
| Stateful source target call | Base choose called once at target and per later Token; no shifted call stream |
| Independent run factories | Candidate order does not leak state; reused mutable source rejected |
| CRN across N/D/I | All input streams/fault/config fingerprints match; attempted counts1+c |
| Delayed/fault/event tie | Preserve T1 event order and automatic Replay; no second policy decision |
| Mid-batch failure/incomplete snapshot | Attempted calls retained, incomplete costs absent; no false success |
| Future-dependent winning action | Raw pathwise rows only; no hindsight-min/BR/regret fields |
| Bad inputs | Strict enum/type/numeric/identity/lifecycle validation fails closed |

Suggested arithmetic anchor with all applicable persistent coefficients1:
L=1, no Replay/excess, W_P=2,W_H=1,W_R=0, Hedge winner, W_lose=2,
Q_reserved=1, each model's quote=.5 in its OWN units gives initial1.5,
runtime4.5 and extended6.5. With runtime c_inc=c_waste=0, runtime also gives1.5
but retains its distinct runtime model ID/provenance. This checks formulas,
not relative desirability of different games.

For this design review revision, run and append the actual results of:

    .venv\Scripts\python.exe -m unittest discover -s tests -v
    .venv\Scripts\python.exe -m mfg_hedge check --config configs\v1_minimal.json

The expected current baseline is517 tests; observe and record the real count
and outcomes. File hashes and hand arithmetic do not replace these checks.
After implementation run focused payoff/deviation/T1/legacy affected tests,
the required full suite and config check. Existing historical test counts are
not evidence that unimplemented T2 tests passed.

## 10. Disposition and supersession

This ADR is Accepted for the bounded T2 private-cost and finite pathwise
deviation contract. Both production modules remain unimplemented until the
separate bounded implementation ticket linked to design ticket04 is executed.
Acceptance authorizes only that ticket's scope; it does not authorize T3
continuation/regret machinery or any predictor, price-feedback, best-response,
forward, scaling, Nash, MFG, or formal campaign work.

This ADR would supersede only spec.md's provisional single private-cost choice
and vague finite-deviation wording for T2. ADR-0020 T1 physics, ADR-0009
historical cost behavior, all existing solvers/artifacts and the Expert branch
remain unchanged. External fixed pricing here does not accept spec.md's
proposed endogenous pacing controller or any price-equilibrium interpretation.

## References

- [Accepted Token/T1 boundary](0020-restore-token-player-model.md)
- [Historical runtime cost](0009-persistent-runtime-work-cost.md)
- Initial price-only formula and separate provenance: user's explicit review
  correction of 2026-09-06, recorded in design ticket04's revision log.
- [T2 design ticket](../../.scratch/token-mfg-restoration/issues/04-design-token-payoff-and-deviation.md)
- [T2 implementation ticket](../../.scratch/token-mfg-restoration/issues/05-implement-token-payoff-and-deviations.md)
- [Restoration specification](../../.scratch/token-mfg-restoration/spec.md)
- Source anchors: `src/mfg_hedge/mfg_solver.py:_soft_best_response`,
  `token_online.py:TokenActionSource/OnlineAdmissionDecision/OnlineEpisodeResult`,
  `hedge_simulation.py:HedgeTokenResult/HedgeAttemptRecord/_online_observation`,
  `attribution_episode.py:episode_trace_fingerprint`.
