# ADR-0010: Attribution-first, phase-scoped, per-Replica control protocol

Status: Accepted (confirmed by the user on 2026-09-05)

Date: 2026-09-05

Builds on: ADR-0001, ADR-0004, ADR-0005, ADR-0006, ADR-0007, ADR-0008, ADR-0009

## Context

The runtime-cost correction prevents zero-price work from being literally free,
but it does not isolate a mean-field contribution. The corrected 100,000-Token
pilot contains only 87/110/31 H/D/F arrivals and 99,772 R arrivals; overall
metrics therefore measure almost entirely the indefinitely reused H policy.
No-Hedge is also too weak a sole comparator because any reasonable state-aware
Hedge rule may produce the observed D/F benefit.

The primary controller optimizes mean latency while P99 is interpreted after the
fact, deadline-miss calibration is identically zero, aggregate work hides domain
imbalance, and finite-temperature Softmax deliberately assigns positive mass to
strictly worse H actions. Under the corrected 1/1 cost artifact, H-Regular has
approximately `J_N=1.2646` and `J_D=1.5172`, yet Delayed retains about 22% policy
mass. This is entropy leakage, not a resource-optimal engineering choice.

Per-Replica accounting also invalidates the historical claim that load 0.5 has
clean D headroom. With slowdown 2, Replica A capacity is 0.5 and its round-robin
Primary work rate is 0.5, so its base utilization is already 1.0.

## Decision

1. This ADR governs only the new attribution-study schema/run family. Historical
   schema-3/schema-4 configurations, calibration schemas, code paths, artifacts,
   and pilot interpretations remain governed by ADR-0005 through ADR-0009 and
   are never rewritten.
2. Preserve physical `CommonState={H,D,F}`. Add an arrival-time policy
   `ControlContext=(Phase H/D/F/R, TokenClass, primary_replica)`. R remains a
   Phase whose physical state is H, but it is independently audited and never
   silently obtains an H policy. Every study arm, including diagnostics, forces
   exact Normal in H/F/R; only D may choose N/D/I. ADR-0005 continues to govern
   copies/timers already created before a later boundary.
3. Condition ActionStats on the causally known Primary Replica and add actual
   deadline misses, expected SLO excess, complete action-dependent Primary/
   Hedge/Replay/total/wasted lifecycle work, and work by execution phase and
   Replica. The decision model uses a one-dimensional endogenous destination-B
   load surrogate plus a conservative nominal reservation for A; it does not
   claim closure of a joint two-Replica queue distribution. Economic lifecycle
   work and actual physical phase/Replica work remain distinct audited views.
4. The primary loss is mean latency plus class-weighted expected SLO excess,
   Replay risk, persistent incremental total-work cost, and wasted-work cost.
   Destination-B and common work-budget duals add congestion/resource prices.
   The quota charge table is arm-independent and frozen from the all-Normal
   destination fixed point, even though decision ActionStats vary with endogenous
   load. The new normalized deadlines/weights and 1/1 work costs are frozen in
   the feature spec rather than selected from holdout outcomes.
5. Remove Softmax from confirmatory primary arms. Slack decisions are exact hard
   best responses. A binding mixture is admissible only on final-price best-
   response support with induced destination-load consistency, the nominal A
   reservation and planned B envelope, both non-negative capacity/quota prices,
   complementary slackness for both constraints, stable multi-solution selection,
   and complete witnesses. No convergence failure falls back to Softmax.
   Entropy regularization is permitted only as a named diagnostic with the full
   entropy term and parameter recorded.
6. A result may be called conditional finite-action MFG only when agents take
   the mean field/prices as given and the output satisfies hard best-response
   support plus population/load and dual consistency. A controller obtained by
   directly minimizing a joint population/social objective is MFC; a central LP
   is not called MFG unless it also emits the required price-taking witness.
   MYOPIC fixes exogenous baseline destination load and has no policy-to-load
   closure. Quota is only a causal realization mechanism.
7. Freeze four primary arms—No-Hedge, a state/Primary-aware RULE, a MYOPIC
   controller, and full mean-field+quota—under common ex-ante budgets. Preserve
   separate no-quota and 00/10/01/11 diagnostics. Never match a comparator to
   holdout realized work after the fact.
8. Use load 0.45 for confirmation, with D protection allowed only A -> B; load
   0.5 and above are structural stress under the new per-Replica target. Use
   fixed H/D/F/R episodes ending arrivals at 320, 50 episodes per macro-seed and
   40 independent macro-seeds. The macro-seed is the inferential unit.
9. Use D/F CVaR95 as co-primary efficacy endpoints, Healthy/Recovered
   non-inferiority and execution amplification as safety/resource gates, and a
   fixed claim hierarchy. Existing seeds are development-only; the complete
   holdout registry and all hashes/gates are frozen before one-shot execution.
10. Keep calibration/evaluation separation, CRN, causal quota auditing,
    planned-versus-realized reporting, fail-fast statuses, no grid clamp, and
    immutable transactional artifacts from the earlier ADRs. Quota keeps
    demand-proportional per-class shares; neither class borrows, unused shares
    expire at each window close, and deficits/shares reset each window.

The supersession boundary for this new run family is explicit:

- ADR-0005 is retained in full for copy lifecycle, attempt keys, cancellation,
  failure/replay interaction, and event order.
- ADR-0006 retains calibration/evaluation namespace separation, causal deficit
  accounting and requested/applied/suppressed identities, demand-proportional
  isolated class shares with no borrowing and window-close expiry, and planned-
  versus-realized reporting. This ADR replaces its state/scalar ActionStats,
  Softmax response, and scalar resource model.
- ADR-0007/0008 retain fail-fast/no-clamp behavior, forced-Normal F transparency,
  final-state recomputation, and honest infeasible/nonconverged statuses. This
  ADR replaces their aggregate/joint capacity closure, constant-primary-work
  assumption, H-policy reuse, Softmax-era response, previous official gate, and
  load-0.5 confirmation setting.
- ADR-0009 retains persistent-work and wasted-work definitions. Its
  Hedge-plus-Replay incremental formula is replaced prospectively by the full
  action-dependent lifecycle difference from Normal, which may be negative.
  Schema-3/schema-4 outputs remain historical; the new schema explicitly carries
  the 11/00/10/01 ablations.

No old schema, artifact, or historical solver output is altered.

## Consequences

- A positive result can identify whether value came from Hedge, a fixed rule,
  allocation, mean-field closure, quota, or persistent cost terms.
- The 100,000-Token single-fault artifact stays a pilot rather than becoming
  pseudo-replicated failure evidence.
- H/R speculative latency hedging is deliberately excluded from the primary
  fault-protection claim. It may be studied later as a separate mechanism.
- F/R latency changes are interpreted as D-protection spillover and recovery,
  because F/R themselves choose Normal.
- Load 0.45 places degraded Replica A at its nominal planned boundary; it is not
  proof of realized feasibility. B receives the only permitted D Hedge direction.
  Action-dependent cancellation, cross-phase carry-in, and F/R spillover are
  reported explicitly rather than hidden inside the D planning envelope.
- The first attribution study still uses a finite load-summary approximation;
  policy-composition-consistent background calibration and detector uncertainty
  remain separate future work.
