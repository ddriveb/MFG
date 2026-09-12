# ADR-0012: Bounded transient admission and objective implementation

Status: Accepted (user confirmed the proposed next slice on 2026-09-05)

## Scope

Implement only ADR-0011 proposal sections concerning causal reservation and
outcome scoring, following the physical validation in ticket 09. ADR-0011 as
a whole remains Proposed. No 81-rule search, reduced model, MFG solver,
qualification, holdout, CLI, campaign or historical-schema migration is authorized
by this disposition. Existing attribution tickets 03-07 remain paused.

## Decisions

1. An isolated development API uses fixed four-entry time/class rules (early/late
   D x Regular/Urgent). Only D Primary-A arrivals may request D/I. The API takes
   no callback with access to a workload, engine or future observations. Decisions
   depend only on current arrival metadata and a per-episode reservation ledger.
   A sequential metadata-only prepass is causally equivalent to arrival-time
   decisions for this restricted rule; queue-feedback policies are not supported.
2. Reservation is pooled, FIFO, non-refundable, per half-open D window anchored
   at degraded_start. Defaults: width 25, cap rate scale x 0.45 with scale 0.25,
   mean charge 1. A final short window receives its actual duration's cap.
   Applied actions, never requested actions, go into the unchanged episode engine.
   Debit arithmetic uses decimal representations of the declared parameters to
   avoid floating-point overspend at exact cap boundaries. Unused balance expires.
   This is a reserved-work cap, not a pathwise executed-work cap. No expected-work
   guarantee is inferred for arbitrary user-constructed traces.
3. Score pooled completed episode Tokens by arrival phase/class. Freeze the
   proposed J: quarter-weighted four-phase mean/SLO-excess/Replay losses with
   class weights .8/.2, deadlines 3/2, penalties 1/5; half the sum of D/F
   fractional-tail CVaR95; plus total executed work and non-winner executed work
   per generated Token, coefficients 1/1. Report each term and counts. A missing
   phase/class makes total J null/incomplete, never silently zero or reweighted.
   Raw work is pooled before division, including running losers through drain.
4. Fractional CVaR uses exactly n/20 observations of tail mass; integer divmod
   handles n=20/21 without a rounded-tail or floating-boundary substitution.
   Use a new field/function name; retain historical empirical_cvar95 unchanged.
5. Reuse existing episode metrics' full result invariants before scoring; reject
   duplicate episode keys and mixed protocol/slowdown/delay. New simulation API
   permits only transient-control:v1:fit or transient-control:v1:test namespaces.
   This slice does not generate qualification or holdout data.

## Supersession and limitations

Only for these new transient development APIs, replace ADR-0010's isolated
class-share/endogenous charges with rule 2, and its decision surrogate with
realized pooled J. Historical entry points retain all behavior. Preserve
ADR-0005 lifecycle, CRN and ADR-0009 work/waste interpretation. Enumeration,
statistical gates, economic calibration and full ADR-0011 disposition are deferred.
Pooled FIFO can suppress later Urgent arrivals; expose per-class counts.
The scheduled failure clock is known; no unexpected-failure robustness claim.
