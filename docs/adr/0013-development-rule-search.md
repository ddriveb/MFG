# ADR-0013: Development-only 81-rule reference search

Status: Accepted (user directed the experiment to start on 2026-09-05)

## Scope

Activate ADR-0011's finite reference search only. This is a development fit on
exact finite queue simulations. It does not activate qualification, holdout,
reduced mean-field modeling, MFG claims, or the paused historical tickets 03-07.

## Frozen experiment

1. Generate 64 independently keyed episodes using namespace
   `transient-control:v1:fit`, macro-seed 20260905, indices 0..63, load 0.45,
   the 100/200/220/320 protocol, slowdown 2, and existing CRN streams.
2. Enumerate all 81 rules in lexical `N < D < I` order over early-Regular,
   early-Urgent, late-Regular, late-Urgent. Every rule uses ADR-0012's default
   reservation and `tau0` from the configured healthy service distribution.
   No-Hedge `NNNN` is the immutable comparator and an eligible candidate.
3. For each rule, run all 64 episodes through the same physical engine and
   score ADR-0012's pooled realized objective. No simulation is reused across
   different applied actions; only immutable exogenous traces are shared.
4. A candidate is development-feasible iff its H/R mean-latency ratios versus
   NNNN are each <=1.01, H/R P95 ratios each <=1.02, and total executed-work
   ratio <=1.05. P95 uses the existing linear percentile definition. Missing
   cohorts, failed invariants, nonfinite values, or trace mismatch are invalid.
5. Select minimum scalar J among feasible rules, then minimum total executed
   work, then lexical rule order, using exact computed values with no hidden
   tolerance. Report all 81 rows, all constraints, requested/applied/suppressed
   counts, selected-vs-NNNN D/F tail ratios, and trace fingerprints.
6. Write a new immutable transactionally committed development artifact. A
   complete negative result is retained. Runtime is diagnostic only; timestamps
   and run IDs are not scientific inputs.

## Claim boundary

The selected rule is the sample-best feasible member of the predeclared
81-rule class on burned development data. It is not a global optimum, safety
certificate, statistical comparison, qualification result, or MFG evidence.
Its fit data may never become untouched confirmation data. Coefficients,
budget, rules, sample count and gates are not changed in response to this run.
