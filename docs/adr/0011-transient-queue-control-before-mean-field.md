# ADR-0011: Transient queue control before mean-field claims

Status: Proposed

Date: 2026-09-05

## Context

ADR-0010's scalar destination-load surrogate pools lifecycle work of D-arrival
Tokens even when that work is admitted or executed in F/R. It also closes the
requested policy although quota changes the applied policy, and holds A's queue
environment fixed despite action-dependent cancellations. Small numerical
residuals cannot certify the physical accuracy of these approximations.

The user requested a reasonable mathematical model before further changes.
This proposal specifies the finite stochastic queueing system first and uses a
small, enumerable control class as a reference. It does not declare a newly
designed algorithm successful before development validation.

## Proposed decisions

1. Use the exact joint two-Replica queue process, with ordered copy identities,
   service progress, timers, winner links, failure state, and admission balance.
   Lifecycle costs and time-local workload dynamics have separate equations.
2. Predict and evaluate the composite policy consisting of requests AND the
   admission rule. Any approximate forward map must reproduce this composite
   process on both A and B, including Replay bursts and cancellation.
3. Use mean/SLO/Replay losses and explicit D/F CVaR in the system objective,
   together with persistent total executed-work and non-winner-work costs.
   Coefficients are normalized mechanism preferences, not measured money prices.
4. Propose an arm-independent prospective Hedge reservation equal to one mean
   service requirement for each admitted D/I request. Use a pooled, causal,
   non-refundable per-window ledger; its guarantee is reserved charge and an
   expected-work bound under declared draw independence, not a pathwise work cap.
   Class preference lives in the policy/loss. This prospectively replaces the
   isolated expected-work class shares and the endogenous charge table.
5. Start with 81 deterministic time-by-class rules, all using the same ledger.
   Enumerate them using development-only full-episode simulations. This is a
   restricted finite-system stochastic-control reference, not MFG or a proved
   global optimum over all policies. Select a policy before independent validation.
6. Retain MFG as a subsequent research hypothesis. A transient mean-field
   approximation needs a specified population scaling, joint/environment law,
   causal best responses including admission, and a validated finite-system
   approximation. A policy-generated trajectory law alone is not an MFG theorem.
7. Keep the existing copy lifecycle, CRN, workload distribution, episode clock,
   and historical artifacts. Prospective labels/statistical gates must be
   revised before any freeze; old C0-C3 gates cannot grade a control benchmark as MFG.

## Exact supersession boundary if accepted

- Retain ADR-0004/0005 event ordering and copy lifecycle.
- Retain ADR-0009 persistent execution/waste costs and old schema compatibility.
- Replace ADR-0010 sections governing the 36-cell scalar calibration, fixed A
  decision environment, requested-policy fixed point, two duals, and isolated
  class shares for the new proposed run family.
- Retain ADR-0010 statistical independence, disjoint namespaces, complete-result
  reporting, and no post-hoc holdout tuning. Revise arm identities and claim
  gates in a follow-up design disposition before implementation.
- The new objective uses fractional-tail CVaR. Existing rounded-tail empirical
  CVaR outputs remain immutable and are not silently redefined.

## Consequences

The next step is a physical-model contract and development benchmark, not an
implementation of the old ticket-03 fixed-point solver. Additional mean-field
complexity must earn its place through predictive and decision-quality evidence.
The proposal changes the near-term solver approach and admission semantics;
these are explicit proposed research choices, not already accepted code behavior.

## Design and sources

Full definitions and acceptance fixtures:
`.scratch/mfg-hedge-attribution-study/transient-model-design.md`.

- Whitt and You, *Using Robust Queueing to Expose the Impact of Dependence in
  Single-Server Queues*: https://www.columbia.edu/~ww2040/robust_WY_101016.pdf
- Rockafellar and Uryasev, *Optimization of Conditional Value-at-Risk*:
  https://sites.math.washington.edu/~rtr/papers/rtr179-CVaR1.pdf
- Carmona, Delarue and Lachapelle, *Control of McKean-Vlasov Dynamics versus
  Mean Field Games*: https://arxiv.org/abs/1210.5771
