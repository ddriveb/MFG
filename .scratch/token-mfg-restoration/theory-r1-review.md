# Theory derivation r1: checks and limits

Date: 2026-09-07

## Reviewed identities and source correspondence

- Player is the logical Token; one arrival request, no later action minimum.
- Initial price-only utility has no Primary execution, fixed work/waste,
  SLO, reservation-price or endogenous-price term.
- Continuous latency integral stops at winner; H/R execution cost continues
  through physical loser settlement. Replay cost occurs at creation once.
- Arrival kernel includes the same indivisible Reservation and online policy
  call semantics as the finite engine. The recommendation-conditional memory
  kernel is distinguished from the single-request intervention kernel.
- Exact finite state retains ordered linked attempts and policy information,
  including inert timer events that still append public history after settlement.
- Hazards integrate out hidden service sizes; future scheduled completion times
  are not simultaneously retained in this representation.
- Forward and backward use the same completion/timer/fault/dispatch maps;
  clock drift, dispatch atoms and common jumps are not counted twice.
- Finite joint probability, live Token intensity and event-index distribution
  are three different measures, with their respective conservation identities.
- Candidate FCFS entry mass is constrained by priority selection, capacity and
  non-idling; existence/uniqueness through common Replay atoms remains unproved.
- Candidate scaling and macro policy restrictions are explicit proposals.
- Event-indexed equations retain physical holding times; no unit-cost-per-event
  replacement, repeated post-arrival optimization, or hindsight minimum.
- Finite-state discretization needs a reconstruction/lumpability argument,
  service/timer precision and tail handling; a transition matrix alone does
  not close the original correlated queue game.

## Actual deterministic formula check

Command:

    .venv\Scripts\python.exe .scratch/token-mfg-restoration/theory-r1-algebra-check.py

Exit0. Standard-library Simpson integration checks one declared two-running-copy
event kernel, with arrival rate .9 and next deterministic boundary .4:

- continuous event probability: .7695125894131364;
- no-event boundary atom: .23048741058686353;
- probability normalization error: 1.1102230246251565e-16;
- mean holding time from kernel and survival: .21343298995350288;
- initial utility with winner at1 and running Hedge loser through2 at p=.5:
  direct and integrated cost both2;
- lognormal healthy q90 delay: 1.6385447242959017;
- Decimal cap2.8125 with unit reservation admits at most2 finite requests.

These are illustrative algebra/quadrature checks, not proofs of all equations,
queue simulations, fitted transitions, continuation samples or MFG validation.

## Actual required engineering regression

    .venv\Scripts\python.exe -m unittest discover -s tests -v

Exit0; 517 tests in158.756s; OK.
Log: `theory-r1-full-suite.log`.

    .venv\Scripts\python.exe -m mfg_hedge check --config configs\v1_minimal.json

Exit0; status ok; Python3.10.11.
The historical minimal-config finding remains: single-domain failure load1.4
exceeds target.9. This is not the new theory's load parameter or a new failure.
Log: `theory-r1-config-check.log`.

No production/configuration/accepted-ADR change and no experiment campaign.
Ticket05 remains claimed by its existing T2 workflow; this session did not
implement or modify its modules. Repository checks do not certify the proposed
MFG limit, private regret or an equilibrium.

## Outstanding theory gate

1. Justify the scale family and show it retains the intended strategic coupling.
2. Prove the forward problem/FCFS frontier is well posed in a stated policy class.
3. Specify a computable information state or belief approximation with error.
4. Bound numerical state/tail error in conditional action costs.
5. Establish applicable equilibrium existence/selection and finite-scale
   approximation conditions; do not assume continuity at hard admission edges.

The chosen next formulation is the exact event kernel and Bellman recursion,
followed by explicit finite-state approximation if needed. Continuous PDE
solution is not a prerequisite imposed on this project.
