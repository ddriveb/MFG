# ADR-0015: Shared-Backup capacity and an Expert-level game

Status: Accepted (confirmed by the user on 2026-09-05)

Date: 2026-09-05

## Context

Accepted ADR-0014 has independent Expert queue triplets conditional on a common
fault. There is no strategic transition/cost coupling between Experts. The
user requested a nondegenerate game layer and a route to common-noise MFG.

## Proposed decisions

1. Players are Logical Expert controllers, each minimizing its own population
   latency/tail/Replay/work/waste objective over causal policies. Preserve
   fixed Gate and Primary A with same-Expert Backups B/C.
2. Couple all B/C running heads through one pool of capacity N*c_B with equal
   sharing and speed cap1 per head: h(k)=min(1,c_B/k), h(0)=1. Count Primary,
   Replay, Hedge and running-loser work. Keep local FCFS order. Main proposed
   c_B=.5; .25/1/2 are named diagnostics, not fitted capacity values.
3. Retain independent per-Expert prospective reservations. Actions N/D/S/X
   mean Normal/delayed single/immediate single/immediate dual; charges0/1/1/2.
   Reject an unaffordable request to Normal without partial downgrade.
4. Separate scheduled-fault diagnostics from random common-fault development.
   The latter reveals only past/current phase and age. Its mean field is
   Law(Y_t | public fault history through t), never future fault revelation.
5. Scale arrivals and capacity proportionally to N, keeping per-Expert budgets
   and speeds fixed. Use nested independent Expert traces and stable local
   destination marks in a new namespace, replacing global-ID parity only here.
6. Start with an unpriced congestion game. A separately named occupancy tariff
   can charge actual execution; it is not automatically a dual or social toll.
   Transfers are excluded from social welfare. Complementarity checks require
   a further explicitly defined market and are not imposed on this scheduler.
7. Start with a256-rule, symmetric pure-strategy best-response search; cycles
   and failure to find an equilibrium are reportable outcomes. Risk-sensitive
   mixtures and ordinary softmax do not supply a Nash certificate.
8. Validate exact physical interference, conditional forward predictions,
   finite-system unilateral deviations with the pool recomputed, size scaling,
   and welfare/safety separately. Restricted regret is not unrestricted regret.
   Approximate centralized search is a comparator, not a proved optimum.

## Supersession boundary and consequences

This is design only. No experiment or production change is activated. A later
bounded disposition must enable each implementation slice. For the proposed
new family only, replace ADR-0014 independent Backup capacity, global-ID
destination parity, fixed-backup-count policy semantics, deterministic-only
clock and320 cutoff with the specification's explicit definitions. Preserve
all historical engines/configurations/results and ADR-0005 conservative
lifecycle and failure-first ordering. Keep existing paused tickets paused.

The primary implementation challenge is global rescheduling after active-head
changes. Existing independent-Expert simulation loops cannot simply be summed
to produce a shared-capacity result. Shared bottleneck validity, risk-sensitive
equilibrium existence, restricted strategy expressiveness, unknown-fault
information and finite-N error remain explicit model risks.

## Specification

Full mathematics, observations, fixtures and verification protocol:
`.scratch/shared-backup-game/spec.md`.

Design ticket: `.scratch/shared-backup-game/issues/01-design-shared-backup-game.md`.

Sources: https://arxiv.org/abs/1210.5771 ; https://arxiv.org/abs/1407.6181 ;
https://arxiv.org/abs/1210.5780 . These do not prove this proposed model valid.
