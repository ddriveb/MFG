# Design reliability-aware multi-Replica routing

Type: design
Status: resolved
Blocked by: none

## Scope

Produce the pure design slice for reliability-aware routing of one Logical
Expert across eight exogenously placed Replicas. The design must remain
single-Primary and no-Hedge. It will become executable only after Proposed
ADR-0035 is accepted and this ticket is resolved.

## Required design decisions

Freeze, without running an experiment or changing production code:

1. the eight-Replica placement representation and failure-domain map;
2. the exact observable queue/load proxy and current service/health fields;
3. the causal failure-history window and hazard estimator;
4. failure-first, unavailable-Replica, tie-break, Replay, and complete-drain
   semantics using the existing physical reference;
5. the four scenario families: homogeneous reliability, two high-risk
   Replicas, non-stationary risk, and correlated failure domains;
6. deterministic definitions for Lazarus Algorithm 1, RR, JSQ, least
   unfinished work, reliability-only, and risk-aware JSQ;
7. paired finite-system metrics, per-Replica audits, CRN identity rules,
   provenance fields, and the future implementation ticket's sample/call
   budget;
8. the explicit boundary that MFG, prices, Hedge, placement optimization, and
   best-response claims are not part of the first implementation ticket.

## Acceptance criteria

- `.scratch/reliability-aware-multi-replica-routing/spec.md` is complete and
  consistent with accepted repository terminology.
- Proposed ADR-0035 records the physical boundary and claim boundary.
- No production source, test, config, or artifact is changed or created.
- No experiment or scheduler call is run.
- The next implementation ticket can freeze executable scenario and sampling
  values without reopening the routing semantics.

## Progress log

### Update: 2026-09-09 — Feature and design ticket created

Status: partial

#### Goal

Start the independent pure-design feature after closing the Selective Hedge
validation ticket as `development_no_candidate`.

#### Changed

- Created the feature spec for one Logical Expert with eight exogenously
  placed Replicas, one Primary per Token, and no Hedge.
- Created Proposed ADR-0035 for the causal routing boundary and the finite
  baseline family.
- Created and claimed this design ticket.

#### Verification

- Confirmed ticket 16 is resolved with the development-only terminal result.
- Confirmed this feature has exactly one claimed ticket and no production
  implementation ticket.
- No experiment, scheduler call, configuration change, or artifact was run or
  generated in this design session.

#### Artifacts

- `.scratch/reliability-aware-multi-replica-routing/spec.md`
- `docs/adr/0035-reliability-aware-multi-replica-routing.md`
- `.scratch/reliability-aware-multi-replica-routing/issues/01-design-reliability-aware-routing.md`

#### Decisions and risks

The current negative Selective Hedge result is retained as a finite
development finding only. It is not generalized to all Hedge mechanisms. The
new feature starts with routing so reliability and congestion effects remain
separable from active redundancy.

#### Next

Complete the remaining pure design decisions, then request review of
Proposed ADR-0035 before creating any implementation ticket.

### Update: 2026-09-09 — Reliability-aware routing design completed

Status: completed

#### Goal

Finish the finite-system design for routing one Logical Expert across eight
exogenously placed Replicas, while keeping placement, Hedge, prices, and MFG
outside the first implementation slice.

#### Changed

- Froze one Logical Expert, eight equivalent Replicas, four failure domains,
  per-Replica FCFS queues, and a global unassigned queue.
- Defined causal idiosyncratic and domain failure histories, a trailing-window
  Gamma-Poisson hazard estimator, burn-in, failure/requeue ordering, repeated
  reactive Replay, and complete drain.
- Defined six deterministic policies: round-robin, JSQ, least observable
  expected work, reliability-only, risk-aware JSQ, and a batch-faithful
  Lazarus Algorithm 1 adaptation.
- Froze four mechanism scenarios, stable SHA-256 CRN identities, metrics,
  per-Replica audits, invariants, and a 192-call non-inferential smoke bound.
- Separated the future decentralized Token MFG interface from centralized
  mean-field control, without implementing either solver.
- Expanded ADR-0035 to record the complete physical, causal, comparator, and
  claim boundary. Its status is `Proposed (ready for confirmation)`.

#### Verification

- Cross-checked the specification against the existing failure, Replay, CRN,
  Token-player, shared-capacity, and Lazarus-comparator terminology.
- Confirmed that the design conditions on placement and does not claim to
  improve Expert allocation or fault-tolerant placement.
- Confirmed that no production source, test, configuration, or historical
  artifact was changed and that no scheduler or experiment was run.
- `git diff --check` passed after the final document update.
- `.venv/Scripts/python.exe -m mfg_hedge check --config
  configs/v1_minimal.json` returned exit 0 with `status: ok`; the existing
  single-domain failure headroom diagnostic remains unchanged.

#### Artifacts

- `.scratch/reliability-aware-multi-replica-routing/spec.md`
- `docs/adr/0035-reliability-aware-multi-replica-routing.md`
- `.scratch/reliability-aware-multi-replica-routing/issues/02-implement-finite-multi-replica-routing.md`

#### Decisions and risks

The isolated engine permits repeated reactive Replay because a Token may be
routed through several failing Replicas, but never permits two live copies.
This deliberately does not change historical two-Replica engines. Exact
failure rates, estimator hyperparameters, workload scale, policy coefficients,
seeds, and namespaces still have to be frozen before implementation.

#### Next

Wait for explicit acceptance of ADR-0035. After acceptance, claim ticket 02,
freeze its numerical protocol, and implement only the bounded finite-system
slice and 192-call mechanism smoke.

## Answer

The feature now has a complete, reviewable routing model. It treats placement
as an external input, evaluates one-copy routing under causal reliability and
congestion observations, includes strong named comparators, and exposes a
future population-state interface without calling the finite heuristic an
MFG. No implementation or experiment is authorized until ADR-0035 is accepted.
