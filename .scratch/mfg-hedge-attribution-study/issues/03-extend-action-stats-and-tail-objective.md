# Extend ActionStats and implement the tail/work objective

Type: task
Status: blocked
Blocked by: 08 (model redesign disposition)

## Goal

Implement the attribution-only ActionStats schema, D/Primary-aware calibration,
complete work accounting, and hard-response cost contract without changing
historical schema-3/schema-4 behavior.

## Scope

- Add schema-5 configuration for the frozen SLOs, penalties, load 0.45, episode
  dimensions, quota scales, and controller semantics.
- Calibrate the exact 36-cell D/Primary-0 destination-B-load surrogate on the
  fixed rho grid with at least 2,000 independent probes per cell and cell-
  internal CRN; use only stable calibration load-carrier namespaces.
- Add actual deadline miss, expected excess, Primary/Hedge/Replay/total/wasted
  lifecycle work, and execution-phase x Replica work.
- Implement MYOPIC hard argmin inputs and the A3 hard-best-response/dual solver
  certificate; no Softmax fallback and no holdout execution.

## Acceptance criteria

1. Algebra, finite/type validation, legacy-table rejection, provenance mismatch,
   36-cell completeness, sample floor, unique in-grid all-Normal fixed point,
   arm-independent `g_budget`, and no-clamp behavior are tested.
2. A strictly more expensive slack action receives exactly zero probability;
   ties follow Normal < Delayed < Immediate.
3. MYOPIC queries only the frozen all-Normal baseline destination load and has
   no induced-load feedback.
4. A3 reports valid support, the nominal A reservation, planned B load/price,
   the separate quota price, complementarity for both constraints, destination-
   load residuals, and deterministic multi-solution witness, or an honest stable
   infeasible/nonconverged status; each quota scale is solved separately.
5. Historical configs/solvers/calibration serialization remain reproducible;
   full tests and environment check pass.

## Progress log

### Update: 2026-09-05 — Pause implementation for mathematical redesign

Status: partial

#### Goal

Inspect the attribution calibration/objective scope before implementation.

#### Changed

Claimed the ticket, then paused it following the user's mathematical-review
and design-first instructions. Added a pointer to ticket 08/Proposed ADR-0011.
No Python, test, configuration or calibration table was written.

#### Verification

Read the active spec, ADRs and existing configuration/calibration/solver APIs.
The subsequent ticket-08 documentation session ran the existing suite:
334 tests OK; the minimal environment check returned status ok.
No ticket-03 Red/Green implementation cycle has occurred.

#### Artifacts

Ticket status/pointer only; see ticket 08 for the proposed model documents.

#### Decisions and risks

The old requested-policy scalar fixed point is under redesign. A resolved
design ticket does not by itself authorize resuming the obsolete scope.

#### Next

Reconcile this scope after disposition of ADR-0011.

## Answer

## Comments

2026-09-05: The user requested mathematical review and design before further
implementation. Only this ticket's claim was made; no ticket-03 production
code was written. Implementation is paused pending disposition of ticket 08
and Proposed ADR-0011. The existing scope is retained as history and must not
be implemented unchanged merely because ticket 08's design work is resolved.
