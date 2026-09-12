# Design Budgeted LÆDGE work-constrained idle-release baseline

Type: design
Status: resolved
Blocked by: none

## Scope

Produce the pure design for a conservative-cancellation Budgeted LÆDGE finite-
system baseline and its later development/holdout evaluation. The design
preserves the existing idle-release scheduler, fault-first semantics, complete
drain, and CRN contracts while adding causal work-based admission and the
frozen seven-point work-target scan:
`{0%, 1%, 3%, 5%, 8%, 12%, 18%}` relative to NIIN conservative mean total
executed work.

This ticket changes no production code, configuration, historical artifact, or
experiment result. It does not run development, holdout, or Pareto evaluation.
The next implementation ticket must not be created until ADR-0033 is explicitly
accepted.

## Design contract

The complete protocol is specified by [ADR-0033](../../docs/adr/0033-budgeted-laedge-work-constrained-idle-release.md),
including:

- unassigned arrival waiting and work-conserving idle-release order;
- Primary/Replay priority before Hedge and stable tie-breaking;
- conservative running-loser completion, queued-loser zero-work cancellation,
  and preemptive oracle-only semantics;
- causal expected-work commitments without future service reads;
- `W=25` windows with forced H/D/F/R cuts, nonrefundable charged work, and no
  carry-over;
- admission `nonrefundable_charged_work_w + expected_work_i <= planned_budget_w`
  with fixed `expected_work_i=1.0`, plus separate launch-cohort and execution-
  time work ledgers;
- separate planned, committed, realized, overshoot, unused, launch, and
  suppression audits;
- development-to-budget-rate mapping for the seven delta targets;
- disjoint development/holdout namespaces and fail-closed calibration;
- exactly 11 unique comparison arms, with Budgeted LÆDGE `delta=0` serving as
  the sole work-conserving No-Hedge row, plus the required metrics, paired CRN,
  and Pareto/dominance rules;
- finite-system-only claim boundary with no optimality, BR, Nash, or MFG claim.

Concrete development/holdout episode counts, namespaces, macro seeds,
fingerprint rules, scheduler-call budgets, and artifact run IDs remain to be
frozen in a later execution ticket; ADR-0033 authorizes no run without them.

## Acceptance criteria

1. A Proposed ADR-0033 exists and freezes all ten requested contract groups.
2. This ticket records that the work is design-only and that no implementation
   ticket may be created before ADR acceptance.
3. No production source, configuration, historical artifact, or experiment
   result is modified or generated.
4. The design keeps conservative LÆDGE as the primary baseline and preemptive
   LÆDGE as oracle/upper-bound semantics only.

## Progress log

### Update: 2026-09-09 — Complete Proposed ADR design

Status: completed

#### Goal

Freeze the causal, work-constrained Budgeted LÆDGE idle-release protocol as a
pure design, without modifying production code or running an experiment.

#### Changed

- Added Proposed [ADR-0033](../../docs/adr/0033-budgeted-laedge-work-constrained-idle-release.md).
- Added this design-only ticket and explicitly deferred implementation until
  ADR-0033 is accepted.
- Froze scheduler semantics, conservative/preemptive cancellation roles,
  causal work-budget ledger, `W=25` window cuts, seven delta targets,
  development/holdout isolation, comparison arms, metrics, Pareto analysis,
  and claim boundaries.

#### Verification

- Read and applied `AGENTS.md`, `CONTEXT.md`, the local issue-tracker and
  update-format instructions, ADR-0030, ADR-0031, ADR-0032, and the active
  replica-routing baseline spec.
- Confirmed the next unused ADR number is 0033 and the next unused ticket
  number is 08.
- No production tests, config check, experiment, or artifact generation was
  run because this session is documentation-only.

#### Artifacts

- `docs/adr/0033-budgeted-laedge-work-constrained-idle-release.md`
- `.scratch/replica-routing-baselines/issues/08-design-budgeted-laedge.md`

#### Decisions and risks

ADR-0033 remains Proposed. Exact development/holdout sample counts,
namespaces, seeds, fingerprints, call budget, and run IDs are intentionally
not invented here and must be frozen before execution. No result can claim a
pathwise realized hard cap because conservative cancellation and unknown
service draws permit overshoot.

#### Next

Obtain explicit ADR-0033 acceptance; only then create a separate implementation
ticket and freeze the execution sample/provenance protocol.

### Update: 2026-09-09 — Narrow contract revision requested

Status: completed

#### Goal

Resolve the three design ambiguities identified during ADR review while
keeping this ticket design-only and resolved.

#### Changed

- Replaced refundable outstanding-commitment admission with the monotone
  nonrefundable formula and fixed v1 normalized expected work of `1.0`.
- Split launch-cohort realized work from execution-time realized work and
  specified that cross-window running work continues physically without refund
  or double counting.
- Reduced the comparison-arm contract to exactly 11 unique arms by making
  Budgeted LÆDGE `delta=0` the sole work-conserving No-Hedge row.
- Synchronized the budget, window-audit, calibration, metric, and arm-count
  wording in Proposed ADR-0033.

#### Verification

- Confirmed ADR-0033 remains `Status: Proposed`.
- Confirmed this ticket remains `Status: resolved`.
- No production source, configuration, version, historical artifact, test
  result, or experiment output was modified or generated.

#### Artifacts

- [ADR-0033](../../docs/adr/0033-budgeted-laedge-work-constrained-idle-release.md)
- This design ticket.

#### Decisions and risks

The outstanding expected-work field is audit-only and cannot restore budget
allowance. The two work views are deliberately complementary rather than
additive. Exact execution sample/provenance values remain deferred until ADR
acceptance and a separate implementation/experiment ticket.

#### Next

Obtain explicit acceptance of ADR-0033; then create the implementation ticket.
