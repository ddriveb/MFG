# Implement Budgeted LÆDGE work-constrained idle-release baseline

Type: implementation
Status: resolved
Blocked by: none

## Scope

Implement the isolated Budgeted LÆDGE conservative-cancellation kernel,
causal work-budget ledger, forced window accounting, and deterministic audit
surface specified by accepted ADR-0033. Preserve the existing LÆDGE,
fixed-dispatcher, NIIN, cancellation, CRN, workload, and historical artifact
behavior. The implementation must not add prices, best response, regret, Nash,
MFG, or a development/holdout execution runner in this ticket.

The implementation must support the unique 11-arm protocol needed by the
later evaluation, while this ticket only validates the core engine and arm
identity contract:

1. fixed-dispatcher No-Hedge;
2. NIIN conservative;
3. Budgeted LÆDGE conservative at `delta={0%,1%,3%,5%,8%,12%,18%}`;
4. unconstrained LÆDGE conservative;
5. unconstrained LÆDGE preemptive oracle.

The Budgeted LÆDGE `delta=0` row is the sole work-conserving No-Hedge row; no
separate duplicate arm may be created.

## Frozen later-experiment protocol

These values are frozen now so implementation choices cannot change the later
sample or call arithmetic:

- Development: `256` episodes, namespace
  `replica-routing-baselines:budgeted-laedge:v1:development`, macro seed
  `20260910`.
- Holdout: `1,024` episodes, namespace
  `replica-routing-baselines:budgeted-laedge:v1:holdout`, macro seed `20260911`.
- Namespaces are disjoint from one another and from all existing baseline,
  cancellation, Token, and MFG runs. No episode identity may be reused across
  the two splits.
- Every episode's four external streams and common fault timeline are shared
  by all 11 arms. A split's episode identities are the dense ordered indices
  `0..episode_count-1`.
- `episode_trace_fingerprint_v1` is SHA-256 over UTF-8 canonical JSON using
  `sort_keys=true`, compact separators, and `allow_nan=false`, covering
  namespace, macro seed, episode index, protocol, arrivals, Token classes,
  attempt-0/1/2 streams, and common fault timeline. The exact per-episode
  digest is generated from these frozen inputs and recorded before simulation.
- `protocol_fingerprint_v1` uses the same canonical encoding over ADR-0033,
  the frozen environment, arm list, delta list, and accounting schema.
- `source_bundle_fingerprint_v1` is the SHA-256 of the sorted relative-path
  and byte-content manifest for the complete transitive execution source
  bundle; it must be recorded in later artifacts and cannot be inferred from
  a partial module list.
- Scheduler-call budget is exactly `14,080 = (256 + 1,024) * 11`, split as
  `2,816` development calls and `11,264` holdout calls. Bootstrap/statistics
  work is not a scheduler call. Failed calls are fail-closed, never retried,
  supplemented, deleted, or replaced.
- Later paired statistics, if authorized, use episode as the cluster unit and
  a separate frozen bootstrap namespace; those statistics are not run by this
  ticket.

## Implementation contract

- Arrival enters unassigned waiting; idle-release serves unserved Primary and
  Replay before considering a Hedge.
- Each Token has at most one Hedge and one Replay; ordering is stable by
  `(arrival_time, token_id)`.
- F cannot start work on failed Replica A; fault-first and complete drain are
  unchanged.
- Conservative running losers continue to completion and remain in realized
  and wasted work. Preemptive cancellation is oracle-only.
- Admission uses
  `nonrefundable_charged_work_w + expected_work_i <= planned_budget_w` with
  `expected_work_i=1.0`; charged work only increases within a window and
  outstanding expected work never refunds it.
- Windows have maximum width `W=25` and are cut at H/D/F/R boundaries. A
  running attempt crosses a boundary physically; only new admission allowance
  resets.
- Audits preserve both `launch_cohort_realized_work` and
  `execution_time_realized_work`, without double counting them as physical
  work.
- Invalid IDs, non-finite values, mutable input, CRN mismatch, conservation
  failure, stale-generation misuse, and budget ledger inconsistency fail fast.

## Acceptance criteria

1. Failing tests precede production implementation and cover idle-release
   priority, causal budget admission, nonrefundable charging, cross-window
   execution, both work ledgers, conservative loser behavior, δ=0 semantics,
   deterministic arm identities, and invalid inputs.
2. The new core is isolated from historical LÆDGE and fixed-dispatcher paths.
3. Small deterministic traces prove exact queue/winner/Replay/drain behavior
   and per-window accounting.
4. No development/holdout experiment or artifact is run or generated.
5. Focused, affected, full suite, and config checks pass before resolution.

## Progress log

### Update: 2026-09-09 — Claimed implementation slice

Status: claimed

#### Goal

Implement and test the accepted ADR-0033 Budgeted LÆDGE kernel and causal
work-ledger layer, while freezing the later evaluation sample/provenance
contract and not running experiments.

#### Changed

- Accepted ADR-0033 with the user's confirmation date.
- Created and claimed this implementation ticket.
- Frozen development/holdout episode counts, namespaces, macro seeds,
  fingerprint schemas, 11 unique arms, and exact 14,080-call arithmetic.
- No production code changed yet; the next action is the red test set.

#### Verification

- Read `AGENTS.md`, `CONTEXT.md`, issue-tracker/update-format instructions,
  ADR-0030, ADR-0031, ADR-0032, ADR-0033, the active baseline spec, and the
  existing LÆDGE/workload/test modules.
- Confirmed no other ticket is claimed.
- No experiment, artifact generation, development calibration, or holdout run
  was performed.

#### Artifacts

- `docs/adr/0033-budgeted-laedge-work-constrained-idle-release.md`
- `.scratch/replica-routing-baselines/issues/09-implement-budgeted-laedge.md`

#### Decisions and risks

The frozen call budget covers the later 11-arm development plus holdout panel;
this implementation ticket does not authorize using it. Preemptive behavior
remains an oracle comparator and no result from this slice can support a
Pareto, optimality, Nash, or MFG claim.

#### Next

Add the deterministic failing tests, then implement the smallest isolated
Budgeted LÆDGE engine and ledger.

### Update: 2026-09-09 — Implemented and verified

Status: resolved

#### Goal

Implement only the finite-system Budgeted LÆDGE conservative idle-release
kernel, causal work ledger, forced-window audit, and deterministic tests.  No
development calibration, holdout evaluation, Pareto selection, price
feedback, best response, Nash, or MFG work was run.

#### Changed

- Added the isolated internal module
  `src/mfg_hedge/budgeted_laedge.py`.
- Added strict immutable `BudgetWindowAudit` and `BudgetedLaedgeResult`
  values, including non-refundable charged work, audit-only outstanding
  expected work, signed reconciliation, launch-cohort realized work,
  execution-time realized work, budget error, overshoot, unused budget,
  launches, and suppressions.
- Added the causal admission rule with fixed `expected_work_i=1.0`; it never
  reads the future Hedge service draw and never refunds completed work.
- Added 25-unit windows cut at common-state boundaries, while running work
  continues physically across boundaries and is reported under both frozen
  work views.
- Added conservative running-loser completion accounting and support for the
  explicitly oracle-only preemptive switch without changing historical
  LÆDGE behavior.
- Frozen the unique 11-arm identity set, seven delta points (with one and
  only one delta-zero dynamic No-Hedge row), later development/holdout
  namespaces, seeds, fingerprints, episode counts, and `14,080` scheduler
  calls in this ticket.
- Added `tests/test_budgeted_laedge.py`; the module is not exported from
  `mfg_hedge.__init__`, so the package version was not bumped.

#### Verification

- Real Red before implementation: focused test collection failed with
  `ModuleNotFoundError: No module named 'mfg_hedge.budgeted_laedge'`.
- Focused suite: `10/10` passed.
- Affected suites (Budgeted LÆDGE, LÆDGE, attribution, and cancellation
  protocols): `92/92` passed.
- Full suite: `677/677` passed.
- Config check: passed with `status: ok`; the existing transient
  single-domain failure-load diagnostic remains informational and unchanged.
- Deterministic tests cover the 11 unique arms, delta-zero semantics,
  non-refundable admission, no future-draw access, forced state-boundary
  windows, cross-window work views, conservative running losers, existing
  attribution invariants, strict inputs, and immutable repeatability.
- No development/holdout calibration or formal experiment was run; no
  experiment artifact was generated.

#### Artifacts

- Implementation: `src/mfg_hedge/budgeted_laedge.py`
- Tests: `tests/test_budgeted_laedge.py`
- Frozen implementation ticket: this file
- No development, holdout, calibration, Pareto, candidate, or other formal
  experiment artifact was created.

#### Decisions and risks

- Historical fixed-dispatcher, NIIN, LÆDGE, cancellation, configuration, and
  artifact behavior is unchanged.
- Conservative cancellation is the default; preemptive cancellation remains
  an oracle/upper-bound option and is not a deployable or scientific claim.
- A planned budget is not a pathwise realized-work hard cap.  Overshoot is
  reported because the service draw is unknown and conservative losers are
  not forcibly terminated.
- The frozen future call arithmetic is implementation provenance only; it was
  not consumed by this ticket.

#### Next

After review, create a separate experiment ticket for development calibration.
Only after its mapping is frozen may an independent holdout Pareto run be
authorized.
