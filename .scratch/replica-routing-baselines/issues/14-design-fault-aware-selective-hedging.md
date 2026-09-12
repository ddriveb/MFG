# Design fault-aware, deadline-aware selective hedging

Type: design
Status: resolved
Blocked by: none

## Scope

Design the next finite-system selective-Hedge slice suggested by the
Budgeted LÆDGE holdout. This ticket is design-only: it may add a Proposed ADR
and protocol documentation, but must not modify production code, prices,
Reservation semantics, historical engines, existing configurations, or
artifacts, and must not run a new experiment.

The design must preserve Budgeted LÆDGE's work-conserving Primary/Replay
priority and conservative cancellation. The new policy is intended to test
whether fault-domain direction and deadline slack can make a small amount of
Hedge useful relative to dynamic No-Hedge.

## Required decisions before implementation

Freeze explicitly:

- eligibility states and Replica direction;
- H/D/F/R handling;
- deadline and Token-class ordering;
- stable tie-breaking;
- whether any minimum-slack gate exists;
- causal observation fields and prohibited information;
- expected-work budget admission and non-refundable accounting;
- suppression reason vocabulary;
- exact baseline arms and mechanism comparisons;
- development/holdout namespaces, seeds, fingerprints, sample sizes, and
  scheduler-call budget for the later experiment;
- success, insufficient, physical-failure, and non-improvement dispositions.

The later experiment must separate dynamic Primary-routing benefit from
incremental Hedge benefit and compare first against Budgeted LÆDGE `delta=0`.
It must retain preemptive cancellation only as an oracle comparator, if used.

## Explicit non-goals

- no production implementation in this design ticket;
- no price feedback or Token-MFG loop;
- no continuation predictor or learned score;
- no best response, regret, Nash, or MFG claim;
- no retuning based on the existing holdout artifact;
- no reuse, overwrite, or reinterpretation of historical artifacts.

## Acceptance criteria

1. ADR-0034 remains `Proposed` until its required parameters are complete and
   explicitly reviewed.
2. The protocol states the exact causal observation boundary and proves that
   future service draws, future faults, future arrivals, other Tokens' private
   queues, and counterfactual baseline trajectories are unavailable.
3. The protocol distinguishes routing-only effects from Hedge effects and
   preserves all existing work, cancellation, Replay, CRN, and invariant
   semantics.
4. The later experiment arithmetic is closed before any implementation or
   execution ticket is created.
5. Existing tests, configs, and artifacts are untouched by this ticket.

## Progress log

### Update: 2026-09-09 — Started selective-hedging design

Status: partial

#### Goal

Start the selective-hedging work from the finite holdout evidence while
keeping the existing Budgeted LÆDGE result immutable and avoiding premature
production or experiment changes.

#### Changed

- Created and claimed this design ticket.
- Added Proposed ADR-0034 defining causal fault-aware and deadline-aware
  selective-Hedge semantics, conservative cancellation, and the separation
  between routing and Hedge effects.
- Recorded the required parameters that must be frozen before implementation.

#### Verification

- Read `AGENTS.md`, `CONTEXT.md`, issue-tracker/update-format instructions,
  ADR-0032, ADR-0033, the replica-routing spec, ticket 09, ticket 12, and
  the current holdout artifact contract.
- Confirmed ticket 12 is resolved and the existing holdout artifact remains
  the immutable reference.
- No production tests, experiment calls, or artifact writes were run.

#### Artifacts

- `docs/adr/0034-fault-aware-selective-hedging.md`
- `.scratch/replica-routing-baselines/issues/14-design-fault-aware-selective-hedging.md`

#### Decisions and risks

The proposed policy does not yet have frozen slack thresholds, sample sizes,
or experiment identities, so implementation and execution remain blocked by
design completion. No claim is made that selective hedging improves the
finite system.

#### Next

Review and accept ADR-0034 only after the causal parameters and later
experiment protocol are fully frozen; then create a separate implementation
ticket.

### Update: 2026-09-09 — Frozen selective-hedging protocol

Status: completed

#### Goal

Close the design ticket by freezing the causal slack definition, development
grid, qualification gate, and independent development/holdout protocol.

#### Changed

- Froze `s_i(t) = deadline_class - (t - arrival_time)` and the observable
  ratio `rho=s/deadline`; expired Tokens are explicitly suppressed.
- Froze the 3×3 Regular/Urgent threshold grid over `{0.15, 0.30, 0.50}`.
- Froze stable candidate ordering by slack, class priority, arrival time, and
  Token ID.
- Froze the paired qualification gate against Budgeted LÆDGE `delta=0%`:
  one of D/F CVaR95 or miss must improve with a paired 95% CI, the other may
  not have a positive upper CI, work increase must be at most 5%, and storm
  peak must be at most 2.
- Froze 256 development episodes / 3,072 calls and 1,024 holdout episodes /
  4,096 calls under disjoint namespaces and macro seeds.
- Updated ADR-0034 to `Proposed (ready for confirmation)`; it remains
  unaccepted until explicit confirmation.

#### Verification

- `git diff --check` passed.
- Confirmed no production source, configuration, test, or artifact was
  modified or generated.
- No scheduler call, calibration, holdout, or bootstrap was run.

#### Artifacts

- `docs/adr/0034-fault-aware-selective-hedging.md`
- `.scratch/replica-routing-baselines/issues/14-design-fault-aware-selective-hedging.md`

#### Decisions and risks

The design now has closed sample and call arithmetic, but production
implementation remains gated on ADR-0034 acceptance. The success gate is a
pre-registered finite-system comparison rule, not a claim that the candidate
will pass.

#### Next

After ADR-0034 is explicitly accepted, start the separate implementation
ticket; do not run the development panel in the implementation slice.

## Answer

ADR-0034 is ready for explicit confirmation. Ticket 14 is resolved after
freezing the causal observation, threshold grid, paired delta-zero gate, and
development/holdout provenance without changing production behavior.
