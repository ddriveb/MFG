# Implement fault-aware, deadline-aware selective hedging

Type: implementation
Status: resolved
Blocked by: none

## Scope

Implement only the accepted ADR-0034 selective-Hedge policy after the ADR is
explicitly accepted. Preserve the Budgeted LÆDGE work-conserving
Primary/Replay priority, conservative cancellation, non-refundable budget
ledger, complete drain, CRN streams, and all historical behavior.

The implementation must support the frozen 3×3 development threshold grid,
the frozen causal observation boundary, stable candidate ordering, explicit
suppression reasons, and deterministic arm identity. It must not run the
development or holdout panels in this ticket.

## Frozen protocol to preserve

- Absolute slack: `s_i(t) = deadline_class - (t - arrival_time)`.
- Unitless normalized slack: `bar_s_i(t) = s_i(t)/deadline_class`.
- Eligibility: `0 < bar_s_i(t) <= theta_class`; expired Tokens are suppressed
  as `deadline_expired`.
- `theta_Regular, theta_Urgent ∈ {0.15, 0.30, 0.50}` for development.
- Only D state, Primary on degraded A, healthy/startable B, no existing
  Hedge, and no startable unserved Primary/Replay.
- Hedges are disabled in H/R and forbidden in F.
- Candidate order: `(deadline_slack, class_priority, arrival_time, token_id)`.
- Development: 256 episodes, 3,072 calls, namespace and seed from ADR-0034.
- Holdout: 1,024 episodes, 4,096 calls, namespace and seed from ADR-0034;
  execution is outside this implementation ticket.

## Acceptance criteria

1. Red tests precede production changes and cover causal observation,
   D/A-to-healthy-B eligibility, H/F/R suppression, deadline expiry,
   stable ordering, Primary/Replay priority, budget admission, conservative
   loser completion, CRN/fingerprint preservation, and deterministic repeat.
2. The implementation is isolated from historical fixed-dispatcher, NIIN,
   LÆDGE, cancellation, and Budgeted LÆDGE behavior.
3. No future service draw, remaining work, future fault/arrival, private queue,
   or counterfactual baseline trajectory is observable to the policy.
4. Focused, affected, full-suite, and config checks pass.
5. No development/holdout scheduler call or formal artifact is created.

## Progress log

### Update: 2026-09-09 — Created implementation gate

Status: open

#### Goal

Create the bounded implementation ticket after design closure, while waiting
for explicit ADR-0034 acceptance.

#### Changed

- Created this unclaimed implementation ticket.
- Copied the frozen causal and sample/call contract from ADR-0034.

#### Verification

- Ticket 14 is resolved and no other current ticket is claimed.
- No production code, test, configuration, experiment, or artifact changed.

#### Artifacts

- `.scratch/replica-routing-baselines/issues/15-implement-selective-hedging.md`

#### Decisions and risks

Implementation and experiment execution remain gated on explicit acceptance
of ADR-0034. The existing Budgeted LÆDGE holdout remains immutable.

#### Next

After ADR-0034 acceptance, begin with the failing test set.

### Update: 2026-09-09 — ADR accepted and implementation claimed

Status: partial

#### Goal

Enter the implementation slice after resolving the slack units and receiving
explicit confirmation of ADR-0034.

#### Changed

- Accepted ADR-0034 with the user's confirmation date.
- Claimed ticket 15 as the only active ticket.
- Made the absolute-versus-normalized slack distinction explicit in the
  implementation contract.

#### Verification

- Confirmed `bar_s=s/deadline` is dimensionless and the threshold grid is
  applied to `bar_s`, not to absolute time.
- No production code, test, scheduler call, or artifact was changed or run.

#### Artifacts

- `docs/adr/0034-fault-aware-selective-hedging.md`
- `.scratch/replica-routing-baselines/issues/15-implement-selective-hedging.md`

#### Decisions and risks

Implementation remains bounded to the causal selective-Hedge policy. The
development and holdout panels remain outside this ticket.

#### Next

Write the required failing tests before modifying production code.

### Update: 2026-09-09 — Implemented selective-Hedge kernel

Status: completed

#### Goal

Implement the causal fault-aware, deadline-aware selective-Hedge kernel from
accepted ADR-0034 without changing historical engines or running development
or holdout experiments.

#### Changed

- Added isolated `src/mfg_hedge/selective_hedging.py`.
- Added immutable `SelectiveHedgeThresholds`, `SelectiveSuppression`, and
  `SelectiveHedgeResult` values.
- Implemented explicit unitless normalized slack
  `bar_s=s/deadline`, deterministic arm identity, and the frozen threshold
  grid `{0.15, 0.30, 0.50}`.
- Restricted Hedge eligibility to degraded state, Primary on Replica A, an
  available healthy Replica B, no existing Hedge, and no startable
  Primary/Replay waiting work.
- Preserved Primary/Replay priority, conservative running-loser completion,
  non-refundable budget admission, CRN draw binding, complete drain, and
  historical engines.
- Added explicit suppression reason auditing for deadline, state, direction,
  threshold, and budget decisions.
- Added `tests/test_selective_hedging.py` with causal, ordering, budget,
  deterministic, and δ=0 physical-equivalence coverage.
- Kept the module internal; no package export and no version bump.

#### Verification

- Real Red before implementation: focused collection failed with
  `ModuleNotFoundError: No module named 'mfg_hedge.selective_hedging'`.
- Focused suite: `7/7` passed:
  `.venv\Scripts\python.exe -m unittest tests.test_selective_hedging -v`.
- Affected suites: `62/62` passed, covering selective-Hedge, Budgeted LÆDGE,
  attribution, and Budgeted LÆDGE campaign contracts.
- Full suite: `704/704` passed in `206.066s`:
  `.venv\Scripts\python.exe -m unittest discover -s tests -v`.
- Config check returned `status: ok`; the existing informational
  single-domain failure-load finding remains unchanged.
- `git diff --check` passed.
- No development/holdout scheduler call, calibration, bootstrap, or formal
  artifact was generated.

#### Artifacts

- `src/mfg_hedge/selective_hedging.py`
- `tests/test_selective_hedging.py`
- `.scratch/replica-routing-baselines/issues/15-implement-selective-hedging.md`

#### Decisions and risks

The implementation is a finite causal policy kernel, not a selected policy
and not an experiment result. The 3×3 threshold search and later 256/1,024
episode panels remain governed by ADR-0034 and were not run. No price,
predictor, BR, regret, Nash, or MFG behavior was added.

#### Next

Create a separate experiment ticket only if the development threshold search
is explicitly authorized; do not consume the frozen panel calls from this
implementation ticket.

## Answer

ADR-0034 selective-Hedge implementation is resolved and verified. It adds
only the isolated causal kernel and tests; historical scientific behavior and
artifacts are unchanged, and no formal experiment has started.
