# Design Token conditional continuation cost oracle (T3A)

Type: design
Status: resolved
Blocked by: none

## Scope

Design the next bounded Token slice after resolved T2.  The deliverable is a
Proposed ADR for a pure Monte Carlo oracle over pre-registered observable bins,
estimating `Q_m(u|B(O)=b,S)` separately for the three ADR-0021 cost models and
`u in {N,D,I}`.  This ticket does not implement the oracle or run a pilot.

## Acceptance criteria

- Freeze a causal observation boundary, canonical observation fingerprint, and
  sampling-time bin schema/boundaries.
- Define an episode factory keyed by episode index/namespace/protocol, with no
  raw-observation or prefix-matching input.
- Define observable-bin episode identity, independent episode namespaces, split
  isolation, episode-local observation construction, and fail-closed missing
  cohorts.
- Bind the estimand to a pre-registered target-selection rule `S`; enforce at
  most one future-blind target per independent episode.
- Preserve paired N/D/I CRN within each episode while keeping different
  episodes independent; fail the whole episode if any branch fails and never
  auto-supplement it.
- Require cross-episode aggregation to use only `(selection_rule, bin_id)`;
  raw observations and prefix fingerprints need not match across episodes.
- Require complete fresh reruns with other Tokens using policy functions online,
  including Reservation projection, queue physics, Replay, timers, failures,
  losers and drain.
- Keep the three named cost models and their provenance separate.
- Define immutable input/rollout/estimate records with bin-conditioned mean,
  SE, effective episode count and missing status, without BR/regret/Nash/MFG
  fields or labels.
- Define the pre-implementation red test matrix and the later freeze gate for
  sample counts, failure policy and execution budget.
- Leave posted prices, Soft BR, forward feedback, mean field, solver and
  experiments explicitly deferred.

## Progress log

### Update: 2026-09-07 — Claim T3A design

Status: partial

#### Goal

Design a narrow conditional continuation oracle without activating predictor,
price feedback, best response, regret or MFG work.

#### Changed

Created and claimed this design ticket. No production source, configuration,
campaign or formal experiment artifact was changed.

#### Verification

Read `AGENTS.md`, `CONTEXT.md`, the local issue/update rules, accepted
ADR-0020/0021, the resolved T2 ticket, the resolved finite evolution/closure
ticket, T1/T2 source boundaries, and the restoration specification. Confirmed
that ticket 06 is a theory reference and does not already accept T3A.

#### Artifacts

None at claim time.

#### Decisions and risks

The design will use fixed observable-prefix conditioning, paired action CRN,
independent continuation namespaces, fresh complete counterfactual reruns, and
separate model estimands. No conditional law is assumed valid merely because a
prefix matches; its provenance and missing state will be audited.

#### Next

Write the Proposed ADR and record the explicit implementation/test gate.

### Update: 2026-09-07 — T3A observable-bin oracle design completed

Status: completed

#### Goal

Deliver the reviewable T3A design and stop before implementation or any
expectation-based action selection.

#### Changed

- Revised `docs/adr/0022-token-conditional-continuation-oracle.md` to estimate
  only `Q_m(u|B(O)=b)` under a pre-registered observable-bin schema; exact
  `Q(u|O)` is no longer claimed.
- Added the hard episode contract: at most one future-blind target per
  independent episode, paired N/D/I branches, all-or-nothing panel failure,
  no automatic supplemental draws, and episode-level effective sample size.
- Kept the causal observation/fingerprint contract, Reservation projection,
  complete fresh policy/queue reruns, separate cost-model estimands, and
  fail-closed oracle output.
- Defined proposed immutable records for observation/bin, episode request,
  rollout, and action-conditioned estimates. Effective sample size counts
  independent complete episode panels, not paired action branches.
- Added calibration/validation split isolation, deterministic namespace and
  ordering requirements, no-retry/incomplete-cohort handling, and the
  pre-implementation red regression matrix.
- Explicitly deferred posted-price pacing, Soft BR, best response, regret,
  forward evolution, conditional mean-field closure, Nash/MFG and campaigns.

#### Verification

- Confirmed the ADR is `Status: Proposed`; no acceptance or implementation
  authorization was inferred.
- Confirmed ticket 06 remains resolved and unchanged; its finite derivation is
  referenced as theory only.
- This was documentation-only design work. No predictor, rollout, experiment,
  campaign, solver, price feedback or MFG execution was run; no production
  tests were claimed.

#### Artifacts

- `docs/adr/0022-token-conditional-continuation-oracle.md`
- `.scratch/token-mfg-restoration/issues/07-design-token-conditional-continuation-oracle.md`

#### Decisions and risks

T3A estimates observable-bin-conditioned action costs only. A lower sample
mean is not a best-response certificate, and an observed action comparison is
not regret, Nash or MFG evidence. Conditional continuation-law correctness,
bin adequacy, coverage, sample allocation, and execution budget remain gates
for the future implementation ticket.

#### Next

User review and explicit acceptance of ADR-0022; then create a separate
implementation ticket that freezes sample counts, namespaces, failure policy,
and budget before any T3A pilot.

## Answer

The T3A design is complete as a separate Proposed ADR. Ticket 07 is resolved
as a documentation deliverable only. No exact `Q(u|O)` claim, predictor,
expected best response, regret, price feedback, Nash, MFG or experiment result
exists.

### Update: 2026-09-07 — Final document boundary check

Status: completed

#### Goal

Verify the new design files and the retained ticket boundary after writing the
ADR and closing this documentation ticket.

#### Changed

None; this was a read-only verification pass.

#### Verification

A PowerShell document check confirmed all of the following: ADR-0022, ticket
07, and ticket 06 exist; ADR-0022 remains `Status: Proposed`; the oracle
evidence label and all four false claim flags are present; calibration and
validation split isolation and complete paired branches are specified; ticket
07 remains resolved; and the explicit no-predictor/no-best-response boundary
is present.

No production test command was run because this session changed documentation
only and added no executable code.

#### Artifacts

None beyond the two documentation files already listed above.

#### Decisions and risks

No ADR acceptance, version change, implementation authorization, or experiment
authorization was inferred from the document check.

#### Next

None for this design ticket; await explicit user review of ADR-0022.

### Update: 2026-09-07 — Revise to pre-registered observable bins

Status: partial

#### Goal

Apply the user's narrow statistical-contract correction before any T3A
implementation: estimate only `Q_m(u|B(O)=b)` and make episode-level pairing
and failure semantics explicit.

#### Changed

- Replaced the exact-observation estimand language with the
  `observable_bin_conditional_cost_oracle` label and a pre-sampling-frozen bin
  schema, boundaries, endpoint rules, and fingerprint.
- Made each independent episode select at most one target using only the causal
  prefix; N/D/I remain paired within that episode.
- Made the episode the independent statistical unit. Any failed N/D/I branch
  fails the entire paired panel, with no retry or automatic supplemental draw.
- Updated the output records, effective sample size, missing/insufficient
  status, and pre-implementation test matrix accordingly. ADR-0022 remains
  Proposed.

#### Verification

The revised ADR and ticket were inspected for the new estimand, frozen-bin
boundary, one-target rule, episode-level independence, all-or-nothing branch
failure, no-supplement rule, and the absence of BR/regret/Nash/MFG claims.
The requested full suite and configuration check are recorded in the next
verification update after execution.

#### Artifacts

`docs/adr/0022-token-conditional-continuation-oracle.md` and this ticket.

#### Decisions and risks

The binned oracle deliberately does not identify exact `Q(u|O)`. Bin adequacy,
conditional-law validity, and precision remain future implementation gates.
No production code, version, sample, or experiment artifact changed.

#### Next

Run the full unittest discovery and minimal configuration check, then stop for
user review; do not implement T3A.

### Update: 2026-09-07 — Observable-bin revision regression check complete

Status: completed

#### Goal

Run the requested repository regression checks after the narrow ADR-0022
observable-bin revision.

#### Changed

None; no production source, configuration, version, sample library, or
experiment artifact changed during verification.

#### Verification

- `.venv\Scripts\python.exe -m unittest discover -s tests -v` — exit code 0;
  **Ran 535 tests in 149.378s; OK (535/535)**.  The current repository has
  535 tests because the resolved T2 suites are present; 517 is the older
  pre-T2 baseline, not the current full-suite count.
- `.venv\Scripts\python.exe -m mfg_hedge check --config
  configs\v1_minimal.json` — exit code 0; `status: ok`, Python 3.10.11.
  The existing diagnostic that single-domain failure gives base load 1.400
  above target 0.900 remains unchanged.

The revised documents retain `Status: Proposed`, the exact-observation claim
is explicitly disclaimed, and the observable-bin label,
pre-sampling-frozen boundaries, one-target episode rule, all-or-nothing
N/D/I failure rule, and no-supplement rule are present.

#### Artifacts

No test, rollout, predictor, campaign, or experiment artifact was generated.

#### Decisions and risks

The 535-test result is the accurate current verification count; it does not
authorize ADR-0022 or T3A implementation.  No BR, regret, Nash, MFG, price
feedback, or exact `Q(u|O)` claim was produced.

#### Next

User review and explicit acceptance of the revised ADR-0022, followed by a
separate implementation ticket if accepted.

### Update: 2026-09-07 — Remove cross-episode prefix conditioning

Status: completed

#### Goal

Resolve the remaining design contradiction by making the sampling order
episode-first and binding the estimand to a pre-registered target-selection
rule.

#### Changed

- Changed `ContinuationFactory` to receive only episode index, split/namespace,
  protocol, and exogenous-generation parameters; it no longer receives a raw
  observation, target, or bin to match.
- Specified the order `episode -> causal S target selection -> episode-local
  O_e -> B(O_e) -> paired N/D/I`, with at most one target per episode.
- Allowed different episodes to have different arrival times, queues, raw
  observations, and prefix fingerprints; only `(S, bin_id)` is shared for
  aggregation.
- Added `missing_target` consumption without replacement and made the
  estimand `Q_m(u|B(O)=b,S)` explicit.
- Restricted exact target-prefix equality to the three branches within one
  episode and retained the existing all-or-nothing failure rule.

#### Verification

Inspected ADR-0022 and this ticket to confirm that no cross-episode raw-prefix
matching or exact `Q(u|O)` claim remains, while ADR-0022 stays Proposed.  The
previous `535/535` full-suite and configuration-check results remain valid for
this documentation-only revision; no executable code changed, so they were
not rerun in this narrow correction.

#### Artifacts

`docs/adr/0022-token-conditional-continuation-oracle.md` and this ticket.

#### Decisions and risks

The resulting oracle estimates a selection-rule-weighted observable-bin
quantity.  Bin adequacy, target-selection representativeness, conditional-law
validity, and precision remain implementation gates.  No production code,
version, sample, or experiment artifact changed.

#### Next

User review and explicit acceptance of the corrected ADR-0022; only then may a
separate implementation ticket be created.

### Update: 2026-09-07 — Finalize episode-first target selection semantics

Status: completed

#### Goal

Remove the residual cross-episode raw-prefix conditioning contradiction and
make the observable-bin estimand depend on the pre-registered selector `S`.

#### Changed

- Finalized the flow as independent episode generation, causal `S` selection,
  episode-local `O_e`, binning `B(O_e)`, then paired N/D/I reruns.
- Made the factory independent of raw observations, targets, and bins. Each
  episode now owns its own observation and fingerprint; cross-episode prefixes
  may differ and only `(S, bin_id)` is aggregated.
- Added explicit `missing_target` consumption without replacement and
  `insufficient` cell handling without bin merging or sample expansion.
- Restricted exact target-prefix equality to the three branches of one episode.

#### Verification

Document inspection confirms ADR-0022 remains Proposed and contains the
selection-bound estimand `Q_m(u|B(O)=b,S)`, episode-indexed factory contract,
episode-local observations, cross-episode prefix freedom, targetless status,
and within-episode paired-prefix equality. No production test was rerun
because this was documentation-only and the prior 535/535 full-suite and
configuration results remain applicable.

#### Artifacts

`docs/adr/0022-token-conditional-continuation-oracle.md` and this ticket.

#### Decisions and risks

The oracle is now explicitly selection-rule weighted; it does not represent an
undefined “typical Token” in a bin. Target-selection representativeness and bin
adequacy remain future implementation gates. No implementation, campaign,
BR, regret, Nash, or MFG work was started.

#### Next

User review and explicit acceptance of ADR-0022; only after acceptance may a
separate T3A implementation ticket be created.

### Update: 2026-09-07 — ADR-0022 accepted

Status: completed

#### Goal

Record the user's explicit acceptance of the corrected observable-bin T3A
design without starting implementation.

#### Changed

- Updated ADR-0022 to `Status: Accepted (confirmed on 2026-09-07)`.
- Confirmed the accepted estimand is `Q_m(u|B(O)=b,S)` with episode-first
  generation, future-blind target selection, episode-local observations and
  prefixes, within-episode N/D/I pairing, episode-level independence,
  no-supplement targetless/missing-bin handling, all-or-nothing failed panels,
  separate three-model estimates, and no BR/regret/Nash/MFG claims.
- Did not create the T3A implementation ticket or change executable code.

#### Verification

Confirmed the ADR status line and accepted scope in the ADR and this ticket.
The previously recorded `535/535` full-suite result and `status: ok`
configuration check remain the applicable executable verification because this
acceptance changes design governance only.

#### Artifacts

`docs/adr/0022-token-conditional-continuation-oracle.md`.

#### Decisions and risks

Acceptance authorizes a separate T3A implementation ticket, not implementation
itself.  That ticket must freeze the bin schema, target-selection rule `S`,
episode count/minimum effective sample count, calibration/validation
namespaces, CRN key, maximum scheduler-call budget, and targetless/failed-panel
disposition before code or rollout execution.

#### Next

Create the independent T3A implementation ticket, then implement only the
observable-bin oracle; do not add automatic action selection or price feedback.
