# Implement and run the bounded Token price-feedback pilot

Type: narrow implementation and experiment

Status: resolved

Blocked by: none

## Goal

Add the ADR-0026 round-level price feedback after the NIIN and observation
repairs, then compare it on a bounded 256-identity, four-round CRN profile.

## Scope

- Add immutable per-round execution quote support to the existing T4c round
  evaluator while preserving the zero-price default.
- Record requested/admitted/suppressed Reservation work and capacity.
- Add an isolated price-feedback runner using initial price 0, step 0.25,
  `beta=4.0`, `eta=0.2`, 256 identities, four rounds, and panel floor 4.
- Write a fresh artifact at
  `artifacts/token-t4d-price-feedback-exploratory-20260908-r1/`.
- Do not implement the within-episode pacing controller, training, predictor,
  formal MFG qualification, or any Nash/MFG claim.

## Acceptance criteria

1. A real Red proves quote propagation, feedback arithmetic, and claim/status
   boundaries before implementation.
2. Price zero reproduces the existing T4c Q and policy path.
3. Every branch in a round uses the same quote; the next quote uses only that
   completed round's pre-projection requested work and declared capacity.
4. The artifact records price, demand pressure, admitted/suppressed work,
   price residual, policy/occupancy residuals, and physics metrics per round.
5. Focused tests, full suite, and config gate pass; the bounded run stays below
   5,120 scheduler calls and is labeled exploratory.

## Progress log

### Update: 2026-09-08 — Claimed price-feedback pilot

Status: partial

#### Goal

Implement and run the smallest price-mediated comparison after the user stopped
the undisclosed multi-hour T4c profile.

#### Changed

- ADR-0026 and this bounded implementation/experiment ticket were created.

#### Verification

- Existing exploratory T4c artifact was reviewed: final weighted policy
  residual 0.085973, mean latency 6.826988, P99 27.862127, Replay rate
  0.0346651, and 7,046 Hedge launches.

#### Artifacts

- `docs/adr/0026-token-mfg-round-price-feedback.md`

#### Decisions and risks

- The first feedback loop is between response rounds.  It is deliberately not
  presented as the still-unimplemented within-episode pacing controller.

#### Next

Write failing focused tests, implement the isolated runner, and execute the
bounded CRN comparison.

### Update: 2026-09-08 — Stopped after invalid baseline discovery

Status: blocked

#### Goal

Prevent a price experiment from continuing on top of an incorrectly encoded
NIIN fallback.

#### Changed

- Stopped the running 1,024-by-six price pilot and removed its uncompleted
  implementation and tests from the production tree.
- ADR-0026 returned to Proposed/deferred status because the user explicitly
  requested baseline and observation/sampling corrections before pricing.
- This ticket is blocked by the exact NIIN repair and the later
  observation/sampling revision.

#### Verification

- Both Python processes were stopped.
- `artifacts/token-t4d-price-feedback-exploratory-20260908-r1/` does not exist;
  transactional output left no partial result.
- Direct source comparison confirmed T3A's historical rule is `(N,I,I,N)`,
  while T4c's fallback condition implements `(N,I,I,I)`.

#### Artifacts

- None. The interrupted price run produced no artifact.

#### Decisions and risks

- No price result will be interpreted or retained.  Pricing remains a later,
  separately identifiable mechanism change.

#### Next

Resolve ticket 20 before changing bins, sampling, temperature, or price.

### Update: 2026-09-08 — Resumed after corrected fixed-grid evidence

Status: partial

#### Goal

Run the smallest real population/price feedback loop after removing the NIIN,
queue-direction, target-selection, and requested/applied interpretation defects.

#### Changed

- ADR-0026 was accepted for a 256-identity, four-round bounded diagnostic.
- The runner must use ADR-0027 refined bins and random thresholds.

#### Verification

- Ticket 21 completed with 603/603 tests passing.
- Fixed-grid price 0 to 1 reduced expected applied protection monotonically;
  at beta 4 the change was 0.2595 to 0.1951.

#### Artifacts

- `artifacts/token-refined-fixed-grid-20260908-r1/`

#### Decisions and risks

- The maximum is 5,120 full calls, not the deferred 30,720-call profile.
- This remains an exploratory feedback controller, not a price equilibrium.

#### Next

Write focused Red tests, implement quote propagation and the isolated runner,
then execute the four-round profile.

## Answer

The bounded feedback runner and experiment completed. Price propagation is
correct, but the feedback/charge-basis mismatch prevented demand regulation;
the result is negative exploratory evidence, not a fixed point.

### Update: 2026-09-08 — Completed bounded price-feedback diagnostic

Status: completed

#### Goal

Test whether the round-level requested-demand controller regulates protection
after the exact NIIN and refined-observation repairs.

#### Changed

- Added `token_price_feedback.py`: exact-NIIN refined population policy,
  coupled action draws, four-round response update, pre-projection demand
  feedback, physics metrics, strict call accounting, and transactional output.
- Added optional non-negative `public_price` propagation through the online
  engine; the default remains zero and historical behavior is unchanged.
- Added five focused tests for quote propagation, feedback arithmetic, exact
  NIIN, call bounds, and false equilibrium claims.

#### Verification

- Real Red: five errors (`token_price_feedback` missing and
  `simulate_episode_online` rejecting `public_price`).
- Focused: 5/5 passed; affected suites: 86/86 passed.
- Full suite: 608/608 passed in 165.220 seconds.
- Config gate: exit 0, `status: ok`; the known transient single-domain warning
  remains unchanged.
- Formal run: 256 identities x four rounds, 4,960/5,120 calls, no physical
  failure. Prices were 0, 0.5210, 1.0981, and 1.7025, with terminal next price
  2.3350.
- Requested Reservation work increased 8,882 -> 9,528 -> 9,843 -> 10,166;
  admitted work stayed at 1,978 -> 2,035 -> 2,038 -> 2,034; suppression
  increased 6,904 -> 8,132. The accounting identity held exactly.
- Mean latency was 6.6058, 6.6055, 6.5910, 6.5880; P99 was 27.3096, 27.3593,
  27.3339, 27.3282. Hedge launches were 1,978, 2,035, 2,037, 2,033.
- Undamped response L1 declined 1.4094 -> 0.6383, while the actual eta=0.2
  update L1 declined 0.2819 -> 0.1277; neither is convergence.

#### Artifacts

- `artifacts/token-t4d-price-feedback-exploratory-20260908-r1/manifest.json`
- `artifacts/token-t4d-price-feedback-exploratory-20260908-r1/summary.json`

#### Decisions and risks

- The runner is correctly wired, but the controller is not self-consistent:
  denied requested work raises the signal while paying zero execution price.
  More rounds would mostly raise a price that cannot reach this demand.
- The small latency change is descriptive and not a paired performance claim.
  All BR/regret/Nash/MFG flags remain false.
- Version remains 0.25.0 because the new runner is internal and the added
  online-engine argument is backward-compatible with a zero default.

#### Next

Use the already modeled reservation-price basis
(`token_extended_reservation_v1`) to charge eligible requested units before
projection, then test it on a bounded fixed grid before another feedback loop.
