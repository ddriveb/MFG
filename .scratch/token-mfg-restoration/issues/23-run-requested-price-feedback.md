# Run bounded requested-price population feedback

Type: bounded implementation and experiment

Status: resolved

Blocked by: none

## Goal

Test whether matching the feedback variable and Token charge basis regulates
population requests under full queue/Reservation feedback.

## Acceptance criteria

1. Price is visible online and D/I target costs pay one requested unit even if
   Reservation applies N.
2. All other ADR-0026 parameters remain identical.
3. At most 5,120 calls; no retries or adaptive extension.
4. Output compares every round with the execution-price artifact and keeps all
   equilibrium claims false.
5. Focused/full tests and config gate pass.

## Progress log

### Update: 2026-09-08 — Claimed requested-price feedback

Status: partial

#### Goal

Run the full-physics confirmation justified by ADR-0028.

#### Changed

- ADR-0029 and this ticket were created.

#### Verification

- ADR-0028 completed with zero scheduler calls and 612/612 tests passing.

#### Artifacts

- `artifacts/token-requested-reservation-price-grid-20260908-r1/`

#### Decisions and risks

- The run is fixed at 256 identities and four rounds; no convergence claim.

#### Next

Add the isolated cost-basis switch, focused Red tests, and execute the run.

## Answer

The matched requested-demand price reduced excess requests after the expected
one-round policy-update lag. Four rounds were insufficient to leave the hard
quota saturation region, so this is directional evidence rather than a fixed
point or performance improvement claim.

### Update: 2026-09-08 — Completed requested-price feedback

Status: completed

#### Goal

Compare requested-unit pricing with ADR-0026 execution-work pricing under the
same bounded population feedback protocol.

#### Changed

- Added the isolated `requested_reserved_work` basis to the existing internal
  price runner while preserving its execution-work default.
- Added a distinct model/evidence identity and transactional requested-price
  artifact with per-round comparison to the ADR-0026 artifact.
- Added two focused tests: denied D/I request toll arithmetic and bounded
  requested-price runner identity.

#### Verification

- Focused price suite: 7/7 passed.
- Full suite: 614/614 passed in 174.178 seconds.
- Config gate: exit 0, `status: ok`; known transient warning unchanged.
- Formal run: 4,960/5,120 calls, four rounds, no physical failure or retry.
- Price path: 0 -> 0.5210 -> 1.0981 -> 1.6115, terminal next price 2.0955.
- Requested work: 8,882 -> 9,528 -> 8,794 -> 7,755. Relative to the
  execution-price controller this is 0, 0, -1,049, and -2,411 by round.
- Admitted work: 1,978 -> 2,035 -> 2,037 -> 2,034. It remains on the hard
  quota plateau; suppression fell to 6,757 in round 2 and 5,721 in round 3.
- Mean latency: 6.6058 -> 6.6055 -> 6.5929 -> 6.5875. P99: 27.3096 ->
  27.3593 -> 27.3414 -> 27.3240. These changes versus execution pricing are
  at most about 0.0076 and are descriptive only.
- Artifact source and comparison-parent SHA-256 fingerprints match.

#### Artifacts

- `artifacts/token-t4d-requested-price-feedback-exploratory-20260908-r1/manifest.json`
- `artifacts/token-t4d-requested-price-feedback-exploratory-20260908-r1/summary.json`

#### Decisions and risks

- The expected lag is explicit: rounds 0 and 1 match because both controllers
  start at price zero and use eta=0.2; the changed round-1 response first alters
  round-2 population actions.
- Demand regulation now has the right sign, but requested work is still 2.69x
  the 2,880-unit round capacity at round 3. Hence admitted work and latency
  remain nearly unchanged.
- No extra rounds were added after seeing the result. All BR/regret/Nash/MFG
  claim flags remain false; version remains 0.25.0 with no public export.

#### Next

Before a longer run, raise the controller step or solve a one-dimensional
request-price target from the fixed grid, then run a bounded paired population
A/B at that frozen price. Do not spend more rounds slowly walking through the
known quota-saturated region.
