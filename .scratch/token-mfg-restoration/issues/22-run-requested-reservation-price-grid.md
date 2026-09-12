# Run the requested-Reservation-price fixed grid

Type: bounded implementation and experiment

Status: resolved

Blocked by: none

## Goal

Test a price that reaches pre-projection protection demand without rerunning
the scheduler or changing the accepted admitted-reservation model.

## Acceptance criteria

1. A distinct model/evidence ID prevents confusion with all ADR-0021 models.
2. Price zero reproduces the ADR-0027 beta-matched grid cells.
3. D/I pay one requested unit even when their recorded applied action is N.
4. The 3x5 grid reuses the immutable 978-panel parent artifact and makes zero
   scheduler calls.
5. Artifact records parent hash, formulas, unsupported coverage, false claim
   flags, and deterministic cells.
6. Focused tests, full suite, and config gate pass.

## Progress log

### Update: 2026-09-08 — Claimed requested-price grid

Status: partial

#### Goal

Repair the feedback/charge-basis mismatch with an explicitly separate demand
toll before attempting another population loop.

#### Changed

- ADR-0028 and this ticket were created.

#### Verification

- ADR-0026 diagnostic completed with 608/608 tests passing.
- Its requested work rose 8,882 to 10,166 while price rose 0 to 1.7025.

#### Artifacts

- Parent: `artifacts/token-refined-fixed-grid-20260908-r1/`

#### Decisions and risks

- This does not mutate or alias `token_extended_reservation_v1`.
- The fixed grid is conditional re-scoring, not population feedback.

#### Next

Write Red tests, implement deterministic parent-panel re-scoring, and commit a
fresh zero-call artifact.

## Answer

The requested-reservation toll reaches denied demand and produces the intended
monotone response.  A bounded population-feedback implementation is now
justified; this retrospective grid itself is not that feedback run.

### Update: 2026-09-08 — Completed zero-call requested-price grid

Status: completed

#### Goal

Verify that pricing pre-projection requested units corrects ADR-0026's
feedback/charge mismatch before spending scheduler calls on another loop.

#### Changed

- Added `token_requested_price_grid.py` with the distinct model ID
  `token_runtime_requested_reservation_price_v1`.
- Reused the immutable ADR-0027 bin statistics and 978 physical panels for
  beta {1,2,4} x request price {0,0.5,1,2,4}.
- Added four focused tests covering exact price-zero reproduction, denied
  request billing, monotone demand response, and false claim boundaries.

#### Verification

- Real Red: four `ModuleNotFoundError` errors before implementation.
- Focused: 4/4 passed.
- Full suite: 612/612 passed in 177.632 seconds.
- Config check: exit 0, `status: ok`; the known transient-capacity warning is
  unchanged.
- Scheduler calls: exactly zero. Parent source and summary SHA-256 fingerprints
  match the on-disk files.
- At beta 4, request price 0/0.5/1/2/4 produced requested protection
  probabilities 0.7399/0.3732/0.2187/0.1846/0.1839. The corresponding
  one-Token counterfactual applied probabilities were
  0.2595/0.2197/0.1932/0.1841/0.1839.
- Conditional unpriced expected cost at beta 4 changed only from 14.2153 at
  price zero to 14.2374 at price one and 14.2425 at price four; the correct
  NIIN conditional cost in the parent bank is 15.6377.

#### Artifacts

- `artifacts/token-requested-reservation-price-grid-20260908-r1/manifest.json`
- `artifacts/token-requested-reservation-price-grid-20260908-r1/summary.json`

#### Decisions and risks

- The response plateaus near 0.184: those bins retain material protection value
  even under a high request toll. This is preferable to forcing all-N.
- `one_token_counterfactual_applied_hedge_probability` is explicitly not a
  population outcome; changed population demand must rerun full physics.
- This new model is not ADR-0021's admitted-reservation model and is not an
  alias for it. No public API was exported; version remains 0.25.0.

#### Next

Implement a 256-identity, four-round population loop using this requested-unit
toll, with the same 5,120-call ceiling and direct comparison to the completed
ADR-0026 execution-price loop.
