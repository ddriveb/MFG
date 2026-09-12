# Run the refined observation and fixed-grid pilot

Type: bounded implementation and experiment

Status: resolved

Blocked by: none

## Goal

Implement ADR-0027 and obtain a fast, interpretable fixed price-temperature
diagnostic before revisiting dynamic feedback.

## Acceptance criteria

1. Red tests cover separate Primary/Backup load bins, deterministic future-blind
   target thresholds, exact NIIN use, grid arithmetic, and false claim flags.
2. One physical panel bank is reused for all nine grid cells; no cell reruns
   the event simulator.
3. Calls do not exceed 5,120; missing targets are not supplemented; supported
   bins require at least eight complete panels.
4. Artifact records schema/selector/source/identity fingerprints, raw panel
   rows, per-grid summaries, unsupported occupancy, and all claim flags false.
5. Focused tests, full suite, and config gate pass.

## Progress log

### Update: 2026-09-08 — Claimed refined fixed-grid pilot

Status: partial

#### Goal

Remove the known observation and target-selection confounds before pricing.

#### Changed

- ADR-0027 and this implementation/experiment ticket were created.

#### Verification

- Ticket 20 completed the exact NIIN repair with 598/598 tests passing.

#### Artifacts

- `docs/adr/0027-refined-token-observation-and-fixed-grid-pilot.md`

#### Decisions and risks

- The nine grid cells reuse one physical bank and remain descriptive; they do
  not imply a population fixed point.

#### Next

Write focused Red tests and implement the 5,120-call runner.

### Update: 2026-09-08 — Completed refined fixed-grid pilot

Status: completed

#### Goal

Run one bounded conditional-cost diagnostic after repairing the exact NIIN
fallback, without introducing dynamic price feedback.

#### Changed

- Added `token_refined_grid.py` with separate Primary/Backup live-load bins,
  four D-age strata, deterministic future-blind random thresholds, one shared
  physical N/D/I panel bank, and a retrospective 3x3 price-temperature grid.
- Added five focused tests.  A result-audit correction additionally split
  requested protection probability from expected applied protection
  probability; the latter accounts for Reservation projection.
- Added transactional four-file output and immutable provenance.  No public API
  was exported; package version remains 0.25.0 after the preceding NIIN fix.

#### Verification

- Initial Red: all five focused tests failed with `ModuleNotFoundError`.
- Requested/applied audit Red: the focused arithmetic test failed with
  `KeyError: requested_hedge_probability` before the fields were split.
- Focused: 5/5 passed.
- Full suite after the final requested/applied accounting correction: 603/603
  passed in 150.520 seconds.
- Config check: exit 0, `status: ok`; the known single-domain transient-load
  warning remains unchanged.
- Formal run: 1,024 episodes, 4,936/5,120 calls, 978 complete panels, 46
  missing targets, zero failed episodes.  Thirty-one supported bins cover
  870/978 = 88.9571% of complete panels.
- The 3x3 grid reused those 978 panels and made no per-cell simulator calls.
  Requested/applied protection probabilities at beta 4 were 0.7399/0.2595 at
  price 0 and 0.6755/0.1951 at price 1.  The NIIN conditional unpriced cost was
  15.6377 versus 14.2153--14.2836 for the nine descriptive mixtures.

#### Artifacts

- `artifacts/token-refined-fixed-grid-20260908-r1/manifest.json`
- `artifacts/token-refined-fixed-grid-20260908-r1/summary.json`
- `artifacts/token-refined-fixed-grid-20260908-r1/panel_rows.json`
- `artifacts/token-refined-fixed-grid-20260908-r1/grid_cells.json`

#### Decisions and risks

- This is conditional, supported-bin evidence under a fixed NIIN population,
  not a population A/B result.  Unsupported bins comprise 11.0429% of complete
  panels and are retained explicitly.
- Runtime work pricing changes actual applied protection, but the tested
  price range is weak: at beta 4, price 1 reduces expected applied protection
  by only 0.0644.  Requested protection remains much higher because exhausted
  Reservation states project D/I to N and therefore make those requests
  physically indistinguishable in the panel costs.
- `response_l1` remains large (1.2362--1.3649); the reported eta-0.2 quantity
  is only a hypothetical update step, not evidence of convergence.
- All BR, regret, Nash, and MFG claim flags remain false.

#### Next

Use this evidence to revise the price-feedback design so it prices reservation
demand/opportunity cost separately from executed incremental work, then run a
small corrected-NIIN population A/B before any long fixed-point campaign.

## Answer

The refined fixed-grid diagnostic completed within budget and isolated the
requested-versus-applied protection gap.  It supports a narrowly revised price
feedback experiment, but it does not establish a best response or equilibrium.
