# Implement and run the T3A statistical oracle

Type: implementation and experiment

Status: resolved

Blocked by: none

## Goal

Implement the accepted ADR-0023 occupancy/validation protocol as one bounded
vertical slice, then execute it without adding BR, regret, price feedback,
Nash, or MFG claims.

## Scope

- Add the named `token_t3a_source_bundle_v1` execution module.
- Freeze and validate `theta_token_load0p7_v1`, `pi_NIIN_v1`, `S_v2`, the
  source bundle, model/quote provenance, namespaces, seeds, call limits, and
  deterministic serialization.
- Run 512 occupancy episodes, retain only bins with `n_occ >= 16`, and derive
  the single validation allocation before observing costs.
- Reuse one selection plus baseline/N/D/I physical runs per validation episode
  across the three pure scorers.
- Write immutable occupancy and validation artifacts; never overwrite an old
  run and never publish a partial validation result.

## Acceptance criteria

1. Focused red-green tests cover anchor selection, occupancy screening,
   allocation, shared physical runs, model separation, source provenance,
   budget enforcement, deterministic output, and fail-closed behavior.
2. Occupancy uses exactly 512 scheduler calls and validation never exceeds
   10,240 calls; combined calls never exceed 10,752.
3. Every retained `(model_id, bin_id)` has either at least 32 complete panels
   and finite N/D/I estimates or explicit `insufficient` status with null
   numeric fields.
4. Output claims for best response, regret, Nash and MFG are all false.
5. The focused suite, full test suite, and config gate pass.
6. The real occupancy and validation artifacts are generated under fresh run
   IDs, or the run fails closed with the exact reason recorded here.

## Progress log

### Update: 2026-09-07 — Started combined T3A implementation and execution

Status: partial

#### Goal

Remove the document-only bottleneck and deliver the accepted statistical
oracle through real experiment artifacts.

#### Changed

- ADR-0023 marked Accepted and its stale design-only execution wording fixed.
- This single implementation/experiment ticket was created and claimed.

#### Verification

- Read `AGENTS.md`, `CONTEXT.md`, the issue/update rules, ADR-0022/0023, the
  Token restoration spec, ticket 09, and the current T1/T2/T3A modules/tests.

#### Artifacts

- None yet.

#### Decisions and risks

- Occupancy and validation are implemented in one ticket, but occupancy is
  sealed and verified before validation begins.
- No downstream action selection or MFG claim is in scope.

#### Next

Write failing focused tests for the frozen ADR-0023 execution contract.

### Update: 2026-09-07 — Implemented and executed the statistical oracle

Status: completed

#### Goal

Finish the accepted ADR-0023 implementation and produce the first statistically
screened Token conditional-cost artifacts without another design-only pause.

#### Changed

- Added `src/mfg_hedge/token_t3a_execution.py` with the frozen four-anchor
  `S_v2`, environment/policy/source-bundle fingerprints, 512-episode occupancy
  screen, deterministic validation allocation, shared physical N/D/I runs,
  three independent pure scorers, strict call accounting, and transactional
  artifacts.
- Added `tests/test_token_t3a_execution.py` with eight focused red-green tests.
- ADR-0023 is Accepted and its stale design-only execution wording is removed.
- Kept the module internal and package version at `0.24.0`; no public API was
  added.

#### Verification

- Real Red: `.venv/Scripts/python.exe -m unittest tests.test_token_t3a_execution -v`
  failed with `ModuleNotFoundError: mfg_hedge.token_t3a_execution`; the second
  Red failed because `_run_validation` did not yet exist.
- Focused final: the same command ran 8 tests, all `OK`.
- Affected final: Token T1/T2/T3A plus attribution suites ran 71 tests, all
  `OK`.
- Full final: `.venv/Scripts/python.exe -m unittest discover -s tests -v`
  ran 552 tests in 160.357 seconds, all `OK`.
- Config gate: `.venv/Scripts/python.exe -m mfg_hedge check --config
  configs/v1_minimal.json` exited 0 with `status: ok`; the known
  `single_domain_failure_load=1.4 > 0.9` transient warning remains.
- Real occupancy: 512 calls, 441 selected targets, 71 `missing_target`, 44
  observed bins, 9 retained bins, validation allocation 1,928 episodes.
- Real validation: 1,928 selection calls plus 1,683 complete four-run physical
  panels = 8,660 calls; 245 `missing_target`; no failed panel. Combined calls
  were 9,172, below the 10,752 cap.
- All 81 estimate rows completed; effective independent panel counts range from
  53 to 134, so no retained `(model_id, bin_id)` cell was insufficient.

#### Artifacts

- `artifacts/token-t3a-oracle-20260907-r1-occupancy/`
- `artifacts/token-t3a-oracle-20260907-r1-validation/`
- Source bundle fingerprint:
  `b71b485fa164a6edb1a5f9c2eb1d1c767672c48ae4cf95cd58eabd23267183ac`.
- Occupancy fingerprint:
  `439d7c5dfdefc14120fc4fd2ed780e65a753150e56f99b1a24f04621a05f5ef9`.
- Validation fingerprint:
  `08a149f7db31aa20e037d3d6e8c672f72d30e298086a915e8977ee8f17349922`.

#### Decisions and risks

- The three cost models share exactly the same baseline/N/D/I physical results;
  model IDs and quote bases remain separate in every estimate row.
- The data are conditional-cost evidence only. Descriptive action minima are
  not promoted to BR, regret, Nash, or MFG claims.
- Only Regular-token bins passed the pre-registered occupancy floor. Urgent
  bins remain unsupported under this fixed 512-episode protocol and were not
  topped up or merged.

#### Next

Design and implement the separate Token best-response rule using these frozen
conditional-cost estimates and an independent qualification split.

## Answer

ADR-0023 is implemented and the authorized occupancy/validation experiment
completed successfully. All retained cells passed their sample floor; the
artifacts above are the canonical result.
