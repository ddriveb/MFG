Type: implementation and bounded diagnostic

Status: resolved

Blocked by: none

## Goal

Implement the Proposed ADR-0025 T4c support-weighted zero-price Soft-BR
diagnostic without changing the T4b artifact or claiming equilibrium.

## Frozen scope

- 4,096 immutable episode identities, deterministic on-demand trace
  reconstruction, and no retained full trace library.
- `beta=1.0`, `eta=0.2`, maximum 20 rounds, the T4 physical protocol, nine
  retained bins, N/D/I, full online reruns, complete drain, and paired panel
  statistics.
- Per `(bin, round)`: zero occupancy is `inactive`; positive occupancy with
  fewer than 32 complete panels is `unresolved`; at least 32 complete panels
  is `active`.
- Only active bins update Soft-BR.  Unresolved bins fail closed; inactive bins
  do not block.  Residuals include current-occupancy-weighted policy L1,
  maximum active-bin policy L1, and occupancy L1.
- T4c call ceiling is `20,480` per round and `409,600` total.  No retry,
  supplement, merge, price feedback, predictor, BR claim, Nash claim, or MFG
  claim.

## Acceptance criteria

1. Red tests cover identity-only library immutability, deterministic
   reconstruction, support classification, active-only updates, weighted and
   maximum residuals, call accounting, and non-overwriting artifacts.
2. The implementation keeps T4b behavior and artifact unchanged.
3. A bounded physical smoke proves same identity fingerprints across rounds,
   fresh endogenous state, complete online counterfactual panels, and exact
   call accounting.
4. If a bounded T4c run is executed, its output is a fresh artifact and
   reports `statistics_insufficient`, `not_converged_20`, or
   `soft_fixed_point` only; all claim flags remain false.
5. Focused tests, full suite, and config check pass.

## Progress log

### Update: 2026-09-08 — Claimed T4c support-weighted diagnostic

Status: partial

#### Goal

Freeze and implement the support-aware T4c continuation of the honest T4b
diagnostic while preserving all finite-physical semantics.

#### Changed

- Added Proposed ADR-0025 for the 4,096 identity-library, on-demand
  reconstruction, support classes, and weighted residual gate.
- Claimed this ticket; T4b ticket 14 remains resolved and ADR-0024 remains
  Proposed.
- No production code or T4c artifact has been added yet.

#### Verification

- Read `AGENTS.md`, `CONTEXT.md`, issue/update conventions, ADR-0024, ticket
  14, the T4b implementation, and its focused tests.
- Red implementation tests are the next action; no T4c long run has started.

#### Artifacts

None.

#### Decisions and risks

- Positive occupancy below the panel floor is explicitly `unresolved`, never
  silently `inactive`.
- This ticket does not accept ADR-0024, add price feedback, or produce an
  equilibrium claim.

#### Next

Write the failing T4c tests before implementing the module.

### Update: 2026-09-08 — T4c implementation and bounded verification

Status: completed

#### Goal

Implement the support-weighted zero-price T4c diagnostic with identity-only
episode storage, deterministic on-demand reconstruction, active/inactive/
unresolved support gates, and no equilibrium claim.

#### Changed

- Added `src/mfg_hedge/token_mfg_t4c_supported_soft_response.py`.
- Added immutable `T4CIdentityLibrary` with 4,096-slot capacity and a
  canonical identity fingerprint; full workload traces are reconstructed per
  round and are not retained in the library.
- Added immutable T4c policy state.  Only active bins are updated; existing
  policy probabilities for inactive bins are preserved.
- Added support classification: zero occupancy is `inactive`, positive
  occupancy below the panel floor is `unresolved`, and panel-complete bins
  are `active`.
- Added support-weighted policy L1 residual, maximum active-bin residual,
  occupancy residual, hard-BR repeat diagnostics, exact call constants, and
  transactional artifact entry points.
- Added Proposed [ADR-0025](D:/project/mfg_hedge_v1/docs/adr/0025-token-mfg-t4c-supported-soft-response.md).
- T4b source, behavior, artifact, ADR-0024, and package version `0.24.0`
  were not changed.

#### Verification

- Real Red: `.venv\\Scripts\\python.exe -m unittest
  tests.test_token_mfg_t4c_supported_soft_response -v` initially ran `8/8`
  errors, each a real `ModuleNotFoundError` before the T4c module existed.
- Focused Green: the final focused command ran `9/9` passed.
- Bounded physical smoke: two rounds reused the same reconstructed identity
  fingerprint, executed complete online baseline/N/D/I panels when a target
  was selected, and satisfied reserved/attempted call accounting.
- Full suite after the final test change:
  `.venv\\Scripts\\python.exe -m unittest discover -s tests -v` —
  `594/594` passed in `174.722` seconds.
- Config check:
  `.venv\\Scripts\\python.exe -m mfg_hedge check --config
  configs\\v1_minimal.json` — `status: ok`; the existing 1.4 transient
  single-domain load warning remains diagnostic.

#### Artifacts

- No T4c formal 4,096-episode artifact was generated.
- No existing T4b artifact was overwritten.

#### Decisions and risks

- Positive support below 32 panels remains fail-closed as `unresolved`; only
  zero-support bins are non-blocking `inactive`.
- The implementation is ready for a separately authorized bounded/full T4c
  diagnostic, but this update is not a statistical result.
- ADR-0024 remains Proposed, ADR-0025 remains Proposed, and all
  best-response, regret, Nash, and MFG claim flags remain false.
- No price feedback, predictor, or solver was added.

#### Next

If desired, run the separately authorized 4,096-identity T4c diagnostic under
ADR-0025 and write a fresh artifact; do not interpret this implementation
validation as a fixed point.

## Answer

Status: resolved

T4c support-weighted execution is implemented and verified.  The code now
follows current population support without silently discarding positive but
undersampled bins, while retaining the zero-price Soft-BR and complete online
physical rerun semantics.

#### Artifacts

None.  This ticket produced implementation validation only; no formal T4c
statistical artifact was generated.

#### Decisions and risks

- Package version remains `0.24.0` because the new module is internal.
- No equilibrium or MFG conclusion is licensed.

#### Next

Run the full T4c diagnostic only as a separate execution decision under
Proposed ADR-0025.
