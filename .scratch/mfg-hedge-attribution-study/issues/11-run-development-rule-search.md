# Run development-only 81-rule reference search

Type: implementation and development experiment
Status: resolved
Blocked by: none

## Scope

Implement ADR-0013's deterministic enumerator/reporting boundary, verify it,
then execute the one frozen 64-episode fit search and commit a fresh artifact.
Do not generate qualification/holdout data or implement/relabel an MFG solver.

## Acceptance criteria

1. Exactly 81 unique rules in N<D<I lexical order, NNNN first and included.
2. Every rule consumes the same 64 immutable fit traces and actual applied
   actions; fingerprints are checked before selection.
3. Objective completeness and H/R mean/P95 plus total-work filters implement
   ADR-0013 exactly; tie-break is J, work, lexical rule.
4. Invalid inputs, mixed namespaces/protocols, missing rules/results and any
   nonfinite/failed invariant abort without a scientific artifact.
5. Results include all rule rows, selected and NNNN summaries, D/F tail ratios,
   reservation counts, safety reasons, parameters and provenance.
6. Focused Red/Green, full suite and environment check pass. Execute a complete
   64x81 development run, retain negative or positive outcome without tuning.
7. One fresh transactional artifact; rerun cannot overwrite it.

## Progress log

### Update: 2026-09-05 — Run the frozen 81-rule development experiment

Status: completed

#### Goal

Implement the reproducible exact-queue rule search and execute the frozen
64-episode development fit requested by the user, without qualification,
holdout access, parameter tuning or an MFG claim.

#### Changed

- Added Accepted ADR-0013 with the exact fit identity, 81-rule order, safety
  filters, selection rule, artifact and claim boundary. Full ADR-0011 remains
  Proposed; the historical attribution tickets remain paused.
- Added `configs/v1_transient_development.json`: load 0.45 self-contained base
  configuration; no existing configuration changed.
- Added `src/mfg_hedge/transient_search.py`: exact lexical bank, episode-bank
  identity validation, all-rule physical evaluation, pooled H/R mean and P95,
  total-work filters, deterministic J/work/lexical selection, admission and CRN
  provenance, frozen 64-episode entry point and transactional artifact builder.
- Added `search-transient-rules` CLI, public exports and reproducibility docs.
  Package 0.16.0 -> 0.17.0 for new public experiment APIs; old behavior unchanged.
- Added `tests/test_transient_search.py`: seven tests. A second Red/Green cycle
  tightened artifact validation so duplicate/missing/reordered 81-rule rows are
  rejected before any directory is written.
- Added `development-rule-search-report.md` with numerical results and limits.

#### Verification

- Initial real Red:
  `.\.venv\Scripts\python.exe -m unittest tests.test_transient_search -v`
  -> `Ran 1 test ... FAILED (errors=1)`, missing `mfg_hedge.transient_search`.
- Boundary-hardening Red:
  `... tests.test_transient_search.ArtifactTests.test_incomplete_or_duplicate_rule_bank_writes_nothing -v`
  -> `FAILED (failures=1)`: duplicate NNNN/missing IIII was incorrectly accepted.
  Fixed with exact complete ordered bank validation.
- Focused final:
  `.\.venv\Scripts\python.exe -m unittest tests.test_transient_search -v`
  -> `Ran 7 tests in 0.489s ... OK`.
- Full required regression:
  `.\.venv\Scripts\python.exe -m unittest discover -s .\tests -v`
  -> `Ran 373 tests in 19.677s ... OK` (366 prior + 7 new).
- `.\.venv\Scripts\python.exe -m mfg_hedge check --config .\configs\v1_minimal.json`
  -> exit 0, status ok, Python 3.10.11; known historical headroom finding retained.
- New config check before experiment -> exit 0, no audit findings,
  single-domain failure load 0.9.
- Actual frozen CLI run, run-id
  `transient-control-81-fit-m20260905-e64-20260905` -> exit 0. All 64 episodes,
  18,709 Tokens and 81 rules completed; 81 feasible; selected NIIN.
- Full computation repeated in memory: all 81 rule rows and all 64 fingerprints
  equal the committed files exactly; selected rule and comparison equal.
- Existing-directory write probe after the experiment raised FileExistsError;
  three JSON files remained valid and unchanged. SHA-256:
  manifest C9C5C33289EFFF5652AEDE9DC7B1641CB2A4A6E2F46DE28B699E2C342767A4A7;
  rows 8C76DAA2A985F28570497A53F1C44106440E6D6682F4F0119DB5E27E84161694;
  summary 5453F11B04BC4A85F728FCCCD87A2AB5224549C14A93D7A592AFC2F4DBF9CCEB.

#### Artifacts

- `artifacts/transient-control-81-fit-m20260905-e64-20260905/manifest.json`
- Same directory: `rule_results.json` (all rows/fingerprints) and `summary.json`.
- `.scratch/mfg-hedge-attribution-study/development-rule-search-report.md`.

#### Decisions and risks

- Development result: NIIN means early-Regular N, early-Urgent I,
  late-Regular I, late-Urgent N. Versus NNNN: J 14.887604 vs 20.578092
  (ratio .723469); D CVaR95 10.432193 vs 16.373484 (ratio .637139);
  F CVaR95 10.229315 vs 13.130886 (ratio .779027); total work ratio 1.012448.
  H mean/P95 ratios were exactly 1; R mean/P95 .972934/.968298.
- NIIN requested 1,401 Hedges, applied 451, and suppressed 950 under the frozen
  pooled reservation. Engine requested count equals applied count. Non-winner
  work increased 38.563332 -> 342.960114 and is charged, not omitted.
- All 81 candidates passed safety filters, so those filters were nonbinding at
  this low cap. Selection was by development J; this does not validate the gates.
- NIIN's surprising late-Urgent N must not be interpreted causally from fit data.
  Pooled FIFO and sampling variation can drive it. The weights/budget/gates were
  not changed after seeing the result.
- These 64 episodes are permanently burned fit data. This is neither independent
  superiority evidence nor MFG evidence; no qualification/holdout namespace was
  generated. A complete negative qualification remains an acceptable next result.

#### Next

Freeze a separate qualification-only ticket and evaluate NIIN and NNNN on 128
independently keyed episodes once, without reranking any of the 81 rules.

## Answer

All acceptance criteria passed and the complete development experiment is
committed. NIIN is the sample-best feasible rule in the frozen 81-rule class;
its promising fit result remains unqualified development evidence.

## Comments

- User instruction “开始实验吧” activates only this previously recommended
  development comparison. It does not authorize qualification/holdout unblinding.
