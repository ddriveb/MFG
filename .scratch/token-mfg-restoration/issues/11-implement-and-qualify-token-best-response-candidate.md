Type: implementation and experiment

Status: resolved

Blocked by: none

## Goal

Construct the T3B conditional best-response candidate from the sealed T3A
validation artifact and run the independent 2,048-episode holdout
qualification.  The output is descriptive finite-token evidence only:
`conditional_best_response_candidate_with_holdout_diagnostics`.

## Scope

- Load and validate the existing T3A occupancy and validation artifacts without
  reinterpreting or overwriting them.
- Select the runtime-model minimum action per retained bin with deterministic
  `N < D < I` tie breaking and `N` fallback for unknown bins.
- Add a causal, stateless `ConditionalTokenCandidatePolicy` whose action is
  selected from the current `TokenObservation` only.
- Run the independent T3B qualification using one target-selection run plus
  baseline/N/D/I paired physical reruns per selected target, with fixed NIIN
  for all non-target Tokens and the candidate applied only to the tagged Token.
- Emit a fresh transactional artifact under
  `artifacts/token-t3b-qualification-20260907-r1/`.
- Preserve paired episode costs, paired differences, per-bin floors, and
  descriptive one-sided 95% diagnostics; never emit BR, regret, Nash, MFG,
  price, or policy-iteration claims.

## Frozen qualification contract

- Namespace: `token-mfg-restoration:t3b:qualification:v1`.
- Macro seed: `20260911`.
- Episode count: `2048`; one future-blind `S_v2` target at most per episode.
- Physical environment: `theta_token_load0p7_v1`; population policy:
  `pi_NIIN_v1`; schema, reservation, slowdown, delay, dispatcher, and
  complete-drain semantics are inherited from the sealed T3A provenance.
- Only the nine T3A-retained bins are aggregated.  Episodes targeting other
  bins still consume their scheduled selection/physical calls and are recorded
  as out of scope; missing targets and failed panels are never replaced.
- A complete panel is selection plus baseline/N/D/I physical runs using the
  same target prefix and CRN.  Any physical branch failure fails the whole
  panel.  The maximum is `2048 * (1 + 4) = 10240` scheduler calls; no retry,
  supplement, bin merge, or sample-driven early stop is permitted.
- Each `(model_id, bin_id)` needs 32 complete panels.  Report means, SE,
  effective n, descriptive minimum, candidate-minus-alternative paired
  diagnostics, and `mean + NormalDist().inv_cdf(.95) * SE` only when the floor
  is met.  This is not a simultaneous Nash bound and has no epsilon threshold.
- Required artifact files are `manifest.json`, `candidate_policy.json`,
  `qualification_summary.json`, `qualification_rows.json`, and
  `episode_audit.json`.  Writes are transactional and never overwrite an
  existing run directory.

## Acceptance criteria

1. Red tests first cover malformed/fingerprint-mismatched parent artifacts,
   deterministic argmin/tie handling, unknown-bin fallback, fresh episode
   policy state, causal selection, CRN/prefix equality, paired statistics,
   insufficient bins, false claim flags, exact five-call accounting, and
   artifact no-overwrite behavior.
2. The real qualification uses exactly the frozen 2,048 episode slots and
   does not reuse T3A episodes or calls.  Actual calls never exceed 10,240.
3. The candidate policy is applied only to the tagged Token; other Tokens are
   still invoked through their fresh online NIIN policy functions.
4. The result label is exactly
   `conditional_best_response_candidate_with_holdout_diagnostics`, and all
   BR/regret/Nash/MFG claim flags are false.
5. Focused tests, one full test suite, and one config check pass.
6. The ticket is resolved only after the fresh artifact and all verification
   evidence are recorded in this ticket.

## Progress log

### Update: 2026-09-07 — Claimed T3B candidate and holdout qualification

Status: partial

#### Goal

Implement the bounded T3B candidate construction and independent holdout
diagnostics without entering any downstream equilibrium or MFG stage.

#### Changed

- Created and claimed this single implementation/experiment ticket.
- Added `token_best_response.py` for sealed T3A parent validation, runtime
  minimum-action selection, deterministic tie-breaking, and causal N fallback.
- Added `token_t3b_execution.py` for the tagged Token holdout, paired runtime
  scorer, exact call accounting, diagnostics, and transactional artifact
  writing.
- Added 13 focused tests covering parent integrity, policy causality, paired
  statistics, CRN/prefix identity, call accounting, and no-overwrite safety.

#### Verification

- Read the repository rules, T3A ticket/provenance, and the existing T1/T2/T3A
  execution and test modules before implementation.
- Real Red captured with
  `.venv/Scripts/python.exe -m unittest tests.test_token_best_response tests.test_token_t3b_execution -v`:
  both suites failed at import because the new T3B modules do not yet exist
  (`ModuleNotFoundError` for `mfg_hedge.token_best_response` and
  `mfg_hedge.token_t3b_execution`).
- Focused implementation suite currently passes 13/13 tests.
- Real qualification command completed successfully:
  `.venv/Scripts/python.exe -m mfg_hedge.token_t3b_execution --project-root . --artifacts-root artifacts --run-id token-t3b-qualification-20260907-r1 --validation-dir artifacts/token-t3a-oracle-20260907-r1-validation --occupancy-dir artifacts/token-t3a-oracle-20260907-r1-occupancy`.
- The fresh holdout has 2,048 episode slots, 916 complete retained-bin panels,
  269 missing targets, 863 out-of-scope-bin panels, zero failed panels, and
  9,164 actual calls out of the 10,240 maximum.

#### Artifacts

- `artifacts/token-t3b-qualification-20260907-r1/` was committed
  transactionally with the five required JSON files.  The two sealed T3A
  parent directories remain unchanged.

#### Decisions and risks

- Parent T3A artifacts are treated as sealed inputs; the current source bundle
  is recorded separately for T3B and is not used to reinterpret parent rows.
- This ticket does not authorize price feedback, best-response iteration,
  regret, Nash, or MFG conclusions.

#### Next

Run the single required full test suite and the single config check, then
record the exact per-bin diagnostics and resolve this ticket if both gates
pass.

### Update: 2026-09-07 — Completed T3B candidate qualification

Status: completed

#### Goal

Construct the finite conditional candidate from the sealed T3A runtime rows
and run the independent 2,048-episode holdout without producing BR, regret,
Nash, MFG, price-feedback, or group-policy claims.

#### Changed

- Added `src/mfg_hedge/token_best_response.py` with strict T3A parent
  artifact validation, immutable runtime estimate rows, deterministic
  `N < D < I` argmin selection, and causal `N` fallback for unknown bins.
- Added `src/mfg_hedge/token_t3b_execution.py` with the frozen T3B namespace,
  macro seed, tagged-Token complete physical reruns, paired runtime scoring,
  per-bin diagnostics, exact call accounting, and transactional artifact
  emission.
- Added 13 focused tests.  T3B modules remain internal and were not exported
  through `mfg_hedge.__init__`; version remains `0.24.0`.

#### Verification

- Real Red: the pre-implementation focused command failed with two import
  errors for the not-yet-created T3B modules.  After implementation the
  focused suite passed `13/13`.
- Qualification completed with exactly `2,048` episode slots and `9,164`
  actual scheduler calls of the `10,240` maximum:
  `916` complete retained-bin panels, `269` `missing_target`, `863`
  out-of-scope-bin panels, and `0` failed panels.  No retry, supplement, or
  bin merge occurred.
- All nine retained bins met the 32-panel floor; per-bin effective n,
  construction candidate, holdout descriptive minimum, and diagnostic
  candidate upper 95% value were:

  | bin | n | candidate | holdout min | candidate upper95 |
  | --- | ---: | :---: | :---: | ---: |
  | `D/R/0/[100,200)/[0,1.5)/[1,3)/[2.8125,+inf)` | 138 | N | N | 2.6989275408855695 |
  | `D/R/0/[100,200)/[0,1.5)/[3,+inf)/[2.8125,+inf)` | 101 | I | I | 4.2871411679276035 |
  | `D/R/0/[100,200)/[1.5,3)/[1,3)/[2.8125,+inf)` | 112 | N | I | 3.331441315982233 |
  | `D/R/0/[100,200)/[1.5,3)/[3,+inf)/[2.8125,+inf)` | 93 | I | I | 3.849349924139453 |
  | `D/R/0/[100,200)/[10,+inf)/[3,+inf)/[0,1)` | 115 | N | N | 8.838158486958797 |
  | `D/R/0/[100,200)/[10,+inf)/[3,+inf)/[1,2.8125)` | 83 | I | I | 4.110086429029265 |
  | `D/R/0/[100,200)/[3,10)/[1,3)/[2.8125,+inf)` | 113 | I | I | 3.083807183652082 |
  | `D/R/0/[100,200)/[3,10)/[3,+inf)/[1,2.8125)` | 67 | I | I | 4.857102587444235 |
  | `D/R/0/[100,200)/[3,10)/[3,+inf)/[2.8125,+inf)` | 94 | I | I | 3.9117307806315607 |

- The construction candidate remained the holdout descriptive minimum in
  `8/9` bins; bin 3 is retained as a diagnostic disagreement, not hidden or
  reselected.  Candidate action counts are `N=3, D=0, I=6`.
- Paired candidate-minus-alternative diagnostics were retained for every bin;
  for bin 3 they were `N-I = 0.015684328021020728` with SE
  `0.23553242929288817`, and for all other bins the stored paired rows are in
  `qualification_rows.json`/`qualification_summary.json`.
- Parent validation fingerprint is
  `08a149f7db31aa20e037d3d6e8c672f72d30e298086a915e8977ee8f17349922`;
  parent occupancy fingerprint is
  `439d7c5dfdefc14120fc4fd2ed780e65a753150e56f99b1a24f04621a05f5ef9`.
  The current T3B source-bundle fingerprint is recorded in the manifest as
  `e8752f90e8089da1a7335cf45485b7565a6572839135b1196762c50f3a71ffdc`.
- Full suite passed `565/565` with
  `.venv/Scripts/python.exe -m unittest discover -s tests -v`.
- Config gate passed with
  `.venv/Scripts/python.exe -m mfg_hedge check --config configs/v1_minimal.json`
  and returned `status: ok`.
- `git diff --check` passed.  The five artifact files were read back from the
  fresh run directory and no existing T3A artifact was modified.

#### Artifacts

- `artifacts/token-t3b-qualification-20260907-r1/manifest.json`
- `artifacts/token-t3b-qualification-20260907-r1/candidate_policy.json`
- `artifacts/token-t3b-qualification-20260907-r1/qualification_summary.json`
- `artifacts/token-t3b-qualification-20260907-r1/qualification_rows.json`
- `artifacts/token-t3b-qualification-20260907-r1/episode_audit.json`

#### Decisions and risks

- The result label is exactly
  `conditional_best_response_candidate_with_holdout_diagnostics`; all
  best-response, regret, Nash, MFG, and simultaneous-bound flags are false.
- The one bin-level holdout disagreement is an explicit diagnostic and does
  not upgrade or alter the sealed construction candidate.
- The holdout used fixed NIIN for non-target Tokens, applied the candidate only
  to the tagged Token's requested-action branches, and preserved complete
  online physics and paired CRN reruns.
- No formal experiment conclusion, Nash candidate, continuation predictor,
  price feedback, group-policy iteration, or MFG result was produced.

#### Next

None for this ticket.  Any downstream action selection or MFG work requires a
separate explicitly authorized ticket and must not reinterpret this artifact.

## Answer

T3B implementation and the independent qualification completed under the
frozen protocol.  The artifact is a finite conditional best-response
candidate with holdout diagnostics only; it is not a verified best response,
Nash equilibrium, or MFG result.
