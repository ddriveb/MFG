# Three-round finite reliability-aware routing evaluation

Type: experiment
Status: resolved
Blocked by: none

## Scope

Run the three-round finite-system evaluation authorized by ADR-0036 over the
accepted ADR-0035 engine. This ticket owns only the experiment runner,
development selection/diagnosis, one permitted Round-2 revision or robustness
confirmation, independent holdout, paired cluster statistics, analysis
artifacts, and final report. It does not alter the research object or add
Hedge, prices, placement optimization, training, cross-Expert capacity, or an
MFG solver.

## Frozen protocol

The complete protocol is in ADR-0036. In particular: 24-call maximum timing
preflight; hard cap 60,000; Round-1 namespace/seed
`reliability-aware-routing:v1:development:r1`/`20260910`, 256 episodes and
14,336 calls; Round-2 fresh namespace/seed
`...:development:r2`/`20260911`, at most 14,336 calls; and Round-3 fresh
namespace/seed `reliability-aware-routing:v1:holdout:r1`/`20260912`, 1024
episodes and 24,576 calls. Episode is the cluster unit and the holdout is
never used for selection.

## Acceptance criteria

- Red tests cover call arithmetic, preflight isolation, trace/CRN identity,
  candidate selection guardrails/tie-breaks, R2 branch freezing, max-t
  bootstrap determinism, fail-closed incomplete panels, and transactional
  artifacts.
- All baseline and candidate rows use complete drain and identical immutable
  exogenous traces per episode.
- R1, R2, and R3 each have unique artifacts and analysis Markdown; no old
  artifact is overwritten.
- The final report gives exact calls, failures, runtime, peak RSS if
  available, tests, config check, candidate status, estimates, paired SE/CI,
  and claim boundary.
- Formal development/holdout is not silently expanded, reweighted, or
  reinterpreted based on observed results.

## Progress log

### Update: 2026-09-10 — Three-round evaluation ticket opened

Status: partial

#### Goal

Freeze and execute the three-round finite routing evaluation without changing
the accepted physical routing object.

#### Changed

- Created and claimed ticket 04.
- Added and accepted ADR-0036 with the three-round protocol, branch rules,
  paired max-t statistics, 60,000-call cap, and claim boundary.
- No production code, scheduler call, or experiment artifact has been changed
  or generated in this ticket update.

#### Verification

- Confirmed tickets 01–03 are resolved and no other ticket is claimed in this
  feature.
- Protocol arithmetic: `24 + 14,336 + 14,336 + 24,576 = 53,272 <= 60,000`.

#### Artifacts

- `docs/adr/0036-three-round-reliability-aware-routing-evaluation.md`
- `.scratch/reliability-aware-multi-replica-routing/issues/04-three-round-routing-evaluation.md`

#### Decisions and risks

Round 2 is branch-controlled by Round 1 only. Round 3 remains independent and
cannot be used to tune a candidate. A no-candidate result is a valid terminal
scientific outcome, not a reason to relax a guardrail.

#### Next

Add the experiment runner contract tests, then implement the smallest
deterministic runner before the timing preflight.

### Update: 2026-09-10 — Three-round evaluation completed

Status: completed

#### Goal

Execute the frozen R1 development selection, R2 robustness confirmation, and
independent R3 holdout for finite reliability-aware multi-Replica routing,
without changing the physical routing object or making MFG/equilibrium claims.

#### Changed

- Added the isolated internal runner
  `src/mfg_hedge/reliability_aware_routing_experiment.py` and its focused
  contract tests. The only post-Red correction was to make paired-statistic
  rows accept the production nested `metrics` shape while retaining the
  metric-only helper shape used by the deterministic test.
- Ran R1 with all five fixed baselines and nine frozen risk-aware-JSQ
  configurations. The frozen development candidate is
  `risk_aware_jsq_w100_g4_6` (`history_window=100`, `gamma=(4,6)`).
- Ran R2 with the candidate frozen across all four robustness cells. It
  qualified for the holdout gate and was not retuned.
- Ran R3 on 1,024 fresh episodes per scenario with the candidate and five
  fixed baselines, followed by the fixed 4,096-replicate paired episode
  cluster max-t calculation.
- No production routing semantics, CRN stream, workload, policy definition,
  or historical artifact was changed by the campaign.

#### Verification

- Real Red: the first focused run failed with
  `ModuleNotFoundError: No module named
  'mfg_hedge.reliability_aware_routing_experiment'` before the runner existed.
  After implementation and the narrow row-shape correction,
  `tests.test_reliability_aware_routing_experiment`: `4/4` passed.
- Affected routing/experiment suites:
  `.venv\Scripts\python.exe -m unittest tests.test_reliability_aware_routing tests.test_reliability_aware_routing_lazarus tests.test_reliability_aware_routing_experiment -v`:
  `17/17` passed.
- Full suite:
  `.venv\Scripts\python.exe -m unittest discover -s tests -v`:
  `730/730` passed in `229.306s`.
- Config check:
  `.venv\Scripts\python.exe -m mfg_hedge check --config configs/v1_minimal.json`:
  exit 0, `status: ok`.
- Timing preflight: `24` isolated calls, `3.8570803862373357` calls/s,
  projected `3.8347946192379347` hours for the declared maximum; formal
  identity consumption was false.
- R1: `14,336/14,336` calls, `0` failures, `2,867.908659799956s`;
  terminal status `development_candidate`.
- R2: `6,144/6,144` calls, `0` failures, `1,438.2495463000378s`;
  terminal status `candidate_qualified_for_holdout`.
- R3: `24,576/24,576` calls, `0` failures, `5,359.447802299983s`;
  terminal status `holdout_not_improving`.
- Formal scheduler calls were `45,056`; including preflight, `45,080`.
  Bootstrap used `4,096` resamples and added no scheduler calls. The runner
  reported `peak_rss_bytes: null`; RSS was not fabricated from Python heap
  measurements.
- R3 candidate overall means were: mean `2.3050143614`, P95 `5.4665005348`,
  P99 `6.9960704229`, CVaR95 `6.3382658145`, urgent miss `0.3714326312`,
  replay rate `0.0224931471`, executed work `226.4372805448`, lost work
  `3.0884842529`, and drain duration `4.0847435306`.
- The primary paired candidate-minus-LOEW overall-CVaR95 point was
  `+0.0481962483`, paired SE `0.0200346670`, simultaneous 95% CI
  `[-0.0016118716, +0.0980043683]`; its relative point was `+0.7662%`.
  The candidate-minus-JSQ point was `-0.0929311312`, with simultaneous CI
  `[-0.1428161136, -0.0430461488]`.
- All holdout arms had zero summed
  `completed_without_winner`, `down_starts`, `duplicate_live_attempts`, and
  `future_work_observations`. Replica audit, trace fingerprints, complete
  drain, and all metric rows are retained in the holdout artifact.

#### Artifacts

- `artifacts/reliability-aware-routing-evaluation-20260910/reliability-aware-routing-development-r1-20260910/`
- `artifacts/reliability-aware-routing-evaluation-20260910/reliability-aware-routing-development-r2-20260911/`
- `artifacts/reliability-aware-routing-evaluation-20260910/reliability-aware-routing-holdout-r1-20260912/`
- `.scratch/reliability-aware-multi-replica-routing/three-round-evaluation-report-20260910.md`

Each round has `summary.json`, `paired_statistics.json`,
`episode_rows.jsonl`, `manifest.json`, `protocol.json`, and `analysis.md`.
All artifact directories were new and committed transactionally; no prior
artifact was overwritten.

#### Decisions and risks

R1 produced a development candidate and R2 passed its frozen robustness gate,
but independent R3 did not establish improvement over LOEW: the primary
candidate-minus-LOEW CVaR95 interval includes zero and points slightly worse.
The result is therefore `holdout_not_improving`, not a validated candidate or
equilibrium. The routing engine is single-copy, so Hedge/Protection-Storm
outputs are structurally zero/not applicable here. RSS was unavailable and is
explicitly reported as unavailable. No MFG, Nash, optimality, or solver claim
was made.

#### Next

None for this ticket; any follow-up algorithm design requires a new protocol
and ticket.
