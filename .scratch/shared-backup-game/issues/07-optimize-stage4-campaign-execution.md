# Optimize Stage 4 campaign execution layer

Type: implementation
Status: resolved
Blocked by: none

## Scope

Implement and verify only the semantics-preserving Stage 4 performance layer.
This ticket must not run Fit, Validation, or any formal campaign. It may not
change the shared Backup physics, policy observation boundary, scorer
mathematics, CRN/sample protocol, rule bank, solver stopping rules, call
budget, historical engines/configurations, or artifact claim boundary.

## Frozen performance contract

- Keep all 256 rules, N=8 Experts, frozen common paths and populations, three
  starts, 32-round maximum, and 921,088 planned-call upper bound.
- Keep every counterfactual as a complete fresh shared-physics rerun. Do not
  freeze baseline speed, queues, Replay, actions, completion, or budget state;
  all Experts and all copies remain in the event loop.
- Add immutable `PreparedTrace` input preparation with one-time validation,
  deterministic event/identity projections, and cached trace/fault
  fingerprints. It must contain no endogenous policy, queue, speed, budget,
  completion, Replay, winner, timer, scorer, or deviation state.
- Add exact tagged evaluation that retains only the tagged Expert's scorer
  sufficient data while physically evolving every Expert/copy/action/timer/
  Replay/running loser. Tagged and full reference scoring must agree exactly
  (or with an explicitly justified `abs_tol <= 1e-12`).
- Optimize observation scans with local canonical Expert indexes and optimize
  queue audits with an O(1) remaining-work ledger while retaining every
  QueueAuditEvent and capacity interval audit.
- Add a Windows spawn-safe process runner with task granularity
  `(deviating_expert, candidate_rule)`, deterministic scenario order, at most
  8 workers and 8 in-flight tasks, parent-only call reservation/settlement,
  no automatic retries, fixed parent merge order, and no worker file writes.
- Enforce/report 512 MiB worker RSS and 4 GiB parent working-set limits using
  real RSS measurement or fail explicitly when unsupported. A failed or
  over-limit task produces partial/failed output and no candidate.
- Use only Python standard library. No campaign artifact is authorized in
  this ticket.

## Required equivalence and performance checks

Add or extend focused tests for:

1. Reference full vs optimized full vs optimized tagged field-by-field
   equality across N/D/S/X, Replay burst, failure/completion/timer ties,
   simultaneous completion tie, queued/running losers, failure drain,
   survivor acceleration, and budget suppression.
2. Interleaved Expert/global Token IDs retain canonical local observation
   history ordering and causal information boundaries.
3. Incremental remaining-work ledger equals a scan reference at every audit
   event and preserves enqueue/executed/cancelled/discarded conservation.
4. PreparedTrace is immutable, reusable without contamination, and preserves
   fingerprints and results under different online policies.
5. Serial and 1-worker/2-worker process results have identical rows and fixed
   merge order; completion order does not affect output.
6. Parent call reservation/settlement is exact; budget 10 never dispatches an
   eleventh call; worker failure is recorded, not retried, and cannot produce
   a candidate; duplicate/missing results fail closed.
7. Windows spawn-safe top-level worker functions, real RSS fields, bounded
   worker lifecycle, and no residual workers.
8. Real N=8 timing probe uses the frozen fit namespace/seed/path/population,
   1,276 realized Tokens, c_B=.5, slowdown 2.0, delay 1.5, and NNNN/NSSN/XXXX.
   Warm up once, measure at least five times, report median/max, throughput,
   speedup, RSS, and the 921,088-call worst-case estimate. This is a timing
   gate only; do not start campaign calls.

## Required verification commands

`.venv\\Scripts\\python.exe -m unittest tests.test_shared_backup_performance -v`

`.venv\\Scripts\\python.exe -m unittest tests.test_campaign_execution -v`

`.venv\\Scripts\\python.exe -m unittest tests.test_shared_backup -v`

`.venv\\Scripts\\python.exe -m unittest tests.test_game_workload -v`

`.venv\\Scripts\\python.exe -m unittest tests.test_expert_game -v`

`.venv\\Scripts\\python.exe -m unittest tests.test_game_deviations -v`

`.venv\\Scripts\\python.exe -m unittest tests.test_game_solver -v`

`.venv\\Scripts\\python.exe -m unittest discover -s tests -v`

`.venv\\Scripts\\python.exe -m mfg_hedge check --config configs\\v1_minimal.json`

## Progress log

### Update: 2026-09-06 — Claim performance implementation slice

Status: partial

#### Goal

Implement the accepted ADR-0017 performance layer without running any Fit,
Validation, or formal campaign execution.

#### Changed

- Created and claimed this sole implementation ticket after blocking ticket
  06. No source, configuration, or campaign artifact was changed in this
  claim step.

#### Verification

- No implementation verification has run yet; the required Red tests and
  reference-equivalence harness are outstanding.

#### Artifacts

- This ticket only. No campaign artifact.

#### Decisions and risks

- ADR-0017 is accepted for performance-layer implementation, with its frozen
  semantics and no-campaign boundary unchanged.

#### Next

Write the failing performance equivalence tests first, then implement one
single-scheduler optimization at a time.

### Update: 2026-09-06 — Performance layer verified; campaign remains unstarted

Status: completed

#### Goal

Implement only the accepted ADR-0017 Stage 4 execution/performance layer.
Fit, Validation, formal campaign execution, candidate selection, and MFG work
remain out of scope and were not run.

#### Changed

- Added immutable `PreparedTrace` and `prepare_shared_trace`; it contains only
  validated exogenous trace projections, canonical local indexes, work arrays,
  event projection, fingerprints, and schema/count metadata. It contains no
  policy, budget, queue, speed, completion, Replay, winner, timer, scorer, or
  deviation state.
- Added exact optimized full mode with Expert-local observation indexes, an
  O(1) remaining-work ledger, fail-fast ledger underflow checks, and a retained
  per-event queue audit. The existing `simulate_shared_backup` remains the
  reference default path.
- Added exact tagged mode and `score_tagged_result`; all Experts, online action
  calls, copies, queues, timers, Replay, processor-sharing speed changes, and
  running losers still participate. Only the tagged Expert's materialized rows
  and compact global pool/invariant summary are returned.
- Added spawn-safe `campaign_execution.py` with top-level worker functions,
  fixed task grain `(deviating_expert, candidate_rule)`, scenario preparation
  in worker initialization, parent-only call reservation/settlement, no
  retries, deterministic merge order, canonical rule-bank validation, and
  compact scorer sufficient data. It does not choose a best rule or write
  files.
- Exported the isolated performance API and bumped the package consistently
  from `0.22.0` to `0.23.0` in `pyproject.toml` and `mfg_hedge.__version__`.

#### Verification

- True Red was recorded before implementation: the performance suite first
  failed to import the missing `score_tagged_result`; the batch suite first
  raised `ModuleNotFoundError` for the missing `campaign_execution` module.
  The first tagged scorer run then exposed and fixed a fault-fingerprint
  wrapper mismatch. No failure was hidden or converted to a skipped test.
- Focused suites passed with exact counts: performance `6`, campaign execution
  `7`, shared backup `14`, game workload `15`, expert game `9`, game
  deviations `10`, and game solver `10`.
- Full command `.venv\Scripts\python.exe -m unittest discover -s tests -v`
  passed `456/456`. Config command
  `.venv\Scripts\python.exe -m mfg_hedge check --config
  configs\v1_minimal.json` returned `status: ok` (with the pre-existing
  diagnostic that single-domain failure load is above target capacity).
- Reference versus optimized full was asserted as complete immutable result
  equality, covering Token/attempt/action identity, lifecycle times/work,
  winner/Replay/timer fields, pool intervals, queue audits/events, counters,
  drain, and trace/fault fingerprints. Dedicated equivalence cases include
  N/D/S/X, failure Replay, failure/completion/timer ties, simultaneous winner
  tie, queued loser, running loser, survivor acceleration, and budget
  suppression.
- Raw optimized and reused PreparedTrace runs were equal; PreparedTrace was
  frozen, fingerprint-stable, reusable without state contamination, and its
  interleaved Expert/global-ID observation histories matched reference order.
  Tagged scorer output matched the full-reference Expert projection exactly,
  and tagged runs called every Expert online while retaining global counters
  and pool integration.
- One-worker and two-worker rows were identical; reversing task submission
  order did not change rows or merge keys. Parent budget `2` reserved and
  completed exactly `2` one-call tasks and dispatched no third task. A worker
  exception settled one reserved call as failed without retry and returned no
  candidate; duplicate tasks, missing/duplicate/reordered canonical rule banks
  were rejected.
- Queue audit events all held the conservation equation at `1e-12`; terminal
  queue audits matched `enqueued - executed - cancelled - discarded`, and pool
  interval execution stayed at its capacity bound whenever constrained. The
  tagged pool summary integrated the same full-reference intervals.
- Windows RSS was measured with `GetProcessMemoryInfo`, not Python heap
  accounting: parent peak `64,913,408` bytes; worker peaks
  `32,821,248`–`33,841,152` bytes. These are below the `4 GiB` parent and
  `512 MiB` worker limits. The valid batch used `spawn`, eight workers, eight
  calls, and zero failed calls; no residual worker remained after shutdown.
- Frozen real-load probe (namespace `shared-backup-game:v1:stage4-fit`, macro
  seed `20260905`, common path `0`, population `0`, `N=8`, `1,276` Tokens,
  `c_B=.5`, slowdown `2.0`, delay `1.5`) used one warm-up and five measured
  runs per mode. Median/max seconds were:

  | rule | reference full | optimized raw full | prepared full | prepared tagged |
  |---|---:|---:|---:|---:|
  | NNNN | `0.5012425000022631 / 0.5116986999928486` | `0.3755918999959249 / 0.3849624000140466` | `0.36952649999875575 / 0.3699672000075225` | `0.2800223000231199 / 0.2834067000076175` |
  | NSSN | `0.4922091999906115 / 0.49794630001997575` | `0.3897220000217203 / 0.39355609999620356` | `0.37962160000461154 / 0.3833575999888126` | `0.29547519999323413 / 0.29787959999521263` |
  | XXXX | `0.5095436999981757 / 0.5190275000059046` | `0.39466510000056587 / 0.3966039000079036` | `0.3854713000182528 / 0.39043390000006184` | `0.3038101000129245 / 0.30827729997690767` |

  The eight-worker real-load batch completed `8/8/0` reserved/completed/
  failed calls in `1.8538537999847904` seconds, throughput
  `4.315334898612628` calls/second. Prepared-full median speedups versus
  reference were `1.3564453429022028x`, `1.2965784875903592x`, and
  `1.3218719525268103x`; NNNN prepared-tagged median speedup was
  `1.7900092241256438x`. At the frozen `921,088`-call upper bound, the observed
  batch throughput gives `213,445.31111504883` seconds, or
  `59.29036419862467` hours. This is a timing estimate only.

#### Artifacts

- Performance-only report: `.scratch/shared-backup-game/stage4-performance-report-20260906.md`.
- No Fit, Validation, candidate, qualification, holdout, campaign, or root
  `artifacts/` output was generated. The temporary spawn probe script was
  deleted after measurement.

#### Decisions and risks

- The optimized and tagged paths preserve the finite game's physics and
  causal observation boundary; tagged mode does not remove other Experts from
  physical evolution. Parallel execution does not select rules and does not
  change deterministic rows.
- The timing estimate is not a scientific experiment result and does not
  authorize campaign execution. No Nash candidate or MFG result exists.
- RSS measurement is real and supported on this Windows host. A first stdin
  probe demonstrated that Windows spawn requires a real guarded top-level
  script; the production runner itself uses top-level spawn-safe workers.

#### Next

Use the timing and memory report to make a separate campaign-launch
disposition in ticket 06; ticket 06 is open but remains unclaimed, and no
campaign should start in this session.

### Update: 2026-09-06 — Final source verification after PreparedScenario input support

Status: completed

#### Goal

Close the performance implementation only after the final source state passes
the required focused, full, and configuration checks.

#### Changed

- Accepted already prepared `PreparedScenario` rows directly in the batch
  preparation path; this is an input compatibility path only and does not
  change any scheduler, scorer, CRN, or process semantics.

#### Verification

- Re-ran `tests.test_campaign_execution`: `7/7` passed.
- Re-ran `tests.test_shared_backup_performance`: `6/6` passed.
- Re-ran `unittest discover -s tests -v`: `456/456` passed in `28.399s`.
- Re-ran `mfg_hedge check --config configs\v1_minimal.json`: `status: ok`;
  the only finding is the pre-existing transient-load diagnostic.
- No Fit, Validation, solver campaign, candidate selection, or MFG execution
  was started by these final checks.

#### Artifacts

None beyond the performance-only report already recorded above.

#### Decisions and risks

The ticket remains resolved because the final compatibility change preserved
all equivalence and budget checks. Ticket 06 is open and unclaimed; timing
alone does not authorize its campaign.

#### Next

None for ticket 07. Ticket 06 must receive a separate launch disposition.

## Answer

Resolved. The accepted ADR-0017 performance layer and its required
equivalence, deterministic batching, RSS, and timing-gate verification are
complete; the separately scoped campaign in ticket 06 was not run.
