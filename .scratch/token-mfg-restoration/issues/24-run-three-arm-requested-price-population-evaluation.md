# Run the frozen three-arm requested-price population evaluation

Type: implementation and bounded experiment

Status: resolved
Blocked by: none

## Goal

Implement and run one frozen finite-system, paired three-arm population
evaluation for the requested-Reservation-price Token policy.  The output is
descriptive finite-system performance evidence only.  It must not claim best
response, regret, Nash, or MFG.

## Frozen protocol

- Namespace: `token-mfg-restoration:three-arm-holdout:v1`.
- Macro seed: `20260918`.
- Holdout episode count: `1024`; every episode uses the same immutable trace
  for all three arms.
- Exactly three complete scheduler calls per episode: A, B, C.  Maximum and
  intended call count: `3072`.  No retry, supplement, adaptive extension, or
  zero-fill after an arm failure.  Complete drain is required.
- Physical environment: `theta_token_load0p7_v1`, with the frozen episode
  protocol and service/fault workload keys, slowdown `2.0`, hedge delay `2.0`,
  and the accepted Reservation contract.

### Arms

- A `No-Hedge`: every Token requests Normal.
- B `NIIN`: exact frozen rule `(early Regular=N, early Urgent=I, late
  Regular=I, late Urgent=N)`, with `late_after=50.0`; all other
  phase/Primary cases request Normal.  This is tested directly and must not
  be replaced by NIII.
- C `Requested-Price Candidate`: candidate table loaded from the sealed
  `artifacts/token-refined-fixed-grid-20260908-r1/summary.json` parent, with
  model `token_runtime_requested_reservation_price_v1`, requested reservation
  price `2.0`, `beta=4.0`, and refined observation ADR-0027.  On supported
  bins, use `Q_runtime(N)`, `Q_runtime(D)+2`, and `Q_runtime(I)+2` in a stable
  softmax.  Unsupported bins fall back exactly to NIIN.  The SHA-256
  policy-key draw and policy table fingerprint are frozen before execution.
  The public price is `2.0` for C policy observations only; it affects policy
  choice, not physics, faults, or the Reservation cap.

### Statistics and gates

- Per arm report overall and H/D/F/R latency sample count, mean, P50, P95,
  P99, CVaR95, and deadline misses; Replay; Hedge and Reservation counters;
  work/waste; and the existing Protection-Storm metrics.
- Paired comparisons are episode-clustered for `C-B`, `C-A`, and `B-A`.
  Report point difference, paired SE, 95% CI, and relative change.  Primary
  endpoints are pooled D/F CVaR95 and D/F deadline-miss rate.  Secondary
  endpoints are D/F P95/P99, Replay, overall mean/P99, H/R P99,
  total/wasted work, and storm peak.
- Use deterministic paired cluster bootstrap with `4096` replicates,
  namespace `token-mfg-restoration:three-arm-holdout:v1:paired-bootstrap`,
  seed `20260919`, and one shared resample-index library for all comparisons.
  Bootstrap resampling consumes no scheduler calls.
- The requested-price gate passes only when C versus NIIN reduces D/F CVaR95
  by at least 10%, its 95% CI supports improvement without crossing zero,
  D/F miss rate is no worse, Replay is no worse, H/R P99 deterioration is at
  most 2%, total work increase is at most 5%, storm peak is no higher, and all
  invariants pass.  Requested/capacity ratio, suppression, and quota
  saturation are reported but are not gate conditions.
- Any failed arm invalidates its entire episode.  No zero-fill is permitted;
  any physical/invariant failure is fail-closed and cannot produce a passing
  gate.  No formal claim is emitted.

## Artifact contract

Publish only to fresh run id `token-three-arm-holdout-20260908-r1` using the
repository's transactional non-overwrite writer.  At minimum include a
manifest, summary, episode rows, paired statistics, and candidate policy.
Record arm and source fingerprints, parent artifact SHA-256, full source
bundle/environment/config provenance, namespace/seed/counts, bootstrap
protocol, calls, and all claim flags false.  Do not write to `artifacts/`
until the bounded smoke and focused tests pass; never overwrite an existing
directory.

## Required verification

Before the formal 1024-episode run, add Red tests for exact NIIN, candidate
softmax and fallback, trace sharing, stable hash draw, three calls per
episode, full-group failure, paired bootstrap, gate boundary, false claims,
atomic/non-overwriting artifact creation, and the `3072` call ceiling.  Run a
bounded smoke and report its timing before starting the formal run.  Then run
the focused suite, the full suite, and config check.  Do not tune, change
sample counts, or launch any other experiment.

## Progress log

### Update: 2026-09-08 — Ticket claimed; protocol frozen

#### Goal

Implement and run the frozen three-arm requested-price population evaluation
as a finite-system performance gate only.

#### Changed

- Created and claimed this ticket as the only active ticket.
- Frozen the three arms, shared-trace CRN, 1024 episodes, 3072 scheduler-call
  ceiling, candidate parent artifact, requested-price softmax, bootstrap
  namespace/seed, paired endpoints, and no-claim boundary above.

#### Verification

- Confirmed ticket 16 is no longer claimed and no ticket was concurrently
  active.
- Confirmed `token-three-arm-holdout-20260908-r1` does not exist.
- Formal holdout has not started; no calls have been consumed by this ticket.
- Real Red: `tests.test_token_three_arm_evaluation` failed during collection
  with `ModuleNotFoundError: mfg_hedge.token_three_arm_evaluation`; `0` tests
  ran.

#### Artifacts

None.

#### Decisions and risks

- The evaluation compares finite-system population performance; it is not a
  best-response, regret, Nash, or MFG experiment.
- The parent candidate is sealed before the holdout and cannot be changed by
  holdout results.

#### Next

Write the required failing tests, implement the runner, execute a bounded
smoke, then launch the frozen holdout only after reporting its expected
duration.

### Update: 2026-09-08 — Red, implementation, and bounded smoke

#### Goal

Complete the frozen execution layer without changing the finite-system
physics, candidate source artifact, or claim boundary.

#### Changed

- Added `src/mfg_hedge/token_three_arm_evaluation.py` with exact No-Hedge and
  NIIN sources, requested-price supported-bin softmax, exact NIIN fallback,
  SHA-256 policy-key draws, shared-trace three-arm execution, existing
  attribution metric aggregation, paired episode-cluster bootstrap, gate
  evaluation, and transactional artifact output.
- Added the focused contract tests in
  `tests/test_token_three_arm_evaluation.py`.
- Added requested/capacity ratio, admitted-capacity ratio, and suppression
  rate to the existing Reservation audit output.  No new physical metric
  definition was introduced.

#### Verification

- Real Red: initial collection failed with
  `ModuleNotFoundError: mfg_hedge.token_three_arm_evaluation`; `0` tests ran.
- Focused suite after implementation: `10/10` passed.
- Bounded smoke: `4` episodes, `12` calls, `0` failed episodes, `0.849`
  seconds wall time; all three arms produced H/D/F/R metrics and paired
  statistics.  No formal artifact was written.
- The measured smoke rate implies roughly `3.6–4.0` seconds per 16-episode
  block for physical calls and an expected total around five minutes after
  the frozen bootstrap.  This is only a launch-time estimate.

#### Artifacts

None.  The formal run id remains unused.

#### Decisions and risks

- The candidate table is loaded from the sealed fixed-grid summary and is not
  selected from holdout data.
- Failed arms invalidate the entire episode without zero-fill; no retry is
  performed.

#### Next

Run the pre-launch full suite and config check.  If both pass, launch exactly
`1024` shared episodes with three complete calls each, then publish one fresh
non-overwriting artifact.

### Update: 2026-09-08 — Holdout completed; requested-price gate failed

#### Goal

Execute the frozen three-arm population evaluation and report finite-system
performance without converting it into a response, equilibrium, or MFG claim.

#### Changed

- Ran the sealed candidate from
  `artifacts/token-refined-fixed-grid-20260908-r1/summary.json` against the
  independent holdout namespace.  No holdout-driven policy selection or
  parameter change was made.
- Published the fresh non-overwriting artifact
  `artifacts/token-three-arm-holdout-20260908-r1` containing manifest,
  summary, episode rows, paired statistics, candidate policy, and metric
  contract.
- Kept package version unchanged: the new runner is an internal execution
  module and was not exported as a public API.

#### Verification

- The first formal launch command failed during shell quoting before Python
  started: `0` calls and no artifact were produced.  The corrected command
  then ran once; no scheduler call was retried.
- Formal holdout: `1024/1024` complete episodes, exactly `3072/3072`
  scheduler calls, `0` failed episodes, wall time `149.5s`.
- Pre-launch focused suite: `10/10` passed.  Pre-launch full suite and
  config check: `624/624` passed; `status: ok`.
- Post-artifact full suite and config check: `624/624` passed in `161.445s`;
  `status: ok`.
- All episodes used one shared trace across A/B/C.  The paired bootstrap used
  `4096` replicates, the frozen namespace/seed, one shared index-library
  fingerprint, and `0` scheduler calls.
- Artifact manifest records the parent summary SHA-256, current source-bundle
  fingerprint with `50` entries, frozen namespace/seed/counts, and all four
  claim flags false.  Transactional/non-overwrite behavior passed focused
  tests.

#### Results

- Arm A No-Hedge: `457925` tokens; overall mean latency `8.8090657966`,
  overall P99 `43.1739319612`, Replay rate `0.0484948409`, total work
  `458025.6143203`, wasted work `607.9991502`, storm peak `0`.
- Arm B exact NIIN: overall mean `6.6550285942`, overall P99
  `36.2010474763`, Replay rate `0.0338112136`, Hedge requests/launches
  `7854/7854`, Reservation suppression `99471`, total work
  `458819.3748736`, wasted work `1643.4189202`, storm peak `2`.
- Arm C requested-price candidate: overall mean `6.5590678123`, overall P99
  `35.9154977093`, Replay rate `0.0334050336`, Hedge requests/launches
  `8162/8162`, Reservation suppression `73995`, total work
  `458878.8001882`, wasted work `1738.8790801`, storm peak `2`.
- C minus B pooled D/F CVaR95: point `-0.2657118972`, relative change
  `-0.9873%`, paired SE `0.0474627957`, bootstrap 95% CI
  `[-0.3586924494, -0.1746198710]`; this improves in direction but misses
  the required `10%` reduction.
- C minus B D/F miss-rate change `-0.0039699999`, Replay-rate change
  `-0.0004122541`, H/R P99 relative change `-0.7919%`, total-work relative
  change `+0.01295%`, wasted-work relative change `+5.8086%`, and storm-peak
  change `+0.09765625`.
- The requested-price gate is `gate_pass: false`: the D/F CVaR reduction is
  below `10%`, and the storm peak is higher than NIIN.  C-A and B-A paired
  rows are retained in the artifact; no result-based policy adjustment was
  made.

#### Artifacts

- [Three-arm holdout artifact](../../artifacts/token-three-arm-holdout-20260908-r1)
  (fresh run id, never overwritten).
- No Nash, MFG, BR, regret, or training artifact was generated.

#### Decisions and risks

- This is a finite-system, paired population performance result only.  The
  candidate does not qualify as a best response or equilibrium.
- The candidate's reduced latency versus NIIN is small relative to the frozen
  gate and comes with a higher Protection-Storm peak and higher wasted-work
  relative change; it must not be interpreted as a successful price design.
- The artifact is valid descriptive evidence despite the failed performance
  gate; no additional samples or tuning are authorized by this ticket.

#### Next

Leave the candidate gate failed and preserve the artifact.  Any later policy
  or price change requires a separately frozen protocol and ticket; do not
  reuse this holdout to retune the policy or claim Token-MFG closure.
