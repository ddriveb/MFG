Type: implementation and experiment

Status: resolved

Blocked by: none

## Goal

Run the first population-level A/B diagnostic using the sealed T3B candidate:

- A: every Token uses the original NIIN online policy;
- B: every Token uses `ConditionalTokenCandidatePolicy` from the sealed T3B
  runtime candidate table.

This is a finite population policy comparison only.  It does not claim a best
response, regret, Nash equilibrium, MFG fixed point, price equilibrium, or
group-policy iteration.

## Frozen A/B contract

- Environment: `theta_token_load0p7_v1`, with the complete T3A physical
  protocol, slowdown `2.0`, hedge delay `2.0`, reservation parameters,
  dispatcher, and complete drain unchanged.
- A and B use the exact same immutable `EpisodeTrace` for each episode.  The
  candidate is loaded from the sealed T3B artifact and applied online to every
  Token in B; it is not applied as a tagged-only intervention.
- Namespace: `token-mfg-restoration:t3c:population-ab:v1`.
- Macro seed: `20260912`; episode count: `2048`; canonical episode order is
  `0..2047`; no retry, supplement, filtering, or result-driven stopping.
- Maximum scheduler calls: `2048 * 2 = 4096`, one complete run for A and one
  complete run for B per immutable trace.
- Population comparison metrics are frozen as follows:
  - overall and arrival-phase `D`/`F` latency distributions with count, mean,
    P95, and P99;
  - Replay token count/rate and Replay execution count;
  - requested/applied/launched Hedge counts and executor suppression;
  - executed primary/Hedge/Replay work, total executed work, and wasted work;
  - Reservation suppression/admission and timer suppression counts;
  - Protection Storm diagnostics from the existing attribution contract:
    maximum/mean Hedge launch rate, maximum peak-to-mean ratio, maximum
    sustained overload duration, and overloaded-bin totals/means.
- Phase latency means P95/P99 are pooled over Tokens by arrival phase. Count,
  work, and lifecycle fields report both pooled totals and episode means where
  applicable. `delta_B_minus_A` is descriptive only.
- Every episode must satisfy the existing online/scenario invariants.  A/B
  trace fingerprints, token identity/order, arrival/class stream, and all
  exogenous workload/fault fields must match exactly.
- Required fresh artifact run ID:
  `token-t3c-population-ab-20260908-r1/`.  It must be transactional and never
  overwrite an existing run.  No source, config, or old artifact is changed.

## Acceptance criteria

1. Red tests first cover same-trace A/B execution, all-Token online policy
   invocation, deterministic results, metric hand calculations, phase/D/F
   latency fields, work/lifecycle/storm fields, exact call accounting,
   candidate parent fingerprint, and artifact no-overwrite behavior.
2. The real run completes all 2,048 episode slots and exactly 4,096 calls,
   with zero retries and no failed episode.
3. The artifact contains manifest, metric contract, episode rows, and summary
   with source/parent fingerprints and all claim flags false.
4. The report explicitly states whether B is descriptively better or worse
   than A for each frozen metric, without upgrading that statement to a game
   or MFG result.
5. Focused tests, one full test suite, and one config check pass.

## Progress log

### Update: 2026-09-08 — Claimed population A/B diagnostic

Status: partial

#### Goal

Compare original NIIN against the T3B candidate applied to every Token on
paired immutable traces, before any policy iteration or price feedback.

#### Changed

- Created and claimed this single T3C implementation/experiment ticket.
- Added `src/mfg_hedge/token_population_ab.py` with paired same-trace A/B
  execution, all-Token online candidate application, pooled overall/D/F
  latency tails, lifecycle/work/reservation metrics, and Protection Storm
  diagnostics.
- Added `tests/test_token_population_ab.py` covering the frozen call budget,
  causal online execution, same-trace identity, deterministic output,
  diagnostics, claim boundary, and transactional artifact safety.
- Frozen the A/B namespace, macro seed, 2,048 episode slots, 4,096-call
  budget, and metric definitions above.  The 2,048 count is inherited from
  the T3B holdout allocation; the namespace and seed are independent.

#### Verification

- Read repository rules, T3B ticket/artifact contract, online engine, episode
  protocol, and attribution metric definitions.
- Real Red captured with
  `.venv/Scripts/python.exe -m unittest tests.test_token_population_ab -v`:
  the suite failed at import because `mfg_hedge.token_population_ab` did not
  yet exist.
- Focused implementation suite currently passes `7/7`.
- Real A/B command completed successfully:
  `.venv/Scripts/python.exe -m mfg_hedge.token_population_ab --project-root . --artifacts-root artifacts --run-id token-t3c-population-ab-20260908-r1 --validation-dir artifacts/token-t3a-oracle-20260907-r1-validation --occupancy-dir artifacts/token-t3a-oracle-20260907-r1-occupancy`.
- The run completed all `2048/2048` episodes and exactly `4096/4096`
  scheduler calls, with no failed episode and no retry.
- Pooled latency comparison (`B - A`) was:

  | cohort | A mean / P95 / P99 | B mean / P95 / P99 | B−A |
  | --- | --- | --- | --- |
  | overall | 6.8028086492 / 27.3719499292 / 36.7425670110 | 8.6042143021 / 33.5262131757 / 42.4501732110 | +1.8014056529 / +6.1542632465 / +5.7076061991 |
  | arrival D | 8.5180677177 / 28.5349147644 / 37.0379296315 | 10.8967511213 / 35.1420379801 / 42.5135206030 | +2.3786834036 / +6.6071232156 / +5.4755909715 |
  | arrival F | 21.5873488849 / 36.3865627568 / 43.9706015792 | 27.2224623509 / 42.1760017234 / 49.7581706502 | +5.6351134660 / +5.7894389667 / +5.7875690710 |

- Replay increased from `31,770` tokens / `0.0346010586` to `43,567` /
  `0.0474493019`; rate delta was `+0.0128482433`.
- Hedge launches decreased from `15,847` (`7.7377929688` per episode) to
  `4,019` (`1.9624023438` per episode). Reservation suppression decreased
  from `56,509` (`0.0615445773`) to zero.
- Mean total executed work changed from `449.8653716018` to
  `449.8567048947` per episode (`-0.0086667072`). Mean wasted work changed
  from `1.6811673618` to `1.6271120914` (`-0.0540552704`).
- Protection Storm diagnostics: maximum peak launch rate stayed `2.0`,
  maximum peak-to-mean ratio increased `128.0 -> 320.0`, maximum sustained
  overload duration changed `22.0 -> 21.0`, and mean overloaded-bin count
  changed `165.7885742188 -> 165.5815429688`.
- All A/B episode invariant checks passed: `83,968/83,968` per arm. Trace
  fingerprints and Token identity fingerprints matched for every pair.
- The T3B candidate policy payload matched the runtime candidate table used by
  the run in all retained rows, selected actions, schema, and parent
  fingerprint fields. The candidate remains `3 N + 6 I`, with no D.
- Full suite passed `572/572` with
  `.venv/Scripts/python.exe -m unittest discover -s tests -v`.
- Config gate passed with
  `.venv/Scripts/python.exe -m mfg_hedge check --config configs/v1_minimal.json`
  and returned `status: ok`.
- `git diff --check` passed.

#### Artifacts

- `artifacts/token-t3c-population-ab-20260908-r1/manifest.json`
- `artifacts/token-t3c-population-ab-20260908-r1/candidate_policy.json`
- `artifacts/token-t3c-population-ab-20260908-r1/metric_contract.json`
- `artifacts/token-t3c-population-ab-20260908-r1/summary.json`
- `artifacts/token-t3c-population-ab-20260908-r1/episode_rows.json`
- The T3B candidate artifact remains unchanged.

#### Decisions and risks

- This is a descriptive finite-population A/B diagnostic.  It is not a
  conditional continuation re-estimation and cannot establish MFG closure.
- The same trace is reused only within each A/B pair; no T3B episode or result
  row is reused as a T3C observation.
- The population result is descriptively worse for latency and Replay under B,
  despite lower Hedge activity and slightly lower wasted work. This is direct
  evidence that the fixed-NIIN conditional candidate is not self-consistent
  when installed by the whole population.
- This does not establish a best response, regret, Nash equilibrium, or MFG
  fixed point. No price feedback, group-policy iteration, or continuation
  re-estimation was run.
- No package version bump was made; the new runner remains internal.

#### Next

None for this ticket; any policy iteration or MFG work requires a separate
explicitly authorized ticket.

## Answer

The first true population A/B is complete. Applying the T3B candidate to all
Tokens is not an improvement over NIIN in this frozen environment: latency and
Replay worsen, while Hedge/suppression activity falls and total work remains
nearly unchanged. The result is a finite descriptive population diagnostic,
not a Nash or MFG result.
