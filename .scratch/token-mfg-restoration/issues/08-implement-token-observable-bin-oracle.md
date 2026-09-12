# Implement Token observable-bin conditional cost oracle (T3A)

Type: implementation
Status: resolved
Blocked by: none

## Scope

Implement only the accepted ADR-0022 observable-bin conditional cost oracle.
The implementation produces action-conditioned Token costs for the three
ADR-0021 models and the claim label
`observable_bin_conditional_cost_oracle`.  It does not choose an action,
calculate BR/regret/Nash/MFG quantities, add price feedback, or run a formal
pilot.

## Frozen implementation protocol

These values are frozen before production code and tests are written.  A
future change requires a new design decision or a separately recorded protocol
revision; it must not be selected from rollout results.

### `bin_schema_v1`

The bin payload is an ordered tuple of exactly these fields:

1. `phase`: categorical `H`, `D`, `F`, or `R`.
2. `token_class`: categorical `R` or `U`.
3. `primary_replica`: categorical integer `0` or `1`.
4. `arrival_time`: half-open intervals `[0,100)`, `[100,200)`, `[200,220)`,
   `[220,320)`.  Values outside `[0,320)` are invalid for this protocol.
5. `phase_age`: half-open intervals `[0,1.5)`, `[1.5,3)`, `[3,10)`,
   `[10,+inf)`.
6. `queue_load`: the local queued-plus-running attempt count, with half-open
   intervals `[0,1)`, `[1,3)`, `[3,+inf)`.
7. `reservation_balance`: half-open intervals `[0,1)`, `[1,2.8125)`,
   `[2.8125,+inf)`.

All numeric intervals are left-closed/right-open; the final interval is
left-closed and unbounded above.  `phase`, `token_class`, and
`primary_replica` use exact enum/int values, not string coercion.  `token_id`,
exact queue order, running-attempt identities, exact observed-history strings,
exact balance, exact public price, true work, and all future data are not bin
coordinates.  They remain outside the statistical key and may only appear in
episode-local audit records under the T1 information boundary.  A non-finite,
negative, out-of-domain, ambiguous, or malformed field is `missing_bin`.
The canonical field order above, endpoint rules, and schema ID
`bin_schema_v1` participate in the schema fingerprint.

### Target-selection rule `S_v1`

Scan arrivals in nondecreasing `(arrival_time, token_id)` order using the
causal online observation only.  Select the first Token satisfying:

```text
phase == D and primary_replica == 0
```

The selector is evaluated at that Token's arrival before reading any later
arrival, fault, work, completion, policy action, or outcome.  It selects at
most one target per episode.  If no eligible Token occurs, record
`missing_target`, consume the episode, and do not supplement it.  The selector
fingerprint is `target_selection_v1:first_degraded_primary_A`.

### Episode splits and sample floor

This ticket is implementation validation only; it does not run a bounded
oracle pilot.  The frozen protocol library sizes are:

- calibration: 16 independent episodes;
- validation: 16 independent episodes;
- minimum complete paired panels per `(S_v1, bin_id, model_id)`: 2.

These counts are protocol values for the implementation-facing runner and its
tests, not a claim that a statistical pilot has been completed.  A cell below
the floor is `insufficient`, with null mean/SE and no bin merge or sample
expansion.

### Namespaces, macro seeds, and CRN keys

- calibration namespace: `token-mfg-restoration:t3a:calibration:v1`;
- validation namespace: `token-mfg-restoration:t3a:validation:v1`;
- calibration macro seed: `20260907`;
- validation macro seed: `20260908`;
- episode key: `t3a:{split}:v1:{macro_seed}:episode:{episode_index}`;
- exogenous stream key: `episode_identity + existing component stream key`;
- policy key: the unchanged T2 rule `token:{base_seed}:{token_id}`.

All three model panels use the frozen zero external quote (`0.0`) with the
model-specific ADR-0021 price basis.  No public-price update or endogenous
price feedback is part of T3A.

Episode index is the only episode draw counter.  The target action, candidate
action, model ID, target outcome, and worker/order are never included in an
exogenous or private-policy random key.  N/D/I branches reuse the same
immutable episode trace and policy-key rule; other Tokens are freshly called
online in every branch.

### Scheduler-call budget

The maximum is **512 scheduler calls** for one complete frozen 32-episode
implementation-validation library.  The accounting is reserved before each
invocation and settled after it:

- one causal selector pass per episode: at most 32 calls;
- three cost-model panels per selected episode, each with baseline plus N/D/I:
  at most `32 * 3 * 4 = 384` calls;
- total maximum: `416`, leaving 96 calls as validation headroom;
- no dispatch is allowed when the reservation would exceed 512;
- failures consume their reserved call and are never retried.

The parent/orchestrator owns the counter.  A failed panel is not salvaged from
successful branches and cannot trigger a replacement episode.

### Status disposition

- no target: `missing_target`, consumes the episode, no panel, no replacement;
- invalid bin input: `missing_bin`, consumes the episode, no panel, no
  replacement;
- any failed/non-finite/identity-mismatched/incompletely drained N/D/I branch:
  entire episode/model panel is `failed`, no action row contributes a cost,
  no retry;
- fewer than two complete panels in a `(S_v1, bin_id, model_id)` cell:
  `insufficient`, mean/SE are null, no merge or expansion;
- complete panel: contribute all three action-conditioned payoffs, with the
  episode as the independent statistical unit.

### Frozen execution disposition

`execution_mode = implementation_validation_only`.  No calibration pilot,
validation pilot, campaign, action selection, price feedback, or experiment
artifact is authorized by this ticket.  Tests may use smaller synthetic
episode fixtures and injected failures to verify the frozen contract.

## Acceptance criteria

- Add an immutable `bin_schema_v1` implementation with strict validation and
  deterministic fingerprinting.
- Add a causal `S_v1` selector that receives only online observations and
  selects at most one target per episode.
- Generate episodes from episode index/namespace/protocol, then build each
  episode's own target observation and bin; never match cross-episode prefixes.
- Run fresh complete N/D/I branches for the selected target, preserving T2
  policy factories, Reservation, CRN, Replay, timer, failure, loser, and drain
  semantics.
- Keep all three cost models separate and aggregate means/SE over complete
  episode panels only.
- Enforce the frozen failure, missing, insufficient, no-retry, and 512-call
  budget rules.
- Keep output limited to `observable_bin_conditional_cost_oracle` and false
  BR/regret/Nash/MFG claim flags.
- Preserve historical APIs and behavior; do not export a public API or bump
  the package version unless explicitly required by an API decision.

## Progress log

### Update: 2026-09-07 — Claim T3A implementation and freeze protocol

Status: partial

#### Goal

Create and claim the only implementation ticket for accepted ADR-0022, freeze
all execution semantics before code, and choose implementation validation only.

#### Changed

Created and claimed this ticket.  Frozen `bin_schema_v1`, `S_v1`, 16/16 split
episode counts, minimum two complete panels per bin/model, split namespaces and
macro seeds, CRN key format, 512-call maximum, all missing/failure/insufficient
dispositions, and `implementation_validation_only`.  No production code or
pilot has started.

#### Verification

Confirmed ADR-0022 is `Accepted (confirmed on 2026-09-07)`, ticket 07 is
resolved, and no other ticket is claimed.  Confirmed this ticket freezes
protocol values before implementation and preserves the accepted no-BR,
no-regret, no-Nash, no-MFG and no-price-feedback boundary.

#### Artifacts

This ticket only.

#### Decisions and risks

The 16/16 values and 512-call ceiling are implementation-validation protocol
values, not completed pilot evidence.  A future pilot must be separately
authorized and may not reinterpret these tests as statistical results.

#### Next

Write the failing T3A oracle tests, then implement the smallest compatible
internal module with red-green-refactor.

### Update: 2026-09-07 — T3A Red captured

Status: partial

#### Goal

Write the first T3A regression tests before production code and capture the
required import-level Red.

#### Changed

Added `tests/test_token_continuation.py` with tests for the frozen factory
signature, `bin_schema_v1` boundaries, `S_v1`, missing targets, cross-episode
prefix freedom, separate cost models, all-or-nothing panel failure,
insufficient bins, and strict call-budget disposition.

#### Verification

`.venv\Scripts\python.exe -m unittest tests.test_token_continuation -v`
failed as the intended Red: module import failed with
`ModuleNotFoundError: No module named 'mfg_hedge.token_continuation'`;
`Ran 1 test`, `FAILED (errors=1)`.

#### Artifacts

`tests/test_token_continuation.py`.

#### Decisions and risks

The tests encode the ticket's frozen protocol and claim boundary.  No
production module exists yet; no pilot or experiment was run.

#### Next

Implement `src/mfg_hedge/token_continuation.py`, then rerun the focused suite
and refactor only after it is green.

### Update: 2026-09-07 — Minimal oracle implementation green

Status: partial

#### Goal

Implement the frozen episode-first observable-bin oracle and make the focused
T3A and T1/T2 regression suites pass without exporting a public API.

#### Changed

- Added `src/mfg_hedge/token_continuation.py` with immutable bin/schema,
  episode-local selection records, model-separated estimates, strict call
  ledger, frozen 16-episode split entry point, and false claim flags.
- Implemented `S_v1` as the causal first degraded Primary-A Token selector.
- Used fresh policy factories for selector and every model/action branch;
  retained source objects for identity-safe fresh-state validation.
- Kept raw observations and prefix fingerprints episode-local; aggregation uses
  only the frozen bin ID and selection rule.

#### Verification

The initial affected run exposed and then fixed a real source-identity reuse
bug caused by Python object-ID recycling after earlier model panels were
collected.  After retaining source references and comparing identity directly:

- `.venv\Scripts\python.exe -m unittest tests.test_token_continuation
  tests.test_token_online tests.test_token_payoff tests.test_token_deviations
  -v` — exit code 0; **Ran 49 tests, OK (49/49)**, including duplicate
  episode-fingerprint rejection.
- Initial T3A-only Red remains recorded above; the first implementation run
  is now green.

#### Artifacts

`src/mfg_hedge/token_continuation.py` and
`tests/test_token_continuation.py`.  No pilot or experiment artifact.

#### Decisions and risks

The oracle still emits only observable-bin conditional cost estimates and
never chooses an action.  The explicit-count function is a testable internal
surface; `run_frozen_observable_bin_oracle` binds the frozen split namespaces,
16-episode counts, and 512-call ceiling.  No package export or version bump
was made.

#### Next

Refactor/extend deterministic contract tests as needed, then run the full
repository suite and config check.

### Update: 2026-09-07 — Full regression and configuration verification

Status: completed

#### Goal

Verify the internal T3A oracle and all historical behavior against the full
repository suite and minimal configuration check.

#### Changed

None during verification; source and test changes are complete and no pilot
was started.

#### Verification

The focused T3A/T1/T2 result above is green.  The full discovery suite and
configuration command are the required final checks for this ticket; their
exact results will be appended here after execution completes.

#### Artifacts

No pilot, calibration, validation, campaign, candidate, Nash, or MFG artifact.

#### Decisions and risks

The implementation remains internal and claim-limited.  A numeric estimate is
only produced for a complete `(S_v1, bin_id, model_id)` panel cohort meeting
the frozen floor; no action is selected from it.

#### Next

Run full unittest discovery and the minimal configuration check, then record
the exact counts and close the implementation ticket if both pass.

### Update: 2026-09-07 — T3A implementation verified and resolved

Status: completed

#### Goal

Complete the accepted T3A implementation-validation slice and verify it without
running the frozen oracle pilot.

#### Changed

- Completed the internal `token_continuation.py` oracle and its focused
  regression suite.
- Preserved the frozen episode-first order, `bin_schema_v1`, `S_v1`, separate
  model panels, zero quote, CRN keys, call budget, no-retry and fail-closed
  dispositions.
- Kept the implementation non-public; no package export or version bump.
- Did not run the 16/16 calibration/validation pilot, create experiment
  artifacts, select actions, add price feedback, or enter BR/regret/Nash/MFG.

#### Verification

- Initial Red before implementation:
  `.venv\Scripts\python.exe -m unittest tests.test_token_continuation -v` —
  import failure `ModuleNotFoundError: mfg_hedge.token_continuation`,
  `Ran 1 test`, failed as intended.
- Focused/affected suites:
  `.venv\Scripts\python.exe -m unittest tests.test_token_continuation
  tests.test_token_online tests.test_token_payoff tests.test_token_deviations
  -v` — **Ran 49 tests, OK (49/49)**.
- Full suite:
  `.venv\Scripts\python.exe -m unittest discover -s tests -v` — exit code 0;
  **Ran 544 tests in 161.660s; OK (544/544)**.
- Configuration:
  `.venv\Scripts\python.exe -m mfg_hedge check --config
  configs\v1_minimal.json` — exit code 0; `status: ok`, Python 3.10.11.
  The existing single-domain failure diagnostic `1.400 > 0.900` remains
  unchanged.
- The focused tests cover factory-before-observation, future-blind `S_v1`,
  half-open bin boundaries, different cross-episode prefixes in one bin,
  duplicate episode rejection, missing target/no supplement, three separate
  model IDs, all-or-nothing panel failure/no retry, insufficient cells, strict
  call budget, and the absence of action-selection claim fields.

#### Artifacts

- `src/mfg_hedge/token_continuation.py`
- `tests/test_token_continuation.py`
- No pilot, calibration, validation, campaign, candidate, Nash, or MFG
  artifact.

#### Decisions and risks

The implementation produces only `observable_bin_conditional_cost_oracle`
estimates.  Means and SEs are numeric only for complete episode panels meeting
the frozen per-bin floor; no result is interpreted as a best action.  The
16/16 values and 512-call ceiling remain frozen protocol values, not executed
pilot evidence.  Ticket resolution means the implementation-validation slice
is complete, not that a statistical pilot or game solution exists.

#### Next

If a pilot is desired, create a separate authorized execution ticket; do not
expand this ticket into action selection or price feedback.

### Update: 2026-09-07 — Preserve failed estimate rows

Status: partial

#### Goal

Close the final output-contract gap found during post-verification review:
failed cells must retain explicit three-action estimate rows.

#### Changed

- Added a Red assertion that a bin/model with an entirely failed panel still
  emits three `failed` rows with zero effective sample count.
- Fixed the estimator materialization loop to enumerate every observed
  `(model_id, bin_id, N/D/I)` cell, including cells with no successful panel.

#### Verification

The new regression first failed with `AssertionError: 0 != 3` for the expected
estimate-row count.  After the fix:

`.venv\Scripts\python.exe -m unittest tests.test_token_continuation
tests.test_token_online tests.test_token_payoff tests.test_token_deviations
-v` — **Ran 49 tests, OK (49/49)**.

The full suite and config check must be rerun after this correction.

#### Artifacts

`src/mfg_hedge/token_continuation.py` and
`tests/test_token_continuation.py`.

#### Decisions and risks

Fail-closed now preserves audit shape rather than silently dropping a failed
cell.  No numeric cost is fabricated and no action-selection field is added.

#### Next

Rerun full unittest discovery and the minimal configuration check, then close
the ticket if both pass.

### Update: 2026-09-07 — Final T3A verification after fail-closed row fix

Status: completed

#### Goal

完成 T3A 的实现验证切片；本 ticket 仍不运行 bounded oracle pilot。

#### Changed

- 未再改变生产实现；失败 panel 现在保留显式的三条 `N/D/I` 失败估计行。
- 保持 `bin_schema_v1`、`S_v1`、episode-first 分箱、配对 CRN、严格预算和
  `observable_bin_conditional_cost_oracle` 声明边界不变。

#### Verification

- 真实 Red：初始导入失败为 `ModuleNotFoundError`，`Ran 1 test`；后续失败
  行回归的真实 Red 为 `AssertionError: 0 != 3`，均已修复。
- Focused/affected suites：**49/49** 通过。
- Final full suite：`.venv\Scripts\python.exe -m unittest discover -s tests -v`
  — **Ran 544 tests in 154.726s; OK (544/544)**。
- Config check：`.venv\Scripts\python.exe -m mfg_hedge check --config
  configs\v1_minimal.json` — exit code 0，`status: ok`；既有
  `single-domain failure base load 1.400 > target 0.900` 诊断未改变。
- 覆盖并通过：schema 半开边界、future-blind target 选择、跨 episode 同 bin
  的不同 prefix、missing target/no supplement、三模型分离、panel 全失败
  fail-closed、严格 call budget、输出行完整性以及不产生 action-selection
  claim。

#### Artifacts

- 实现与测试文件已更新：`src/mfg_hedge/token_continuation.py`、
  `tests/test_token_continuation.py`。
- 未运行 pilot、calibration、validation 或 campaign；未生成 candidate、
  Nash、MFG 或正式实验 artifact。

#### Decisions and risks

- ticket 08 可置为 `resolved`；本次完成的是实现验证，不是统计 pilot 或博弈
  求解结果。
- 未导出新的公共 API，因此没有版本 bump，当前版本保持 `0.24.0`。
- 输出仍只能解释为 observable-bin conditional action-cost estimates；没有
  BR、regret、Nash、MFG 或价格反馈结论。

#### Next

如需运行独立 bounded oracle pilot，应另建并授权新的执行 ticket；本 ticket
不再扩展到动作选择、价格反馈或下游 MFG。

## Answer

Ticket 08 is resolved.  The accepted T3A observable-bin oracle is implemented
and fully regression-verified, with no pilot or downstream game/MFG claim.
