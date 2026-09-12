# Design T3A statistical execution protocol

Type: design
Status: resolved
Blocked by: none

## Scope

Design-only ticket for a separate observation-occupancy calibration and the
subsequent stratified T3A oracle allocation.  The existing 16/16
implementation-validation configuration is not statistical evidence and is
not reused.  This ticket does not run calibration, run N/D/I panels, produce
cost estimates, choose actions, or enter price feedback, BR, regret, Nash, or
MFG work.

## Frozen design contract

The accepted estimand remains:

```text
Q_m(u | B(O)=b, S_v2)
```

with the existing `bin_schema_v1`, episode-level independence, complete
within-episode N/D/I CRN, and all-or-nothing panel failure from ADR-0022.

### Occupancy calibration

- Namespace: `token-mfg-restoration:t3a:occupancy:v1`.
- Macro seed: `20260909`.
- Protocol: `occupancy_v1`.
- Episode count: `512`, assigned deterministically by
  `anchor_id = episode_index % 4`.
- The four pre-registered D-phase anchor windows are exactly the existing
  `phase_age` bins: `[0,1.5)`, `[1.5,3)`, `[3,10)`, and `[10,+inf)`.
- `S_v2` selects at most one target: the first arrival in canonical
  `(arrival_time, token_id)` order whose current observation is phase `D`,
  Primary `A` (`primary_replica == 0`), and whose phase age is in the assigned
  anchor window.  It never falls back to another anchor or later target.
- The selector is called only on the causal prefix at each arrival.  The
  anchor assignment depends only on the frozen episode index; no future
  arrival, fault, work, action, completion, or outcome is inspected.
- Calibration records only target selection, `bin_id`, anchor, observation and
  trace fingerprints, `missing_target`, and `missing_bin` counts.  It does not
  run N/D/I cost panels and cannot produce action comparisons.
- Calibration consumes all 512 episode slots.  Missing targets/bins are not
  replaced, and duplicate or invalid episode identity is fail-closed.

### Occupancy-to-validation allocation

- The target-eligible support is the 72 combinations of Token class,
  phase-age bin, queue-load bin, and reservation-balance bin; fixed D phase,
  Primary A, and D arrival-time support are not treated as free dimensions.
- A bin is retained only when the occupancy library contains at least
  `16` valid target observations for that canonical `bin_id`.  This is a
  deterministic support screen based only on occupancy, never on cost.
- Each retained bin requires `32` complete independent episode panels.  This
  replaces the implementation smoke floor of `2`; the three action branches
  are not counted as three independent samples.
- Let `p_min` be the smallest retained-bin occupancy count divided by `512`.
  Before validation generation, freeze one common validation episode count:

  ```text
  validation_episode_count = ceil(2 * 32 / p_min)
  ```

  Since retained bins have at least `16/512` occupancy, this count is at most
  `2048`.  The common library is assigned the same four anchor strata by
  `episode_index % 4`; there is no bin-specific top-up, filtering, or
  replacement after validation starts.  A retained bin with fewer than 32
  complete panels is `insufficient`.
- Validation namespace: `token-mfg-restoration:t3a:oracle-validation:v1`.
- Validation macro seed: `20260910`.
- The calibration library is never used as validation input, and the previous
  `t3a:calibration:v1`/`t3a:validation:v1` 16/16 smoke libraries are never
  reused as statistical data.

### CRN and call budget

- Exogenous episode key:
  `token-mfg-restoration:t3a:{split}:v1:{macro_seed}:episode:{episode_index}`.
  The anchor is a deterministic projection of the episode index and is not a
  random-stream key.  Action and model IDs are never appended to the
  exogenous key.
- All N/D/I branches for one episode share its complete exogenous trace and
  prefix.  Different episode indices use independent namespaces.
- Calibration call ceiling: `512` selector calls.
- Validation worst case: `2048 * (1 + 3 * 4) = 26,624` scheduler calls
  (one selector call plus four calls for each of three cost models).
- Combined protocol ceiling: `27,136` scheduler calls.  No call is retried;
  reservation and settlement are exact, and a failed task/panel remains in
  the audit.

### Dispositions

- `missing_target`: consumes its episode slot; no replacement.
- `missing_bin`: consumes its episode slot; no neighboring-bin merge.
- Any failed/non-finite/identity-mismatched/incompletely drained N/D/I branch
  fails the whole model panel; no surviving branch contributes a cost and no
  retry is allowed.
- A retained bin below 32 complete panels is `insufficient`, with null
  estimates and no action claim.  It is not removed after seeing costs and is
  not repaired by extra samples.
- A calibration result is occupancy metadata only.  The later validation
  result, if separately authorized, may carry only the
  `observable_bin_conditional_cost_oracle` label.

## Acceptance criteria

- Add a Proposed ADR describing this two-stage occupancy/allocation protocol.
- Explicitly label 16/16 as implementation-validation smoke, not a statistical
  oracle result.
- Preserve ADR-0022's accepted status and remove any stale Proposed-status
  activation wording.
- Define deterministic occupancy records, allocation-manifest fields, split
  fingerprints, and the fail-closed dispositions above.
- Do not modify production source, run calibration, run validation, or create
  cost/candidate/BR/Nash/MFG artifacts.

## Progress log

### Update: 2026-09-07 — Claim statistical execution design

Status: partial

#### Goal

Design a statistically meaningful follow-up to the T3A implementation smoke,
starting with independent target-bin occupancy calibration and stopping before
any conditional cost panel.

#### Changed

Created and claimed this design ticket. No production code or experiment data
has been changed.

#### Verification

Read `AGENTS.md`, `CONTEXT.md`, the issue/update rules, ADR-0022, ticket 07,
and the resolved ticket 08. Confirmed that the current 16/16 configuration
cannot support the 72 target-eligible observable-bin combinations and must not
be interpreted as a statistical estimate.

#### Artifacts

None at claim time.

#### Decisions and risks

The proposed protocol separates occupancy from cost estimation, changes the
selector to deterministic D-phase anchor strata, and uses a new validation
namespace. Numeric allocation values are protocol design, not executed sample
evidence.

#### Next

Write ADR-0023 and perform a documentation-only review; do not execute the
occupancy library in this ticket.

### Update: 2026-09-07 — Occupancy-gated statistical protocol drafted

Status: completed

#### Goal

将 T3A 从 16/16 implementation-validation smoke 推进到可审查的统计执行
设计，但停在 occupancy calibration 之前，不运行任何 oracle panel。

#### Changed

- 新增 Proposed [ADR-0023](../../../docs/adr/0023-t3a-statistical-execution-protocol.md)。
- 冻结 `S_v2` 的四个 D phase-age anchor、512 个 occupancy episode、独立
  namespace/seed、`n_occ >= 16` 的支持筛选、每 bin 32 个完整 panel 的最低
  floor，以及由 `p_min` 决定的 validation library size。
- 冻结 worst-case call accounting：occupancy 512、validation 26,624、
  combined 27,136 scheduler calls；失败不重试、不补抽。
- 修正 ADR-0022 中残留的 Proposed activation wording，明确其已于
  2026-09-07 接受，但只授权实现层，不授权统计 pilot。

#### Verification

- 只读核对了 `AGENTS.md`、`CONTEXT.md`、issue/update 规则、ADR-0022、
  tickets 07/08 及 T3A 实现边界。
- 文档检查确认 16/16 被标记为 implementation-validation smoke，旧
  `t3a:calibration:v1`/`t3a:validation:v1` 不复用，ADR-0022 状态和激活门
  一致，ADR-0023 保持 `Proposed`。
- 未运行 occupancy calibration、N/D/I cost panel、pilot、validation 或
  任何正式实验；未运行生产测试，因为本轮仅修改设计文档。

#### Artifacts

- `docs/adr/0023-t3a-statistical-execution-protocol.md`
- `docs/adr/0022-token-conditional-continuation-oracle.md`
- 本 ticket；无数据、cost、candidate、BR、Nash、MFG 或实验 artifact。

#### Decisions and risks

- occupancy calibration 与正式 validation 完全隔离；calibration 只统计
  bin occupancy，不估计动作优劣。
- 32-panel floor 是统计执行门槛，不是 Nash 或 MFG 认证；即使达到也不
  授权选择动作。
- `S_v2` 的四 strata 是预注册设计值，尚未产生 occupancy 结果；实际保留
  bin manifest 必须在独立 calibration 完成后密封，且不能读取 cost。

#### Next

等待 ADR-0023 审查与明确授权；之后另建执行 ticket 运行独立 occupancy
calibration。不要复用 16/16 数据或直接运行 N/D/I pilot。

## Answer

统计执行协议设计完成，ticket 09 已 resolved。当前仍只有 T3A 执行机制，
没有统计意义的 `Q_m(u|B(O)=b,S)` 估计，也没有 BR、regret、Nash、MFG 或
价格反馈结果。

### Update: 2026-09-07 — Freeze physical and population provenance

Status: completed

#### Goal

修订 ADR-0023，使统计 estimand 明确依赖完整物理环境 `theta_v1` 与唯一
群体策略 `pi_NIIN_v1`，并补齐模型价格 provenance 与逐模型统计 floor。

#### Changed

- 将 estimand 改为
  `Q_m(u | B(O)=b, S_v2; theta_v1, pi_NIIN_v1)`。
- 冻结一个 Expert、A/B 两副本、`arrival_rate=1.4`、`[0,320)` cutoff、
  `R/U=.8/.2`、lognormal `mean=1/CV=.5`、确定性
  `H=[0,100), D=[100,200), F=[200,220), R=[220,+inf)`、Dispatcher、
  slowdown `2.0`、hedge delay `2.0`、complete drain、workload/fault key
  schema，以及 Reservation `window=25, budget_rate=.45, scale=.25,
  mean_requirement=1` 和 cap `2.8125`。
- 冻结 occupancy/validation 共用的 `pi_NIIN_v1`：静态 `NIIN` 规则
  `(N,I,I,N)`、`late_after=50.0`、fresh `new_episode()`、无私有随机数、
  T1 policy-key schema；仅 namespace/macro seed 不同。
- 分别冻结三种 ADR-0021 模型的 gamma、work/waste、SLO、deadline、零价格
  和 price basis，并要求所有输出带 environment、policy、schema、selector
  和 model fingerprints。
- 明确 32-panel floor 对每个 `(model_id, bin_id)` 分别检查；模型之间不得
  借用有效样本。
- 明确 `ceil(2*32/p_min)` 只是工程分配规则，不是统计功效保证。

#### Verification

- 文档核对确认 occupancy 与 validation 使用相同 `theta_v1`、
  `pi_NIIN_v1` 和模型 provenance，仅 namespace/macro seed 分离。
- `.venv\Scripts\python.exe -m unittest discover -s tests -v` —
  **Ran 544 tests in 153.610s; OK (544/544)**。
- `.venv\Scripts\python.exe -m mfg_hedge check --config
  configs\v1_minimal.json` — exit code 0，`status: ok`；既有
  `single-domain failure ... 1.400 > 0.900` 诊断保持不变。
- 本轮为文档修订，没有生产代码 Red；没有运行 calibration、validation、
  N/D/I cost panel 或任何 pilot。

#### Artifacts

- `docs/adr/0023-t3a-statistical-execution-protocol.md`
- `docs/adr/0022-token-conditional-continuation-oracle.md`
- 本 ticket；无数据、cost、candidate、BR、Nash、MFG 或实验 artifact。

#### Decisions and risks

- ADR-0023 仍保持 `Proposed`，等待用户确认；本 ticket 只是设计修订，
  不授权执行。
- `E_validation` 只是在 occupancy 后冻结的工程分配，不保证任何
  `(model_id, bin_id)` 必然达到 32；不足仍为 `insufficient`。
- T3A 结果仍不能解释为动作选择、BR、regret、Nash、MFG 或价格反馈。

#### Next

等待用户审查并明确接受 ADR-0023；接受后再另建执行 ticket 运行独立
occupancy calibration，不能复用 16/16 smoke 数据。

### Update: 2026-09-07 — Separate load baseline and collapse physical reruns

Status: completed

#### Goal

完成 ADR-0023 的最后三项统计 provenance 收口，避免 T1 历史夹具、模型
scorer 和物理 rerun 被混为一谈。

#### Changed

- 将实验环境命名为 `theta_token_load0p7_v1`，明确 arrival rate `1.4`
  来自 `healthy_offered_load=0.7`；明确 T1 的 `0.9` 是历史等价夹具，D
  基础负载比约 `0.9333`、F 基础负载比 `1.4`，且 cap `2.8125` 是历史
  comparator quota，不是 `1.4` headroom 推导值。
- 明确三模型共享同一物理结果：每个 validation episode 为一次
  target-selection run 加 baseline/N/D/I 四次 physical run，随后将四个
  结果分别交给三个独立 scorer；预算改为 validation `10,240`、combined
  `10,752`。
- 用 `token_t3a_source_bundle_v1` 替代少数文件 hash：包含指定根文件、
  完整本地 import closure、当前 package source set、配置字节和未来执行
  模块；缺失、变更或未列出的依赖 fail-closed。

#### Verification

- `.venv\Scripts\python.exe -m unittest discover -s tests -v` —
  **Ran 544 tests in 161.597s; OK (544/544)**。
- `.venv\Scripts\python.exe -m mfg_hedge check --config
  configs\v1_minimal.json` — exit code 0，`status: ok`；既有
  `single-domain failure ... 1.400 > 0.900` 诊断保持不变。
- 只修改 ADR/ticket 文档；未运行 occupancy、validation、cost panel、pilot
  或正式实验。

#### Artifacts

`docs/adr/0023-t3a-statistical-execution-protocol.md` 与本 ticket；无数据或
正式实验 artifact。

#### Decisions and risks

- ADR-0023 仍保持 `Proposed`，等待用户确认；本 ticket 的 `resolved` 只
  表示设计修订和回归验证完成。
- 三个模型仍拥有独立参数、price basis、model ID 和 provenance；共享物理
  run 不合并 scorer 统计，也不产生动作选择含义。
- `10,752` 是当前冻结环境的最大 scheduler-call 预算，不是样本精度或
  statistical-power 保证。

#### Next

等待用户接受 ADR-0023；接受后再另建执行 ticket，先运行独立 occupancy
calibration，不能复用 16/16 smoke 数据。

### Update: 2026-09-07 — Seal source-manifest provenance

Status: completed

#### Goal

补充群体策略与模型实现的 source-manifest fingerprint，完成本轮文档修订的
可复现性封口。

#### Changed

- 在 ADR-0023 中加入 `token_online.py`、`transient_control.py` 和
  `token_payoff.py` 的 SHA-256 source manifest。
- 保持物理参数、`pi_NIIN_v1`、模型参数、price basis 和逐
  `(model_id, bin_id)` floor 不变。

#### Verification

- source manifest 由当前工作树文件的 SHA-256 只读计算得到；未修改生产源。
- 上一条验证已实际完成：full suite **544/544**，config check
  `status: ok`；本次仅补充文档 provenance。
- 未运行 occupancy calibration、validation、pilot 或任何 cost campaign。

#### Artifacts

`docs/adr/0023-t3a-statistical-execution-protocol.md` 与本 ticket；无实验
数据或正式 artifact。

#### Decisions and risks

ADR-0023 仍为 `Proposed`，等待用户接受；source manifest 不构成执行授权，
也不把 16/16 smoke 变成统计结果。

#### Next

等待 ADR-0023 的用户确认；确认后再单独建立 occupancy calibration 执行
ticket。
