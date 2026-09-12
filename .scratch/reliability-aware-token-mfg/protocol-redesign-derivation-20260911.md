# Ticket 06 协议重设计推导：从 r2 `statistics_insufficient` 到可裁定量化提案

Date: 2026-09-11
Status: 分析稿（无新实验；不改动任何冻结定义；所有协议变更仅为提案，待用户裁定）
Data: 仅使用 r2 已提交产物
(`artifacts/reliability-aware-token-mfg-qualification-20260911-r2/summary.json`、
`.scratch/reliability-aware-token-mfg/qualification-runs/reliability-aware-token-mfg-qualification-20260911-r2/`
checkpoint + 32 个内容寻址 blob）与源码冻结定义。
标注约定：【代码已核实】【产物已核实】【报告陈述】【推断】【待确认】。

辅助脚本（只读解码，零调度调用）：
- `.scratch/reliability-aware-token-mfg/analyze_r2_cell_occupancy.py` → `r2_cell_occupancy_report.json`
- `.scratch/reliability-aware-token-mfg/analyze_r2_coarsening.py` → `r2_cell_bucket_table.json`

验证锚点：对 blob 重算 `max((gain_mean + 2·SE)/baseline)` = **1.190440**，与 summary 记录的
normalized finite-K UCB 1.1904399923892601 完全一致 → 解码与分组逻辑可信【产物已核实】。

---

## 0. 冻结定义速查（推导依据）

- 状态身份 `_state_id_from_cell`【代码已核实】
  (`token_mfg_qualification_backend.py:159-160`)：cell 的 `asdict` SHA-256；cell =
  `PopulationCellKey(token_bucket, replica_state, locality_signature, topology)`
  (`population_estimator.py:246-265`)。token 侧 4 桶：class(2)×age(5)×retry(3)×domain(4)，
  上限 120；replica 侧为**全部 K 个 Replica 各自的 8 桶元组** → 状态粒度随 K 增长。
- floor=8【代码已核实】：`min_complete_samples=8` (`token_mfg_qualification_campaign.py:635`)，
  冻结校验 :709-710；生效于 `_finite_k_simultaneous_bound` 的
  `effective_n < minimum_samples → statistics_insufficient` (`token_mfg_qualification_backend.py:135`)。
- active-cell 规则【报告陈述】：仅存在于 ticket 06 协议文本 (:40-43)——occupancy ≥ 0.5% 且
  ≥8 episodes；0.5% 无代码实现（0.5% 阈值是协议约定）。
- UCB 公式【代码已核实】：逐 cell `(state, action)` 分组
  (`token_mfg_qualification_backend.py:99-129`)，gain = baseline − candidate；
  `normalized_upper = (gain_mean + 2.0·SE)/mean(baseline_cost)`；complete 时
  `normalized_simultaneous_ucb = max over cells` (`token_mfg_qualification.py:148-187`)。
- 0.02 门槛【代码已核实】：`token_mfg_formal_backend.py:429-434`——K=64 normalized UCB ≤ 0.02
  且沿 K 单调不增；policy L1 ≤ 0.05 (:417-427)。
- 调用计价【代码已核实】：每 episode `1 + (1+K) + (1+K)`（forward 1 + continuation 1+K +
  deviation 1+K，`token_mfg_streaming_formal.py:53,156`；K=8 → 19）；每 run 上限
  `2·7·(32+32·9+32·9)=8,512`，六 run 51,072，加 holdout 8,064 = 59,168
  (`token_mfg_qualification_campaign.py:45-51,755-763`)。
- share residual【代码已核实】：`max over states L1(predicted_share, realized_share)` ∈ [0,2]
  (`token_mfg_response_solver.py:286-291`)；realized 为逐 episode one-hot 的组内均值
  (`token_mfg_streaming_formal.py:217`)。
- preflight【产物已核实】：K=64 吞吐 0.274 calls/s、最坏单调用 0.92s、8-worker 投影
  31,567s；K=8 最坏单调用 0.055s。

---

## 1. 占用分布画像【产物已核实】

63 个 cell 在 32 episodes 下的占用（解码 32 个 blob 重算）：

- cell 计数直方图：`{1: 48, 2: 7, 3: 4, 11: 4}`；min=1, q25=median=1, mean=1.87, max=11；
  **59/63 未达 floor=8**。
- 结构事实：cell 计数 == 其所在 target state 的 episode 计数（continuation 与 finite 分组
  计数零失配）。18 个 target state 的 episode 计数 `{11:1, 3:1, 2:2, 1:14}`；63 = Σ state 的
  action 支持数（每 state 3–7 个 action bucket，均值 3.5）。
- forward 快照 43 个 state 的 occupancy_mass：min=q25=median=1.266%, max=13.9%；
  **0 个 state < 0.5%**；但只有 18/43 成为过 target——**25 个 occupancy ≥1.27% 的 active state
  在 32 episodes 中从未被抽中**。
- 每 episode 实际调用 23×11 + 8×9 + 1×7 = 332（reserved 608 = 32×19）。

稀有 cell 频率估计与 floor 所需 episode 数【推断，基于 occupancy∝target 概率假设】：

| 情形 | p̂ | N（点估计 8/p） | N（95% Poisson, λ=13.9） |
|---|---|---|---|
| 已观测最稀有 target state | 1/32 = 3.1% | 256 | ~440 |
| 43 个 forward state 的 occupancy 下界 | 1.266% | 632 | **1,098** |
| active 边界 state（occupancy=0.5%） | 0.5% | 1,600 | **2,780** |

不确定度：p̂ 来自单个 32-episode 样本；从未被抽中的 state 真实 p 可能低至 0.5%（active 阈值）。
且 N 增大时尾部 state 会陆续进入 active 集合（coupon-collector 效应）→ **在冻结定义下
floor 的 N 需求是开口的，量级 10³–3×10³，而非 32**。

## 2. 分叉一：粗化分箱

关键发现【产物已核实】：63 个 cell 的 action bucket 在 replica 侧完全同标签
（queue_0_1 / work_0_1 / idle / hazard_0p01_0p05 / UP），仅 locality 分两档
（other_domain 48 / same_domain 15）。**replica 侧 action 合并对 cell 数零效果**；
cell 多样性全部来自 token 侧 + 全 K-replica 元组的状态身份。

缺失字段【待确认】：r2 blob 未持久化每个 state 的 token 侧 age/retry/domain 桶标签
（只有 token_class 与 state 哈希），故合并后的精确占用分布无法从 r2 产物计算，
以下 N 估计均为【推断】：

| 方案 | 状态身份变更 | cell 总数上界 | 合并后 p_min【推断】 | floor 所需 N（95%） | 信息损失 |
|---|---|---|---|---|---|
| C1 | 状态 = TokenBucket（剔除 replica_state 元组与 locality） | 120 × ≤7 ≈ ≤840（观测 ≤43×3.5） | ~3–5% | **280–460** | 策略不再条件于全局 replica 环境（队列/风险剖面）；但 mean-field 型空间保留，且状态空间变为 K-不变 |
| C2 | 状态 = class×age（retry 3→1、domain 4→1 也合并） | 10 × ≤7 ≈ ≤70 | ~5–10% | **140–280** | retry 次数与 ingress domain 区分被合并掉 |
| C3 | 仅合并 hazard 4→2、queue 5→3 | 不变（r2 中 action 侧已塌缩） | 不变 | 无改善 | 无（也基本无用） |

补充论点【代码已核实 + 推断】：冻结状态身份含全 K-replica 桶元组，状态数随 K 增长、
每状态 occupancy 随 K 稀释 → floor 问题在 K=64 比 K=8 严格更糟；C1 同时消除这个
跨 K 不可比性（policy L1 当前比较的是不同粒度的对象）。

## 3. 分叉二：门槛重定

UCB 缩放【推断，SE 主导假设 UCB ∝ 1/√n】：观测最坏 cell 当前 n=2、normalized upper
1.1904。

| normalized UCB 门槛 | 样本倍数 (1.1904/t)² | 该 cell 所需 n | 经 occupancy(p̂=6.25%)折算 N |
|---|---|---|---|
| 0.02（冻结） | 3,543× | 7,086 | ~113,000 |
| 0.05 | 567× | 1,134 | ~18,000 |
| 0.10 | 142× | 283 | ~4,500 |
| 0.25 | 23× | 45 | ~720 |

关键事实【代码已核实】：本次门禁失败是 **floor=8（59/63 cell）** 触发的
`statistics_insufficient`，与 0.02 门槛无关——**只放宽门槛不改变任何结论**；
门槛必须与 N 增长或粗化联用才有意义。
科学含义：0.02 ≈ gain UCB 压在 baseline 私有成本的 2% 内（近精确的有限-N gain 证据）；
0.05/0.10 是"正则化近似"容差，主张强度从 ε-gain 降为方向性/近似性结论；
0.25 以上基本不再约束。

## 4. 分叉三：share residual 升主指标

- 1.7143 的来源【产物已核实 + 推导】：12/7 = 2(m−1)/m，m=7 个支持 action 下
  one-hot realized（n_state=1 时）对 uniform predicted 1/7 的 L1——**结构性小样本伪影**，
  不是校准失败。
- 收敛速率【推断】：realized 是 one-hot 的组内均值，E[L1] ≈ √(2(m−1)/(πn))；
  要 max-over-states L1 ≤ 0.01（冻结门槛）：m=7 需 n≈3.8×10⁴/状态，m=3 需 n≈1.3×10⁴/状态，
  再乘 43 状态的 max 多重性校正 → **每状态 10⁴–10⁵ episodes**。
- 结论：share residual 升主指标不但躲不开样本墙，其 0.01 门槛比 UCB 门槛**严格更难**
  （每状态 n 需求大一个量级以上），且同样是逐状态 coupon-collector 结构。
  除非同步把 0.01 门槛放宽到 ~0.2（one-hot 伪影量级以下仍紧），否则不可行。

## 5. 成本墙核算

吞吐锚点【产物已核实】：K=8 最坏单调用 0.055s → 8-worker 上界 ~145 calls/s；
K=64 0.274 calls/s（8-worker 投影 31,567s = 8.77h，已逼近 12h 门禁）。
r2 panel 实际墙钟未持久化【待确认】，K=8 列均为 preflight 上界估计【推断】。

每 episode 调用 = 1+2(1+K)：K=8→19，K=16→35，K=32→67，K=64→131【代码已核实】。

| 协议候选 | K=8 全资格（3 starts×2 models×7 iter） | K=64 每 iteration | K=64 可行性 |
|---|---|---|---|
| 冻结协议 N=32（ceiling 51,072 calls） | ~176s | 4,192 calls → 4.2h | 单 iteration 可行；全 campaign 51.8h > 12h 门禁 → **不可行**（已知门禁失败） |
| N=280（C2 粗化后） | 223,440 calls → ~0.43h | 36,680 → 37.2h | **不可行**（单 iteration 超门禁 3 倍） |
| N=560（C1 粗化后） | ~0.85h | 73,360 → 74.4h | 不可行 |
| N=1,100（不粗化 95% floor） | ~1.7h | 144,100 → 146h | 不可行 |
| N=2,800（active 边界 95%） | ~4.3h | 366,800 → 372h | 不可行 |

结论：**任何能真正满足 floor 的 N（≥~300）都使 K=64 单 iteration 超过 12h 门禁**
（N=280 时已 37h）；K=8 侧在 N≤2,800 内全程 ≤4.3h，完全可行。

## 6. 推荐方案【提案，待用户裁定】

三个分叉各自都不闭环：分叉 2/3 需要 N→10³–10⁵（K=64 成本不可行）；
分叉 1 把 N 降到 300–600 但 K=64 仍不可行，且需改冻结 schema。推荐组合：

1. **C1 粗化**：状态身份降为 TokenBucket（剔除全 K-replica 元组）。状态空间 K-不变、
   cell 上界 120，floor 所需 N≈300–600（95%）。——这是对冻结定义的最小科学改动，
   顺带修复跨 K policy 比较的对象粒度问题。
2. **K=8-only 正式资格**：N=600/iteration，全 K=8 campaign ≈0.9h（8-worker 12h 门禁内）。
3. **K=16/32/64 降级为样本外诊断**：冻结 N=32 panel，UCB 以区间/上界形式发布，
   标注为 finite-N epsilon 证据，不再作为硬性 gate。
4. 若 1–3 均被否决（协议一字不改）：替代研究问题——**支持度充分 cell 子集资格**：
   仅在达到 floor 的 cell 子集（r2 中为 n=11 的 4 个 cell 所属状态）上发布 simultaneous
   UCB 与 holdout 九臂对比，结论显式限定在该子空间，不做全空间断言。

理由：冻结定义把状态粒度与 K 耦合（replica 元组），使 floor 在 K=64 下无可行解；
解耦（C1）是恢复可执行跨 K 比较的最小改动；其余分叉不改变这一结构性结论。
