# MFG-Hedge v1 方案评估

## 结论

第一版的研究边界是合理的：不做完整 HJB–FPK、不跨层、不训练、不改 Gate、不动态配副本，能把问题压缩成一个可证伪的机制实验。建议先做“单 Expert、两 Replica、两 Token 类型、三动作”，通过后再扩到 8 Experts。

完整 MFG-Hedge 可以进入第一版，但必须和一个更强的“容量感知优先级配额”对照。否则即使胜过固定阈值，也无法回答收益究竟来自 MFG/价格均衡，还是仅仅来自动态 quota。

## 各方案的定位

| 方案 | 价值 | 主要问题 | 在论文中的角色 |
| --- | --- | --- | --- |
| No Hedge | 给出最低额外计算下界，暴露无保护时的尾延迟和 Replay | 可靠性最弱，不能作为主要竞争者 | 必须保留的下界 |
| Immediate Hedge All | 给出强保护、最大冗余的参考点 | 会吞掉近一倍工作量，故障时可能加剧排队 | 必须保留的资源上界 |
| Fixed Delay | 是工程中最自然的 hedged request 基线 | 延迟阈值固定，负载变高会触发“更多 Hedge → 更拥塞 → 更多 Hedge”正反馈 | Protection Storm 的主要对照之一 |
| Risk Threshold | 最直接利用 H/D/F 的启发式 | 公共状态跳变会让同类 Token 集体切换，最容易产生策略突变 | 展示同步切换风险 |
| Static Quota | 能隔离“限制保护比例”本身的效果 | 不随剩余容量和状态变化；20% 必须扫描，不能只报单点 | 回答 MFG 是否只是 quota |
| MFG without Common State | 保留负载与价格协调，去掉 Z | 若队列/价格已间接编码故障，可能与完整方案接近 | 必须保留的状态消融 |
| Proposed MFG-Hedge | 能按 Token 类型、负载、公共状态和稀缺容量联合分配保护 | 依赖动作统计表、收敛速度与价格/阻尼参数；工程上仍需配额投影保证硬容量约束 | 提议方法 |
| Capacity-aware Priority Quota（建议新增） | 按实时 headroom 算可用份额，再按 urgency/边际收益分配；简单、强、可解释 | 不是均衡方法，可能忽略群体反馈和未来触发负载 | 最重要的强启发式对照 |

建议另加一个只用于小规模实验的 Central Planner / Oracle：在已知同一条 trace 的前提下最小化目标函数。它不是可部署基线，而是衡量 Proposed 与可达到上界的差距。

## 需要先修正的建模口径

### 1. 故障后的容量可行性

若健康 offered load 0.7 是相对两副本总容量定义，两个同容量副本掉一个后，基础负载变为 1.4。此时 `B=[0.9C-lambda]+ = 0`，而且即使完全不 Hedge 也已过载。这个场景可用于研究短时瞬态恢复，但不能用于声称稳态容量出清。

第一版应把实验分成两类：

- Degraded / transient failure：保留 0.7，研究短时间冲击和 storm；
- steady failure feasibility：将健康负载降到不高于 0.45，或增加 N+1 冗余容量，再比较保护策略。

报告中需同时给出 `base overload` 与 `hedge budget violation`，不要把基础流量造成的不可行归因于策略。

### 2. Replay 工作量不能从负载固定点中消失

原定义写 `g_N=0`，但 Normal 失败后会 Replay；若 `G` 表示 Hedge 和 Replay 的总额外负载，这两者冲突。建议动作统计拆成：

- `expected_hedge_work`；
- `expected_replay_work`；
- `expected_extra_work = hedge + replay`。

价格可以只对可控的 Hedge 工作收费，但有效利用率必须计入两类额外工作。本环境的 `ActionStats` 已按此拆分。

### 3. Replay 与 deadline miss 分开报告

二者机制和量纲不同，不宜只合并为一个 `q`。求解器可以配置不同权重，最终结果也应分别报告。否则某策略可能只是把 Replay 换成 deadline miss，却在复合指标上看起来更好。

### 4. 有限价格与 Softmax 不自动保证硬约束

在有限逆温度和有限价格下，Softmax 通常仍会给 Hedge 正概率；当 `B=0` 时尤其如此。价格迭代用于得到目标份额，窗口落地时仍应做确定性的 quota projection，并把被投影掉的质量记录成指标。KKT 只能在明确的求解条件下声称近似满足。

### 5. Fixed Delay 的“同步风暴”需要测量而非预设

相对到达时间设置的 timer 在 Poisson 流下天然有一定错峰。Risk Threshold 的公共状态切换反而更容易严格同步。主实验应通过小时间窗峰值/均值比、峰值启动率和持续过载时间验证 storm，不能只凭总额外工作下结论。

## 推荐实验顺序

1. 单 Expert、无故障 sanity：验证队列、分布采样、取消/保守 loser 计费和可重复性。
2. 单 Expert、确定性 H→D trace：比较全部基线，确认 storm 是否真实出现。
3. 加入短时 F 与恢复：区分基础容量不可行和保护负载导致的放大。
4. 负载扫描与 Pareto：所有 timer、quota、gamma 都在校准 seeds 上选取，在独立 evaluation seeds 上报告。
5. 模型失配：查表用 lognormal，事件仿真改为强长尾、混合分布和 bursty arrival。
6. 机制成立后扩到 8 Experts；第一版仍让每个 Expert 独立定价，不引入跨池耦合。

所有方案必须共享 arrival、service-time、故障轨迹和 Gate 结果（common random numbers）。不要让不同策略各自随机生成 trace，否则方差会掩盖差异。开发阶段用 `smoke_tokens=1000`，正式结果再使用至少 100,000 Token × 20 seeds。

## 第一版的通过门槛

- Healthy 时 Proposed 的无意义冗余接近 No Hedge；
- D/F 窗口相对 No Hedge 明显改善 Replay 或 P99；
- 在相同额外工作下，Proposed 相对 Fixed Delay、Risk Threshold、Static Quota 和 Capacity-aware Priority Quota 至少改善一个主要指标，且另一个不恶化；
- 结论在独立 seeds 和至少一种模型失配下仍成立；
- 求解迭代、残差和 quota 投影量足够小，控制开销可接受。

如果 Proposed 只能小幅胜过 Static Quota，却不能胜过 Capacity-aware Priority Quota，就不应继续堆叠完整动态 MFG 理论；应先判断价格均衡是否真的提供额外机制价值。

