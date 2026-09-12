# 瞬态双队列物理验证报告

日期：2026-09-05。范围：ticket 09 的第一步物理验证。

结论：三个关键反馈与手算一致，独立工作量账本和现有引擎的存活状态一致。
本轮没有发现需要修复的生产引擎问题。结果支持后续在这个物理过程上研究控制，
尚不能证明新目标、81 规则搜索或 MFG 的效果。

## 方法

新增 `tests/transient_physics_support.py`，仅用于测试：

- 每个事件处理前按运行中的副本和物理速度独立积分已执行工作；
- 记录实际入队工作、取消移除的工作和故障丢弃的剩余工作；
- 每个状态/完成/计时器/到达事件处理前后，以及 dispatch 后，直接读取队列；
- 检查 `入队 - 执行 - 取消 - 丢弃 = 存活剩余工作`，并检查 Token/copy 关联；
- 将逐阶段服务积分与最终逐 attempt 执行工作交叉核对；
- 每次被观测运行的完整公开结果与未观测运行逐字段相等。

故障失效与 Replay 入队是同一个 state handler 的有序子步骤；审计在这个
handler 前后取值，不声称在其每一条内部语句后采样。浮点对账使用绝对/相对
1e-9 的数值容差，手算夹具使用二进制精确数值，没有统计近似。

## 1. Replay 集中入队

手工时间线 10/20/50，A 在 D 的速度为 0.5。
三个 A Token 的 Primary 工作为 20、4、6，分别在 10、11、12 到达。
20 时刻前，首个 Primary 已执行 5，其余两项仍排队。

| 量 | 故障处理前 | 故障处理后 |
| --- | ---: | ---: |
| A 剩余工作 | 25 | 0 |
| B 剩余工作 | 0 | 20 |
| A 累计执行工作 | 5 | 5 |
| A 累计丢弃剩余工作 | 0 | 25 |

Replay 工作为 8、6、6，均在 t=20 入 B，按 running 再 FCFS 排列；
分别于 28、34、40 完成。此前 B 的两个小 Primary 共执行 1，因此 B 的
生命周期执行总量为 21，其中 Replay 为 20。

另一个隔离夹具把同样的 B 工作 8、6、6 改成两种入队时间，均在 F 中走 B：

| B 入队时刻 | 每项延迟 | 平均延迟 | t=20 的 B 剩余工作 |
| --- | --- | ---: | ---: |
| 20、20、20 | 8、14、20 | 14 | 20 |
| 20、28、34 | 8、6、6 | 20/3 ≈ 6.667 | 8 |

两组的服务抽样、总执行工作 20 和最后完成时刻 40 都相同，等待过程不同。
这个例子说明：生命周期总工作和排空终点无法替代时间局部的队列信息。
两组到达时间是刻意改变的机制夹具，不是共享 arrival trace 的政策效果实验。

## 2. Backup 胜出取消排队 Primary

时间线 10/30/40。A 的首个 Primary 在 t=10 开始，工作为 4，于 t=18 完成。
Token 2 在 t=11 到达，A Primary 工作为 2；若 Immediate Hedge，B 的工作
为 0.5，t=11.5 胜出，取消尚未开始的 A Primary。

| 量 | No-Hedge | Token 2 Immediate |
| --- | ---: | ---: |
| 后续 Token 4 开始时刻 | 22 | 18 |
| 后续 Token 4 完成时刻 | 24 | 20 |
| A 总执行工作 | 7 | 5 |
| Token 2 被取消 Primary 执行工作 | 不适用 | 0 |

两组共享完整 workload。取消使后续 A Token 提前 4 个时间单位完成，证实 A
本身受保护策略影响；固定 A 队列环境会漏掉该反馈。

对照夹具还验证了保守取消：Primary 若已经运行，即使 Backup 在 10.5 胜出，
它仍执行到 18，消耗完整 4 单位工作，记录为 completed_loser。

## 3. 准入拒绝必须改变后续物理结果

同一个 Token 在 t=19 到达，A Primary 工作为 4，故障 t=20；B Hedge 工作
0.5，B Replay 工作 3。两组共享全部外生数据，仅测试预算不同：

| 量 | 测试预算 1，批准 Immediate | 测试预算 0.5，拒绝 Immediate |
| --- | ---: | ---: |
| requested | Immediate | Immediate |
| applied | Immediate | Normal |
| Replay 次数 | 0 | 1 |
| Token 完成时刻 | 19.5 | 23 |
| A 执行工作 | 0.5 | 0.5 |
| B 执行工作 | 0.5 | 3 |
| 总执行工作 | 1 | 3.5 |

拒绝组完整结果与直接 No-Hedge 一致。这个夹具验证“少发 Hedge 不保证减少
总工作”，不能被解释为批准组在相同预算下的算法优势。

准入函数是测试专用的 mean-one reservation 夹具，只接收 id/到达/request，
没有服务抽样或执行结果输入。它没有替换生产 quota，也没有实现生产级参数校验。

附加验证：cap=2.8125 批准前两次请求、拒绝第三次，累计扣费 2；已运行 Hedge
的抽样工作可为 4，超过 cap 而没有违反预留扣费契约。Delayed 因 Primary
完成而作废时不退款；后续请求变化不改变先前准入；新窗口重置额度。

## 4. 审计可靠性与边界

- 故意漏记 Replay 入队工作：守恒断言拒绝。
- 故意漏记 queued cancellation：守恒断言拒绝。
- 故意漏记 failure discard：守恒断言拒绝。
- Primary 物理完成恰在 failure、Delayed timer 同刻到期：先失效并 Replay，
  完成事件随后失效，timer 作废；即使丢弃剩余工作为零，也保留故障语义。
- 8 个确定性开发种子，每个 100 Tokens，混合 N/D/I：所有事件对账通过，
  观测与未观测公开结果相同，排空后两队列剩余工作为零。

## 5. 验证命令与范围

初始 Red：

`.\.venv\Scripts\python.exe -m unittest tests.test_transient_physics -v`

`Ran 1 test ... FAILED (errors=1)`，缺少 `tests.transient_physics_support`。
这是新增测试工具尚未实现的真实 Red，不是生产引擎存在故障的证据。

实现测试工具后，同一命令：`Ran 13 tests in 0.127s ... OK`。

全量：`.\.venv\Scripts\python.exe -m unittest discover -s .\tests -v`
→ `Ran 347 tests in 22.361s ... OK`，334 既有 + 13 新增。

环境：`.\.venv\Scripts\python.exe -m mfg_hedge check --config .\configs\v1_minimal.json`
→ exit 0，status ok，保留旧配置的 single-domain headroom 提醒。

生产源码、版本、配置、历史指标和 artifact 未修改。测试辅助代码依赖引擎私有
hook；未来内部重构可能需要同步测试工具。当前结果验证有限夹具下的物理契约，
不构成形式证明、统计效果验证、完整新 quota 验证或 MFG 均衡证书。
