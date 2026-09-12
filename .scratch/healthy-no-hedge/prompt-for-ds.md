# Prompt for DS

你将在 `D:\project\mfg_hedge_v1` 中实现第一个最小仿真垂直切片。

开始前必须完整阅读：

1. `AGENTS.md`
2. `CONTEXT.md`
3. `docs/adr/0001-v1-scope-and-model-evaluation-separation.md`
4. `.scratch/healthy-no-hedge/spec.md`
5. `.scratch/healthy-no-hedge/issues/01-implement-minimal-simulator.md`

先把 ticket 的 `Status` 从 `open` 改为 `claimed`，然后严格采用 red-green-refactor：先添加会失败的测试，再实现最小代码使其通过，最后在测试保持通过的前提下整理结构。

只实现 Healthy + No Hedge：一个逻辑 Expert、两个健康 Replica、Poisson 到达、Regular/Urgent 类型、Lognormal 服务需求、固定 round-robin Dispatcher、每 Replica FCFS 队列、确定性 workload trace、汇总指标以及唯一 artifacts 目录中的 `summary.json`。不得实现故障、Delayed/Immediate Hedge、Backup、Replay、MFG、价格、quota、多 Expert、绘图或任何第三方依赖。

不要重新解释或扩大需求；spec 是唯一功能口径。若 spec 与现有代码或 ADR 冲突，停止并在 ticket 的 `## Comments` 中写明冲突，不要自行改口径。

完成后执行 spec 中全部验证命令，将实际命令和结果追加到 ticket 的 `## Progress log`，满足全部验收条件后才把 `Status` 改为 `resolved`。最终答复必须严格使用 `docs/agents/update-format.md` 的标题和字段，并给出所有新增/修改文件路径。

