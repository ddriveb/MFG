# MoE 专家复制、Token 路由与 Hedge 方法统计

更新日期：2026-09-09

## 1. 结论先行

在本文归一整理的 30 项代表性工作中，**没有一项把“同一个 Token 同时或延迟发送到同一 Logical Expert 的两个物理 Replica，并取最先完成结果”作为核心 MoE 方法**。这不证明全球文献绝对不存在此做法，但足以说明它不是当前 MoE 专家复制文献的常规含义。

MoE 论文中的 `expert replication` 通常指：复制热门 Expert 的权重，然后把**不同 Token**分摊给不同副本，以消除热点、提高吞吐或改善故障恢复。它通常不是 request hedging。最接近真正 Hedge 的公开方法主要来自通用在线服务领域，例如 [The Tail at Scale](https://research.google/pubs/the-tail-at-scale/) 的延迟副本请求和 [LÆDGE](https://www.usenix.org/conference/nsdi21/presentation/primorac) 的空闲副本 work-conserving Hedge，而不是 MoE 专用路由文献。

因此，本项目更准确的定位不是“新的 Expert placement 算法”，而是：

> 在已有同一 Logical Expert 跨故障域复制的拓扑上，研究故障感知、队列感知和工作预算约束的 Token 执行路由与选择性 same-Expert replica hedging。

目前实验又进一步表明：在现有负载和故障配置下，动态 Primary 路由的收益大于增加 Hedge 的收益；正预算 Hedge 反而增加了工作量、miss 和 Protection Storm。这意味着未来方法不能以“多发副本”为主卖点，而应以“何时只改 Primary 路由、何时才值得额外执行一个副本”为核心。

## 2. 统计口径

### 2.1 样本边界

本文统计 2020--2026 年间 30 项有代表性的 MoE 模型、训练系统、推理系统与容错工作。纳入条件是明确涉及以下至少一项：Token-to-Expert 路由、负载均衡、Expert placement/replication、Token dispatch、推理调度、Expert offload/prefetch 或故障恢复。

这是一份**结构化代表样本**，不是声称穷尽 DBLP、Semantic Scholar 或全部 arXiv 的严格文献计量普查。百分比只描述本文 30 项样本。每项工作只按其主要贡献计入一个类别，避免一篇论文重复计数。

精确关键词还检索了 `MoE request hedging`、`mixture of experts hedged requests`、`expert replica hedge token inference` 和 `MoE redundant execution straggler replica token`。检索没有找到一项清晰的、以 same-Expert same-Token first-completion racing 为核心的 MoE 论文。这个结果可以支持“样本文献中未见”，不能直接支持“我们是世界首创”。

### 2.2 四种容易混淆的“一个 Token 去多个地方”

| 机制 | 是否同一 Logical Expert | 是否重复执行同一 Token | 是否取最先完成者 | 常见目的 |
| --- | ---: | ---: | ---: | --- |
| Top-k MoE 路由 | 否，不同 Expert | 是，但不同参数 | 否，通常组合多个输出 | 模型质量和专家专门化 |
| Expert replica 分流 | 是，相同参数副本 | 否，每个 Token 通常只去一个副本 | 否 | 热点负载均衡、吞吐和局部性 |
| Replay / failover | 是 | 失败后重新执行 | 否，通常是失效替代 | 容错 |
| 真正 request hedge | 是 | 是，副本可并行存活 | 是 | 抑制尾延迟或故障风险 |

本仓库中的 `Hedge` 专指最后一行。不能把 Top-2 Gate、权重预取、Expert 复制或故障后的 Replay 统计成 Hedge。

## 3. 主要方法分布

按每篇论文的主要贡献归类，30 项样本分布如下：

| 主要方法族 | 数量 | 样本占比 | 典型做法 |
| --- | ---: | ---: | --- |
| 运行时调度、并行与通信优化 | 9 | 30.0% | all-to-all 调度、通信计算重叠、动态并行、Token/device 协同 |
| Expert placement、replication 与跨副本分流 | 7 | 23.3% | 复制热门 Expert、迁移/扩缩副本、按历史或预测负载放置 |
| 模型级语义路由与负载平衡 | 6 | 20.0% | Top-1/Top-2 Gate、均衡分配、Expert Choice、容量感知 drop/reroute |
| 内存、缓存、offload 与 prefetch | 5 | 16.7% | CPU/GPU 混合、Expert cache、预测激活、权重预取 |
| 故障恢复与 checkpoint | 3 | 10.0% | 重放、重配置、恢复 Expert 覆盖、稀疏 checkpoint |
| same-Expert same-Token Hedge | **0** | **0.0%** | 本样本中没有作为核心 MoE 机制出现 |

另外，至少 6 项工作明确使用 Expert replica/replication：FasterMoE、FlexMoE、Lazarus、CRAFT、预测式 Expert replication 和 Libra。它们复制的是 Expert 参数或执行位置，随后做 Token 分流、预取或故障恢复；**这 6 项中仍是 0 项采用同 Token first-completion Hedge 作为主机制**。

## 4. 逐篇方法矩阵

### 4.1 模型级语义路由与负载平衡：6 项

| 工作 | 年份/出处 | 主要方法 | 同 Token same-Expert Hedge? |
| --- | --- | --- | ---: |
| [GShard](https://openreview.net/pdf?id=qrwe7XHTmYb) | ICLR 2021 | 稀疏 Gate、容量约束、自动分片；模型层 Token-to-Expert 路由 | 否 |
| [BASE Layers](https://proceedings.mlr.press/v139/lewis21a.html) | ICML 2021 | 将 Token-to-Expert 变成等容量线性指派，每 Token 一个 Expert | 否 |
| [Switch Transformer](https://www.jmlr.org/beta/papers/v23/21-0998.html) | JMLR 2022 | Top-1 路由、容量因子和负载平衡损失 | 否 |
| [GLaM](https://proceedings.mlr.press/v162/du22c.html) | ICML 2022 | 稀疏 Top-k 语义 Expert 激活 | 否；不同 Expert 的输出是模型计算 |
| [Expert Choice Routing](https://proceedings.neurips.cc/paper_files/paper/2022/hash/2f00ecd787b432c1d36f3de9800728eb-Abstract-Conference.html) | NeurIPS 2022 | Expert 选择其 Top-k Token，保证每个 Expert 容量 | 否 |
| [Capacity-Aware Inference](https://proceedings.iclr.cc/paper_files/paper/2026/hash/94e845868a9ace4bc239d0c529d32f4c-Abstract-Conference.html) | ICLR 2026 | 对热点 Expert 做容量感知 Token drop，或扩展候选后 reroute 到低负载 Expert | 否；可能改变 Logical Expert/质量 |

这组方法主要决定“Token 应该由哪个语义 Expert 计算”。它们可能改善负载，但会改变 Gate/模型语义，因而不属于本项目“Gate 固定后的 same-Expert replica 路由”问题。

### 4.2 Expert placement、replication 与跨副本分流：7 项

| 工作 | 年份/出处 | 主要方法 | Replica 如何使用 | 同 Token Hedge? |
| --- | --- | --- | --- | ---: |
| [FasterMoE](https://doi.org/10.1145/3503221.3508418) | PPoPP 2022 | shadow popular experts，基于负载复制/广播热门 Expert | 不同 Token 分摊到 Expert 副本 | 否 |
| [FlexMoE](https://arxiv.org/abs/2304.03946) | 2023 | 根据动态负载 expand、shrink、migrate Expert | 调整副本数量和设备放置 | 否 |
| [MegaScale-Infer](https://chaojin0310.github.io/files/papers/SIGCOMM25-MegaScale-Infer.pdf) | SIGCOMM 2025 | 根据历史流量规划大规模 Expert deployment | placement 与通信优化 | 否 |
| [CRAFT](https://proceedings.mlsys.org/paper_files/paper/2026/hash/3a7f9e485845dac27423375c934cb4db-Abstract-Conference.html) | MLSys 2026 | 在显存预算下按层选择有收益的 Expert replica，避免过度复制 | 不同 Token 分摊到副本，提高吞吐 | 否 |
| [Fast MoE Inference via Predictive Prefetching and Expert Replication](https://arxiv.org/abs/2605.11537) | 2026 预印本 | 预测即将过载的 Expert 并为后续 batch 复制 | batch 中不同 Token 并行分流 | 否 |
| [Mixture-of-Experts Serving](https://arxiv.org/abs/2607.17880) | 2026 预印本 | 联合决定每个 Expert 的 GPU 数量并考虑重配置成本 | 资源分配和 placement | 否 |
| [Scaling Multi-Node MoE Inference Using Expert Activation Patterns](https://arxiv.org/abs/2604.23150) | 2026 预印本 | 按激活轨迹进行 micro-batch grouping 与 Expert placement，提高 Token locality | 降低跨节点 all-to-all | 否 |

这组最接近“专家备份/复制”，但其核心问题是**副本放多少、放哪里，以及把不同 Token 分给哪个副本**。通常一个 Token 只在一个副本执行一次。

### 4.3 运行时调度、并行与通信优化：9 项

| 工作 | 年份/出处 | 主要方法 | 同 Token Hedge? |
| --- | --- | --- | ---: |
| [DeepSpeed-MoE](https://arxiv.org/abs/2201.05596) | ICML 2022 | Expert/tensor/data parallelism 与 MoE 推理优化 | 否 |
| [Tutel](https://proceedings.mlsys.org/paper_files/paper/2023/hash/5616d34cf8ff73942cfd5aa922842556-Abstract-mlsys2023.html) | MLSys 2023 | 自适应 MoE 并行、pipelining 和运行时优化 | 否 |
| [Lina](https://www.usenix.org/conference/atc23/presentation/li-jiamin) | USENIX ATC 2023 | all-to-all 优先调度；推理时按 Expert 热度动态平衡资源和传输 | 否 |
| [SmartMoE](https://www.usenix.org/conference/atc23/presentation/zhai) | USENIX ATC 2023 | 离线候选并行池加在线策略选择 | 否 |
| [MegaBlocks](https://proceedings.mlsys.org/paper_files/paper/2023/hash/5a54f79333768effe7e8927bcccffe40-Abstract-mlsys2023.html) | MLSys 2023 | dropless block-sparse MoE 计算，避免因容量丢 Token | 否 |
| [ScheMoE](https://doi.org/10.1145/3627703.3650083) | EuroSys 2024 | 调度通信和计算任务、形成高效流水 | 否 |
| [Lancet](https://proceedings.mlsys.org/paper_files/paper/2024/file/339caf45a6fa281cae8adc6465343464-Paper-Conference.pdf) | MLSys 2024 | 全图通信-计算重叠 | 否 |
| [Semantic Parallelism](https://proceedings.iclr.cc/paper_files/paper/2026/hash/f0552f14388d95b19740dee809f5cad1-Abstract-Conference.html) | ICLR 2026 | 联合 Expert device placement 与 request/Token device scheduling | 否 |
| [Libra](https://proceedings.iclr.cc/paper_files/paper/2026/hash/9ff1ac9a659085fed0735362cafe5e53-Abstract-Conference.html) | ICLR 2026 | 预测下一层 Expert 激活、局部性复制与 Token sharding | 否；其 speculative 是 Expert 预测/预取，不是重复 Token 竞速 |

这些方法优化的是执行图、通信、批处理、Token sharding 或设备调度。即使使用 “speculative” 一词，也可能只是提前预测和预取 Expert 权重，不能自动归为 request hedging。

### 4.4 内存、offload、cache 与 prefetch：5 项

| 工作 | 年份/出处 | 主要方法 | 同 Token Hedge? |
| --- | --- | --- | ---: |
| [SiDA-MoE](https://proceedings.mlsys.org/paper_files/paper/2024/hash/698cfaf72a208aef2e78bcac55b74328-Abstract-Conference.html) | MLSys 2024 | 数据感知 Expert 激活预测、CPU/GPU 内存协同 | 否 |
| [MoE-Infinity](https://arxiv.org/abs/2401.14361) | 2024 | request-level Expert activation trace、prefetch 与 cache | 否 |
| [ExpertFlow](https://arxiv.org/abs/2410.17954) | 2024 | 预测 Expert 激活、CPU/GPU 调度与跨 batch Token 重排 | 否 |
| [Fiddler](https://openreview.net/pdf?id=N5fVv6PZGz) | ICLR 2025 | 在 GPU 执行 attention、在 CPU/GPU 间调度 Expert 计算 | 否 |
| [KTransformers](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf) | SOSP 2025 | CPU/GPU 混合放置和高效异构算子 | 否 |

这里优化的是“权重在哪里、何时搬运、在哪类处理器执行”。预取错误可能浪费带宽或内存，但不等同于两个 Replica 对同一 Token 的重复 Expert 计算。

### 4.5 故障恢复：3 项

| 工作 | 年份/出处 | 主要方法 | 与 Hedge 的区别 |
| --- | --- | --- | --- |
| [Lazarus](https://arxiv.org/abs/2407.04656) | 2024 | 自适应 Expert replica placement、故障后重新规划并继续训练 | Replica 用于恢复能力和负载；不是每 Token 主动竞速 |
| [Surviving Partial Rank Failures in Wide Expert-Parallel MoE Inference](https://arxiv.org/abs/2605.10670) | 2026 | 修改 Expert-parallel membership、恢复丢失 Expert coverage、重新并入节点 | 故障发生后恢复执行覆盖 |
| [MoEvement](https://www.usenix.org/conference/nsdi26/presentation/gandhi) | NSDI 2026 | 稀疏增量 checkpoint、上游日志与局部恢复 | checkpoint/re-execution，不是在线 Token Hedge |

故障论文最接近本项目的 common-state 场景，但多数处理的是节点失效后的系统恢复、重新放置、checkpoint 或 Replay，而不是在失效前为少量高风险 Token 启动并发副本。

## 5. “专家复制后普遍怎么路由”

综合上述工作，Expert 被复制以后，常见执行链是：

1. Gate 先决定 Logical Expert；
2. 系统根据历史负载、预测热度、显存和拓扑决定每个 Expert 的副本数与位置；
3. Dispatcher 在该 Logical Expert 的副本中选择一个目的地；
4. 选择依据常见为轮询、Token 数、队列/未完成工作、设备局部性、通信代价或 batch min-max；
5. 一个副本执行该 Token；失败时再 Replay/failover；
6. 周期性调整副本数量、placement 或批次划分。

代表性例子是 FasterMoE/FlexMoE/CRAFT：它们复制热门 Expert 后，是把热门 Expert 的 Token 流量摊到多个副本，而不是让每个热门 Token 在所有副本上重复计算。CRAFT 还明确指出过度复制会占用显存并降低吞吐，这与本项目观察到“冗余保护有真实资源代价”在方向上相符。[CRAFT](https://proceedings.mlsys.org/paper_files/paper/2026/hash/3a7f9e485845dac27423375c934cb4db-Abstract-Conference.html)

## 6. 与通用 Request Hedging 的关系

通用 request hedging 的经典定义是：先向一个 Replica 发请求；等待一段时间仍未完成，就向另一个等价 Replica 发相同请求；接受最先完成的结果，并取消剩余请求。[The Tail at Scale](https://research.google/pubs/the-tail-at-scale/) 还讨论了 P95 延迟触发和低优先级副本，以限制额外负载。

[LÆDGE](https://www.usenix.org/conference/nsdi21/presentation/primorac) 进一步将 Hedge 约束为只有 Replica 空闲时才启动，从而保持 work-conserving。这些方法默认 Replica 可以互换，关注尾延迟与冗余工作；它们不理解 MoE 特有的 Gate、Logical Expert、Expert 热度、跨故障域 placement、Replay 脉冲和 Token 类别。

本项目可以把通用 Hedge 作为方法部件，但论文贡献必须来自 MoE 场景新增的控制问题，例如：

- Gate 固定后，只允许 same-Expert replica protection；
- Expert 热点和故障域导致 Primary/Backup 队列高度不对称；
- Replay 与 Hedge 竞争同一 Backup 容量；
- 工作预算、故障状态、deadline 和 Token class 共同决定是否值得发 Hedge；
- 多个 Token 同时自我保护会形成 Protection Storm。

## 7. 本仓库已有方法处在什么位置

| 仓库方法 | 文献类别 | 是否标准公开基线 | 当前解释 |
| --- | --- | ---: | --- |
| Fixed No-Hedge | 固定 Dispatcher + failure Replay | 是，朴素下界 | 无主动重复执行 |
| RR / JSQ / least unfinished work | replica routing | 是，经典调度基线 | Gate 后在同 Expert 副本中选一个目的地 |
| EPLB-style | Expert replication/placement | 是，placement 基线 | 决定副本数量和设备放置，不是 Token Hedge |
| LPLB-style | batch Token-to-replica assignment | 是，负载均衡基线 | 最小化 batch 负载，不重复 Token |
| P95 delayed Hedge | 通用 request hedging | 是 | 延迟超过阈值后重复执行 |
| LÆDGE | 通用 request hedging | 是，NSDI 2021 | 仅空闲 Replica 发 Hedge |
| NIIN | 手工故障感知 N/D/I 规则 | 否，项目内部 comparator | 不是公认文献算法 |
| Budgeted LÆDGE | LÆDGE + 因果工作预算 | 否，本项目 adaptation | 当前正预算点在 holdout 上均被支配 |
| requested-price candidate | 价格引导有限 Token 策略 | 否，本项目方法候选 | 尚不是经过均衡验证的 MFG |

因此，论文实验不应该只把 NIIN 当成“业界基线”。更可信的比较结构是三层：

1. **placement 层**：EPLB-style、CRAFT-style 或固定均匀复制；
2. **单执行路由层**：RR、JSQ、least-unfinished-work、LPLB-style、work-conserving dynamic Primary routing；
3. **保护层**：failover-only、P95 delayed hedge、LÆDGE、Budgeted LÆDGE、拟议的 MoE-specific selective hedge。

Tutel、Lancet、SiDA、Fiddler 等解决正交的通信或内存问题，不宜直接当成同一仿真器中的“算法臂”；它们更适合作为兼容性讨论或真实系统后端。

## 8. 对当前实验的直接解释

当前 Budgeted LÆDGE holdout 的最重要结果不是“多发 Hedge 有效”，而是：

- dynamic work-conserving No-Hedge (`delta=0`) 相对 fixed No-Hedge 大幅降低 D/F CVaR95，工作量几乎不变；
- 增加正 Hedge 预算后，D/F CVaR95、deadline miss、工作量与 Storm 没有形成更优点；
- 因而当前场景首先证明的是 **Gate 后的 same-Expert replica routing 有价值**；
- 尚未证明主动 Hedge 有价值，更没有证明 MFG 有价值。

这与文献主流并不冲突：主流 Expert replication 本来就以“把不同 Token 更好地分给副本”为主，而不是以重复同一个 Token 为主。你的结果反而提示一个更稳妥的论文故事：

> 先建立 MoE replicated-expert routing 的强基线；再证明只有在特定故障风险、队列差异与剩余工作预算条件下，选择性 Hedge 才能越过纯路由 Pareto 前沿。

如果未来选择性 Hedge 仍不能越过该前沿，论文贡献也应收缩为“fault-aware dynamic replica routing”，而不能继续以 Hedge 或 MFG 为中心。

## 9. 可以支持和不能支持的声明

当前文献统计可以支持：

- Expert replication 在 MoE 系统中很常见；
- 主流用途是热点分流、placement、局部性、吞吐或故障恢复；
- Top-k Expert 激活不等于 same-Expert request hedging；
- 在本文 30 项代表样本和精确关键词检索中，没有找到把 same-Expert same-Token first-completion Hedge 作为核心机制的 MoE 工作；
- 将通用 Hedge 与 MoE 故障、Replay、预算和 replica routing 联合建模，是一个值得验证的研究交叉点。

当前不能支持：

- “这是首个 MoE Hedge 算法”；
- “专家备份论文通常都发 Hedge”；
- “Budgeted LÆDGE 或价格策略已经优于现有方法”；
- “当前 requested-price candidate 已经是 MFG 均衡”；
- “仿真中的计算工作改善一定转化成真实 GPU wall-clock 改善”。

正式论文若要写 `first`，还需对 ACM DL、IEEE Xplore、USENIX、MLSys、ICLR、NeurIPS、arXiv 做可复现的系统检索，并由第二位研究者复核纳入/排除结果。

## 10. 推荐下一步

下一步不应立即加入更多 Hedge 或直接进入 MFG。应先把当前 `delta=0` work-conserving dynamic Primary routing 提升为正式核心 baseline，并在多 Expert、不同复制因子和随机故障下与 RR、JSQ、least unfinished work、LPLB-style 比较。

只有当强单执行路由基线确定后，才增加一个严格选择性 Hedge：它必须以可观测队列差、故障风险、deadline 紧迫性和剩余工作预算为条件，并用 paired holdout 证明在不显著增加 miss、work 和 Storm 的前提下改善 D/F tail。通过这一关后，价格协调或 Token-MFG 才有明确的必要性。

## Sources

分类框架参考了 2026 年的 MoE 推理优化综述，但所有方法判定尽量回到原论文或官方会议页面：[A Survey on Inference Optimization Techniques for Mixture of Experts Models](https://doi.org/10.1145/3794845)。通用 Hedge 定义来自 [The Tail at Scale](https://research.google/pubs/the-tail-at-scale/)；空闲副本约束来自 [LÆDGE](https://www.usenix.org/conference/nsdi21/presentation/primorac)。其余原始来源逐项链接在第 4 节矩阵中。

