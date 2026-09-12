# 原 Token 模型的工程恢复评估

日期：2026-09-06。性质：设计与只读审查；没有修改 src、tests、configs，
没有启动 Fit、Validation、Stage4b 或 MFG campaign。

## 结论

不用回滚整个项目。旧 Token 路线仍完整保留；需要以旧双副本引擎为基础，
补上在线决策、Token 偏离、动作条件预测和群体反馈求解，再单独做规模化验证。
Expert 路线的统计、因果观测和复现工程可借鉴；其玩家、目标、策略搜索及
共享算力物理假设不能直接继承为 Token MFG。

更准确的工作量判断是：恢复有限 Token 反馈博弈属于数个清晰实现切片；
获得经验证的 Token MFG 仍有新的前向模型与规模化研究工作。不是改一个
player_id 参数，也不是删除三分之二代码。当前证据不足以给出可信工时或
“完成了百分之多少”；下面按具体交付物列出范围。

完整定义见 [spec.md](spec.md)，新增设计边界见
[ADR-0020](../../docs/adr/0020-restore-token-player-model.md)。

## 1. 实際保留的四条路径

| 路径 | 当前实现 | 能说明什么 |
| --- | --- | --- |
| 早期 Token 有限动作近似 | calibration.py -> mfg_solver.py -> quota.py -> paired.py | 标量负载表、价格/策略固定点，再生成动作计划；不等于瞬态队列闭环 |
| 原双队列瞬态参照 | transient_control.py -> attribution_episode.py -> hedge_simulation.py；transient_search.py | 同预算81规则、NIIN及完整生命周期，适合恢复为物理基座 |
| 8 Expert 独立执行扩展 | scaled_experiment.py | 固定Primary、双Backup的另一物理配置，未必有Expert间策略耦合 |
| 共享Backup Expert博弈 | shared_backup.py -> expert_game.py -> game_deviations.py -> game_solver.py -> stage4_campaign.py | 当前NSNS Fit及后续精度工作；玩家和新模型不同 |

旧程序的名称里含 `mfg`，只能说明历史接口名称，不能当作当前数学模型已经
实现。Shared-Backup action callback 虽然在每个Token到达时调用，究竟谁是
玩家仍由成本函数和偏离范围决定。

## 2. 文件级复用和修改边界

| 文件／已检查入口 | 处理建议 | 具体理由／缺口 |
| --- | --- | --- |
| domain.py:ProtectionAction | 直接保留N/D/I | 原词汇未删除，Dual不是必需 |
| workload.py:WorkloadTrace、generate_workload_with_hedge | 保留原服务/到达CRN | 新增独立策略随机流，不能让决策消耗外生流 |
| hedge_simulation.py:_Engine、_handle_arrival、simulate_hedge_common_state | 最小扩展／新在线入口 | 现在构造器把actions写入所有Token，未来动作由预制表确定；需在真实到达事件调用只读观察接口 |
| attribution_episode.py:simulate_episode | 保留结果/排空；增加在线包装 | 目前仅接收actions Mapping，replica_count=2；这是原物理基座 |
| transient_control.py:plan_transient_actions | 抽出在线Ledger；保留旧函数 | 现在明确只接受TimeClassRule，拒绝callback；已有Decimal预留、请求/准入审计可复用 |
| transient_control.py:plan_transient_actions 的midpoint/final window | 随机场景需新规则 | 当前用已知failed_start计算中点和末窗cap；用于已知时刻正确，用于未知故障会泄露未来 |
| transient_objective.py:build_transient_objective、cvar95_fractional_tail | 保留系统评分 | 不能作为每个Token的私有Q；新增独立Token payoff |
| transient_search.py:evaluate_rule_bank | 保留81规则和NIIN对照 | 搜索总体J是集中式有限规则参照，不是Token BR |
| interfaces.py:PolicyContext | 新增Token观察协议 | 只有expert/state/utilization/price，缺时间/预算/队列进度等；不改变历史协议语义 |
| calibration.py:calibrate_cohort | 借鉴tagged CRN框架，重建预测接口 | 每个action plan只改变probe，其余背景Normal；背景不能表达现行保护、计时器、取消及准入组成 |
| mfg_solver.py:_soft_best_response、_work_rates、solve_mfg | 保留历史；新增Token求解器 | 旧cost不含SLO excess，价格乘Hedge+Replay；Primary固定按lambda*mean计，rho为标量阶段率；与新私有成本/预留价格不匹配 |
| quota.py:project_actions | 作为旧基线保留 | 类别份额和deficit prescription与当前pooled FIFO不同；按静态policy预生成整条计划，不能接运行中反馈 |
| paired.py:run_paired_comparison | 新建在线campaign入口 | solve -> project_actions -> simulate，控制器看不到其动作产生的实时队列；不能简单复用CLI名称 |
| shared_backup.py:ActionObservation | 借鉴因果字段及timer修复 | 已有队列顺序/执行进度/预算/phase_age；但含Expert/三副本/共享head耦合，不能直接拷贝成新物理 |
| shared_backup.py:_reschedule_pool | 不进入原物理主路径 | min(1,N*c_B/Kheads)就是新增跨Expert池；把c_B设大仍不能恢复双副本Active-Active Dispatcher |
| game_workload.py:CommonFaultPath、PublicFaultObservation | 复用设计/适配独立工具 | common/private分流、future隐藏有价值；当前生成器固定3副本、.45/Expert、cutoff360，不能整条搬过来 |
| expert_game.py:score_expert、score_population | 不用于Token私有损失 | 玩家是整个Expert，phase/class pooled CVaR按Expert聚合；仅取通用统计思想 |
| game_deviations.py:evaluate_unilateral_deviations | 必须新建Token偏离器 | 当前替换candidate_profile[deviating_expert]，改变其一整条请求流；Token偏离只改一个stable Token key |
| game_solver.py:solve_symmetric_pi256 | 保留另一模型结果 | 四上下文N/D/S/X组成256规则，不能重命名为Token最优策略 |
| stage4_campaign.py、campaign_execution.py、precision_pilot.py | 选择性复用基础设施 | call计数、CRN、hash、资源边界、cluster统计有价值；样本单位、profile和结论标签需要重设计 |
| tests/test_mfg_readiness_fixes.py | 保留缺陷修复 | Dual绑定、timer、stateful生命周期、统计方向等已修复，恢复模型不是撤销这些修复 |

源码入口行号及SHA256已冻结在 engineering-evidence.json，避免依赖父目录Git
状态推断项目版本。项目本身在父仓库呈未跟踪目录；没有可安全假定的单一
“回滚到Token版本”的提交。此次没有执行reset、checkout、删除或改分支。

## 3. 已复现：原引擎已有Token间外部性

使用ONE Expert、A/B各单服务器，D从10开始，A速度.5，B速度1：

- Token0在11到达，Primary A需要2单位工作，Hedge B需要1。
- Token1在11.1到达，Primary B需要1，动作固定Normal。
- 只把Token0动作从Normal改为Immediate，其余trace完全一致。

| Token0动作 | Token0延迟 | Token1延迟 | Token0执行工作 |
| --- | ---: | ---: | ---: |
| N | 4.0 | 1.0 | 2.0 |
| I | 1.0 | 1.9 | 3.0 |

保护自己会改变其他Token的成本，无需跨Expert池。此例只证明物理外部性，
不证明均衡、策略优越或任何大规模极限。共2次确定性scheduler调用，无拟合。

## 4. 当前最需要防止的六种错误迁移

1. 只改玩家名字：callback按Token调用，但payoff及deviation仍按Expert，
   得到的还是Expert博弈。
2. 直接重启旧mfg_solver：标量rho/Normal背景、未含实际准入的Q和预制动作表
   的闭合缺口仍在；旧代码存在不等于原研究目标已完成。
3. 混用预留和执行预算：当前q=1，旧价格项是Hedge+Replay执行工作；将二者
   相乘/相减得到的price residual不具备统一单位和资源意义。
4. 用有限配额的已准入量更新供需价：准入已被硬截断，超额需求消失；会把
   严重拒绝误报成“价格清算”。本设计显式用requested demand做pacing。
5. 每条trace选最佳Token动作：这给予未来完成/故障预知；必须先估计给定
   可观察信息的动作期望成本，再取min，报告限制策略或分箱误差。
6. 把增加Expert数或episode数当Token极限：两者都不能让K=1时单个请求占
   35.56%预算的影响消失。K规模化必须同时改变执行容量和到达率。

## 5. 实施顺序和工作量

以下是候选工作包，不自动创建执行票，也不授权新campaign。

### T1：恢复原物理上的在线入口（较小到中等）

新增TokenObservation、在线policy source及episode入口，复用两副本事件引擎。
Observation在正确到达边界构造，禁止暴露trace；保留旧actions Mapping入口。
把pooled reservation抽成在线组件。新增公共/策略记忆的episode reset合同。
注意同一时刻事件批处理中“派发前的等待状态”必须有一致定义。

验收：把NIIN等静态规则通过在线入口运行，应与旧预计划路径逐Token、逐
attempt、预算、work/waste、drain一致；同观察前缀+同policy key在隐藏未来
不同情况下仍作同一决定。测试后续在线策略确实能对queue变化作不同响应。

### T2：单Token成本与偏离（中等）

新增token_payoff.py、token_deviations.py（建议名称，尚不存在）。复用底层完整
rerun，不复用Expert profile替换。一个Token改变动作后，其他Token保持策略
函数，但队列/价格/准入导致的实际动作允许变化。Token所有running loser
完成后成本才结算完整。先做已知小例子及3-action CRN比较。

验收：只覆盖一个Token key；相同动作重放一致；证明影响其他Token；拒绝
动作按N；没有conditional/hindsight混淆。此阶段能做有限Token博弈参照。

### T3：动作条件环境预测与价格接口（较大）

先用小型独立 continuation rollouts 做“给定可观察状态”的N/D/I成本参照，
固定候选群体策略；未来随机故障按历史条件分支。模型使用实际预算投影、
队列及副本组成，不再只查询旧Normal背景rho表。实现明确的posted-price
pacing；预留价和work成本分开。模型生成数据不得来自validation。

验收：预测与独立执行的动作差异、Replay及工作项一致到预声明误差；缺失
状态/样本时fail closed；训练和验证namespace隔离；价格更新审计精确。

### T4：Token策略—群体前向闭环（较大，含研究风险）

新增token_solver.py和token_forward.py。初期可做限制状态/策略的固定点，
明确regularization、field representation及残差；循环、非收敛和高方差均保留。
不会通过换名把有限互动粒子BR当成nonatomic frozen-field BR。

验收：BR、独立forward、实际准入三者一致；残差与原始成本regret分开；
可重复因果策略；外循环索引与物理时间分开。完成此包才有“候选闭环控制器”。

### T5：同预算机制归因（中等到较大，需冻结实验协议）

NIIN、urgency、fixed delay、risk/load greedy、no-field、no-price、full均
使用同物理和FIFO reservation。全局J_sys及H/R/work安全检查沿用通用评分，
但新主成本和价格不进入旧artifact。先fit后独立validate；SE按common path。

验收：完整对照、样本精度、资源曲线与Token偏离；不以“赢No-Hedge”代替归因。
不在测试数据上追平451等历史实现次数。可证实反馈价值，仍不自动证明MFG极限。

### T6：Token MFG规模化与前向误差（最大、另立研究切片）

实现同一Expert内部A/B各K执行器、FCFS中心队列；K=1必须复现T1。
队列服务年龄和双副本关联构成带到达/退出的marked measure前向参照。
对K规模、网格/状态压缩、共同故障脉冲、条件偏离分别测误差。

验收：不能只让仿真总负载增大；保留individual service distributions；检验
预算边界与Replay峰值；无误差支持则停止MFG声明。可先研究有限Token机制，
不必把T6伪装成T1的一项小参数变更。

### 范围汇总

实际建议：扩展原事件引擎与episode包装约2处入口，拆出/新增观察和ledger，
新增Token payoff/deviation、动作预测、forward/solver、独立实验协议。
现有Expert模块无需删除。具体文件数会受隔离方式影响，以上按交付模块计，
不是承诺补丁行数。最大的工作来自此前未实现的Token MFG闭环，而非回退代码。

## 6. 当前活动与验证状态

读取时 Shared-Backup ticket15 `Status: claimed`，ADR-0019 Accepted；
本审查没有接管、停止或推进那项工作。它的结果只能回答Expert模型的精度
问题，不能作为恢复Token模型的前置条件或Token MFG证据。

历史ticket13记录475个测试通过；本轮不把这当新运行结果。本轮实际执行：

    .venv\Scripts\python.exe .scratch/token-mfg-restoration/audit_probes.py

结果：2次确定性调用，externality断言通过，probe期间源文件hash无变化。

    .venv\Scripts\python.exe -m unittest tests.test_hedge_simulation tests.test_transient_control tests.test_game_workload tests.test_mfg_readiness_fixes -q

结果：75 tests，2.292秒，OK。

    .venv\Scripts\python.exe -m mfg_hedge check --config configs/v1_minimal.json

结果：status ok；同时如实报告该历史minimal配置F基础负载1.4高于target .9。
它不是新Token参考到达率.9的配置验收，也不是新MFG实验的preflight。

没有新增生产实现，所以本轮不运行完整campaign或以全套历史测试代替模型
验证。文档完整性检查另见本票最终Progress log和review verification输出。
