# MFG 实验准备度代码审查 — 2026-09-06

Status: review completed; fixes and new experiments not executed.

## 判断

当前实现是有真实共享容量交互的 N=8、Pi256 有限博弈实验框架。
它尚不是条件均值场求解器。审查确认六个可复现的实现/统计连接问题，
另有一个正式 artifact 溯源缺口。现有467测试覆盖了不少单模块不变量，
但不能排除跨模块语义、单侧推断方向和策略生命周期错误。

本轮只新增审查脚本、报告与工作记录。生产代码、配置、版本和历史结果不变。
用7次合成 scheduler 调用（含1次故意注入失败）完成反例，未调用正式
Fit/Validation，也未生成新故障随机样本。

## P1-1：Dual 没有保留 Single 默认目的副本的 CRN 绑定

位置：`src/mfg_hedge/shared_backup.py:1262-1265`，及`:1497-1499`。

Single 向 destination_replica 发 attempt2；Dual 固定枚举 B/C=(1,2)，
将 attempt2 绑定 B、attempt3 绑定 C。因此 destination=C 时，新增第二份
Backup 同时替换了原 C 副本的服务随机数。违反 spec 第4节“first Hedge 使用
默认目的标记、与 Single 共享同一 draw”的约定。

合成反例：一个 Expert、c_B=2（不存在共享限速）、默认目的地C。
同一 trace 的 C/attempt2工作=.25、C/attempt3=5、B/attempt2=8。
Single 完成延迟=.25，Dual 完成延迟=5。两者在 C 上并没有保留同一份工作。

影响：r2 所有含 X 的候选可能使用了与声明不同的配对样本；全表排序、
最大SE和XXXX起点需在修复后重新评估。独立同分布的服务抽样不因此自动产生
总体均值偏差，但配对反事实和有限样本结果已经不同。NSNS/NSSS本身没有X，
上一轮对这两条规则的方差诊断不受这一点影响。

修复方向：Dual 的目的顺序应为(default_destination,other_destination)，
冻结其attempt ID绑定；新增destination=C的跨动作测试。不能只检查
reference与optimized一致，因为它们共用同一错误绑定。

## P1-2：单侧 max-t 上界使用了相反方向的中心化统计量

位置：`src/mfg_hedge/stage4_campaign.py:754-768`。

令 d 是真实可获偏离收益，d_hat为其估计。要构造所有 d_j 的同时上界
U_j=d_hat_j+q*SE_j，需要校准 max_j((d_j-d_hat_j)/SE_j)。
对当前采用的中心化 bootstrap 反演，这对应

    max_j((pseudo_mean_j-bootstrap_mean_j)/SE_j).

当前程序取的是 bootstrap_mean-pseudo_mean 的最大值，再将其正向加到
observed 上。这是另一侧的有向统计量。若误差联合分布对称，两者可能近似
相同；目前采用非对称经验重采样且尾部显著偏斜，不能默认这项性质。

已复现：32个伪值为(-31,1,...,1)，另加一条全0不偏离行，observed=0，
SE=1。使用同一4096重采样与seed20260906：当前实现的95%临界值=1，
上界=1，target=1.5时返回pass；改用上界反演所需的相反统计量，临界值=2。
这是方向差异的精确合成反例，不是对真实总体覆盖率的定理或测量。

影响：未执行的 Validation 统计认证；不影响 r2 Fit 的 jackknife SE。
现有测试只检查确定性、对称小样本和预期pass，无法发现两侧在偏斜时的差异。
应明确采用上界反演还是另一种区间，修正相应实现并在偏斜/重尾/相关案例下
验证方向与覆盖率；不能仅依赖变量名simultaneous_upper_bound。

参考：R boot的basic和studentized区间同样显式反演分位数方向，见
[boot源代码 basic.ci / stud.ci](https://github.com/cran/boot/blob/master/R/bootfuns.q#L1115-L1135)。
该参考支持反演关系，不证明本项目的高维小样本覆盖率。

## P1-3：通用偏离接口复用有状态策略对象，污染跨场景与跨候选实验

位置：`src/mfg_hedge/game_deviations.py:295-299`，以及偏离候选循环。

_run_profile每个场景新建的是包装器_OnlineRuleSource，但内部policy仍为
同一个对象；候选循环也继续使用原opponent对象。一个带记忆、内部RNG或
计数器的反馈策略会携带前一场景或前一候选的运行状态。

反例：策略“本episode首次决策Single，之后Normal”，同一原始trace作为
两个场景输入，得到动作[S,N]，而不是两个独立episode都为S。
这会使候选顺序、场景顺序改变所谓最佳响应，也可能人为制造非零自偏离收益。

影响：当前冻结ActionRule无状态，r2不受影响；计划接入学习策略、混合策略
或带状态的MFG反馈策略时是核心阻断项。单一source对象广播给多Expert的
接口也需明确禁止共享可变私有状态。

修复方向：不可变策略配置与episode私有状态分离；按Expert/episode创建
policy实例并绑定独立稳定私有RNG。测试self-deviation、候选顺序不变性、
相同场景重复运行、不同Expert实例隔离。

## P2-4：已经作废的 Delayed timer 在观察中仍显示 pending

位置：`src/mfg_hedge/shared_backup.py:1418-1420`。

timer_history仅根据timer_fired判断fired或pending，没有读取timer_voided
或timer_pending。若Primary先完成，timer已经失效，之后的观察仍永远pending。

反例：t=110到达并选择Delayed，Primary在110.2完成，111.5处理作废timer；
t=112下一次决策仍见((token0,'pending'),)，而实际token.timer_voided=True。

影响：依赖待发副本数量或timer历史的反馈策略会错误推断未来负载。
现有四位置固定规则不读取该字段，所以没有影响r2排序。
需明确pending/fired/voided互斥状态及可观察deadline，测试winner/failure作废路径。

## P2-5：缺失 jackknife 数据在 Validation 连接处变成异常，而非可报告结果

位置：`stage4_campaign.py:269-277`、`:699-711`和campaign调用bound的路径。

jackknife_pseudo_values遇到任何None返回空tuple；其他正常偏离行长度仍为32。
compute_simultaneous_regret_bound先检查所有行长度相同，然后才处理缺失，
于是抛出ValueError。campaign没有将此异常转换为statistics_insufficient。

合成反例：一条完整32伪值行加一条由None生成的空行，抛出
“all pseudo-value rows must share cluster observations”。现有测试分别验证
None->空tuple与正常bound，却没有把两个模块连起来测试。

影响：未来 Validation 某个delete-one池缺少cohort时，可能在昂贵运行后
异常退出，无法按承诺保留完整的统计不足结果。它不会错误通过，但会破坏
失败记录和运行结果完整性。应保留带None的固定长度行或提前统一判定。

## P2-6：baseline中途失败的调用未进入计数

位置：`stage4_campaign.py:465-470`；Validation baseline也有相同结构。

代码等整批_full_profile_runs成功才增加call_count。若前面已完成若干
场景、后面一个失败，except返回的call_count没有任何本批调用。

反例：第二次baseline调用故意抛异常，第一次正常完成；实际调用2次，
result.status=execution_failed，但result.call_count=0。

影响：不改变成功r2的49344计数，但失败重试与总预算审计会低估实际开销。
应在每次dispatch前记账，区分reserved/attempted/completed/failed，
不能依赖整批返回后补计。已有串行_CallBudget的逐调用observer可作参照。

## P2-7：正式科学 artifact 缺少代码与推断协议指纹

位置：`stage4_campaign.py:1108-1115`与成功分支manifest构建。

r2 manifest只有运行名、状态、维度/调用计划及trace指纹，没有代码快照hash、
package版本、完整物理参数、动作/attempt绑定、统计实现及阈值hash。
candidate_fingerprint也主要哈希拟合结果和traces，不能证明使用哪个版本
的scheduler与推断函数。此仓库当前项目文件仍显示为untracked，因此只加
一个父仓库git revision也不足以解决问题。

影响：修复Dual或上界之后，相同trace仍可能被错误描述为“相同实验”；
将来无法单靠artifact确认历史结果的精确语义。
应在调度前冻结源码内容hash和全部科学配置/协议hash，结果引用该freeze。
这是直接检查manifest/code得到的缺口，不计入六个合成反例。

## 对已有 Fit 的影响边界

- 保留fit_no_candidate，当前不能提升为candidate/Nash/MFG。
- 不能用本次统计上界问题解释上轮高SE：该函数根本未在r2执行。
- 上轮NSNS/NSSS配对方差的解释仍有逐请求支持，两条规则均不使用Dual。
- 256表含X的部分存在CRN合同差异；修复后应以新版本/新run重新评估，
  旧artifact保留为旧实现证据，不原地改写。
- 反馈策略生命周期和timer观察问题在当前无状态规则中没有触发。
- 成功r2没有本次故意注入的baseline异常，计数发现不推翻其成功调用计数。

## 距离 MFG 尚缺的建模和实验层

这些不是靠修几个bug即可得到的功能：

1. **条件群体演化模型尚未实现。** 当前game_solver的真实评估器和stage4
   都逐候选重跑N=8全部玩家；environment_fingerprint是一个运行摘要，
   不是m_t=Law(Y_t|F_t^0)的前向解。计划中的conditional_field模块不存在，
   也没有另一实现承担条件分布闭合、粒子数误差或分布残差验证。
2. **nonatomic最佳响应尚未实现。** 需要在给定、适应于公共历史的m下求
   代表性Expert的因果最佳响应，再让实施后的策略产生新m。目前做的是有限
   游戏单边偏离，自己的动作会改变其他玩家速度；应保留它作为验收基准，
   不能把它的纯策略固定点改名为MFG均衡。
3. **反馈/风险策略契约需要完成。** Pi256只用时间块和类别，不使用队列、
   预算或k反馈。当前CVaR是事前分布目标；未来Bellman/HJB不能直接当逐步
   加性奖励使用。需明确固定阈值/风险状态、可观测状态与私有策略初始化。
4. **有限规模验证尚未完成。** 需要同一冻结limit策略在N=8/16/32/64运行，
   按公共路径条件比较群体分布，并独立复核各Expert偏离收益。不能在每个N
   分别重新搜索一条不同规则后，将不同结果解释为一个MFG的收敛。
5. **部署质量与归因实验尚未覆盖。** Stage4的pass最多表示受限偏离收益
   统计门，不等于H/R安全、工作比、公平性或优于NSSN。还需相同预算/物理
   机制下的No-Hedge、冻结规则、集中控制、MFG及必要消融对照。
6. **精度预算尚不匹配目标。** 上轮16公共路径、2人口与F尾部稀疏性已暴露；
   不能通过重复起点、增加N、删除不利路径或临时放松阈值替代新精度设计。

## 建议实施顺序

第一步做独立修复切片：Dual CRN绑定、单侧统计方向与缺失值连接、策略
生命周期和timer观察，并补跨模块反例；同时冻结新的实验溯源字段。

第二步在有限游戏层验证统计门和精度方案；保持旧Fit为负结果。

第三步实现条件均值场前向模型与代表玩家最佳响应，先做模型预测误差验证。

第四步冻结一个候选策略，再运行有限N的独立偏离、规模、安全和价值实验。
无需为了称作MFG先引入价格；物理共享已产生非退化交互，价格若引入须有
独立定义和对照。

## Verification 与限制

反例脚本：`.scratch/shared-backup-game/review_readiness_probes.py`。
证据：`artifacts/mfg-readiness-review-20260906/probes.json`。
六个反例全部复现，7次合成scheduler调用中1次故意失败；源码/配置hash不变。
完整回归与配置检查结果记录在ticket12。未改生产代码或启动正式实验。

这是针对相关数据流的定向审查，不是对整个仓库不存在其他缺陷的保证。
没有测量真实Validation覆盖率、证明MFG极限或评估实际部署资源拓扑。
