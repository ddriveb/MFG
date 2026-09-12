# Token 保护博弈的演化与闭环推导：第一稿

日期：2026-09-07。性质：数学设计推导，供审阅；不是实验报告或已证明的 MFG。

依据：Accepted ADR-0020 的 Token/T1 物理过程与 Accepted ADR-0021 的
`token_initial_price_v0`。本文不改变已接受 ADR，不接受旧 spec 中的动态价格、
扩展效用或 K-server 假设。第 9 节之后的规模族是明确标出的候选扩展。
工作票：[06](issues/06-derive-token-evolution-and-closure.md)。

用户随后明确允许按适用性选择离散或连续方程。本文据此优先采用第15—17节的
**事件步转移核 + Bellman** 作为下一理论实现路线；第2—14节保留为相同过程的
生成元/守恒参照。事件步离散不等于状态有限；有限状态近似另须给出误差。

## 1. 先锁定对象与推导顺序

玩家是一个逻辑 Token，Primary/Hedge/Replay 都属于同一玩家。
一个 Expert 的 A/B Replica 是资源。Gate、Primary Dispatcher 外生固定。
Token 在到达时只请求一次 a∈{N,D,I}；随后不再优化，D 定时器和 Replay 是机制。

先推导精确有限队列过程及其条件成本，再讨论它能否有一个非退化的均值场极限。
“写出生成元”“定义了一个方程系统”“方程有唯一解”“存在均衡”“有限系统趋近
该均衡”是五个不同层次。本文不把最后三项当成已经证明。

第一版效用固定为：

\[
C_i=L_i+\gamma_{k_i}R_i+p(W_i^H+W_i^R),\qquad p\geq0.\tag{1}
\]

p 是整局事先固定的外生执行工作价格，没有价格反馈或清算方程。
`token_runtime_adr0009_v1` 与 `token_extended_reservation_v1` 保留独立身份，
本推导不把它们的固定工作、waste、SLO 或 reservation 价格项混入式 (1)。

## 2. 概率输入、信息与物理状态

有限基准：初始空队列；到达率 λ=.9，出生区间 [0,320)；类别权重 (.8,.2)；
A/B 各一台 FCFS 单服务器。非 F 按 Token 序号交替 Primary，F 到达用 B。
A 速度 H/R 为1、D 为.5、F 为0；B 始终为1。
已知日程基准为 D=100、F=200、R=220；延迟 τ 是健康服务分布 .9 分位数。
这些日程参数属于已声明场景；不能把随机故障扩展的预抽样未来当成观测。

各 attempt 的需求 S 独立同分布，lognormal 均值1、CV=.5。因此

\[
\sigma^2=\log(1.25),\quad\mu=-\tfrac12\log(1.25),\quad
\bar F(e)=P(S>e),\quad h(e)=f(e)/\bar F(e).\tag{2}
\]

端点按连续延拓定义：\(\bar F(0)=1, h(0)=0\)，覆盖刚开始执行的状态。

随机过程意义上的推导积分所有未观测随机量。CRN 是把不同动作的完整过程
耦合到相同输入上的计算约定，不意味着玩家知道已抽样的未来。

一个 Token 的标记状态可写成

\[
x_i=(k_i,t_i,r_i,a_i,\bar a_i,w_i,R_i,d_i,
       \{\ell,r_{i\ell},s_{i\ell},\kappa_{i\ell},e_{i\ell}\}_{\ell=P,H,R}).\tag{3}
\]

- w_i∈{0,1}：winner 是否产生；winner 角色另外记录，不能把 Primary 的 ID=0
  当作尚无 winner。R_i：Replay 是否创建过。
- d_i：D 的截止时刻与 pending/fired/void 状态。
- s：not-created/queued/running/completed/failed/cancelled。
- κ：入队的全序优先级；e：已执行工作，不是服务需求或真实剩余量。
- 已终结 attempt 的执行量保留；完整结算后可以把 Token 送入墓地状态 ∂。

精确全局物理状态另外含所有 S、两条有序队列、运行对象、事件同刻顺序、
预算窗口和余额、下一 Token 序号、时钟/阶段，以及策略需要的历史与运行内存。
事件表也保留已失效但未来仍会产生公开 `timer:voided` 日志的定时器：其 Token
即使已经结算，该日志仍可能影响后续读取历史的策略。不能仅因物理净效果为0
就从精确信息过程删掉它。过期完成事件按引擎 generation 规则忽略。
这显然不是只含两个队长的低维状态。

为方便写生成元，可以积分掉隐藏 S，使用“全体标记状态+已执行进度+历史”的
增广状态 Y。服务完成强度为

\[
\alpha_{i\ell}(Y)=1_{\{s_{i\ell}=running\}}v_{r_{i\ell}}(Z)h(e_{i\ell}),\qquad
P(S-e>u\mid S>e)=\frac{\bar F(e+u)}{\bar F(e)}.\tag{4}
\]

在这个积分掉需求的表示中，不保留由隐藏 S 算出的未来完成时间；完成用强度
(4)，而已知 timer 截止仍保留。这是在 iid 服务、非预知调度假设下与隐藏需求模型等分布的表示，并非把
lognormal 改成指数分布。固定 CRN 路径仍可用 S 和确定性完成边界计算。
若未来引入相关需求或透露需求的特征，式 (4) 必须改成条件 hazard/联合信念。

**信息边界：** Y 是分析状态，不是传给策略的输入。令 O_i 是 T1 实际提供的
观测及策略允许保留的记忆，动作必须对它可测。当前 TokenObservation 没有
直接提供 e、需求 S、真实剩余量，也没有完整的群体分布。不能把分析变量直接
放进策略冒充现有信息集。需要更丰富观测时必须另作设计。

定义到达决策时的条件信念（按到达事件条件化，而非用一个概率零的固定时刻事件）

\[
\nu_i(dy\mid O_i)=\operatorname{Law}(Y_{t_i}^{pre}\in dy\mid O_i,
                                      \text{该 Token 到达}).\tag{5}
\]

这是从概率输入、过去策略和观测投影导出的条件分布，不是另一张自由拟合的表。
若用简化状态代替 O 或 ν，需证明其充分性，否则只能声明受限信息/近似策略。
保留任意历史和控制器内存使过程闭合，但可能是无限维；本文没有声称找到了
最小有限维信息状态。

## 3. 准入、漂移和所有事件映射

令 q(N)=0、q(D)=q(I)=1。D-relative 半开窗口宽25，名义额度 c=2.8125，
每次批准预留1、不退款。只在 Z=D 且 Primary=A 时允许保护。定义

\[
\Gamma(a,Y,k,r)=
\begin{cases}
a,&a\in\{D,I\},\ Z=D,\ r=A,\ B\geq1,\\
N,&\text{其他情况},
\end{cases}\qquad
B^+=B-q(\Gamma(a,Y,k,r)).\tag{6}
\]

N 本来就映射为 N；拒绝的 D/I 保留 requested 标签，但物理上按 N 继续。
窗口打开余额重置 c；F 禁用准入。引擎在首个相关到达时懒重置，与独立窗口
边界在到达前重置对决策等价；其实现不是每窗独立计划未来 Token。

无事件区间：绝对时钟 t 作为外部时间，阶段 age 以1增长；每个运行 attempt 的 e 以 v_r(Z) 增长；
排队 attempt 的 e 不变。timer 剩余时间以1减少。记状态坐标漂移为 D，
它不含对外部绝对时间 t 的导数。

| 映射 | 完整操作；动作只在第一行由策略产生 |
| --- | --- |
| A(y,k,a)：到达 | 确定 Primary；产生当下观测；原策略调用一次并更新其内存；请求经 Γ；扣额度；Primary 先入队；I 同时 enqueue Hedge；D 设置 t_i+τ 定时器；最后按批次 dispatch |
| C_{iℓ}：完成 | 移除完成 attempt；无 winner 则设 winner、取消本 Token 所有 queued 同伴并 void timer；其他 running 同伴保留；已有 winner 则本次记 loser；符合条件时结算；最后 dispatch |
| T_i：timer | 若 winner 已有或 pending 已取消则无物理动作；否则清除 pending；F 时 suppress，否则 enqueue Hedge；无第二次策略调用或收费；最后 dispatch |
| F：故障 | 删除 A 的 running/queued，保留已执行量；依 running Primary 优先、再 A 队列顺序扫描丢失 Primary；仅无 winner、无 live Hedge 且未 Replay 的 Token 创建 B Replay，设置 R=1 并 void timer；最后 dispatch |
| G_D/G_R：阶段 | D 调整 A 后续速度；R 恢复可服务状态，不恢复已丢弃副本；阶段 age 归零 |
| W：窗口 | 进入新 D 窗口，余额为 c；旧余额作废，不因 timer void/cancel 退款 |
| Ψ：dispatch | 每个可服务且空闲的 Replica 从 live 队首开始服务；取消墓碑跳过；F 的 A 不开始新服务 |

同一时刻组合顺序是 state/fault/Replay → completion → timer → arrival → dispatch。
同类事件保持引擎 sequence 次序。同刻不能每处理一个事件就提前 dispatch，
表中“最后”指整个同刻批次末尾。连续随机输入下多数平局概率为零，但确定性
测试、阶段边界及 Replay 批次仍必须规定。故障与完成同刻时故障先处理；
已经执行到需求量也不能凭另一套完成优先规则暗自更换 winner。

R_i 在 **创建 Replay** 时变1，不在开始/完成 Replay 时变1。
只有本 Token 已有 winner、所有已创建 attempt 均物理终结且无有效 pending
动作时才能进入 ∂；墓地的值函数为0。无效旧 timer 事件不延长成本结算时间。

## 4. 两种守恒检查

真实隐藏需求表示下，对 Replica r 的未完成工作 V_r：

\[
V_r(t)=V_r(0)+A_r^{work}(t)-E_r^{work}(t)
                  -C_r^{remaining}(t)-F_r^{remaining}(t).\tag{7}
\]

A 包含 P/H/R 入队需求；E 包含全部已执行工作（包括 running loser）；C/F 只
移除取消/故障时尚未执行的量。不能把已经执行的失败工作再从 E 中删掉。

对未结算 Token 数 M：

\[
M(t)=M(0)+N_{arrival}(t)-N_{settled}(t).\tag{8}
\]

winner 数不等于 settled 数；复制与 Replay 不增加玩家数。
取 (7)、(8) 是两个独立检查：工作守恒和玩家质量守恒。

## 5. 精确有限系统的前向方程

固定所有后续 Token 的因果策略函数 π，包含已声明的内存/随机化规则。
在非确定性边界处，对测试函数 f：

\[
\begin{aligned}
\mathcal A_t^\pi f(y)
={}&D f(y)
 +\sum_{i,\ell}\alpha_{i\ell}(y)
                 [f(C_{i\ell}y)-f(y)]\\
 &+\lambda1_{\{t<320\}}\sum_k w_k
       \sum_{a\in\{N,D,I\}}\pi(a\mid O(y,k))
                 [f(A(y,k,a))-f(y)]. \tag{9}
\end{aligned}
\]

此处 A/C 包含事件后的合法 dispatch；式 (9) 用于没有同刻其他边界的区间。
若策略有隐私内存或抽样，A 是包含其一次调用/状态更新的转移核，f(Ay) 表示
对核积分，不能假定调用只改变一个枚举值。在 (9) 的求和中，该核的内存更新
条件于本次推荐恰为 a；不能乘过 π(a|O) 后再独立抽一次推荐。目标覆盖核 A_i^a
则对原推荐及其内存更新积分，再把送入 Γ 的请求换成 a。下一 Token 序号使交替 Dispatcher
在有限系统中精确，而非逐 Token 独立掷硬币。

设 P_t 是 Y 的概率律。对带有限变差边界项的测试函数，有弱形式

\[
\begin{aligned}
\langle P_t,f\rangle-\langle P_0,f\rangle
={}&\int_0^t\langle P_s,\mathcal A_s^\pi f\rangle ds\\
 &+E\!\sum_{s\leq t:\;timer/window/scheduled\ phase}
              [f(G_sY_{s-})-f(Y_{s-})]. \tag{10}
\end{aligned}
\]

G_s 按第3节组合所有该时刻的边界；timer 的时刻由过去到达和动作决定。
式 (10) 的边界项由明确的 timer 坐标/边界命中定义，不能再另加一项
“平均 timer 强度”重复计数。在光滑内部简写为 ∂_t P=(A_t^π)^*P，边界
使用 G 的推前。P 是完整联合律，故没有未说明的队列独立性假设。

在 iid 服务和有限到达窗口下，有限系统过程本身良定义：有限时间内到达数
几乎必然有限；每个 Token 至多三个 attempt；每个需求几乎必然有限；恢复后
可服务且无新故障、B 始终可服务，因此全局物理 drain 几乎必然完成。
用所有到达 Token 的潜在 P/H/R 需求之和作上界，可得有限均值的 drain/成本。
这给定 π 定义了概率演化和有限期望 Q；不等于已证明存在某个均衡 π*。

## 6. 把初始效用转换成后向方程

令 T_i^win 是 winner 时刻，T_i^set 是物理结算时刻，N_i^R 是 Replay 创建
计数。由执行工作定义，逐条样本路径恒等：

\[
\begin{aligned}
C_i={}&\int_{t_i}^{T_i^{set}}c_i(Y_s)ds
               +\gamma_{k_i}\int_{t_i}^{T_i^{set}}dN_i^R(s),\\
c_i(y)={}&1_{\{w_i=0\}}+
p\sum_{\ell\in\{H,R\}}1_{\{s_{i\ell}=running\}}v_{r_{i\ell}}(Z).
\end{aligned}\tag{11}
\]

注意：Primary running loser 仍占容量，但式 (1) 不对其执行收费；H/R running
loser 同时占容量并继续收费。完成 winner 时不再重复加一笔 L；Replay 创建
时收 γ 后也不在结算重复收费。p 乘 work，不乘占用墙钟时间。

给定其他玩家的策略函数，V_i^π(t,y) 是已提交动作的该 Token 从现在至结算的
剩余期望成本。在原已知日程的区间内部：

\[
0=\partial_t V_i^\pi+\mathcal A_t^\pi V_i^\pi+c_i,
\qquad V_i^\pi(t,y)=0\quad(i\text{ 已结算}).\tag{12}
\]

这里 D 不含已由 ∂_t 写出的绝对时钟导数，避免时间漂移重复计算。阶段 age、
执行进度和 timer 坐标仍属于 D。

任意确定性边界使用同一物理映射：

\[
V_i^\pi(t-,y)=\gamma_{k_i}\Delta R_i(G_t,y)
                         +V_i^\pi(t+,G_ty).\tag{13}
\]

timer/window/普通完成的 ΔR=0；故障才可能创建 Replay。
离开出生区间 t=320 只删除新到达项；**不能设置 V(320)=0**。值函数的定义是
期望直到结算的累计成本，即满足相应可积性条件的非负成本解；不能给一个任意
远期时间强行零截断，尤其 lognormal 在无限支撑下没有统一的有限 drain 上界。

在到达前，固定其他 Token 的策略函数，并只覆盖目标这次 requested action：

\[
Q_i^\pi(a\mid O_i)=\int V_i^\pi(t_i+,A_i^a y)\nu_i(dy\mid O_i),\qquad
V_{entry}^\pi(O_i)=\min_{a\in\{N,D,I\}}Q_i^\pi(a\mid O_i).\tag{14}
\]

A_i^a 仍调用原策略一次，再覆盖返回给 Γ 的请求；保留其内存更新。
其他 Token 对新状态作反应，不能冻结它们的动作轨迹。准入费在初始模型为0，
预留不是货币效用项；初始模型的 p 只在执行 H/R 时通过式 (11) 计费。

因此该模型准确的“后向最优性”是 **线性继续价值 + 到达处 Bellman 最小化**。
在每个服务事件后都写 min_a 会把模型改成重复决策/可撤销控制游戏。
另外 min 必须在条件期望之后：min_a E[C^a|O]，不能换成 E[min_a C^a|O]。

式 (9)—(14) 对给定策略给出了精确有限的前向/成本系统。若进一步求解
supp π_i*(·|O_i)⊆argmin_a Q_i^{π*}(a|O_i)，得到的是有限 Token 博弈的
一致性条件，而不是仅凭记号就得到一个 MFG。

## 7. 公共随机故障如何加入，且不泄漏未来

原已知日程先用 (13)。若采用另行提案的随机阶段长度，则加入公共阶段年龄 s：
D duration∼U[75,125]、F duration∼U[10,30] 时 hazard 分别为
0（下界前）和1/(上界−s)（上下界间）；H→D 仍为已知100时刻。
只在这个扩展中使用下式。设 N^0 计数公共跳变，G_z 同时改变所有玩家和资源。

对公共历史 F_t^0=σ(Z_u:u≤t) 条件化的完整联合律 P_t^0，在公共跳变间按
(10) 演化；公共跳变处：

\[
P_t^0=(G_z)_\#P_{t-}^0.\tag{15}
\]

等价地写成非补偿计数形式：

\[
d\langle P_t^0,f\rangle=
\langle P_t^0,\mathcal A_t^\pi f\rangle dt
+\text{timer/window 边界项}
+\langle P_{t-}^0,f\circ G_z-f\rangle dN_t^0.\tag{16}
\]

假设公共故障外生且独立于私有负载；若故障强度依赖未观测负载，还要额外
过滤，不能直接套 (16)。式 (16) 不再加相同的 q_z dt 跳变漂移。

事前成本后向方程必须平均未来尚未发生的公共跳变，故式 (12) 增加

\[
q_z(s)\{\gamma_{k_i}\Delta R_i(G_z,y)+V_i(t,G_zy)-V_i(t,y)\}.\tag{17}
\]

条件前向沿真实公共历史推进；后向在当前信息下考虑所有未来分支。先固定
整条未来故障路径再让玩家最优化，会给玩家未来信息，除非本来就是已知日程。

## 8. 为什么小状态的 FPK 暂时不能等同于原模型

下面是结构性反例，不依赖统计实验：

1. 同队长、同当前总工作，队首先后是短任务/长任务时，下一个开始服务时刻不同。
2. 即使各队列长度和需求分布相同，改变 A/B 上 attempt 的 Token 配对，A 完成
   后在 B 取消的是队首还是队尾不同，故后续等待不同。
3. 同队长、同已执行工作均值，运行年龄分布不同，则 ∑v h(e) 一般不同。
   lognormal hazard 非常数，不能只保留一个均值。
4. 同物理队列，B=.9 与 B=1.1 时，同一个 I 请求分别拒绝与批准。

因此 ρ、Hedge 比例、两个队长的组合不是精确 Markov 状态。
把全局联合律 P 边缘化成 Token 律时，会出现“该 Token 在给定群体状态下何时
开始服务”等条件关联；这正是要解决的闭合问题。

## 9. 候选规模族：明确是新增假设

为了让单个 Token 对归一化资源的影响变小，考察候选 K 族：A/B 每域 K 个
同速 executor、每域一个共享 FCFS 队列；λ_K=.9K；每窗 cap_K=2.8125K；
每 Token 需求、τ、效用、阶段时刻不变。K=1 返回有限基准。
固定 Gate 和非 F 的交替 Primary；不增加 Expert 玩家。

允许的 π^K 必须有明确的归一化观测极限：队长/余额除 K，保留类别、Primary、
阶段、年龄等。不能允许策略任意依赖原始 Token ID 的数论性质、未缩放队长
阈值或隐藏种子后仍默认存在同一个极限策略。第一候选策略类限定为声明的当前
宏观可观测特征的反馈；不直接依赖任意逐条审计日志。若要保留原 T1 的全部
历史依赖能力，必须另保留这些历史的极限状态，尤其已结算 Token 的失效 timer
仍会产生公开日志。仅有 live m 并不闭合这种广义策略类。这是策略类限制提案，
不是宣称当前所有 TokenActionSource 都满足它。交替 Primary 的半半流量极限
也须在这个非预知策略/负载框架中证明，不能改称有限系统独立路由。

K>1 同刻 Replay 新定义稳定顺序：丢失的 running Primary 按原始到达顺序，
然后丢失的 queued Primary 按 A 的 FCFS 顺序，追加到既有 B 队列之后。
K=1 与原顺序一致；这是候选扩展的 tie 约定，不是现有引擎已经实现的能力。

初始 m_0=0；定义未结算的非归一化质量

\[
m_t^K=K^{-1}\sum_{i:T_i^{set}>t\geq t_i}\delta_{x_i(t)}.\tag{18}
\]

m 的总质量不是1。极限对象若存在，是条件于公共历史的 cohort 强度测度，
不是从所有历史出生 Token 中挑一个后得到的普通概率律。

**原参数下的非退化检查不能跳过。** 平滑 D 流量中，即使所有可选 Token
都 Immediate，A 的 offered work=.45<.5，B 的 offered work≤.9<1；所以
不能靠“无限 K 后 D 一定拥堵”证明耦合。这一计算只针对平滑到达，不能当成
忽略 Delayed 的延迟流量和 Replay 脉冲的路径上界。许多单服务器随机等待会
在 pooling 极限消失；可能留下的交互包括稀缺准入和 F 的 Replay 脉冲。
必须证明这些交互在目标极限仍影响 Q，且有限 K=1 的决策相关机制被保留。

## 10. 候选均值场前向方程：把 FCFS 也作为方程写出

采用式 (3) 的整 Token 配对标记，保留 service age、两个副本状态、timer，
并为每个 queued attempt 保留全序 κ。故障批次若在一个粗 κ 上产生原子，
再加入批内顺序坐标；不允许丢掉坐标后随机插队。以下限定在这种标记/优先级
有极限且策略可测的候选类。为看清闭合，不把 K1 的观测自动升级为可见 m。

定义每域运行/排队质量和完成率：

\[
n_r=\int\!\sum_\ell1_{run,r}(x)\,m(dx),\quad
q_r=\int\!\sum_\ell1_{queue,r}(x)\,m(dx),\quad
d_r=\int\!\sum_\ell1_{run,r}v_rh(e_\ell)\,m(dx).\tag{19}
\]

I_r(dt,dx,dℓ) 是开始服务的质量，H(dt,dx) 是 pending timer 到期的质量。
它们不是可以自由校准的系数。H 是 timer 剩余时间以速度−1输运在0的出流，
仅限届时仍 pending 的状态；等价地将出生时承诺的 t_i+τ 推到截止时刻，并
先应用此前的 winner/fault 取消，再取幸存 timer 的出流。

设 C_ℓ^0、T^0、S_{rℓ} 分别为 Token 内部完成、timer 创建、开始服务映射，
**不包括其他 Token 的 dispatch**。在公共跳变之间，对 φ(∂)=0：

\[
\begin{aligned}
d\langle m,\phi\rangle={}&\langle m,D_x\phi\rangle dt\\
 &+\int\sum_\ell1_{run}v_rh(e_\ell)
                    [\phi(C_\ell^0x)-\phi(x)]m(dx)dt\\
 &+\lambda1_{t<320}\sum_{k,r}w_k\eta_z(r)
        \sum_a\pi(a\mid o(t,k,r,m,b))\phi(A_{k,r,\Gamma_b(a)}^0)dt\\
 &+\sum_r\int[\phi(S_{r\ell}x)-\phi(x)]I_r(dt,dx,d\ell)\\
 &+\int[\phi(T^0x)-\phi(x)]H(dt,dx).\tag{20}
\end{aligned}
\]

η_z 在非 F 为(.5,.5)，F 为(0,1)。A^0 是先入队的 newborn，开始服务只经 I，
避免到达项和 dispatch 项重复计数。若策略观察私有信号/公开历史摘要，出生项
还对该信号的指定条件核积分；o 只能是原观测的声明极限，不能包含隐藏需求。
整 Token 的 C^0 同时取消自己的 queued 同伴，避免错误的两队独立边缘模型。

**I 的 FCFS 约束如下。** 记 μ_r^q(dκ) 为当前 queued 优先级质量。
在一个原子 dispatch 时刻，先做该时刻所有非 dispatch 事件，得到带 ~ 的状态。
可开始的总质量

\[
\Delta I_r=\min\{\widetilde q_r,(1-\widetilde n_r)_+\},\quad v_r>0.\tag{21}
\]

令 K_r^* 为最小分位点，使 μ_r^q((−∞,K_r^*])≥ΔI_r；选取全部较小优先级，
再取边界原子所需质量。原子内按保留的批内顺序选取；不能仅按类别比例随意分配。
I_r(dt,dx,dℓ) 是该选取在完整 Token 标记上的提升，S 将所选 queue→running。

无原子的区间要求

\[
dn_r=dI_r-d_rdt,\quad0\leq n_r\leq1,\quad q_r(1-n_r)=0,\tag{22}
\]

且开始服务质量始终从当前未取消队列最老优先级取。具体为：有正队列时 n_r=1、
dI_r=d_rdt；有空闲且无队列时，所有进入队列且未被同刻取消的质量立即开始，
直至容量达到1；空队列且满容量的接触点使用非闲置/容量约束选择可接受流量，
其余形成队列。F 的 A 设 n_A=q_A=0、I_A=0，清除操作发生在故障跳变处。
这是一组带 FCFS 分位选择的输运/反射边界条件，不是一个待拟合的平均等待率。

规则已说明 I 的物理选择，但 (20)—(22) 的存在、唯一性、接触点连续性与
Replay 原子的弱极限仍须证明。仅列出容量互补式而没有优先级选择是不够的；
反过来列出这些选择规则也不能直接冒充已经证明了解的唯一性。

## 11. 预算方程、公共跳变与出生/退出边界

令 b=B/K。每个窗口起点 t_j 的 b=c。假定出生流无时间原子，归一化 eligible
请求率为

\[
r_t=\lambda1_{\{t<320,Z=D\}}\eta_z(A)
         \sum_kw_k[\pi(D\mid o_{k,A})+\pi(I\mid o_{k,A})].\tag{23}
\]

在窗口内：

\[
b_t=\left[c-\int_{t_j}^t r_sds\right]_+,\qquad
dU_t=1_{\{b_t>0\}}r_tdt.\tag{24}
\]

Γ_b 对 b>0 的 eligible D/I 准入，b=0 后按 N；π 与 r 依赖实际 b，故 (24)
是联立关系。名义 cap 在下窗重置；故障禁用后 r=0，余额记账与可用性分开。
边界单点不影响无原子出生积分。若允许批量出生，必须再指定按到达全序分配
的质量截断；不能将 (24) 的微分公式直接用于原子。
有限 K 仍必须 B≥1 才批准，不能在 K1 用 b>0 替换 indivisible Ledger。

公共故障时先全体应用 F^0（含 Replay 标记和队列优先级），再按 (21) dispatch：

\[
m_t=\Psi_\#(F^0_\#m_{t-}),\tag{25}
\]

推前包含进入墓地的质量删除。Ψ 依赖变换后的整体队列测度，不能拆成各 Token
彼此独立的随机选择。恢复只改变速度和资源可用性，不复活失败 attempt。
如果 (20) 中 I 已包含此次 dispatch 原子，则 (25) 只写 F^0 推前；二者选一，
不重复计数。

令 φ 为 live 状态的常数1（在 ∂ 为0），由 (20) 得

\[
d\|m_t\|=\lambda1_{t<320}dt-dS_t^{settled}.\tag{26}
\]

Hedge/Replay、queue→run 不创造 Token 质量；win 但 loser 未结束不退出。
初始为空，t≥320 删除出生项，继续所有输运/跳变直到成本结算。
无限支持服务的极限质量可能只在 t→∞趋0，不能默认某个固定 drain_end 全部为0。

## 12. 给定场的 tagged 后向方程和入口最优性

令 E 包含公共阶段/年龄、整个标记场 m、b、FCFS 前沿和策略所需的公共记忆。
给定一个因果的候选场演化，tagged Token 在其环境中不改变 m,b 的极限轨迹。
其排队 attempt 依入队 κ 等待前沿到达；开始服务是由 (21)—(22) 决定的
边界命中，不凭空引入一个指数服务启动率。含批内 rank 的完整状态决定原子
内何时开始；若只有粗 rank，则须积分其声明的条件分布。

对已提交动作的 tagged Token，在无 timer/dispatch/窗口边界的内部：

\[
\begin{aligned}
0={}&\partial_t v+\mathcal D_E^\pi v+D_xv+c(x,E)\\
 &+\sum_\ell1_{run}v_r(Z)h(e_\ell)
                 [v(t,C_\ell^0x,E)-v(t,x,E)]\\
 &+q_z(s)\{\gamma_k\Delta R(x)+v(t,F^0x,G^EE)-v(t,x,E)\}.
\tag{27}
\end{aligned}
\]

已知日程版本删掉最后一行，改用已知时刻 value matching。
c(x,E) 是 (11) 的初始模型运行成本。
D_E^π 是沿 (20)—(24) 中连续环境流的方向导数，**不含已单列的公共故障**；
式 (27) 只在这些流/导数存在的区间解释为经典方程。一般情况下以相同 tagged
过程的条件期望/动态规划积分关系定义 v，不假装非光滑测度一定可微。

边界：timer 用 T^0；开始服务用 S；窗口用 b 重置；公共故障同时更新 x,E 并
收一次 γΔR；结算 v=0；winner 后 v 一般不为0；到达截止无零终值。
这使 tagged 预测和 population forward 共用同一取消/Replay/FCFS/准入机制。

令 ν_{entry}(dx,dE|o) 为当前信息的指定条件信念，出生动作建立 A_{Γ_b(a)}^0：

\[
Q(a\mid o;m,b,p)=E[v(t+,A_{\Gamma_b(a)}^0,E^+)\mid o],\quad
\operatorname{supp}\pi^*(\cdot\mid o)\subseteq\arg\min_a Q(a\mid o;m^*,b^*,p).
\tag{28}
\]

E^+ 不因 tagged 一次动作改变宏观场；若它导致即时开始服务，由相同前沿规则
处理。有限系统的单次动作仍能改变真实队列/余额，不能把本段的忽略反馈用于
T2 有限偏离。全场只有投影被观测时，式 (28) 仍必须对其条件信念平均。

完整候选一致性是：给定 π*，(20)—(26) 产生 m*,b* 和前沿；同一环境经
(27)—(28) 产生的最优策略支撑包含 π*；出生/观察条件核也来自该前向系统。
也可逐出生 cohort (t,k,r) 写 m_t 为过去出生强度乘以各 cohort 在当前时刻
尚未结算的条件状态概率之积分。这比 m=Law(X) 更准确地保留持续出生和退出。

公共随机扩展中，一致性沿公共历史定义；后向使用未来公共分支的条件期望，
绝不为每条已揭示未来路径分别选一套事后最优 π。
求每个 Token 的私人最优与求整个系统的总成本最小化不同；式 (28) 没有把
其对其他 Token 的全部损失直接加进自己的效用。

## 13. 本轮得到什么，理论门槛还剩什么

| 层次 | 本轮状态 |
| --- | --- |
| 玩家/控制/初始效用 | 对齐 accepted Token 模型，没有换成 Expert 玩家 |
| 精确有限物理过程 | 状态、事件、准入与结算已定义，可写 (9)—(10) |
| 精确有限条件成本 | (11)—(14)；取条件期望后再最小化 |
| 当前 T1 最小充分观测 | 未证明；精确表示保留信念/历史，不能直接删掉 |
| K 规模族 | 明确候选扩展，未作为 accepted 工程模型 |
| 候选前向闭合规则 | (20)—(26)，FCFS 分位/非闲置与 timer 出流已明写 |
| 前向解的存在/唯一性 | 尚未证明；重点是相关取消、Replay 原子、预算接触点 |
| 均衡存在/选择 | 尚未证明；有限动作存在最小值不代表固定点存在 |
| 有限 K 到候选场 | 尚未证明，K1 误差尤其未知 |
| MFG 实验/PPO | 未启动；本文件不产生实验或均衡结论 |

下一理论切片应先处理 **给定可接受策略时的前向解与 tagged 入服前沿**，再证明
成本映射可积且随候选场适当连续，并确定均衡存在需要的条件。硬准入可能导致
不连续；不得直接引用一个连续映射不动点定理。若需混合、松弛、熵正则或
离散化，应单独说明改变了哪个数学对象，而非默认为原始 Nash。

只有确认该规模族保留研究中的交互，才能将其作为正式 MFG 的建模候选提交。
如果它抹去了主要保护拥堵机制，应回到“有限 Token 随机排队博弈”或另行设计
不同规模族；不能为了 HJB-FPK 的名称而暗自改物理过程。

## 14. 与现有代码的对应及资料边界

| 方程部位 | 当前工程依据/缺口 |
| --- | --- |
| A、观测、一次请求 | hedge_simulation._handle_arrival / _online_observation；token_online.TokenObservation |
| Γ、窗口、Decimal | token_online.ReservationLedger.preview/admit；归一化 b 方程未实现 |
| C、T、F、Ψ | hedge_simulation._handle_complete / _handle_timer / _enter_failed / _dispatch_pass |
| 原始成本与单次偏离 | accepted ADR-0021；ticket05 仍 claimed，本轮未发现两生产模块 |
| 条件 ν、生成元/前向求解、入口最优性 | 无上述完整理论实现；旧 scalar-rho solver 不能替代 |
| 多 executor、前沿、公共条件场 | 候选理论扩展，未实现或验证 |

本文方程是针对本仓库机制的推导。以下资料提供方法依据，不提供本模型的现成
存在、收敛或近似 Nash 定理：

- Kang & Ramanan, [Fluid limits of many-server queues with reneging](https://arxiv.org/abs/1011.2921)：
  服务/等待年龄测度与 arrival/server 同尺度增长；其队列没有本文的配对取消与公共 Replay。
- Belak, Hoffmann & Seifried, [Continuous-Time Mean Field Games with Finite State Space and Common Noise](https://link.springer.com/article/10.1007/s00245-020-09743-7)：
  离散状态也可以采用公共噪声的前向/后向方法；不证明本文无界混合队列模型。
- Carmona, Delarue & Lachapelle, [Control of McKean-Vlasov Dynamics versus Mean Field Games](https://arxiv.org/abs/1210.5771)：
  群体控制与玩家最佳响应的一致性是不同数学对象。

状态离散不是障碍；本模型实际混合离散状态与连续时钟/服务进度。合适的语言是
输运—跳变前向方程、FCFS 边界和到达处 Bellman 条件，不必人为增加 Brownian
扩散项，也不必把一个复杂排队问题写成看似标准但不对应物理过程的两条 PDE。

## 15. 优先路线：不求连续 PDE，直接推导精确事件步核

选择原因：当前引擎本来按事件推进，FCFS、取消、Replay、整份准入和同刻次序
都由事件映射自然表达。先用离散事件步闭合有限模型，随后按需要近似其状态。
这保留第2—7节的全部物理与信息定义，不新增玩家决策时刻。

设 θ_n 是第 n 个事件批次结束的时刻，Y_n 是该时刻 dispatch 后的增广状态。
这里 n 是事件序号，不是等间隔物理时间；Δ_n=θ_{n+1}−θ_n 可变。
令 Φ_s y 为没有新事件时沿第3节漂移 s 的状态。d(y) 是最近的已知 timer、
窗口边界、到达截止或已知阶段切换的剩余时间。无效 timer 若会进入公开历史
也计入；没有未来事件且系统已结束时直接吸收，不引入无穷等待成本。

对于 0≤s<d(y)，尚未发生任何随机事件的生存概率为

\[
G_y(s)=\exp\!\left[-\int_0^s\lambda1_{\{t+u<320\}}du\right]
\prod_{(i,\ell):run}\frac{\bar F(e_{i\ell}+v_{r_{i\ell}}s)}{\bar F(e_{i\ell})}
\;G_z^0(s).\tag{29}
\]

原已知日程 G_z^0=1；随机故障扩展中
G_z^0(s)=exp(−∫_0^s q_z(age+u)du)。最近的已知阶段边界已在 d 中，因此
这里 v 在区间内固定。式 (29) 来自独立到达、各运行需求的条件剩余分布和
外生故障，不需要假设 lognormal 无记忆。

定义持有时间与下一状态的联合转移核 K^π(y,ds,dy')：

\[
\begin{aligned}
K^\pi(y,ds,dy')={}&1_{s<d}G_y(s)\Big[
\lambda1_{t+s<320}\sum_kw_k\sum_a\pi(a\mid O(\Phi_s y,k))
                       A_{k,a}(\Phi_s y,dy')\\
&\quad+\sum_{i,\ell:run}v_rh(e_{i\ell}+v_rs)
                       \delta_{C_{i\ell}\Phi_s y}(dy')
+q_z(age+s)\delta_{G_z\Phi_s y}(dy')\Big]ds\\
&+G_y(d-)\delta_d(ds)\delta_{G_{boundary}\Phi_d y}(dy').\tag{30}
\end{aligned}
\]

原日程删掉 q_z 项。A 是推荐与内存更新的条件核，含准入和批末 dispatch。
G_boundary 按引擎同刻顺序组合所有到期边界。概率零的随机完成平局可在核中
忽略，但路径测试仍按已定义 tie 次序。d=∞ 时不写边界原子；若永远无事件则
进入结束吸收状态。核的总质量由“首个随机事件或边界”分解为1。

嵌入链的前向方程是

\[
P_{n+1}(B)=\int P_n(dy)\int K^\pi(y,ds,B).\tag{31}
\]

它是完整联合状态的离散 Kolmogorov 方程，不是独立 Token 分布的无条件闭合。
时间包含在 Y 中；若需要某个固定物理时刻 t 的分布，还必须用持有时间和 Φ
重建，不能把“第100个事件”的分布当作“t=100”的分布。

已提交动作的 tagged 剩余成本满足纯积分递推：

\[
V_i^\pi(y)=\int K^\pi(y,ds,dy')\left[
\int_0^s c_i(\Phi_u y)du
+\gamma_{k_i}(R_i(y')-R_i(y))+V_i^\pi(y')\right].\tag{32}
\]

结算后 V_i=0；吸收状态保留成本所需的 R 标记，或把 γΔR 放入事件核的奖励
记录，不能因删除标记而得到负 Replay 费用。区间中没有 winner/阶段事件，
故运行成本不变时内层积分就是 c_i(y)s；事件时的 γ 仍只收一次。
min 仍只在式 (14) 的入口出现，式 (32) 中没有“每一步重新挑 N/D/I”。

这就是适合原模型的离散事件 Bellman—前向系统。它不要求先证明值函数对
连续进度可微，并且保持物理持有时间的成本单位。给定 π 的核由已声明输入和
所有事件映射唯一指定，不需要用一条事后轨迹的最优动作来定义它。

## 16. 若要全部有限状态化，哪些是精确的，哪些是近似的

| 表示 | 对当前模型的意义 | 本轮选择 |
| --- | --- | --- |
| 事件步 + 完整有序标记状态 | 事件步离散；时间/进度连续；保留一般服务与硬 timer | 精确理论参照，优先 |
| 有限状态、可变持有时间的半 Markov 近似 | 类别/动作/状态/整份预算精确保留，时间/进度/队列尾部近似 | 需要可计算表格时优先评估 |
| 等时步离散模型 | 需处理步内多事件、边界顺序和成本积分 | 可作独立数值近似，不能每步至多一个事件而不计误差 |
| 纯离散 CTMC | lognormal 与确定性 timer 不能只凭队长变成无记忆转移 | 若采用 phase-type 服务/时钟，须声明分布近似 |
| 连续输运/测度 PDE | 能保留进度、质量和服务年龄，FCFS边界分析复杂 | 作为极限/守恒工具；不强制作为首个求解器 |

精确保留：N/D/I、类别、attempt 关联与 FCFS 全序、winner/Replay 标志、
requested/applied、逐份预算。K1 的 c=2.8125 减去整数批准数，余额只有
2.8125、1.8125、.8125 这些 D 窗内状态，不需要对这部分再做粗格点近似。

需要处理：服务进度、等待/阶段年龄、timer 剩余时间和无界队列。一个典型状态
可写成有序 attempt 列表及其进度格点、timer 格点和精确的离散标记；不能退回
只存两个队长。缩小队列上限要显式保留 overflow/误差，不可把超限 Token 丢弃
后赋0成本。连续格点与队列截断带来的时间/成本尾误差必须记录。

分箱还涉及 Markov 性：如果两个真实状态 y1,y2 落在同一格点，但
K(y1,下一格点)≠K(y2,下一格点)，分箱过程一般不是精确 Markov 链。
应声明格点代表状态或格内重建核 R_h(dy|s)，再定义近似核

\[
K_h^\pi(s,d\tau,s')=\int R_h(dy\mid s)
               K^\pi(y,d\tau,\{y':\Pi_h(y')=s'\}).\tag{33}
\]

R_h 是数值近似的一部分，需要误差分析；不能宣称分箱自动继承原过程全部
条件独立性。策略仍只使用可观测分箱/条件信念，不使用隐藏 service draw。
式 (33) 的概率和、持有时间及事件费用共用一个核，避免前向与成本来自两套模型。

若使用固定时格，精确转移定义为整段间隔内的原过程条件分布（含任意多个事件）；
用 I+ΔtQ 替代只是额外数值近似，需要正概率、步长和边界误差约束。
已知 fault/window/cutoff 可作为独立精确事件保留，从而避免跨边界的费用错位。

选择离散精度的理论标准应落在动作成本上：若在某个固定环境和信息条件下
max_a |Q_h(a|o)−Q(a|o)|≤ε，则按 Q_h 选择的动作相对该环境真最佳响应的
损失≤2ε。这是逐条件的动作选择界；还需另加场误差、统计误差、策略限制以及
有限规模误差，不能把它单独称为 Nash 保证。

## 17. 离散均值场方程仍须通过同一个闭合门槛

在明确的近似状态/规模族下，可以把第10—12节写成矩阵/求和形式。取公共
物理时格 t_j（不是各 Token 各自不同的事件序号），使用包含完整标记关联的
状态 s。给定候选环境路径和策略，令 P_j^π(s,s') 是该间隔内一个已存在 Token
未结算并进入 s' 的概率，因此行和≤1；墓地质量单独记录。

\[
m_{j+1}(s')=\sum_s m_j(s)P_j^\pi(s,s')+\beta_j^\pi(s').\tag{34}
\]

β_j 不是把 λΔt 全放在步末的 newborn：它是该间隔内所有出生，经各自到达
时的策略/准入，再演化到 t_{j+1} 后仍未结算的状态质量。这样到达后同一步内
完成的 Token 才不会被误保留。故障产生的 Replay 改标记，不算 β 的新玩家。

后向继续价值为

\[
v_j(s)=g_j(s)+\sum_{s'}P_j^\pi(s,s')v_{j+1}(s'),\qquad
\pi_j^*(\cdot|o)\text{ 支撑于 }\arg\min_a Q_j(a|o).\tag{35}
\]

g_j 是原初始模型在该间隔内的预期未获胜时间、实际 H/R 执行价与 Replay
创建费用；含间隔内结算的 Token 费用。全局 cutoff 仍非零终值边界。
随机公共故障下，P_j/g_j 是对当前可用公共信息的下一步条件核，后向平均所有
公共后继；已实现的前向按所观测公共更新分支推进，不能预见完整未来。

预算和 FCFS 选择必须参与构造同一个 P_j；它一般依赖间隔内的场/余额演化，
不应无说明地冻结成仅看步初 m_j 的负载表。只有状态和机制已经闭合后，才可
写成 P_j[m_j,b_j,π_j] 的简记。离散化不会使丢失的副本关联、优先级或信念
自动出现，也不会证明 K 极限存在。

因此下一步优先深化 (29)—(33) 的可计算状态和误差边界，再判断是否需要求解
连续 PDE。理论选择依据是信息充分性、事件语义、计算规模和动作成本误差，
不是预先要求某一种方程形式。

补充方法资料：[Uniformization: Basics, extensions and applications](https://www.sciencedirect.com/science/article/pii/S0166531617301451)
讨论 CTMC 与离散链的转换及适用范围；这里没有因为存在这种方法就把原
lognormal/timer 系统声明成有限状态 CTMC。式 (29)—(35) 为本文按本项目机制
构造的方程，文献不替本模型提供近似误差或均衡证明。
