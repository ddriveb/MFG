# MFG 代码阅读地图

Gate 已决定 Logical Expert，策略在同一专家的物理副本间备份/路由，不改变专家身份。
本仓库与频谱分配的 SRP/XLRON 没有代码、依赖或实验指标耦合。

| 层次 | `src/mfg_hedge/` 入口 | 关键边界 |
|---|---|---|
| 物理执行 | `reliability_aware_routing.py`、`simultaneous_token_routing.py` | 同批 Token 共用决策前观察；故障/恢复/Replay |
| 人群统计 | `population_estimator.py`、`token_population_response.py` | 在途 Token 与新/Replay 决策 cohort 分开 |
| 响应求解 | `token_mfg_response_solver.py`、`token_mfg_fixed_point.py` | 求解与评估 trace 隔离，不读取未来 |
| 资格评估 | `token_mfg_qualification*.py`、`token_mfg_streaming*.py` | 初始化、跨 K 传递、预算与 checkpoint |
| 历史基线 | `shared_backup.py`、`expert_game.py`、Hedge/Replay 模块 | 保留原指标和物理语义 |

验收位于 `tests/`，受控输入位于 `configs/`，历史脚本/研究决策在 `.scratch/`。
当前契约以 ADR-0037/0038/0039 和对应 ticket 为准；实现存在不代表 qualification 已通过。
