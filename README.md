# MFG-Hedge v1

> **2026-09-12 独立仓库入口**：本仓库只研究专家副本备份／Token-MFG，
> 与 SRP、XLRON、JAX、CUDA 无关。新电脑使用 Windows Python **3.10**。
> [安装与验收](docs/NEW_COMPUTER.md) · [代码阅读地图](docs/CODE_MAP_ZH.md)
>
> 当前同时 Token-MFG 主线见 ADR-0037/0038/0039 与
> `.scratch/reliability-aware-token-mfg/`。下面保留早期 Hedge 项目说明作为历史背景；
> 不代表当前模型仅停留于最初阶段，也不表示正式 qualification 已通过。
> 实验产物与续跑账本需单独从旧机备份恢复，clone 本身只迁移代码与研究文档。

这是一个与仓库现有实验隔离的、可复现的 Python 3.10 仿真子项目。第一阶段只研究：在单层 Top-1 MoE、固定副本布局与单故障域扰动下，价格协调的 Token 保护策略能否缓解 Protection Storm。

当前环境刻意保持很小：

- 零第三方运行依赖，避免在模型与指标尚未定稿前引入框架耦合；
- 配置、源码、测试、生成物严格分目录；
- 已定义公共状态、Token 类型、保护动作及动作统计接口；
- 已实现 Healthy/故障/Replay/Hedge 仿真及历史求解器；新的瞬态模型目前仅提供开发用准入与结果评分，尚未做策略搜索或证明 MFG 有效。

## 快速开始

在 PowerShell 中运行：

```powershell
cd D:\project\mfg_hedge_v1
.\scripts\bootstrap.ps1
.\.venv\Scripts\python.exe -m mfg_hedge check --config .\configs\v1_minimal.json
```

`bootstrap.ps1` 只创建本地 `.venv`、在该环境中注册本项目的 `src` 路径并执行标准库测试，不访问网络、不安装第三方包。源码修改会立即生效，行为等同于本地 editable 开发环境。

## 目录

```text
mfg_hedge_v1/
├── configs/                # 受版本控制的实验配置
├── docs/                   # 方案评估与研究口径
├── src/mfg_hedge/          # 可复用核心代码
├── tests/                  # 快速、确定性的单元测试
├── scripts/                # 环境引导脚本
└── artifacts/              # 生成结果（默认不纳入 Git）
```

详细的方案判断和下一步顺序见 `docs/design_review_zh.md`。

## 配对实验

单 seed 同 trace 对照：

```powershell
.\.venv\Scripts\python.exe -m mfg_hedge compare-mfg-hedge-vs-no-hedge --config .\configs\v1_paired_comparison.json --tokens 1000
```

The schema-3 command above reproduces the original zero-persistent-cost
experiment. The corrected ADR-0009 objective uses schema 4:

```powershell
.\.venv\Scripts\python.exe -m mfg_hedge compare-mfg-hedge-vs-no-hedge --config .\configs\v2_paired_runtime_cost.json --tokens 1000
```

配置冻结的多 seed campaign（读取 `tokens_per_run`、`base_seed` 和
`seed_count`）：

```powershell
.\.venv\Scripts\python.exe -m mfg_hedge campaign-mfg-hedge-vs-no-hedge --config .\configs\v1_paired_comparison.json
```

0.6/0.7 只作 solver/容量压力诊断；门禁失败时不会生成伪 paired 结果：

```powershell
.\.venv\Scripts\python.exe -m mfg_hedge diagnose-mfg-hedge-loads --config .\configs\v1_paired_comparison.json --loads 0.6 0.7
```

## 工作流

新增开发 API 与可运行示例见 `docs/transient-control.md`。它与上述历史
配对 CLI 隔离，使用真实队列结果评分，不复用旧求解器的预测成本。

8-Expert / 3-Replica 扩展属于独立的开发实验，入口和拓扑限制同样记录在
`docs/transient-control.md`，不改变历史双副本结果。

- 开始工作前阅读 `AGENTS.md`、`CONTEXT.md` 和相关 ADR。
- 每个功能在 `.scratch/<feature-slug>/` 下拥有独立 spec 和 tickets。
- 每次更新按 `docs/agents/update-format.md` 记录到当前 ticket。
- 生成结果只写入唯一的 `artifacts/<run-id>/`，不得覆盖历史运行。
