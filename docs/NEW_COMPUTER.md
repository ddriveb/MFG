# 新电脑：MFG 独立运行与验收

MFG 使用 Windows Python **3.10** 和标准库，与 SRP/XLRON 完全独立。
安装 Python 3.10 后，在 PowerShell 中执行：

```powershell
git clone git@github.com:ddriveb/MFG.git
cd MFG
.\scripts\bootstrap.ps1
```

若没有 Windows `py` 启动器，明确传入解释器：

```powershell
.\scripts\bootstrap.ps1 -PythonExe C:\Python310\python.exe
```

脚本在当前仓库创建 `.venv`，注册 `src`，校验并还原七个冻结候选输入，
然后运行全套 unittest 和最小配置检查。冻结输入仅约 2.46 MiB，原始哈希不变，
详见 `docs/migration/FROZEN_INPUTS.md`；它们不包含正式 r2 续跑账本。
不安装第三方依赖，不启动正式 qualification/holdout。任一步失败都会停止。

## 验收标准

- 测试成功且导入来自当前 clone；比较 skipped tests，不仅看退出码。
- `test_engine_equivalence.py` 中内嵌的确定性 golden 保持一致。
- config check 返回 `status: ok`；负载可行性提示与检查失败不同。
- `test_common_state_metrics` 的旧 schema-2 artifact 回归在没有旧数据时会 skip；
  这是源码 checkout 的已知差别，不意味着历史数据已经迁移。

## 阅读与数据恢复

从 `docs/CODE_MAP_ZH.md`、ADR-0037/0038/0039 和
`.scratch/reliability-aware-token-mfg/` 开始。早期 Hedge/shared-backup 保留为历史基线。

Git 不包含 `artifacts/` 和 `.scratch` 生成的结果/日志。续跑前从旧机备份还原完整 run
目录及关联输入，核验哈希，读取原 ticket 的恢复规则。r2 qualification 的 checkpoint、
预算、已保留身份必须整体恢复；不能在新机清零或重复消费原正式身份。
代码迁移不自动启动 campaign，不把未完成试验标记完成。
