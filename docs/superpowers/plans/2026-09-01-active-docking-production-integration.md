# 主动 docking 生产集成实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 让新的主动 ligand-receptor docking 工作流从既有 full workflow 的 `prepare` 产物开始，按 3 个旧配置 seed 执行选中任务并以 median score 更新部分观测状态，同时保持旧流程和 receptor-subset QUBO 不变。

**架构：** 保留 `FullExperimentRunner` 的 canonical 七阶段和 `scripts/run_experiment.py`。新增 `active_docking/production.py` 作为桥接层，调用既有 preparation 和 `DockingAdapter`，使用脱敏 active manifest、独立 active run directory、原子 round checkpoint 和配置 fingerprint。现有 `run_active_docking.py` 继续负责完整矩阵 masked replay；新增 `run_active_experiment.py` 负责真实生产 active run。

**技术栈：** Python 3.11、现有 dataclass/JSON/CSV、RDKit、NumPy、既有 Uni-Dock/Vina adapter、pytest。

---

### 任务 1：锁定配置与旧产物路径合同

**文件：**
- 创建：`src/qubo_receptor_ensemble/active_docking/production_config.py`
- 创建：`tests/test_active_docking_production_config.py`
- 创建：`configs/active_docking/mk14_active_docking_remote.json`

- [ ] **步骤 1：编写失败的测试**

测试 schema 解析、`base_experiment_config`、`prepared_run_directory`、`active_run_directory`、配置 fingerprint、3-seed median 协议、baseline `11OY` 和禁止 active workflow 使用旧 receptor-subset problem。

- [ ] **步骤 2：运行测试确认失败**

运行：

```bash
PYTHONPATH=src python -m pytest -q tests/test_active_docking_production_config.py
```

预期：因生产配置模块不存在而失败。

- [ ] **步骤 3：实现最小配置加载器和 MK14 配置**

解析 active 配置，并调用 `load_full_experiment_config()` 验证 base schema 3.0 配置。相对 `prepared_run_directory` 和 `active_run_directory` 以 `data_root` 解析；允许 CLI 显式覆盖 prepared run。校验 workflow、engine、3 个唯一整数 seed、`score_fusion=median`、正预算和独立输出目录。

- [ ] **步骤 4：运行测试确认通过**

运行同一 focused pytest，预期全部通过。

- [ ] **步骤 5：Commit**

```bash
git add src/qubo_receptor_ensemble/active_docking/production_config.py tests/test_active_docking_production_config.py configs/active_docking/mk14_active_docking_remote.json
git commit -m "feat: add active docking production config"
```

### 任务 2：桥接旧 prepare manifest 并生成可见特征

**文件：**
- 创建：`src/qubo_receptor_ensemble/active_docking/manifest_bridge.py`
- 创建：`tests/test_active_docking_manifest_bridge.py`
- 修改：`src/qubo_receptor_ensemble/active_docking/warm_start.py`

- [ ] **步骤 1：编写失败的测试**

使用旧格式 `prepared_ligands.csv` 和 `selected_receptors.csv` 测试：脱敏 ligand/receptor manifest、RDKit ligand descriptor、scaffold、aligned receptor structural descriptor、固定 cluster、`11OY` baseline、缺失 baseline 失败、标签不进入可见 manifest。

- [ ] **步骤 2：运行测试确认失败**

运行：

```bash
PYTHONPATH=src python -m pytest -q tests/test_active_docking_manifest_bridge.py
```

预期：桥接函数不存在或输出 schema 不完整。

- [ ] **步骤 3：实现最小桥接**

读取旧 CSV；从 ligand SMILES 生成确定性数值特征；从 aligned PDB 的 C-alpha 坐标和已存在的 alignment RMSD 生成稳定 receptor 特征；用固定结构距离阈值和按 conformer ID 排序生成 cluster。输出只包含 predictor 所需字段和 PDBQT 路径，不复制 `label`、`active`、`decoy`、`selection_role`。将 receptor 的真实 ID 保留为 `conformer_id`/`receptor_id`。

- [ ] **步骤 4：运行测试确认通过**

运行同一 focused pytest，预期通过，并运行现有 warm-start 测试确认 `r0` fixture 行为不变。

- [ ] **步骤 5：Commit**

```bash
git add src/qubo_receptor_ensemble/active_docking/manifest_bridge.py src/qubo_receptor_ensemble/active_docking/warm_start.py tests/test_active_docking_manifest_bridge.py
git commit -m "feat: bridge prepared manifests for active docking"
```

### 任务 3：实现 3-seed selected-task docking 和 score fusion

**文件：**
- 创建：`src/qubo_receptor_ensemble/active_docking/executor.py`
- 创建：`tests/test_active_docking_executor.py`
- 修改：`src/qubo_receptor_ensemble/docking_adapters.py`

- [ ] **步骤 1：编写失败的测试**

使用 fake adapter 测试每个 selected task 只被提交一次到每个配置 seed；未选 ligand-receptor pair 不会传入 adapter；读取 `docking_score` 后按 task 求 3-seed median；不接受缺 seed、重复 task、失败 score 或非有限 score；active adapter 输入不需要 label；旧 adapter 输出行为保持兼容。

- [ ] **步骤 2：运行测试确认失败**

运行：

```bash
PYTHONPATH=src python -m pytest -q tests/test_active_docking_executor.py
```

预期：executor 不存在，或 adapter 对无 label 输入抛出 `KeyError`。

- [ ] **步骤 3：实现最小 executor 和 adapter 兼容层**

按 receptor/seed 分组 selected tasks，调用已有 `get_docking_adapter()` 和 `run_batch()`。保留 `root=data_root` 的路径解析和 resume 语义。将 adapter 的 score row label 改为可选，不改变旧 full workflow 的 label 输出；active executor 单独写无标签 score table，并生成任务级 `seed_scores`、`fused_score`、elapsed time 和 cost。

- [ ] **步骤 4：运行测试确认通过**

运行 focused executor 测试，以及 `tests/test_docking_adapters.py` 和旧 full workflow 的 docking 测试。

- [ ] **步骤 5：Commit**

```bash
git add src/qubo_receptor_ensemble/active_docking/executor.py src/qubo_receptor_ensemble/docking_adapters.py tests/test_active_docking_executor.py
git commit -m "feat: execute selected docking tasks with seed fusion"
```

### 任务 4：生产 runner、checkpoint 和恢复

**文件：**
- 创建：`src/qubo_receptor_ensemble/active_docking/production.py`
- 创建：`tests/test_active_docking_production.py`
- 修改：`src/qubo_receptor_ensemble/active_docking/state.py`
- 修改：`src/qubo_receptor_ensemble/active_docking/replay.py`

- [ ] **步骤 1：编写失败的测试**

覆盖：从 base full config 执行既有 prepare、生成 active run；warm-start 调用真实 executor；每轮执行 predictor → acquisition → QUBO → solver → selected docking → reveal；成本按 seed 数计费；state 和 round artifact 原子落盘；中断后恢复；配置或 prepared input fingerprint 不一致时拒绝恢复；决策前审计不包含 hidden score/label；final evaluation 只在结束读取原 prepared label。

- [ ] **步骤 2：运行测试确认失败**

运行：

```bash
PYTHONPATH=src python -m pytest -q tests/test_active_docking_production.py
```

预期：production runner 不存在。

- [ ] **步骤 3：实现生产 runner**

`prepare` 直接调用 `prepare_experiment_inputs()`，并将旧产物路径记录到 active manifest。`initialize` 对所有 ligand 的 baseline receptor 和结构 cluster coverage 执行 docking。每个 round 使用现有 active predictor/acquisition/QUBO/solver；只在 solver 返回后调用 executor；收到 fused score 后一次性调用 `state.reveal()`。每轮先写临时目录，再用 `os.replace` 更新 checkpoint。active run 目录位于旧 run 下的 `active_docking/`，不覆盖旧 score tables/matrices/problem/selection/evaluation。

- [ ] **步骤 4：运行测试确认通过**

运行 focused production tests，并运行现有 `tests/test_active_docking.py`。

- [ ] **步骤 5：Commit**

```bash
git add src/qubo_receptor_ensemble/active_docking/production.py src/qubo_receptor_ensemble/active_docking/state.py src/qubo_receptor_ensemble/active_docking/replay.py tests/test_active_docking_production.py
git commit -m "feat: add resumable active docking runner"
```

### 任务 5：新增 CLI 和真实数据验证

**文件：**
- 创建：`scripts/run_active_experiment.py`
- 创建：`tests/test_active_docking_production_cli.py`
- 修改：`src/qubo_receptor_ensemble/active_docking/__init__.py`

- [ ] **步骤 1：编写失败的测试**

测试 CLI `validate`、`prepare`、`run`、`resume`、`finalize` 的参数解析和 dry-run 行为；`validate` 不启动 docking；生产 CLI 不接受完整矩阵作为必需输入；`--prepared-run-directory` 能覆盖本地历史目录。

- [ ] **步骤 2：运行测试确认失败**

运行：

```bash
PYTHONPATH=src python -m pytest -q tests/test_active_docking_production_cli.py
```

预期：脚本不存在。

- [ ] **步骤 3：实现 CLI**

参数包含 `--config`、`--data-root`、`--prepared-run-directory`、`--resume`、`--overwrite`。默认命令只验证或运行本地配置指定的 workflow；生产 `run` 显式标记 `real_docking_executed=true`，不连接量子硬件。旧 `scripts/run_active_docking.py` 保留离线 replay 语义。

- [ ] **步骤 4：运行测试确认通过**

运行 CLI focused 测试和 `python scripts/run_active_experiment.py --help`。

- [ ] **步骤 5：Commit**

```bash
git add scripts/run_active_experiment.py src/qubo_receptor_ensemble/active_docking/__init__.py tests/test_active_docking_production_cli.py
git commit -m "feat: add active docking production cli"
```

### 任务 6：文档、远程路径和 end-to-end smoke

**文件：**
- 修改：`docs/active_docking_workflow_zh.md`
- 修改：`docs/experiment_workflow_zh.md`
- 修改：`configs/active_docking/mk14_active_docking_remote.json`
- 创建：`tests/fixtures/active_docking/production_base_config.json`
- 创建：`tests/test_active_docking_end_to_end.py`

- [ ] **步骤 1：编写失败的 end-to-end 测试**

使用 fake prepare/fake adapter 完成小型多轮流程，断言旧 prepare output 被消费、每个任务执行 3 seed、状态和结果目录正确、相同 seed 复现 task sequence。

- [ ] **步骤 2：运行测试确认失败**

运行：

```bash
PYTHONPATH=src python -m pytest -q tests/test_active_docking_end_to_end.py
```

- [ ] **步骤 3：实现文档和 smoke fixture**

补充本地 `E:\Quant\qubo_receptor_ensemble_experiment_data` 与远程 `/root/autodl-tmp/qubo_data_root` 的映射，明确旧 MK14 run directory、raw 输入、prepare-only 命令、active run 命令、3-seed cost、resume、真实 docking 和 masked replay 的边界。所有公式使用 `$$` 块。

- [ ] **步骤 4：运行测试确认通过**

运行 end-to-end focused 测试和 `git diff --check`。

- [ ] **步骤 5：Commit**

```bash
git add docs/active_docking_workflow_zh.md docs/experiment_workflow_zh.md configs/active_docking/mk14_active_docking_remote.json tests/fixtures/active_docking/production_base_config.json tests/test_active_docking_end_to_end.py
git commit -m "docs: document active docking production workflow"
```

### 任务 7：完整验证和最终提交检查

**文件：**
- 无新增职责文件；检查所有上述变更。

- [ ] **步骤 1：运行新增 focused tests**

```bash
PYTHONPATH=src python -m pytest -q tests/test_active_docking_production_config.py tests/test_active_docking_manifest_bridge.py tests/test_active_docking_executor.py tests/test_active_docking_production.py tests/test_active_docking_production_cli.py tests/test_active_docking_end_to_end.py
```

- [ ] **步骤 2：运行 replay 和配置验证**

在小型 fixture 上运行原有 `run_active_docking.py validate/predict/replay`，并运行新 CLI `validate` 和 fake-adapter smoke；不运行真实 docking 或量子硬件。

- [ ] **步骤 3：运行回归测试**

```bash
PYTHONPATH=src python -m pytest -q
python -m compileall -q src scripts
git diff --check
```

- [ ] **步骤 4：检查审计边界**

扫描 active state、prediction/acquisition/QUBO/solver round artifacts，确认没有 hidden score、active/decoy label 或未选任务 score；确认旧配置和旧 CLI 没有语义变更。

- [ ] **步骤 5：Commit**

```bash
git status --short --branch
git log -5 --oneline
```

只保留 `feat/quantum-active-docking` 分支，不合并、不推送、不删除其他分支。

