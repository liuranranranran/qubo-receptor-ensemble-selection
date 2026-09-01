# 量子辅助预算约束主动 ligand-receptor docking 工作流

本文档说明仓库中 `active_docking` 的离线 masked replay 和生产集成实现。离线 replay 只使用已有完整 score matrix；生产流程从旧 full workflow 的 `prepare` 产物切出，按策略选择任务后才启动真实 docking。两者都不默认访问远程任务或量子硬件。

## 1. 研究边界

预测器负责估计未完成 ligand-receptor 任务的 score 分布；QUBO 和 solver 只负责从候选任务中选择下一批。预测器不会读取未揭示 score，solver 也不会承担 score prediction。hidden active/decoy 标签只在 replay 结束时计算评价指标。

工作流的闭环是：

$$
\text{已有 docking 结果}
\rightarrow
\text{可见 score 状态}
\rightarrow
\text{预测未完成 score 分布}
\rightarrow
\text{后验任务价值与批内互补}
\rightarrow
\text{预算约束 QUBO}
\rightarrow
\text{下一批任务}
\rightarrow
\text{揭示所选 score 并更新状态}
$$

当前代码对应设计文档的 v0.1 最小实现。Bayesian 主模型是具有显式后验协方差的 Bayesian linear residual model，不是完整层次模型，也没有把普通点预测包装成 posterior。量子兼容后端是明确标记的模拟适配器，不代表量子硬件结果。

## 2. 部分观测状态

`PartialObservationState` 保存 ligand/receptor manifest、候选任务、已观察 score、任务成本、当前轮次、warm-start 审计和 scaffold/cluster 元数据。隐藏 oracle 不属于状态对象。

对任务 `a=(l,r)`，只有以下状态转移会使 score 可见：

$$
D_{t+1}=D_t\cup\{(a,s_a):a\text{ 被当前策略选中}\}
$$

`reveal` 先验证所有任务都在候选集合、没有重复揭示、score 有限，再一次性更新 score 和成本。任何一项失败都会使整个操作失败。状态序列化只写 observed score，不写 hidden score 或 hidden label；manifest 也拒绝 `label`、`active`、`decoy` 等评价字段。

## 3. Warm-start

第一轮为每个 ligand 固定执行基准受体 `r0`。随后按 scaffold 稳定排序，从每个非基准受体 structural cluster 选择固定比例或最小数量的 ligand。排序使用 SHA-256 和配置 seed，不使用 docking score 或最终指标。`WarmStartConfig` 独立于预测和 solver，报告中应单独标记 warm-start 成本与覆盖。

完整矩阵 replay 会先将所有任务放入外部 oracle，再只把 warm-start score 写入状态。外部 oracle 在选择之后才把所选任务的 score 传给 `reveal`。

## 4. Score prediction

主模型预测相对于基准受体的 residual：

$$
\Delta_{lr}=s_{lr}-s_{l,r_0}
$$

恢复 score 的公式为：

$$
\hat{s}_{lr}=s_{l,r_0}+\hat{\Delta}_{lr}
$$

实现使用 ligand 数值特征、receptor 数值特征、两者的交互乘积以及 scaffold/cluster 的稳定哈希特征。共轭 Bayesian 线性模型的预测输出为：

$$
\mu_{lr}=s_{l,r_0}+x_{lr}^{\mathsf T}\mathbb{E}[\beta\mid D_t]
$$

$$
\sigma^2_{lr}=\sigma^2_\epsilon+x_{lr}^{\mathsf T}\operatorname{Cov}(\beta\mid D_t)x_{lr}
$$

`sample` 使用确定性 seed 产生 posterior predictive score。另有 `NearestReceptorPredictor` 和 `ObservedScoreMeanPredictor` 两个可审计基线。`run_masked_prediction_gate` 输出 MAE、RMSE、95% 区间覆盖率和 NLL；只有在 masked score 验证优于简单基线时，复杂主模型才可作为研究协议中的主模型。

预测门应使用 scaffold-disjoint 或结构分层的 mask。若区间校准失败，或 top-q 任务质量不优于简单基线，不应把后续 QUBO 结果解释为可靠的主动 docking。

## 5. 任务价值与 posterior sampling

对候选任务集合 `S`，每个 posterior sample 都把假设 score 加入可见矩阵副本，然后按锁定的 mean score fusion 生成 ligand 排名。默认无活性先验，使用排序 score/information utility；如果配置训练期 `activity_prior`，它只能来自训练边界，不能来自当前 hidden 标签。

$$
F_t(S)=\mathbb{E}_m[U(\operatorname{rank}(M_t\cup\tilde{S}^{(m)}))]
$$

$$
v_a=F_t(\{a\})-F_t(\varnothing)
$$

$$
v_a^{\mathrm{cost}}=\frac{v_a}{c_a}
$$

风险项通过 top-q ligand 的 posterior fused-score 标准差进入 utility：

$$
F_t(S)=\mathbb{E}[U]-\lambda_{\mathrm{risk}}\mathbb{E}[\text{top-q uncertainty}]
$$

空集合的边际价值严格为零。Monte Carlo 数量和 seed 在配置中固定，不能按 outer/test 指标后验调整。

## 6. 批内互补与候选裁剪

后验效用直接生成 pairwise interaction：

$$
\Gamma_{ab}=F_t(\{a,b\})-F_t(\{a\})-F_t(\{b\})+F_t(\varnothing)
$$

实现缓存每个任务的 posterior draw，使单任务和 pair utility 使用同一随机协议。候选裁剪先去除已完成任务，再按固定 value-per-cost、uncertainty、scaffold representative 和 receptor cluster representative 取候选。replay 比较时固定共享候选池，以便每个 solver 接收相同变量集合；候选池规模、规则和数量在审计 JSON 中单独报告。

候选裁剪是经典预处理，不得将它产生的收益归因于 QUBO 或量子后端。

## 7. 批量 QUBO 与约束

对每个任务定义二值变量 `z_lr`。目标采用最小化形式：

$$
E(z)=-\sum_a v_a z_a-\lambda_{\mathrm{batch}}\sum_{a<b}\Gamma_{ab}z_az_b+P\Phi(z)
$$

`BatchQUBO.matrix` 是对称矩阵，能量由 `x^T Q x + constant` 计算。实现支持：

- 总 docking 成本预算；
- 每个 ligand 每轮最多新增 receptor 数；
- 每个 receptor 每轮最多处理 ligand 数；
- 每个 scaffold 每轮最多任务数；
- 可选 receptor activation cost 和 `x_r` 变量；
- 等成本任务的 cardinality 约束；
- 不同成本任务的整数化 budget/slack 变量；
- 固定 penalty、成本单位和矩阵缩放审计。

预算和 `<=` 覆盖约束转换为固定整数单位的等式 slack。可行性由原始浮点成本和约束重新诊断，不依赖 QUBO 能量的表面数值。成本离散化误差记录在 `FeasibilityResult.integerization_error`。

惩罚系数来自配置冻结值，不能根据 outer/test 的 BEDROC、EF 或 PR-AUC 调整。

## 8. Solver 对比

统一入口 `solve_batch_qubo` 支持：

- `exact`：枚举可行任务集合；
- `value_greedy`：单位成本价值贪心；
- `greedy_one_swap`：贪心后局部一换一；
- `simulated_annealing`：任务变量上的可行模拟退火；
- `quantum_compatible_simulator`：量子兼容 QUBO 输入/输出协议的模拟适配器。

所有这些后端共享当前候选、QUBO、约束和配置时间预算。量子兼容模拟器的输出含 `backend_type=quantum_compatible_simulation`、`quantum_hardware_used=false` 和 `quantum_execution_result=not_run`。模拟退火结果不能称为量子结果。CP-SAT/MILP 在依赖可用前不静默降级，当前未安装时会明确报告 unavailable。

只有在相同候选集、相同约束和相同时间预算下，量子后端相对 exact/CP-SAT/强经典基线产生端到端解质量、time-to-solution 或 docking 资源收益时，才可讨论量子优势。QUBO 能量更低、解可行或复现 exact 解都不能单独证明量子优势。

## 9. Masked replay

入口为 `scripts/run_active_docking.py`：

```text
validate -> 验证独立 active-docking 配置
predict  -> warm-start 后运行 masked score prediction gate
replay   -> 多策略逐轮选择、揭示、更新和最终评估
compare  -> replay 的多策略比较别名
```

示例命令：

```bash
python scripts/run_active_docking.py validate --config configs/active_docking/default_masked_replay.json
python scripts/run_active_docking.py predict --config configs/active_docking/default_masked_replay.json --matrix matrix.json --ligand-manifest ligands.json --receptor-manifest receptors.json
python scripts/run_active_docking.py replay --config configs/active_docking/default_masked_replay.json --matrix matrix.json --ligand-manifest ligands.json --receptor-manifest receptors.json --labels labels.json
```

`matrix.json` 可以是宽矩阵行，也可以是 `{ "scores": [{"ligand_id": ..., "receptor_id": ..., "score": ...}], "labels": {...} }`。标签在 replay 期间只保存在调用方，最终计算 BEDROC20、PR-AUC、EF1% 等指标时才读取。

replay 审计记录每轮候选池、可用任务、selected/revealed 任务、solver backend、观察任务数量、成本和停止原因。审计不写 hidden score 或 hidden label。相同 seed 对同一输入复现任务序列和确定性结果；wall-clock 测量不写入确定性 replay artifact。

## 10. 指标和研究结论

最终报告应分开回答：

1. 预测模型是否能预测未完成 score，尤其是误差、排名和区间校准是否优于简单基线；
2. 带 `Gamma_ab` 的批量策略是否优于 value-greedy，且收益不是候选裁剪产生；
3. QUBO 是否优于 exact/CP-SAT 或其他强经典方法；
4. 量子后端是否有端到端收益；
5. 收益是否来自量子优化，而不是 predictor、warm-start、候选预筛选或后处理。

当前实现和小型 replay 只提供软件闭环、泄漏边界和可重复性证据，不证明 H1、H2 或 H3。不能写成“QUBO 能量更低所以筛选更好”“量子硬件返回可行解所以有量子优势”“模拟退火代表量子结果”或“replay 成功所以真实 docking 一定节省成本”。

统计上应以 target-level 为主要单位，报告 paired bootstrap 或 target-level hierarchical interval、每个 target、最坏 target 和失败类型。20% active rate 的内部面板不能作为真实筛选泛化的充分证据。

## 11. 失败停止条件

应在配置或预注册中冻结以下停止门：预算耗尽、连续轮边际效用低于阈值、连续两轮排名变化低于稳定性阈值、不确定性达到目标、预测门失败或 solver 可行性/时间门失败。预测门或批量收益门失败时，不应跳过门控直接申请真实 docking；真实 docking 和硬件实验属于后续独立阶段。

## 12. 从旧 prepare 产物运行生产 active docking

生产入口是 `scripts/run_active_experiment.py`，它与 `scripts/run_active_docking.py` 分离。旧 full workflow 的 canonical 阶段仍然是：

```text
prepare -> dock -> aggregate -> build_problem -> solve -> evaluate -> persist
```

主动流程新增的阶段是：

```text
prepare -> active_initialize -> warm_start -> active_rounds -> active_finalize
```

`active_initialize` 只读取旧 `prepare` 生成的 `prepared_ligands.csv` 和 `selected_receptors.csv`，通过 `manifest_bridge` 生成脱敏 manifest；它不会读取旧完整 matrix，也不会扩展旧 `receptor_subset` QUBO。生产 active 产物写入旧运行目录下的 `active_docking/`，不会覆盖旧 score tables、matrices、problem 或 evaluation。

### 12.1 MK14 配置与输入

生产配置位于 `configs/active_docking/mk14_active_docking_remote.json`。MK14 的 baseline receptor 是旧 manifest 中真实的 `11OY`，不是抽象的 `r0`。旧 prepare 的 raw 输入仍由 full config 声明：

```text
data/raw/dude/mk14/actives_final.ism
data/raw/dude/mk14/decoys_final.ism
data/raw/external_targets/mk14_dude/mk14/receptor.pdb
data/raw/external_targets/mk14_dude/mk14/crystal_ligand.sdf
data/raw/rcsb/mk14/
```

旧 `prepared_ligands.csv` 中的 `label` 和 `selection_role` 只保留在 evaluation 边界。桥接后的 active ligand manifest 只包含 ligand ID、SMILES、scaffold、RDKit 数值特征和 PDBQT 路径；active receptor manifest 只包含 receptor ID、对齐结构特征、cluster 和结构路径。

每个被选中的 ligand-receptor 任务固定执行三个 Uni-Dock seed：

```text
20260821, 20260822, 20260823
```

三个 `pose_rank=1` docking score 使用 median 融合。任务成本为：

$$
c_{lr}=3\times c_{seed}
$$

warm-start 成本计入总预算，但在报告中单独列出；warm-start 本身不作为量子优势证据。

### 12.2 远程服务器运行

以下命令适用于远程 Linux 服务器。仓库名为 `qubo-receptor-ensemble-selection`，数据根目录为 `/root/autodl-tmp/qubo_data_root`：

```bash
REPO_ROOT=/root/qubo-receptor-ensemble-selection
DATA_ROOT=/root/autodl-tmp/qubo_data_root
CONFIG="$REPO_ROOT/configs/active_docking/mk14_active_docking_remote.json"

cd "$REPO_ROOT"
conda activate qubo-receptor-ensemble
python -m pip install --editable .
python scripts/run_active_experiment.py validate \
  --config "$CONFIG" \
  --data-root "$DATA_ROOT"
```

如果数据根目录中还没有旧的 `results/runs/mk14_adaptive_remote/prepared_ligands.csv` 和 `selected_receptors.csv`，先只运行旧流程的 `prepare`：

```bash
python scripts/run_experiment.py run \
  --config "$REPO_ROOT/configs/experiments/mk14_adaptive_remote.json" \
  --data-root "$DATA_ROOT" \
  --from prepare \
  --to prepare
```

这一步只生成旧 canonical prepare 产物，不生成旧 receptor-subset problem。已有旧运行目录时，不要对旧目录使用 `--overwrite`；直接把它作为 active 输入运行：

```bash
PREPARED_RUN="$DATA_ROOT/results/runs/mk14_adaptive_remote"

python scripts/run_active_experiment.py run \
  --config "$CONFIG" \
  --data-root "$DATA_ROOT" \
  --prepared-run-directory "$PREPARED_RUN"
```

生产 active 输出位于：

```text
/root/autodl-tmp/qubo_data_root/results/runs/mk14_adaptive_remote/active_docking/
```

中断后使用相同配置和 prepared 目录恢复。恢复会重新校验 active 配置 fingerprint 和两个旧 manifest 的 fingerprint；不一致时停止，不会混用不同 prepare 产物：

```bash
python scripts/run_active_experiment.py resume \
  --config "$CONFIG" \
  --data-root "$DATA_ROOT" \
  --prepared-run-directory "$PREPARED_RUN"
```

active rounds 结束后，只有 `finalize` 才读取旧 ligand manifest 的 hidden active/decoy label 并生成评价指标：

```bash
python scripts/run_active_experiment.py finalize \
  --config "$CONFIG" \
  --data-root "$DATA_ROOT" \
  --prepared-run-directory "$PREPARED_RUN" \
  --format markdown \
  --output "$DATA_ROOT/results/runs/mk14_adaptive_remote/active_docking/evaluation.md"
```

`validate`、`prepare`、`run`、`resume` 和 `finalize` 的职责分别是配置校验、调用旧 prepare、执行新 active rounds、恢复 checkpoint 和最终评估。生产入口不要求手工创建 `matrix.json`、`ligands.json`、`receptors.json` 或 `labels.json`；这些 JSON 是离线 masked replay 的测试/输入格式，不是从旧 full workflow 切生产 active 的前置条件。

### 12.3 生产审计与研究结论边界

每轮 checkpoint 至少记录 candidate pool、selected/revealed tasks、预测均值和方差、QUBO fingerprint、solver backend、观察任务数量和 docking 成本。state、active manifest、预测器、acquisition、QUBO 和 solver 都不得出现 hidden score 或 hidden label。未被 solver 选中的任务不会传给 docking adapter，也不会自动进入 state。

当 backend 为 `quantum_compatible_simulator` 时，审计必须记录 `quantum_hardware_used=false` 和 `quantum_execution_result=not_run`。模拟退火结果只能作为量子兼容模拟后端结果，不能称为量子结果。

生产 replay 或真实 docking 运行后，报告必须分别回答：预测模型是否预测未完成 score；批量 QUBO 是否优于逐任务 value-greedy；QUBO 是否优于 exact/CP-SAT 或其他强经典方法；量子后端是否产生端到端收益；收益是否来自量子优化而不是候选裁剪、预测模型或 warm-start。当前实现可以证明软件闭环、状态隔离、成本记账和恢复行为，但不能单独证明这些研究结论，也不能由 masked replay 推断真实 docking 一定节省成本。
