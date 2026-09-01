# 主动 ligand-receptor docking 生产集成设计

## 目标

在不改变既有 `prepare -> dock -> aggregate -> build_problem -> solve -> evaluate -> persist` 流程和 receptor-subset QUBO 语义的前提下，新增一条从既有 `prepare` 产物开始的预算约束主动 ligand-receptor docking 流程。

新的生产闭环为：

```text
prepare -> active_initialize -> warm_start -> active_rounds -> active_finalize
```

`prepare` 复用 `FullExperimentRunner` 已有的原始数据、配体 3D/PDBQT、受体对齐/PDBQT 和 docking box 实现。主动流程不调用旧的全量 `dock` 阶段，不等待完整矩阵，也不复用旧 receptor-subset QUBO。

## 路径合同

新配置引用一个 schema 3.0 的完整实验配置作为 preparation source。远程 MK14 的数据根目录和旧运行目录为：

```text
data_root: /root/autodl-tmp/qubo_data_root
prepared_run_directory: /root/autodl-tmp/qubo_data_root/results/runs/mk14_adaptive_remote
active_run_directory: /root/autodl-tmp/qubo_data_root/results/runs/mk14_adaptive_remote/active_docking
```

本地验证允许通过 CLI 覆盖 prepared run directory，以支持 `E:\Quant\remote_runs\mk14\adaptive_remote` 这种不位于本地 data root 下的历史产物目录。配置文件不写死本地或远程绝对路径。

## 数据和泄漏边界

旧 `prepared_ligands.csv` 和 `selected_receptors.csv` 是 preparation 输入。主动流程转换出不含 `label`、`active`、`decoy` 或 selection role 的可见 manifest。原始标签只保留在 final evaluator 的输入边界中，不进入状态、预测器、采集器、QUBO、solver 或 round audit。

主动状态只保存已揭示的 fused score。隐藏 score 不复制到状态或 round audit；生产执行器从未完成任务中接收策略选择后才调用 docking adapter。`state.json` 可序列化并恢复，恢复时重新验证所有可见数据边界。

## 真实 docking 协议

每个选中的 ligand-receptor 任务都执行旧 MK14 配置中的 3 个 seed。每个 seed 使用已有 `DockingAdapter`，但 batch 只包含本轮选中的 ligand。任务的 revealed score 是 3 个 seed 的 `pose_rank_1` representative score 的 median。

预算默认按 seed docking job-equivalent 计费：一个 ligand-receptor 任务的成本为 3。每轮和全局预算均包含 warm-start 成本；warm-start 的成本单独报告。adapter 原有完整流程仍保留标签列，主动路径使用无标签输入并由 active score parser 读取 score。

## Manifest 特征

配体特征从旧 manifest 的 SMILES 通过确定性 RDKit 描述符生成，并保留 scaffold。受体特征从本次 prepare 产生的 aligned PDB/PDBQT 生成；结构 cluster 采用固定阈值和稳定排序，不读取 docking score 或评价标签。MK14 的 warm-start 基准受体固定为当前真实 selected manifest 中的 `11OY`，若 manifest 不含该 ID 直接失败。

## 运行和恢复

新增 `scripts/run_active_experiment.py`，提供：

- `validate`：检查 active 配置、base full 配置、raw 输入或 prepared 产物、baseline receptor、3-seed 协议和输出路径；不执行 docking。
- `prepare`：调用现有 preparation 实现生成或恢复旧流程的准备产物。
- `run`：执行 warm-start 和逐轮主动 docking；每个 round 在提交 adapter 前落盘预测、采集、互补项、QUBO 和 selected task 审计，完成后原子更新状态。
- `resume`：从最后一个完整 round 的状态继续，拒绝不匹配的配置或输入 fingerprint。
- `finalize`：只使用已经揭示的 score 生成最终部分矩阵和报告；标签只在最终指标计算时读取。

主动产物写入旧 run directory 下的 `active_docking/`，不覆盖旧 `score_tables`、`matrices`、`problem.json` 或旧 summary。

## 预测和求解门

生产 runner 使用现有 Bayesian residual predictor、posterior acquisition、batch QUBO 和统一 solver adapter。预测模型只能拟合当前状态可见 score；prediction gate 必须来自授权的 masked development replay，gate 失败时生产 run 停止。默认实际 solver backend 为配置中的 `quantum_compatible_simulator` 时，报告中明确写出它是模拟后端，不称为量子硬件结果。

所有策略比较继续在离线 masked replay 中完成。生产 run 只执行一个冻结 strategy，不能为了比较策略在同一个真实 docking 预算中重复提交任务。候选裁剪、warm-start、预测模型和 solver 的收益分别记录。

## 停止条件和审计

生产流程在预算耗尽、无可行 batch、预测 gate 失败、solver 不可行、连续低边际收益或达到显式最大轮数时停止。每轮记录候选池 fingerprint、可用任务、选中任务、seed score table、fused score、累计成本、solver backend、配置 fingerprint 和停止原因，但不记录隐藏标签或未选任务 score。

## 测试范围

新增测试覆盖：

1. 旧 preparation manifest 到脱敏 active manifest 的转换；
2. 真实 conformer ID、`11OY` baseline 和 aligned structure cluster；
3. adapter 只收到 selected task，3-seed score fusion 正确；
4. 状态和 round checkpoint 的原子更新、恢复和配置 fingerprint 检查；
5. hidden label/score 不进入决策前输入；
6. 小型 fake adapter 的完整多轮 runner smoke test；
7. 旧 full workflow、旧 CLI 和现有 active replay 回归不变。

