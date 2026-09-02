# 主动 ligand-receptor docking replay 操作与审计

本文档把旧 full workflow 的 `prepare` 产物切出为离线 masked replay。它解决的是“已有完整 docking 结果后，模拟有限预算下逐轮选择任务”的研究验证问题，不会启动真实 docking、远程任务或量子硬件。

## 1. 两条输入路径

旧流程仍然保持如下 canonical 阶段：

```text
prepare -> dock -> aggregate -> build_problem -> solve -> evaluate -> persist
```

主动生产流程从旧 `prepare` 产物开始，独立写入 `active_docking/`：

```text
prepare -> active_initialize -> warm_start -> active_rounds -> active_finalize
```

离线 replay 不要求手工创建 `matrix.json`、`ligands.json`、`receptors.json` 或 `labels.json`。这些文件由本仓库的输入准备脚本从已有运行产物生成：

```text
active_manifest.json
    来源：旧 prepare 产物桥接后的 active 运行目录

matrices/primary_median_matrix.csv
    来源：旧 full workflow 的 aggregate 阶段

prepare_active_replay_inputs.py
    输出：replay_inputs_anon/{ligands,receptors,matrix,labels,id_map}.json
```

对于 MK14，目录映射为：

| 用途 | 本地示意 | 远程服务器 |
|---|---|---|
| 数据根目录 | `E:\Quant\qubo_receptor_ensemble_experiment_data` | `/root/autodl-tmp/qubo_data_root` |
| 旧运行目录 | `E:\Quant\remote_runs\mk14_adaptive_remote` | `/root/autodl-tmp/qubo_data_root/results/runs/mk14_adaptive_remote` |
| active manifest | 旧运行目录下的 `active_docking/active_manifest.json` | 同左 |
| 完整 score matrix | 旧运行目录下的 `matrices/primary_median_matrix.csv` | 同左 |

如果旧运行目录只有 `prepare` 产物，没有 aggregate 矩阵，不能直接做完整矩阵 replay；应先按旧流程完成需要的 aggregate 阶段。生产 active docking 则只需要 `prepared_ligands.csv` 和 `selected_receptors.csv`，不走本文件的 replay 输入准备。

## 2. 远程服务器准备

以下命令在远程 Linux 服务器执行。命令中的 `python` 是远程环境，不是本地 Windows 命令。

```bash
REPO_ROOT=/root/qubo-receptor-ensemble-selection
DATA_ROOT=/root/autodl-tmp/qubo_data_root
RUN_ROOT=$DATA_ROOT/results/runs/mk14_adaptive_remote
ACTIVE_RUN=$RUN_ROOT/active_docking
REPLAY_ROOT=$RUN_ROOT/replay_inputs_anon
REPLAY_CONFIG=$REPO_ROOT/configs/active_docking/mk14_masked_replay.json

cd "$REPO_ROOT"
conda activate qubo-receptor-ensemble
python -m pip install --editable .
```

如果旧 canonical prepare 还没有完成，先只运行旧流程的 `prepare`：

```bash
python scripts/run_experiment.py run \
  --config "$REPO_ROOT/configs/experiments/mk14_adaptive_remote.json" \
  --data-root "$DATA_ROOT" \
  --from prepare \
  --to prepare
```

这一步是旧流程的输入准备，不是新的 active 选择器，也不会生成旧 receptor-subset QUBO。已有准备结果时不要用 `--overwrite` 重写旧运行目录。

## 3. 生成匿名 replay 输入

先确认两个源文件存在：

```bash
test -f "$ACTIVE_RUN/active_manifest.json"
test -f "$RUN_ROOT/matrices/primary_median_matrix.csv"
```

然后生成输入：

```bash
python scripts/prepare_active_replay_inputs.py \
  --active-manifest "$ACTIVE_RUN/active_manifest.json" \
  --matrix "$RUN_ROOT/matrices/primary_median_matrix.csv" \
  --output "$REPLAY_ROOT"
```

输出目录内容及安全边界如下：

```text
replay_inputs_anon/
  ligands.json   # L0000 等不透明 ID、scaffold、可见 ligand 特征
  receptors.json # receptor ID、cluster、可见 receptor 特征
  matrix.json    # 只有 score rows，不含 label
  labels.json    # 评价边界使用的 opaque-ID -> active/decoy 映射
  id_map.json    # 仅供运行后分析，不传给 predictor、acquisition 或 solver
```

脚本会检查 ligand 和 receptor 集合、矩阵完整性、重复 ligand、有限 score 和合法标签。源 ligand ID 以及 PDBQT 路径不会进入 predictor 输入。源 CSV 中的 `label` 只用于单独生成 `labels.json`；它不参与 warm-start、预测、acquisition、候选裁剪、QUBO 或 solver。

## 4. 配置验证和 predictor gate

本实验使用与已完成生产实验相同的总成本和批成本：总预算 `6000.0`，每轮批成本 `300.0`，每个任务成本 `3.0`。warm-start 成本也计入总预算，但 warm-start 是独立的初始化基线，不能作为量子优势证据。

先验证独立 replay 配置：

```bash
python scripts/run_active_docking.py validate \
  --config "$REPLAY_CONFIG"
```

再运行 masked score prediction gate：

```bash
python scripts/run_active_docking.py predict \
  --config "$REPLAY_CONFIG" \
  --matrix "$REPLAY_ROOT/matrix.json" \
  --ligand-manifest "$REPLAY_ROOT/ligands.json" \
  --receptor-manifest "$REPLAY_ROOT/receptors.json" \
  --format json \
  --output "$REPLAY_ROOT/prediction_gate.json"
```

检查 gate：

```bash
python -c 'import json; p=json.load(open("'"$REPLAY_ROOT"'/prediction_gate.json")); print("passed=", p["passed"]); print("primary_rmse=", p["primary_report"]["rmse"]); print("baseline_rmse=", {k:v["rmse"] for k,v in p["baseline_reports"].items()})'
```

主模型是具有显式后验协方差的 Bayesian linear residual model，实际预测目标为：

$$
\Delta_{lr}=s_{lr}-s_{l,r0},\qquad \hat{s}_{lr}=s_{l,r0}+\hat{\Delta}_{lr}
$$

它只使用当前可见 score 和 manifest 特征，输出均值、方差和确定性 posterior samples。gate 同时报告 Bayesian residual、nearest receptor 和 observed-score mean 基线的 masked score 误差。若 `passed` 为 `false`，应停止解释后续 QUBO 结果，不要强行运行生产 docking。

## 5. 运行 replay 和策略比较

`replay` 会用同一 warm-start、同一候选集合、同一 QUBO 约束和同一预算运行配置中的所有策略。只有任务被策略选中后，replay 才从完整 matrix 的外部 oracle 取出对应 score 并写入部分观测状态。

```bash
python scripts/run_active_docking.py replay \
  --config "$REPLAY_CONFIG" \
  --matrix "$REPLAY_ROOT/matrix.json" \
  --ligand-manifest "$REPLAY_ROOT/ligands.json" \
  --receptor-manifest "$REPLAY_ROOT/receptors.json" \
  --labels "$REPLAY_ROOT/labels.json" \
  --format json \
  --output "$REPLAY_ROOT/replay.json"
```

需要 Markdown 或 CSV 审计表时，使用同一输入再次调用 `compare`：

```bash
python scripts/run_active_docking.py compare \
  --config "$REPLAY_CONFIG" \
  --matrix "$REPLAY_ROOT/matrix.json" \
  --ligand-manifest "$REPLAY_ROOT/ligands.json" \
  --receptor-manifest "$REPLAY_ROOT/receptors.json" \
  --labels "$REPLAY_ROOT/labels.json" \
  --format markdown \
  --output "$REPLAY_ROOT/comparison.md"
```

配置中的策略含义是：

| 策略 | 解释 |
|---|---|
| `value_greedy` | 逐任务单位成本价值贪心，作为主要强基线 |
| `greedy_one_swap` | 贪心后执行局部一换一 |
| `simulated_annealing` | 在任务变量上运行可行模拟退火 |
| `quantum_compatible_simulator` | 接受 QUBO 协议的模拟适配器，实际复用模拟退火 |

当前配置故意不把大候选池上的 `exact` 放进生产规模 replay。现有 `exact` 是指数枚举，只适合小候选集验证；CP-SAT/MILP 依赖未安装时必须明确记录 unavailable，不能静默降级。

## 6. 审计结果

重点检查 replay JSON 中的以下字段：

```bash
python -c 'import json; p=json.load(open("'"$REPLAY_ROOT"'/replay.json")); print(p["metadata"]); [(print(s["name"], s["evaluation"], len(s["task_sequence"]))) for s in p["strategies"]]'
```

必须满足：

- `candidate_pool_is_shared` 为 `true`，所有策略的 candidate pool 相同；
- `oracle_scores_used_only_after_selection` 为 `true`；
- `hidden_labels_used_only_for_final_evaluation` 为 `true`；
- `real_docking_executed` 为 `false`；
- `quantum_hardware_used` 为 `false`；
- 每轮 `revealed_tasks` 是 `selected_tasks` 的子集；
- round audit 不包含 hidden score 或 hidden label。

候选裁剪必须单独归因。`metadata.candidate_pruning` 中记录未完成任务数、候选数、`candidate_cap` 和裁剪规则。要做裁剪消融，应复制本配置，只改变 `candidate_cap`，并保持 seed、预算、warm-start、预测器、约束和策略不变；候选池变化带来的收益不能记到 QUBO 或量子后端名下。

预测、warm-start 和候选裁剪都要单独报告。不能用“QUBO 能量更低”“解可行”或“模拟退火结果”替代端到端策略比较。

## 7. 真实 active docking 的边界

真实 production active docking 使用另一套入口和配置，不读取本节匿名 replay JSON：

```bash
ACTIVE_CONFIG="$REPO_ROOT/configs/active_docking/mk14_active_docking_remote.json"
PREPARED_RUN="$RUN_ROOT"

python scripts/run_active_experiment.py validate \
  --config "$ACTIVE_CONFIG" \
  --data-root "$DATA_ROOT" \
  --prepared-run-directory "$PREPARED_RUN"

python scripts/run_active_experiment.py run \
  --config "$ACTIVE_CONFIG" \
  --data-root "$DATA_ROOT" \
  --prepared-run-directory "$PREPARED_RUN"
```

该命令会在选中任务后调用 Uni-Dock，必须确认用户确实要运行真实 docking。离线 replay 的默认 CLI 永远不会启动真实 docking；本仓库验证阶段也不运行真实量子硬件。

## 8. 研究结论和停止条件

每次报告必须分别回答：

1. predictor 是否在 scaffold-disjoint 或结构分层 masked score 上预测了未完成 score，且误差/校准优于简单基线；
2. 带 batch interaction 的 QUBO 是否在相同候选集和预算下优于逐任务 value-greedy；
3. QUBO 是否优于 exact、CP-SAT/MILP 或其他强经典方法；
4. 量子后端是否带来端到端收益；
5. 收益是否来自量子优化，而不是 predictor、warm-start、candidate pruning 或后处理。

当前软件闭环不能证明上述任何量子优势结论。尤其不能把 `quantum_compatible_simulator` 或 simulated annealing 称为量子执行，也不能从 replay 成功推断真实 docking 一定节省成本。

应在实验开始前冻结失败停止条件：预算耗尽、连续轮边际效用低于阈值、排名稳定性达到阈值、预测区间达到目标、predictor gate 失败，或 solver 可行性/时间门失败。触发 predictor gate 或批量收益门时，应停止研究解释并记录失败原因，不得为了得到结果而调整惩罚系数或 outer/test 后验选择策略。
