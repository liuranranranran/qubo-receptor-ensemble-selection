# 脚本说明

## 当前主入口

完整实验使用 `run_experiment.py`。Uni-Dock 需要在 Linux 环境中执行。首次建立 Conda 环境、安装仓库包和检查依赖，请按[完整实验流程](../docs/experiment_workflow_zh.md)第 1 节执行：

```bash
REPO_ROOT=/path/to/qubo-receptor-ensemble-selection
DATA_ROOT=/path/to/qubo_receptor_ensemble_experiment_data_20260815
cd "$REPO_ROOT"
test -d "$DATA_ROOT/data/raw"

CONFIG="$REPO_ROOT/configs/experiments/stage102a_fa10_full.json"

python scripts/run_experiment.py validate --config "$CONFIG" --data-root "$DATA_ROOT"
python scripts/run_experiment.py plan --config "$CONFIG" --data-root "$DATA_ROOT"
python scripts/run_experiment.py run --config "$CONFIG" --data-root "$DATA_ROOT"
```

`run_experiment.py` 会自动加载仓库内的 `src`，不需要手工设置
`PYTHONPATH`。`--data-root` 应指向前面设置的 `$DATA_ROOT`。

可用命令：

- `validate`：检查当前起始阶段需要的前置路径；
- `plan`：展开阶段、engine、redock 和输出路径，不启动实验；
- `run`：执行实际流程，支持 `--from`、`--to`、`--resume`、`--overwrite`。

## 阶段实现

核心实现位于 `src/qubo_receptor_ensemble/`：

- `full_workflow.py`：schema 3.0 配置、源数据选择和阶段边界；
- `docking_adapters.py`：Uni-Dock/VinaCPU 适配器；
- `experiment.py`：准备、docking、聚合、QUBO、评估和归档；
- `matrix.py`：long score table、seed 聚合和矩阵构造。

旧脚本仍保留为兼容或独立工具：

- `run_pipeline.py`：schema 2.0 matrix replay，不是完整实验入口；
- `prepare_ligand_3d_sdf.py`、`batch_prepare_ligand_pdbqt_parallel.py`：通用配体准备工具；
- `prepare_receptor.py`：通用受体准备和审计工具；
- `batch_vina_docking_parallel.py`：旧的 VinaCPU 单受体批量入口；
- `build_score_matrix.py`、`aggregate_seed_replicates.py`：独立矩阵工具。

新实验不要把旧 matrix replay 的输入、seed、box 或运行结果混入完整流程。
不同 engine 的 score 也不能混在同一次聚合中。

离线 masked active-docking replay 的输入准备、predictor gate、同预算策略比较和泄漏边界见[主动 docking replay 操作与审计](../docs/active_docking_replay_zh.md)。从已有 active 生产结果生成匿名 replay 输入时，使用：

```bash
python scripts/prepare_active_replay_inputs.py \
  --active-manifest /path/to/active_docking/active_manifest.json \
  --matrix /path/to/run/matrices/primary_median_matrix.csv \
  --output /path/to/run/replay_inputs_anon
```

## 旧入口的 Linux 调用

旧 schema 2.0 pipeline 只从已有矩阵开始：

```bash
cd "$REPO_ROOT"

python scripts/run_pipeline.py validate \
  --config configs/pipelines/stage102a_fa10_development_selection.json \
  --root .
python scripts/run_pipeline.py plan \
  --config configs/pipelines/stage102a_fa10_development_selection.json \
  --root .
python scripts/run_pipeline.py run \
  --config configs/pipelines/stage102a_fa10_development_selection.json \
  --root .
```
