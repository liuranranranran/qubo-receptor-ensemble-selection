# Linux 完整实验流程

从实验数据准备开始执行完整的受体集合选择流程。Uni-Dock 需要在 Linux 环境中执行。

## 1. 建立 Linux 运行环境

### 1.1 设置路径并拉取仓库

以下路径按服务器实际目录填写。`REPO_ROOT` 是代码仓库目录，`DATA_ROOT` 是外部实验数据根目录，`ENV_NAME` 是本次创建的 Conda 环境名称。

```bash
REPO_ROOT=/path/to/qubo-receptor-ensemble-selection
DATA_ROOT=/path/to/qubo_receptor_ensemble_experiment_data_20260815
ENV_NAME=qubo-receptor-ensemble

git clone https://github.com/Sinking-tenderness/qubo-receptor-ensemble-selection.git "$REPO_ROOT"
cd "$REPO_ROOT"
```

### 1.2 根据环境文件创建 Conda 环境

仓库依赖以 `environment/environment.yml` 为准。创建环境时直接使用该文件，不要手工逐项安装 Python 依赖：

```bash
conda env create --name "$ENV_NAME" --file "$REPO_ROOT/environment/environment.yml"
conda activate "$ENV_NAME"
```

### 1.3 安装仓库包并测试环境

在激活的环境中以 editable 方式安装仓库包，然后检查 Python 依赖、Meeko、Uni-Dock、实验入口、字节码编译和 pytest：

```bash
python -m pip install --editable .

python --version
python -c "import numpy, pandas, scipy, sklearn, xgboost, rdkit, meeko, gemmi; print('Python dependencies: OK')"
python -c "from qubo_receptor_ensemble.preparation import find_meeko_script; print(find_meeko_script())"
command -v unidock
unidock --help >/dev/null
python scripts/run_experiment.py --help
python -m compileall -q src scripts
python -m pytest -q --basetemp /tmp/qubo-receptor-ensemble-selection-pytest -o cache_dir=/tmp/qubo-receptor-ensemble-selection-pytest-cache
```

### 1.4 准备外部原始数据

实验数据不复制进仓库。将数据包放在 `DATA_ROOT`，并确认原始输入位于 `DATA_ROOT/data/raw`：

```bash
test -d "$DATA_ROOT/data/raw"
find "$DATA_ROOT/data/raw" -maxdepth 2 -type d | sort
```

完整流程从 `data/raw/external_targets/<target>_dude/` 读取 active/decoy ISM、参考受体 PDB 和晶体配体，从 `data/raw/rcsb/<target>/` 读取 RCSB 结构池。`prepare` 会从这些 raw 输入生成配体中间数据、受体 PDB、对齐后的受体 PDB、受体 PDBQT、受体 manifest 和 docking box。首次运行不要把旧的 `results/` 或 `data/processed/` 当作 raw 输入。

`--data-root` 是外部实验数据根目录。配置文件中的 `sources` 和 `paths` 相对路径都相对于该目录解析，运行结果默认也写入该数据包。

## 2. 完整流程入口

完整实验的主入口是 `scripts/run_experiment.py`。FA10 和 EGFR 使用以下配置：

```text
configs/experiments/stage102a_fa10_full.json
configs/experiments/stage102a_egfr_full.json
```

以 FA10 为例，先完成第 1 节的环境和 raw 数据检查，再执行：

```bash
CONFIG="$REPO_ROOT/configs/experiments/stage102a_fa10_full.json"

python scripts/run_experiment.py validate \
  --config "$CONFIG" \
  --data-root "$DATA_ROOT"

python scripts/run_experiment.py plan \
  --config "$CONFIG" \
  --data-root "$DATA_ROOT"

python scripts/run_experiment.py run \
  --config "$CONFIG" \
  --data-root "$DATA_ROOT"
```

完整流程的固定阶段顺序是：

```text
prepare -> dock -> aggregate -> build_problem -> solve -> evaluate -> persist
```

各阶段职责如下：

1. `prepare`：读取 raw ISM、参考受体、晶体配体和 RCSB 结构池，选择配体和受体，准备配体 3D/PDBQT，复制或解析 RCSB PDB，完成受体结构对齐和 PDBQT 准备，并根据晶体配体坐标计算本次运行的 docking box。
2. `dock`：调用激活环境中的 Uni-Dock，按配置的受体、配体和 seed 重新 docking，写出每个 seed 的 score table 和 pose 文件。
3. `aggregate`：读取多个 seed 的 score table，生成长表、primary median matrix 和 sensitivity minimum matrix。
4. `build_problem`：读取 primary matrix、配体标签和受体 manifest，按照配置的 utility metric、权重和 QUBO 方法构造问题。
5. `solve`：调用配置的 solver 求解受体子集。
6. `evaluate`：在配置指定的数据划分上计算 BEDROC20、ROC-AUC、PR-AUC、EF1%、EF5% 和 EF10% 等指标。
7. `persist`：保存 selection、evaluation、配置快照、运行 manifest 和 summary。

默认选择目标是 BEDROC20，即 `utility_metric: "bedroc"` 和 `bedroc_alpha: 20.0`。ROC-AUC 作为辅助指标记录，但不参与默认的 QUBO 选择或自适应 `k` 决策。

### 2.1.7 自适应构象数（可选）

如需让流程先判断目标蛋白是否需要多构象，可在 `problem` 中显式启用
`k_policy`。该策略从 `k=1` 开始，按候选顺序逐步评估；`k=1` 只是搜索起点，不是固定的
最终选择。每一步只为当前候选生成 inner scaffold fold 的 OOF 预测，并与前一个候选计算
相邻边际增益。每个相邻候选对共享该转换自己的 scaffold bootstrap 抽样，不同转换不要求
共享同一组抽样。第 k 步的边际净收益为：

```text
risk_adjusted_gain(k-1 -> k) = mean_OOF_gain(k, k-1) - cost_per_receptor
```

每个转换会记录 `supported`、`uncertain` 或 `harmful` 状态。`supported` 要求边际净收益
超过 `minimum_effect` 且 bootstrap 正收益概率不低于 `required_probability`；默认
`required_probability` 为 `0.9`。`harmful` 表示负增益概率严格大于 0.5，会立即停止。
`uncertain` 不立即否决后续，但只允许紧接着计算一个后续候选作为 lookahead；如果该
lookahead 为 `supported`，则清除本次确认状态并继续向下探索，而不是立即停止。如果
lookahead 仍为 `uncertain`，则停止；`harmful` 仍然立即停止。这样可以识别“第一步信号弱、
第二步增益明确”后仍存在更大构象数收益的情况，同时控制额外计算量。

候选必须位于没有 `harmful` transition 的路径上，且当前 transition 必须为 `supported`；
累计净收益只用于可用路径上的候选排序，不能跨过 `harmful` transition 选择更大的 k。
差异在 `selection_tie_tolerance` 内时偏好较小 k。`bootstrap_lcb` 和 rescue
contrast 作为审计指标保存；只有显式设置 `require_rescue_contrast: true` 时，rescue
contrast 才会成为选择硬门槛。候选 k 不限制为 1、2、3：可以显式配置为从 1 开始的连续更大
范围；不填写 `candidates` 时默认检查 `1..selection.receptor_count`，但仍会按上述停止规则
提前结束。

```json
"problem": {
  "type": "receptor_subset",
  "strategy": "qubo",
  "target_size": 1,
  "weights": {"redundancy": 0.25, "count": 0.10, "size": 1.0},
  "k_policy": {
    "mode": "adaptive",
    "selector": "risk_adjusted_oof",
    "candidates": [1, 2, 3, 4, 5, 6],
    "scaffold_field": "scaffold_smiles",
    "inner_fold_count": 3,
    "bootstrap_iterations": 1000,
    "lower_quantile": 0.05,
    "minimum_effect": 0.0,
    "required_probability": 0.9,
    "cost_per_receptor": 0.0,
    "selection_tie_tolerance": 0.0,
    "require_rescue_contrast": false,
    "rescue_fractions": [0.01, 0.05],
    "random_seed": 0
  }
}
```

`lower_quantile: 0.05` 表示审计用的单侧 95% 下置信界。自适应选择只使用开发数据的内层
scaffold fold，不读取 outer/test 标签；当前版本只支持单问题 QUBO，不支持
`problem.mode: "compare"`。运行目录会保存 `adaptive_cardinality.json`，并在
`problem.json`、`selection.json`、`evaluation.json`、`summary.json` 和 `manifest.json` 中
保留相同的决定审计信息。其中 `transitions` 保存每个已计算的相邻转换，
`evaluated_candidates` 保存实际计算到的候选序列，`stop_reason` 说明是达到候选上限、
遇到 harmful transition，还是连续不确定确认后停止；同时保留累计候选净收益、
bootstrap 状态和 rescue 诊断。固定 `problem.target_size` 且不配置 `k_policy` 时，旧流程不变。

如果需要在同一套 docking 矩阵上比较多个历史 QUBO 方法，可以将 `problem.mode` 设为 `compare` 并列出 `methods`。比较从 `build_problem` 开始即可，不需要重新执行 `prepare`、`dock` 或 `aggregate`。

`validate` 只检查当前起始阶段需要的输入，`plan` 只展开阶段和输出路径；这两个命令都不会启动结构准备或 docking。

### 2.1 配体和受体的选择逻辑

完整配置默认使用 `selection.ordering: "scaffold_hash_allocation"`。

配置中的核心字段是：

```json
"selection": {
  "receptor_count": 13,
  "ligand_count": 600,
  "label_counts": {
    "active": 120,
    "decoy": 480
  },
  "ordering": "scaffold_hash_allocation",
  "allocation": {
    "outer_fold_count": 5,
    "minimum_label_counts_per_outer_fold": {
      "active": 20,
      "decoy": 80
    }
  }
}
```

#### 2.1.1 配体解析

程序分别读取 `sources.active_ism` 和 `sources.decoy_ism`。ISM 每个非空行至少包含 SMILES 和 source molecule ID，第三列如果存在则作为 source extra ID 保存。程序用 RDKit 解析 SMILES，并为每条记录保存原始 SMILES、canonical SMILES、标签和原始行号；解析失败或文件为空会直接报错。

#### 2.1.2 分组准则

程序为每条配体计算无手性的 Bemis-Murcko scaffold，并通过三个键建立连通组：`source_molecule_id`、`canonical_smiles` 和 `scaffold_smiles`。任意一个键相同的记录都会进入同一组。后续选择以组为单位，不能把同一组拆到不同选择结果或 outer fold 中。

#### 2.1.3 active 选择准则

active 组按固定、可复现的顺序排序，并以整组为单位选择，直到恰好达到配置的 active 配额。选择结果不依赖文件系统顺序、随机状态或 docking 分数，因此相同 raw 输入和配置会得到相同结果。

#### 2.1.4 decoy 选择准则

程序收集已选 active 的 scaffold 集合，丢弃任何与 active scaffold 重叠的 decoy 组，再按同样的可复现规则选择 decoy，直到恰好达到配置的 decoy 配额。若剩余合格组无法组成指定数量，`prepare` 会失败而不会静默降低配额。

#### 2.1.5 fold 分配和输出

active 和 decoy 分别按组大小从大到小处理，每个组分配到当前配体数最少的 outer fold，同数时取编号较小的 fold。程序检查每个 fold 至少包含配置要求的 20 个 active 和 80 个 decoy，并检查 ligand ID 不重复。选择结果写入运行目录的 `source_ligands.csv`，其中包含 source 行号、source ID、canonical SMILES、scaffold、allocation group、hash、label、split 和 outer fold；无分数审计写入 `source_ligand_allocation_summary.json`。

#### 2.1.6 其他选择模式

`manifest_order` 仍然保留为显式选项。启用后，程序分别按 active 和 decoy raw ISM 的文件行顺序截取配置的数量，例如前 120 个 active 加前 480 个 decoy；该模式不做 scaffold hash 排序、不排除 active/decoy scaffold 重叠，也不使用 docking 分数。

`seeded_sample` 按配置 seed 对 active 和 decoy 分别抽样。`preselected_manifest` 从配置指定的 ligand manifest 读取。两者都不改变 active/decoy 标签和配置配额。

受体选择独立于配体选择。`prepare` 生成受体 manifest 后，只保留 `status` 为 `ok` 的记录，按 manifest 顺序取前 `selection.receptor_count` 个；有效受体数量不足时直接报错。

## 3. 示例数据

示例的数据和默认规模如下：

| 目标 | 受体构象数 | 配体总数 | active | decoy | docking seed 数 |
|---|---:|---:|---:|---:|---:|
| FA10 | 13 | 600 | 120 | 480 | 3 |
| EGFR | 12 | 600 | 120 | 480 | 3 |

两个示例都从外部数据包的 raw 目录读取配体和受体。FA10 的配置路径例如：

```text
data/raw/external_targets/fa10_dude/fa10/actives_final.ism
data/raw/external_targets/fa10_dude/fa10/decoys_final.ism
data/raw/external_targets/fa10_dude/fa10/receptor.pdb
data/raw/external_targets/fa10_dude/fa10/crystal_ligand.mol2
data/raw/rcsb/fa10/
```

FA10 的 RCSB CIF 候选也可以位于该目录下的 `coordinate_pool/`；程序会递归发现 `.cif` 和 `.pdb`，同一结构 ID 优先使用 CIF。EGFR 配置使用对应的 `data/raw/external_targets/egfr_dude/egfr/` 和 `data/raw/rcsb/egfr/`。这些路径都在 `DATA_ROOT` 下解析，而不是在仓库根目录下寻找。

## 4. Docking 配置

默认配置使用激活 Conda 环境中的 Uni-Dock，并重新执行 docking：

```json
"docking": {
  "redock": true,
  "engine": "unidock",
  "executable": "unidock",
  "seeds": [20260821, 20260822, 20260823]
}
```

`executable: "unidock"` 表示从当前激活环境的 `PATH` 查找 Uni-Dock。若 `command -v unidock` 失败，应先检查环境创建和激活是否成功。

完整模式的 box 只配置计算规则，不填写固定中心坐标：

```json
"box": {
  "method": "ligand_bounds",
  "padding": 5.0,
  "minimum_size": [22.0, 22.0, 28.0]
}
```

`prepare` 会用 raw `crystal_ligand.mol2`/`.sdf` 的坐标计算中心和尺寸，写入 `docking_box.json`，随后把该结果传给 docking adapter。`allow_bad_res` 只在明确配置时启用，并会在受体准备审计中记录 Meeko 删除的非模板残基。

VinaCPU 仍可作为显式的替代适配器，但不是当前默认流程：

```json
"docking": {
  "redock": true,
  "engine": "vina_cpu",
  "executable": "/path/to/vina"
}
```

不同 engine 的 score table 不得在同一次聚合中混用。完整模式要求 `docking.redock` 为 `true`；只有显式使用 `workflow_mode: "reference_replay"` 并提供已有 score table 或 matrix 时，才允许关闭 redock。

## 5. 从中间阶段继续

可以在 JSON 中设置起止阶段，也可以用命令行覆盖：

```bash
python scripts/run_experiment.py run --config "$CONFIG" --data-root "$DATA_ROOT" --from aggregate --to persist
```

从中间阶段开始时，运行器只使用配置中明确声明的前置路径，不会自动搜索仓库或数据包中的旧矩阵：

| 起始阶段 | 必须存在的前置路径 |
|---|---|
| `prepare` | raw `.ism`、参考受体 PDB、晶体配体和 RCSB 结构目录 |
| `dock` | `paths.prepared_ligand_manifest`、`paths.selected_receptor_manifest`、`paths.docking_box` |
| `aggregate` | 上述两个 manifest、`paths.score_tables` |
| `build_problem` | `paths.primary_matrix`、`paths.selected_receptor_manifest`；启用自适应 `k` 时还需要 `paths.prepared_ligand_manifest` |
| `solve` | `paths.problem` |
| `evaluate` | `paths.selection` |
| `persist` | `paths.evaluation` |

`--from` 和 `--to` 必须遵守配置声明的 canonical 阶段顺序。若重新执行已经完成的阶段，先确认输出目录是否需要使用 `--overwrite`。

## 6. 输出、续跑和覆盖

默认运行目录由配置中的 `paths.run_directory` 指定，并相对于 `DATA_ROOT` 解析。典型结构如下：

```text
results/runs/stage102a_fa10_full_local/
  source_ligands.csv
  prepared_ligands.csv
  receptors/
    source_pdb/
    aligned_pdb/
    prepared/
  receptor_preparation_audit.json
  selected_receptors.csv
  docking_box.json
  score_tables/
  matrices/
    aggregated_long.csv
    primary_median_matrix.csv
    sensitivity_minimum_matrix.csv
  problem.json
  adaptive_cardinality.json  # 仅在启用自适应 k 时生成
  selection.json
  evaluation.json
  config.snapshot.json
  manifest.json
  summary.json
```

中断后可以在同一环境中续跑：

```bash
python scripts/run_experiment.py run --config "$CONFIG" --data-root "$DATA_ROOT" --resume
```

已有输出默认不会覆盖。只有确认需要重写当前运行目录时才使用 `--overwrite`。文件 SHA-256 和其他 provenance 由程序自动记录，不需要手工计算或填写。

`docking_box.json` 的中心来自 raw 晶体配体坐标包围盒中心，尺寸为 `max(坐标范围 + 2 * padding, minimum_size)`。它记录晶体配体路径和 SHA-256；`docking` 阶段只使用本次运行生成的 box，不接受旧的 `common_box.json` 或配置中预先填写的六个坐标值。

## 7. 旧入口边界

`scripts/run_pipeline.py` 和 `configs/pipelines/*.json` 是 schema `2.0` 的 matrix replay 兼容入口。它们从已有 score matrix 开始，不负责配体准备和 docking，也不是本文的默认完整实验入口。

旧入口的 Linux 执行方式见 [配置说明](../configs/pipelines/README.md)。新实验统一使用：

```text
scripts/run_experiment.py
configs/experiments/*.json
```

## 8. 开发者检查

在第 1 节创建并激活的 Conda 环境中、仓库根目录执行轻量检查：

```bash
cd "$REPO_ROOT"
python scripts/run_experiment.py --help
python -m pytest -q --basetemp /tmp/qubo-receptor-ensemble-selection-pytest
git diff --check
```

回归测试不会启动 600 个配体乘多受体乘多 seed 的生产 docking。完整运行前应先执行 `validate` 和 `plan`，确认数据根目录、配置、engine 和输出目录。

## 9. 从 canonical prepare 切出 active docking

新的主动 ligand-receptor docking 不改变本文件前面定义的 full workflow，也不复用旧的 receptor-subset problem。它复用本流程的 `prepare` 产物，然后进入独立阶段：

```text
prepare -> active_initialize -> warm_start -> active_rounds -> active_finalize
```

主动配置为 `configs/active_docking/mk14_active_docking_remote.json`，生产入口为 `scripts/run_active_experiment.py`。它读取旧运行目录中的：

```text
prepared_ligands.csv
selected_receptors.csv
```

不要求用户手工准备完整 score matrix。旧 ligand manifest 的 label 不会进入 active manifest、预测器、采集器、QUBO 或 solver；只有 `finalize` 才读取 label 计算最终评价。

远程服务器上的路径映射如下：

```text
本地数据根目录示意：E:\Quant\qubo_receptor_ensemble_experiment_data
远程数据根目录：/root/autodl-tmp/qubo_data_root

本地旧运行目录示意：E:\Quant\remote_runs\mk14\adaptive_remote
远程旧运行目录：/root/autodl-tmp/qubo_data_root/results/runs/mk14_adaptive_remote
```

远程服务器先按本文件前面的环境说明安装仓库和依赖。若旧 prepare 尚未完成，只生成 prepare：

```bash
REPO_ROOT=/root/qubo-receptor-ensemble-selection
DATA_ROOT=/root/autodl-tmp/qubo_data_root
BASE_CONFIG="$REPO_ROOT/configs/experiments/mk14_adaptive_remote.json"
ACTIVE_CONFIG="$REPO_ROOT/configs/active_docking/mk14_active_docking_remote.json"

cd "$REPO_ROOT"
conda activate qubo-receptor-ensemble
python scripts/run_experiment.py run \
  --config "$BASE_CONFIG" \
  --data-root "$DATA_ROOT" \
  --from prepare \
  --to prepare
```

已有旧 prepare 产物时，直接从它切出 active workflow，不要覆盖旧 canonical 运行目录：

```bash
PREPARED_RUN="$DATA_ROOT/results/runs/mk14_adaptive_remote"

python scripts/run_active_experiment.py validate \
  --config "$ACTIVE_CONFIG" \
  --data-root "$DATA_ROOT" \
  --prepared-run-directory "$PREPARED_RUN"

python scripts/run_active_experiment.py run \
  --config "$ACTIVE_CONFIG" \
  --data-root "$DATA_ROOT" \
  --prepared-run-directory "$PREPARED_RUN"
```

三个 Uni-Dock seed `20260821`、`20260822`、`20260823` 的 `pose_rank=1` score 使用 median；每个 task 成本按三个 seed 计费。active 结果写到：

```text
/root/autodl-tmp/qubo_data_root/results/runs/mk14_adaptive_remote/active_docking/
```

中断后运行 `resume`，完成后运行 `finalize`。旧入口 run_active_docking.py 仍然只负责离线完整矩阵 masked replay；它与上面的生产入口不是同一个输入流程。
