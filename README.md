# QUBO Receptor Ensemble Selection

面向受体构象子集选择的研究代码。当前主入口从配体结构和受体 manifest
开始，在本机重新 docking，生成 score matrix 后执行 QUBO、评估和归档。

## 运行环境和路径

Uni-Dock 需要在 Linux 环境中执行。首次建立 Conda 环境、安装仓库包和检查依赖，请按[完整实验流程](docs/experiment_workflow_zh.md)第 1 节执行。仓库和数据路径通过变量提供：

```bash
REPO_ROOT=/path/to/qubo-receptor-ensemble-selection
DATA_ROOT=/path/to/qubo_receptor_ensemble_experiment_data_20260815

cd "$REPO_ROOT"
test -d "$DATA_ROOT/data/raw"
python --version
command -v unidock
```

默认配置中的 `docking.executable` 为 `unidock`，从激活环境的 `PATH` 查找。

## 快速运行

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

默认流程为：

```text
prepare -> dock -> aggregate -> build_problem -> solve -> evaluate -> persist
```

EGFR 使用 `configs/experiments/stage102a_egfr_full.json`。FA10 和 EGFR 示例
分别使用 13 和 12 个受体、每个目标 600 个配体，以及 3 个 Uni-Dock seed。

## 中间阶段继续

跳过的阶段必须在配置 `paths` 中给出前置文件或目录，运行器不会自动搜索
仓库中的旧矩阵：

```bash
python scripts/run_experiment.py run \
  --config "$CONFIG" \
  --data-root "$DATA_ROOT" \
  --from aggregate \
  --to persist
```

从 `aggregate` 开始至少需要准备好的 ligand manifest、receptor manifest 和
score tables；从 `build_problem` 开始需要 primary matrix；从 `solve` 开始
需要 problem；从 `evaluate` 开始需要 selection。

默认 `docking.redock=true`、`docking.engine=unidock`。完整模式下不使用已有
score matrix 代替 docking。只有显式使用
`workflow_mode: "reference_replay"`，并提供已有 score tables 或 matrix，才
属于重放流程。

## 目录

```text
configs/experiments/  schema 3.0 的完整实验配置
configs/pipelines/   schema 2.0 的旧 matrix replay 配置
data/                 仓库内保留的小型输入和说明
results/              仓库内已有结果，不作为默认新实验输入
docs/                 实验流程和边界说明
scripts/              命令行入口及兼容脚本
src/                  可复用 Python 实现
tests/                自动化测试
```

大规模新数据、原始 `.ism`、准备好的受体和运行结果位于 `--data-root` 指定
的数据包。运行器自动记录配置快照、数量、引擎、seed、阶段状态和输出位置。

## E1 headroom 扫描（离线诊断）

`scripts/headroom_scan.py` 实现 `E:\Quant\docs\qubo\E1实现计划_headroom扫描_20260911.md`
的预注册诊断：8 条融合规则 φ × 精确子集枚举 × 三层 headroom（`H_raw` /
`H_nested` / `H_perm`）+ scaffold 聚类 bootstrap 噪声地板 + G1 判定。
不新增任何 docking，只重算已有稠密矩阵。

```bash
python scripts/headroom_scan.py run \
  --prereg configs/experiments/e1_headroom_preregistration.json \
  --assets configs/e1_assets.json \
  --output-dir results/headroom/e1_20260911 --jobs 24 --resume
```

资产表：本地 `configs/e1_assets.json`，远程服务器 `configs/e1_assets_remote.json`
（主判定）与 `configs/e1_assets_remote_sensitivity.json`（min 聚合 + 三 seed 独立）。
远程正式运行看 [远程运行手册](docs/headroom_scan_remote_runbook_zh.md)：
`JOBS=32 bash scripts/run_e1_headroom_remote.sh`。
实现与产物说明见 [docs/headroom_scan_zh.md](docs/headroom_scan_zh.md)。
**远程权威运行 `e1_20260912`（352 主分片 + 800 敏感性分片）判定 G1 = NO-GO**
（逐格 4.0% / 折内 oracle-φ 18.0% / train-selected φ 2.0%）；唯一通过三 seed 检验的灰区线索
是 `PPARA × min`（BEmin），按计划交棒 E4。产物在 `results/headroom/e1_20260912/`。

## 验证

```bash
cd "$REPO_ROOT"
python -m pytest -q --basetemp /tmp/qubo-receptor-ensemble-selection-pytest
python scripts/run_experiment.py --help
git diff --check
```

FA10 的 `k=2` 和 EGFR 的 `k=1` 是当前 development 案例，不是跨蛋白通用
规则。development/train 结果不能称为独立验证，也不能据此声称 QUBO 优势
或量子优势。

详见 [实验流程](docs/experiment_workflow_zh.md)、[配置说明](configs/README.md)
和 [脚本说明](scripts/README.md)。

## 许可

MIT License，见 [LICENSE](LICENSE)。
