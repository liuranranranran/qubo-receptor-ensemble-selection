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

## E2 预算分配律（离线诊断）

`scripts/budget_scan.py` 实现 `E:\Quant\docs\qubo\下一步实验计划_20260911.md` §3 E2 的预注册电池：
固定 B 次 docking（B ∈ {600, 1200, 2400, 3600, 4800}），在既有稠密矩阵上遮蔽模拟 8 条分配策略
（宽度极限 / 均匀深度 / top-x% 两阶段 / scaffold 两阶段 / 逐配体分数 oracle），按冻结 φ
（mean、min）做 V5 五折外评、scaffold 聚类 bootstrap 与 MDE，输出收益–预算曲线、B*、G2 判定
和训练折特征表（探索性分配律）。零新增 docking。

```bash
python scripts/budget_scan.py run \
  --prereg configs/experiments/e2_budget_preregistration.json \
  --assets configs/e1_assets.json \
  --output-dir results/budget/e2_20260913 --jobs 16 --resume
```

远程一键：`JOBS=32 bash scripts/run_e2_budget_remote.sh`（canonical 矩阵用
`configs/e1_assets_remote.json`）。实现与产物说明见 [docs/budget_scan_zh.md](docs/budget_scan_zh.md)。

**本地全量运行 `e2_20260913`（10 资产 / 98 分片 / 3920 格）判定 G2 = `NONTRIVIAL_REGION`**：
在同一 600 次 docking 预算下，"scaffold 代表 + 前 25% 组补测"（`s4_scaffold25`）比
"全库 × 单受体"宽度基线宏平均 PR-AUC 高 +0.086（mean）/ +0.075（min），bootstrap CI>0 覆盖
4/5 与 5/5 主判定靶点（min 下 25/25 折为正）；深度预算（1200–4800）不再增加宏平均收益，
宽度基线只在 ESR1(mean) 的深度预算上胜出（3/30 格）。边界：多格低于 MDE、30 受体池不显著、
开发对照 CDK2(mean) 为负 → 只作假设生成。结论见
`E:\Quant\docs\qubo\E2结论总结_20260913.md`。

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
