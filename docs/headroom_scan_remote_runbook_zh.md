# E1 Headroom 远程运行手册（32 vCPU Linux）

> 适用环境（见 `E:\Quant\docs\qubo\qubo_receptor_ensemble_experiment_status_zh.md` §1）：
> `REPO_ROOT=/root/qubo-receptor-ensemble-selection`、`DATA_ROOT=/root/autodl-tmp/qubo_data_root`。
> 本文只讲**怎么跑**；协议与产物定义见 `docs/headroom_scan_zh.md`。
> 特点：**零新增 docking**，只读既有 run 的 `matrices/` 与 `score_tables/`。

## 0. 一页速览

```bash
cd /root/qubo-receptor-ensemble-selection
conda activate qubo-receptor-ensemble
git fetch origin && git checkout feat/e1-headroom-scan     # 或已合并的目标分支
pip install -e .                                            # 首次才需要

# 全流程（T1-T6 测试 → D1 → 主扫描 → 置换/φ选择/图 → seed 矩阵 → 敏感性 → 产物审计）
JOBS=32 bash scripts/run_e1_headroom_remote.sh             # JOBS=$(nproc) 亦可
```

> **`JOBS` 不要用 1**：pool30 的 40 个分片单核各需 ≈85 s，串行光这一项就 ≈1 小时；
> 32 并行整轮 10–20 分钟；JOBS=1 单进程约 3–3.5 小时（pool30 的 40 片占 ≈77 min）。
> 分片 checkpoint 是**逐个**落盘到 `<OUT>/cells/` 的，任何时刻中断、重跑同一条命令加 `--resume`
> 只重算未完成分片（不会丢整轮）；中途改 `JOBS` 重跑同样只是续跑。
>
> 日志位置：`nohup` 命令里的 `$LOG` 依赖启动 shell 的 `$DATA_ROOT`；没 export 时会落到
> `/results/headroom/<RUN_ID>.log`。找不到就用
> `ls -t /root/autodl-tmp/qubo_data_root/results/headroom/*.log /results/headroom/*.log 2>/dev/null`。
> 查看进度：`echo "shards: $(ls <OUT>/cells | wc -l)"` 与 `tail -5 <日志>`。

预计墙钟 **10–20 分钟**（32 并行分片；主运行 3–5 min、敏感性 2–3 min、组装/图 1–2 min），
零 docking、零 GPU。产物默认写到
`$DATA_ROOT/results/headroom/e1_<YYYYMMDD>/`（主）与 `..._sensitivity/`（敏感性）。

## 1. 环境准备（首次）

```bash
REPO_ROOT=/root/qubo-receptor-ensemble-selection
DATA_ROOT=/root/autodl-tmp/qubo_data_root

git clone https://github.com/Sinking-tenderness/qubo-receptor-ensemble-selection.git "$REPO_ROOT" 2>/dev/null || true
cd "$REPO_ROOT"
conda env create --name qubo-receptor-ensemble --file environment/environment.yml   # 已有则跳过
conda activate qubo-receptor-ensemble
python -m pip install --editable .

# 依赖自检：numpy / pandas / sklearn 必装；matplotlib 仅用于两张图
python -c "import numpy, pandas, sklearn; print('deps ok')"
python -c "import matplotlib; print('matplotlib', matplotlib.__version__)" || pip install matplotlib
```

输入文件随仓库分发：FA10/EGFR 的矩阵与配体 manifest、CDK2 的矩阵与 fold 分配都在
`data/processed/`（`git add -f` 跟踪），所以 `git clone` 即可跑全部 9 个资产；
MK14 的 canonical 矩阵仍以 `$DATA_ROOT/results/runs/mk14_adaptive_remote/` 为准。
测试的 MK14 路径支持环境变量覆盖：`E1_MK14_MATRIX`、`E1_MK14_MANIFEST`、`E1_MK14_RUN_DIR`
（默认同时探测 Windows 本地路径与 `/root/autodl-tmp/...` 远程路径，远程无需设置）。

**BLAS 线程必须收敛**（否则 32 个分片进程 × 每进程多线程 = 过度订阅，反而变慢）：

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

`scripts/run_e1_headroom_remote.sh` 已内置这四个变量，手工跑时请自行 export。

## 2. 路径与产物

| 变量 | 默认值 | 说明 |
|---|---|---|
| `REPO_ROOT` | `/root/qubo-receptor-ensemble-selection` | 代码仓库 |
| `DATA_ROOT` | `/root/autodl-tmp/qubo_data_root` | 数据根；canonical run 在 `results/runs/` |
| `RUN_ID` | `e1_$(date +%Y%m%d)` | 主运行 ID，决定产物目录名 |
| `OUT` | `$DATA_ROOT/results/headroom/$RUN_ID` | 主产物 |
| `SENSITIVITY_OUT` | `${OUT}_sensitivity` | 敏感性产物 |
| `JOBS` | `32` | 分片并行进程数 |

主产物（与 `docs/headroom_scan_zh.md` §5 相同）：
`input_manifest.json`、`cell_metrics.csv`、`headroom_map.csv`、`bootstrap_report.json`、
`phi_selection.json`、`permutations.json`、`gate_g1.json`、`run_manifest.json`、
`figures/*.png`、`cells/*.json`（分片 checkpoint，可断点续跑）。

## 3. 分步执行（与脚本一一对应）

### 3.1 T1–T6 与实现自检

```bash
python -m pytest -q tests/test_headroom_fusion.py tests/test_headroom_metrics_parity.py \
  tests/test_headroom_protocol_parity.py tests/test_headroom_subsets.py \
  tests/test_headroom_headroom.py tests/test_headroom_bootstrap.py \
  tests/test_headroom_gate.py tests/test_headroom_assets.py \
  tests/test_headroom_seed_matrices.py tests/test_headroom_runner.py
```

预期 **59 passed**（远程 canonical run 在位时，T2 与 min 矩阵校准都会真正执行；
若看到 `2 skipped`，说明 MK14 载体没探测到，用上面的 `E1_MK14_*` 变量指向它）。

### 3.2 D1 输入核验（硬门，先跑）

```bash
python scripts/headroom_scan.py verify \
  --prereg configs/experiments/e1_headroom_preregistration.json \
  --assets  configs/e1_assets_remote.json \
  --output-dir "$OUT"
```

预期：9 个资产 `status=ok`，五个主判定靶点 `source=matrix_csv`（**不是** problem.json 回退），
`PPARA_pool30` 的 `backfilled_scaffold=600`，`primary_targets_missing=[]`。
这一步同时写入 `input_manifest.json`（含全部输入 SHA-256）。

### 3.3 主运行（精确枚举，9 资产 = 352 分片）

```bash
python scripts/headroom_scan.py run \
  --prereg configs/experiments/e1_headroom_preregistration.json \
  --assets  configs/e1_assets_remote.json \
  --output-dir "$OUT" \
  --jobs 32 --resume --skip-perm --skip-phi-selection --skip-figures
```

- `(target, fold, phi)` 为一个分片，写 `cells/<target>_<fold>_<phi>.json`；
- 中断后同命令 `--resume` 续跑（已完成分片跳过）；
- 9 资产 = 5 主 + FA10 + EGFR + pool30 + CDK2：8×40 + 32 = **352 分片**；
- pool30 按预注册跑满 **k=1..6**（本地验证机受算力限制只跑了 k≤4；32 vCPU 不必缩水）。

### 3.4 组装：置换校正 + train-only φ 选择 + 产物 + 图

```bash
python scripts/headroom_scan.py report \
  --prereg configs/experiments/e1_headroom_preregistration.json \
  --assets  configs/e1_assets_remote.json \
  --output-dir "$OUT" --jobs 32
```

`report` 只补算缺失的置换/φ 记录（已存在的不重算），然后重建全部产物与两张图；
可以反复执行，幂等。

### 3.5 敏感性：从既有 score_tables 重建 3 个 seed 矩阵

```bash
for T in mk14 ppara pparg bace1 esr1; do
  python scripts/headroom_scan.py extract-seeds \
    --run-dir "$DATA_ROOT/results/runs/${T}_adaptive_remote" \
    --output-dir "$OUT/seed_matrices/$T" \
    --seeds 20260821,20260822,20260823
done
```

每个目录产出 `seed_<seed>_matrix.csv`、`seed_min_matrix.csv`（三 seed 逐格取 min）和
`seed_extraction_audit.json`（文件清单 + SHA-256 + 丢弃行数）。
提取器已用 MK14 真数据校准：重建的 `seed_min_matrix.csv` 与 canonical
`sensitivity_minimum_matrix.csv` **逐值相等（max |Δ| = 0）**。

### 3.6 敏感性运行（20 资产 = 800 分片）

```bash
python scripts/headroom_scan.py run \
  --prereg configs/experiments/e1_headroom_preregistration.json \
  --assets  configs/e1_assets_remote_sensitivity.json \
  --root "run_root=$OUT" \
  --output-dir "$SENSITIVITY_OUT" \
  --jobs 32 --resume --skip-perm --skip-phi-selection --skip-figures

python scripts/headroom_scan.py report \
  --prereg configs/experiments/e1_headroom_preregistration.json \
  --assets  configs/e1_assets_remote_sensitivity.json \
  --root "run_root=$OUT" \
  --output-dir "$SENSITIVITY_OUT" --jobs 32 --skip-figures
```

敏感性资产 = 5 个 canonical `sensitivity_minimum_matrix.csv` + 15 个单 seed 矩阵
（`{target}_min` / `{target}_seed<seed>`）。这些 id 不在预注册主判定靶点里，
所以敏感性运行的 `gate_g1.json` 是 `NOT_EVALUATED`——**这是预期的**，判据只看主运行。

### 3.7 质量门

脚本最后一步自动审计，等价的手工核对：

| 检查 | 期望 |
|---|---|
| `input_manifest.json` | `primary_targets_missing=[]`，全部 `status=ok` |
| `cells/` | 主运行 352 个分片；敏感性 800 个 |
| `cell_metrics.csv` | ≥ 4000 行，`method` 含 `oracle/ref/greedy/single/train_selected` |
| `gate_g1.json` | `n_cells_considered_per_phi_cell = 400`，decision ∈ {NO_GO, GO_PHI, GREY_ZONE} |
| `bootstrap_report.json` | `per_cell` ≥ 300，每格有 `noise_floor.se` 与 `mde` |
| `permutations.json` | `per_cell` ≥ 300（主格 400 全有） |
| `phi_selection.json` | `targets` 覆盖 7 个主+扩展靶点 |
| `figures/` | 两张 PNG 非空 |

## 4. 时间与资源预算（32 vCPU）

单分片核心秒（本地单核实测，MK14/R=15，含 2000 次 bootstrap）：

| 分片类型 | 子集数/分片 | 单分片 | 数量 | 核心秒 | 32 并行墙钟 |
|---|---:|---:|---:|---:|---:|
| R=15 主靶点 k≤6 | 9,948 | ≈4.2 s | 240 | ≈1,000 | ≈0.5 min |
| R=16（PPARG）k≤6 | 14,892 | ≈5.5 s | 40 | ≈220 | — |
| FA10/EGFR k≤6 | 4,095 / 2,509 | ≈2 s | 80 | ≈160 | — |
| **pool30 k≤6** | 768,211 | ≈85 s | 40 | ≈3,400 | **≈1.8 min** |
| CDK2（4 折） | 14,892 | ≈4 s | 32 | ≈130 | — |
| 敏感性 20 资产 | 同上 | ≈4 s | 800 | ≈3,200 | ≈1.7 min |
| 置换校正 | — | ≈0.4 s/格 | ≈1,600 | ≈640 | ≈0.3 min |
| φ 选择（7 靶点×5 折） | 24 次枚举/折 | ≈7 s/折 | 35 | ≈250 | ≈0.1 min |

总量约 9,000 核·秒 ⇒ **32 并行下 5–10 分钟**；加上 Python 启动/IO/图，按 10–20 分钟安排。

> 注意：本机（受限 Windows 沙箱）进程池被拦截时 runner 会自动回退串行并打印
> `[parallel] process pool unavailable ...`；Linux 上应看到 `--jobs 32` 真正并行。
> 若日志出现回退，检查是否被容器/沙箱限制（`python -c "import multiprocessing as m; print(m.get_start_method())"`）。

## 5. 结果怎么读

1. **G1**：`gate_g1.json`
   - `fraction_cells_go_train_selected_phi`：可落地层面（headline）；
   - `fraction_cells_go_fold_oracle_phi`：φ 上界；
   - `fraction_cells_go_per_phi_cell`：字面 400 格；
   - `<10%` → NO_GO；`≥30%` → GO_PHI；之间 → GREY_ZONE + `tie_break_trace`。
2. **地图**：`headroom_map.csv` 的 `verdict/go_folds/ratio`，按 `target × phi × k` 看哪些格子过线。
3. **噪声地板/MDE**：`bootstrap_report.json` 的 `noise_floor.se` 与 `mde`；
   任何小于 MDE 的差值不得解释为方法差异。
4. **φ 与 k\***：`phi_selection.json` 的 `selected_phi_by_k` 与 `k_star`。
5. **Sanity 对照**（本地验证运行 2026-09-11 的参考值，主格 400）：
   `H_raw` 均值 ≈ +0.062（92% 为正）、`H_nested` 均值 ≈ −0.017、噪声地板 ≈ 0.05、
   train-selected φ 通过率 ≈ 2%（NO_GO）。远程若量级差一个数量级，先查矩阵列序与折。

## 6. 回传与归档

```bash
# 远程 → 本地（示例）
rsync -av "$OUT" "$SENSITIVITY_OUT" <local_user>@<local_host>:/E/Quant/remote_runs/
```

归档到仓库时只提交产物（分片 checkpoint 已在 `.gitignore` 中排除）：

```bash
mkdir -p results/headroom/$RUN_ID
cp -r "$OUT"/{input_manifest.json,cell_metrics.csv,headroom_map.csv,bootstrap_report.json,phi_selection.json,permutations.json,gate_g1.json,run_manifest.json,figures} results/headroom/$RUN_ID/
git add results/headroom/$RUN_ID && git commit -m "results(headroom): E1 远程正式运行 $RUN_ID"
```

## 7. 排错

| 症状 | 处理 |
|---|---|
| 某靶点 `matrices/` 不存在 | 自动回退 `problem.json` 载体（`input_manifest.source=problem_json`，D1 已复核）；或先重跑该 run 的 `aggregate` 阶段 |
| FA10/EGFR 报 missing | 输入已随仓库分发（`data/processed/stage102a_*`）；若仍缺，可从其 run 的 score_tables 重建中位聚合矩阵：`python scripts/headroom_scan.py extract-seeds --run-dir $DATA_ROOT/results/runs/stage102a_fa10_full_local --output-dir /tmp/fa10_rebuild --aggregation median`，再把 `seed_median_matrix.csv` 拷成 `data/processed/stage102a_fa10_phase_a_primary_median_score_matrix.csv` |
| CDK2 报 missing | 控制面板（不进 G1）：矩阵与 fold 分配已在 `data/processed/stage04_cdk2_expanded16_development_*`；确实要跳过就从资产表删掉该条目并标注 |
| 想先只跑主判定 | `--targets MK14,PPARG,BACE1,ESR1,PPARA,PPARA_pool30`（6 资产 = 240 分片）；敏感性资产只依赖这 5 个靶点，可独立跑满 |
| `gate = NOT_EVALUATED` | 主格不完整：查 `input_manifest.primary_targets_missing` 与 `cells/` 是否 352 |
| 想改 `RUN_ID` | `RUN_ID=e1_x bash scripts/run_e1_headroom_remote.sh`；若手工跑敏感性，`--root run_root=<主目录>` 必须指对 |
| 中断/超时 | 同一命令加 `--resume`；分片粒度 = (target, fold, phi)，重跑代价极小 |
| 内存 | 单分片峰值 < 300 MB（只保存 top-M 掩码，不存全部子集分数）；pool30 k=6 也安全 |
| 想快速冒烟 | `--k-max 3 --no-train-oracle --bootstrap-iterations 100 --permutations 20`（**必须标注为偏离预注册**，不能用于 G1） |
| 图报错 | `pip install matplotlib` 或 `--skip-figures`（图不影响 G1） |
| 数据被改过 | `run_manifest.json` 记录 prereg sha256 与全部输入 SHA-256；不匹配会体现在 D1 复核里 |

## 8. 预注册纪律（跑之前再看一眼）

- `configs/experiments/e1_headroom_preregistration.json` 是冻结配置，其 SHA-256 会写进
  `input_manifest.json`/`run_manifest.json`；**跑完不得修改门限、指标、融合族或判据**。
- 任何偏离（例如只跑 k≤4、跳过敏感性）都要写成 amendment，并在结论里标注；
  本地那次 `e1_local_validation_20260911` 就是这样标注的验证件，**不作为权威结论**。
- 预注册的 `H_perm` 公式量纲问题已在 `docs/headroom_scan_zh.md` §6.4 记录；
  G1 只用 `H_nested`，不受影响。