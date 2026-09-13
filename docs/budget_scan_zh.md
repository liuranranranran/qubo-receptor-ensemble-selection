# E2 预算分配律：实现与运行

> 计划依据：`E:\Quant\docs\qubo\下一步实验计划_20260911.md` §3 E2（零 docking，2 周）
> 预注册：`configs/experiments/e2_budget_preregistration.json`
> CLI：`scripts/budget_scan.py`（`run` / `report`）；一键：`scripts/run_e2_budget_remote.sh`
> 关系：E1 证明"选构象子集"层无可赢空间 → E2 转向**固定预算下的分配几何**（配体宽度 vs 构象深度）。

## 0. 口径（冻结）

- **预算 B**：docking 任务数 = (配体, 受体) 单元数；B ∈ {600, 1200, 2400, 3600, 4800}，
  600 ≈ 整库规模（600 配体 × 1 受体）。
- **库 = 整块 600 配体面板**（15–16 个受体，共 9000 单元）；**评估 = 留出折的 120 配体**
  （V5 五折外评）：折外参数只能来自训练折与已揭示分数。
- **排序约定**：已测配体用冻结 φ 的**异质深度融合**（每个配体融合自己的 k 个观测受体）；
  **未测配体排在最后**（"测不到就找不到"，筛选口径）。
- 主指标 **PR-AUC**；次要 BEDROC20、recall@1%/5%/10%、EF1/5/10；配对统计用 scaffold 聚类
  bootstrap（B=2000）与 MDE = t₉₇.₅·SE·√2。

## 1. 策略族（8 条，冻结）

| 策略 | 定义 | 类别 |
|---|---|---|
| `s1_width` | 训练最优受体 × min(B, N) 配体（宽度极限；B>N 的余量不花） | 宽度基线 |
| `s2_uniform` | ⌊B/N⌋ 个受体（训练序）× 全部配体（余量不花） | 均匀深度 |
| `s3_top10/25/50` | 阶段 1：最优受体 × 全部；阶段 2：按阶段 1 分数取前 x%，逐轮补受体 | 两阶段 |
| `s4_scaffold25/50` | 阶段 1：每个 scaffold 一个代表 × 最优受体；阶段 2：按代表分数取前 x% 组，组内全成员补受体 | scaffold 两阶段 |
| `s5_score_oracle` | 每个配体用隐藏矩阵取"对自己最好"的 k 个受体（逐配体分数上界，不用标签） | 上界 |

可选（不在默认族内）：`s5_metric_oracle` = 逐单元贪心地最大化**留出折**指标（隐藏矩阵 + 折标签）。
在"评估限定在留出折"的口径下它退化为"把留出折所有单元都测掉"，且约 0.37 s/步；
需要指标上界时用 `--policies s5_metric_oracle` 显式单独跑。

**诚实性纪律**：除 `s5_*` 外，策略只用（a）训练折标签决定受体序、（b）已揭示的 docking 分数、
（c）配体 scaffold；`s5_*` 只用隐藏矩阵/标签并**标注为上界**。

## 2. 判定（G2，跑前冻结）

- **B\***：某 (靶点, φ) 上"最好策略不再等于 `s1_width`"的最小预算（并列时基线胜，保守）。
- **G2 决策**：
  - `WIDTH_LOCKED`：所有深度预算（2400/3600/4800）下，全部主判定靶点（× 冻结 φ）的最优策略
    都是 `s1_width` → 集成 docking 的实践上限被钉死（强结论，可写）。
  - `NONTRIVIAL_REGION`：存在某非基线策略，在**同一预算、同一冻结 φ**下 ≥4/5 主判定靶点上
    `mean_delta > 0 且 ci95_low > 0` → 得到可处方的分配规则（本计划最高价值产出）。
  - 其他 → `GREY_ZONE`（报 B\* 与逐靶点曲线）。

**判定口径（2026-09-13 跑前冻结修订）**：稳定性只统计主判定靶点（`targets`），且按
`(policy, budget, phi)` 组合计数，不允许跨 φ 拼 family-max；配对差值一律按 `outer_fold` id
对齐（不依赖行序）。宽松口径（跨 φ 计数）写入 `gate_g2.json` 的
`stable_policy_budget_any_phi` 作为审计字段，不参与判定。

## 3. 代码布局

```text
src/qubo_receptor_ensemble/budget/
    fusion_ragged.py  异质深度融合（与 headroom.fusion 在均匀掩码下逐位一致）
    policies.py       预算模拟器 + 8 条策略（预算不变式：Σ≤B、单元唯一、受体≤R）
    metrics.py        筛选口径指标（未测=最后）+ recall@q
    law.py            曲线 / B* / 配对 bootstrap / G2
    features.py       训练折特征（单受体质量、pair-gain 互补性、池多样性、库/活性率）
    runner.py         分片（target×fold×φ）checkpoint、产物、图
scripts/budget_scan.py              CLI：run / report
scripts/run_e2_budget_remote.sh     远程一键（测试 → 运行 → 组装 → 归档 → 可选关机）
configs/experiments/e2_budget_preregistration.json
tests/test_budget_core.py           10 项（融合 parity、预算不变式、oracle 上界性、B*/G2、端到端 resume）
```

## 4. 产物（`results/budget/<run_id>/`）

| 文件 | 内容 |
|---|---|
| `budget_cells.csv` | 逐格：target/fold/phi/policy/budget + 预算用量、覆盖、深度、全部指标 + 对基线差值 |
| `budget_curves.csv` | target×phi×policy×budget 的均值/最坏折/正收益折数/覆盖/深度 |
| `policy_comparisons.csv` | 策略 vs `s1_width` 的配对差值、bootstrap CI、MDE |
| `b_star.json` | 每 (target, φ) 的逐预算最优策略与 B\* |
| `law_features.csv` | 每 (target, φ)：训练折特征（单受体质量 / pair-gain / 池多样性 / 库规模 / 活性率）+ B\* |
| `law_summary.json` | 分配律的探索性汇总（特征 vs B\* 的秩相关、width-locked/切换组均值） |
| `gate_g2.json` | G2 判定 + 证据链 |
| `input_manifest.json` / `run_manifest.json` | D1 输入、commit、环境、配置哈希 |
| `figures/fig_budget_curves.png` `figures/fig_b_star.png` `figures/fig_law.png` | 收益–预算曲线、B\* 图、B\* vs 特征图 |
| `cells/<target>_<fold>_<phi>.json` | 分片 checkpoint（含各策略融合分数，供 bootstrap；`--resume` 复用） |

## 5. 运行（远程 32 vCPU）

```bash
cd /root/qubo-receptor-ensemble-selection && git pull --ff-only
export DATA_ROOT=/root/autodl-tmp/qubo_data_root
RUN_ID=e2_$(date +%Y%m%d)

JOBS=32 bash scripts/run_e2_budget_remote.sh          # 一键（约 5–10 分钟）
```

分步：

```bash
python scripts/budget_scan.py run \
  --prereg configs/experiments/e2_budget_preregistration.json \
  --assets  configs/e1_assets_remote.json \
  --output-dir "$DATA_ROOT/results/budget/$RUN_ID" \
  --jobs 32 --resume
python scripts/budget_scan.py report \
  --prereg configs/experiments/e2_budget_preregistration.json \
  --assets  configs/e1_assets_remote.json \
  --output-dir "$DATA_ROOT/results/budget/$RUN_ID"
```

- 分片 = (target, fold, φ)，共 9 资产 × 5 折 × 2 φ ≈ 90 片；
- 预算/融合/策略/折都可用 `--budgets/--fusions/--policies/--folds/--targets` 覆盖（覆盖即偏离预注册，需在结论里标注）；
- `report` 只聚合 checkpoint（含 630 个配对的 2000 次聚类 bootstrap，约 3–5 分钟）。

## 6. 预期时间与限制

| 项 | 估计 |
|---|---|
| 分片模拟（8 策略 × 5 预算，含 oracle 上界） | 每片 ≈1–2 s；32 并行 ≈ 1 分钟 |
| 组装（曲线 + 630 个配对 bootstrap + 两张图） | ≈3–5 分钟（单进程） |

限制：① 矩阵遮蔽模拟，默认"已测即揭示"，未建模 docking 失败/重试成本；
② 未测配体排最后使低覆盖策略天然吃亏（这是筛选口径，不是 bug）；
③ `s5_score_oracle` 是**逐配体分数上界**而非指标上界；
④ 探索性重分析（父计划 §5.1），结论只作假设生成与处方候选。

## 7. 与 E1 的关系

E1 的结论（决策层无可赢空间）决定了 E2 的问法：**不再找"更聪明的选择器"，
而是问"固定预算下宽度与深度怎么分配"**。E2 直接复用 E1 的矩阵加载/D1 核验/聚类 bootstrap 基建，
并把 E1 的 φ 线索（`min`）作为第二个融合族成员。

## 5.1 运行状态（2026-09-13）

| 运行 | 环境 | 资产表 | 结果 | 产物 |
|---|---|---|---|---|
| `e2_20260913`（全量，权威件） | 本地 Windows 28 vCPU | `configs/e1_assets.json`（9+1 资产；MK14 canonical，PPARG/BACE1/ESR1/PPARA/pool30 用 problem.json 载体） | 98 分片 / 3920 格 / 700 配对；**G2 = `NONTRIVIAL_REGION`**；稳定格 `s4_scaffold25@600`（mean 4/5、min 5/5） | `results/budget/e2_20260913/` |
| 远程 canonical 复跑 | 32 vCPU Linux | `configs/e1_assets_remote.json` | 未执行（实例下线）；脚本就绪 | `JOBS=32 bash scripts/run_e2_budget_remote.sh` |

结论总结：`E:\Quant\docs\qubo\E2结论总结_20260913.md`。

## 8. 与计划 §4 统计规范的对应

| 规范 | 落点 |
|---|---|
| 主指标 PR-AUC，BEDROC20 次要，EF1% 不作主指标 | `primary_metric=pr_auc`；`budget_cells.csv` 同时写 BEDROC20、recall@1/5/10%、EF1/5/10% |
| 统计单位 = scaffold 聚类 | `law.macro_bootstrap_delta` 对每折按 scaffold 整簇重采样，再算宏平均 |
| 每靶点 + 最坏折，不只报宏平均 | `budget_curves.csv` 写 `delta_worst_fold`、`positive_folds`；结论按靶点分列 |
| 事先报 MDE | `policy_comparisons.csv` 每行带 `mde` 与配对 SE；MDE 以下差值不得解释为方法差异 |
| φ 家族冻结、禁止 family-max | φ 只含 `mean`/`min`，逐格判定；G2 稳定性按同 φ 计数 |
| 不确定性与预注册 | 预注册 JSON 冻结判据；所有产物带 `preregistration.sha256` 与输入 SHA-256 |
