# E1 Headroom 扫描：实现与结果

> 运行 ID：`e1_20260911`　|　产物：`results/headroom/e1_20260911/`
> 计划依据：`E:\Quant\docs\qubo\E1实现计划_headroom扫描_20260911.md`（父计划 §3 E1）
> 状态：已实现、已全量运行、G1 = **NO-GO**

## 0. 一句话结论

在预注册主格（5 靶点 × 5 折 × 8 融合规则 × k∈{2,3} = 400 格）上：
偷看测试标签的 oracle 上界 `H_raw` 平均 **+0.062**（92% 为正、52% 超过噪声地板），
但"一半选、另一半评"的可落地 `H_nested` 平均 **−0.017**（仅 4.0% 格子通过
`H_nested_lower95 > 噪声地板 且 ratio > 1`）。折内 oracle-φ 通过率 18%、
train-selected φ 通过率 2%，均低于 30% 的 GO 门；k\* 也**没有**出现 5/5 靶点
一致的右移。按预注册判据：**"选构象子集"线以"上界告负"收口**，主线转 E2
（预算分配）与 E5 臂 A（holo-like 池）。

唯一值得单列的信号：`min`（BEmin）在 PPARA / PPARG 上达到可落地的
`H_nested > 噪声`（PPARA min：ratio 3.8–6.5，3/5 折通过），属于灰区线索，
按计划交棒 E4 作为"φ 选择"的新决策问题，而不是复活主线。

## 1. 实验协议（冻结）

| 项 | 冻结值 |
|---|---|
| 靶点（主判定） | MK14 / PPARG / BACE1 / ESR1 / PPARA |
| 主格 | `k ∈ {2,3}` × 8 φ × 5 靶点 × 5 折 = 400 格 |
| k 扫描范围 | 1–6（pool30 为 1–4，见 §7.3） |
| 融合族 φ | `mean,min,max,zmean,gmean,hmean,ranksum,rrf`（rrf K=60） |
| 主指标 | PR-AUC（average precision）；次指标 BEDROC20 / ROC-AUC / EF1,5,10 |
| 评价单元 | V5 外层留出折（每折约 120 配体，约 24 active），与 `scripts/nested_outer_k_evaluation.py` 同口径 |
| `S_ref` | 训练折上 `{greedy, single}` 中较优者（主指标） |
| `H_raw` | `max_{|S|=k} U_test(S) − U_test(S_ref)`（精确枚举全部子集） |
| `H_nested` | 测试折按 scaffold 分 A/B：A 上选 `argmax_S U_A`、B 上评 `U_B`，反向再平均 |
| `H_perm` | `H_raw − q95(max_{S∈TopM} U_test(S^perm))`，TopM=2000、200 次标签置换 |
| 噪声地板 | 同格 `greedy vs single` 的 scaffold 聚类配对 bootstrap（B=2000）的 **SE** |
| 通过判据 | `H_nested_lower95 > noise_floor` 且 `ratio > 1`（ratio = H_nested / SE） |
| MDE | `t(0.975, df) · SE · √2` |
| G1 门 | 占比 <10% → NO-GO；≥30% → GO-φ；10–30% → 灰区 + 预注册 tie-break |

全部子集**精确枚举**，无近似搜索；φ 的平移常数、z 统计量与秩参照系都只由训练折拟合。

## 2. 代码结构

```text
src/qubo_receptor_ensemble/headroom/
    fusion.py       8 条 φ + train-only 参数拟合 + 物化 scorer（子集评分热路径）
    metrics_fast.py O(n) 版 BEDROC / PR-AUC / ROC-AUC / EF（与 screening.py 数值一致）
    subsets.py      位掩码枚举、utility、精确 oracle
    headroom.py     H_raw / H_nested / H_perm、A/B scaffold 划分、内层折分配
    bootstrap.py    scaffold 聚类 bootstrap、噪声地板、MDE
    gate.py         预注册 G1 判定
    assets.py       D1 输入核验、matrix/problem.json 双载体、SHA-256
    runner.py       分片扫描、checkpoint、并行、产物、图
scripts/headroom_scan.py        CLI：verify / run / report（rebuild+补齐）
configs/experiments/e1_headroom_preregistration.json   冻结配置
configs/e1_assets.json         输入资产表（{repo}/{quant}/{data_root} 占位符）
tests/test_headroom_*.py       T1–T6 与实现单测（51 项）
```

CLI：

```bash
python scripts/headroom_scan.py verify --prereg ... --assets ... --output-dir results/headroom/<run_id>
python scripts/headroom_scan.py run    --prereg ... --assets ... --output-dir ... --jobs 24 --resume
python scripts/headroom_scan.py report --prereg ... --assets ... --output-dir ...   # 从 checkpoint 重建/补齐
```

- 分片键 `(target, fold, phi)`，每片写 `cells/<target>_<fold>_<phi>.json`，`--resume` 跳过已完成分片；
- `report` 只补算缺失的置换校正与 φ 选择，再重建全部产物，可安全重跑；
- 沙箱/受限 Windows 上进程池不可用时自动回退串行（协议与结果不变，只损失墙钟时间）。

## 3. D1 输入核验（`input_manifest.json`）

| 资产 | role | 载体 | 配体×受体 | 折 | scaffold 数 | 备注 |
|---|---|---|---|---|---|---|
| MK14 | primary | matrix + prepared manifest | 600×15 | 5 | 414 | 本地矩阵齐全 |
| PPARG | primary | **problem.json 载体** | 600×16 | 5 | 338 | `matrices/` 未下载；V5 报告 §7 已确立该载体，D1 复核通过 |
| BACE1 | primary | problem.json 载体 | 600×15 | 5 | 358 | 同上 |
| ESR1 | primary | problem.json 载体 | 600×15 | 5 | 379 | 同上 |
| PPARA | primary | problem.json 载体 | 600×15 | 5 | 246 | 同上 |
| FA10 | secondary | matrix + ligand manifest | 600×13 | 5 | 330 | stage102a phase-A |
| EGFR | secondary | matrix + ligand manifest | 600×12 | 5 | 437 | stage102a phase-A |
| PPARA_pool30 | pool | problem.json + 回填 | 600×**30** | 5 | 246 | scaffold/outer_fold 由 PPARA 载体回填 600/600（已计数） |
| CDK2 | control | matrix + fold_assignments | 160×16 | 4 | 160 | 开发对照，不进 G1 |
| MK14_min | sensitivity | sensitivity_minimum_matrix | 600×15 | 5 | 414 | 仅 MK14 有下载的 min 聚合矩阵 |

核验项：受体列顺序冻结并写盘、无缺失/非有限分数、`label ∈ {active, decoy}`、
每配体唯一 `outer_fold`、折覆盖、`scaffold_smiles` 100% 非空、每个输入文件 SHA-256。
所有靶点 `status=ok`，主判定靶点无缺失。

## 4. 正确性验证（T1–T6）

| 测试 | 内容 | 结果 |
|---|---|---|
| **T1** 指标 parity | `metrics_fast` vs `screening.py`：200 个随机向量 + 224 个含并列值的向量 | max abs diff **1.7e-16**（门限 1e-12） |
| **T2** 协议 parity | MK14 + mean + k=1..6：现跑 `nested_outer_k_evaluation.py` 的 `solve_subset`/`subset_metrics` 与快路径对比；子集与归档 V5 decision log 对比 | 子集 **30/30 完全一致**；max abs diff **2e-16**（门限 1e-9）；归档 CSV（6 位小数）另设 1e-5 护栏 |
| **T3** 单受体退化 | `\|S\|=1` 时 8 条 φ 的秩相关 | 全部 **1.0** |
| **T4** 泄漏测试 | 只改测试折分数 → z 统计 / 平移常数 / 秩参照逐位不变 | 通过（bitwise equal） |
| **T5** 方向单调性 | 整列平移：`ranksum`/`rrf` 数值不变；`mean` 随之移动 | 通过 |
| **T6** 可复现 | 同 seed/输入连跑两次，产物 SHA-256 逐位相同 | 通过（含 `cell_metrics.csv` / `gate_g1.json`） |

> T2 的一个归档差异已记录：`E:\Quant\nested_v5\mk14\folds_long.csv` 的
> fixed-k BEDROC20 与"现跑 golden 脚本"最大差 1.74e-6（该 CSV 只有 6 位小数，
> 且是 2026-08-27 的旧运行）。子集完全一致，且现跑 golden 路径与快路径一致到
> 2e-16，因此 T2 以"现跑 golden 路径"为准，归档值仅作 1e-5 护栏。

## 5. 运行与产物

```text
results/headroom/e1_20260911/
  input_manifest.json      D1 核验 + SHA-256
  cell_metrics.csv         长表：target/fold/phi/k/method（oracle/ref/greedy/single/train_selected）
  headroom_map.csv         target × phi × k 的 h_nested / lower95 / noise / ratio / verdict
  bootstrap_report.json    每格 noise_floor、MDE、H_nested 条件 bootstrap
  phi_selection.json       内层 CV（训练折）φ 选择 + k* 位移
  gate_g1.json             G1 判定 + 证据链
  permutations.json        H_perm 逐格置换零分布
  run_manifest.json        commit/dirty 状态、环境、seed、输入哈希、配置哈希
  figures/fig_headroom_map.png, fig_phi_ranking.png
  cells/*.json             392 个分片 checkpoint（可 --resume）
```

规模：10 个资产 × (5 折或 4 折) × 8 φ = **392 分片**；主格 400 格；全量 cell 行 12k+。

## 6. 结果

### 6.1 主判定 G1（`gate_g1.json`）

| 级别 | 通过率 | 计数 |
|---|---|---|
| 逐格（400 格，每格自己的 φ） | **4.0%** | 16 / 400 |
| 折内 oracle φ（50 个 target×fold×k 单元，取该折最好的 φ） | **18.0%** | 9 / 50 |
| train-selected φ（50 单元，φ 只用训练折内层 CV 选） | **2.0%** | 1 / 50 |

`decision = NO_GO`（headline = train-selected 2% < 10%）。
tie-break 记录：oracle−train 差 16 个百分点；k\* 位移 `[0, −1, −1, +1, −1]`，
**0/5** 靶点出现 ≥2 的右移 → 灰区第二条也不成立。

### 6.2 有没有可赢的东西？

| 统计量（主格 400 格） | 均值 | 中位 | >0 比例 | >噪声地板比例 |
|---|---:|---:|---:|---:|
| `H_raw`（偷看测试标签的 oracle 上界） | **+0.062** | +0.049 | 92.2% | 52.2% |
| `H_nested`（A 选 B 评，可落地） | **−0.017** | −0.025 | 34.8% | 4.0% |
| 噪声地板（greedy−single 配对 SE） | 0.050 | 0.045 | — | — |

解读：**"有东西可赢"只存在于后视镜里**。oracle 对 105 个子集取最优，平均能多
拿 0.062 PR-AUC（≈1.2× 噪声地板）；但把这半折选出来的子集放到另半折，收益
完全消失（甚至为负）。这正是 E1 要区分的"方法不行"与"没东西可赢"：
**在该工况下，"用一半数据选出更好的 k 子集"这件事本身没有可转移的信息量。**

### 6.3 φ 是不是主因？

主格（k∈{2,3}）各 φ 的平均 `H_nested`：

| φ | min | rrf | hmean | zmean | gmean | mean | ranksum | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mean H_nested | **+0.028** | −0.000 | −0.008 | −0.019 | −0.024 | −0.026 | −0.040 | −0.044 |

- `mean` 在 50 个单元里的**平均排名 4.7/8**（中位 5，22% 落在倒数两名）——
  "mean 垫底"的预期**不成立**；`min`（BEmin）是最好的平均规则，`max`/`ranksum` 最差。
- k\* 位移：以 oracle-k\* 计，best-φ 相对 mean 的位移为 `0/−1/−1/+1/−1`，
  没有"φ 让 k\* 系统性右移"的证据（k 饱和点的旧结论不因 φ 而改观）。
- 通过判据的 16 个格子高度集中：PPARA 8 个、PPARG 8 个；其中 11 个是 `min`，
  且集中在 fold 1/2/5。换言之：**φ 的效果是靶点特异、折特异的小样本信号**，
  不是全局规则。

### 6.4 H_perm 的（不）可用性

按预注册公式 `H_perm = H_raw − q95(max_{S∈TopM} U_test(S^perm))` 得到的值
**全部为负**（主格均值 −0.329）。原因是该式把两个不同量纲项相减：`q95` 是
"置换标签下 2000 个子集的最大 AP"，其绝对水平由指标本身的量纲决定（≈0.3–0.4），
而 `H_raw` 是差值（≈0.06）。因此 `H_perm` 不是可用的膨胀校正量；产物中同时
保留了可解释的替代量 `h_perm_alt = U_test(oracle) − q95(置换最大值)`（主格均值
+0.274，反映"真实标签下最优子集高于随机标签下的最优"），但它衡量的是
"评分是否有信号"，不是"选择是否膨胀"。**G1 判定不使用 H_perm**（预注册门只
用 `H_nested`），该公式问题作为方法学备注写入复现记录。

### 6.5 敏感性与扩展面板

| 资产 | role | H_raw (k=2,3) | H_nested | 通过格（k=2,3） |
|---|---|---:|---:|---:|
| FA10 | secondary | +0.063 | **+0.021** | 0 / 80 |
| EGFR | secondary | +0.046 | **+0.018** | 0 / 80（但 map 上有 10 行 GO，含 k=4,5） |
| PPARA_pool30 | pool | +0.074 | −0.043 | 2 / 80（map 上 `min` k=2,3,4 各 1 折通过） |
| CDK2 | control | +0.083 | −0.017 | 1 / 64 |
| MK14_min | sensitivity | +0.088 | +0.000 | 0 / 80 |

- **seed 聚合敏感性**：MK14（median 矩阵）与 MK14_min（min 聚合）在 80 个
  k∈{2,3} 单元上的通过模式**完全一致**（80/80 都不过），`H_raw` 相关 0.84、
  `H_nested` 相关 0.75 → 结论对该敏感性是稳的。
- FA10/EGFR 的 `H_nested` 均值为正（+0.02），但没有任何格子在 lower95 意义上
  通过；EGFR 的 map GO 行来自 k=4/5 的单折，样本量太小，只作线索。
- pool30 放大到 30 个构象后 `H_nested` 仍为负（−0.043），与父计划
  "扩池不能救目标错位"（C4）一致。

### 6.6 全靶点 headroom 地图

`headroom_map.csv` 共 464 行，verdict=GO 的 34 行，分布：
PPARA 6、PPARG 6、EGFR 10、FA10 3、PPARA_pool30 3、CDK2 3、MK14_min 1、
**MK14 / BACE1 / ESR1 = 0**。主判定五靶点中只有 PPARA、PPARG 出现过 GO 行，
且几乎都由 `min` 贡献。

## 7. 限制与解释（写结论时必须带上）

1. **统计功效**：每个留出折约 120 配体（~24 active），PR-AUC 的 bootstrap
   SE ≈ 0.05；`min` 在 PPARA 上的 +0.15 是少数几个超过 MDE 的效应。
   EF1% 只基于 ~2 个分子，一律只作参考（计划 §G5 已冻结）。
2. **A/B 半折更小**：`H_nested` 在 ~60 配体（~12 active）上选子集，本身噪声很大；
   这正是"半数据选择不可转移"的机制，但也意味着 4% 的通过率是**上界意义下**
   的低估值还是高估值需要 E4 才能分辨（这也是计划要把 φ 选择交给 E4 的原因）。
3. **敏感性输入不齐**：本地只下载到 MK14 的 min 聚合矩阵；三个 seed 的独立矩阵、
   多套 scaffold 划分不可得（远程 `matrices/` 未同步）。已在 D1/manifest 中登记
   为**未完成项**，不冒充已做。
4. **pool30 只跑到 k≤4**：按计划 §10 的预案（"k≤4 精确 + k=5,6 延长作业；仍不行
   则报告 k≤4 精确上界并标注"）。pool30 本就不进主判定。
5. **四个远程靶点用 problem.json 载体**：与 V5 报告 §7 一致，D1 复核了受体顺序、
   折覆盖与 scaffold 完整率；但矩阵文件本身仍是"未下载"状态，写论文时应改为
   直接使用 `matrices/primary_median_matrix.csv` 并复跑一次作交叉验证。
6. **探索性定位**：E1 是对已被看过的矩阵的重分析（父计划 §5.1），结论只能作为
   假设生成；G1 的 NO-GO 是"上界结论"，不是对某个新方法的判决。

## 8. G1 之后的直接衔接

按计划 §11 / 判定树：

- **NO-GO（本次）**：归档 headroom 地图 + 上界结论；主线转 **E2（预算分配律）**
  与 **E5 臂 A（holo-like 池）**——即"没东西可赢"的结论直接把 docking 预算导向
  "换池子/换预算几何"，而不是继续在决策层换算法。
- 灰区线索（`min` × PPARA/PPARG、oracle-φ 18%）按两级门规则交棒 **E4**：
  把"选 φ"形式化为新的无标签决策问题（φ 的选择规则、可迁移判据、regret 报告）。
- 旧的"QUBO 打不过 greedy"结论**不需要重算**：φ 家族在可落地层面没有改变
  `mean` 的排名结构（mean 平均排名 4.7/8），k 饱和点也没有系统性移动。

## 9. 复现命令

```bash
cd E:/Quant/qubo-receptor-ensemble-selection
PY="C:/Users/18089/.conda/envs/qubo-receptor-ensemble/python.exe"

# D1
$PY scripts/headroom_scan.py verify \
  --prereg configs/experiments/e1_headroom_preregistration.json \
  --assets configs/e1_assets.json --output-dir results/headroom/e1_20260911

# 扫描（分片可切分，--resume 幂等）
for T in MK14 BACE1 ESR1 PPARA PPARG FA10 EGFR CDK2 MK14_min; do
  $PY scripts/headroom_scan.py run --prereg ... --assets ... \
      --output-dir results/headroom/e1_20260911 --targets $T --jobs 1 --resume \
      --skip-perm --skip-phi-selection --skip-figures --allow-missing-primary
done
$PY scripts/headroom_scan.py run --prereg ... --assets ... --output-dir ... \
    --targets PPARA_pool30 --k-max 4 --jobs 1 --resume \
    --skip-perm --skip-phi-selection --skip-figures --allow-missing-primary

# 补齐置换校正 + train-only φ 选择，重建全部产物与图
$PY scripts/headroom_scan.py report --prereg ... --assets ... \
    --output-dir results/headroom/e1_20260911 --jobs 1

# 验证
$PY -m pytest -q tests/test_headroom_fusion.py tests/test_headroom_metrics_parity.py \
    tests/test_headroom_protocol_parity.py tests/test_headroom_subsets.py \
    tests/test_headroom_headroom.py tests/test_headroom_bootstrap.py \
    tests/test_headroom_gate.py tests/test_headroom_assets.py tests/test_headroom_runner.py
```