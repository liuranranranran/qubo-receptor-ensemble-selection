# E1 Headroom 扫描：实现与结果

> 权威运行：`e1_20260912`（远程 32 vCPU，`REPO_ROOT=/root/qubo-receptor-ensemble-selection`，
> commit `acbb582`）　|　产物：`results/headroom/e1_20260912/`（含完整归档 tar.gz）
> 对照：`e1_local_validation_20260911`（本地验证件，流水线与口径的回归锚点）
> 计划依据：`E:\Quant\docs\qubo\E1实现计划_headroom扫描_20260911.md`（父计划 §3 E1）
> 状态：已实现、已远程全量运行、**G1 = NO-GO**；唯一稳健灰区线索 = PPARA × `min`

运行手册：`docs/headroom_scan_remote_runbook_zh.md`（怎么在服务器上跑、汇总、打包、关机）。

## 0. 一句话结论

在预注册主格（5 靶点 × 5 折 × 8 融合规则 × k∈{2,3} = **400 格**）上：

- 偷看测试标签的 oracle 上界 **`H_raw` 均值 +0.062**（92% 为正、52% 超过噪声地板），
- 但"一半选、另一半评"的可落地 **`H_nested` 均值 −0.017**（仅 35% 为正，**16/400 格通过**），
- 判定门：逐格通过率 **4.0%**、折内 oracle-φ **18.0%**、train-selected φ **2.0%**，k\* 位移 **0/5** 靶点同向。

⇒ 按预注册判据 **NO_GO**："标准工况下'用一半数据选出更好的 k 子集'这件事本身没有可转移的信息量"——
上界告负，主线转 E2（预算分配）+ E5 臂 A（holo-like 池）。

**唯一通过三 seed 敏感性检验的效应**：`PPARA × min`（BEmin）。三个 seed 的 `H_nested` 目标级均值
分别 +0.019 / +0.034 / +0.033（std 0.007），min 聚合 +0.029，逐格通过 10–14/80；
median 矩阵上 `min` 在 k=2..5、3/5 折通过（ratio 4.0–6.5）。这是"靶点特异 × φ 特异"的灰区线索，
按两级门交棒 **E4**（把"选 φ"形式化为新的无标签决策问题），不复活主线。
对照：PPARG 在 median 矩阵上同样有 GO 行，但 seed 间符号翻转（+0.009 / −0.053 / −0.002）→ 判为噪声。

## 1. 实验协议（冻结）

| 项 | 冻结值 |
|---|---|
| 靶点（主判定） | MK14 / PPARG / BACE1 / ESR1 / PPARA |
| 扩展/对照 | FA10（13 受体）、EGFR（12 受体）、PPARA pool30（30 受体）、CDK2（160 配体 4 折对照） |
| 主格 | `k ∈ {2,3}` × 8 φ × 5 靶点 × 5 折 = 400 格 |
| k 扫描范围 | 1–6（全部资产跑满，含 pool30） |
| 融合族 φ | `mean,min,max,zmean,gmean,hmean,ranksum,rrf`（rrf K=60） |
| 主指标 | PR-AUC（average precision）；次指标 BEDROC20 / ROC-AUC / EF1,5,10 |
| 评价单元 | V5 外层留出折（每折约 120 配体、约 24 active），与 `scripts/nested_outer_k_evaluation.py` 同口径 |
| `S_ref` | 训练折上 `{greedy, single}` 中较优者（主指标） |
| `H_raw` | `max_{|S|=k} U_test(S) − U_test(S_ref)`（精确枚举全部子集） |
| `H_nested` | 测试折按 scaffold 分 A/B：A 上选 `argmax_S U_A`、B 上评 `U_B`，反向再平均 |
| `H_perm` | `H_raw − q95(max_{S∈TopM} U_test(S^perm))`，TopM=2000、200 次标签置换 |
| 噪声地板 | 同格 `greedy vs single` 的 scaffold 聚类配对 bootstrap（B=2000）的 **SE** |
| 通过判据 | `H_nested_lower95 > noise_floor` 且 `ratio > 1` |
| MDE | `t(0.975, df) · SE · √2` |
| G1 门 | <10% → NO-GO；≥30% → GO-φ；10–30% → 灰区 + 预注册 tie-break |

## 2. 代码结构

```text
src/qubo_receptor_ensemble/headroom/
    fusion.py        8 条 φ + train-only 参数拟合 + 子集评分热路径
    metrics_fast.py  O(n) 版 BEDROC / PR-AUC / ROC-AUC / EF（与 screening.py 数值一致）
    subsets.py       位掩码枚举、utility、精确 oracle
    headroom.py      H_raw / H_nested / H_perm、A/B scaffold 划分、内层折分配
    bootstrap.py     scaffold 聚类 bootstrap、噪声地板、MDE
    gate.py          预注册 G1 判定
    assets.py        D1 输入核验、matrix/problem.json 双载体、SHA-256
    seed_matrices.py 从既有 score_tables 重建单 seed / min / median 矩阵
    summary.py       运行汇总（看护脚本与归档用）
    runner.py        分片扫描、逐片 checkpoint、并行、产物、图
scripts/headroom_scan.py                  CLI：verify / run / report / extract-seeds / summarize
scripts/run_e1_headroom_remote.sh         远程一键电池（T1–T6 → D1 → 主扫描 → 组装 → seed → 敏感性 → 审计）
scripts/e1_headroom_watch_and_shutdown.sh 看护：等进程结束 → 汇总 → 打包 → （可选）关机
configs/experiments/e1_headroom_preregistration.json  冻结预注册
configs/e1_assets.json / e1_assets_remote.json / e1_assets_remote_sensitivity.json  资产表
tests/test_headroom_*.py                  62 项测试（T1–T6 + 回归）
```

CLI 速查：

```bash
python scripts/headroom_scan.py verify  --prereg ... --assets ... --output-dir <OUT>
python scripts/headroom_scan.py run     --prereg ... --assets ... --output-dir <OUT> --jobs 32 --resume
python scripts/headroom_scan.py report  --prereg ... --assets ... --output-dir <OUT> --jobs 32
python scripts/headroom_scan.py extract-seeds --run-dir <canonical run> --output-dir <OUT/seed_matrices/<t>>
python scripts/headroom_scan.py summarize --run-dir <OUT> [--sensitivity-dir <OUT_sens>]
```

## 3. D1 输入核验（`e1_20260912/input_manifest.json`）

| 资产 | role | 载体 | 配体×受体 | 折 | 备注 |
|---|---|---|---|---|---|
| MK14 | primary | matrix_csv | 600×15 | 5 | canonical `matrices/primary_median_matrix.csv` |
| PPARG | primary | matrix_csv | 600×16 | 5 | canonical |
| BACE1 | primary | matrix_csv | 600×15 | 5 | canonical |
| ESR1 | primary | matrix_csv | 600×15 | 5 | canonical |
| PPARA | primary | matrix_csv | 600×15 | 5 | canonical |
| FA10 | secondary | matrix_csv | 600×13 | 5 | 输入随仓库分发（`data/processed/`） |
| EGFR | secondary | matrix_csv | 600×12 | 5 | 同上 |
| PPARA_pool30 | pool | problem.json | 600×**30** | 5 | 矩阵文件不存在，按设计用 problem.json 载体 + scaffold/outer_fold 回填（600/600 计数） |
| CDK2 | control | matrix_csv | 160×16 | 4 | 对照面板，不进 G1 |

全部 `status=ok`，主判定 5 靶点全部 `source=matrix_csv`；所有输入 SHA-256 记录在 manifest。

## 4. 正确性验证（T1–T6 + 回归，62 项全绿）

| 测试 | 内容 | 结果 |
|---|---|---|
| **T1** 指标 parity | `metrics_fast` vs `screening.py`（200 随机 + 224 含并列向量） | max abs diff **1.7e-16**（门限 1e-12） |
| **T2** 协议 parity | MK14 + mean + k=1..6：现跑 golden 脚本 vs 快路径；固定-k 子集 vs 归档 V5 decision log | 子集 **30/30 一致**；max abs diff **2e-16**（门限 1e-9） |
| **T3** 单受体退化 | `\|S\|=1` 时 8 条 φ 秩相关 | 全 **1.0** |
| **T4** 泄漏测试 | 只改测试折 → z 统计/平移常数/秩参照逐位不变 | 通过 |
| **T5** 方向单调性 | 整列平移：`ranksum`/`rrf` 不变；`mean` 随之移动 | 通过 |
| **T6** 可复现 | 同 seed/输入连跑两次产物 SHA-256 一致 | 通过 |
| 回归 | `--resume` 必须复用分片 checkpoint（mtime 不变）；空 run_id 的旧 checkpoint 兼容 | 通过 |

> 归档 `E:\Quant\nested_v5\mk14\folds_long.csv` 与现跑 golden 脚本有最大 1.74e-6 的旧运行偏差
> （该 CSV 只有 6 位小数，且为 2026-08-27 的旧运行）；子集完全一致，T2 以"现跑 golden 路径"为准。

## 5. 运行与产物

权威运行规模：9 资产 → **352 分片**（精确枚举，pool30 跑满 k≤6）；置换校正 **704 格**；
train-only φ 选择 **7 个靶点 × 5 折**；敏感性 **20 资产 / 800 分片**（5 个 min 聚合矩阵 + 15 个单 seed 矩阵）。

```text
results/headroom/e1_20260912/
  input_manifest.json  cell_metrics.csv  headroom_map.csv  bootstrap_report.json
  phi_selection.json   permutations.json gate_g1.json      run_manifest.json
  summary.json summary.txt STATUS.txt   figures/*.png
  e1_20260912_products_20260913_042229.tar.gz   # 远程归档（含敏感性 run 的完整 cell_metrics）
results/headroom/e1_20260912_sensitivity/       # 敏感性 gate/map/manifest（大表在 tar.gz 内）
results/headroom/e1_local_validation_20260911/  # 本地验证件（同一口径的回归对照）
```

看护脚本 `scripts/e1_headroom_watch_and_shutdown.sh` 在电池结束后自动：`summarize` → tar 打包
（默认排除 `cells/` 分片）→ 写 `*_STATUS.txt` → 满足"gate 存在且敏感性 ≥640 分片"时 `shutdown -h now`。

## 6. 结果

### 6.1 主判定 G1（`gate_g1.json`）

| 级别 | 通过率 | 计数 |
|---|---|---|
| 逐格（400 格） | **4.0%** | 16 / 400 |
| 折内 oracle φ（50 单元） | **18.0%** | 9 / 50 |
| train-selected φ（50 单元，可落地） | **2.0%** | 1 / 50 |

`decision = NO_GO`（headline 2% < 10%）；tie-break：oracle−train 差 16 个百分点，k\* 位移
`[0, −1, −1, +1, −1]`（0/5 靶点右移 ≥2）。

### 6.2 有没有可赢的东西？

| 主格 400 格 | 均值 | 中位 | >0 | >噪声地板 |
|---|---:|---:|---:|---:|
| `H_raw`（后视 oracle 上界） | **+0.0621** | +0.0494 | 92.2% | 52.2% |
| `H_nested`（A 选 B 评，可落地） | **−0.0165** | −0.0253 | 34.8% | 4.0% |
| 噪声地板（greedy−single 配对 SE） | 0.0500 | 0.0450 | — | — |
| MDE（配对，均值） | 0.1413 | — | — | — |

> oracle 对 105 个子集取最优平均多拿 0.062 PR-AUC（≈1.2× 噪声地板），但把半折选出的子集
> 放到另半折收益全消 → **"有可赢空间"只存在于后视镜里**。

### 6.3 φ 是不是主因？

主格（k∈{2,3}）各 φ 的平均 `H_nested`：

| φ | min | rrf | hmean | zmean | gmean | mean | ranksum | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mean H_nested | **+0.0284** | −0.0004 | −0.0079 | −0.0187 | −0.0241 | −0.0257 | −0.0397 | −0.0439 |

- `mean` 的 50 单元排名**平均 4.72/8**（中位 5，22% 落倒数两名）→"mean 垫底"的预期不成立；
  `min` 平均最好，`max`/`ranksum` 最差。
- best-φ 的 oracle-k\* 相对 `mean` 位移 `0/−1/−1/+1/−1` → **没有 φ 让 k 饱和点系统右移**。
- 16 个通过格集中在 PPARA(8)/PPARG(8)，其中 11 个是 `min` → φ 效果是靶点特异、折特异。

### 6.4 H_perm 的（不）可用性

预注册公式 `H_perm = H_raw − q95(max_S U_test(S^perm))` 全部为负（主格均值 −0.329）：两项量纲不同
（`q95` 是置换标签下 2000 个子集的最大 AP，绝对水平 ≈0.3–0.4；`H_raw` 是差值 ≈0.06）。
产物另存 `h_perm_alt = U_test(oracle) − q95(置换最大值)`（均值 +0.274）但它衡量的是"评分是否有信号"，
不是"选择是否膨胀"。**G1 只用 `H_nested`**，该公式问题作为方法学备注记录。

### 6.5 敏感性：min 聚合 + 三 seed 独立（计划 §7.1 / §7.2）

目标级 `H_nested` 均值（k∈{2,3}，每变体 80 格）：

| 靶点 | min 聚合 | seed 20260821 | seed 20260822 | seed 20260823 | seed 均值±std | 通过格（min / 三 seed） |
|---|---:|---:|---:|---:|---:|---:|
| **PPARA** | **+0.0287** | +0.0192 | +0.0336 | +0.0332 | **+0.0287 ± 0.0067** | 12 / 10,13,14 |
| PPARG | +0.0280 | +0.0087 | **−0.0532** | −0.0019 | −0.0154 ± 0.0270 | 6 / 6,2,6 |
| MK14 | +0.0002 | −0.0051 | −0.0062 | −0.0255 | −0.0122 ± 0.0094 | 0 / 0,0,0 |
| BACE1 | −0.0517 | −0.0414 | −0.0527 | −0.0741 | −0.0561 ± 0.0136 | 0 / 0,0,0 |
| ESR1 | −0.0423 | −0.0268 | −0.0347 | −0.0341 | −0.0319 ± 0.0036 | 0 / 0,0,0 |

- **PPARA 是唯一三 seed 同号为正的靶点**（std 0.007，远小于均值）；
- **PPARG 是"median 矩阵制造的 GO"**：三 seed 符号翻转（一个 seed −0.053），min 聚合值高出
  seed 均值 +0.0435（min 聚合按格取最优 seed，属上界量，不是可落地选择）；
- 逐格 seed 噪声：同格三 seed 的 std 0.026–0.087、符号一致率 50–80%，
  与效应量（≈0.03）同量级 → **逐格差异不得解释为方法差异**（MDE≈0.14），只有目标级聚合可读。

主运行地图 GO 行（432 行中 35 行 GO）：

| 靶点 | GO 行 | 说明 |
|---|---:|---|
| PPARA | 6 | `min` k=2..5（**go_folds=3**，ratio 4.0–6.5）+ `gmean`/`hmean` k=2（go_folds=1） |
| PPARG | 6 | `mean`/`min`/`ranksum`/`rrf`×2/`zmean` 集中在 k=2，go_folds 1–2 → 见敏感性，判为不稳 |
| EGFR（扩展） | 12 | 全部 **go_folds=0**（均值过线但无单折过线）→ 弱线索 |
| PPARA_pool30 | 5 | `min` k=2..6，go_folds=1，ratio 1.6–2.1（与 PPARA 同面板一致） |
| FA10（扩展） | 3 | go_folds=0 |
| CDK2（对照） | 3 | go_folds ≤1 |
| MK14 / BACE1 / ESR1 | **0** | 稳定无 GO |

### 6.6 PPARA × `min` 的细节（灰区线索的全部证据）

- 主运行（median 矩阵）：`min` k=2/3/4/5 → `H_nested` +0.148/+0.183/+0.170/+0.163，
  ratio 6.5/4.6/4.0/3.8，**3/5 折通过**（fold 1/2/5 为正，fold 3/4 为负）；
- 三 seed：目标级 +0.019/+0.034/+0.033，`min` 变体 +0.029，通过格 10–14/80；
- 同面板扩池（pool30）：`min` k=2..6 ratio 1.6–2.1，1/5 折通过 → 方向一致但更弱；
- `mean` 在 PPARA 上也有弱正值（+0.037/+0.096/+0.056）→ 效应部分来自靶点本身，
  部分是 φ=`min` 的贡献（主格上 `min` 比 `mean` 高 +0.054）。

## 7. 限制与解释边界

1. **逐格功效**：留出折约 120 配体（~24 active），配对 bootstrap SE ≈ 0.05、MDE ≈ 0.14；
   同格 seed 噪声 0.026–0.087 与效应同量级。**逐格/单折结论一律不可采信**，
   本文只用目标级（80 格）聚合与三 seed 一致性。
2. **A/B 半折更小**（~60 配体、~12 active）：`H_nested` 的 4% 通过率本身是选择噪声下的结果，
   这也是为什么"φ 选择"必须交给 E4 用新的判据处理。
3. **min 聚合矩阵是上界**：它按格取最优 seed（PPARG 上比 seed 均值高 +0.044），
   因此 §7.1 的"min 敏感性"只能当上限；可落地的稳健性判定必须看 §7.2 的三个 seed。
4. **pool30 用 problem.json 载体**（矩阵文件未同步），D1 已复核受体顺序/折/scaffold；
   论文前建议回补 `matrices/` 复跑一次。
5. **H_perm 公式**见 §6.4，G1 不依赖它。
6. **探索性定位**：E1 是对已看过的矩阵的重分析（父计划 §5.1），结论只作为假设生成与上界证据。

## 8. G1 之后的直接衔接

| 结果 | 动作 |
|---|---|
| **NO-GO（本次权威判定）** | 归档 headroom 地图 + 上界结论；主线转 **E2（预算分配律）** 与 **E5 臂 A（holo-like 池）** |
| 灰区线索：**PPARA × `min`** | 交棒 **E4**：把"选 φ"形式化为无标签决策问题（特征—判据—regret），并在 E2 的分配几何里复核 `min` 行为 |
| 反例：**PPARG 的 median-GO** | 写入失败案例：median-of-3 会在目标级制造假阳性；E4 的判据必须用 seed 一致性约束 |
| 旧结论 | "QUBO 打不过 greedy""k 在 2–3 饱和"**不需要重算**（`mean` 平均排名 4.7/8，k\* 无系统位移） |

## 9. 复现命令

```bash
# 远程一键（32 vCPU，10–20 分钟；JOBS=1 约 3–3.5 小时）
cd /root/qubo-receptor-ensemble-selection
JOBS=32 bash scripts/run_e1_headroom_remote.sh

# 只补跑敏感性（主运行 checkpoint 会被 --resume 复用）
JOBS=32 RUN_ID=e1_20260912 bash scripts/run_e1_headroom_remote.sh

# 跑完自动汇总 + 打包 +（可选）关机
nohup env RUN_ID=e1_20260912 AUTO_SHUTDOWN=1 \
  bash scripts/e1_headroom_watch_and_shutdown.sh \
  > "$DATA_ROOT/results/headroom/e1_20260912.watch.log" 2>&1 &

# 本地复核（对下载回来的产物）
python scripts/headroom_scan.py summarize \
  --run-dir results/headroom/e1_20260912 \
  --sensitivity-dir results/headroom/e1_20260912_sensitivity
python -m pytest -q tests/test_headroom_*.py
```

详细步骤、时间预算、排错表见 `docs/headroom_scan_remote_runbook_zh.md`。