# 量子辅助预算约束主动 ligand-receptor docking：组会汇报稿

> 更新时间：2026-09-03
>
> 当前状态：真实 active docking 已完成；离线 masked replay 正在远程半核服务器上运行，最终策略比较尚未完成。

## 1. 文献调研：为什么要这样设计

本节先放文献背景，再介绍代码和结果。本次是围绕本项目的定向调研，不是对整个领域的系统综述。文献选择优先覆盖 5 个问题：docking score 是什么、为什么使用多个 receptor、如何选择昂贵任务、如何处理批量任务、什么时候才能声称量子优势。文献链接和元数据以 DOI 官方页面或正式论文页面为准；未对所有论文做完整复现。

### 1.1 Docking score 只能先当作计算代理信号

[Trott and Olson, 2010](https://doi.org/10.1002/jcc.21334) 介绍了 AutoDock Vina 的 scoring function、搜索过程和效率改进。它说明 docking 可以快速给出一个用于排序的计算分数，但这个分数不是实验结合自由能，也不是生物活性标签。

[Mysinger et al., 2012](https://doi.org/10.1021/jm3006879) 发布了 DUD-E 数据集，用 active 和 decoy 评估虚拟筛选方法。它适合做方法学 benchmark，但 active/decoy 标签属于评价信息。若在选择任务时提前使用这些标签，得到的结果就不再是公平的主动筛选。

对本项目的直接启示是：预测器可以预测尚未完成的 docking score，但不能把 hidden active/decoy 标签当作 score 预测输入，也不能在每一轮选择时直接计算真实 BEDROC 或 EF。

### 1.2 多 receptor 的意义与代价

[Bottegoni et al., 2008](https://doi.org/10.1021/ci800025s) 对 ensemble docking 做了实验验证，讨论了使用多个 receptor 构象处理蛋白柔性的价值。多个构象可能提供互补的结合信息，但每增加一个 ligand-receptor 对，就增加一次 docking 成本。

这正是本项目从旧问题转向新问题的原因：旧流程先完成全部矩阵，再选择 receptor 子集；新流程面对的是一个尚未完成的任务表，需要决定“下一步算哪些格子”。

### 1.3 主动学习与 Bayesian optimization

[Settles, 2009](https://minds.wisconsin.edu/handle/1793/60660) 系统总结了 active learning 的基本思想：模型不只被动接收数据，还主动选择最值得获取标签的新样本。

[Shahriari et al., 2016](https://doi.org/10.1109/JPROC.2015.2494218) 总结了 Bayesian optimization。它适用于一次评估成本高、不能穷举所有候选的黑盒优化问题。模型需要同时给出预测均值和不确定性，用 acquisition function 决定下一次评估什么。

[Kandasamy et al., 2018](https://proceedings.mlr.press/v80/kandasamy18a.html) 讨论了并行 Bayesian optimization 和 Thompson sampling。一次选择多个任务时，不能只把单任务分数从高到低排列，还要考虑任务之间是否重复、是否互相提供信息。

对本项目的启示是：

- predictor 负责回答“这个未完成任务的 score 可能是多少”；
- acquisition 负责回答“完成这个任务后，最终 ligand 排名预计改善多少”；
- batch interaction 负责回答“两个任务一起做是否比各自单独做更有价值”；
- 真实 score 只能在任务被选中后揭示。

### 1.4 QUBO 与量子优势的边界

[Lucas, 2014](https://doi.org/10.3389/fphy.2014.00005) 说明了如何把许多组合优化问题写成 Ising/QUBO 形式。预算、容量和覆盖约束可以被转换为惩罚项，从而交给不同 solver 处理。

但“可以写成 QUBO”不等于“量子方法已经有优势”。[Rønnow et al., 2014](https://doi.org/10.1126/science.1252319) 讨论了 quantum speedup 的定义和检测，强调必须明确比较对象、问题规模、总运行时间和成功概率。[Preskill, 2018](https://doi.org/10.22331/q-2018-08-06-79) 对 NISQ 时代的硬件限制也给出了清晰背景。

所以本项目的判定标准不是“QUBO energy 更低”，而是：在同一候选集、同一预算、同一约束和可比时间预算下，是否带来更好的端到端任务发现效率或实际资源收益。当前代码中的 `quantum_compatible_simulator` 只是兼容 QUBO 接口的模拟后端，实际复用 simulated annealing，不能称为量子执行。

### 1.5 文献形成的研究空缺

现有文献分别讨论了 docking、受体柔性、active learning、Bayesian optimization 和 QUBO，但这些组件之间仍有一个工程和研究上的连接问题：在有限 docking 预算下，如何用可审计的 score 预测和批量组合优化选择下一批 ligand-receptor 任务，同时把预测误差、批内重复信息、预算和评价泄漏分开记录。

本项目先实现这个闭环，再判断 QUBO 是否比简单的 value-greedy 更好。量子优势是最后一个待验证的问题，不是实现的前提结论。

## 2. 一句话说明项目在做什么

把完整的 ligand-receptor docking 表想象成一张有 9,000 个格子的表。每计算一个格子都要花 docking 成本，但预算不足以一次完成全部格子。系统每一轮做 5 件事：

1. 根据已经看见的 score 预测还没计算的格子；
2. 估计完成某个格子对最终 ligand 排名的预期帮助；
3. 估计一批格子放在一起是否互补；
4. 用带预算和覆盖约束的 QUBO 选择下一批；
5. 只有被选中的格子才揭示真实 score，然后进入下一轮。

预测模型和 QUBO 的职责完全不同：预测模型预测 score，QUBO/solver 选择任务集合。

## 3. 为什么不继续扩大旧 receptor-subset QUBO

旧流程的核心问题是：全部 docking 完成后，从已有矩阵里选择少量 receptor。这个问题的变量数量通常不大，强经典方法很容易解决；而且 QUBO 目标和 outer-fold 的筛选指标并不天然一致。

新流程把决策变量改为任务变量：

```text
z(ligand, receptor) = 1：本轮执行这个 docking 任务
z(ligand, receptor) = 0：本轮不执行
```

这样研究问题变成了预算约束的主动任务选择，更接近真实虚拟筛选中的资源分配问题，也不会把新的逻辑继续堆到旧的 receptor-subset QUBO API 上。

## 4. 当前实现的完整流程

```text
旧流程 prepare
    |
    +--> 旧流程 aggregate 生成完整 score matrix
    |
    +--> active manifest bridge，去掉评价标签
              |
              +--> warm-start：固定基准 receptor 和结构 cluster 覆盖
              |
              +--> predictor：Bayesian residual score 分布
              |
              +--> acquisition：单任务价值和单位成本价值
              |
              +--> batch interaction：后验效用生成互补项
              |
              +--> batch QUBO：预算、ligand、receptor、scaffold 约束
              |
              +--> solver：greedy、局部搜索、模拟退火、量子兼容模拟器
              |
              +--> 只揭示选中任务的真实 score
              |
              +--> 更新部分观测矩阵，进入下一轮
```

离线 replay 使用完整 matrix 作为外部 oracle，但 oracle score 只在策略选中任务之后进入状态。真实 production active docking 则在选中后调用 Uni-Dock；两条路径的输入和输出边界不同。

## 5. 各模块用通俗语言解释

### 5.1 部分观测矩阵

系统维护两类信息：

- 已经完成的任务及其真实 score；
- 尚未完成的任务，但不保存它们的 hidden score。

状态还记录当前轮次、已消耗成本、每个任务成本、scaffold 和 receptor cluster。序列化时只保存可见 score，不把完整矩阵或 hidden label 写进状态。

### 5.2 Warm-start

如果一开始一个 score 都没有，预测器没有任何锚点。因此第一轮固定完成：

- 每个 ligand 在基准 receptor `11OY` 上的 score；
- 额外覆盖 receptor structural cluster；
- ligand 按 scaffold 分层，避免 warm-start 只偏向某一类分子。

warm-start 是初始化策略，不是量子优化的一部分。它的成本和覆盖必须单独报告。

### 5.3 Score predictor

第一版模型不是直接猜原始 score，而是预测相对于基准 receptor 的 residual：

$$
\Delta_{lr}=s_{lr}-s_{l,r0}
$$

最终再恢复预测 score：

$$
\hat{s}_{lr}=s_{l,r0}+\hat{\Delta}_{lr}
$$

模型输入包括 ligand 特征、receptor 特征、两者的交互特征，以及当前已经观测到的 score。模型输出预测均值、方差和 posterior samples。

当前实际实现是具有显式后验协方差的 Bayesian linear residual model，不是完整的层次 Bayesian 模型，也没有把普通点预测模型包装成 Bayesian posterior。它同时与两个简单基线比较：nearest receptor 和 observed-score mean。

### 5.4 Acquisition value

对于一个还没完成的任务，系统从预测分布采样多个假设 score，把假设 score 放入可见矩阵副本，重新生成 ligand 排名，再计算排名效用。多个 posterior samples 的平均值就是预期效用。

单任务价值回答：“做完这个任务，预计能给最终排名带来多少帮助？”单位成本价值再除以任务成本。这里不能直接使用真实 BEDROC、EF1% 或当前 hidden active 标签。

### 5.5 批内互补性

如果两个任务一起完成，能够提供互补信息，它们的联合效用应高于两个单任务效用之和。代码用后验效用计算：

$$
\Gamma_{ab}=F(\{a,b\})-F(\{a\})-F(\{b\})+F(\varnothing)
$$

这不是手工规定“不同 scaffold 加分、相同 receptor 扣分”，而是让互补项来自同一套 score 后验和排名效用。

### 5.6 Batch QUBO

QUBO 接收预测器和 acquisition 已经算好的数值，不负责预测 score。它同时表达：

- 总 docking 成本预算；
- 每个 ligand 每轮最多增加多少个 receptor；
- 每个 receptor 每轮最多处理多少个 ligand；
- 每个 scaffold 每轮最多选择多少个任务；
- 可选 receptor activation cost；
- 等成本任务的 cardinality 约束；
- 不同成本任务的整数化 budget/slack 约束。

惩罚系数在配置中固定，不能根据 outer/test 结果后验调节。

### 5.7 Solver

当前统一比较接口包括：

| 后端 | 作用 | 当前解释 |
|---|---|---|
| `value_greedy` | 按单位成本价值逐任务选择 | 主要基础基线 |
| `greedy_one_swap` | 贪心后做一换一局部改进 | 更强经典基线 |
| `simulated_annealing` | 在任务变量上搜索可行组合 | 经典随机优化 |
| `quantum_compatible_simulator` | 接受 QUBO 输入输出协议 | 模拟适配器，不是真实量子 |
| `exact` | 枚举所有可行组合 | 只适合小候选集 |
| `CP-SAT/MILP` | 可选强经典优化器 | 依赖可用时再加入比较 |

所有策略应接收相同候选集、相同 QUBO、相同约束和相同时间预算。候选裁剪带来的收益不能算作 solver 或量子收益。

## 6. 当前实验设置

| 项目 | 当前设置 |
|---|---:|
| 目标 | MK14 |
| ligand 数 | 600 |
| receptor 数 | 15 |
| 完整任务数 | 9,000 |
| baseline receptor | `11OY` |
| warm-start 任务数 | 1,440 |
| active docking 任务数 | 560 |
| 总观测任务数 | 2,000 |
| 单任务成本 | 3.0（3 个 Uni-Dock seed） |
| 总 docking 成本 | 6,000 |
| 每轮批成本 | 300 |
| active round 数 | 8 |
| score fusion | 3 个 seed 的 median |
| active solver | `quantum_compatible_simulator` |
| 真实量子硬件 | 未使用 |

预测、acquisition、batch interaction、QUBO 和 solver 都是 CPU 阶段。真实 production active docking 的 Uni-Dock 阶段才可能使用 GPU；离线 replay 不使用 GPU。

## 7. 当前已经得到的结果

### 7.1 Predictor gate

截至 2026-09-03，远程服务器日志给出的 masked prediction gate 为：

| 模型 | RMSE |
|---|---:|
| Bayesian residual | 0.871536 |
| nearest receptor | 0.939288 |
| observed-score mean | 1.023637 |

gate 输出 `passed = True`。这说明在当前 masked 设置下，Bayesian residual 的 score 预测 RMSE 优于两个简单基线。

但这只是预测门通过，不等于 QUBO 已经优于 greedy，也不等于发现了量子优势。还需要查看分层误差、排名指标、区间覆盖率和独立的 scaffold-disjoint 验证。

### 7.2 已完成的 production active docking

已有 production artifact 的最终评价为：

| 指标 | 数值 | 通俗解释 |
|---|---:|---|
| BEDROC20 | 0.5324 | 更关注排名前部的整体富集表现 |
| EF1% | 3.3333 | 前 1% 相对随机筛选的富集倍数 |
| EF5% | 2.6667 | 前 5% 相对随机筛选的富集倍数 |
| PR-AUC | 0.4341 | active 比例不均衡时的排序质量 |
| ROC-AUC | 0.7712 | active 与 decoy 的整体区分能力 |

这些数字说明这次运行产生了可评价的筛选排序，但没有 baseline 对比时，不能据此说 QUBO 更好；使用的后端也不是量子硬件。

已有 artifact 的真实状态以 `state.json` 和 `evaluation.json` 为准：观测任务数为 2,000，docking cost 为 6,000，预算已经耗尽。旧运行生成的 `run_summary.json` 仍可能显示 `status=running`，这是摘要写入时机问题，不改变 `state.json` 已经记录的预算耗尽事实；新版本代码已经修复正常结束时的摘要状态。

### 7.3 当前离线 replay

当前远程命令正在运行的是完整 masked replay。它会在同一预算下依次比较多个策略，并在最后才写出 `replay.json`。因为候选池约 400 个，batch interaction 需要大量 pairwise posterior utility 计算；半核 CPU 上运行很慢，14 小时没有最终 JSON 不代表已经接近完成。

在 replay 完成前，以下问题都不能下结论：

- 批量 QUBO 是否优于逐任务 `value_greedy`；
- QUBO 是否优于更强经典方法；
- `quantum_compatible_simulator` 是否有端到端收益；
- 收益来自 QUBO，还是来自 predictor、warm-start 或候选裁剪。

## 8. 数据泄漏边界

汇报时建议用下面这句话说明边界：

> 完整 score matrix 只作为离线 replay 的外部 oracle；任务被策略选中前，预测器、acquisition、QUBO 和 solver 都只能看到部分观测 score。hidden active/decoy 标签只在最终评价时使用。

为避免源 ligand ID 中的 `active`/`decoy` 字样泄漏到输入，replay 输入准备脚本会把 ligand ID 映射为 `L0000` 等不透明 ID，并把标签单独写入 `labels.json`。`id_map.json` 只用于运行后分析，不能传给 predictor、acquisition 或 solver。

## 9. 组会中可以明确回答什么

### 已经有证据的内容

- 旧 canonical prepare 产物可以切出独立 active workflow；
- 部分观测状态、warm-start、预测、acquisition、batch interaction、QUBO、solver 和揭示 score 的软件闭环已经实现；
- hidden score 不会在任务选择前进入 replay 状态；
- hidden label 只在最终评价边界使用；
- predictor gate 在当前 masked 设置下优于两个简单基线；
- production active docking 已按预算完成 2,000 个观测任务；
- 当前所谓 quantum backend 没有执行真实量子硬件。

### 目前不能声称的内容

- 不能说“QUBO 能量更低，所以筛选效果更好”；
- 不能说“模拟退火结果代表量子结果”；
- 不能说“量子硬件返回可行解，所以有量子优势”；
- 不能说“replay 成功，所以真实 docking 一定节省成本”；
- 不能说“QUBO 优于 greedy，所以 ensemble docking 的生物学价值已经成立”。

## 10. 下一步实验与停止条件

### 10.1 先完成 replay 性能和结果审计

当前 replay 完成后，先读取：

```text
prediction_gate.json
replay.json
comparison.md（如果生成）
```

重点比较每个策略的 BEDROC20、PR-AUC、EF1%、EF5% 和 ROC-AUC，同时检查所有策略是否共享 candidate pool、预算、seed 和约束。

### 10.2 做候选裁剪消融

保持 predictor、warm-start、预算、seed、约束和 solver 不变，只改变 `candidate_cap` 或候选裁剪规则。将候选池变化带来的收益单独报告，不能归因于 QUBO 或量子后端。

### 10.3 加入强经典基线

在候选数量足够小的实验中加入 exact。依赖和架构允许时加入 CP-SAT/MILP。大候选池不要直接运行指数枚举 exact，否则比较会失去意义。

### 10.4 量子优势判定

只有同时满足以下条件，才进入量子优势讨论：

1. predictor gate 通过；
2. batch QUBO 在相同候选集和预算下优于逐任务 greedy；
3. QUBO 不劣于强经典组合优化器；
4. 量子后端有真实可核验的执行记录；
5. 收益在候选裁剪、warm-start、预测模型和后处理消融后仍然存在；
6. 统计结果按 target 或独立重复实验报告，而不是只报告一次运行。

### 10.5 失败停止条件

实验开始前应冻结以下停止条件：预算耗尽、连续轮边际效用低于阈值、连续两轮排名变化低于阈值、预测区间达到目标、predictor gate 失败，或 solver 可行性/时间门失败。任何门失败时都记录失败原因，不通过调惩罚系数或偷看 hidden label 继续实验。

## 11. 仓库入口和远程操作文档

完整远程命令见 [主动 docking replay 操作与审计](active_docking_replay_zh.md)。主要入口如下：

```text
scripts/prepare_active_replay_inputs.py
    从 active_manifest.json 和 primary_median_matrix.csv 生成匿名 replay 输入

scripts/run_active_docking.py validate
    验证离线 replay 配置

scripts/run_active_docking.py predict
    运行 masked score prediction gate

scripts/run_active_docking.py replay
    运行多策略 masked replay

scripts/run_active_experiment.py
    从旧 prepare 产物切出的真实 active docking 入口
```

离线 replay 默认不启动真实 docking、远程任务或量子硬件。真实 active docking 与 replay 必须分开汇报。

## 12. 参考文献

1. Trott, O.; Olson, A. J. AutoDock Vina: Improving the Speed and Accuracy of Docking with a New Scoring Function, Efficient Optimization, and Multithreading. *Journal of Computational Chemistry*, 2010. [DOI](https://doi.org/10.1002/jcc.21334)
2. Mysinger, M. M. et al. Directory of Useful Decoys, Enhanced (DUD-E): Better Ligand Discovery via Expanded Benchmarks and New Protein Targets. *Journal of Medicinal Chemistry*, 2012. [DOI](https://doi.org/10.1021/jm3006879)
3. Bottegoni, G. et al. Protein Flexibility in Docking: The Experimental Validation of Ensemble Docking. *Journal of Chemical Information and Modeling*, 2008. [DOI](https://doi.org/10.1021/ci800025s)
4. Settles, B. Active Learning Literature Survey. University of Wisconsin-Madison Technical Report, 2009. [正式页面](https://minds.wisconsin.edu/handle/1793/60660)
5. Shahriari, B. et al. Taking the Human Out of the Loop: A Review of Bayesian Optimization. *Proceedings of the IEEE*, 2016. [DOI](https://doi.org/10.1109/JPROC.2015.2494218)
6. Kandasamy, K. et al. Parallelised Bayesian Optimisation via Thompson Sampling. *Proceedings of Machine Learning Research*, 2018. [论文页面](https://proceedings.mlr.press/v80/kandasamy18a.html)
7. Lucas, A. Ising Formulations of Many NP Problems. *Frontiers in Physics*, 2014. [DOI](https://doi.org/10.3389/fphy.2014.00005)
8. Rønnow, T. F. et al. Defining and Detecting Quantum Speedup. *Science*, 2014. [DOI](https://doi.org/10.1126/science.1252319)
9. Preskill, J. Quantum Computing in the NISQ Era and Beyond. *Quantum*, 2018. [DOI](https://doi.org/10.22331/q-2018-08-06-79)
