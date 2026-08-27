# Curious-VLA Dataset V2：CDT-HLA-GRPO 下一步闭环执行台账

> 本文档是 `dataset_v2_20260825` 上验证 `SafetyMix-1K + CDT Hierarchical Lexicographic Advantage` 的唯一实时计划与结果台账。既有数据、模型和已完成实验事实继续以 [`dataset_v2_grpo_experiment_ledger.md`](dataset_v2_grpo_experiment_ledger.md) 为准；本文不覆盖旧证据，只接管下一步执行顺序。

## 1. 研究问题、当前状态与结论边界

本轮只回答一个主问题：

> 在同一 `SafetyMix-1K`、同一 Stage-2、同一 `G=4`、同一训练预算，以及相同 reward server、production SDR 计算和评价字段下，只在组内出现 CDT safety conflict 时改写 advantage，是否比标准 GRPO 提高严格安全表现且不损害总体驾驶质量？

当前状态：

- Dataset V2 数据、10K image/cache、Phase-1 6K shared rollout bank、dev/final split 均已冻结；
- `V2-I0` 已证明 Stage-2 具有有效图像敏感性；
- `V2-S0`、`V2-S1`、`V2-R0`、`V2-T0-SDR/RAW` 均已完成，可直接复用，不重新推理；
- `V2-E0` 已完成，Stage-2 dev anchor 为 PDMS `0.947464`、PDMS scaled `0.908727`、旧 Safe `0.997000`；
- 旧 `SLDR-current` 已由 `V2-R0` 关闭，不恢复、不调 `0.5/0.1/0.6`；
- 原队列中的 `V2-R4-SDR/RAW/FALS/ADAS` 暂停，不作为本轮前置条件；本轮 baseline 必须是同一 SafetyMix manifest 上的 `SM4-SDR`；
- `V2-H0` 已完成，全部技术门控通过，但两个 material-change 科学门控失败；依照冻结规则，本轮已关闭 HLA 正式训练，`V2-HT0-HLA`、`V2-SM4-SDR/HLA`、matched seeds 与 final 均未启动。

执行分支于 2026-08-27 冻结为：

- Git remote：`post-training`；
- branch：`codex/dataset-v2-cdt-hla-execution`；
- fork point：`d8b2c38`（`Record CDT-HLA H0 terminal decision`）；
- 本台账后续若有经明确重新预注册的补充执行，只允许从该分支建立独立 server worktree/实验目录，不得直接复用或切换其他正在运行任务的 source worktree；分支变更必须先回填本台账。

创建该分支时远程服务器正在执行其他任务，因此本次只完成 Git 分支隔离与台账冻结，未连接、切换或修改服务器上的现有 source、进程和实验目录。

本轮结论边界：

- `SafetyMix` 是当前 Stage-2 policy 的安全决策边界 selector，不表述为数据集固有危险度；
- 主 contrast 只识别“在 SafetyMix 上，HLA estimator 相对标准 GRPO”的作用，不单独识别 SafetyMix 相对 Random 的收益；
- `CDTR` 区间 reward 只作为 CPU diagnostics 和未来可选 reward-only baseline，不阻塞 HLA 训练；
- 单 seed dev 结果只作筛选，不能写成最终泛化结论；
- V2 dev 是方法冻结后的 matched exploratory confirmation，不表述为严格 pre-registered confirmation；
- final reserve 在三 seed 确认前继续保持未访问。

## 2. 直接复用的冻结证据

| 资产或阶段 | 复用内容 | 本轮处理 |
| --- | --- | --- |
| `V2-D0` | Dataset V2 parquet、manifest、image/cache、Stage-2 hash 与 namespace | 不重建数据；代码完成后只做一次 source rebind |
| `V2-I0` | correct/shuffled image 门控已通过 | 不重跑 |
| `V2-S0` | 同一 500 token 的 4 个独立 G4 block，seed `20260825`–`20260828` | 只复算 CDT mixed-tier membership、CV 和 Jaccard |
| `V2-S1` | 6,000 groups / 24,000 rollout，G=4，字段完整 | 直接构建 SafetyMix 和三种 advantage geometry |
| `V2-M0` | intent quota `634/251/115`、per-log cap `5`、确定性约束选择入口 | 复用现有 constrained selector，不新增优化器 |
| `V2-R0` | production SDR/SLDR 复算与真实 GRPO geometry 入口 | 复用 group 读取、coverage、material-change 统计 |
| `V2-T0-SDR` | 24 GB、reward、parser、checkpoint、TensorBoard、资源回收已通过 | `SM4-SDR` 不再单独 smoke |
| `V2-E0` | 同一 2,000-token dev 的 Stage-2 anchor | 直接复用，不重新评估 |
| cluster bootstrap | `projects/dataset_v2/experiment_pipeline.py::cluster_bootstrap` | 扩展 StrictClear/tier 指标，不另建统计框架 |

不得为了新路线重复 D0/I0/S0/S1 rollout、旧 reward smoke 或 Stage-2 dev evaluation。若代码提交导致现有 freeze marker 要求 source 一致，只调用既有冻结入口重新绑定 source 和输入 hash，不重建任何资产。

## 3. 公平比较与最小实验矩阵

### 3.1 主 contrast

\[
\Delta_{HLA}=V2\text{-}SM4\text{-}HLA-V2\text{-}SM4\text{-}SDR
\]

两侧必须完全相同：

- SafetyMix-1K manifest、token 顺序；
- Stage-2 初始化与 model hash；
- 相同 reward server、production SDR 计算和完整 NAVSIM 分项记录；
- `G=4`、1,000 scene、250 steps、4 groups/step、4,000 train rollout；
- LoRA、optimizer、LR、KL loss、clip、bf16、batch 和显存设置；
- train/dev generation seed 与 decode；
- step250 checkpoint 和同一 2,000-token dev evaluation；
- 唯一主要变量为从轨迹级指标构造 policy advantage 的规则：`algorithm.adv_estimator: grpo -> cdt_hla_grpo`。

这里的“matched reward”只表示两侧调用相同 reward server 并保留相同的 production SDR 与评价字段，不表示有效优化信号未变：`SM4-SDR` 始终优化 scalar SDR preference；`SM4-HLA` 在 valid mixed-tier group 中优化 `CDT hierarchy > bounded within-tier SDR residual`，在 all-valid same-L3/L2 group 中完全退化为 standard SDR-GRPO。这正是主 contrast 要识别的方法变量，而不只是一次 numerical normalization 替换。

### 3.2 首轮正式矩阵

| 顺序 | ID | Selector | Reward/记录 | Advantage | 状态 |
| ---: | --- | --- | --- | --- | --- |
| 0 | `V2-E0` | 无训练 | production SDR | — | `COMPLETE` |
| 1 | `V2-H0` | Phase-1 6K replay | SDR/CDTR diagnostics | SDR/CDTR/HLA replay | `COMPLETE_CLOSE_HLA` |
| 2 | `V2-HT0-HLA` | SafetyMix mixed-tier 40 | production SDR | CDT-HLA + scale-only normalization | `SKIPPED_BY_H0_SCIENTIFIC_GATE` |
| 3 | `V2-SM4-SDR` | SafetyMix-1K | production SDR | standard GRPO | `SKIPPED_NO_ACTIVE_HLA_CONTRAST` |
| 4 | `V2-SM4-HLA` | 同一 SafetyMix-1K | production SDR | CDT-HLA + scale-only normalization | `SKIPPED_BY_H0_SCIENTIFIC_GATE` |
| 5 | `V2-SM4-CDTR` | 同一 SafetyMix-1K | CDT interval reward | standard GRPO | `OPTIONAL_NOT_IN_FIRST_ROUND` |

首个单 seed 方法结果只需要一次 CPU replay、一次 HLA smoke 和两个正式 run。`SM4-CDTR` 不延迟主 contrast；只有后续明确需要区分“tier 进入 reward”与“tier 进入 advantage”时才单独预注册。

## 4. CDT 安全定义

记：

- `C = no_at_fault_collisions`；
- `D = drivable_area_compliance`；
- `T = time_to_collision_within_bound`。

所有 parsed rollout 必须唯一映射：

| Tier | 名称 | 定义 |
| ---: | --- | --- |
| `L3` | Fully Clear | `C == 1 and D == 1 and T == 1` |
| `L2` | TTC Risk | `C == 1 and D == 1 and T < 1` |
| `L1` | Hard Violation | `C > 0 and (C < 1 or D < 1)` |
| `L0` | Critical Collision | `C == 0` |

Parse failure 使用独立 `validity=invalid`，报告时不得记作正式 safety tier，也不得进入 CDT comparison。HLA 对 invalid completion 的 advantage 固定为 0；若 valid completion 少于 2，整个 group 的 HLA advantage 为 0。parser/reward 仍照常记录，但本方法不声称其 scalar penalty 会穿过 HLA estimator 产生 policy gradient，parse failure 由既有 `>=99.5%` 技术门控约束。

进入 tier 判定前，CDT 单一真源必须以 `epsilon=1e-6` canonicalize：

\[
C\rightarrow\{0,0.5,1\},\qquad D,T\rightarrow\{0,1\}
\]

距离某个合法离散值不超过 `epsilon` 时映射到该值。`parsed_ok=true` 但无法落入合法集合的数值必须记为 `mapping_error` 并使 H0 technical gate 失败；不得猜测、截断或静默降级为 `invalid/L0`。只有真实 parse failure 使用 `validity=invalid`。

硬约束：

- `C=0.5` 只能进入 `L1`，不得进入 `L2/L3`；
- 旧 `safe = C>0 and D>0` 只保留 backward-compatible metric，不再用于主门控；
- 新主指标 `StrictClear = parsed_ok and C==1 and D==1 and T==1`；
- `StrictClear`、tier 与 CDTR 必须共用 canonicalized CDT 值和同一映射实现；
- 本版本只声明 CDT safety，不扩展到未进入当前训练日志的其他交通规则指标。

## 5. SafetyMix-1K

### 5.1 Mixed-tier 定义

一个 token 的冻结 G4 group 同时满足：

- 四条 rollout 全部 parse 成功；
- 四条均完成 CDT tier 映射；
- `unique(tier) >= 2`。

它表示 Stage-2 在该 scene 上生成了跨安全等级行为，是 policy-dependent safety-boundary evidence。

### 5.2 选择规则

直接复用 `projects/dataset_v2/experiment_pipeline.py::constrained_ranked_select`：

1. Phase-1 6K token 中，valid mixed-tier 的 priority 为 1，其余 fully-valid token 为 0；
2. 同 priority 内使用 selector seed `20260825` 和 namespace `safetymix-g4` 的 stable hash；
3. intent quota 恰好为 straight/left/right `634/251/115`；
4. per-log cap 固定为 `5`；
5. mixed-first 选满 1,000，不启用 Extension，不从 dev/final 补数据；
6. membership 选满后，忽略 mixed/single priority，使用 seed `20260825` 和独立 namespace `safetymix-train-order` 对选中 1,000 条做一次 stable-hash 排序；mixed-first 只决定成员，不形成隐式 curriculum；
7. membership hash、最终 order hash、mixed 数、tier composition、intent/log 分布全部写入报告，`SM4-SDR/HLA` 共用完全相同的冻结顺序。

该规则不要求“544 全收”，而是在现有 quota/cap 下用项目已有确定性选择器尽可能优先 mixed-tier。若选择后 safety signal 不足，由 H0 geometry gate 决定是否训练，不再增加 selector 参数。

### 5.3 稳定性复算

复用 V2-S0 相同 500 token 的前四个独立 G4 block：

- mixed-tier ratio CV `<= 0.20`；
- mixed-tier membership median pairwise Jaccard `>= 0.50`；
- 四个 block 均为 500×4、字段完整且 finite。

当前只读预估约为 CV `0.051`、median Jaccard `0.704`；正式 H0 必须由冻结代码重新生成报告，不能手填该数值。

## 6. CDT Hierarchical Lexicographic Advantage

### 6.1 Validity boundary 与 tier advantage

对同一 G4 group，先定义 valid completion 子集：

\[
V=\{i:\operatorname{parsed\_ok}_i=1\},\qquad G_v=|V|
\]

invalid completion 不属于 safety tier、不参与 pairwise comparison，最终 advantage 固定为 0。若 `G_v<2`，整个 group 为 0。仅当 valid subset 中存在至少两个 CDT tier 时，对 `i \in V` 定义：

\[
A_i^{tier}=\frac{1}{G_v-1}\sum_{j\in V,j\ne i}\operatorname{sign}(L_i-L_j)
\]

CDT hierarchy 仅为 `L0 < L1 < L2 < L3`；validity 与 safety 完全分离。

必须满足：

- `A_tier` 范围位于 `[-1,1]`；
- group 内和为 0；
- 对任意 `L_i > L_j`，均有 `A_i^{tier} > A_j^{tier}`。

这是 weak lexicographic 约束：它保证 valid completion 的跨等级 advantage 顺序严格服从 CDT hierarchy，但不要求只要更高等级存在，所有非最高等级样本都取得绝对负 advantage。例如 `[L3,L2,L0,L0]` 中 `L2` 的 tier advantage 仍为 `1/3`，这是预期行为。

固定例：

- `[L3,L2,L0,L0] -> [1,1/3,-2/3,-2/3]`；
- `[L3,L3,L2,L2] -> [2/3,2/3,-2/3,-2/3]`；
- `[L3,L2,L1,L0] -> [1,1/3,-1/3,-1]`。

### 6.2 Mixed-tier secondary quality residual

令 `Q_i` 为 standard GRPO 实际接收的 production scalar SDR（当前为 `pdms_scaled`）。仅在 valid mixed-tier group 的 `L3/L2` tier 内，对成员集合 `S_l` 定义：

\[
d_i=Q_i-\overline{Q}_{S_l},\qquad
m_l=\max_{j\in S_l}|d_j|
\]

\[
A_i^{within}=\begin{cases}
0,&m_l=0\\[2mm]
\dfrac{d_i}{m_l+10^{-6}},&m_l>0
\end{cases}
\]

因此 `A_within \in [-1,1]`、每个 tier 内和为 0，并保留 SDR 的连续 relative geometry；同 tier 成员数为 1 或全部同分时自然为 0。`L1/L0/invalid` 的 `A_within=0`，第一版不实现 Pareto 分支。

### 6.3 分支、组合与 scale

HLA 先按 valid subset 分支：

1. `G_v<2`：整个 group `A_train=0`；
2. valid subset 全为 `L3` 或全为 `L2`：直接调用现有 standard SDR-GRPO estimator 处理 `Q_V`，invalid 位置补 0，不再做额外 scale；当 `G_v=G=4` 时必须与 baseline advantage 完全一致；
3. valid subset 全为 `L1` 或全为 `L0`：整个 group `A_train=0`，不允许低优先级质量补偿硬失败；
4. valid subset 包含多个 CDT tier：对 `i in V` 构造：

\[
A_i^{raw}=A_i^{tier}+0.125A_i^{within}
\]

其中 `0.125 = 1/(2G)`，仍按原始 G4 固定，不随 `G_v` 调整、不做 sweep。由于 `2<=G_v<=4`，两个 occupied tier 的最小 tier-advantage 间距至少为 `2/(G_v-1)>=2/3`，而 within-tier 项最坏可缩小 `2*0.125=1/4`，因此任意 valid cross-tier raw margin 满足：

\[
A_i^{raw}-A_j^{raw}\ge\frac{2}{G_v-1}-\frac{1}{4}\ge\frac{5}{12}>0,\qquad L_i>L_j
\]

mixed-tier 完成 hierarchical construction 后，只在 valid subset 上做与现有 standard GRPO 路径相同的正尺度归一化，invalid 位置保持 0：

\[
A_i^{train}=\begin{cases}
0,&i\notin V\ \text{or}\ \operatorname{std}_{V}(A^{raw})=0\\[2mm]
\dfrac{A_i^{raw}}{\operatorname{std}_{V}(A^{raw})+10^{-6}},&i\in V
\end{cases}
\]

实现必须直接复用 standard GRPO 的 `std` convention 和 `epsilon`，避免引入新的尺度定义。这一步不改变 valid subset 内的 tier ordering、tier/within 相对 geometry 或 raw-margin 正号，也不从 scalar reward 重新构造 advantage。valid subset 及整个 group 的 advantage 和均为 0。

最终 `A_train` 作为 outcome advantages/returns 进入现有 actor update；PPO ratio、clipping、KL loss、LoRA、optimizer 和 generation 全部不变。

必须区分两个不变量：all-valid same-L3/L2 满足 `A_HLA=A_standard_SDR`，即 no eligible safety conflict 时不干预原质量学习；all-L1/L0 仍为 0，这是非补偿性安全设计，不宣称 baseline identity。mixed valid/invalid group 只对 valid subset 学习，invalid 为零梯度，其风险由 parse-rate gate 显式约束。

## 7. CDTR diagnostics

CPU replay 同时计算但首轮不训练：

\[
R_{CDT}=\frac{2L+q_L}{7},\qquad
q_L=\begin{cases}
SDR,&L\in\{L2,L3\}\\
0,&L\in\{L0,L1\}
\end{cases}
\]

它只回答 tier 写入 scalar reward 后，标准 GRPO 能利用多少几何变化。HLA 正式训练仍调用相同 production SDR reward entrypoint，但 CDT tier 与 `pdms_scaled` 通过 reward metrics 进入 estimator，在 valid mixed-tier group 中构造 `CDT hierarchy > bounded within-tier SDR residual` 的有效优化信号。

## 8. 最小实现范围与测试

### 8.1 必要改动

只实现以下生产路径：

1. 新增一个 CDT 单一真源，负责 CDT canonicalization、tier、StrictClear 和 CDTR diagnostics；
2. 在现有 Dataset V2 pipeline 中增加一次性 `V2-H0`：复算 S0 稳定性、构建 SafetyMix、计算 SDR/CDTR/HLA geometry；不另建选择框架；
3. 在 `core_algos.py` 注册 `cdt_hla_grpo`，复用 standard GRPO estimator 实现 same-L3/L2 identity，并直接实现 mixed-tier HLA 分支；
4. reward manager 返回的 `pdms_scaled/C/D/T/parsed_ok` 在训练 driver 上按原 batch 顺序传给 HLA estimator；标准 GRPO 路径不变；
5. Dataset V2 launcher 允许显式选择 `grpo` 或 `cdt_hla_grpo`，其余 resolved config 不变；
6. 现有 paired cluster-bootstrap 增加 StrictClear、tier rates、transition matrix、`P_up/P_down/NetTierGain`。

不实现：

- L0/L1 Pareto advantage；
- 新 critic、constraint optimizer、Lagrangian、额外 reward server；
- 自动 fallback、参数 sweep、Extension、ADAS/FALS hybrid；
- CDTR 正式 launcher，除非主结果后另行授权。

### 8.2 最小测试集合

优先扩展已有 `tests/test_dataset_v2.py` 和 `tests/test_safe_grpo.py`，只增加：

- CDT 分类表：四级、`C=0.5`、`epsilon=1e-6` 边界、非法离散值 mapping error、parse invalid；
- 三个固定 tier-advantage 示例、和为 0、范围与 inversion invariant；
- centered-SDR residual 的范围、tier 内和为 0、连续 gap 保留、singleton/tie，以及 L1/L0 zero-within；
- all-valid all-L3/all-L2 与现有 standard SDR-GRPO 输出一致，all-L1/all-L0 为 0；
- mixed valid/invalid 不参与 CDT comparison、invalid advantage 为 0、`G_v<2` 全组为 0；
- mixed-tier valid subset 与全组 `sum(A_raw)=sum(A_train)=0`（数值容差内），且 `lambda=0.125` 的最小 valid cross-tier raw margin `>=5/12`；
- scale-only normalization 的 zero-std 分支、正序保持，以及非零组 `A_train` std 与 standard GRPO convention 一致；
- reward metrics 到 estimator 的 batch 顺序/shape；
- SafetyMix 恰好 1,000、quota `634/251/115`、per-log `<=5`、无 dev/final overlap、membership/order 确定性复跑一致，最终顺序不保留 mixed-first 分块；
- 现有 cluster-bootstrap 对 StrictClear/tier transition 的小 fixture。

提交前只运行：

- 上述两个已有 focused test 文件；
- changed Python modules 的 compile；
- Dataset V2 launcher `bash -n`；
- `git diff --check`；
- 一次真实 24K CPU replay。

不重复全仓测试、D0 asset rebuild、I0/S0/S1 inference 或旧 smoke。若 focused test 暴露共享基础设施回归，再扩大检查范围。

## 9. `V2-H0`：CPU replay、manifest 与 source freeze

### 9.1 输入

- V2-S0 block 1–4：500 token × 4 rollouts/block；
- V2-S1：6,000 groups / 24,000 rollouts；
- V2-E0：既有 2,000-token dev rollout，仅复算 `StrictClear_E0` 与 headroom，不重新生成；
- Phase-1 6K、dev 2K、final 1K manifest；
- `master_index.csv`；
- production SDR implementation；
- selector rollout/tie-break seed `20260825`；正式训练 generation seed 与其解耦。

### 9.2 输出

正式目录使用 `experiments/dataset_v2_20260825/v2_h0_cdt_hla_seed20260825/`，至少包含：

- `cdt_stability_report.json`；
- `safetymix_1k.txt`；
- `safetymix_report.json`；
- `advantage_geometry_report.json`；
- `group_geometry.csv`；
- 在 `advantage_geometry_report.json` 中记录 `StrictClear_E0` 与 `Headroom_E0=1-StrictClear_E0`；
- input/source hash、membership hash、final order hash、resolved method definition、`COMPLETE` 和 `exit_code`。

### 9.3 技术门控

- S0/S1 coverage、group size、字段和 production SDR recompute 完整；
- parsed-ok rollout 100% 在 canonicalization 后唯一映射到 L0–L3，invalid 单独计数且不进入 CDT comparison，`mapping_error=0`；
- `C=0.5` 进入 L2/L3 的数量为 0；
- SafetyMix 恰好 1,000，quota/cap/overlap、membership 与 independent final order 的确定性全部通过；
- 所有 SDR/CDTR/HLA advantage finite；
- same-L3/L2 identity 分支直接复用 standard estimator；所有 mixed-tier HLA nonzero valid subset 的 scale-only normalization 与 standard GRPO 的 `std/epsilon` convention 一致；
- focused tests、compile、launcher syntax、diff check 通过；
- 数据与 Stage-2 hash 未变化，source 只重新绑定一次。

### 9.4 科学门控

- mixed-tier ratio CV `<=0.20`；
- membership median Jaccard `>=0.50`；
- HLA tier inversion 数为 0；
- HLA 的最小 cross-tier raw margin `>=5/12`，scale 后所有 cross-tier order 保持；
- all-valid same-L3/L2 group 的 `max(|A_train_HLA-A_train_SDR|)=0`（实现数值容差内）；
- invalid 不参与 pair audit、其 HLA advantage 全为 0，`G_v<2` group 全为 0；
- 在 SafetyMix 全部 group 中，至少 10% 满足 `mean(|A_train_HLA-A_train_SDR|)>=0.10`；
- 在 mixed-tier group 中，至少 80% 满足同一 material-change 条件；
- 报告 standard SDR 在所有 valid `L_i>L_j` pair 上的 cross-tier inversion/tie rate，以及 HLA 修正的 pair 数；HLA 对应 inversion/tie 必须为 0；
- 报告 `EffectiveGroupRate=P(max_i|A_i^{train}|>0)`，并将 zero group 拆成 all-L3、all-L2、all-L1、all-L0、all-invalid、`G_v<2`、partial-invalid、SDR tie；
- exact-zero、`A_tier/A_within/A_raw/A_train` 分布和 `StrictClear_E0/Headroom_E0` 完整报告，但 exact-zero 与 headroom 不作为关闭门槛，也不据此修改 `+0.01` 晋级线。

任一技术门控失败：只修已定位问题并用 `retryN` 重跑 H0。任一科学门控失败：关闭 HLA 正式训练，不调 tier、lambda、quota、cap 或 material threshold。

### 9.5 实际结果与终止决定

`V2-H0` 在 source `d30ba3b4e42d6e1bc2059854645cb714db49832f` 上完成，`exit_code=0`，证据目录为 `experiments/dataset_v2_20260825/v2_h0_cdt_hla_seed20260825/`。

- source rebind 完成，Dataset V2 的 10K image/cache、6K/dev/final manifests、Stage-2 model hashes 均未变化；server focused tests 为 `58 passed`，compile、launcher syntax 与 diff check 通过；
- S0 四块 mixed-tier ratio 为 `0.106/0.096/0.094/0.094`，CV `0.058919`；六个 pairwise Jaccard 的中位数为 `0.703704`；
- 6K bank 的 tier rollout counts 为 `L3=23087, L2=163, L1=587, L0=153, invalid=10`，`mapping_error=0`，`C=0.5` 进入 L2/L3 的数量为 0；
- SafetyMix 恰好 1,000 groups，其中 mixed-tier 540；intent 为 `634/251/115`，494 个 logs，per-log max 5，dev/final overlap 均为 0；membership/order SHA256 分别为 `0102c164db07537393b54dc8b4ee84cc23befc12bfa23885bfd2a2a664004808` 与 `acd72e792b48999c9bcc99b13e762173d7bfb1b5c11d10658b28698812c7a77a`；
- HLA cross-tier inversion 为 0，修正 SDR inversion/tie pair 27 个，最小 raw margin `0.666667`；same-L3/L2 identity max diff 与 invalid advantage max abs 均为 0；
- SafetyMix 的 EffectiveGroupRate 为 `0.847`，zero groups 为 152 个 all-L3 SDR tie 和 1 个 all-L2 SDR tie；`StrictClear_E0=0.995`，`Headroom_E0=0.005`；
- 科学门控中，稳定性、Jaccard、tier order、raw margin 均通过；但 SafetyMix 全组 material-change ratio 仅 `0.042 < 0.10`，mixed-tier material-change ratio 仅 `0.077778 < 0.80`，两项失败。

终止决定：`close_hla`。HLA 的层级排序实现正确，但在冻结的真实 rollout geometry 上只对 4.2% 的 SafetyMix groups、7.78% 的 mixed-tier groups 产生达到阈值的训练 advantage 改写，信号覆盖远低于预注册要求。依照本节冻结规则，不调 tier、`lambda`、quota、cap 或 material threshold，不启动 smoke、正式训练、matched seeds 或 final；`SM4-CDTR` 也不在本轮追加执行。

## 10. `V2-HT0-HLA`：唯一新增 smoke

H0 通过后，从 SafetyMix 中确定性选择 40 个 mixed-tier token：

- 10 steps、40 groups、160 train rollout、G=4；
- Stage-2、production SDR、`cdt_hla_grpo`、training seed `20260829`；
- 无 dev/final access；
- 只验收 metrics plumbing、`A_raw/A_train` tensor、scale-only normalization、loss/KL/clip/grad、checkpoint、TensorBoard、显存和资源回收；
- 必须 parse `>=99.5%`、clipping `0`、全部 finite、无 tier inversion/OOM/CUDA/no-space/traceback；
- smoke 不产生方法效果结论。

`SM4-SDR` 不另跑 smoke：production SDR/standard GRPO 已由 `V2-T0-SDR` 验证，新 manifest 的加载与约束由 H0 测试覆盖。

## 11. 正式训练共同协议

| 项目 | 冻结值 |
| --- | --- |
| 初始化 | 每个 run 从同一 Stage-2 独立开始 |
| Manifest | 同一 `safetymix_1k.txt`，同一顺序 |
| Group/steps | `G=4`，250 steps，4 groups/step |
| Train rollout | 4,000 = 1,000×4 |
| Reward/字段 | 相同 reward server、production SDR 计算；完整记录 CDT/PDMS 分项 |
| 唯一变量 | 由同一批轨迹级指标构造 advantage 的规则：`grpo` vs `cdt_hla_grpo` |
| Train decode | temperature/top-p `1.0/1.0` |
| Dev decode | temperature/top-p `0.6/0.95`，每 token 1 response |
| Selector seed | 冻结 rollout/tie-break seed `20260825`，不作为正式训练 seed |
| Main seed | `20260829` |
| Confirmation | `20260830`、`20260831`，仅在主 seed 晋级后执行 |
| Checkpoint | step125 可仅供恢复；科学评估只读取 step250 |
| Dev | 同一 `dev_2000.txt`，每个正式 model 只生成一次 |
| 统计 | 20,000 次 paired log-cluster bootstrap |

`SM4-SDR` 和 `SM4-HLA` 的代码、manifest、config、source hash 必须在 baseline 首次访问 dev 前同时冻结。看到 `SM4-SDR` dev 后不得修改 HLA 定义或配置。

## 12. 正式 run 技术门控

直接沿用 Dataset V2 已有技术标准，保留必要项：

- source clean，Stage-2/data/manifest/cache hash 与 H0 一致；
- active manifest 与 dev/final token、log overlap 为 0；
- 24 GB GPU、端口/进程空闲、磁盘满足既有 launcher 门槛；
- `exit_code=0`、`COMPLETE`、1,000×4 train 和 2,000×1 dev 覆盖完整；
- train/dev parse `>=99.5%`，clipping `0`，全部 reward/advantage/loss/metric finite；
- HLA run 的 tier coverage、`A_raw/A_train` scale、cross-tier inversion audit 和 EffectiveGroupRate 完整；
- HLA run 的 invalid exclusion、same-L3/L2 identity 与 `G_v<2` zero audit 完整；
- 无 OOM/CUDA/no-space/killed/traceback，结束后资源全部回收；
- 原始 rollout、LoRA、resolved config、曲线、paired report 和 result hash 完整。

技术失败只允许修复已定位的执行问题，不改变方法。正式 step250 dev 结果产生后，不重选 checkpoint、tier、lambda、selector 或 decode。

## 13. 主指标、transition 与晋级门控

### 13.1 完整报告

- StrictClear、旧 Safe；
- L3/L2/L1/L0/invalid rate；
- raw PDMS、PDMS scaled；
- Collision、DAC、TTC、Progress、Comfort；
- parse/clipping；
- train mixed-tier ratio、tier composition、基于最终 train advantage 的 material-change ratio、SDR cross-tier inversion/tie 与 HLA correction、EffectiveGroupRate、zero group composition；
- HLA run 每 25 steps 汇总 mixed-tier ratio、all-L3/L2/L1/L0、partial-invalid、EffectiveGroupRate、mean `|A_tier|`、mean `|0.125*A_within|`、valid cross-tier pair 数和 HLA correction 数；只作机制曲线，不设新 gate；
- GPU wall time、reward query、峰值显存和磁盘成本。

### 13.2 Paired tier transition

在相同 2,000 dev token 上生成 baseline→HLA 的 `4×4` L0–L3 transition matrix；invalid 单独列出，不并入四级矩阵。同时计算：

\[
P_{up}=P(L_{HLA}>L_{SDR}),\qquad
P_{down}=P(L_{HLA}<L_{SDR})
\]

\[
NetTierGain=P_{up}-P_{down}
\]

### 13.3 主 seed 晋级线

方向均为 `SM4-HLA - SM4-SDR`，必须同时满足：

- `Delta StrictClear >= +0.01000`；
- `Delta raw PDMS >= -0.00500`；
- `Delta Collision/DAC/TTC >= -0.00500`；
- `Rate(L0)_HLA <= Rate(L0)_SDR`；
- `Rate(L1)_HLA <= Rate(L1)_SDR`；
- StrictClear paired log-cluster 95% CI 满足 `CI_upper > 0`，允许跨 0；
- 全部技术门控通过。

解释规则：

- 全部通过：标记 `PROMOTED_SINGLE_SEED`，立即进入 matched seeds；
- 任一 PDMS/安全分项越界或 L0/L1 上升，或 StrictClear `<=0`：`NEGATIVE_OR_TRADEOFF`，关闭 HLA；
- 其余情况下，StrictClear 为正但未达 `+0.01`，或 `CI_upper<=0`：`INCONCLUSIVE_SINGLE_SEED`，停止，不调参数；CI 跨 0 但上界大于 0 不构成自动关闭条件；
- train geometry 改善不能覆盖 dev 门控失败；
- 不因 HLA 结果补跑 CDTR，除非用户另行授权机制消融。

## 14. Matched seed 与 final

只在主 seed 晋级后运行 seeds `20260830/20260831`：

- 每个 seed 都从 Stage-2 独立训练完整的 `SM4-SDR` 和 `SM4-HLA`；
- 每个 seed 内 manifest、顺序、decode、预算完全匹配；
- 每个 seed 分别执行 20,000 次 paired log-cluster bootstrap，报告 `Delta_s` 与各自 95% CI；
- 3/3 seed 的 StrictClear 差值均为正；
- 三 seed 平均 `Delta StrictClear >= +0.01000`；
- 汇总 95% CI 使用 20,000 次 two-level bootstrap：第一层重采样 seed，第二层在被采样 seed 内重采样 matched log cluster，禁止把 6,000 条 dev 结果当作独立 observation 直接拼接；汇总 CI 下界大于 0；
- 平均 PDMS 与 Collision/DAC/TTC 仍满足主门控。

最终主证据同时展示 `3/3 direction + mean effect + each-seed CI`；two-level CI 只作为明确考虑 seed 与 log 两级不确定性的汇总量。该结论命名为冻结方法后的 `matched exploratory confirmation on V2 dev`，不写成严格 pre-registered confirmation。全部通过才标记 `CONFIRMED_ON_V2_DEV`。之后直接复用原 V2 台账的 final reserve 一次性独立确认规则；否则 final 永不访问。

## 15. 结果记录

### 记录 HLA-001：V2-H0 CPU replay 与 SafetyMix 冻结

- ID / 状态：`V2-H0-CDT-HLA / COMPLETE_CLOSE_HLA`；
- source/model/data hash：source `d30ba3b4e42d6e1bc2059854645cb714db49832f`；Stage-2 两个 shard 分别为 `870666c2...f10f0f`、`4f264c53...98744`；S1 rollout 为 `a4f1f9ae...9f4d7`，完整输入 hash 见 `input_sha256.json`；
- CDT mapping 与 tier counts：`L3=23087, L2=163, L1=587, L0=153, invalid=10`；`mapping_error=0`；
- S0 mixed ratio / CV / Jaccard：`0.106/0.096/0.094/0.094 / 0.058919 / 0.703704`；
- SafetyMix mixed 数、intent/log、overlap、membership/order hash：`540 / 634-251-115 / 494 / 0-0 / 0102c164...004808 / acd72e79...c7a77a`；
- SDR/CDTR/HLA geometry：HLA inversion `0`，SDR inversion/tie rate `0.015067`，HLA correction `27`，minimum raw margin `0.666667`；
- same-L3/L2 identity、invalid exclusion、material/EffectiveGroupRate/zero diagnostics：identity diff `0`，invalid advantage `0`；all/mixed material `0.042/0.077778`，EGR `0.847`，zero groups `152 all-L3 SDR tie + 1 all-L2 SDR tie`；
- `StrictClear_E0 / Headroom_E0`：`0.995 / 0.005`；
- focused tests 与 source rebind：server `58 passed`；source rebind、compile、launcher syntax、diff check 全部通过；
- 门控结论：技术门控全部通过；科学门控中的 all-group 与 mixed-group material-change 两项失败，决策 `close_hla`；
- 唯一下一动作：无；闭环停止并保留证据。

### 记录 HLA-002：V2-HT0-HLA smoke

- ID / 状态：`V2-HT0-HLA / SKIPPED_BY_H0_SCIENTIFIC_GATE`；
- 门控结论：H0 material-change 科学门控失败，按冻结规则不启动 smoke；
- 唯一下一动作：无。

### 记录 HLA-003：V2-SM4-SDR baseline

- ID / 状态：`V2-SM4-SDR / SKIPPED_NO_ACTIVE_HLA_CONTRAST`；
- 门控结论：HLA 已在 H0 关闭，matched baseline 不再具有本轮主 contrast 用途，因此未启动；
- 唯一下一动作：无。

### 记录 HLA-004：V2-SM4-HLA main contrast

- ID / 状态：`V2-SM4-HLA / SKIPPED_BY_H0_SCIENTIFIC_GATE`；
- 科学边界：H0 只证明当前冻结定义的 HLA 信号覆盖不足，不否定安全层级建模这一研究动机；
- 门控结论：未进入正式训练，因而不存在模型效果、paired CI 或 tier transition 结论；
- 唯一下一动作：无。

### 主结果表

| ID | 状态 | PDMS | StrictClear | L0 | L1 | Collision | DAC | TTC | 直接差值 / CI | 决策 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| V2-E0 | COMPLETE | 0.947464 | 0.995000 | — | — | 0.999500 | 0.997500 | 0.997500 | — | Stage-2 anchor |
| V2-SM4-SDR | SKIPPED | — | — | — | — | — | — | — | 未形成 contrast | H0 后关闭 |
| V2-SM4-HLA | SKIPPED | — | — | — | — | — | — | — | 未训练 | H0 科学门控失败 |
| V2-SM4-CDTR | OPTIONAL | — | — | — | — | — | — | — | CDTR−SDR | 首轮不运行 |

## 16. 证据、空间与执行队列

证据和大文件清理直接沿用原 Dataset V2 台账：保留 run.env、source/input/model hash、raw rollout、LoRA、config、TensorBoard/曲线、paired report、代表样本、COMPLETE/exit code；只有这些全部固化且进程回收后，才允许精确删除 full actor state。不得删除旧 V2/SLDR/SDR/RAW 证据。

冻结的条件执行顺序：

1. `V2-H0`：最小实现、focused tests、S0/S1 CPU replay、SafetyMix 与 geometry freeze；
2. `V2-HT0-HLA`：唯一新增 10-step smoke；
3. `V2-SM4-SDR`：SafetyMix matched baseline；
4. `V2-SM4-HLA`：唯一主方法 run；
5. 立即生成 paired cluster report 与 tier transition，写回单-seed 结论；
6. 仅在晋级时运行两个 matched seeds；
7. 仅在三 seed 确认后访问 final；
8. 写入最终结论并停止，不追加 reward/selector/estimator sweep。

实际执行在第 1 步结束：`V2-H0` 已完成并返回 `close_hla`；第 2–7 步均按门控跳过，final reserve 未访问。当前没有待执行实验动作。
