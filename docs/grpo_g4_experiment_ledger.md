# Curious-VLA G=4 GRPO 奖励与场景选择实验台账

> 生效日期：2026-08-24。本文档是用户重新开启 `G=4` GRPO 探索后的唯一实时计划与结果台账。
> [`post_training_execution_loop.md`](post_training_execution_loop.md) 保存旧 `G=2` GRPO 的历史证据，
> [`offline_preference_post_training_execution_loop.md`](offline_preference_post_training_execution_loop.md) 保存离线偏好路线历史；二者均不回写既有结论。
> 本轮只把旧 566-token dev 当作已访问的 exploratory development set。旧 565-token held-out 已失去 unseen 资格，任何新结果都不能表述为最终确认性证据。

## 1. 当前决策快照

- 当前阶段：`R4-SDR`。S0 已完成并触发多项科学失败门控，全部 SLDR 正式训练关闭；现在只运行 Random-1k + SDR + GRPO, `G=4`。
- 当前唯一动作：从同一 Stage-2 checkpoint 独立训练 `R4-SDR`，在预注册 step 125/250 上与 E1 的 `G=2` Random-SDR 结果比较，同时报告 rollout/query 翻倍的成本边界。
- 当前最佳历史候选：`E2 = FALS-1k + SDR + GRPO, G=2`。它只是在单一训练 seed 的旧 dev 上优于其他已训练变体，不能声明稳定超过 Stage-2。
- SLDR 当前状态：S0 科学负向，永久关闭本轮 `R4-SLDR` 与 `AF4-SLDR`；不调 `0.5/0.1/0.6` 系数，不做 dev sweep。
- SDR 当前状态：继续作为主 reward。若 SLDR 未通过，先完成 `G=4 + SDR` 的 Random/FALS/ADAS 单变量对照，再根据失败类型决定是否设计新的非线性映射或 tie-aware 方法。
- ADAS 当前状态：历史发布配置不是当前 `1k scenes / G=2 / 250 steps` 的同协议基线，必须重新建立标准化 ADAS-1k 的 `G=2/G=4` 配对实验。
- `ADAS + FALS` 当前状态：视为一个待定义的新 Hybrid selector，而不是两个可直接叠加的独立模块。最小定义固定为 `ADAS gate → FALS ranking → Top-1,000`。

| 阶段 | 状态 | GPU | 回答的问题 | 下一动作 |
| --- | --- | ---: | --- | --- |
| T0 | 技术通过 | smoke | 目标 RTX 4090 24 GB GPU 是否能保持科学协议运行 `G=4` | 峰值 21,222 MiB，进入 S0 |
| S0 | 科学负向 | 0 | SLDR 在 `G=4` 下是否真的改变有效 advantage，而不只是改变 raw reward 数值 | 多项门控失败；关闭 SLDR |
| R4-SDR | 待执行 | full | Random 场景下 `G=4` 的系统收益与额外查询成本 | 当前唯一动作；与 E1 对照 |
| F4-SDR | 被 R4-SDR 阻塞 | full | FALS 在 `G=4` 下是否仍优于 Random | 与 R4-SDR、E2 对照 |
| A2/A4-SDR | 待建立 ADAS-1k | full ×2 | 同一 ADAS 场景集下 `G=2→4` 的变化 | 先冻结标准化 ADAS manifest |
| R4-SLDR | 按门控跳过 | 0 | SLDR 在 `G=4` 下是否优于同协议 SDR | S0 失败，禁止运行 |
| AF4-SDR | 被 F4/A4 阻塞 | full | ADAS gate 后再用 FALS 排序是否有独立贡献 | 先完成 Hybrid train-only 审计 |
| AF4-SLDR | 按门控跳过 | 0 | 同一 Hybrid selector 下 SLDR 是否优于 SDR | S0 失败，禁止运行 |
| C0 | 未开放 | paired seeds | 胜出差值是否跨训练 seed 稳定 | 只确认一个胜出对照 |

## 2. 名词与比较口径

### 2.1 本台账中的方法名

- `G`：同一 prompt/scene 在一次 group 内生成的 completion 数，不是整步生成的轨迹总数。
- `SDR`：当前代码中的 `pdms_scaled`，训练入口为 `compute_score_group_fast`。它已经包含 focal-style 非线性映射，不是原始线性 PDMS。
- `SLDR`：当前项目实现的 Safety-Lexicographic Dense Reward，训练入口为 `compute_score_sldr`。
- `Random`：冻结的 Random-1k manifest。
- `FALS`：从冻结 D0 train rollout 计算 difficulty × headroom 并选 Top-1,000。
- `ADAS-1k`：先通过 ADAS gate 得到 eligible pool，再以固定 seed 从 pool 中均匀选择 1,000 个 token；它与 Hybrid 使用同一个 eligible pool。
- `ADAS+FALS`：同一 ADAS eligible pool 内按冻结 FALS score 排序并取 Top-1,000；若 eligible pool 少于 1,000，不允许用 Random 补齐。

### 2.2 已有 `G=2` 结果的正确边界

| ID | 场景选择 | Reward / estimator | G | PDMS scaled | PDMS | Safe | 结论 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| E0 | 无 GRPO | Stage-2 | — | 0.65938 | 0.68361 | 0.72438 | 冻结基线 |
| E1 | Random-1k | SDR + GRPO | 2 | 0.64281 | 0.66691 | 0.70671 | 同协议负对照 |
| E2 | FALS-1k | SDR + GRPO | 2 | 0.67230 | 0.69758 | 0.74028 | 当前 dev 最佳候选 |
| E3 | Random-1k | SLDR + GRPO | 2 | 0.62994 | 0.65266 | 0.68905 | SLDR 独立贡献为负 |
| E4 | Random-1k | SLDR + Std-Floor | 2 | 0.64344 | 0.66691 | 0.70848 | 仅部分补救 E3 |
| ADAS-history | 历史 ADAS/约 6k | 发布路线配置 | train G=8 / selector n=32 | — | — | — | 仅作历史参考，不是当前 ADAS-G2 基线 |

因此，当前真正可直接做 `G=2→4` 比较的只有：

1. Random + SDR；
2. FALS + SDR；
3. Random + SLDR，但必须先通过 S0，不能因为矩阵对称就自动重跑；
4. ADAS + SDR 需要先补建同协议 `G=2` 基线。

## 3. SLDR 历史结果与机制复盘

### 3.1 SDR 和 SLDR 实际在做什么

令

\[
f_{0.6}(x)=1-(1-x)^{0.6},
\qquad
q=\frac{5f_{0.6}(P)+5f_{0.6}(T)+2C}{12},
\]

其中 `P/T/C` 分别是 progress、TTC 和 comfort。当前 SDR 大致为：

\[
R_{\mathrm{SDR}}=
\begin{cases}
0,&\text{collision gate 或 drivable gate 为 0},\\
q,&\text{否则}.
\end{cases}
\]

当前 SLDR 定义为：

\[
R_{\mathrm{SLDR}}=
\begin{cases}
0.5+0.5q,&\text{代码判定为 safe},\\
0.1T,&\text{代码判定为 unsafe}.
\end{cases}
\]

这意味着：

- 对全 safe group，SLDR 只是 SDR 的正仿射变换；标准 GRPO 组内归一化后 advantage 完全相同。
- 在 `G=2` 的 safe/unsafe 非平局 pair 中，两种 reward 都把 safe 排在前面；归一化后通常仍是固定的 `±0.707`，raw reward 跨度不会变成更大梯度。
- 在 `G=2` 下，SLDR 真正新增的主要信号来自“两个轨迹都 unsafe、SDR 都为 0，但 TTC 不同”的 pair；此时它只按 TTC 打破平局。
- 在 `G=4` 的 mixed-safety group 中，SLDR 的分段映射可能改变四条轨迹归一化后的相对间距。这是它相对 `G=2` 唯一值得重新审计的主要机制。

实现证据见 [`safety_dense_reward.py`](../EasyR1/verl/utils/reward_score/navsim/safety_dense_reward.py) 与 [`navsim_reward_grouped.py`](../EasyR1/verl/utils/reward_score/navsim/navsim_reward_grouped.py)。

### 3.2 E3 相对 E1 的已观察结果

两者使用同一个 Random-1k、`G=2`、250 steps 和相同评估协议，唯一主要变化是 SDR → SLDR。

| 指标 | E1 SDR | E3 SLDR | E3 − E1 |
| --- | ---: | ---: | ---: |
| PDMS scaled | 0.64281 | 0.62994 | -0.01287 |
| PDMS | 0.66691 | 0.65266 | -0.01425 |
| Safe | 0.70671 | 0.68905 | -0.01766 |
| Collision | 0.95936 | 0.95760 | -0.00176 |
| DAC | 0.74205 | 0.72792 | -0.01413 |
| Progress | 0.91071 | 0.90941 | -0.00130 |
| TTC | 0.94170 | 0.93816 | -0.00354 |
| Comfort | 0.91873 | 0.91873 | 0.00000 |
| Exact-zero group | 46.30% | 44.60% | -1.70 pp |
| `0 < std < 0.05` | 13.60% | 15.40% | +1.80 pp |
| Headroom | 0.17365 | 0.16895 | -0.00470 |

当前能够成立的判断：

1. SLDR 没有把 raw reward 的设计跨度转化为 dev 收益，主要质量与安全指标全部下降。
2. 它只小幅减少 exact-zero group，同时增加低非零方差 group；没有显示“有效 advantage 明显增多”。
3. E4 的 Std-Floor 相对 E3 有部分补救，但仍未超过 E0/E2，说明问题不能简单归因于一个过小的 std 分母。
4. 因为 `G=2` 会抹平绝大多数非平局 pair 的 reward gap，E3 不能彻底否定 SLDR 在 `G=4` mixed group 中的作用；但它足以要求严格的零 GPU 前置门控。

D0 的四 rollout/group 诊断中，SDR exact-zero group 为 18.14%，明显低于 E1 的 46.30%。D0 覆盖全量 train、policy 与场景分布也不同，因此这不是 `G=4` 因果效果；但它提示 `G=4 + SDR` 本身可能已经消除大量平局，SLDR 能额外解决的问题可能更少。

### 3.3 S0：SLDR train-only advantage-geometry 审计

输入只允许使用冻结 D0 的 4,525 个 train scene、每 scene 4 条 rollout。不得读取 dev 或旧 held-out，不重新生成轨迹。E3/E4 原始大体积产物已经删除，不能假装进行逐轨迹复算；S0 使用 D0 是机制代理，不是 E3 效果复现。

S0 必须输出：

1. 必需 NAVSIM 分项字段覆盖、token 数、每组 rollout 数和输入 hash。
2. 使用当前生产代码复算每条轨迹的 SDR、SLDR，不复制一份近似公式。
3. 对每个四轨迹 group 统计 `all-safe / mixed-safety / all-unsafe`；同时检查代码的 `>0` safe 判定是否把部分 collision score 错当成完整 safe。
4. 用训练器的真实 GRPO estimator 计算 `G=4` SDR/SLDR advantage；不得用手写 population-std 近似。
5. 枚举每个四轨迹 group 的全部 6 个 `G=2` pair，验证前述 `±0.707` 与 tie-breaking 机制。
6. 分别报告 exact-zero group、unique reward 数、reward gap、advantage span，以及

   \[
   \Delta_A=\frac{1}{G}\sum_i\left|A_i^{\mathrm{SLDR}}-A_i^{\mathrm{SDR}}\right|.
   \]

7. 单独报告 advantage 改变来自 mixed-safety 还是 all-unsafe/TTC-only group。
8. 对 SLDR 新偏好的 unsafe trajectory，报告 Collision、DAC、Progress、TTC、Comfort 的 paired 差值与 train-only bootstrap CI，防止只改善 TTC 却系统性损害其他行为。

S0 只有同时满足以下条件才通过：

- 数据和复算完整，safe 语义不存在未解释的系统性错标；
- 至少 10% 的 `G=4` group 满足 `ΔA ≥ 0.10`；
- 发生实质变化的 group 中，至少一半是 `G=4` 新出现的 mixed-safety advantage geometry，而不是已经在 E3 中暴露过的 all-unsafe/TTC-only tie-break；
- SLDR 相对 SDR 将 exact-zero group 至少降低 5 个百分点；
- 新偏好轨迹在 Collision、DAC 或 Progress 上不存在 bootstrap CI 完全低于 0 的一致性退化。

门控决策：

- 全部通过：允许一个且仅一个 `R4-SLDR` 正式配置，与 `R4-SDR` 单变量比较。
- 任一科学门控失败：SLDR 关闭，不调 `0.5/0.1/0.6` 系数，不做 dev sweep，不运行 `R4-SLDR` 或 `AF4-SLDR`。
- 数据字段或实现复算失败：只标记技术证据不足；先做最小修复，不把技术失败解释为 SLDR 科学失败。

## 4. 冻结实验协议

### 4.1 所有正式训练共同项

| 项目 | 冻结要求 |
| --- | --- |
| 初始化 | 每个方法从同一 Stage-2 SFT checkpoint 独立开始；不得从 E2 接着训练 |
| 场景预算 | 每个正式 selector 固定 1,000 个 scene，每个 scene 一次 group exposure |
| 训练 | 250 optimizer steps，4 scene groups/step |
| G2 | 2 responses/group，共 2,000 train trajectories |
| G4 | 4 responses/group，共 4,000 train trajectories |
| LoRA / optimizer | 继承 E1/E2 的 rank、target modules、LR、KL、clip 与 seed |
| 生成 | 除 group size 外，temperature、top-p、response length、parser 全部冻结 |
| Reward | SDR 对照固定 `compute_score_group_fast`；SLDR 只在 S0 通过后启用 |
| Dev | 同一已访问 566-token exploratory dev，每 token 1 response；不得称为 unseen final |
| Checkpoint | 预注册读取 step 125 和 step 250；不得看 dev 后追加挑选其他 step |
| 记录 | resolved config、source commit/status、manifest/hash、日志、rollout、诊断、指标、成本和资源回收 |

`G=4` 不是免费改变：在同样 1,000 个 scene 和 250 steps 下，它把 train reward query 从 2,000 增至 4,000。结果必须同时给出：

- step 250：相同 scene exposure / optimizer steps，但 `G=4` 使用两倍 rollout；只能称为系统级效果比较。
- step 125：约 2,000 rollout 的成本快照，但只覆盖约一半 scene、只有一半 updates；只能作成本曲线，不能称为严格等预算因果对照。

不存在一个单独实验能同时固定 scene coverage、optimizer updates 和 rollout 数而只改变 G；报告中必须保留这个边界。

### 4.2 ADAS 与 Hybrid 冻结规则

1. 为研究训练 group size，ADAS eligible pool 必须先固定，并在 A2/A4 中使用同一份 ADAS-1k manifest；不得随 G 重新筛场景。
2. eligible pool 必须完全位于冻结的 4,525-token train split，与 dev/旧 held-out 重叠为 0。历史发布的约 6k filter 不能未经交集、覆盖和来源审计直接作为本轮 manifest。
3. ADAS-1k 定义为：ADAS gate → 固定 seed 均匀抽取 1,000；Hybrid 定义为：同一 ADAS gate → FALS score 排序 → Top-1,000。
4. eligible pool 少于 1,000 时阶段阻塞；不得静默用 Random 或 pool 外 FALS token 补齐。
5. 如果另行研究“随 G 重算 Bernoulli ADAS”，必须新建立系统实验。因为

   \[
   \min_p\left[p^4+(1-p)^4\right]=0.125,
   \]

   沿用 `ε_div=0.1` 时没有场景能通过。该系统实验不能与固定 selector 的 `G=2→4` 因果比较混写。
6. 当前发布实现默认 `n_rollout=8`、`group_size=32`，发布训练/ADAS 路线还出现过 `n=8/32`；必须把它们标为历史配置，不能静默解释成当前训练 G。实现见 [`filter_dynamic.py`](../EasyR1/scripts/adas/filter_dynamic.py)。

### 4.3 硬件 T0 门控

T0 是本轮第一门控。在目标 RTX 4090 24 GB GPU 上先运行不访问 dev 的 10-step throwaway smoke：

- 必须保持 `G=4` 和 4 scene groups/step；不能为了过显存门禁把 G 或 group 数改回去。
- 允许调整只影响显存/吞吐、不改变数学批次的 micro-batch、activation checkpointing 或 rollout memory allocation，但每项必须写入 resolved config。
- 通过条件：完成 optimizer update、无 OOM/NaN、每步严格 16 条训练轨迹、reward 覆盖完整、checkpoint 可保存、进程和 GPU 可回收。
- 若出现 OOM，或必须降低 G、scene group 数等科学协议才能运行，则判定显存不支持 `G=4`，立即终止整条 G4 路线，不执行 S0 或后续阶段。只有能够明确证明与显存容量无关的技术故障，才允许一次新目录最小 retry。

## 5. 分阶段实验矩阵

| 顺序 | ID | Selector | Reward | G | 唯一问题 | 直接对照 | 启动条件 |
| ---: | --- | --- | --- | ---: | --- | --- | --- |
| 0 | T0 | Random smoke | SDR | 4 | RTX 4090 24 GB GPU 是否满足原协议 | 无 dev | 已通过 |
| 1 | S0 | D0 train-only | SDR vs SLDR offline | 2/4 replay | SLDR 是否值得一次 G4 训练 | 无 GPU | 已完成；科学负向 |
| 2 | R4-SDR | Random-1k | SDR + GRPO | 4 | Random 下 G4 系统效果 | E1 | 当前唯一动作；S0 已写回 |
| 3 | F4-SDR | FALS-1k | SDR + GRPO | 4 | FALS 在 G4 下的贡献 | R4-SDR；E2 仅作 G2 参考 | R4-SDR 完成 |
| 4 | A2-SDR | ADAS-1k | SDR + GRPO | 2 | 建立同协议 ADAS-G2 | E1/E2 | ADAS pool/manifest 冻结 |
| 5 | A4-SDR | 同一 ADAS-1k | SDR + GRPO | 4 | ADAS 内部的 G2→4 变化 | A2-SDR | A2 完成、T0 通过 |
| 6 | R4-SLDR | Random-1k | SLDR + GRPO | 4 | SLDR 在 G4 下的独立贡献 | R4-SDR | S0 全通过、R4-SDR 完成 |
| 7 | AF4-SDR | ADAS+FALS-1k | SDR + GRPO | 4 | Hybrid selector 的独立贡献 | A4-SDR、F4-SDR | 二者完成且 Hybrid 审计通过 |
| 8 | AF4-SLDR | 同一 ADAS+FALS-1k | SLDR + GRPO | 4 | 同一 Hybrid 下 SLDR 的贡献 | AF4-SDR | R4-SLDR 与 AF4-SDR 均晋级 |
| 9 | C0 | 胜出 selector/reward | paired rerun | 2/4 | 差值是否跨训练 seed 稳定 | 对应完整 pair | 只允许一个胜出假设 |

`FALS + SLDR` 与 `ADAS + SLDR` 不属于当前最小矩阵。只有 AF4-SLDR 出现无法定位的 selector×reward interaction 时，才允许预注册补充；不能为了填满 factorial table 自动运行。

## 6. 统一效果门控与结果解释

### 6.1 每项正式实验的验收顺序

1. **技术验收**：退出码、COMPLETE、覆盖、每组 G、checkpoint、parse、NaN/OOM、进程与 GPU 回收。
2. **预算验收**：scene groups、train trajectories、reward queries、墙钟、峰值显存与重采样次数。
3. **训练信号**：exact-zero group、低非零 std、headroom、group reward gap、归一化 advantage 分布、all-safe/mixed/all-unsafe 占比。
4. **探索性效果**：566-dev 的 PDMS scaled、PDMS、Safe、Collision、DAC、Progress、TTC、Comfort。
5. **不确定性**：相同 token 的 paired difference 与 20,000 次 bootstrap CI；明确它不覆盖训练 seed 不确定性。
6. **决策写回**：正向 / 负向 / 证据不足 / 技术失败，以及唯一下一动作。

### 6.2 发现阶段晋级线

相对表中直接对照，候选必须同时满足：

- PDMS scaled 点估计至少 `+0.01000`；
- Safe、Collision、DAC 的点估计均不下降；
- parse success 不低于 99.5%，无 reward clipping、NaN 或覆盖缺口；
- 收益解释与实际 rollout/query 成本一起报告。

如果只有 step 250 正向、step 125 不正向，结论写为“在两倍 rollout 预算下获得系统收益”，不能写成“G4 更高效”。如果 train signal 诊断变好但 dev 未过门槛，科学结论仍为未晋级，不能用次要训练统计替代效果指标。

### 6.3 各路线的闭环分支

```text
T0 显存门控
├─ 通过 ──> S0 train-only 审计
└─ 不支持 G4 ──> 整条 G4 路线终止，不执行任何后续阶段

S0 SLDR 审计
├─ 通过 ──> R4-SLDR，与 R4-SDR 单变量比较
└─ 不通过 ──> 关闭全部 SLDR 正式训练，进入第 7 节的 SDR 诊断分支

R4/F4/A2/A4-SDR
├─ G4 仅在 step250 正向 ──> 记录为额外采样预算收益
├─ G4 在 step125 与 step250 均正向 ──> 进入 paired second-seed 候选
└─ G4 不正向 ──> 对该 selector 关闭 G4 扩展

AF4-SDR
├─ 优于 A4-SDR 与 F4-SDR ──> Hybrid selector 晋级
└─ 未同时优于 ──> Hybrid 关闭，不运行 AF4-SLDR

R4-SLDR / AF4-SLDR
├─ 过统一晋级线 ──> 只保留一个 reward 假设进入 C0
└─ 未过 ──> SLDR 最终关闭，不调 coefficient 追分

C0 paired second seed
├─ 同方向且满足安全约束 ──> 形成“跨两个训练 seed 的 exploratory dev 证据”
└─ 不稳定 ──> 结论降级为单 seed 偶然候选
```

## 7. SLDR 失败后的 SDR 优化分支

SDR 本身已经是 nonlinear focal reward。SLDR 失败后，不默认再叠加 `square/exp/sigmoid`；先按 `G=4` SDR train diagnosis 分类：

| 诊断 | 原因判断 | 下一种方法 | 为什么 |
| --- | --- | --- | --- |
| Exact-zero group 仍高 | 相同 SDR 值没有组内排序 | 优先 FALS/ADAS/G4；必要时设计 SDR-primary lexicographic tie-break 或 rank advantage | 严格单调变换不能打破相等输入 |
| 非零组很多，但 G4 advantage 相对间距过于集中 | reward geometry 问题 | train-only 拟合一个有界、严格单调、分位数校准的 SDR transform | G4 下单调变换能改变四轨迹相对间距 |
| Mixed-safety group 的更新损害安全分项 | 多目标权重/约束问题 | 保留 SDR 主排序，单独设计 safety-constrained 或 component-wise advantage | 单纯扩大方差不能修复目标冲突 |
| 训练信号健康但 dev 仍无收益 | 不是 reward span 的主要问题 | 停止 reward 变换，检查 selector、policy update 或数据覆盖 | 避免用 reward sweep 追逐 dev 噪声 |

任何新 reward/advantage 候选必须先通过 train-only 离线门控：

1. 不反转任何原本不相等的 SDR pair，除非新假设明确预注册了安全约束并提供原因；
2. 若目标是解 tie，exact-zero group 至少下降 5 个百分点；
3. 若目标是改变 G4 geometry，至少 10% group 达到 `ΔA ≥ 0.10`；
4. 不读取 dev 选择 transform、温度、阈值或辅助权重；
5. 多个离线候选只能按预注册 train-only score 选一个进入正式训练，不做 dev coefficient sweep。

优先级固定为：

1. 先验证 `FALS + SDR + G4`，因为 FALS 是现有唯一正向 selector 证据；
2. 再判断 ADAS-1k 与 Hybrid 是否改善预算分配；
3. 只有上述方法仍留下明确 reward-geometry 问题，才实现新的 SDR transform 或 tie-aware advantage。

## 8. 结果总表

### 8.1 已完成历史结果

| ID | 配置 | G | Train trajectories | PDMS scaled | Safe | Exact-zero group | 状态 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| E1 | Random + SDR + GRPO | 2 | 2,000 | 0.64281 | 0.70671 | 46.30% | 科学负向 |
| E2 | FALS + SDR + GRPO | 2 | 2,000 | 0.67230 | 0.74028 | 38.80% | 单 seed dev 候选 |
| E3 | Random + SLDR + GRPO | 2 | 2,000 | 0.62994 | 0.68905 | 44.60% | 科学负向 |
| E4 | Random + SLDR + Std-Floor | 2 | 2,000 | 0.64344 | 0.70848 | 43.70% | 部分补救、未晋级 |

### 8.2 新路线待写回

| ID | 配置 | 状态 | Step125 PDMS scaled / Safe | Step250 PDMS scaled / Safe | Train signal | 成本 | 决策 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T0 | Random + SDR G4 smoke | 技术通过 | 不访问 dev | 不访问 dev | 40 groups × 4；parse 100%；zero group 12.5% | 160 queries；918 秒；峰值 21,222 MiB | 24 GB 支持完整 G4 协议 |
| S0 | D0 SDR/SLDR geometry audit | 科学负向 | 不适用 | 不适用 | `ΔA≥0.10` 2.59%；zero 仅降 0.575 pp；30 条 safe 错标；DAC CI 全负 | 0 GPU；6.27 秒 | 关闭全部 SLDR 正式训练 |
| R4-SDR | Random + SDR + GRPO | 待执行 | 待填 | 待填 | 待填 | 待填 | 当前唯一动作 |
| F4-SDR | FALS + SDR + GRPO | 被阻塞 | 待填 | 待填 | 待填 | 待填 | 待填 |
| A2-SDR | ADAS + SDR + GRPO | 待建 manifest | 不适用 | 待填 | 待填 | 待填 | 待填 |
| A4-SDR | ADAS + SDR + GRPO | 被阻塞 | 待填 | 待填 | 待填 | 待填 | 待填 |
| R4-SLDR | Random + SLDR + GRPO | 按门控跳过 | 不适用 | 不适用 | S0 科学门控失败 | 0 | 禁止运行 |
| AF4-SDR | ADAS+FALS + SDR + GRPO | 被阻塞 | 待填 | 待填 | 待填 | 待填 | 待填 |
| AF4-SLDR | ADAS+FALS + SLDR + GRPO | 按门控跳过 | 不适用 | 不适用 | S0 科学门控失败 | 0 | 禁止运行 |

## 9. 证据与数据边界

1. 旧 566-token dev 已参与多轮方法判断，本轮只能作为 exploratory development evidence。
2. 旧 565-token held-out 已生成 520/565 条部分 rollout，永久禁止补跑或重新称为 unseen。
3. 在建立新的版本化 final set 前，任何方法最多得到“train-only 机制证据 + 已访问 dev 效果 + 两个训练 seed 稳定性”。
4. E1/E3/E4 的原始大体积训练产物已删除；历史数字由旧台账保存。不能虚构逐轨迹复分析结果。
5. D0、E2 和 manifests 是当前保留证据；S0 开始前必须重新核对远端实际存在、字段覆盖和 hash。
6. 新实验使用独立命名空间，不覆盖旧 E/R/P/M 目录、访问锁、checkpoint 或日志。

## 10. 新记录模板

每个阶段结束后追加一条记录，并同步更新第 1 节与第 8.2 节：

```text
### 记录 G4-NNN：<阶段与事件>

- 状态：计划中 / 运行中 / 技术通过 / 科学正向 / 科学负向 / 证据不足 / 按门控跳过
- 假设与直接对照：<只回答一个问题>
- 预注册门控：<启动条件、晋级线、停止条件>
- 代码与配置：<commit、source status、resolved config、seed、G>
- 数据：<manifest、hash、scene/group/trajectory 覆盖、split 重叠>
- 技术结果：<退出码、checkpoint、OOM/NaN/parse、资源回收>
- 训练信号：<zero group、headroom、advantage geometry、group composition>
- 探索性效果：<step125/250 指标、直接差值、paired bootstrap>
- 成本：<reward queries、墙钟、峰值显存、磁盘>
- 分析边界：<能够说明什么、不能说明什么>
- 决策：<推进 / 回退 / 跳过 / 最小技术重试 / 关闭路线>
- 下一动作：<唯一动作>
```

### 记录 G4-001：T0 RTX 4090 24 GB 显存门控完成

- 状态：技术通过。
- 假设与直接对照：只回答 RTX 4090 24 GB 是否能在不缩减科学协议的条件下运行 `G=4`；不访问 dev，不判断模型效果。
- 预注册门控：10 optimizer steps、4 scene groups/step、4 trajectories/group；OOM 或必须降低 G/group 数则整条路线终止。
- 代码与配置：source `b8d16d82fb04ce7415cde4bc4efd98ff09241cac`，source status 为空；Stage-2、SDR、标准 GRPO、seed `20260812`、micro-batch 1、gradient checkpointing、`skip_final_validation=true`、`save_model_only=true`。
- 数据：冻结 Random-1k manifest SHA-256 `3ae99bb940fad6fab3b488bc4ea7d01e8755a3677161f0c29dffb5e476721fa8`；1,000 个唯一 train token，与 dev/旧 held-out 重叠均为 0；实际覆盖 40 个唯一 group、160 条 rollout，每组严格 4 条。
- 技术结果：`COMPLETE`、`exit_code=0`、10/10 step；step-10 actor checkpoint 可保存且完整，约 7.7 GB；全部训练数值 finite，无 OOM、NaN、traceback、CUDA error、no-space 或 killed；GPU、Ray、Gunicorn/8901 和训练进程全部回收。
- 训练信号：parse success 与必需 reward 字段覆盖均为 100%，clipping 为 0；SDR exact-zero group `12.5%`、低非零 std group `5.0%`、平均 headroom `0.32864`。这些 40-group smoke 统计只作技术旁证，不作正式方法结论。
- 探索性效果：不适用；未访问 dev。
- 成本：160 次 train reward query；10 step 合计 `776.67` 秒、平均 `77.67` 秒/step；1 秒采样共 837 点，墙钟覆盖 918 秒，峰值显存 `21,222 MiB`、最低剩余 `2,860 MiB`、GPU utilization 峰值 100%；run 目录约 7.7 GB，结束后磁盘剩余约 23 GB。
- 产物清理：S0 证据闭环且 R4 正式启动前，重新确认 GPU、训练进程和 8901 均为空后，只删除 throwaway checkpoint 的 `/root/autodl-tmp/curious-vla-workspace/experiments/safe_grpo/t0_g4_sdr_smoke10_seed20260812/checkpoints/global_step_10/actor/model_world_size_1_rank_0.pt`（`8,144,550,392` bytes，不可恢复），磁盘空余回升至约 31 GB；LoRA、config、tracker、训练日志、rollout、显存采样、报告与 `COMPLETE` 全部保留。
- 分析边界：证明当前 24 GB GPU 支持冻结的 G4 数学批次和 checkpoint 保存；不证明 250-step 科学收益或长程稳定性。
- 决策：T0 通过，允许进入 S0；不调整 G、scene groups、生成协议或 micro-batch。
- 下一动作：只执行 D0 train-only S0 advantage-geometry 审计。

### 记录 G4-002：S0 SLDR train-only advantage-geometry 审计完成

- 状态：科学负向。
- 假设与直接对照：只使用冻结 D0 的 `4,525 × 4` train rollout，对生产 SDR 与 SLDR 的 `G=4` reward/advantage geometry 做离线比较；不访问 dev 或旧 held-out。
- 预注册门控：数据和 safe 语义完整；`ΔA ≥ 0.10` group 至少 10%；material group 中 strict mixed-safety 至少 50%；exact-zero 至少下降 5 pp；SLDR 新偏好的 unsafe trajectory 在 Collision、DAC、Progress 上不得出现 20,000 次 group-bootstrap CI 全负。
- 代码与配置：source `e4dfd64ee64be55a43708868b6dd082339c223a7`，source status 为空；生产 `compute_sldr`、真实 `compute_grpo_outcome_advantage`、float32 与 `eps=1e-6`；seed `20260812`，CPU-only。最终证据目录为 `s0_sldr_geometry_seed20260812_retry2`；此前输出只因补齐 G2 极小 gap 的 production-eps/float32 解释而被替代，科学数值和门控结论未改变。
- 数据：D0 SHA-256 `2ededee1d08d754c251a1f1777d2df4e44e52f4a859e884afeed95521e6ef9d6`，train manifest SHA-256 `4a19947abd86d4265e055a6408fc8a6d579fcc083cb5bc4c207159d5c60d8168`；4,525 token、18,100 rollout、每组严格 4 条，dev/held-out 访问为 false。18,088 条 parsed row 必需分项字段完整；12 条 parse-failure 日志缺少 5 个 component 字段，只按生产 `_zero_result` 的确定性零 score 恢复 60 个值，正常 parsed row 未填充。
- 技术结果：`COMPLETE`、`exit_code=0`；远端 35 tests passed，报告、4,525 行 group geometry 与 unsafe-preference group 差值文件完整；无 GPU compute process。
- 训练信号：严格 safety composition 为 all-safe/mixed/all-unsafe `1,401 / 2,833 / 291`。SDR exact-zero 为 `821/4,525 = 18.1436%`，SLDR 为 `795/4,525 = 17.5691%`，只下降 `0.5746 pp`；`ΔA` 均值 `0.01233`，仅 `117/4,525 = 2.5856%` group 达到 `0.10`，其中 strict mixed-safety `91/117 = 77.78%`。material group 的 mixed 占比通过，但信号覆盖与解 tie 幅度均未达门槛。
- G2 机制复核：枚举全部 `4,525 × 6 = 27,150` pair；SDR/SLDR exact tie 分别为 `10,051 / 9,789`，全部产生零 advantage；非 tie 中 `98.44% / 96.98%` 在 `eps` 误差内接近理想 `±0.707`。SLDR 只新增 262 个 tie-break pair，未反转任何原本不相等的 SDR pair。
- 安全语义：原日志 `safe` 与生产 `>0` 判定完全一致，但 NAVSIM 将 `no_at_fault_collisions=0.5` 明确定义为与非 agent 物体的 at-fault collision。D0 有 189 条该分值，其中 30 条、15 个 token 因 DAC 也为正而被生产规则系统性标为 safe；因此 safe 语义门控失败。
- unsafe 新偏好：262 个 pair、139 个独立 group；winner-minus-loser 的 group-bootstrap 95% CI 为 Collision `[+0.2050,+0.3435]`、DAC `[-0.1559,-0.0372]`、Progress `[+0.0058,+0.0736]`、TTC `[+1,+1]`、Comfort `[+0.0144,+0.0791]`。SLDR 的 TTC tie-break 伴随一致性 DAC 退化，DAC 门控失败。
- 探索性效果：不适用；未访问 dev。
- 成本：0 GPU、0 新 reward query、20,000 次 train-only group bootstrap；最终分析墙钟 `6.27` 秒，产物不足 1 MB。
- 分析边界：能够否定当前 SLDR 映射进入 G4 正式训练的资格；不能把 D0 机制代理解释为新的模型效果实验。
- 决策：S0 未通过 safe 语义、material group 覆盖、exact-zero 降幅和 DAC bootstrap 四项门控。关闭本轮全部 SLDR 正式训练，不运行 `R4-SLDR` 或 `AF4-SLDR`，不调系数追逐 dev。
- 下一动作：只执行 `R4-SDR`，比较 Random selector 下 `G=4` 的系统效果与两倍 rollout/query 成本。

## 11. 当前下一动作

只执行 `R4-SDR`：先在正式启动前确认 RTX 4090 24 GB 的空闲显存、无残留训练进程、source clean、Random-1k manifest/checkpoint/hash 完整和目标目录不存在；随后从 Stage-2 独立运行 250-step `Random + SDR + GRPO, G=4`，保存预注册 step 125/250 并由 Luna 只读监控。`R4-SDR` 完成前不启动 F4、ADAS、Hybrid 或任何 SLDR 实验。
