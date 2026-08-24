# Curious-VLA G=4 GRPO 奖励与场景选择实验台账

> 生效日期：2026-08-24。本文档是用户重新开启 `G=4` GRPO 探索后的唯一实时计划与结果台账。
> [`post_training_execution_loop.md`](post_training_execution_loop.md) 保存旧 `G=2` GRPO 的历史证据，
> [`offline_preference_post_training_execution_loop.md`](offline_preference_post_training_execution_loop.md) 保存离线偏好路线历史；二者均不回写既有结论。
> 本轮只把旧 566-token dev 当作已访问的 exploratory development set。旧 565-token held-out 已失去 unseen 资格，任何新结果都不能表述为最终确认性证据。

## 1. 当前决策快照

- 当前阶段：`R4-RAW` 已完整闭环，回答 `Random-1k + raw-PDMS + GRPO, G=4` 的 SDR 消融问题。
- 当前唯一动作：保留并固化 R4-RAW/R4-SDR 对照证据；按用户当前范围锁定，不根据已完成的零 GPU P0 审计启动 A4-SDR 或其他训练。
- 当前最佳历史候选：`E2 = FALS-1k + SDR + GRPO, G=2`。它只是在单一训练 seed 的旧 dev 上优于其他已训练变体，不能声明稳定超过 Stage-2。
- SLDR 当前状态：S0 科学负向，永久关闭本轮 `R4-SLDR` 与 `AF4-SLDR`；不调 `0.5/0.1/0.6` 系数，不做 dev sweep。
- SDR 当前状态：技术闭环，但科学证据不足并存在 trade-off。step250 `Δ_SDR(PDMS scaled)=+0.00354 < +0.01000`，且 `Δ_SDR(DAC)=-0.00353`；不能声称 SDR 相对 raw-PDMS 已建立独立正向贡献，也不能声称 raw-PDMS 全面更优。
- ADAS 当前状态：P0 train-only CPU 审计已完成并归档，但不属于当前训练范围。除非用户再次明确启动，不据此开始 ADAS 训练。
- 证据保存状态：R4-RAW 已固化 250-step policy/reward/optimization/output/system 曲线、raw response、代表样本与资源证据；R4-SDR 已用历史原始日志回溯同口径曲线，但其旧 rollout 不含 raw response。

| 阶段 | 状态 | GPU | 回答的问题 | 下一动作 |
| --- | --- | ---: | --- | --- |
| T0 | 技术通过 | smoke | 目标 RTX 4090 24 GB GPU 是否能保持科学协议运行 `G=4` | 峰值 21,222 MiB，进入 S0 |
| S0 | 科学负向 | 0 | SLDR 在 `G=4` 下是否真的改变有效 advantage，而不只是改变 raw reward 数值 | 多项门控失败；关闭 SLDR |
| R4-SDR | 系统级正向、未超 E0 | full | Random 场景下 `G=4` 的 SDR 基线 | 作为 R4-RAW 直接对照 |
| R4-RAW | 技术通过、科学证据不足/trade-off | full | `pdms_scaled` 相对 raw `pdms` 是否有独立贡献 | 已闭环；当前停在本消融结论 |
| F4-SDR | 科学未晋级 | full | FALS 在 `G=4` 下是否仍优于 Random | 相对 R4 仅 +0.00576；关闭 FALS-G4 扩展 |
| A0 | 定义门控失败 | 0 | 发布 ADAS gate 在冻结 train 内是否形成有选择性的 eligible pool | train 排除 0；未写 manifest |
| P0-ADAS | CPU 审计已完成、当前冻结 | 0 | G4 下哪些 ADAS 参数能形成有选择性且有训练信号的 1k pool | 不据此启动训练 |
| A4-SDR | 未启动 | full | 固定 G4/SDR 后 ADAS 相对 Random 的独立贡献 | 等待用户后续明确授权 |
| R4-SLDR | 按门控跳过 | 0 | SLDR 在 `G=4` 下是否优于同协议 SDR | S0 失败，禁止运行 |
| C0 | 用户终止并已清理 | canceled | Random-SDR 的 G4−G2 差值是否跨训练 seed 同方向 | G2 在 step80 停止，无科学结论 |

## 2. 名词与比较口径

### 2.1 本台账中的方法名

- `G`：同一 prompt/scene 在一次 group 内生成的 completion 数，不是整步生成的轨迹总数。
- `SDR`：当前代码中的 `pdms_scaled`，训练入口为 `compute_score_group_fast`。它已经包含 focal-style 非线性映射，不是原始线性 PDMS。
- `raw-PDMS`：NAVSIM 返回的未缩放 `pdms`，训练入口固定为 `compute_score_group_raw_pdms`；R4-RAW 仍持久化 `pdms_scaled` 和各安全分项用于同口径评测。
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

1. P0-ADAS 必须先用冻结 D0 train-only G4 rollout 固定 eligible pool、唯一参数与 ADAS-1k manifest；A4 不得根据 dev 重新筛场景。
2. eligible pool 必须完全位于冻结的 4,525-token train split，与 dev/旧 held-out 重叠为 0。历史发布的约 6k filter 不能未经交集、覆盖和来源审计直接作为本轮 manifest。
3. ADAS-1k 定义为：ADAS gate → 固定 seed 均匀抽取 1,000；Hybrid 定义为：同一 ADAS gate → FALS score 排序 → Top-1,000。
4. eligible pool 少于 1,000 时阶段阻塞；不得静默用 Random 或 pool 外 FALS token 补齐。
5. 历史严格 `ε_div=0.1` 在 G4 下不可用，因为

   \[
   \min_p\left[p^4+(1-p)^4\right]=0.125,
   \]

   沿用该阈值时没有场景能通过。P0 必须预注册只读 train 的参数候选和唯一选择规则，`n_rollout/group_size` 均固定为 4；不允许用 dev 选阈值。
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
| 2 | R4-SDR | Random-1k | SDR + GRPO | 4 | 建立 G4 SDR 基线 | E1 | 已完成；作为 reward 消融对照 |
| 3 | F4-SDR | FALS-1k | SDR + GRPO | 4 | FALS 在 G4 下的贡献 | R4-SDR；E2 仅作 G2 参考 | 已完成；未晋级 |
| 4 | A0 | 发布 ADAS file | 定义审计 | — | 发布参数是否能用于 G4 | 无 GPU | 已完成；定义门控失败 |
| 5 | C0 | Random-1k + SDR | paired rerun | 2/4 | G4−G2 是否跨 seed 稳定 | R4-SDR / E1 | 用户在 C0-G2 step80 终止并清理 |
| 6 | R4-RAW | Random-1k | raw-PDMS + GRPO | 4 | SDR 相对 raw-PDMS 的独立作用 | R4-SDR | 已完成；科学证据不足/trade-off |
| 7 | P0-ADAS | D0 train-only | 参数审计 | 4 replay | G4 下冻结唯一 ADAS 参数/manifest | Random-1k | CPU 审计已完成；当前冻结 |
| 8 | A4-SDR | ADAS-1k | SDR + GRPO | 4 | ADAS 相对 Random 的独立作用 | R4-SDR | 当前冻结；等待用户后续明确授权 |

`FALS/Hybrid/SLDR` 均不属于用户调整后的最小矩阵；不得为了填满 factorial table 自动恢复。

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

R4-RAW 的 reward 消融单独定义 `Δ_SDR = R4-SDR − R4-RAW`，因为正式运行的候选文件方向是 raw-minus-SDR：

- 若 step250 `Δ_SDR(PDMS scaled) ≥ +0.01000`，且 `Δ_SDR(Safe/Collision/DAC) ≥ 0`，则 SDR 记为单 seed 已访问 dev 上的正向 reward 证据；
- 若 `Δ_SDR(PDMS scaled) ≤ 0`，则当前 SDR 映射没有建立正向贡献；
- 若介于二者之间或安全分项退化，则记为证据不足/有 trade-off，不用 train reward 曲线替代固定 dev 结论；
- step125、step250 都必须报告同 token paired bootstrap，但 CI 只覆盖场景不确定性。

如果只有 step 250 正向、step 125 不正向，结论写为“在两倍 rollout 预算下获得系统收益”，不能写成“G4 更高效”。如果 train signal 诊断变好但 dev 未过门槛，科学结论仍为未晋级，不能用次要训练统计替代效果指标。

### 6.3 各路线的闭环分支

```text
T0 显存门控
├─ 通过 ──> S0 train-only 审计
└─ 不支持 G4 ──> 整条 G4 路线终止，不执行任何后续阶段

S0 SLDR 审计
├─ 通过 ──> R4-SLDR，与 R4-SDR 单变量比较
└─ 不通过 ──> 关闭全部 SLDR 正式训练，进入第 7 节的 SDR 诊断分支

R4-RAW reward 消融
├─ SDR 达到预注册正向线 ──> 记录 SDR 的单 seed 正向作用，再进入 P0-ADAS
├─ raw-PDMS 不差于 SDR ──> 记录 SDR 未建立贡献，仍完成用户指定的 P0 定义审计
└─ 技术失败 ──> 只允许修已定位的实现问题，不作 reward 结论

P0-ADAS train-only 参数审计
├─ 形成至少 1,000 个且真正有选择性的 G4 pool ──> 冻结唯一参数和 ADAS-1k
└─ 无合法参数 ──> ADAS 路线关闭，不用 dev 调阈值

A4-SDR
├─ 相对 R4-SDR 过统一晋级线 ──> ADAS 形成单 seed dev 候选
└─ 未过 ──> ADAS 科学未晋级，停止 selector sweep
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

1. 先以 R4-RAW 对 R4-SDR 做同 G4 的 reward 单变量消融；
2. 再用 P0-ADAS 在 train-only 上冻结适合 G4 的唯一参数/manifest，并以 A4-SDR 对 R4-SDR 做 selector 单变量比较；
3. 不恢复 FALS/Hybrid/SLDR，也不在看到 dev 后新增 reward transform。

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
| R4-SDR | Random + SDR + GRPO | 技术通过、系统级正向 | `0.65750 / 0.72085` | `0.65889 / 0.72438` | zero 22.1%；parse 99.9%；headroom 0.28310 | 4,000 train queries；峰值 21,294 MiB | 相对 E1 晋级但未超 E0；进入 F4 |
| R4-RAW | Random + raw-PDMS + GRPO | 技术通过、科学证据不足/trade-off | `0.64849 / 0.71201` | `0.65535 / 0.71731` | raw zero 22.2%；low 10.9%；parse 99.9%；headroom 0.29054 | 4,000 train + 1,132 dev queries；15,352 秒；峰值 21,266 MiB | `Δ_SDR=+0.00354` 且 DAC `-0.00353`，正向门控未过 |
| F4-SDR | FALS + SDR + GRPO | 技术通过、科学未晋级 | `0.65653 / 0.72085` | `0.66465 / 0.72968` | zero 7.5%；parse 99.95%；headroom 0.49786 | 4,000 train queries；峰值 21,266 MiB | 相对 R4 +0.00576，未达线 |
| A0 | 发布 ADAS pool 定义审计 | 定义门控失败 | 不适用 | 不适用 | 5,656/5,656 全覆盖；train 排除 0；G4 Bernoulli 最小值 0.125 | 0 GPU | 冻结 ADAS 与 Hybrid 路线，未写 manifest |
| A2-SDR | 旧 ADAS 定义 + SDR + GRPO | 按旧定义门控跳过 | 不适用 | 不适用 | 发布 selector 未定义 | 0 | 不恢复旧参数 |
| P0-ADAS | G4 train-only 参数审计 | 技术通过、当前冻结 | 不适用 | 不适用 | `ε_div=0.20`；eligible 1,022；ADAS-1k zero-std 0 | 0 GPU | 只归档参数/manifest，不启动 A4 |
| A4-SDR | 新 ADAS-1k + SDR + GRPO | 未启动、当前冻结 | 待填 | 待填 | 待填 | 0 | 等待用户后续明确授权 |
| R4-SLDR | Random + SLDR + GRPO | 按门控跳过 | 不适用 | 不适用 | S0 科学门控失败 | 0 | 禁止运行 |
| AF4-SDR | ADAS+FALS + SDR + GRPO | 按定义门控跳过 | 不适用 | 不适用 | 全通过 ADAS gate 后退化为 FALS | 0 | F4 已未晋级，禁止重复运行 |
| AF4-SLDR | ADAS+FALS + SLDR + GRPO | 按门控跳过 | 不适用 | 不适用 | S0 科学门控失败 | 0 | 禁止运行 |
| C0 | Random + SDR matched G2/G4，seed 20260813 | 用户终止、无科学结果 | 不适用 | 不适用 | G2 在 step80 停止 | 640 train queries | 中间产物已全部清理 |

### 8.3 面向后续追问的正式训练证据保存契约

| 证据组 | 每轮必须保存 | 能回答的问题 |
| --- | --- | --- |
| 配置与边界 | source commit/status、启动命令、`run.env`、resolved config、seed、manifest/model hash、split overlap | 从哪里开始、唯一变量是什么、是否混入旧状态或数据泄漏 |
| Reward / group | 每条 rollout 的 raw response、parsed poses、`training_reward`、raw `pdms`、`pdms_scaled`、Safe/Collision/DAC/Progress/TTC/Comfort；group mean/std/min/max/gap/headroom、exact-zero/low-std | 模型实际收到什么信号、SDR 是否改变排序/方差、是否出现 verifier/parser 问题 |
| Policy 曲线 | 每 step `pg_loss`、`entropy_loss`、KL loss、PPO KL、high/low clip fraction | 策略是否更新过快、坍缩或大面积被裁剪；`pg_loss` 只能称 policy loss，不虚构不存在的 critic loss |
| Optimization 曲线 | grad norm、learning rate、advantage mean/min/max、NaN/Inf 扫描 | 数值是否稳定，raw-PDMS 是否造成异常 advantage/gradient |
| Output 与样本 | response length mean/max、clipping、parse success；高/低 reward、最长、parse failure、最大 group gap 的代表性样本 | 曲线变化是否对应生成行为，是否出现长度偏置、模板化或 parse hacking |
| System 曲线 | step/generation/reward/reference/update 时间、throughput、逐秒 GPU used/free/utilization、峰值显存、墙钟、query 数 | 主要瓶颈、G4 成本和 24 GB 资源边界 |
| Checkpoint / evaluation | step125/250 LoRA/hash、固定 566-dev rollout/metrics、paired bootstrap、进程回收记录 | checkpoint 选择、固定协议效果、不确定性与可复核性 |

原始 `checkpoints/experiment_log.jsonl` 与 `gpu_memory.csv` 是事实源；正式 run 还必须生成 `training_evidence/training_history.csv`、`training_curves.svg`、`training_curve_summary.json`、`representative_train_samples.jsonl` 和 `training_evidence_manifest.json`。图只用于展示，结论以原始 JSONL/CSV 和固定评测为准。R4-SDR 的旧 rollout 没有 raw response 字段，只能回溯 poses/分项与曲线；R4-RAW 起新增 raw response，不能伪称历史对照也保存过该字段。

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

### 记录 G4-003：R4-SDR Random-G4 正式实验完成

- 状态：技术通过；相对 E1 为系统级正向候选，但未超过 Stage-2 E0。
- 假设与直接对照：只改变 Random-SDR 的 group size `G=2→4`；step 250 与 E1 比较固定 1,000 scene/250 update 下两倍 rollout 的系统效果，step 125 是约 2,000 rollout 的成本快照但只覆盖一半 scene/update。
- 预注册门控：相对 E1，PDMS scaled 至少 `+0.01000`，Safe/Collision/DAC 均不下降，parse 至少 99.5%，无 clipping/NaN/覆盖缺口。step125 与 step250 均正向才可列为 second-seed 候选；这不替代后续 F4 单变量比较。
- 代码与配置：source `43e479cb7203c2cf1c8d0cb41a474952cf8b6966`，source status 为空；Stage-2 权重两个 shard SHA-256 为 `870666c2...b10f0f`、`4f264c53...859744`；Random-1k manifest SHA-256 `3ae99bb9...21fa8`；seed `20260812`、SDR、标准 GRPO、`G=4`、4 groups/step、250 steps、LoRA rank 8、step125/250 双 checkpoint、训练生成 temperature/top-p `1.0/1.0`，dev `0.6/0.95`。
- 数据与技术结果：`COMPLETE`、`exit_code=0`；1,000 个唯一 train group、4,000 train rollout，每组严格 4 条；final step250 dev 与独立 step125 dev 各 566 条。train parse `99.9%`（4 条 parse-failure），两套 dev parse 100%，clipping 0，全部指标 finite；无 OOM、NaN、traceback、CUDA、no-space 或 killed。step125/250 actor checkpoint 各约 7.7 GB，LoRA SHA-256 分别为 `2b6c912f...a586b`、`6f437dee...a487`；GPU、Ray、Gunicorn/8901 与训练进程全部回收。
- 训练信号：SDR exact-zero group `22.1%`、`0<std<0.05` group `9.7%`、平均 headroom `0.28310`；train PDMS scaled `0.61773`、Safe `0.66075`。
- 探索性效果：step125 PDMS scaled/PDMS/Safe 为 `0.65750 / 0.68197 / 0.72085`，step250 为 `0.65889 / 0.68325 / 0.72438`。相对 E1，step125 的 PDMS scaled/Safe/Collision/DAC 差值为 `+0.01469 / +0.01414 / +0.00000 / +0.01590`，step250 为 `+0.01608 / +0.01767 / +0.00442 / +0.01060`；两时点均通过预注册 E1 点估计门控。
- 不确定性与基线边界：E1 原始逐 token rollout 已按旧台账删除，无法虚构 paired CI。使用仍保留的同 566-token Stage-2 E0 做 20,000 次 paired bootstrap：step125/250 的 PDMS scaled 差值分别为 `-0.00188`（95% CI `[-0.02380,+0.02017]`）与 `-0.00049`（`[-0.02453,+0.02342]`）；step250−step125 为 `+0.00139`（`[-0.02310,+0.02608]`）。R4 恢复了历史 E1 的 G2 退化，但没有证明超过 Stage-2，也未显示 125→250 的确定性继续收益。
- 成本：4,000 train reward query，另有 566×2 次预注册 dev query；250 step 合计 `12,963.39` 秒（均值 `51.85` 秒/step），launcher 墙钟约 4 小时 9 分；13,561 个 1 秒 GPU 样本，峰值 `21,294 MiB`、最低剩余 `2,788 MiB`、utilization 峰值 100%；run 约 16 GB，结束后磁盘剩余约 15 GB。
- 产物清理：step125 评估、paired report 与 LoRA hash 固化且进程完全回收后，只删除 `/root/autodl-tmp/curious-vla-workspace/experiments/safe_grpo/r4_sdr_random_lora_1k_g4_seed20260812/checkpoints/global_step_125/actor/model_world_size_1_rank_0.pt`（`8,144,550,392` bytes，不可恢复），磁盘空余回升至约 22 GB；step125 LoRA/optimizer/config、全部 rollout/report 和 step250 完整 checkpoint 保留。
- 分析边界：step250 只能称为在相同 scene/update、两倍 rollout 成本下相对 E1 的系统级恢复；step125 虽与 E1 同为约 2,000 train rollout，却只有一半 scene exposure/update，不能称为严格等预算效率因果证据。所有 dev 均已访问，只是 exploratory evidence。
- 决策：R4-SDR 相对预注册直接对照 E1 通过发现门控并列入后续 paired second-seed 候选；由于仍未超过 E0，不能宣称 Random-G4 带来净模型提升。按矩阵先执行 F4-SDR，检验 FALS 在相同 G4 协议下是否优于 Random。
- 下一动作：只执行 `F4-SDR`；在启动前保留 R4 step250 与全部评估证据，只清理已完成评估的 step125 大体积 full-state 文件以满足磁盘预算。

### 记录 G4-004：F4-SDR FALS-G4 正式实验完成

- 状态：技术通过，科学未晋级。
- 假设与直接对照：在同一 Stage-2、SDR、G4、1,000 scene、250 update 协议下，只把 Random-1k 改为冻结 FALS-1k；直接对照 R4-SDR，E2 只作 FALS-G2 参考。
- 预注册门控：相对 R4-SDR，PDMS scaled 至少 `+0.01000`，Safe/Collision/DAC 均不下降，parse 至少 99.5%，无 clipping/NaN/覆盖缺口。
- 代码与配置：source `75e767aed4a9a00d2a7ec84878c5e3ef368eccf4`，source status 为空；FALS-1k SHA-256 `fd62a6f204806beff51fa7e1fb0f853027655b4b47f00f9633c787b04e0ffed0`，1,000 个唯一 train token，dev/held-out 重叠 0；其余配置与 R4 字节级同类，step125/250 均保存并评估。
- 数据与技术结果：`COMPLETE`、`exit_code=0`；1,000 group、4,000 train rollout，每组严格 4 条；step125/250 dev 各 566 条。train parse `99.95%`（2 条 parse-failure），两套 dev 100%，clipping 0，全部指标 finite；无 OOM、NaN、traceback、CUDA、no-space 或 killed。step125/250 LoRA SHA-256 为 `27531d6b...9d35`、`3b84e01c...9de4`；GPU、Ray、Gunicorn/8901 与训练进程全部回收。
- 训练信号：exact-zero group `7.5%`、`0<std<0.05` group `0.4%`、平均 headroom `0.49786`；train PDMS scaled `0.36090`、Safe `0.39450`。FALS 确实把预算集中到更难且更有组内差异的场景，但该信号变化未转化为足够的 dev 增益。
- 探索性效果：step125 PDMS scaled/PDMS/Safe 为 `0.65653 / 0.68065 / 0.72085`，step250 为 `0.66465 / 0.68945 / 0.72968`。step250 相对 R4 的 PDMS scaled/PDMS/Safe/Collision/DAC 为 `+0.00576 / +0.00621 / +0.00530 / +0.00265 / +0.00530`；主要安全点估计不降，但 PDMS scaled 未达到 `+0.01000`。
- 不确定性：step250−R4 的 PDMS scaled 95% paired bootstrap CI 为 `[-0.01617,+0.02747]`；step125−R4 为 `-0.00097`（`[-0.02458,+0.02336]`）。step250 相对 E2-G2 的 PDMS scaled/Safe 为 `-0.00765 / -0.01060`，CI 均跨 0；没有证据表明 FALS 从 G2 扩到 G4 获益。
- 成本：4,000 train reward query，另有 566×2 次 dev query；250 step 合计 `13,331.19` 秒（均值 `53.32` 秒/step），launcher 墙钟约 4 小时 16 分；13,894 个 1 秒 GPU 样本，峰值 `21,266 MiB`、最低剩余 `2,816 MiB`、utilization 峰值 100%；完成时磁盘剩余约 6.1 GB。
- 产物清理：全部评估、paired report 与 LoRA hash 固化且进程回收后，删除 F4 step125/250 和 R4 step250 的三个 `model_world_size_1_rank_0.pt`（各 `8,144,550,392` bytes，不可恢复）；保留所有 LoRA/optimizer/config/rollout/report，磁盘空余回升至约 29 GB。后续 R4-RAW/A4 都从 Stage-2 独立重训，不依赖这些 full-state。
- 分析边界：F4 说明在当前单 seed 已访问 dev 上，FALS-G4 相对 Random-G4 的点增益不足预注册阈值；不能据此否定历史 E2-G2，也不能用更好的 train variance 替代效果门控。
- 决策：F4-SDR 未晋级，关闭 FALS 的 G4 扩展，不把 F4 纳入 second-seed 候选；继续前置审计 ADAS 定义。
- 下一动作：只执行零 GPU `A0`，判断能否冻结一个真实有选择性的 train-only ADAS-1k manifest。

### 记录 G4-005：A0 发布 ADAS eligible-pool 定义审计完成

- 状态：定义门控失败；A2/A4 与全部 ADAS-Hybrid 路线按定义门控跳过。
- 假设与直接对照：只核对发布 ADAS token file 在冻结 train/dev/held-out 中的覆盖和 G4 Bernoulli 边界；不做模型推理，不访问 rollout，不生成候选 manifest。
- 预注册门控：eligible train pool 至少 1,000 个且必须在冻结 train 内有真实选择性；否则禁止把全 train 抽样表述为 ADAS。另行核对历史严格 `p^4+(1-p)^4 < 0.1` 是否存在可行场景。
- 代码与配置：source `fb967d67666b40d5d3b35231316426ac092d7713`，source status 为空；CPU-only、seed `20260812`、`manifest_write=false`。
- 数据：发布 ADAS file SHA-256 `594b20aa...82414`，共 5,656 个 token；它包含全部 train `4,525`、dev `566` 和旧 held-out `565`，split 外为 0。train 内被 gate 排除数为 0，eligible ratio 为 `1.0`；冻结 Random-1k 只用于边界核对，未被改写。
- 技术结果：`COMPLETE`、`exit_code=0`；报告、输入 hash、source 与运行配置完整，远端 `38 passed`；未写 ADAS manifest，未启动 GPU 或模型进程。
- 定义结果：发布 ADAS gate 接纳整个冻结 train，因此 gate 后均匀抽 1,000 与 Random 抽样同义。对 G4，`p^4+(1-p)^4` 的理论最小值为 `0.125`，所以历史严格 `<0.1` 门控不可能接纳任何场景。
- 成本：0 GPU、0 reward query、0 模型推理；只产生约 KB 级审计证据。
- 分析边界：能够否定当前发布列表和历史 Bernoulli 规则作为本轮 ADAS-G4 selector 的资格；不能否定重新提出、独立预注册的其他 ADAS 定义，但本轮禁止据 dev 反向设计新阈值。
- 决策：冻结使用发布参数的旧 A2/A4、AF4-SDR 和所有 ADAS+SLDR 分支；全通过 gate 不能伪装为 selector。用户在 G4-007 重新授权的是一个参数重新预注册的 P0-ADAS 新定义，不恢复本条失败配置。
- 下一动作（当时）：开放 `C0` matched second-seed；该动作随后在 G4-006 被用户终止并由 G4-007 取代。

### 记录 G4-006：C0 Random-SDR matched second-seed 预注册

- 状态：用户终止，未形成科学结果；中间产物已清理。
- 原计划：使用新 seed `20260813` matched 重训 Random-SDR G2/G4，确认 G4−G2 是否跨 seed 同方向。
- 实际执行：C0-G2 使用 source `6d0fb73ff751663c01beff7ae1d3e38c23f07652`、冻结 Random-1k、Stage-2、SDR、G2；运行到 step80 时用户调整研究问题，要求先做 raw-PDMS reward 消融，因此精确终止进程组。
- 技术终态：停止后 `exit_code=1`，无 step125 checkpoint、无 dev 评估；GPU 回到 0 MiB 占用，8901、Ray、Gunicorn 和训练进程全部回收。该退出码代表外部终止后的非完整 run，不是算法技术失败。
- 成本：已产生 `80 × 4 × 2 = 640` 次 train query；没有产生 dev query，不能读取中途 train reward 作方法结论。
- 产物清理：删除本次 36 MB run 目录、576 KB EasyR1 debug 目录和 4 KB launcher log；均不可恢复。目标路径逐一核对后删除，未触碰任何历史保留集合。
- 决策：C0 被新用户指令取代，不运行 C0-G4，不把 step80 当作 G2 结果。
- 下一动作：预注册并执行 `R4-RAW`。

### 记录 G4-007：R4-RAW Random-G4 raw-PDMS reward 消融预注册

- 状态：已预注册、待执行。以下门控在 R4-RAW 生成任何 dev 结果前冻结。
- 假设与直接对照：只回答 `pdms_scaled`（SDR）相对 NAVSIM raw `pdms` 是否改善 G4 GRPO。R4-RAW 与已完成 R4-SDR 都从同一 Stage-2 独立训练；Random-1k、G4、seed、steps、LoRA、optimizer、KL、生成和评测协议保持一致，唯一训练变量为 reward scalar。
- 代码与配置：R4-RAW 固定使用 `compute_score_group_raw_pdms`，其 `overall/training_reward=pdms`；`accuracy=pdms_scaled` 以及 Safe/Collision/DAC/Progress/TTC/Comfort 继续完整记录。seed `20260812`、4 groups/step、每组 4 条、250 steps、step125/250 双 checkpoint、train temperature/top-p `1.0/1.0`、dev `0.6/0.95`。
- 数据：Random-1k SHA-256 `3ae99bb9...21fa8`，1,000 个唯一 train token，与 dev/旧 held-out 重叠 0；Stage-2 model hash 必须与 R4-SDR 一致。只访问既有 566-token exploratory dev，禁止访问旧 held-out。
- 技术门控：`COMPLETE`、`exit_code=0`；4,000 train rollout = 1,000 group × 4，step125/250 dev 各 566；parse 至少 99.5%、clipping 0、全部数值 finite、无 OOM/CUDA/no-space/killed，GPU/Ray/8901/训练进程完整回收。
- 科学门控：按第 6.2 节计算 `Δ_SDR = R4-SDR − R4-RAW`；step125/250 都报告 PDMS scaled、PDMS、Safe 及六个 NAVSIM 分项和 20,000 次 paired bootstrap。只有 step250 PDMS scaled `≥ +0.01000` 且 Safe/Collision/DAC 不降，才称 SDR 正向；`≤0` 为未建立贡献，中间区间为证据不足/trade-off。
- 训练证据：逐 step 保存 policy loss、entropy、KL、clip、grad、LR、advantage、reward 分项、response length/parse/clipping、分阶段耗时/throughput及逐秒显存；保存 raw response、parsed poses 和代表性样本。曲线异常必须与样本和固定 dev 共同解释，不能单凭 reward/loss 宣称效果。
- 成本预算：4,000 次 train query，加 step125/250 共 1,132 次 dev query；预计墙钟约 4–4.5 小时、峰值显存约 21.3 GiB。full-state 只在 LoRA/hash/rollout/曲线/paired report 固化且资源回收后精确清理。
- 分析边界：这是同一 seed、已访问 dev 上的单变量 reward 消融；paired CI 不覆盖训练 seed，不构成 unseen final confirmation。
- 下一动作：冻结 source 并通过远端测试后，只启动 `r4raw`。

### 记录 G4-008：R4-RAW Random-G4 raw-PDMS reward 消融闭环

- 状态：技术通过，科学证据不足并存在安全分项 trade-off。预注册门控未过，不把点估计包装成 SDR 正向证据。
- 假设与直接对照：R4-RAW 与 R4-SDR 均从同一 Stage-2 独立训练；Random-1k、G4、seed `20260812`、250 steps、LoRA/optimizer/KL、train/dev generation 与 split 完全一致，唯一训练变量是 `overall/training_reward: pdms_scaled → pdms`。
- 代码与硬件：source `aacc4381c8afe74b47cdd61806fd8c2d4d9c289a`，source status 为空；RTX 4090 `24,564 MiB` 首要门控通过；远端完整测试 `41 passed`，launcher shell/compile/diff check 通过。
- 技术验收：`COMPLETE`、`exit_code=0`；4,000/4,000 train rollout、step125/250 各 566/566 dev；train parse 99.9%、dev parse 100%、clipping 0、无非有限值/OOM/CUDA/no-space/killed。完成后 GPU 0 MiB，trainer、Ray、Gunicorn 与 8901 全部回收。
- 固定 dev：R4-RAW step125 `PDMS scaled/PDMS/Safe = 0.64849/0.67197/0.71201`；step250 为 `0.65535/0.67916/0.71731`。step250−step125 的 PDMS scaled 为 `+0.00686`，95% paired CI `[-0.01785,+0.03150]`，不构成 checkpoint 优势证据。
- SDR 消融（方向均为 `R4-SDR − R4-RAW`）：step125 的 PDMS scaled/PDMS/Safe/Collision/DAC/Progress/TTC/Comfort 差值依次为 `+0.00901/+0.01000/+0.00883/+0.00265/+0.00530/-0.00076/+0.00530/-0.00177`，PDMS scaled 95% paired CI `[-0.01283,+0.03086]`；step250 依次为 `+0.00354/+0.00409/+0.00707/+0.00530/-0.00353/-0.00124/-0.00707/+0.00177`，PDMS scaled 95% paired CI `[-0.02098,+0.02871]`。两次均使用同 566 token、20,000 bootstrap。
- 科学决策：step250 主差值 `+0.00354` 低于 `+0.01000`，且 DAC 为负；因此 SDR 没有通过预注册正向门控。差值又大于 0，不能按 `≤0` 分支称 SDR 未建立任何贡献；结论固定为“单 seed、已访问 dev 上证据不足并有 Collision/Safe 与 DAC/TTC 的 trade-off”。paired CI 只覆盖场景，不覆盖训练随机性。
- Train signal：raw reward mean/std `0.62023/0.46076`，exact-zero `22.2%`、low-nonzero `10.9%`、headroom `0.29054`；SDR 对照分别为 reward mean/std `0.61773/0.44952`、zero `22.1%`、low `9.7%`、headroom `0.28310`。两者组内信号几何接近，不能用 train reward 均值替代 dev 结论。
- 曲线与样本：250 个 step 的 policy loss、entropy、KL/PPO-KL、clip、grad、LR、advantage、reward 分项、response length、timing、throughput 和逐秒 GPU 全覆盖。entropy 首末 `0.17597→0.17213`，KL mean/max `0.000136/0.000454`，grad mean/max `0.02038/0.03504`，LR 恒为 `1e-6`，high/low clip fraction 均为 0；平均 step/generation/update 为 `53.19/24.80/16.52` 秒，平均 throughput `495.44 tokens/s`。代表样本保留高/低 reward、最长、parse failure 和最大 group gap，且 raw response 可用。
- 成本与资源：4,000 train query + 1,132 dev query；GPU 采样墙钟 `15,352` 秒，峰值 `21,266 MiB`，最低空闲 `2,816 MiB`。训练与双评估完成后两个 full-state model 合计 `16,289,100,784` bytes。
- 证据与清理：生成 `training_history.csv`、`training_curves.svg`、curve summary、代表样本、evidence manifest、三份 paired report 和 30 项 `result_sha256.txt`，全部校验通过。资源回收后精确删除 step125/250 的两个 `model_world_size_1_rank_0.pt`（各 `8,144,550,392` bytes，不可恢复）；保留双 LoRA、optimizer/config、原始 rollout、metrics/paired/曲线/hash，run 约 614 MB，磁盘恢复到约 29 GB。
- 下一动作：实际仅执行了零 GPU 的 P0-ADAS train-only 审计；随后按用户范围锁定，不启动 A4-SDR。

### 记录 G4-009：P0-ADAS G4 train-only 参数审计预注册

- 状态：已预注册、待执行。以下候选、顺序和门控在读取候选 eligible count 前冻结；执行不访问任何 dev/held-out model output。
- 假设：发布 ADAS 的 `n_rollout=8/group_size=32/ε_div=0.1` 不能直接解释为 G4。P0 只判断在保留生产三阶段语义时，是否存在一个适合 `n_rollout=group_size=4`、能从冻结 train 形成真实选择性的参数。
- 冻结输入：D0 Stage-2 G4 train-only `adas_scores.csv`，18,100 rows = 4,525 token × 4，SHA-256 `4ade4d7f...930785`；冻结 train/dev/held-out manifests 只用于覆盖与泄漏门控，Random-1k 作为 train-signal 参考。禁止读取 R4/A4 dev 分数选择参数。
- 固定参数：`n_rollout=4`、`group_size=4`、`std_threshold=0.01`、`confidence_threshold=0.10`、manifest seed `20260812`。不 sweep std/conf，不修改 `p_est=pdms_mean/pdms_range` 或 predicted-std 公式。
- 唯一候选顺序：先 `ε_div=0.20`，再 `ε_div=0.35`。理想二值 G4 下 `p=0.5` 的 metric 为 `0.125`，`p=0.25/0.75` 为 `0.3203125`；所以 0.20 表示只接纳 2/2 mixed，0.35 表示接纳所有非恒定 1/3、2/2、3/1 mixed。只选择按此严格到宽松顺序的第一个全门控通过者，不因实际计数新增阈值。
- Pool 门控：eligible 至少 1,000，且至多为冻结 train 的 80%（3,620），所有输出 finite，eligible 的 `p_est` 均位于 `[0,1]`。通过 gate 后用固定 seed 从 eligible 均匀取 1,000 并排序写 manifest；不得用 pool 外 token 补齐。
- Signal 门控：冻结 ADAS-1k 在同一 D0 G4 scores 上的 `pdms_scaled` exact-zero ratio 必须严格低于 Random-1k，且 mean `pdms_scaled` group std 必须严格高于 Random-1k；这只证明 train signal 分配改变，不预判 A4 dev 效果。
- 失败分支：若 0.20 和 0.35 都未全通过，则 `manifest_written=false` 并关闭 ADAS/A4；不放宽 pool ratio，不调 conf/std，不看 dev 后追加候选。若通过，只冻结一个 eligible pool、参数和 ADAS-1k，随后才预注册唯一一组 A4-SDR。
- 证据：保存 group stats、两候选逐门控报告、Random/selected signal 对照、输入/结果 hash、source/status/run.env/exit code；CPU-only，GPU/query 均为 0。
- 下一动作：远端通过 shell/pytest/compile 检查后，只运行 `run_p0_adas_g4_parameter_audit.sh`。

### 记录 G4-010：P0-ADAS G4 train-only 参数审计闭环与范围冻结

- 状态：CPU-only 技术通过；不访问 dev/held-out model output，GPU 和训练 query 均为 0。该结果只作为已归档的后续备选，不改变当前 R4-RAW 的 SDR 消融范围。
- 冻结选择：两个候选均过门控；`ε_div=0.20` 为 `3,349 → 1,071 → 1,022`，`ε_div=0.35` 为 `3,349 → 2,832 → 2,782`。按预注册的严格到宽松顺序，唯一冻结 `ε_div=0.20`。
- Pool 与 manifest：eligible pool 为 1,022，占冻结 train `22.5856%`，eligible SHA-256 `b6d7f99b9f390764898b371fe2be3c88a24865d7d33678afe1fafc0b87c49fcc`；ADAS-1k manifest SHA-256 `e4703e46d0f580ff6ff883646899a0e109d2f7462661030097a80bc353617fcc`。
- Train signal：Random-1k 的 `pdms_scaled` exact-zero group ratio/mean group std 为 `0.23/0.318914`，ADAS-1k 为 `0/0.525640`；两项预注册 signal gate 均通过。这只证明 train signal 分配改变，不表示 ADAS 具有 dev 收益。
- 完整性：`COMPLETE`、`exit_code=0`，source `7ede0337b145f1fab02ed15db728d82231e9db77` 且状态为空；实验产物保留在 `p0_adas_g4_parameter_audit_seed20260812`。
- 当前决策：不启动 `ADAS-1k + SDR + GRPO, G=4`。只有在用户后续再次明确授权后，才可以该冻结参数和 manifest 单独预注册 A4-SDR。

## 11. 当前下一动作

当前无下一实验动作。`Random-1k + raw-PDMS + GRPO, G=4` 已完成 SDR 消融闭环；P0 CPU 审计仅归档为未来备选。除非用户再次明确授权，不启动 A4-SDR 或其他训练。
