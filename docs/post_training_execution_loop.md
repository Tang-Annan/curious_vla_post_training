# Curious-VLA 后训练科学增量实施与证据闭环

> 本文档是当前后训练方法开发的唯一执行方案与决策台账。服务器实验目录中的日志、配置和产物是原始证据，本文档保存可追溯事实、解释边界、阶段门控和下一步决策。已完成记录只追加、不回写；外部论文只提供待验证假设，不能替代 Curious-VLA 自身证据。

## 1. 当前决策快照

- 最后更新：2026-08-14
- 开发分支：`codex/post-training-analysis`；本次改写前 Git HEAD：`3d1eeb5`
- 当前最佳已训练候选：E2 FALS-only `global_step_250/actor`
- 当前唯一下一动作：R0 离线诊断；完成并写回门控结论前不启动新 GPU 训练
- 当前封存动作：暂停 E2 step-50 dev 审计；held-out 继续完全封存
- 保留的服务器核心证据：E0、D0、E2、全部 manifest，以及 E2 step 50/250 checkpoint
- 已排除方向：继续调 SLDR、把 Std-Floor 直接叠加到 E2、为了吞吐改动正式生成协议
- 当前主线：先验证“FALS 选择目标与 GRPO 优化尺度是否错配”，再分别判断 Dr.GRPO 与 Dynamic Sampling；Recovery 仅保留为条件研究分支

| 阶段 | 状态 | 目的 | 当前动作 |
| --- | --- | --- | --- |
| E0–E4 | 已闭环 | 建立 Stage-2、Vanilla GRPO、FALS、SLDR、Std-Floor 的证据基线 | 不重跑、不调参 |
| R0 | 待执行 | 用现有 D0/E2 rollout 诊断 advantage-scale mismatch 与 zero-signal group | 当前唯一动作 |
| R1 | 条件执行 | FALS + Dr.GRPO 单因素消融 | 仅在 R0-R1 门控通过后启动 |
| R2-D / R2-G | 条件执行 | 在线过滤 zero-variance group；父方法由 R1 结果决定 | 不默认依赖 R1 成功 |
| R3 | 可选 | Failure-Guided Recovery 等预算可行性对照 | 核心 R1/R2 路线完成后再决定 |
| C0 | 条件执行 | 最终新方法与 E2 的匹配训练种子确认 | 仅对达到晋级线的方法执行 |
| F0 | 暂停 | 只审计最终胜出方法的预注册 checkpoint | 方法开发结束后恢复 |
| F1 | 封存 | 一次性 held-out 确认 | 代码、方法、checkpoint 全部冻结后执行 |

当前自适应路线为：

```text
R0
├─ R1 动机成立 ──> R1
│                  ├─ R1 晋级 + dead-group 门控成立 ──> R2-D（Dr.GRPO + Dynamic）
│                  ├─ R1 不晋级 + dead-group 门控成立 ─> R2-G（GRPO + Dynamic）
│                  └─ dead-group 门控不成立 ──────────> C0 / F0
├─ R1 动机不成立、dead-group 门控成立 ───────────────> R2-G
└─ 两个门控均不成立 ─────────────────────────────────> 以 E2 进入 F0
```

R3 不在这条硬主线中；它的成功与否不得阻塞最终审计。

## 2. 已完成证据基线

### 2.1 同协议 566-token dev 结果

| 阶段 | 唯一变化 | PDMS scaled | PDMS | Safe | Collision | DAC | Progress | TTC | Comfort | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| E0 | Stage-2 baseline | 0.65938 | 0.68361 | 0.72438 | 0.96643 | 0.75265 | 0.91135 | 0.94876 | 0.92049 | 冻结基线 |
| E1 | 随机 1k + Vanilla GRPO | 0.64281 | 0.66691 | 0.70671 | 0.95936 | 0.74205 | 0.91071 | 0.94170 | 0.91873 | 完整负对照 |
| E2 | FALS 1k + Vanilla GRPO | 0.67230 | 0.69758 | 0.74028 | 0.96908 | 0.76678 | 0.90938 | 0.95406 | 0.92049 | 当前最佳候选 |
| E3 | 随机 1k + SLDR | 0.62994 | 0.65266 | 0.68905 | 0.95760 | 0.72792 | 0.90941 | 0.93816 | 0.91873 | SLDR 独立贡献为负 |
| E4 | E3 + Std-Floor | 0.64344 | 0.66691 | 0.70848 | 0.95760 | 0.74558 | 0.91016 | 0.94346 | 0.92049 | 部分补救 E3，未恢复 E0 |

E2 相对 E0 的点估计为：PDMS scaled `+0.01292`、PDMS `+0.01397`、Safe `+0.01590`；Progress `-0.00197`。

固定 seed `20260814`、20,000 次 paired bootstrap 的结果是：

- E2 − E0 PDMS scaled：95% CI `[-0.00924, +0.03516]`；
- E2 − E0 Safe：95% CI `[-0.00883, +0.04064]`；
- E2 − E0 Progress：95% CI `[-0.00378, -0.00027]`；
- E2 相对 E1/E4 的 PDMS scaled 与 Safe CI 为正。

因此，E2 可以声明“明显优于已训练的 Vanilla GRPO 与 Std-Floor 变体”，但不能声明“已稳定超过 Stage-2”。paired bootstrap 只覆盖 dev scene 不确定性，不等价于训练随机种子稳定性。

### 2.2 Train rollout 诊断

| 阶段 | 训练 reward | Reward mean/std | Exact-zero std | `0 < std < 0.05` | Headroom | Safe | 解释边界 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| D0 | PDMS scaled | 0.59636 / 0.43850 | 18.14% | 16.24% | 0.26838 | 65.71% | 冻结 Stage-2，4 rollout/group |
| E1 | PDMS scaled | 0.61446 / 0.45183 | 46.30% | 13.60% | 0.17365 | 65.55% | 随机 1k，2 rollout/group |
| E2 | PDMS scaled | 0.35843 / 0.45048 | 38.80% | 6.70% | 0.24294 | 39.25% | FALS 主动集中困难样本 |
| E3 | SLDR | 0.66047 / 0.42596 | 44.60% | 15.40% | 0.16895 | 65.40% | reward 定义不同，不与 E1/E2 均值直比 |
| E4 | SLDR | 0.65341 / 0.42762 | 43.70% | 15.90% | 0.16945 | 64.75% | reward 定义不同，不与 E1/E2 均值直比 |

已成立的事实：

1. 随机 1k Vanilla GRPO 在当前预算下退化，且 46.30% group 没有相对优势信号。
2. FALS 将 exact-zero group 降低 7.5 个百分点，并把 headroom 从 0.17365 提高到 0.24294；它解决了部分离线预算选择问题，但仍留下 38.80% zero-signal group。
3. SLDR-only 明确退化；Std-Floor 只部分补救 SLDR，不能替代 FALS。
4. E2 的训练 reward 较低是其困难样本分布的结果，不能和 E1 的随机 train 均值直接解释为 policy 退化。
5. held-out 尚未使用；所有方法判断都来自 train 诊断与同一 dev 协议。
6. 当前只有一个正式训练 seed。后续不能把单 seed + scene bootstrap 写成训练稳定性结论。
7. E1/E3/E4 的大体积原始产物已清理，指标、配置和结论仅由本台账保留；E0/D0/E2 原始证据仍在服务器。

## 3. 新路线要回答的科学问题

| 假设 | 当前证据 | 仍缺什么 | 对应阶段 |
| --- | --- | --- | --- |
| H-S：FALS 能把预算移向困难且可学习场景 | E2 优于 E1，zero std 下降、headroom 上升 | 相对 Stage-2 的稳定优势 | C0 / F1 |
| H-O：FALS 的 raw headroom 信号被标准 GRPO 的组内 std normalization 抹平 | 设计上高度可疑；E2 使用 `n=2`，非零组归一化后 advantage 幅值近乎恒定 | 同 policy 的分布诊断与 Dr.GRPO 单因素结果 | R0 / R1 |
| H-D：当前 policy 下的 zero-variance group 浪费在线 rollout 与 reward 预算 | E2 exact-zero std 为 38.80% | 有界补采能否提升模型，成本是否可接受 | R0 / R2 |
| H-R：部分 persistent failure 需要结构化反馈，而非继续采样 | 仅有外部工作动机 | 相对 blind resampling 的等预算本项目证据 | R3 |

最终叙事只能由通过门控的模块组成：

- R1/R2 未通过时，不能预先宣称 “Select–Optimize” 已成立；
- R3 未通过时，不能把 “Recover” 写成项目贡献；
- NoRD、DAPO、ELF-VLA、DriveDPO 等只用于形成假设，不作为 Curious-VLA 的效果证据。

## 4. 冻结比较协议

### 4.1 数据与选择边界

1. 冻结 split：train 4,525、dev 566、held-out 565，三者两两重叠为 0。
2. FALS 只能来自 D0 train rollout；唯一 top-1,000 manifest SHA-256 为 `fd62a6f204806beff51fa7e1fb0f853027655b4b47f00f9633c787b04e0ffed0`。
3. dev 只用于正式方法比较和最终一次 checkpoint 审计；不得用于调阈值、改 feedback 模板、选 Dynamic Sampling 上限或反复重跑。
4. held-out 只允许在 F1 使用一次，不参与模型、checkpoint、seed、阈值或路线选择。
5. R0/R3 的场景、阈值和统计都只允许使用 train 证据；R3 的 feedback 不得把 GT trajectory 坐标暴露给模型。

### 4.2 E2 对照协议

R1/R2 默认逐项继承 E2 的 resolved config；未列出的配置也必须保持一致。

| 项目 | 冻结值 |
| --- | --- |
| base model | `models/sft_stage2` |
| train manifest | 唯一 FALS top-1,000 |
| training budget | 250 steps |
| rollout | 2 responses/group，temperature 1.0，top_p 1.0 |
| LoRA | rank 8，alpha 16，`q/k/v/o_proj`，exclude visual |
| actor attention | `sdpa` |
| validation backend | vLLM 0.11.0 内置 `FLASH_ATTN`，CUDA Graph enabled |
| train / validation batch | 4 / 4 |
| max response length | 512 |
| vLLM memory utilization | 0.55 |
| max num batched tokens | 4608 |
| train reward | 原始 grouped PDMS reward `compute_score_group_fast` |
| validation | 同一 566-token dev、每 token 1 response |
| seed | discovery seed `20260812` |
| checkpoint | 正式比较先统一使用 step 250；F0 再审计预注册 checkpoint |

单因素约束：

- R1 相对 E2 只允许 `adv_estimator: grpo → dr_grpo`；
- R2 相对其父方法只允许启用 exact-zero group filtering 与有界补采；
- 不同时改 reward、FALS budget、learning rate、clip、KL、LoRA、生成参数或 validation 协议；
- 任一变更若改变 validation 随机协议，必须在新协议下重跑对应 baseline，不能沿用现有 E0/E2。

### 4.3 已关闭与条件重开项

- 不再安装独立 `flash_attn`，不采用 batch 8、token budget 8192 或 LRU reward cache。
- 不继续调整 SLDR 系数，不把 E3/E4 的负结果变成超参数搜索起点。
- reward 4-worker 并发不自动恢复。只有 R2 profiling 显示 reward 等待占 step wall time 至少 20%，才允许先做固定输入等价性验证；否则保持单 worker。
- F0-A step-50 审计暂停。R1/R2 方法开发完成前，不新增任何 checkpoint dev 查询。
- 每个正式方法只运行一个预注册 discovery 配置；除明确实现错误外，不因 dev 结果重跑或调参。

### 4.4 完整性硬门控

每个正式阶段必须保存：

- source commit 与 source status；
- resolved config、seed、active manifest 和 manifest hash；
- launcher/run log、退出码、`RUNNING/COMPLETE/FAILED` 状态；
- 原始 rollout、拆分覆盖、train diagnosis、final-dev metrics；
- checkpoint tracker 与目标 actor checkpoint；
- 主进程、Ray、Gunicorn、端口和 GPU 回收证据。

证据不完整时只允许标记“技术失败”或“证据不足”，不得作效果结论。

## 5. 实时闭环规则

每一阶段必须按以下顺序执行，不能在结果写回前启动下一阶段：

1. **预注册**：在本节写清唯一变量、输入、输出、硬门控、晋级线和失败分支。
2. **实现**：只修改该阶段必需代码；补最小单元测试和启动器门控。
3. **执行**：使用新目录启动，禁止覆盖正式产物；正常运行只读监控。
4. **技术验收**：先核对覆盖、边界、checkpoint、日志、退出码和资源回收。
5. **效果分析**：区分点估计、paired bootstrap、训练 seed 稳定性和成本。
6. **决策写回**：记录“推进 / 回退 / 跳过 / 最小修复重试 / 结束方法开发”。
7. **调整下一步**：只按预注册分支改变下一阶段，不根据好看的次要指标临时换门槛。

通用决策语义：

- **技术失败**：不作算法效果结论；保留失败目录，只修一个已定位问题后用新目录重试。
- **科学负结果**：技术验收通过但未达晋级线；保留消融结论，回到父方法，不调参追分。
- **证据不足**：只补能解除当前歧义的最小诊断；不得把它写成提升。
- **工程晋级**：达到预注册点估计与安全约束，可进入下一单因素消融。
- **科学确认**：除工程晋级外，还需 C0 匹配 seed 和 F1 held-out；未完成前统一称“dev 候选”。

## 6. 分阶段实施方案

### R0：Selection–Optimization Mismatch 离线诊断

**目标**

用现有证据回答两个独立问题：

1. FALS 依赖的 raw reward gap/headroom 在标准 GRPO 中被多大程度地消除；
2. exact-zero group 的比例是否高到值得引入有界 Dynamic Sampling。

**输入**

- D0：4,525 train token × 4 rollout，同一冻结 Stage-2 policy；
- D0 `fals_ranking.csv`、唯一 FALS top-1,000 与冻结随机 train 1k；
- E2：1,000 FALS token × 2 train rollout；
- 不访问 dev 或 held-out，不运行 GPU 推理。

D0 是 difficulty/variance 关联的主证据，因为所有 scene 来自同一 policy。E2 rollout 来自训练中的变化 policy，只用于描述实际训练信号；若日志没有 step 元数据，不得声称 zero-group 随 step 上升或下降。

**分析项**

1. 对每个 group 输出 `mean/std/min/max/reward_gap/headroom/difficulty/learnability/safe/parse`。
2. 在 D0 上按 difficulty quintile 与 headroom quintile 报告 reward gap/std；给出固定 seed `20260814`、20,000 次 bootstrap CI 和 Spearman 相关。
3. FALS learnability 本身包含 headroom，因此“FALS top-1k 的 std 更高”只能作描述，不能作为独立因果证据。
4. 计算：
   - `A_GRPO = (r - group_mean) / (group_std + eps)`；
   - `A_Dr = r - group_mean`；
   - 非零组中两者的绝对幅值、分位数、比例和 group ranking 变化。
5. 单独报告 E2 `n=2` 的性质：使用 sample std 时，任意非零二元组的标准 GRPO advantage 绝对值近似 `1/sqrt(2)`；R0 要量化 raw reward gap 的异质性，而不是把这个数学事实误写成训练提升。
6. 用 E2 empirical zero rate 对每 step 目标 4 个有效 group、2–8 次 generation batch 做 Monte Carlo，选择使 250-step 整体填充失败概率低于 1% 的最小补采上限，并报告期望 rollout/reward 开销。

**拟新增产物**

远程 `experiments/safe_grpo/r0_difficulty_bias_seed20260812/`：

- `group_metrics.csv`；
- `r0_report.json`；
- `advantage_scale.csv`；
- `difficulty_bias.svg`；
- source、命令、输入 manifest/hash 和 `COMPLETE`。

**R1 启动门控**

以下两项同时成立才启动 R1：

1. E2 informative group ratio `1 - exact_zero_std_ratio >= 0.50`；
2. E2 非零 group 的 `reward_gap` 四分位距 `P75 - P25 >= 0.10`。

第一项依据现有 E2 为 61.20%，已满足；第二项由 R0 冻结计算。门槛不得在看过 R0 图后修改。

若 R1 门控失败，则不以 “difficulty bias” 为由训练 Dr.GRPO。

**R2 启动门控**

- E2 exact-zero std ratio `>= 0.25`；
- Monte Carlo 估计在最多 8 个 generation batch 内可把 250-step 整体填充失败概率压到 1% 以下；
- 预计平均 raw rollout 开销不超过 E2 的 2.0 倍。

现有 exact-zero 38.80% 已满足第一项。若成本门控失败，记录 Dynamic Sampling 在当前单卡预算下不可行，不能通过接受不足 batch 来绕过。

**结果分支**

- R1、R2 门控都通过：先执行 R1；
- 仅 R1 通过：执行 R1，随后进入 C0/F0；
- 仅 R2 通过：跳过 R1，执行 R2-G；
- 两者都不通过：停止方法扩展，以 E2 进入 F0。

### R1：FALS + Dr.GRPO

**唯一变量**

```text
E2: FALS top-1k + GRPO
R1: FALS top-1k + Dr.GRPO
```

Dr.GRPO 使用 `A = r - group_mean`，不做组内标准差归一化。不得同时启用 Dynamic Sampling、SLDR、Std-Floor 或 reward/learning-rate 变更。

**最小实现范围**

- 在 `EasyR1/verl/trainer/core_algos.py` 注册 `dr_grpo`；
- 在 `EasyR1/verl/trainer/config.py` 更新合法 estimator 说明；
- 在 `scripts/run_safe_grpo_experiment.sh` 增加 R1 单因素入口与独立目录；
- 在 `tests/test_safe_grpo.py` 验证去均值、零方差为零、不除 std、group size 门控和启动器唯一变量；
- 不为未来 estimator 创建新抽象。

**运行前硬门控**

- R0 `COMPLETE` 且 R1 gate 通过；
- FALS manifest hash 与 E2 完全一致；
- resolved config 与 E2 逐项 diff，除 estimator 外无有效差异；
- 本地相关测试、服务器 `py_compile/bash -n` 和 5-step 无 dev smoke 通过；
- GPU、8901、目标目录空闲，source clean。

**正式验收**

- 250 steps，train 1,000×2、dev 566×1；
- step-250 actor 完整；
- 无 train/dev/held-out 泄漏、未知 token、OOM、traceback 或 clipping；
- 记录 gradient norm、KL、policy loss、reward gap 与 wall time，检查 raw advantage 是否造成不稳定。

**工程晋级线（相对 E2）**

- `Delta PDMS_scaled >= +0.01000`；
- `Safe >= 0.74028`；
- `Collision >= 0.96908`；
- `TTC >= 0.95406`；
- parse success 为 1.0，clipping 为 0；
- 同时报告 20,000 次 paired bootstrap，不以单个次要指标替代主门槛。

达到点估计门槛只代表可作为 R2 父方法；若 CI 仍跨 0，仍称“exploratory dev candidate”，必须经过 C0/F1 才能作稳定提升结论。

**结果分支**

- 达到全部晋级线：R1 成为当前父候选；若 R2 gate 通过，进入 R2-D。
- 主指标为正但 `< +0.01`，或任一安全约束下降：记为弱/负结果，不叠加；若 R2 gate 通过，回到 E2 执行 R2-G。
- `Delta PDMS_scaled <= 0` 或训练不稳定：拒绝 R1；只有确认实现错误时允许一次最小修复重试。

### R2-D / R2-G：有界 Dynamic Sampling

**父方法选择**

- R1 达到全部晋级线：R2-D = R1 + Dynamic Sampling；
- R1 未晋级或被跳过：R2-G = E2 + Dynamic Sampling。

这样 Dynamic Sampling 的独立价值不会被 Dr.GRPO 的成败绑架，也不会形成无法归因的多 trick 实验。

**唯一变量**

在每个 optimizer step 中，按实际训练 reward 对同一 `uid` 的 rollout 分组：

```text
group reward 完全相同
    -> 丢弃该 group，不进入 optimizer batch
    -> 从同一 FALS manifest 补采新 group

group reward 存在差异
    -> 保留
```

只过滤 exact-zero group；不使用 `std < 0.05`。optimizer 的有效 group 数始终保持 4，禁止达到上限后静默使用不足 batch。

**实现原则**

当前 `ray_trainer._make_batch_data` 已有“生成—过滤—补采”和 `max_try_make_batch` 骨架，但现有 `online_filtering` 按 group mean 区间过滤，不是 zero-variance filtering。R2 只复用其循环和上限，不得把现有 mean-filter 直接当作 DAPO。

- 过滤标量必须与 advantage 使用的 training reward 完全一致；
- 补采上限由 R0 Monte Carlo 选择，硬上限 8；
- 达到上限仍不足 4 个有效 group 时本 step/阶段直接失败，不回退到不足 batch 或未过滤路径；
- 先做 20-step、无 dev pilot，验证 batch、日志、资源与成本，再启动正式 250-step；
- 记录 generated/kept/dropped groups、attempts/step、reward queries、p50/p90 step time 和总 wall time。

**吞吐条件门控**

只有 pilot 显示 reward server 等待占 step wall time 至少 20%，才重新打开“原 reward 服务 4 workers + 有界 client concurrency”。开启前必须对固定输入逐样本验证所有 reward component 完全一致；否则保持 E2 单 worker 路径。Dynamic Sampling 增加 query 数本身不是自动修改 reward 服务的理由。

**工程晋级线**

相对其父方法：

- `Delta PDMS_scaled >= +0.01000`；
- Safe、Collision、TTC 不低于父方法；
- parse/clipping 不退化；
- 平均 raw rollout 与 wall time 不超过父方法 2.0 倍；
- 同时报告相对 E2 的绝对结果和“每 1,000 reward query 的收益”，避免只展示等 step 分数。

未达线则回退到父方法，不调 zero 阈值、不增加 generation 上限追分。

### R3：Failure-Guided Recovery 等预算可行性

R3 是可选研究分支，不是 R1/R2 的必做后继。只有核心方法冻结后、train 上仍有至少 10% persistent-failure group 时才启动。

**Persistent failure 定义与样本**

1. 只从 train split 选候选；
2. 用当前最佳冻结 checkpoint 对候选重新生成 4 个 baseline rollout；
3. 4 个 rollout 全部 unsafe，或 `max PDMS_scaled == 0`，才进入集合；
4. 固定抽取最多 200 个 token；样本和模板在生成结果前冻结。

**等预算对照**

每个 token 使用相同解码预算各生成一次：

- Control：原 prompt 的 blind resampling；
- Treatment：原 prompt + 原失败 trajectory + 固定结构化 feedback。

feedback 只由 collision、DAC、TTC、progress、comfort 等 train reward component 映射，不允许注入 GT trajectory 坐标，也不允许根据 dev 结果改模板。

**Meaningful recovery**

`PDMS_scaled` 相对 4 个 original rollout 的最佳值至少提高 0.05，且 Collision/TTC 不下降；另行报告 unsafe → safe 的比例。

**继续门控**

同时满足才进入后续训练设计：

- Treatment absolute recovery rate `>= 20%`；
- Treatment 比 blind resampling 至少高 10 个百分点；
- paired bootstrap 的 Treatment − Control 95% CI 下界 `> 0`。

不满足即关闭 Recovery 分支，不进行 prompt sweep。

**训练边界**

不得把 feedback-conditioned response 直接塞入原 prompt 的 on-policy Dr.GRPO group：两者条件 prompt 不同，且离线成功样本会引入 off-policy 偏差。若 R3 通过，只允许先写新的设计决策，优先评估“同原始 prompt 的 chosen/rejected preference 数据”或最小 recovery-SFT。当前仓库没有 DPO 训练路径，因此 DPO 不属于本轮默认实施范围。

### C0：匹配训练 seed 确认

只有 R1 或 R2 达到工程晋级线时执行。为控制成本，只新增一组预注册匹配 seed `20260813`：

1. 用相同 FALS manifest 分别运行 E2 comparator 与最终新方法；
2. 两者使用相同训练/生成 seed 和同一 dev 协议；
3. 报告两个训练 seed 的逐 seed 差值、均值、paired scene bootstrap 和安全约束；
4. 不在看到第二 seed 后修改方法或阈值。

最低确认标准：

- 新方法相对 E2 在两个 seed 上 PDMS scaled 差值均为正；
- 两 seed 平均差值 `>= +0.01000`；
- 两 seed 的 Safe、Collision、TTC 均无方向一致的退化。

不满足时，新方法只能作为单 seed 探索结果；最终候选回退 E2。一个额外匹配 seed 只能降低偶然性风险，不能被描述为充分的多 seed 统计证明。

### F0：最终 checkpoint 审计

方法开发结束后恢复，只对最终候选执行一次。

- 新方法训练前明确保留 step 50 与 step 250；不得事后挑选更多 checkpoint。
- 复用 step-250 既有 dev；只新增 step-50 的同协议 566-token dev。
- 只有 step 50 的 PDMS scaled 更高，且 Safe、Collision、TTC 均不低于 step 250，才切换；否则保留 step 250。
- 若所有新方法均未晋级，则直接执行原 E2 step 50/250 审计。
- F0 完成后冻结 source、config、manifest、seed 和唯一 checkpoint。

### F1：一次性 held-out

启动条件：

- R1/R2/R3/C0 已按门控完成、跳过或关闭；
- F0 已冻结唯一 checkpoint；
- held-out manifest 仍未被此前任何阶段读取；
- 评估命令、输出目录和失败恢复规则已预注册。

F1 只运行一次。无论结果好坏都不得再改模型、checkpoint 或阈值；最终报告必须同时列出 dev 与 held-out、所有负结果、训练 seed 限制、rollout/reward 成本和适用边界。

## 7. 结果到下一步的唯一映射

| 最新结果 | 下一步 |
| --- | --- |
| R0：R1/R2 gate 都失败 | 停止方法扩展，E2 → F0 |
| R0：仅 R1 gate 通过 | R1；完成后 C0/F0 |
| R0：仅 R2 gate 通过 | R2-G |
| R1 全部晋级 | R2 gate 通过则 R2-D，否则 C0 |
| R1 弱正向或负向 | 不叠加 Dr.GRPO；R2 gate 通过则 R2-G，否则 E2 → F0 |
| R2 晋级 | R2 成为父候选；决定是否执行非阻塞 R3，然后 C0 |
| R2 不晋级 | 回退其父候选；决定是否执行非阻塞 R3，然后 C0/F0 |
| R3 可行性失败 | 关闭 Recovery，不影响核心候选 |
| R3 可行性通过 | 先新增 preference/SFT 设计决策，不自动训练 |
| C0 不确认 | 新方法降级为 exploratory，E2 → F0 |
| C0 确认 | 最终新方法 → F0 |
| F0 完成 | 冻结唯一 checkpoint → F1 |
| F1 完成 | 只汇总，不再调整 |

## 8. 未来实现影响面

以下文件是后续阶段的预期最小影响面，不代表本次文档改写已经实现对应功能。

| 阶段 | 预计文件 | 最小改动 | 验证 |
| --- | --- | --- | --- |
| R0 | `projects/safe_grpo/analyze_difficulty_bias.py`、`tests/test_safe_grpo.py` | 读取现有 rollout，输出统计与图 | 合成分布、manifest 污染、覆盖、固定 seed |
| R1 | `core_algos.py`、`config.py`、正式 launcher、tests | 注册 `dr_grpo` 与 R1 入口 | 数学单测、唯一变量 diff、5-step smoke |
| R2 | `ray_trainer.py`、`config.py`、正式 launcher、tests | exact-zero group filtering 与有界补采 | 零/非零 group、固定有效 batch、上限失败、指标记录 |
| R3 | 新的 train-only feasibility 工具 | treatment/control 推理与 recovery 统计 | 等预算、无 GT prompt 泄漏、无 dev/held-out |
| C0/F0/F1 | 正式 launcher 与验收脚本 | seed/checkpoint/held-out 隔离入口 | manifest、目录、一次性门控 |

## 9. 历史与实时执行记录

> 记录 001–019 保留当时的事实、判断和“下一动作”，其中旧下一动作可能已被更晚记录取代；判断当前动作时只看第 1 节和最新记录。历史记录所称旧版“2.1 节冻结配置”对应当前第 4.2 节。

### 记录 001：E0 首次 full-actor 尝试失败

- 状态：已归档，不计为正式 baseline。
- 证据：远程 `experiments/safe_grpo/e0_stage2_dev_seed20260812_failed_full_actor/`。
- 事实：完整 actor 约占 15 GiB，vLLM 在 validation 前因单卡显存不足安全退出。
- 分析：问题来自单卡 hybrid-engine 同驻留预算，不是数据或 reward 故障。
- 决策：E0/D0 保留 rank-8 零初始化 LoRA wrapper；失败目录保留，不覆盖。
- 下一动作：用同协议重跑 E0。

### 记录 002：E0 正式 baseline 完成

- 状态：通过。
- 代码：`7c8adda`。
- 证据：远程 `experiments/safe_grpo/e0_stage2_dev_seed20260812/`。
- 覆盖：566 行、566 个唯一 dev token；`COMPLETE` 存在，`exit_code=0`。
- 指标：
  - PDMS scaled / overall：`0.659383745`
  - PDMS：`0.683609782`
  - safe rate：`0.724381625`
  - collision compliance：`0.966431095`
  - drivable-area compliance：`0.752650177`
  - ego progress：`0.911352276`
  - TTC compliance：`0.948763251`
  - history comfort：`0.920494700`
  - parse success：`1.0`
  - reward latency：`260.40 ms/sample`
  - response mean：`366.29`，clipping：`0`
- 资源：主进程、Ray、Gunicorn 和端口 8901 均退出；GPU 回收至 0 MiB。日志观测显存峰值为 19.88 GiB，该值不是连续采样的严格峰值。
- 分析：baseline 完整可信；不得与早期不同生成上限/随机协议的 E1 smoke 直接比较。
- 决策：冻结为正式 E0；进入 D0 前先完成用户要求的 A0 加速测试。
- 下一动作：A0。

### 记录 003：A0 attention backend 与 reward 候选筛选

- 状态：通过，候选已收敛。
- 隔离证据：远程 `experiments/benchmarks/`；独立端口 18901–18903；未修改现有环境。
- attention 事实：当前 vLLM 0.11.0 在 RTX 4090 上自动选择 `vllm.v1.attention.backends.flash_attn.FlashAttentionBackend`。环境没有独立 `flash-attn` 包，但 vLLM 自带并已使用其 FlashAttention backend。
- attention 决策：不安装独立 `flash_attn`；它不会替换当前 validation 的 vLLM 生成路径，且 244 MiB wheel 会给稳定环境增加无证据收益的变更。
- 单 Gunicorn worker/client 并发结果（48 个固定样本，两次重复中位数）：
  - concurrency 1：4.94 samples/s；
  - concurrency 2：3.45 samples/s；
  - concurrency 4：2.50 samples/s；
  - concurrency 8：2.22 samples/s。
- 分析：只提高 client 并发会在单 worker 内争用，拒绝该配置。
- server matrix（64 个固定样本，4 路 client）：
  - 原服务 1 worker：2.20 samples/s；
  - 原服务 4 workers：7.61 samples/s，3.46×，公共 reward 指标逐项一致；
  - 实验 LRU 4 workers：9.83 samples/s，但至少一个样本出现指标漂移。
- 决策：拒绝 LRU 实验实现；保留“原服务 4 workers + 有界 client 并发”为候选，必须在正式应用前补生产路径测试。Gunicorn worker 数单独增加不能加速当前串行 client，因此两侧必须配套验证。
- validation matrix 证据：远程 `experiments/benchmarks/validation_batch_20260813_1855/`。三组均为 64/64 唯一 token、parse 1.0、无 clipping/OOM/traceback、`exit_code=0`；顶层 `COMPLETE` 和 `exit_code=0` 存在，进程、18903 和 GPU 已回收。
  - batch 4 / token 4608：239 s，PDMS scaled `0.673721445`，PDMS `0.691845499`；
  - batch 8 / token 4608：206 s，较基线快 13.8%，但仅 27/64 token 的 pose 完全一致，45/64 token 的全部 reward 指标一致；
  - batch 8 / token 8192：282 s，较基线慢 18.0%，仅 24/64 token 的 pose 完全一致，46/64 token 的全部 reward 指标一致。
- 分析：validation sampling 是随机生成；改变 batch 或 scheduler token budget 会改变随机数消费/调度并形成不同输出协议。batch 8 的小规模墙钟收益不足以抵消重跑正式 E0 和破坏已冻结比较协议的成本；8192 token budget 没有速度收益。
- 决策：正式生成协议保持 `val_batch_size=4`、`max_num_batched_tokens=4608`，现有 E0 继续有效。A0 完成，不安装独立 `flash_attn`，不采用 LRU，不采用 batch 8/8192。
- 下一动作：A1 只实现“原 reward 服务 4 workers + 每 batch 最多 4 路 client concurrency”，默认在正式编排中显式启用；以同协议 64-token validation 检查覆盖、输出、reward、墙钟和资源回收。若生成或 reward 与 batch-4 基线不一致，回退该实现并直接进入 D0。

### 记录 004：终止 A1，进入正式 D0

- 状态：A1 按项目优先级延期；D0 已启动。
- 代码与配置：服务器继续使用已完整远程验证的 `7c8adda`，未同步 A1 并发实现；正式参数见 2.1。
- 原始证据：D0 目录 `experiments/safe_grpo/d0_stage2_train_n4_seed20260812/`，launcher `logs/d0_stage2_train_n4_seed20260812.launcher.log`，启动 PID `259785`。
- 分析：A0 已足够回答 FlashAttention 和 validation 参数问题；A1 尚无端到端远程回归，继续让它阻塞 D0 会偏离后训练主目标。
- 决策：撤回开发分支上的 A1 实验代码；不再重复 A0。reward throughput 优化只保留到原计划 E5 再评估。
- 下一动作：只读监控 D0；完成后验证 4,525 个 train token、每 token 4 rollout、18,100 行、dev/held-out 为 0 和 `diagnosis.json`，再依据诊断确定 E1/E2 行为。

### 记录 005：冻结正式路线并补齐 E1 验收产物

- 状态：通过，不改变实验协议。
- 代码与配置：功能提交 `d3f5083`；E1 仍固定 1k train manifest、250 steps、每个 train token 2 个 rollout、566-token final dev、rank-8 LoRA 和 2.1 节生成/reward 配置。
- 原始证据：本地 `tests/test_safe_grpo.py`，显式工作区临时目录运行结果为 11 passed、4 skipped；`git diff --check` 通过。跳过项是本地缺少可选运行依赖，不涉及新增的 rollout 拆分和覆盖测试。提交后将三个变更文件上传到服务器 `/tmp` 隔离路径，`bash -n` 与服务器 Python `py_compile` 均通过，随后删除临时文件；未修改 D0 checkout 或占用 GPU。
- 覆盖与完整性：E1 结束时将混合原始日志严格拆分为 train/dev 产物；要求 train 1,000×2、dev 566×1，拒绝两个 manifest 重叠、未知 token、缺失或重复覆盖，并分别生成 train diagnosis 与 final-dev metrics。
- 分析：原启动器已经执行 250-step 训练、定期 checkpoint 和最终 dev，但此前只复制混合 rollout 日志，无法单独证明训练与最终验证覆盖。新增逻辑仅在训练结束后整理和验收产物，不改变模型、采样、reward、随机顺序或训练过程。
- 决策：保留该最小验收补丁；D0 运行期间只推送开发分支，不热更新服务器运行 checkout。D0 完成后同步最新提交并在服务器完整环境做最小验证，再启动 E1。
- 下一动作：按四档 ETA 规则只读监控 D0；D0 完整验收、分析和写回通过后，启动正式 E1。

### 记录 006：补齐 E2–E4 正式启动门控

- 状态：通过，尚未启动 E2–E4，也未提前选择 E2 manifest。
- 代码与配置：正式启动器增加 E2 FALS-only、E3 SLDR-only、E4 SLDR + Std-Floor 入口；所有阶段复用 E1 的 1k/250-step/checkpoint/final-dev 路径。E2 只允许显式传入 D0 后生成的 1k FALS manifest；E3 只切换训练 reward；E4 只在 E3 `low_nonzero_std_ratio >= 0.10` 时切换 `adv_estimator=std_floor_grpo`、`std_floor=0.05`，并保持 E3 的 SLDR reward。
- 原始证据：本地 `tests/test_safe_grpo.py` 结果为 13 passed、4 skipped，`git diff --check` 通过；服务器 `/tmp` 隔离路径的 `bash -n`、Python `py_compile` 以及 E4 门控阈值 0.10/0.099 正反例通过；增加 held-out 校验后的最新启动器再次通过服务器 `bash -n`，未修改 D0 checkout 或使用 GPU。
- 覆盖与完整性：E1–E4 启动前均强制 1,000 个非空唯一 token，要求属于冻结 train split 且与 dev、held-out 均无重叠；远程只读核对确认冻结 split 为 train 4,525、dev 566、held-out 565，三者两两重叠均为 0，随机 train 1k 完整属于 train 且与 dev/held-out 重叠为 0。run.env 记录实际 manifest、reward function 和 advantage estimator。
- 分析：SLDR 日志同时保存 `training_reward` 与 `pdms_scaled`。E3/E4 的 group std 必须使用实际 `training_reward`，并与 PyTorch GRPO 一致采用 sample std；原诊断逻辑已相应修正，否则 E4 门控会读取错误信号。
- 决策：保留正式入口和强门控；不从当前代码预设 E2 选择结果，仍只依据 D0 完整诊断生成唯一 FALS 1k manifest。
- 下一动作：继续按 ETA 四档规则监控 D0；完成后先写回 D0 事实与 E2 manifest 决策，再启动 E1。

### 记录 007：强化 FALS 输入边界

- 状态：通过，不改变 FALS 排序公式或预算。
- 代码与配置：FALS 构建器不再静默忽略 train manifest 外 rollout；发现 dev、held-out 或其他未知 token 时直接失败。正式 run.env 改为记录各阶段实际 active manifest，避免 E0/D0 元数据误指向 1k iteration manifest。
- 原始证据：本地 `tests/test_safe_grpo.py` 结果为 14 passed、4 skipped，新增污染 rollout 反例通过；Python compile 与 `git diff --check` 通过。提交后最新启动器与 FALS 构建器在服务器 `/tmp` 隔离路径通过 `bash -n`/`py_compile`，随后删除临时文件，未触碰 D0 checkout 或 GPU。
- 分析：D0 当前只读检查没有发现 train 外 token，但正式 E2 的数据来源边界必须由工具强制，而不能只依赖人工核对。
- 决策：保留强失败边界；D0 完成后仅在全量覆盖和 train/dev/held-out 隔离均通过时生成 FALS manifest。
- 下一动作：保持 D0 自适应只读监控；完整验收后写回 D0 诊断并启动 E1。

### 记录 008：冻结正式 checkpoint 选择语义

- 状态：通过，不改变保存频率或训练过程。
- 代码与配置：E1–E4 继续每 50 steps 保存、最多保留 2 个 checkpoint；阶段完成前新增硬验收，要求 tracker 的 `last_global_step=250` 且 `global_step_250/actor` 存在。
- 原始证据：本地全套相关测试为 14 passed、4 skipped，`git diff --check` 通过；提交后的最新启动器在服务器 `/tmp` 隔离路径通过 `bash -n`，随后删除临时文件，未触碰 D0 checkout 或 GPU。
- 分析：final dev 在 step 250 训练结束后执行，而中途 `val_freq=-1`，因此 tracker 中早期保存时形成的 `best_global_step` 不构成同条件模型选择证据。正式比较统一使用 final `global_step_250`，不误用 step 50。
- 决策：冻结 step 250 为 E1–E4 唯一正式 checkpoint；若缺失则阶段失败，不用较早 checkpoint 兜底。
- 下一动作：等待 D0 完整验收，随后用最新正式启动器执行 E1。

### 记录 009：D0 冻结 train rollout 诊断完成并冻结 E2 输入

- 状态：通过；D0 无需重跑，允许推进 E1。
- 代码与配置：D0 使用提交 `7c8adda90451dbdd8bc8bd9cc8360c4f4d896abc`、seed `20260812`、每 token 4 rollout 和 2.1 节冻结协议；source status 为空。运行时原始 `diagnosis.json` 保留未覆盖；使用开发分支最新版 `analyze_rollouts.py` 复算 sample std 到独立 `diagnosis_sample_std.json`。
- 原始证据：服务器 `experiments/safe_grpo/d0_stage2_train_n4_seed20260812/`；`COMPLETE` 于 21:55:05 CST 写入，`exit_code=0`，无 `RUNNING/FAILED`。复算文件 SHA-256 为 `a3ffa6224e5b2668b2a1285d6cc9bc6c620c21cb564a405fc4fabb3722104944`。
- 覆盖与完整性：`d0_train_rollouts.jsonl` 为 18,100 行、4,525 个唯一 train token，每 token 恰好 4 条；train 缺失 0，dev/held-out/manifest 外重叠均为 0。ADAS CSV 为 18,100 条 score。主 PID、trainer、Ray、Gunicorn 和端口 8901 已退出，GPU 为 0 MiB；无 OOM、traceback 或 fatal error。
- 关键结果：sample-std 诊断为 exact-zero std `18.14%`、`0 < std < 0.05` 为 `16.24%`、`std < 0.05` 合计 `34.39%`；reward mean/std 为 `0.59636/0.43850`，平均 headroom `0.26838`，pairwise ADE/FDE 为 `0.67096/1.60331`。safe rate `65.71%`，PDMS `0.61825`，collision/drivable-area compliance 为 `0.93910/0.71031`。响应长度 mean/p50/p90/p95/p99 为 `367.59/367/377/379/385`，512-token clipping 为 0；parse success 为 `99.9337%`。
- 解析失败分析：18,100 条中 12 条、涉及 11 个 token；均由输出未形成恰好 8 个可解析轨迹点而进入零分路径，response length 为 338–386，排除长度截断。单 token `4730affb7d4d5142` 失败 2 次，其余各 1 次；失败率仅 `0.0663%`，未改变覆盖或边界，不足以触发 D0 重跑。
- E2 manifest：依据全量 D0 train rollout，用冻结公式 `(1 - mean_reward) * headroom` 生成唯一 top-1,000；服务器路径 `manifests/fals_d0_seed20260812/fals_top_1000.txt`，SHA-256 为 `fd62a6f204806beff51fa7e1fb0f853027655b4b47f00f9633c787b04e0ffed0`。清单为 1,000 个唯一 train token，train 外、dev、held-out 重叠均为 0；第 1,000 名 learnability 为 `0.25`，同分按 token 排序，选择规则未事后调整。
- 分析：81.86% group 具有非零 reward 方差，平均 headroom 充足，GRPO 存在有效相对优势信号；与此同时，34.39% group 低于 std 0.05，支持后续分别检验 FALS 与 Std-Floor。D0 是 train 分布诊断，不能与 566-token E0 dev 均值直接作效果比较。
- 决策：接受 D0；不修解析器、不更改冻结配置、不开展额外加速测试。E1 继续作为随机 1k vanilla 对照；E2 只使用上述唯一 top-1,000 manifest；E4 仍由 E3 自身 `low_nonzero_std_ratio >= 0.10` 门控，不能用 D0 代替。
- 下一动作：服务器 checkout 以 Git fast-forward 同步开发分支，完成最小测试与启动门控后启动正式 E1；正常运行按四档 ETA 静默监控。

### 记录 010：E1 Vanilla LoRA-GRPO 正式启动

- 状态：运行中，启动健康检查通过。
- 代码与配置：服务器 checkout 以 Git fast-forward 同步至 `b5a63401813009e43adb11c9506665166561b030`，source status 为空；继续使用冻结随机 train 1k、250 steps、rank-8 LoRA、batch 4、token budget 4608、CUDA Graph、vLLM 内置 FlashAttention、显存比例 0.55 和串行 grouped reward。
- 原始证据：实验目录 `experiments/safe_grpo/e1_vanilla_lora_1k_seed20260812/`，launcher `logs/e1_vanilla_lora_1k_seed20260812.launcher.log`，启动 PID `862991`。
- 启动门控：服务器 `bash -n`、三个 Python 文件 `py_compile` 及 `tests/test_safe_grpo.py` 的 18 项测试全部通过；启动前 GPU、8901 和 E1 目录均空闲。
- 启动健康：`RUNNING`、run.env、train/dev manifest 和 source 证据均已落盘；trainer 占用约 18,994 MiB，reward 8901 返回 HTTP 200，主进程持续运行。
- 决策：按冻结协议继续 E1，不插入配置或加速探索；正常状态静默。
- 下一动作：按 ETA 四档规则监控 E1；完成后先核验 step-250 checkpoint、1,000×2 train rollout、566×1 final dev、指标和资源回收，再写回并推进 E2。

### 记录 011：E1 Vanilla LoRA-GRPO 完成

- 状态：技术验收通过，效果未超过 E0；保留为必要对照，不进入最终候选。
- 代码与配置：source commit `b5a63401813009e43adb11c9506665166561b030`，source status 为空；使用冻结随机 train 1k、250 steps、rank-8 LoRA 与 2.1 节生成/reward 协议。tracker 的 `last_global_step=250`，`global_step_250/actor` 模型、optimizer、extra state、dataloader 和 LoRA adapter 文件完整。
- 原始证据：服务器 `experiments/safe_grpo/e1_vanilla_lora_1k_seed20260812/`；运行时间 2026-08-13 22:37:45 至 2026-08-14 01:34:15 CST，约 2 小时 56 分 30 秒；`COMPLETE` 存在、`exit_code=0`，无 `RUNNING/FAILED`。
- 覆盖与完整性：raw rollout 2,566 行；train 为 2,000 行、1,000 个唯一 token、每 token 2 条；final dev 为 566 行、566 个唯一 token、每 token 1 条。train/dev 均无缺失、未知 token、相互重叠或 held-out 重叠。主 PID、trainer、Ray、Gunicorn/8901 已退出，GPU 回收至 0 MiB；日志无 OOM、traceback、fatal 或 exception。
- train diagnosis：reward mean/std `0.61446/0.45183`，exact-zero std `46.30%`，`0 < std < 0.05` 为 `13.60%`，平均 headroom `0.17365`，pairwise ADE/FDE `0.65745/1.47647`，safe rate `65.55%`，parse success `99.90%`（2/2,000 失败），无 clipping。
- final dev：PDMS scaled `0.64281`、PDMS `0.66691`、safe rate `0.70671`、collision compliance `0.95936`、drivable-area compliance `0.74205`、ego progress `0.91071`、TTC compliance `0.94170`、comfort `0.91873`、parse success `1.0`、clipping `0`。
- 与 E0 同协议差值：PDMS scaled `-0.01657`、PDMS `-0.01670`、safe rate `-0.01767`、collision `-0.00707`、drivable area `-0.01060`、progress `-0.00064`、TTC `-0.00707`、comfort `-0.00177`。各质量指标均未改善；reward latency 下降 `31.70 ms/sample` 不是模型质量收益，也不用于模型选择。
- 分析：E1 证明当前预算下随机 1k vanilla GRPO 没有改善冻结 baseline；1k train rollout 的 exact-zero std 高达 46.30%，表明随机采样浪费了大量相对优势为零的 group。这与 D0 的 FALS 动机一致，但不能预先断言 E2 会改善。
- 决策：接受 E1 作为完整、可复现的负结果，不调参、不重跑、不改变已冻结协议；按原顺序运行只改变样本选择的 E2 FALS-only。最终候选仍由 dev 比较决定，held-out 继续封存。
- 下一动作：确认唯一 FALS top-1,000 manifest 与记录的 SHA-256 一致，服务器 source 干净、GPU/8901/E2 目录空闲后启动 E2。

### 记录 012：E2 FALS-only LoRA-GRPO 正式启动

- 状态：运行中，启动门控与健康检查通过。
- 代码与配置：source commit `89ac72049f8da921e2dd46f82888742ae0eeec0c`，source status 为空；250 steps、rank-8 LoRA、GRPO reward、生成与 final-dev 协议均与 E1 相同，唯一实验变量是 train manifest 改为 D0 冻结的 FALS top-1,000。
- 原始证据：实验目录 `experiments/safe_grpo/e2_fals_lora_1k_seed20260812/`，launcher `logs/e2_fals_lora_1k_seed20260812.launcher.log`，启动 PID `97965`。
- 启动门控：实际 train manifest SHA-256 为 `fd62a6f204806beff51fa7e1fb0f853027655b4b47f00f9633c787b04e0ffed0`；1,000 个唯一 token 全部属于 train，与 dev/held-out 重叠为 0。服务器启动器语法、18 项相关测试、source clean、GPU/8901 和 E2 目录门控均通过。
- 启动健康：`RUNNING`、source、run.env 和 train/dev manifest 已落盘；run.env 确认 `reward_function=compute_score_group_fast`、`adv_estimator=grpo` 和唯一 FALS manifest。Ray/trainer/vLLM 与 Gunicorn/8901 正常，模型已加载至 GPU，日志无异常并进入 250-step loop。
- 决策：保持唯一变量为 FALS 样本选择，不做加速、参数或解析器变更；正常运行静默。
- 下一动作：按 ETA 四档规则监控 E2；完成后按与 E1 相同的 checkpoint、覆盖、final-dev 和资源回收标准验收，并比较 E0/E1/E2 后再推进 E3。

### 记录 013：E2 FALS-only LoRA-GRPO 完成

- 状态：通过；E2 final-dev 超过 E0 与 E1，保留为当前最佳候选。
- 代码与配置：source commit `89ac72049f8da921e2dd46f82888742ae0eeec0c`，source status 为空；实际 train manifest SHA-256 为 `fd62a6f204806beff51fa7e1fb0f853027655b4b47f00f9633c787b04e0ffed0`。除 FALS train 1k 外，训练预算、GRPO reward、LoRA、生成和 final-dev 协议均与 E1 一致。tracker 的 `last_global_step=250` 且 `global_step_250/actor` 完整。
- 原始证据：服务器 `experiments/safe_grpo/e2_fals_lora_1k_seed20260812/`；运行时间 2026-08-14 01:43:32 至 04:34:58 CST，约 2 小时 51 分 26 秒；`COMPLETE` 存在、`exit_code=0`，无 `RUNNING/FAILED`。
- 覆盖与完整性：raw rollout 2,566 行；train 为 2,000 行、1,000 个唯一 FALS token、每 token 2 条；final dev 为 566 行、566 个唯一 token、每 token 1 条。train/dev 均无缺失、未知 token、相互重叠或 held-out 重叠。主 PID、trainer、Ray、Gunicorn/8901 已退出，GPU 回收至 0 MiB；日志无 OOM、traceback、fatal 或 exception。
- train diagnosis：reward mean/std `0.35843/0.45048`，exact-zero std `38.80%`，`0 < std < 0.05` 为 `6.70%`，平均 headroom `0.24294`，pairwise ADE/FDE `0.68908/1.58960`，safe rate `39.25%`，parse success `99.95%`（1/2,000 失败），无 clipping。低 train reward 是 FALS 主动集中困难样本的预期结果，不能与 E1 随机 train 均值直接解释为模型退化。
- final dev：PDMS scaled `0.67230`、PDMS `0.69758`、safe rate `0.74028`、collision compliance `0.96908`、drivable-area compliance `0.76678`、ego progress `0.90938`、TTC compliance `0.95406`、comfort `0.92049`、parse success `1.0`、clipping `0`。
- 同协议差值：相对 E0，PDMS scaled `+0.01292`、PDMS `+0.01397`、safe `+0.01590`、collision `+0.00265`、drivable area `+0.01413`、TTC `+0.00530`，comfort 持平，progress `-0.00197`；相对 E1，PDMS scaled `+0.02949`、PDMS `+0.03067`、safe `+0.03357`。主要质量指标的一致改善支持 FALS 的独立贡献，但单 seed 结果仍需在最终审计中如实注明。
- 分析：E2 比 E1 降低零方差 group 比例 `7.5` 个百分点，并将平均 headroom 从 `0.17365` 提高到 `0.24294`，符合 FALS 将预算移向困难且可学习样本的设计。唯一轻微 trade-off 是 ego progress 相对 E0 下降 `0.00197`，不足以抵消安全与综合分数改善，但需保留在最终报告。
- 决策：接受 E2 为当前 dev 最佳候选，不提前访问 held-out；继续执行 E3 SLDR-only 独立消融。E3 仍使用与 E1 相同的随机 train 1k，不叠加 FALS，确保仅改变训练 reward。
- 下一动作：服务器同步最新台账提交，确认 source clean、随机 train 1k、GPU/8901/E3 目录与测试门控后启动 E3。

### 记录 014：E3 SLDR-only LoRA-GRPO 正式启动

- 状态：运行中，启动门控与健康检查通过。
- 代码与配置：source commit `650548b02c529fd67baa7c03f1e0a2468d862918`，source status 为空；继续使用 E1 的随机 train 1k、250 steps、rank-8 LoRA、GRPO 与相同生成/final-dev 协议。唯一实验变量是训练 reward 切换为 `compute_score_sldr`，未叠加 FALS 或 std-floor。
- 原始证据：实验目录 `experiments/safe_grpo/e3_sldr_lora_1k_seed20260812/`，launcher `logs/e3_sldr_lora_1k_seed20260812.launcher.log`，启动 PID `332016`。
- 启动门控：随机 train manifest 为 1,000 个唯一 train token，与 dev/held-out 重叠为 0；服务器启动器语法、18 项相关测试、source clean、GPU/8901 和 E3 目录门控均通过。
- 启动健康：`RUNNING`、source、run.env 和 train/dev manifest 已落盘；run.env 确认 `reward_function=compute_score_sldr`、`adv_estimator=grpo`。Ray/trainer/vLLM 与 Gunicorn/8901 正常，GPU 训练已进入 step loop，首步约 `39.8s/step`，日志无异常。
- 决策：保持唯一变量为 SLDR reward；正常运行静默。E4 是否执行仍只由 E3 完成后的实际 train diagnosis 门控。
- 下一动作：按 ETA 四档规则监控 E3；完成后比较 E0/E1/E2/E3，并仅在 E3 `low_nonzero_std_ratio >= 0.10` 时启动 E4。

### 记录 015：E3 SLDR-only 完成并通过 E4 门控

- 状态：技术验收通过，效果为负；E4 预注册门控通过。
- 代码与配置：source commit `650548b02c529fd67baa7c03f1e0a2468d862918`，source status 为空；随机 train 1k、250 steps、rank-8 LoRA、GRPO 和生成/final-dev 协议与 E1 相同，唯一实验变量为 SLDR train reward。tracker `last_global_step=250` 且 `global_step_250/actor` 完整。
- 原始证据：服务器 `experiments/safe_grpo/e3_sldr_lora_1k_seed20260812/`；运行时间 2026-08-14 04:40:11 至 07:30:56 CST，约 2 小时 50 分 44 秒；`COMPLETE` 存在、`exit_code=0`，无 `RUNNING/FAILED`。
- 覆盖与完整性：raw 2,566 行；train 2,000 行、1,000×2；final dev 566 行、566×1。无缺失、未知 token、train/dev 或 held-out 重叠。主 PID、trainer、Ray、Gunicorn/8901 已退出，GPU 0 MiB；日志无 OOM、traceback、fatal 或 exception。
- train diagnosis：SLDR reward mean/std `0.66047/0.42596`，PDMS scaled `0.61283`，exact-zero std `44.60%`，`0 < std < 0.05` 为 `15.40%`，平均 headroom `0.16895`，pairwise ADE/FDE `0.65106/1.45392`，safe `65.40%`，parse success `99.90%`（2/2,000 失败），无 clipping。
- final dev：PDMS scaled `0.62994`、PDMS `0.65266`、safe `0.68905`、collision `0.95760`、drivable area `0.72792`、progress `0.90941`、TTC `0.93816`、comfort `0.91873`、parse `1.0`、clipping `0`。
- 同协议差值：相对 E0，PDMS scaled `-0.02945`、PDMS `-0.03095`、safe `-0.03534`；相对 E1 分别为 `-0.01288/-0.01425/-0.01767`；相对 E2 为 `-0.04236/-0.04493/-0.05124`。SLDR-only 未改善任何主要质量指标，不能进入当前候选。
- 门控与分析：E3 有 `15.40%` group 满足 `0 < std < 0.05`，高于预注册阈值 `10%`，因此 E4 必须执行。该门控只证明存在足够的低非零方差样本，不代表 std-floor 必然补救 SLDR；E4 需要与 E3 直接比较。
- 决策：保留 E3 为 SLDR-only 负对照，不调参、不重跑；执行 E4，保持 SLDR、随机 train 1k 和其余协议不变，仅将 advantage estimator 切换为 `std_floor_grpo`、`std_floor=0.05`。
- 下一动作：同步台账提交，确认 E3 diagnosis 门控、source、测试、GPU/8901 与 E4 目录后启动 E4。

### 记录 016：E4 SLDR + Std-Floor 正式启动

- 状态：运行中，门控、启动门控和健康检查通过。
- 代码与配置：source commit `ddf3e64aec7311b20dc3b9cf74079ec342dd5bc5`，source status 为空；随机 train 1k、SLDR reward、250 steps、rank-8 LoRA 和生成/final-dev 协议与 E3 相同。相对 E3 的唯一变化为 `adv_estimator=std_floor_grpo`，`std_floor=0.05`。
- 原始证据：实验目录 `experiments/safe_grpo/e4_std_floor_lora_1k_seed20260812/`，launcher `logs/e4_std_floor_lora_1k_seed20260812.launcher.log`，启动 PID `565886`。
- 启动门控：服务器重新读取 E3 `low_nonzero_std_ratio=0.154`，满足 `>=0.10`；随机 train manifest 隔离、18 项测试、source clean、GPU/8901 和 E4 目录门控均通过。
- 启动健康：`RUNNING`、source、run.env、train/dev manifest 已落盘；run.env 确认 `compute_score_sldr` 与 `std_floor_grpo`。Ray/trainer/vLLM、Gunicorn/8901 和 GPU 正常，首步约 `39.0s/step`，日志无异常。
- 决策：保持 std-floor 为唯一实验变量，不做额外配置或加速变更；正常状态静默。
- 下一动作：按 ETA 四档规则监控 E4；完成后与 E3 直接比较 std-floor 贡献，并与 E0–E2 一并确定进入 E5 的候选。

### 记录 017：E4 SLDR + Std-Floor 完成

- 状态：技术验收通过；std-floor 相对 E3 有补救作用，但未超过 E0 或 E2，不进入最终候选。
- 代码与配置：source commit `ddf3e64aec7311b20dc3b9cf74079ec342dd5bc5`，source status 为空；该 revision 相对 E3 source 仅更新本台账，执行代码未变化。E3/E4 的 train、dev manifest 逐字节一致，随机 train 1k、SLDR reward、250 steps、rank-8 LoRA、seed 与生成/final-dev 协议保持不变；resolved config 确认 E4 使用 `adv_estimator=std_floor_grpo`、`std_floor=0.05`，相对 E3 的唯一有效实验变化是启用 std-floor estimator。
- 原始证据：服务器 `experiments/safe_grpo/e4_std_floor_lora_1k_seed20260812/`；运行时间 2026-08-14 07:41:50 至 11:06:27 CST，约 3 小时 24 分 37 秒；`COMPLETE` 存在、`exit_code=0`，无 `RUNNING/FAILED`。
- 覆盖与完整性：tracker `last_global_step=250`，`global_step_250/actor` 中 model、optimizer、extra state、Hugging Face 配置与 LoRA adapter 完整。raw rollout 2,566 行；train 为 2,000 行、1,000 个唯一 token、每 token 2 条；final dev 为 566 行、566 个唯一 token、每 token 1 条。无缺失、未知 token、train/dev 重叠或 held-out 重叠。主 PID、trainer、Ray、Gunicorn/8901 已退出，GPU 回收至 0 MiB。run log 无 OOM、traceback、fatal 或 exception；reward worker 在正常清理阶段收到 SIGTERM，与 `exit_code=0`、完整产物和资源回收一致，不是训练失败。
- train diagnosis：SLDR reward mean/std `0.65341/0.42762`，PDMS scaled `0.60482`，exact-zero std `43.70%`，`0 < std < 0.05` 为 `15.90%`，平均 headroom `0.16945`，pairwise ADE/FDE `0.64769/1.46148`，safe `64.75%`，parse success `99.85%`（3/2,000 失败），无 clipping。相对 E3，exact-zero std 下降 `0.90` 个百分点、低非零 std 上升 `0.50` 个百分点、headroom 上升 `0.00050`；train reward 与 safe 分别下降 `0.00706/0.00650`，训练诊断不呈单调改善。
- final dev：PDMS scaled `0.64344`、PDMS `0.66691`、safe `0.70848`、collision `0.95760`、drivable area `0.74558`、progress `0.91016`、TTC `0.94346`、comfort `0.92049`、parse `1.0`、clipping `0`。
- 同协议差值：相对 E3，PDMS scaled `+0.01350`、PDMS `+0.01426`、safe `+0.01943`、drivable area `+0.01767`、progress `+0.00074`、TTC `+0.00530`、comfort `+0.00177`，collision 持平，说明 std-floor 对 SLDR 退化存在部分补救。相对 E0，PDMS scaled `-0.01595`、PDMS `-0.01670`、safe `-0.01590`；相对 E2，分别为 `-0.02886/-0.03067/-0.03180`，且 collision、drivable area 与 TTC 均低于 E2。E4 仅与 E1 基本持平，PDMS scaled 和 safe 分别高 `0.00062/0.00177`。
- 分析：E3→E4 的直接对照支持“std-floor 能补救一部分 SLDR 损失”，但补救幅度不足以恢复冻结 baseline，更不足以替代 FALS。训练诊断与 dev 改善方向不完全一致，且当前只有单 seed，因此不把 E4 解读为普遍优于普通 GRPO。E2 仍是唯一同时超过 E0、E1、E3 和 E4 主要 dev 指标的候选；held-out 继续封存。
- 决策：接受 E4 为完整的 std-floor 正对照，不调参、不重跑、不将 SLDR 或 std-floor 叠加到 E2。冻结 E2 `global_step_250/actor` 为进入 E5 的模型候选；E5 只验证 reward 服务吞吐优化，不改变模型选择或生成协议。
- 下一动作：在固定 E2 rollout 输入上执行 E5，比较冻结的单 Gunicorn worker/串行 grouped reward 基线与“原 reward 服务 4 workers + 有界客户端并发”候选；逐样本 reward 指标必须完全一致、无丢失或重排，并重复测量吞吐及 p50/p90 latency。未通过等价性时保留单 worker 基线，不访问 held-out。

### 记录 018：E4 后计划重新审计

- 状态：计划已调整；E5 不再作为下一正式阶段或模型选择门控。
- 现有证据：对 E0/E1/E2/E4 的同一组 566 个 dev token 做固定 seed `20260814`、20,000 次 paired bootstrap。E2 相对 E0 的 PDMS scaled 均值差为 `+0.01292`，95% CI `[-0.00924, +0.03516]`；safe 均值差为 `+0.01590`，95% CI `[-0.00883, +0.04064]`，均跨 0。ego progress 差为 `-0.00197`，95% CI `[-0.00378, -0.00027]`。E2 相对 E1/E4 的 PDMS scaled 与 safe CI 为正，因此它仍是已训练变体中的最佳候选，但尚不足以声称稳定超过 Stage-2 baseline。
- checkpoint 缺口：E2 同时保留完整 `global_step_50` 与 `global_step_250`，当前只有 step 250 做过冻结 566-token dev。原 F0 本就要求只用 dev 确定 checkpoint，因此 step 50 是 held-out 前唯一需要补齐的模型选择证据。
- E5 审计：4 workers + 有界客户端并发只在 A0 小样本筛选中显示约 `2.46×` 吞吐，端到端实现已回退；所有正式训练已经完成，现在执行 E5 不能改善 E2 模型，只会增加评估基础设施变量。
- 决策：把 F0 拆为 F0-A checkpoint dev 审计和 F0-B 一次性 held-out；E5 后移为可选工程验证。F0-A 仅新增一次 step-50 validation，复用 step-250 既有结果；只有 step 50 的 PDMS scaled 更高且 safe、collision、TTC 均不低于 step 250 时才切换，否则保留 step 250。
- 下一动作：在启动 F0-A 前先释放服务器中已排除候选和失败实验的大体积产物，保留 E0/D0/E2、manifest 与闭环结论。

### 记录 019：服务器无效产物清理

- 状态：通过；候选、基线、诊断和隔离证据均保留，磁盘压力解除。
- 删除范围：移除失败的 E0 full-actor 目录、E1/E3/E4 正式实验目录、A0 benchmark 原始目录、被回退的 speed/smoke checkpoint 与对应 debug/launcher 产物。相关事实和结论继续由记录 001、003、011、015、017 保存。
- 保留范围：E0 正式 baseline、D0 全量诊断、E2 正式候选、全部 manifest 和代码 checkout。E2 的 `global_step_50/actor`、`global_step_250/actor`、`final_dev_metrics.json` 均已逐项复核存在；FALS top-1,000 manifest 保留，服务器 source status 为空。
- 空间结果：`/root/autodl-tmp` 从 `113/120 GB`、使用率 `95%`、可用 `7.2 GB` 降至 `58/120 GB`、使用率 `49%`、可用 `63 GB`，实际释放约 `55 GB`。正式实验区现在只剩 E0、D0 和 E2，合计约 `16 GB`。
- 恢复性：上述远程原始产物已直接删除，不能从服务器恢复；其指标、分析、决策与适用边界保留在本台账中。
- 决策：不再删除 E0/D0/E2 或 E2 的任一现存 checkpoint；现有空间足以完成 F0-A 和 F0-B。
- 下一动作：为 E2 `global_step_50` 建立隔离的 validation-only 运行，固定现有 566-token dev、batch 4、token budget 4608、CUDA Graph、vLLM 内置 FlashAttention 和 seed `20260812`，完成后按记录 018 的预注册规则冻结唯一 checkpoint。

### 记录 020：重新开启方法开发并冻结 R0–R2 自适应路线

- 状态：计划审计通过；原 F0-A step-50 审计暂停，held-out 继续封存。
- 触发证据：E2 相对 E0 的 PDMS scaled 与 Safe paired-bootstrap 95% CI 均跨 0；当前只有单训练 seed。E2 仍显著优于 E1/E4，但直接收尾不足以支撑稳定超过 Stage-2 或完整后训练算法贡献。
- 现有结果：E2 FALS 把 exact-zero std 从 E1 的 `46.30%` 降到 `38.80%`，headroom 从 `0.17365` 提高到 `0.24294`；说明离线选择有效但仍有大量 zero-signal group。E3 SLDR 为负，E4 Std-Floor 只部分补救，不再继续调参。
- 工程审计：当前 `core_algos.py` 的相关 grouped estimator 已有 GRPO 与 Std-Floor GRPO，但没有 Dr.GRPO；`ray_trainer._make_batch_data` 已有有界生成/过滤循环，但现有 online filtering 按 group mean 区间过滤，并不等价于 zero-variance Dynamic Sampling；仓库没有现成 DPO 训练路径。
- 方案修正：R0 以同 policy 的 D0 作为 difficulty/variance 主证据，E2 只作训练信号补充；Dynamic Sampling 不绑定 R1 成功，按 R2-D/R2-G 分支执行；补采必须维持固定有效 batch，不能在上限后静默缩小 batch；Recovery 必须先与 blind resampling 做等预算对照，禁止直接把不同 prompt、off-policy refined sample 注入原 GRPO group。
- 决策：采用 `R0 → R1/R2 自适应分支 → C0 → F0 → F1` 为当前正式路线；R3 为非阻塞可选研究分支。NoRD/DAPO/ELF-VLA 只作为假设来源，所有晋级以本项目预注册门控为准。
- 下一动作：只实现并运行 R0 离线诊断，生成 `group_metrics.csv`、`r0_report.json`、`advantage_scale.csv` 和图；完成技术验收、写回 R1/R2 gate 后再决定是否修改训练代码。

## 10. 后续记录模板

每个新结果按以下格式追加；历史记录不回写，第 1 节同步更新：

```text
### 记录 NNN：<阶段与事件>

- 状态：计划中 / 运行中 / 技术通过 / 科学正向 / 科学负向 / 证据不足 / 失败 / 按门控跳过
- 假设与唯一变量：<本阶段回答什么，只改什么>
- 预注册门控：<进入条件、技术验收、工程晋级线>
- 代码与配置：<commit、source status、resolved config、seed>
- 原始证据：<远程目录和关键文件>
- 数据边界：<manifest、hash、覆盖、train/dev/held-out 重叠>
- 技术结果：<退出码、checkpoint、异常、资源回收、墙钟与查询成本>
- 效果结果：<主指标、安全约束、paired bootstrap；如适用则列训练 seed>
- 分析边界：<已知事实、解释、尚未确定部分>
- 决策：<推进 / 回退 / 跳过 / 最小修复重试 / 结束方法开发>
- 下一动作：<唯一动作及启动门控>
```

## 11. 外部动机文献

这些资料只解释为什么提出 R1–R3，不参与本项目效果判定：

1. [NoRD: A Data-Efficient Vision-Language-Action Model that Drives without Reasoning](https://openaccess.thecvf.com/content/CVPR2026/html/Rawal_NoRD_A_Data-Efficient_Vision-Language-Action_Model_that_Drives_without_Reasoning_CVPR_2026_paper.html)
2. [DAPO](https://github.com/BytedTsinghua-SIA/DAPO) 与 [verl DAPO recipe](https://github.com/verl-project/verl-recipe/blob/main/dapo/README.md)
3. [ELF-VLA: Unleashing VLA Potentials in Autonomous Driving via Explicit Learning from Failures](https://arxiv.org/abs/2603.01063)
4. [DriveDPO](https://arxiv.org/abs/2509.17940)
5. [VL-DPO](https://arxiv.org/abs/2605.20082)
