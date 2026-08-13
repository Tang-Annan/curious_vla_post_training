# Curious-VLA 后训练计划执行闭环

> 本文档是后训练阶段的唯一执行台账，不包含环境、数据下载、切分生成等前期准备。每个阶段结束后，必须先在本文档记录证据、分析和决策，再启动下一阶段。实验目录中的日志和产物是原始证据，本文档保存结论及其推导链。

## 1. 当前快照

- 最后更新：2026-08-14 04:45 CST
- 开发分支：`codex/post-training-analysis`
- 开发分支同步状态：已推送，具体 revision 以 `codex/post-training-analysis` 的 Git HEAD 为准
- D0 source commit：`7c8adda`（运行期间未更新 checkout）
- 固定随机种子：`20260812`
- 当前动作：启动 E3 SLDR-only LoRA-GRPO
- 下一科学实验：E3 SLDR-only LoRA-GRPO
- 正式实验顺序：`E0 → D0 → E1 → E2 → E3 → E4（条件门控）→ E5 → F0`

| 阶段 | 状态 | 当前结论 | 下一动作 |
| --- | --- | --- | --- |
| E0 Stage-2 dev baseline | 完成 | 566/566 dev baseline 已冻结 | 作为同协议 dev 比较基线 |
| A0 validation 加速隔离测试 | 完成 | 保持 batch 4 / token budget 4608；拒绝独立 flash-attn、LRU 和 batch 8 | A1 只实现无损 reward 并发 |
| A1 reward 并发回归 | 延期 | 候选尚未完成端到端远程回归，不进入正式路线 | 仅在 E5 吞吐阶段重新评估 |
| D0 train rollout diagnosis | 完成 | 4,525×4 完整通过；存在可学习方差与 headroom | 保留为 E1–E4 train 诊断基线 |
| E1 Vanilla LoRA-GRPO | 完成 | 产物完整，但 final-dev 低于 E0，不作为最终候选 | 保留为 vanilla 对照 |
| E2 FALS only | 完成 | final-dev 超过 E0/E1，支持 FALS 独立贡献 | 保留为当前最佳候选 |
| E3 SLDR only | 启动中 | 仅切换训练 reward，其余与 E1 一致 | 完成后检验 SLDR 独立贡献与 E4 门控 |
| E4 Std-Floor GRPO | 条件待执行 | 受低非零方差占比门控 | E3 中至少 10% group 满足 `0 < std < 0.05` 才运行 |
| E5 grouped reward throughput | 待执行 | A0 仅提供候选配置 | 对最终候选做正式吞吐与等价性验证 |
| F0 最终审计与 held-out | 待执行 | held-out 保持封存 | dev 完成选型后一次性评估 |

## 2. 不可变实验约束

1. train/dev/held-out 严格隔离。held-out 不得参与训练、FALS、阈值选择、checkpoint 选择或超参数调整。
2. 正式 dev 比较固定为 566 个 token；D0 固定覆盖 4,525 个 train token，每 token 4 个 rollout，共 18,100 行。
3. 正式比较统一使用 `max_response_length=512`、seed `20260812`、vLLM CUDA Graph 和冻结的 token manifest。
4. 任何可能改变生成随机序列或输出分布的协议变更，不能直接与现有 E0 比较。若决定采用，必须用新协议重跑并重新冻结 E0。
5. 只允许依据 train rollout 构建 FALS；dev 只用于模型选择和消融比较；held-out 只用于最终一次性确认。
6. 单卡 24 GB 环境中，E0/D0 保留 rank-8、零初始化的 LoRA wrapper。PEFT 的 LoRA B 初始为零，因此不改变 Stage-2 初始输出；移除 wrapper 已被实测证明会突破 hybrid-engine 显存预算。
7. 每个正式阶段必须保存 source commit、source status、resolved config、seed、manifest、日志、rollout、指标和退出状态。证据不完整时不得标记完成。
8. 监控只读。预计剩余时间大于 1 小时时每 1 小时检查；大于 30 分钟且不超过 1 小时时每 30 分钟检查；大于 10 分钟且不超过 30 分钟时每 10 分钟检查；不超过 10 分钟时每 5 分钟检查。正常状态不写入对话和本文档，只有完成、失败或决策相关事件进入台账。

### 2.1 已冻结的正式配置

以下配置在 E0 和 A0 中已完成验证，从 D0 起固定，不再重复做 FlashAttention、batch size、token budget、LRU 或 reward 并发探索：

| 项目 | 固定值 |
| --- | --- |
| source baseline | `7c8adda`（后续功能提交必须保持相同协议） |
| seed | `20260812` |
| model | `models/sft_stage2` |
| LoRA | rank 8, alpha 16, `q/k/v/o_proj`, exclude visual |
| actor attention | `sdpa`；不另装 `flash_attn` |
| validation rollout backend | vLLM 0.11.0 内置 `FLASH_ATTN` |
| max response length | 512 |
| train / validation batch | 4 / 4 |
| vLLM CUDA Graph | enabled (`enforce_eager=false`) |
| vLLM memory utilization | 0.55 |
| max num batched tokens | 4608 |
| reward server | 原始实现，1 Gunicorn worker，串行 grouped request |

只有正式阶段出现 OOM、错误、覆盖失败或无法完成时，才能重新打开相关配置；单纯追求可能的速度收益不再构成变更理由。本地环境缺少依赖或无法代表 GPU/NAVSIM 行为时，直接在服务器当前完整环境执行最小相关测试，不为本地兼容添加代码路径。

## 3. 闭环推进规则

每个阶段按以下顺序闭环，不能跳过“分析与决策”直接启动下一步：

1. **执行**：使用冻结配置和唯一实验目录启动；禁止覆盖已有正式产物。
2. **验收**：核对退出码、覆盖数量、manifest 边界、日志异常、进程/端口/GPU 回收和必要指标。
3. **分析**：区分事实、解释和仍未确定的部分；只在同协议结果之间声明提升或退化。
4. **决策**：依据本节门控选择“推进、重试、回退、跳过或重跑基线”。
5. **写回**：更新“当前快照”和对应执行记录，附原始证据路径与关键数值。
6. **调整**：将新证据转成下一阶段的具体参数、门控或停止条件，然后才启动下一阶段。

决策规则：

- **通过**：全部硬门控成立，推进到计划中的下一阶段。
- **失败且原因明确**：保留失败目录，实施一个最小修复后以新目录重试；不得覆盖失败证据。
- **证据不足**：不作效果结论，补充最小必要测试。
- **协议发生变化**：标记现有跨协议比较无效，先重跑对应 baseline。
- **E4 门控不成立**：记录“按计划跳过”，直接进入 E5，不为了运行 E4 而降低门槛。
- **候选方法在 dev 无优势**：保留消融结果，不进入 held-out；选择 dev 上满足主指标与安全约束的候选。

## 4. 分阶段实施路线与门控

### A0. validation 加速隔离测试

目标：缩短验证墙钟时间，同时不改变正式结果，不污染当前环境与实验产物。

执行项：

1. 核实 vLLM 实际 attention backend，判断独立安装 `flash_attn` 是否会覆盖当前路径。
2. 用冻结 E0 rollout 测试 reward client 并发、Gunicorn worker 数和缓存候选。
3. 用固定 64-token dev 子集测试：
   - `val_batch_size=4`, `max_num_batched_tokens=4608`；
   - `val_batch_size=8`, `max_num_batched_tokens=4608`；
   - `val_batch_size=8`, `max_num_batched_tokens=8192`。
4. 比较 wall time、覆盖、parse、生成输出/pose、reward 指标、异常和显存。

硬门控：

- 独立目录和端口；正式 E0/D0 目录、现有 Python 环境和代码 checkout 不被修改。
- 64/64 token 覆盖，无 OOM/traceback，结束后进程、端口和 GPU 回收。
- reward 并发优化必须逐样本指标等价；任何缓存实现只要出现指标漂移就拒绝。
- batch/token 参数若改变生成结果，只能作为新协议候选，不能直接应用到 E1；采用前须重跑 E0。

通过后的动作：优先应用不改变生成协议的 reward 并发优化；是否改变 batch/token 参数由速度收益与重跑 E0 的成本共同决定。

### E0. Stage-2 冻结 dev baseline

目标：建立所有后续训练方法的同协议 dev 基线。

验收：566 个唯一 dev token、parse success、完整 reward 组件、无 clipping、退出与资源回收正常。

### D0. 冻结 train rollout 诊断

目标：判断 reward 方差、探索空间和可学习 headroom，为 E1–E4 的具体行为提供数据依据。

验收：

- 4,525 个 train token，每个恰好 4 个 rollout，总计 18,100 行；
- dev/held-out token 数均为 0；
- `diagnosis.json` 成功生成；
- 报告 exact-zero std、low-nonzero std、reward/headroom、pairwise ADE/FDE、parse rate 和 safe rate。

自适应决策：

- 根据非零方差和 headroom 分布决定 E2 的 FALS budget/排序范围，不预设阈值结论。
- 若 parse failure 显著，先修正格式/生成问题并重跑 D0，不让解析失败主导 FALS。
- 若绝大多数 group 零方差，记录窄策略证据；E1 仍作为必要 vanilla 对照，E2 优先选择有方差且有 headroom 的场景。

### E1. Vanilla LoRA-GRPO

目标：建立普通 GRPO 后训练对照。

固定项：冻结的 train 1k manifest、250 steps、相同生成/验证协议、相同 LoRA 和 reward。

验收与分析：保存 checkpoint/rollout/final dev；与 E0 比较 PDMS scaled、PDMS、safe rate、collision、drivable area、progress、TTC、comfort、parse 和 clipping。若 A0 改变生成协议，先重跑 E0 后再比较。

### E2. FALS only

目标：只改变样本选择，验证 failure-aware sampling 的独立贡献。

固定项：训练预算、step、reward、LoRA、生成和 dev 协议与 E1 一致。FALS manifest 只能由 D0 train rollout 生成。

决策：根据 D0 排名分布选择与 E1 等预算的主实验；若需要第二预算，只能作为预先记录的补充消融，不能用 held-out 选择。

### E3. SLDR only

目标：只改变训练 reward 为 SLDR，验证 safety-dense reward 的独立贡献。

固定项：使用与 E1 相同的随机 train 1k manifest 和训练预算；不得同时引入 FALS 或 std-floor。

分析：除 dev 总分外，重点检查 unsafe rollout 排序、safe rate、collision、drivable area，以及 group reward std 分布。

### E4. Std-Floor GRPO

目标：验证 std floor 对低但非零 group 方差的稳定作用。

启动门控：E3 rollout 中至少 10% group 满足 `0 < std < 0.05`。否则按计划跳过。

固定项：除 advantage estimator 和 `std_floor=0.05` 外，其余变量与对应对照保持一致。

### E5. grouped reward throughput

目标：在最终训练配置上验证 reward 服务吞吐优化，形成可复用的生产配置。

验收：固定输入逐样本 reward 指标完全一致、无请求丢失/重排、吞吐和 p50/p90 latency 有重复测量、资源回收正常。A0 的 64-token 结果只能作为候选筛选，不能替代此正式验证。

### F0. 最终审计与 held-out

1. 仅使用 dev 结果确定最终候选和 checkpoint。
2. 冻结代码、配置和 checkpoint 后，对 held-out 做一次性评估。
3. 汇总 E0–E5 的效果、吞吐、显存、失败记录和适用边界。
4. 核查所有正式实验可追溯到 source commit、manifest 和原始产物后，关闭计划。

## 5. 执行记录

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

## 6. 后续记录模板

每个新结果按以下格式追加，不改写历史事实；“当前快照”同步更新：

```text
### 记录 NNN：<阶段与事件>

- 状态：通过 / 失败 / 证据不足 / 按门控跳过
- 代码与配置：<commit、关键参数>
- 原始证据：<远程实验目录和文件>
- 覆盖与完整性：<manifest、行数、退出码、资源回收>
- 关键结果：<指标、耗时、显存>
- 分析：<结果说明、限制、是否同协议>
- 决策：<推进、重试、回退、跳过或重跑 baseline>
- 下一动作：<唯一明确动作及启动门控>
```
