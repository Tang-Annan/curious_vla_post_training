# Curious-VLA Dataset V3：干净重启、数据制作与 Selector × Reward 预备台账

> 生效日期：2026-08-27（Asia/Shanghai）。
> 当前状态：`S1_COMPLETE / SELECTORS_FROZEN / R0_COMPLETE / H0_PROTOCOL_FROZEN`；保留现有 `models/sft_stage2`，将 118 个 SFT-unseen logs / 835 个 eligible scenes 全部保留给 Dev/Final，并只从 1,192 个 SFT-seen logs / 103,288 个 SFT tokens 构建受控、可审计的 GRPO train-side pool；Random/TailMix 各 2,000-token manifest/parquet、唯一 CDT `R_task` 公式及 H0 train-only pilot 协议均已冻结并验收，尚未启动正式四格 optimizer training 或访问 Final。
> 本文是 Dataset V3 的执行入口。Dataset V2、G4、CDT-HLA 等旧台账只作为历史证据，不再接管新实验。

## 1. 重启目标与当前冻结结论

本次“完全重启”包含三个相互独立的重启边界：

1. **代码重启**：从固定的干净基线重新建立执行分支，不继承旧 GRPO、selector、reward、HLA 或 Dataset V2 实验逻辑；已验证的通用基础设施修复允许经逐项 diff 审计后迁入；
2. **数据重启**：重新盘点原始数据并按 log 划分；GRPO train-side 明确复用 SFT-seen logs，Dev/Final 严格使用 SFT-unseen logs，两个来源宇宙在 log 层级无重叠；
3. **实验重启**：旧 V2 指标只作历史参考；SFT-E0、Random-Raw 和全部新方法必须在 V3 上重新执行。

已经冻结的科学结论：

- 本轮冻结 `Reuse-SFT + Controlled GRPO Overlap`，不冻结 `Retrain-SFT`，也不等待新增 logs；
- 保留现有 `models/sft_stage2`；它是零更新锚点，回答“GRPO 是否值得”；
- 118 个 SFT-unseen logs / 835 个 eligible scenes 全部只用于 Dev/Final，不得进入任何训练侧筛选、rollout、replay、monitor 或 optimizer manifest；
- GRPO train-side 只能从已审计的 1,192 个 SFT-seen logs / 103,288 个 SFT tokens 构建，并冻结 exact token/log reuse 与 overlap 报告；
- Random-Raw GRPO 是 selector/reward 的 primary matched baseline，回答“方法是否有效”；
- 首轮只研究 selector 与 scalar reward，不恢复 SLDR 或 CDT-HLA advantage 注入；
- 旧 CDT L0–L3 定义、validity 边界和统计方法可以作为 V3 候选定义，但必须在新数据上重新审计；
- primary Tail Dev/Final 必须优先由模型无关的场景或 evaluator/cache 属性定义，不能直接按 SFT rollout 失败挑选；
- Random 与 TailMix 必须使用完全相同的 driving-intent quota；
- 正式 interaction 结论必须来自 RR/TR/RC/TC 四格完全一致的 matched seeds；
- V3 不直接继承 V2 的训练超参数；在新数据、selector 和 reward geometry 冻结后，先用 train-only monitor 校准一套四格共用配置；
- 不从旧分支搬运实验处理逻辑，不删除或覆盖旧证据。

Dev/Final 具体 log 分配、Natural/Tail 几何和每格 2,000-token 训练规模已由 D0R-2/D0S/D0F 冻结；TailMix 四类内部比例仍由 S1 的 train-only Rollout Bank 证据决定，不等待新增 logs，也不读取 Dev/Final。

## 2. Git 与历史证据边界

### 2.1 分支与目录固定

| 项目 | 固定值 / 当前值 | 状态 |
| --- | --- | --- |
| 历史证据分支 | `codex/grpo-g4-execution` | 保留，不清空、不改写历史 |
| 历史封存 commit | `def56881179618efbbef0cadb92f14916feca6c2` | 已推送 `post-training/codex/grpo-g4-execution` |
| 干净代码基线 | `93937eb01905aa5f3983a6a3600fa970ba50ad8b` | `origin/main` 已不再发布；已按完整 SHA 从 origin 重新 fetch 并验证 object |
| V3 执行分支 | `codex/grpo-v3-selector-reward` | D0F source 为 `b46ffdf`；S1 Candidate/Confirm 协议 source 为 `a9ea2c5`；本地、服务器与 `post-training` 在该协议提交上一致 |
| 可写远端 | `post-training` | 新分支只推送到该远端 |
| V3 数据命名空间 | `dataset_v3_controlled_overlap` | 正式命名；明确训练侧受控复用 SFT-seen 数据 |
| V3 实验命名空间 | `experiments/dataset_v3_controlled_overlap/` | 固定名称 |
| 服务器新源码目录 | `/root/autodl-tmp/curious-vla-workspace/src/curious_vla_v3` | 已独立 clone；source clean；当前 `b46ffdf` |

`93937eb` 是本文创建时由 `origin/main` 指向的已知干净基线，而不是自动跟随远端默认分支。执行时远端已不再发布 `main`，因此按完整 SHA 重新 fetch 并固定该 object；没有改用当前默认分支。如果后续决定改用另一个上游 commit，必须先在本文登记新 commit 和理由，不得静默换基线。

D0I 已产生的 `dataset_v3_sft_unseen` 路径是历史 inventory 证据，保留原位且不改写；从 D0S 开始的新正式资产统一写入 `dataset_v3_controlled_overlap`，避免把受控训练重用误称为全数据 SFT-unseen。

### 2.2 历史分支封存

创建 V3 分支前先完成一次只针对历史记录的封存：

1. 检查当前未提交的 `dataset_v2_grpo_experiment_ledger.md` 与 `next_route_execution_handoff.md`；
2. 只提交已经完成实验的真实记录，不暂存 bundle、pytest 临时目录、artifacts 或非台账草稿；
3. 推送历史分支并记录最终 archive commit；
4. 后续不在历史分支实现 V3 代码；
5. 不执行 reset、clean 或递归删除来制造“干净重启”。

### 2.3 V3 允许带入的旧文档

V3 分支从干净基线创建后，只从最终 archive commit 恢复以下历史材料：

- `docs/post_training_execution_loop.md`；
- `docs/offline_preference_post_training_execution_loop.md`；
- `docs/grpo_g4_experiment_ledger.md`；
- `docs/dataset_v2_grpo_experiment_ledger.md`；
- `docs/dataset_v2_cdt_hla_experiment_ledger.md`；
- `docs/next_route_execution_handoff.md`；
- 本文 `docs/dataset_v3_restart_and_data_execution_ledger.md`。

不从旧分支带入：

- `projects/dataset_v2/`、旧 selector、旧 HLA estimator 和旧 reward 修改；
- Dataset V2 launcher、freeze marker、manifest、parquet、cache 或 rollout；
- 旧测试临时目录、bundle、本地 artifacts 和未纳入台账的分析草稿；
- 任何旧 checkpoint、数据 hash 或结果值作为 V3 的正式 baseline。

旧代码仍可在历史分支只读查阅。V3 的实验处理逻辑必须基于干净代码重新做最小实现；通用工程修复按下一节单独处理。

### 2.4 通用基础设施修复审计

“干净重启”要求实验处理变量干净，不要求重新制造已经定位的基础设施故障。以下历史修复可作为 V3 候选补丁：

| 候选 commit | 内容 | V3 处理 |
| --- | --- | --- |
| `683c05c` | 显式加载本地 parquet | 审计后可迁入通用 loader 修改 |
| `69559e3` | `val-only + rollout.n=1` 校验修复 | 审计后可迁入 trainer 修改及对应测试 |
| `e514640` | dev access lock | 只迁入通用 lock 语义；不搬 Dataset V2 launcher/path |
| `2118555` | 训练日志验证器修复 | 复用已验证判定规则；不搬 Dataset V2 pipeline |

迁入规则：

1. 逐个查看 diff，证明修改只解决通用 loader/evaluation/lock/validator 问题；
2. 与 Dataset V2 路径、manifest、selector、reward、CDT/HLA 耦合的部分不得迁入；
3. 纯通用 commit 可以 cherry-pick；混合 commit 只在 V3 中重做必要的最小修改；
4. 每个补丁保留对应 focused test，并在 V3 初始化记录中写明来源 commit；
5. 不允许以“基础设施修复”为名引入旧实验默认值或 fallback。

### 2.5 V3 分支创建与服务器同步顺序

执行顺序固定：

1. 完成历史分支封存并得到 `<ARCHIVE_COMMIT>`；
2. fetch 远端并验证 `93937eb`；
3. 从 `93937eb` 创建 `codex/grpo-v3-selector-reward`；
4. 从 `<ARCHIVE_COMMIT>` 只恢复第 2.3 节文档；
5. 按第 2.4 节审计并迁入通过的通用修复；
6. 运行对应 focused tests，提交 V3 初始化 commit，记录 branch、base、archive commit、补丁来源和 source status；
7. 推送 V3 分支；
8. 服务器在新目录重新 clone，checkout V3 分支；
9. 服务器确认 source clean 后才允许开始数据盘点。

服务器旧训练 checkout 和旧实验目录保持不动。V3 不复用旧源码目录，也不覆盖当前正在运行的其他任务。

## 3. Dataset V3 数据契约

### 3.1 数据集要回答的问题

Dataset V3 首先服务于两个目标：

1. 在严格 SFT-unseen 的 Dev/Final 上，测量受控重用训练数据的 GRPO 相对 SFT 的 post-training 增量；
2. 在同一个冻结的 SFT-seen train-side 宇宙中，用不同 selector manifest 稳定暴露可学习的安全尾部事件。

V3 不要求立即更换到外部驾驶数据集，也不等待新增 logs。当前数据由互斥的两个来源宇宙组成：SFT-seen 只服务训练侧，SFT-unseen 只服务严格评估侧；两者保持相同的图像输入、动作表示、metric cache 和 evaluator。

### 3.2 最小数据层级

V3 只维护一个统一 Master Index，其中训练与评估是互斥的来源分区；每个 selector 只增加 manifest，不复制底层 parquet/image/cache：

```text
1,192 SFT-seen logs / 103,288 tokens        118 SFT-unseen logs / 835 scenes
                 ↓                                           ↓
   Controlled Train Master Pool                  Strict Eval Reserve
                 ↓                                           ↓
      Frozen SFT Rollout Bank                     Dev / Final only
                 ↓
 Random / TailMix / 后续 selector manifests
```

每次测试新 selector，通常只生成新的 sample-ID manifest。只有 selector 需要 Rollout Bank 中不存在的指标时，才为同一 Master Pool 补算字段或 rollout；不得重新划分 dev/final。

### 3.3 全局 split

原始日志必须在制作 token/parquet 前完成 split。逻辑 split 固定为：

| Split | 作用 | 允许的访问 |
| --- | --- | --- |
| `sft_provenance` | 记录 SFT 实际使用的 103,288 tokens / 1,192 logs | 模型溯源和受控训练来源边界 |
| `grpo_screen` | 从 SFT-seen 来源构建 selector 初筛和确认 rollout | train-only |
| `train_monitor` | 训练预算、LR 和 estimator 的固定校准集 | train-only；不进入正式 optimizer manifest |
| `dev_natural` | 正常分布方法选择 | 允许在预注册方法完成后评估 |
| `dev_tail` | 模型无关安全尾部方法选择 | 允许在预注册方法完成后评估 |
| `final_natural` | 最终正常分布确认 | 候选、seed 和门控冻结前禁止访问 |
| `final_tail` | 模型无关安全尾部最终确认 | 候选、seed 和门控冻结前禁止访问 |

必须始终成立：

- `dev_natural/dev_tail/final_natural/final_tail` 的并集覆盖全部 118 个 SFT-unseen logs / 835 个 eligible scenes，并且与 SFT provenance 的 token、log overlap 为 0；
- `grpo_screen`、`train_monitor`、Random/TailMix optimizer manifest 只能来自 SFT provenance；其 SFT overlap 是预注册的受控重用，不得写作 unseen，必须报告 exact unique token/log reuse、选择率与 per-log cap；
- `grpo_screen`、`train_monitor`、两个 dev、两个 final 之间 token、log overlap 均为 0；
- 所有训练侧 selector rollout、reward replay、H0 pilot 和正式 optimizer manifest 与全部 Dev/Final 的 token、log overlap 均为 0；
- 同一连续事件窗口不得跨 split；事件窗口冻结为 14 frames（4 history + 10 future），stride 14，且 center frame 必须具有 route；
- final 不参与 selector 设计、reward 参数、训练预算和非劣界限的确定；
- 所有 prompt 时间描述使用同一正确版本，不重新引入 V2 已发现的 5-second/4-second 不一致。

### 3.4 Tail Dev/Final 构造边界

primary `dev_tail/final_tail` 必须优先根据模型无关属性构建，并在任何 SFT/GRPO evaluation 前冻结。候选证据只能来自原始场景、标注或 evaluator/cache，例如：

- agent interaction、route conflict 或 safety-critical maneuver；
- 场景几何、curvature、obstacle proximity；
- 可由参考轨迹/场景状态得到的 TTC、collision potential 或其他风险 margin；
- 与被评估 policy 输出无关的正式安全字段。

D0I 必须先审计这些字段是否真实存在、含义是否可用于 scene risk，不能为了凑 Tail split 发明代理标签。

如果现有数据无法形成模型无关 Tail split，允许使用冻结 SFT rollout 构建单独的 `dev_sft_challenge/final_sft_challenge`，但必须满足：

- 名称和报告始终写作 **SFT-baseline challenging tail**；
- 只能说明模型相对 SFT 已知困难场景的改善，不能声称代表真实 safety-tail distribution；
- 不替代 `dev_natural/final_natural`；
- 选择规则在 GRPO 训练前冻结，GRPO 结果不得反向改变其成员。

`TAIL_EVAL_ROUTE=POLICY_INDEPENDENT_GT_ACTOR_PROXIMITY` 已冻结：scene flag 为 vehicle distance `<=5.0 m` 或 pedestrian/bicycle distance `<=10.0 m`；在 58 个含 eligible scene 的 logs 中，按 interaction rate、interaction count、minimum actor distance 和 stable hash 排序，前 29 个定义为 Tail，其余 29 个为 Natural；60 个零 eligible-scene logs 只保留在 Natural log reserve，不产生 token。Tail/Natural interaction scene rate 分别为 `83.2%/47.9%`。

### 3.5 SFT 路线决策

D0 首先重建 SFT provenance：

- SFT checkpoint 的精确路径与 hash；
- SFT 训练 parquet/manifest 的 hash；
- 实际 token 数、唯一 log 数和 log-ID blacklist；
- 无法追溯的样本数量及原因。

D0I 证明严格 unseen 容量不足以同时承担原定 GRPO train 与 Dev/Final，但足以作为宝贵的严格评估保留集。D0R 因此冻结第三条路线：

1. **Reuse-SFT + Controlled GRPO Overlap（已选）**：保留 `models/sft_stage2`；将全部 SFT-unseen 数据只分配给 Dev/Final；GRPO train-side 只从 SFT-seen provenance 构建受控、可审计的训练池；
2. **Retrain-SFT（本轮不冻结）**：不作为当前执行路线，也不以等待新增 logs 作为 D0R/D0S 前置条件。

`SFT_ROUTE=REUSE_SFT_CONTROLLED_GRPO_OVERLAP` 已冻结。Dev/Final 的具体 log 分配、Natural/Tail 几何和 train manifest 规模已由 D0R-2/D0S/D0F 冻结；正式 SFT-E0、selector rollout 和训练仍未启动。

该路线的结论边界固定为：允许报告“受控重用 SFT-seen 数据进行 GRPO 后，在严格 SFT-unseen Dev/Final 上相对 SFT 的 post-training 增量”；不得声称 GRPO train 数据本身 unseen、获得了新增 logs，或证明了新数据效率。

### 3.6 容量目标而非冻结规模

以下数字已经由 D0R-2/D0S/D0F 冻结：

| 资产 | 初始容量目标 | 冻结状态 |
| --- | ---: | --- |
| Selector Screen Pool | 8,000 tokens / 1,063 SFT-seen logs；per-log cap 8 | `FROZEN` |
| Train-only Monitor | 256 tokens / 129 disjoint SFT-seen logs；per-log cap 2 | `FROZEN` |
| Random Train Manifest | 2,000 unique tokens；straight/left/right=`1333/434/233` | `FROZEN` |
| TailMix Train Manifest | 2,000 unique tokens；与 Random 使用完全相同 intent quota | `FROZEN_TARGET / MEMBERSHIP_AFTER_S1` |
| Natural Dev + Tail Dev | 416 scenes；Natural/Tail=`210/206` | `FROZEN` |
| Final Natural + Final Tail | 419 scenes；Natural/Tail=`214/205` | `FROZEN / LOCKED` |

评估规模只能根据 118 个 SFT-unseen logs 的 intent 分布、连续事件去重后容量、image/cache 可用性和 safety-conflict coverage 确定；训练规模只根据 1,192 个 SFT-seen logs 的受控候选容量、rollout 成本和完整 2×2 matched-seed 成本确定。D0R 的问题不是“能否凑到 8K”，而是“满足预定 safety-conflict coverage 所需的最小训练规模是多少”。不得通过重复 rare token 填满目标；必须报告 Screen→Train 选择率，避免 selector 覆盖接近全候选池而失去对比分布。

### 3.7 Master Index

数据制作必须产生一个 V3 单一索引，至少包含：

- `token_id`；
- `log_id`；
- scene/时间定位字段；
- split；
- driving intent；
- prompt/version；
- CAM_F0 相对路径；
- metric-cache 相对路径；
- SFT overlap 标记；
- 数据有效性状态。

selector 指标、rollout 统计和 reward 结果不重复写入底层 parquet；它们保存在 Rollout Bank/report，并通过 `token_id` 关联。

### 3.8 数据资产路径

建议固定为：

```text
data/dataset_v3_controlled_overlap/hf/
data/dataset_v3_controlled_overlap/sensor_blobs/
manifests/dataset_v3_controlled_overlap/
experiments/dataset_v3_controlled_overlap/data_build/
experiments/dataset_v3_controlled_overlap/rollout_bank/
```

正式路径在 V3 分支创建后写入配置，不允许自动 fallback 到 Dataset V2、旧 5,656 数据或旧 566/2,000-token dev。

## 4. 数据制作执行阶段

### 4.1 `V3-D0I`：只读 inventory

目标：在不生成训练数据的情况下回答 V3 的严格 SFT-unseen 评估容量、SFT-seen 受控训练容量和可行路线。

输出：

- 原始数据源、row/token/log 数量；
- SFT token/log blacklist；
- 排除 SFT 后的 unique token/log 数；
- intent、log 长度、区域和时间分布；
- 可用于模型无关 Tail split 的场景/evaluator/cache 字段、值域和覆盖率；
- 图像与 metric-cache 可获得性；
- 相邻事件去重后的容量；
- `Reuse-SFT`、`Retrain-SFT` 与 controlled-overlap 路线的容量证据；
- 建议的正式 split 数量，但不创建 final 内容。

通过条件：SFT provenance 可追溯，SFT-seen 与 SFT-unseen 的 token/log 边界可精确复现，并且 D0R 能据此冻结一条诚实、可执行的训练/评估路线；同时明确 `TAIL_EVAL_ROUTE` 的可行候选。D0I 本身只提供证据，不生成 split 或 selector/reward 代码。

### 4.2 `V3-D0S`：split 与基础 manifest

在 `SFT_ROUTE`、Dev/Final 几何和规模冻结后执行：

1. 先锁定全部 118 个 SFT-unseen logs 为 Strict Eval Reserve，再使用固定 seed `20260827` 按 log 分配 Dev/Final；训练侧只读取 SFT-seen provenance；
2. 对连续事件窗口去重；
3. 预留固定 `train_monitor` logs，后续不得进入 Random/TailMix optimizer manifest；
4. 在各 split 内按 intent 做确定性抽样；
5. 按第 3.4 节冻结的模型无关规则构建 primary Tail split；若不可行则使用明确重命名的 SFT-challenge 路线；
6. 生成 Master Index 与基础 manifests；
7. 生成 overlap、SFT token/log reuse、Screen→Train 选择率、分布、Tail 定义和容量报告；
8. final 只生成不可读内容锁和 hash，不进入后续常规入口。

此阶段不根据 SFT rollout、PDMS、CDT tier 或难度挑选 GRPO 训练样本。除已经选择并明确命名的 SFT-challenge 路线外，也不得根据 SFT rollout 构建 primary Tail evaluation。它只建立候选宇宙和评估边界。

### 4.3 `V3-D0A`：image/cache 资产

仅为已经冻结的 active token union 生成或链接：

- CAM_F0；
- NAVSIM metric cache；
- loader 所需 parquet/index；
- image/cache coverage report。

Random、TailMix 和后续 selector 共用同一份资产。不得按 selector 建多份图像或 cache。

### 4.4 `V3-D0F`：数据冻结

正式冻结至少输出：

- `dataset_card.json`；
- `master_index`；
- split manifests；
- `sft_provenance_report.json`；
- `overlap_report.json`；
- `distribution_report.json`；
- `tail_definition_report.json`；
- `asset_coverage_report.json`；
- source/data/model hash；
- `V3_DATA_FROZEN`；
- `COMPLETE` 和 `exit_code`。

技术门控：

- 所有 token 唯一且可回到唯一 log/scene；
- Dev/Final 与 SFT provenance 的 token/log overlap 为 0，训练侧与 Dev/Final 的 token/log overlap 为 0；
- GRPO train-side 与 SFT provenance 的 overlap 精确等于冻结 manifest 声明，且 unique token/log reuse、选择率和 per-log cap 可审计；
- image/cache coverage 为 100%；
- prompt/version 一致；
- primary Tail 的每个选择字段均能证明与被评估 policy 输出无关；若采用 SFT-challenge，split 名称、报告和结论边界全部一致；
- manifest 重跑 membership/order 一致；
- source clean，报告 finite，无 silent fallback；
- final access lock 生效。

任一项失败只修数据或入口并创建 `retryN`，不得启动 rollout 或训练。

## 5. V3 Selector × Reward 预备设计

本节只冻结实验骨架，不在数据完成前冻结具体 selector 配额、rollout 次数或 reward 数值。

### 5.1 研究问题

1. 在相同 Raw-PDMS reward 下，TailMix 是否优于 Random；
2. 在相同 Random manifest 下，CDT scalar reward 是否优于 Raw-PDMS；
3. TailMix 是否通过提高可学习 safety-conflict 的暴露，使 CDT reward 获得额外收益；
4. 完整 GRPO 路线是否相对 SFT-E0 改善安全尾部且保持正常驾驶质量。

### 5.2 最小矩阵

| ID | Selector | Reward | 主要作用 | 当前状态 |
| --- | --- | --- | --- | --- |
| `V3-E0-SFT` | 无训练 | 无 | 零更新锚点 | `BLOCKED_BY_M0` |
| `V3-RR` | Random | Raw-PDMS | primary GRPO baseline | `BLOCKED_BY_E0` |
| `V3-TR` | TailMix | Raw-PDMS | selector 主效应 | `BLOCKED_BY_E0` |
| `V3-RC` | Random | CDT scalar reward | reward 主效应 | `BLOCKED_BY_E0` |
| `V3-TC` | TailMix | CDT scalar reward | selector × reward 协同 | `BLOCKED_BY_E0` |

直接 contrast：

\[
\Delta_{selector}=V3\text{-}TR-V3\text{-}RR
\]

\[
\Delta_{reward}=V3\text{-}RC-V3\text{-}RR
\]

\[
\Delta_{interaction}=(V3\text{-}TC-V3\text{-}TR)-(V3\text{-}RC-V3\text{-}RR)
\]

SFT-E0 不替代 Random-Raw。所有最终候选同时报告相对 SFT-E0 与 Random-Raw 的差值。

seed 协议固定为两阶段：

1. RR/TR/RC/TC 四格首先全部运行 discovery seed `20260827`；执行优先级固定为 `RR → TC → TR → RC`，先取得 primary baseline 与完整方法端点，再补齐两个主效应 cell；该顺序不允许省略 TR/RC，也不改变四格 matched-seed 要求；
2. 只有达到 M0 预注册 promotion gate 的主效应或 interaction 才进入确认；
3. 如果正式声称 selector × reward interaction，四格必须全部补齐完全相同的三个 matched seeds；计划 seed 为 `20260827/20260828/20260829`；
4. interaction 必须先在每个 matched seed 内计算，再汇总 seed 间结果；不得用不同 seed 集合的 cell 均值相减；
5. 未补齐四格 matched seeds 时，interaction 只能标记为 exploratory，不得形成正式协同结论。

### 5.3 Random selector

当前可冻结内容：

- 从同一 `grpo_screen` 候选宇宙选择；
- 使用固定 seed 和 stable hash；
- 样本唯一；
- 按 intent 分层并设置 per-log cap；
- 与 TailMix 使用完全相同的 straight/left/right quota；
- 与 TailMix 样本数、训练 step、group/rollout 预算一致。

精确 intent quota 冻结为 straight/left/right=`1333/434/233`，样本数为每格 2,000，训练候选 per-log cap 为 8；Random/TailMix 同时报告 region、route-type 和有效 log 分布及 JS divergence；是否增加弱分布门槛在 M0 冻结，不要求逐项完全匹配。

### 5.4 TailMix selector

TailMix 首轮只包含四类语义：

- 可复现的严重事件；
- mixed-tier、困难但可恢复事件；
- near-risk 事件；
- 正常分布 Random Anchor。

selector 只能读取 train-side SFT Rollout Bank，不读取 dev/final。FALS 的 difficulty/headroom 可作为“困难但可恢复”特征，不再独立代表安全尾部。

当前流程固定为 `Screen → Candidate → Confirm → Select`：

1. 对 Screen Pool 做共享 SFT `G=4` 广筛；
2. 只有出现安全风险、mixed-tier 或高 headroom 的 token 进入 Candidate；
3. 对 Candidate 使用额外的独立 rollout block 做 Confirm；
4. severe、mixed-recoverable 和 near-risk 只能根据 Confirm 后的频率/跨 block 稳定性归类，不能因单个 G4 failure 直接入选；
5. 从确认 bank 构建 TailMix manifest；
6. Random 与 TailMix 复用同一 Master Pool 和 Rollout Bank。

Confirm 固定为 Screen seed `20260827` 与 Confirm seed `20260828` 两个独立 `G=4` block，总 `G=8`。稳定类别必须在两个 block 均至少出现一次对应事件，按以下互斥优先级归类：

1. `stable_severe`：两个 block 均至少有一个 L0/L1；
2. `stable_mixed_recoverable`：两个 block 均同时含 L3 与至少一个 L0–L2，且各自至少含两个 valid tiers；排除已归入 `stable_severe` 的 token；
3. `stable_near_risk`：两个 block 均至少有一个 L2；排除前两类；
4. `random_anchor`：其余冻结 Screen tokens。

容量审计后冻结 TailMix class×intent 精确配额如下；列顺序为 straight/left/right：

| 类别 | straight | left | right | 合计 |
| --- | ---: | ---: | ---: | ---: |
| `stable_severe` | 258 | 159 | 161 | 578 |
| `stable_mixed_recoverable` | 29 | 35 | 4 | 68 |
| `stable_near_risk` | 1 | 6 | 0 | 7 |
| `random_anchor` | 1,045 | 234 | 68 | 1,347 |
| 合计 | 1,333 | 434 | 233 | 2,000 |

不足类别处理规则冻结为：保留全部 653 个双 block 稳定事件，不重复、不上采样、不放宽类别定义；只用按 seed `20260827` 稳定哈希选择的 `random_anchor` 补足各 intent 缺口。若任一稳定类别容量与审计值不同，Select 硬失败而非重新分配。

### 5.5 Reward

Raw baseline 当前冻结为：

\[
R_{raw}=q,\qquad q=\operatorname{clip}(\text{raw-PDMS},0,1)
\]

CDT scalar reward 只保留以下结构性约束：

- 每个 valid completion 独立计算 scalar reward；
- `L0 < L1 < L2 < L3`；
- 不把 CDT tier 直接写入 advantage estimator；
- 同一 tier 内继续使用连续质量项排序；
- homogeneous-tier group 退化为连续质量学习，mixed-tier group 才产生跨 tier 排序；
- invalid 与正式 safety tier 分离，沿用统一 parser 技术门控。

候选形式为：

\[
R_{CDT}=k(L)+\beta q,\qquad k(L0)<k(L1)<k(L2)<k(L3)
\]

R0 已在 train-side Rollout Bank 完成 CPU geometry replay，并冻结严格不重叠区间、`Q_task` 质量项、`\beta=1/7` 与 tier 间距；没有使用 dev 调参。正式协议与证据见记录 V3-011。

为避免 safety 重复计分，CPU replay 至少比较但不正式训练两种候选：

\[
R_{PDMS}=k(L)+\beta\operatorname{PDMS}
\]

\[
R_{task}=k(L)+\beta Q_{task}
\]

其中 `Q_task` 只能由 evaluator 中未被 CDT tier 重复表达的任务质量项构成；可用分项、定义和值域由 D0/R0 审计。如果无法得到语义完整的 `Q_task`，必须如实报告 PDMS 的 safety double-count，而不是临时拼接新指标。R0 只根据 train-side geometry 冻结唯一公式，正式 2×2 不同时训练两个 CDT reward 版本。

R0 候选审计在执行前固定为相同的不重叠区间：`L0/L1/L2/L3` 分别位于 `[0,1/7]`、`[2/7,3/7]`、`[4/7,5/7]`、`[6/7,1]`，即 `k(L)=2L/7`、`beta=1/7`。`R_PDMS` 使用 `q=clip(pdms,0,1)`；`R_task` 使用生产 PDMS 在移除 Collision、DAC、TTC 后的完整剩余项：

\[
Q_{task}=\frac{5\,ego\_progress+2\,history\_comfort}{7}.
\]

决策门槛预先固定为：若两项 task metric 对全部 valid rollout 均完整、finite 且位于 `[0,1]`，`R_task` 的 empirical cross-tier inversion/tie 为 0、within-tier quality inversion/tie 为 0，且 Random/TailMix 的 EffectiveGroupRate 均不低于对应 Raw-PDMS，则冻结 `R_task`；若仅 task metric 语义完整性失败才允许退回 `R_PDMS` 并显式报告 safety double-count；任一排序或 finite 技术门禁失败则关闭 R0，不启动训练。

### 5.6 公平比较不变量

- 所有 GRPO run 从同一 SFT checkpoint hash 初始化；
- reward 对比使用字节级相同的 selector manifest 和顺序；
- selector 对比使用相同 reward、样本数、完全相同的 intent quota、训练 step 和 rollout 预算；
- optimizer、LoRA、KL、clip、batch、decode 和评估协议相同；
- 正式四格使用 H0 冻结的同一套 LR、advantage estimator、group batch、training group budget、PPO epoch 和 LoRA 配置；
- V3 固定 `data.shuffle=true`、固定 seed，并要求相同 manifest 的 reward 对照得到相同实际样本顺序；shuffle 只打乱单遍顺序，不改变样本使用次数；
- 每个 run 必须保存并验收实际 `experiment_config.json`；resolved config 与台账/launcher 任一关键项不一致时按技术失败处理；
- 四格 matched-seed 规则在 M0 冻结，seed 身份与数量不得按单个 cell 的结果补齐；
- 旧 V2 E0、Random-SDR、Random-Raw 数值不得填入 V3 结果表；
- Final 只在方法、seed 和晋级规则冻结后访问一次。

### 5.7 `V3-H0`：train-only 优化参数校准

#### 5.7.1 V2 证据的正确解释

V2 正式 resolved config 为：4 groups/update、`G=4`、16 sequences/update、`ppo_epochs=1`、`total_epochs=1`、250 updates、constant LR `1e-6`、LoRA rank 8 attention-only、KL coefficient `0.01`。固化的 `experiment_config.json` 显示实际 `shuffle=true`，因此它是 V3 参数审计的权威事实；旧 launcher/台账中的 `shuffle=false` 不能继续作为实际配置证据。

V2 的低 KL、零 clip fraction、稳定 entropy 和较小 grad norm支持“更新偏保守”假设，但不能单独证明 LR 必须升高。尤其 V3 若使用 2K–4K manifest，在 batch 4、单遍条件下自然会产生约 500–1000 updates，已经是 V2 的 2–4 倍；因此不得把 `LR=3e-6` 直接冻结为 V3 默认值。

普通 GRPO 与 Std-Floor 的实现边界必须明确：

- 普通 GRPO 使用 group std 标准化；
- `std_floor_grpo` 使用 `max(std, floor)`，只抑制低非零 std 的噪声放大；
- exact-zero group 在两种 estimator 下都保持零 advantage；
- exact-zero/mixed-tier coverage 优先由 selector、reward 和必要时的 train `G` 解决，不能由 std-floor 冒充新增信号。

#### 5.7.2 参数优先级

V3 按以下顺序校准，前一层未确认前不进入后一层：

1. **信号 geometry 与 train `G`**：先用 S1/R0 报告 RR/TR/RC/TC 的 exact-zero、low-nonzero、mixed-tier 和 EffectiveGroupRate；`G=4` 为首选，只有预注册信号门槛失败时才讨论 `G=8`；
2. **training group budget 与 LR**：把处理过的独立 groups 和总 rollout queries 作为预算，不用固定 250 steps 代表不同数据规模；
3. **advantage estimator**：只有新 reward/selector 下仍存在大量低非零 std，才触发 `grpo` 与 `std_floor_grpo` pilot；
4. **groups/update**：最后比较 batch 4 与 8；按相同总 rollout queries 对齐时，它是 practical compute-matched configuration，不解释成纯 batch 因果效应；
5. **LoRA capacity**：只有策略更新已经健康但 train-monitor 明确平台化，才考虑 rank 或 target modules；
6. **KL coefficient**：V2 实际 KL 很小，首轮保持 `0.01`，只有 H0 出现明确约束证据才调整；
7. **micro batch**：只按显存设置，不进入科学消融。

`ppo_epochs=1` 首轮保持不变。增加 PPO epochs 会重复使用同一批 rollout，并非简单增加新鲜训练信号，不作为 under-training 的第一修复。

#### 5.7.3 最小 pilot

H0 使用从冻结 Random manifest 确定性截取的 `hparam_train` 做 optimizer update，并只在独立、固定、从不进入 optimizer 的 `train_monitor` 上选配置；两者都属于 train-side，不访问 dev/final。所有 pilot 从同一 SFT 初始化，使用 Random-Raw 作为中性校准路径；最后只冻结一套配置供四格共用，正式四格仍全部从原始 SFT 重新开始。

1. `H0-LR1`：standard GRPO、batch 4、首选 `G=4`、LR `1e-6`；
2. `H0-LR3`：除 LR `3e-6` 外与 `H0-LR1` 相同；
3. 在相同 processed-group/rollout budget 下，对固定 monitor 在 0%、20%、40%、60%、80%、100% 预算点生成评估；不得用不同 batch 的 raw step 编号对齐；
4. 根据 train-monitor PDMS/安全指标、KL、clip fraction、entropy、grad norm、parse、response clipping 和样本行为共同冻结 LR，不用训练 reward mean 单独选择；
5. 若 R0 预注册的 low-nonzero gate 触发，在选定 LR 上追加一次 `grpo` vs `std_floor_grpo`；否则保持 standard GRPO；
6. 只有仍有梯度方差证据时，才追加 batch 4 vs 8 的 compute-matched pilot；
7. 不同时测试 LR、estimator、batch、LoRA 和 KL，不根据 monitor 结果开启连续参数 sweep。

H0 已在任何 optimizer pilot 前冻结：`hparam_train=512` groups、独立 `train_monitor=256` tokens、`G=4`、4 groups/update；LR pilot 各处理 512 groups / 2,048 rollout queries，并在 step `0/26/51/77/102/128`（最近似 0/20/40/60/80/100% 的整 update 边界）评估同一 monitor。LR/estimator health 与 promotion gate 见记录 V3-012 及其冻结 protocol；不访问 dev/final 的边界不变。

## 6. 评价框架与待冻结门控

### 6.1 已确定的报告指标

Natural Dev/Final 至少报告：

- raw-PDMS 与项目现有 scaled score；
- Progress、Comfort；
- Collision、DAC、TTC；
- parse、clipping、non-finite；
- SFT-E0 和 Random-Raw 的 paired 差值。

Tail Dev/Final 至少报告：

- CDT L0/L1/L2/L3 rate；
- StrictClear；
- Collision 与 low-TTC；
- worst-tail/CVaR 指标；
- paired tier transition；
- log-cluster bootstrap CI。

训练侧至少报告：

- mixed-tier group rate；
- EffectiveGroupRate 与 exact-zero group；
- tier composition；
- reward/advantage 分布；
- headroom、parse 和 clipping。

### 6.2 数据完成后讨论并冻结

| 决策项 | 需要的新数据证据 | 当前状态 |
| --- | --- | --- |
| `SFT_ROUTE` | 1,192 SFT-seen logs / 103,288 tokens；118 SFT-unseen logs / 835 scenes | `FROZEN: REUSE_SFT_CONTROLLED_GRPO_OVERLAP` |
| split 精确规模 | 118 unseen logs 的 intent、Tail 字段和事件去重容量 | `FROZEN: Dev 416 / Final 419 scenes` |
| `TAIL_EVAL_ROUTE` | 模型无关场景/evaluator/cache 字段；必要时单独定义 SFT-challenge | `FROZEN: POLICY_INDEPENDENT_GT_ACTOR_PROXIMITY` |
| Screen Pool 与 train manifest 规模 | 1,192 SFT-seen logs / 103,288 tokens 的受控容量和 rollout 成本 | `FROZEN: Screen 8,000 / Monitor 256 / each cell 2,000` |
| TailMix 四类比例 | G4/G-confirm tier、headroom 与稳定性 | `FROZEN: 578/68/7/1,347` |
| selector 启动门槛 | Random 与 TailMix mixed-tier coverage | `FROZEN: Random/TailMix 2,000-token manifests` |
| CDT tier 是否沿用 V2 | 新 evaluator 字段和值域审计 | `FROZEN: canonical L0-L3, invalid separate` |
| CDT reward 区间和质量项 | train-only reward geometry | `FROZEN: R_task=(2L+Q_task)/7` |
| 主安全指标 | Tail Dev 事件数量和统计功效 | `TBD_AFTER_D0` |
| Natural non-inferiority margin | 历史 evaluator 方差、D0F split 规模和预期实际容忍度；不读取 V3 treatment dev | `TBD_AFTER_D0` |
| 训练步数和 train rollout `G` | manifest 规模、显存和 pilot 成本 | `TBD_AFTER_D0` |
| training group/rollout budget 与 LR | V3 manifest 规模和 H0 train-monitor 曲线 | `PILOT_PROTOCOL_FROZEN: 512 groups / 2,048 queries; LR pending` |
| advantage estimator | R0 low-nonzero geometry；std-floor 不处理 exact-zero | `TBD_AFTER_D0` |
| groups/update | H0 梯度方差与 compute-matched pilot | `TBD_AFTER_D0` |
| LoRA / KL | H0 policy movement 与平台化证据 | 默认 rank 8 attention-only / KL `0.01`，有证据才改 |
| 主效应确认 seeds | discovery 结果和计算成本 | `TBD_AFTER_D0` |
| interaction 确认 | 四格 discovery 结果和 matched-seed 成本 | `TBD_AFTER_D0`；正式结论固定三组 matched seeds |

`split 精确规模`、`TAIL_EVAL_ROUTE`、`Screen Pool 与 train manifest 规模` 在 D0R-2 冻结并由 D0S/D0F 验证；其余项目在 D0F 后按 S1、R0、H0、M0 对应门控逐项回填。未冻结前不得启动正式方法训练。

### 6.3 `V3-M0` 必须冻结的最小项目

M0 不重新发明算法，只把以下结论写成唯一正式协议：

- Random/TailMix 的相同 intent quota、样本数、per-log cap 和分布报告；
- Tail evaluation 的模型无关定义，或明确重命名的 SFT-challenge 边界；
- TailMix 的 Screen→Candidate→Confirm→Select 稳定性门槛；
- 唯一 CDT reward 公式和 train-only CPU replay 证据；
- H0 冻结的唯一 LR、advantage estimator、groups/update、training group budget、PPO epoch、LoRA、KL 和 shuffle 配置；
- resolved-config 验收规则与固定 train-monitor checkpoint；
- Natural primary metric；
- 唯一 Tail primary safety metric；
- Natural non-inferiority margin；
- discovery seed 的 promotion gate；
- matched multi-seed confirmation 规则；
- interaction 的四格 matched-seed 要求。

任一项未冻结，M0 不得标记 `COMPLETE`。

## 7. 最短执行队列

| 顺序 | ID | 动作 | GPU | 状态 |
| ---: | --- | --- | ---: | --- |
| 0 | `V3-P0` | 创建并按方案、参数审计修订本文 | 0 | `COMPLETE_REVISED_HPARAM` |
| 1 | `V3-A0` | 封存历史台账分支 | 0 | `COMPLETE` |
| 2 | `V3-B0` | 从 `93937eb` 创建并推送 V3 分支 | 0 | `COMPLETE` |
| 3 | `V3-B1` | 审计并迁入通用基础设施修复 | 0 | `COMPLETE` |
| 4 | `V3-S0` | 服务器新目录 clone、环境与基础入口检查 | 0 | `COMPLETE` |
| 5 | `V3-D0I` | SFT provenance、SFT-unseen inventory 与 Tail 字段审计 | 0 | `COMPLETE` |
| 6 | `V3-D0R-1` | 冻结 Reuse-SFT + Controlled GRPO Overlap 与双宇宙边界 | 0 | `COMPLETE / ROUTE_FROZEN` |
| 7 | `V3-D0R-2` | 冻结 Dev/Final 分配、Tail evaluation 定义和训练规模 | 0 | `COMPLETE / FROZEN` |
| 8 | `V3-D0S` | 生成 split、Master Index 和基础 manifests | 0 | `COMPLETE / RETRY2` |
| 9 | `V3-D0A` | 生成/链接 image 与 metric cache | 0 | `COMPLETE` |
| 10 | `V3-D0F` | 数据验收、hash 与 freeze | 0 | `COMPLETE / RETRY1` |
| 11 | `V3-S1` | SFT shared rollout bank、Confirm 与 selector manifests | 1 | `COMPLETE / SELECTORS_FROZEN` |
| 12 | `V3-R0` | 四格 reward/advantage geometry 与 CDT reward freeze | 0 | `COMPLETE / R_TASK_FROZEN` |
| 13 | `V3-H0` | train-only update budget、LR 与条件 estimator/batch pilot | 1 | `IN_PROGRESS / PROTOCOL_FROZEN` |
| 14 | `V3-M0` | 冻结 Selector × Reward、训练配置、指标和晋级规则 | 0 | `BLOCKED_BY_H0` |
| 15 | `V3-E0-SFT` | 在冻结 V3 dev 上生成零更新锚点 | `TBD` | `BLOCKED_BY_M0` |
| 16 | `V3-RR/TC/TR/RC` | 按 `RR → TC → TR → RC` 执行最小 2×2 discovery，并按门控补齐 matched-seed 确认 | `TBD` | `BLOCKED_BY_E0` |

不增加与当前主问题无关的算法分支。数据制作完成前的唯一执行路线为 `A0 → B0 → B1 → S0 → D0I → D0R-1 → D0R-2 → D0S → D0A → D0F`。

## 8. 记录模板

后续每次更新追加一条记录，不重写旧结果：

```text
### 记录 V3-NNN：<阶段与事件>

- 时间：
- branch / source commit / source status：
- 输入与 hash：
- 实际动作：
- 预期/实际数量：
- 技术结果：
- 科学结果：不适用 / 结果摘要
- 产物路径：
- 状态：COMPLETE / FAILED_TECHNICAL / CLOSED_BY_GATE
- 下一唯一动作：
```

### 记录 V3-001：历史封存、干净分支与通用补丁初始化

- 时间：2026-08-27 21:56–22:04（Asia/Shanghai）；
- branch / source commit / source status：历史分支 `codex/grpo-g4-execution@def5688` 已推送；V3 从固定基线 `93937eb01905aa5f3983a6a3600fa970ba50ad8b` 创建，初始化 commit `4c8cf73f5358d25398f212ab3c8f8cc690e6cb72`，source clean；
- 输入与 hash：创建分支前按完整 SHA 从 origin 重新 fetch `93937eb`；GitHub 当前不再发布 `main`，默认 HEAD 为 `codex/post-training-analysis`，因此没有静默改用远端默认分支；archive commit 为 `def56881179618efbbef0cadb92f14916feca6c2`；
- 实际动作：历史分支只提交本文，不纳入 bundle、pytest 临时目录、artifacts 或分析草稿；V3 独立 worktree 只恢复第 2.3 节七份白名单文档；迁入 `683c05c` 的本地 parquet loader、`69559e3` 的 val-only 单 rollout 校验、`e514640` 的通用 config/trainer dev lock；不迁入 Dataset V2 launcher 和 `2118555` 的 Dataset V2 pipeline；
- 预期/实际数量：白名单文档 `7/7`；通用基础设施修改 `3` 处；focused test `2/2` 通过；
- 技术结果：本地 pytest、Python compile 和 `git diff --check` 通过；V3 分支已推送 `post-training/codex/grpo-v3-selector-reward`；
- 科学结果：不适用；没有迁入 selector、reward、HLA、Dataset V2 路径或旧实验默认值；
- 产物路径：本地独立 worktree `.worktrees/grpo-v3-selector-reward`；
- 状态：`COMPLETE`；
- 下一唯一动作：在服务器固定新目录完成 S0。

### 记录 V3-002：远端清理与 S0 初始化

- 时间：2026-08-27 22:01–22:23（Asia/Shanghai）；
- branch / source commit / source status：服务器新 clone `/root/autodl-tmp/curious-vla-workspace/src/curious_vla_v3`，初始 `4c8cf73`，source clean；
- 输入与 hash：服务器 cgroup `cpu.max=50000/100000`，即 0.5 CPU；GPU 不可用；清理前数据盘 `120 GiB / 18 GiB available / 86% used`；本地 V2 R4-RAW 证据归档 SHA-256 `48b1afa076858260af7eada6c93547cbd38fa6b2e81f19cc230eb62e76a6fb1e` 与旧台账一致；
- 实际动作：按用户授权删除除 V2 R4-RAW 外的旧 Dataset V2/G4/preference checkpoint、adapter 和 optimizer/full-state；V2 R4-RAW 删除 step125 与 step250 full FSDP model/optimizer/resume state，只保留正式 step250 LoRA、Hugging Face 配置和全部训练/评估证据；SFT Stage-2 作为 V3 路线决策前的初始化锚点暂时保留；随后在固定新目录独立 clone V3 分支并检查环境与入口；
- 预期/实际数量：清理后数据盘先恢复到 `65 GiB available / 47% used`；R4-RAW step250 `adapter_model.safetensors` SHA-256 清理前后均为 `63ec75166a779c5e0bca192105e47cfd2ced23f068f07e6a07265dbe2cac930b`，`adapter_config.json` 均为 `98f02c09f6623c69f1fab39d7b7cdcf1baff2940944c1035ae9f1ec9af101ed7`；
- 技术结果：现有 `curious` 环境为 Python `3.10.20`、Torch `2.8.0`、datasets `5.0.1`、Transformers `4.57.1`、Accelerate `1.14.0`、PEFT `0.20.0`、Ray `2.48.0`、vLLM `0.11.0`；实际 EasyR1 依赖可导入，`torch.cuda.is_available=false`，远端 focused test `2/2` 通过；`pytorch_lightning` 不在当前 EasyR1 依赖声明中，不构成 S0 阻塞；
- 科学结果：不适用；历史权重清理不改变已经固化的指标、曲线或结论；
- 产物路径：服务器 V3 source 如上；保留权重 `/root/autodl-tmp/curious-vla-workspace/experiments/dataset_v2_20260825/v2_r4_raw_random1k_seed20260825/checkpoints/global_step_250/actor/lora_adapter/`；
- 状态：`COMPLETE`；旧权重/full-state 已从远端不可恢复，R4-RAW 关键证据仍由本地归档覆盖；
- 下一唯一动作：执行 D0I。

### 记录 V3-003：D0I SFT provenance、SFT-unseen 容量与 Tail 字段审计

- 时间：2026-08-27 22:28–22:40（Asia/Shanghai）；
- branch / source commit / source status：`codex/grpo-v3-selector-reward@08ec535140893b6578dc8c87240c41e7ad9d5d20`；服务器 source tracked status clean；
- 输入与 hash：SFT train parquet SHA-256 `86db9581c4bf29552822fdcc7c6bc71dee4a5d7f78c0f9c44b262bad4048f5dd`；旧 master index SHA-256 `887a67ff57e4299e43364ff6d897b655e25918eae44942cd32d3eab914dcb875`；model hash record SHA-256 `1606825c0bd2ede95ded6c16c53571900e1c834f05f284738edf477a7b7a4951`；
- 实际动作：新增受限 NAVSIM Unpickler，只允许静态审计确认的 NumPy `_reconstruct/ndarray/dtype`，拒绝任意 pickle global；按 4 history + 10 future、frame interval 14、必须有 route 的模型无关规则扫描 SFT-unseen logs；在服务器生成 SFT token/log blacklist 和聚合报告，不把 token/log ID 下载到本地；
- 预期/实际数量：SFT parquet/master 均为 `103,288` rows / `103,288` unique tokens，差集 `0/0`，覆盖 `1,192` unique logs；原始 NAVSIM 为 `1,310` unique logs，SFT-unseen 为 `118` logs；其中 `835` eligible unique scenes，token overlap `0`，118 logs 全部可读，但每 log eligible scene 中位数为 `0`、均值 `7.08`；
- 技术结果：实现测试本地与远端均为 `5/5` 通过；report/blacklist 行数和 marker 验收通过；聚合报告 SHA-256 `b101c3d25b83035b6af5e72fe5af72b03078a2f7171fc621b7382c3bc6736742`，token blacklist `5614a8a32a7030bda7cc71696e085f59116df442ba6c0b2604bb0649fea7f083`，log blacklist `b59e4f618b80243ef03ef8f9438f83f4676c42157098a7ff0fdb19d43be1634b`；
- 科学结果：`Reuse-SFT` 在当前本地数据上容量不足，835 scenes 无法同时支撑原定 2K–4K optimizer manifest、train monitor、Natural/Tail dev 和 final；118 个 unseen logs 的 `anns`、`traffic_lights`、`driving_command`、`ego_dynamic_state`、`map_location` 等原始字段完整，可继续审计模型无关 Tail 定义，但 raw sensor 与 metric-cache 文件覆盖均为 `0`，不能直接进入 D0S/D0A；
- 产物路径：服务器 `/root/autodl-tmp/curious-vla-workspace/experiments/dataset_v3_sft_unseen/data_build/v3_d0i_20260827/`；本地 aggregate-only 报告 `artifacts/dataset_v3_sft_unseen/v3_d0i_20260827/inventory_report.json`；
- 清理：D0I 固化 provenance 后删除 V2 可重建派生目录 `data/dataset_v2_20260825`（约 2.0 GiB）和 `exp_root/metric_cache_dataset_v2_20260825`（约 4.1 GiB）；保留 34 MiB provenance manifests、所有实验结论/曲线/log、SFT anchor、运行环境和 R4-RAW step250 LoRA；最终数据盘 `71 GiB available / 42% used`；
- 状态：`COMPLETE / REUSE_SFT_INSUFFICIENT`；
- 下一唯一动作：进入 D0R，只讨论并冻结 `Retrain-SFT`，或先补充新的原始 NAVSIM/同域 logs 后重新执行 D0I；在路线与规模冻结前不生成正式 split、image/cache 或训练入口。

### 记录 V3-004：冻结 Reuse-SFT + Controlled GRPO Overlap 路线

- 时间：2026-08-27 22:57（Asia/Shanghai）；
- branch / source commit / source status：`codex/grpo-v3-selector-reward@f9054d4`；本记录开始前 source clean；
- 决策关系：本记录基于用户新决策取代记录 V3-003 的“只讨论 Retrain-SFT 或等待新 logs”下一动作；V3-003 的 inventory 数值与文件 hash 保持历史有效；
- 输入与证据：D0I 已确认 SFT provenance 为 103,288 unique tokens / 1,192 unique logs，严格 SFT-unseen reserve 为 118 logs / 835 eligible unique scenes，两个来源宇宙 token/log overlap 为 0；
- 冻结决策：本轮采用 `SFT_ROUTE=REUSE_SFT_CONTROLLED_GRPO_OVERLAP`，保留 `models/sft_stage2`，不冻结 `Retrain-SFT`，不等待新增 logs；
- 训练边界：`grpo_screen`、`train_monitor`、SFT Rollout Bank、reward replay、H0 pilot 与 Random/TailMix optimizer manifests 只能从 1,192 个 SFT-seen logs 构建；四格使用同一来源宇宙和 matched quota，并报告 exact token/log reuse、Screen→Train 选择率及 per-log cap；
- 评估边界：118 个 SFT-unseen logs / 835 个 eligible scenes 全部且仅分配给 Dev/Final；不得进入任何训练侧产物；Dev/Final 与 SFT provenance、GRPO train-side 的 token/log overlap 必须为 0；
- 结论边界：只允许声称“受控重用 SFT-seen 数据进行 GRPO 后，在严格 SFT-unseen Dev/Final 上相对 SFT 的 post-training 增量”；不声称 GRPO train 数据 unseen、新增了 logs 或证明了新数据效率；
- 命名空间：D0I 的 `dataset_v3_sft_unseen` 历史证据保留原位；D0S 起正式资产使用 `dataset_v3_controlled_overlap`；
- 状态：`COMPLETE / ROUTE_FROZEN`；本记录只冻结路线和来源边界，未生成 split、rollout、cache 或训练产物；
- 下一唯一动作：执行 `V3-D0R-2`，在现有 118 logs / 835 scenes 内冻结 Dev/Final 具体 log 分配、Natural/Tail 定义，并同步冻结受控 GRPO train manifest 规模。

### 记录 V3-005：D0A 图像/cache 前置资产完成与 ZIP Range 数据源切换

- 时间：2026-08-28 17:30–17:46（Asia/Shanghai）；
- branch / source commit / source status：`codex/grpo-v3-selector-reward@d91db6f4dd35482aa8eb447be4e4fd7d68503c27`；本地与服务器针对性测试均为 `5 passed`，服务器 source tracked status clean；
- 切换原因：原 `WeiXiCZ/navsim-trainval-full-front` 为 148 GB 连续 `tar.zst`，AWS Xet 长连接在当前服务器出口持续降速；Hugging Face 全站候选审计后，改用 `richardyann/navsim-select@7707301e13828b4599b3a0f834b44efed57df90e` 的 `sensor_blobs/trainval.zip`，通过 ZIP64 中央目录与 HTTP Range 只读取目标成员；
- 固定输入：远端 ZIP 为 `148,230,424,017` bytes / `735,584` entries；中央目录为 `120,860,982` bytes，SHA-256 `71c3050bac75dc5e4eac1076202e546c65877e139a5382e50341bf94cdb1f7c8`；active-assets SHA-256 `d39893ab548e4a69b7e50030b540d35d3a63a23b7627d3dbd6bfd67ccf8c6bb8`；
- 技术门禁：精确定位并下载 `835` 张缺失图片和 `64` 张既有 NAVSIM navtrain overlap，共 `899/899` 个 ZIP members；所选成员压缩正文合计 `201,466,300` bytes；每个成员均验证 ZIP CRC/长度，64 张 overlap 均通过 SHA-256 字节一致性检查，门禁完成前未写入正式目录；
- 完成结果：补齐 `835` 张严格 SFT-unseen CAM_F0，image coverage 为 `9,091/9,091`、metric-cache 文件数为 `9,091/9,091`，生成 `D0A_IMAGES_COMPLETE`；`selective_zip_report.json` SHA-256 `2ef2f82c760d06da525f06c0c750fe57a8029a5302f0c80ec4c81c1eab0efcea`，`image_coverage_report.json` SHA-256 `c9e3cc553bc8ad7aa4011e2404fe313a377ea38edd32ef37cb7bc5cf6ee64e82`；
- 空间闭环：删除已替代的旧 `full_front_parts`（9.9 GiB）与旧 staging（5.7 MiB），保留运行日志和报告；正式图像为 1.9 GiB、metric cache 为 3.9 GiB，`/root/autodl-tmp` 最终可用约 65 GiB / 使用率 47%；
- 状态：`COMPLETE / D0A_IMAGE_CACHE_ASSETS_COMPLETE`；未启动 split 后的 rollout、训练或 Dev/Final 访问；下一动作仍由 `V3-D0R-2 → V3-D0S → V3-D0F` 的冻结顺序决定。

### 记录 V3-006：D0R-2、D0S 与 D0F 无卡数据冻结完成

- 时间：2026-08-28 17:56–18:23（Asia/Shanghai）；服务器仍为无卡模式，cgroup `cpu.max=50000/100000`，即 0.5 vCPU；
- branch / source commit / source status：数据准备门控 commit `02d97e10a2f315ba7d144f454ad7bb02bd9b7e19`，prompt horizon 修复 commit `b46ffdf0f63bd7e5931f9e567b34a25aa809f413`；本地与服务器 source 一致且 tracked status clean；本地与服务器相对 GitHub `post-training/codex/grpo-v3-selector-reward@108c609` ahead 2，因当前会话未取得向该外部 GitHub 目的地发送代码的显式授权而未推送；
- D0R-2 冻结：固定 seed `20260827`；14-frame 非重叠事件窗口（4 history + 10 future）；Screen 8,000 / per-log cap 8，Monitor 256 / per-log cap 2，每格 optimizer manifest 2,000；Random/TailMix intent quota 固定为 straight/left/right=`1333/434/233`；2×2 执行优先级固定为 `RR → TC → TR → RC`；
- Tail 与评估几何：`TAIL_EVAL_ROUTE=POLICY_INDEPENDENT_GT_ACTOR_PROXIMITY`；vehicle `<=5.0 m` 或 pedestrian/bicycle `<=10.0 m` 定义 interaction scene，58 个含 eligible scene 的 unseen logs 按 interaction rate/count、minimum actor distance 和 stable hash 排序后等分为 Tail/Natural，各 29 logs；Tail/Natural interaction scene rate 为 `83.2%/47.9%`；Dev/Final 分别为 `416/419` scenes，四个 split 为 `dev_natural=210`、`dev_tail=206`、`final_natural=214`、`final_tail=205`；
- D0S 重跑：新增 Master Index 的 `prompt_version`、`sft_overlap` 和 `data_status` 字段；每次正式生成均执行两次独立重跑，retry2 的 data/manifests/reports tree hash 分别为 `29999e3b2d00a97a405144dc4dcd213d97fe9ad4062222b828755033c68e0b2b`、`2983e134dedfe381358aab71f86efec22b603b3a5581138f132b326088a09eea`、`0357121b1120e5b2e0e5043fd2259eff6d9f3137e1b00595f00e9b312e69b481`，membership/order 和所有文件逐字节一致；
- prompt 门控与修复：首次 D0F 在写入前因四个 parquet 每条 prompt 同时含 `4-second` 与遗留 `5-second` 而失败，其余门控全部通过；修复从只替换一个完整短语改为全局 `5-second → 4-second`，重新双跑 D0S 后，`grpo_screen/train_monitor/dev/final` 共 9,091 rows 的 `4-second` 覆盖为 9,091、`5-second` 计数为 0；本地与服务器 focused tests 均为 `9 passed`；
- D0S 关键 hash：`d0r2_decision_report.json=7d94ff484dbbe55b1e7800831079dd88372e8a48d50cf386fe487de8fae1a306`；`d0s_acceptance_report.json=618e026d0fef4b224c819a9af17b76240e9d12da4861bd3b7f7a9a78762e6af9`；`reproducibility_report.json=40cb917605ae04fbd4117c8365229fe25567add868c03ca06d7840aaa10b5258`；`active_assets.csv=d39893ab548e4a69b7e50030b540d35d3a63a23b7627d3dbd6bfd67ccf8c6bb8`；`master_index.csv=40b3a1fb4a9c12a7a4cce26497aa0058128c7370870477033a2a7e523a90280b`；
- D0F 结果：retry1 在 99 秒内完成；25 项 gate 全部为 true，含 9,091/9,091 CAM_F0、9,091/9,091 metric cache、SFT Stage-2 8 GiB 实物哈希、source clean/exact、报告 finite、prompt/version、训练/评估与四个 eval split 的 token/log 零重叠、Tail policy-independent 和 D0S replay 一致性；`asset_sha256.csv` 含 18,182 个资产 hash；
- D0F 关键 hash：`dataset_card.json=53ff0b4fb5d65bd8597dd681e82f285efa5344575d99a842420764027e3c990b`；`asset_sha256.csv=c2a2e358276e0c47bb6991cde1510b325999a54ec377c46386ffca4e943e2ade`；`asset_coverage_report.json=1c9e8a6222eb8f64279df8c1dae89cd5875ea9f429375e059788b9ef6713f6b3`；`final_access_lock.json=0b1c67094974ab4efb27c4b71f15cf6f61038a78a71888314a90cfa606937fcf`；`V3_DATA_FROZEN`、`COMPLETE` 和 `exit_code=0` 齐全，4 个 Final 文件均为 mode `0400`；
- 产物路径：服务器 `experiments/dataset_v3_controlled_overlap/data_build/v3_d0s_20260828_retry2/` 与 `experiments/dataset_v3_controlled_overlap/data_build/v3_d0f_20260828_retry1/`；未创建 `rollout_bank`，未运行 SFT/GRPO inference 或训练；
- 空间闭环：删除两次重跑的精确临时目录与测试目录；D0S 证据 56 KiB、D0F 证据 2.8 MiB、正式图像 1.9 GiB、metric cache 3.9 GiB；数据盘仍为 120 GiB / 65 GiB available / 47% used；
- 状态：`COMPLETE / V3_DATA_FROZEN`；无卡模式下能完成的数据准备已全部完成；下一唯一动作是将服务器重启到 GPU 模式后执行 `V3-S1` 的 SFT shared rollout bank，当前不得在 0.5 vCPU 无卡模式启动该阶段。

### 记录 V3-007：S1 Screen、metric replay 与 Candidate/Confirm 协议冻结

- 时间：2026-08-28 19:05–2026-08-29 06:16（Asia/Shanghai）；
- branch / source commit / source status：Screen 使用 `22f2cc5`，CPU metric replay 使用 `c049c59`，Candidate/Confirm 协议冻结使用 `a9ea2c51`；各阶段启动和终态审计时服务器 tracked source 均 clean，协议提交已推送 `post-training/codex/grpo-v3-selector-reward`；
- 输入与边界：只读取 D0F 冻结的 8,000-scene `grpo_screen` 与 SFT Stage-2；`seed=20260827`、`G=4`，共 32,000 rollouts；没有读取 118 个 SFT-unseen logs、Dev 或 Final，也没有创建 optimizer manifest 或执行参数更新；Screen manifest SHA-256 `0df963c45c06f0e7590d9e698cc086e5317532672b6031158636ac4ff8b50f00`；
- Screen 技术结果：正式 run `v3_s1_screen8000_g4_seed20260827` 于 2026-08-29 04:21 完成，8,000/8,000 tokens 均精确生成 4 条，共 32,000 条；raw response 与 poses 缺失均为 0，解析失败 11 条，parse success rate `0.99965625`，数值有限；GPU、Ray、Gunicorn 和端口均在终态清理，数据盘仍有约 64 GiB 可用；
- metric replay：因为原 Screen bank 只保存 `pdms/pdms_scaled`，在不重新生成文本和轨迹的前提下，以 CPU 对 32,000 条 rollout 回放冻结 evaluator；run `v3_s1_metric_replay_20260829` 在 5,589 秒内完成，`31,989/32,000` 条成功回放，11 条与原解析失败精确对应；新增保存 `no_at_fault_collisions`、`DAC`、`ego_progress`、`TTC`、`history_comfort`、`pdms` 与 `pdms_scaled`，并补齐 model/input/source/result hash；
- Screen geometry：按 V2 候选 L0–L3 evaluator safety 语义在 V3 上重新审计，而非提前冻结 R0 reward；8,000 个组中 exact-zero std 为 2,384，std `<0.05` 的 low-nonzero 为 3,951；candidate tier 行计数为 L0/L1/L2/L3=`180/1,053/149/30,607`，另有 invalid=`11`；含任一 L0–L2 的组为 833，mixed-tier groups 为 788；风险 rollout 数为 0/1/2/3/4 的组分别为 `7,167/489/189/105/50`；headroom 的 q90/q95 为 `0.136532/0.25`；
- Candidate 冻结：Confirm 候选固定为“所有含任一 L0–L2 的 833 个组”并集“按稳定 token tie-break 排序的 Screen top-10% headroom 800 个组”，两者重叠 727 个；并集后仅按下一 headroom rank 增加 2 个以闭合 batch-4，最终精确 908 unique tokens；intent 为 straight/left/right=`420/269/219`，覆盖 546 个 SFT-seen logs，单 log 最大 6；禁止在 Confirm 结果后改写 Candidate membership；
- Confirm 协议冻结：Confirm run 固定为 `v3_s1_confirm908_g4_seed20260828`，`seed=20260828`、`G=4`；Screen 与 Confirm 构成两个独立 block，总 `G=8`；stable severe、stable near-risk 与 stable mixed-recoverable 均必须在两个 block 同时满足各自语义，四类精确配额只允许在 Confirm 后先做容量审计、再于 Select 前一次性冻结，不能按下游训练或 Dev 结果调节；
- 关键产物与 hash：Candidate manifest `e0437722e19dbb0b8370c553b63bcc31f84d82ed6a6b8492e1a20175c0b48600`，Candidate parquet `2b88eca0eef76ee8eb7b6b6a160ce40d5ea3fbd6de943acf46ce1036da02b8af`，Candidate freeze report `2382941ab1de5b133da4be5654b864144bb7be40988cd863a4383acd2e6d6262`，Candidate 输入的 S1 geometry SHA-256 `cfc8166b4676b9d3ca29688e092f8ba059ec9820ad69ba24bf56d0aec840c7e2`；产物路径为 `experiments/dataset_v3_controlled_overlap/rollout_bank/v3_s1_{screen8000_g4_seed20260827,metric_replay_20260829,candidate_freeze_20260829}/`；
- 科学边界：本阶段只冻结 train-side Candidate/Confirm sampling；L0–L3 仍是待 R0 对照的候选安全分层，不能称为已冻结 CDT scalar reward；Screen/Confirm rollout 不是严格 unseen 评估，任何模型结论仍须由冻结的 SFT-unseen Dev/Final 提供；
- 状态：`SCREEN_COMPLETE / METRIC_REPLAY_COMPLETE / CANDIDATE_FROZEN / CONFIRM_READY`；
- 下一唯一动作：按已冻结协议启动 908-scene、G=4、seed `20260828` 的 S1 Confirm；终态验收后先冻结 TailMix 四类容量与配额，再生成 Random/TailMix 各 2,000-token selector manifests。

### 记录 V3-008：S1 Confirm 完成与稳定类别判定冻结

- 时间：2026-08-29 09:27:33–10:08:31（Asia/Shanghai）；
- branch / source commit / source status：Confirm source 为 `adeb2edace9739ee13d0b34265c0f7e5780cbd04`，source status 为空；稳定类别 classifier 由本记录对应的下一提交冻结，必须先提交再做容量审计；
- 输入与 hash：Candidate 908 unique tokens、`G=4`、seed `20260828`，manifest/parquet SHA-256 分别为 `e0437722e19dbb0b8370c553b63bcc31f84d82ed6a6b8492e1a20175c0b48600` / `2b88eca0eef76ee8eb7b6b6a160ce40d5ea3fbd6de943acf46ce1036da02b8af`；input/model hash records 均验收通过；
- 技术结果：总耗时 2,458 秒；`COMPLETE`、`exit_code=0`，无 `RUNNING/FAILED`；3,632 rollouts、908 tokens、每 token 精确 4 条，manifest membership 精确一致；parse success rate `0.9983480176211453`，解析失败 6，clipping 0，必需字段齐全且非 finite 数值为 0；ADAS 为 3,632 data rows，与 rollout token/score 多重集合一致；
- result hash：`rollouts.jsonl=db03e6131be43ee144b3f9c257cc38bed81092819d0f1544341683a3f046b8f9`，`diagnosis.json=32349c6636f94f288c15bc5c94eaecb29e99284383f49eb08235c0747a551f5a`，`adas_scores.csv=43e7d40ee8585f17bf5c013cc594e0c3e4d00c7c943032efb8c9d9171eb633a0`；
- 稳定性规则冻结：两个独立 block 总 `G=8`；对应事件必须在 Screen 与 Confirm 各至少出现一次；互斥优先级固定为 `stable_severe → stable_mixed_recoverable → stable_near_risk → random_anchor`，精确定义见第 5.4 节；invalid rollout 不计入任何 L0–L3 occurrence；
- 科学边界：本记录在看到稳定类别容量前冻结 classifier，但没有预设四类配额；配额只能在随后一次 train-side capacity audit 后、Select 前冻结；没有读取 Dev/Final，也没有 optimizer update；
- 错误与资源：run/reward/launcher 日志未发现 OOM、Traceback、RuntimeError、HTTP 5xx 或 reward error；main_adas、Ray、Gunicorn、8901 均清理，GPU 0 MiB / 0%，磁盘约 64 GiB 可用；
- 产物路径：`experiments/dataset_v3_controlled_overlap/rollout_bank/v3_s1_confirm908_g4_seed20260828/`；
- 状态：`COMPLETE / STABILITY_CLASSIFIER_FROZEN`；
- 下一唯一动作：只读取 Screen、Confirm、Candidate 与 train-side Master Index 执行稳定类别容量审计；据此在 Select 前冻结四类 × intent 精确配额和不足类别处理规则。

### 记录 V3-009：稳定类别容量审计与 TailMix 配额冻结

- 时间：2026-08-29 10:27:03–10:27:07（Asia/Shanghai）；
- branch / source commit / source status：capacity audit source `0b5bbaecd1ebea8a02bce9fd6754410cbf29e7f6`，source status 为空；
- 输入与 hash：仅使用 Screen enriched rollouts、Confirm rollouts、Candidate manifest 与 train-side Master Index；Screen/Confirm rollout SHA-256 分别为 `6f9aefb8fb3124fd7db331c4d34e114e4c956c557d6542a53ab9db6ef0866277` / `db03e6131be43ee144b3f9c257cc38bed81092819d0f1544341683a3f046b8f9`；Master Index 为 `40b3a1fb4a9c12a7a4cce26497aa0058128c7370870477033a2a7e523a90280b`；
- 容量结果：互斥类别为 `stable_severe=578`、`stable_mixed_recoverable=68`、`stable_near_risk=7`、`random_anchor=7,347`；对应 intent straight/left/right 容量分别为 `258/159/161`、`29/35/4`、`1/6/0`、`5,045/1,535/767`；稳定三类覆盖 653 tokens；
- 配额冻结：保留全部 653 个稳定事件，anchor 精确补 `1,045/234/68`，最终四类×intent 矩阵见第 5.4 节；该方案精确得到 2,000 tokens 与总 intent `1,333/434/233`，不丢弃稳定事件，也不以重复/上采样伪造 near-risk 容量；
- 技术结果：8,001-line capacity CSV（含 header）与 JSON report 均通过 input/result hash 复验；capacity CSV SHA-256 `ebb90ea58bc7e8605f00eaa85ab4247a8f6264b027deb176af2a5a89fe254f41`，report `7937480a27c4308e20aab589f52058172b56a81a98e574dce668b6aa7f735e3c`；
- 科学边界：类别比例完全由预先冻结 classifier 的 train-side 容量决定；没有读取 Dev/Final，没有模型更新，没有按下游结果调节；
- 产物路径：`experiments/dataset_v3_controlled_overlap/rollout_bank/v3_s1_stability_capacity_audit_20260829/`；
- 状态：`COMPLETE / TAILMIX_QUOTAS_FROZEN / SELECT_READY`；
- 下一唯一动作：在硬编码容量与 class×intent 矩阵门禁下生成 Random/TailMix 各 2,000-token manifest/parquet 和分布报告；随后执行 R0 CPU geometry。

### 记录 V3-010：S1 Random/TailMix selector 冻结完成

- 时间：2026-08-29 10:36:25–10:36:27（Asia/Shanghai）；
- branch / source commit / source status：selector freeze source `33569a25da87c8db8b09cc71d5ee9894a46967bb`，source status 为空；
- 实际数量：Random/TailMix 均为 2,000 unique tokens，均精确满足 straight/left/right=`1,333/434/233`；TailMix 为 severe/mixed/near/anchor=`578/68/7/1,347`；Random/TailMix 分别覆盖 944/897 个 logs，per-log max 为 6/7，均低于 cap 8；
- 对比分布：token overlap 493，log overlap 806；intent JS divergence `0`、month `0.0033802894523274262`、log-name `0.20235151597661913`；SFT-seen Master 的 `map_location` 为空且无经审计的 train-side `route_type` 字段，因此 region/route-type 标记为 unavailable，未把 intent 重命名为 route type；该缺口留给 M0 判定弱分布门槛，不反向重选 selector；
- 边界与 overlap：两个 selector 均为 Screen 的 25% 且 2,000/2,000 为预注册 SFT overlap；与 train_monitor、Dev、Final 的 token/log overlap 均为 0；没有读取 Dev/Final；
- 关键 hash：Random/TailMix manifest=`1b2dd1fa05e6c46f08d6a65d15a02dc1c9c3762819597efba65a059ee84f54ed` / `8844ac3589dcdad5c36b6c26684ea038d6f319955c5c9045e1a41107947fa21e`；Random/TailMix parquet=`24f55669474206ce1fa0c051ec8a0e40b25b5244ce7296370a7040a31f6a7537` / `fc29439921be16ac81e0f6a3d8274de69ae03b4a46ecf5c820e5e9fc11f0a339`；membership/report=`e350b7769b1c3d1973b9749a0d2d5abd587b998a4ab4e1de80e42b700b275342` / `b44616c8893e83e9c7d7f17bfd3ac2fbad66e9957e6f2a108247895dbd413425`；
- 技术结果：input/result hash 全部复验，manifest 各 2,000 行，source clean，证据目录约 1.9 MiB，数据盘仍约 64 GiB 可用；
- 产物路径：`experiments/dataset_v3_controlled_overlap/selector_freeze/v3_s1_selector_freeze_20260829/`；
- 状态：`COMPLETE / S1_SELECTORS_FROZEN`；
- 下一唯一动作：按第 5.5 节预注册的 `Raw-PDMS / R_PDMS / R_task` 公式执行 R0 CPU geometry；在 R0 决策前不实现正式 CDT reward 入口或启动训练。

### 记录 V3-011：R0 reward geometry 与唯一 CDT reward 冻结

- 时间：2026-08-29 10:52–11:02（Asia/Shanghai）；
- branch / source commit / source status：候选 geometry retry1 source `62df3af48c29cec7cc67158e4ba3092ebbe14912`；正式协议 source `26d65d0d2e4c4a648a8e55a1636e4bb5f15c7974`；两者 source status 均为空；
- 执行边界：只读取冻结的 8,000-group train-side shared bank 与 Random/TailMix manifests；逐 completion 比较 Raw-PDMS、`R_PDMS=(2L+pdms)/7`、`R_task=(2L+Q_task)/7`，其中 `Q_task=(5*ego_progress+2*history_comfort)/7`；没有 optimizer update，没有读取 Dev/Final；
- 技术过程：首次候选 run 因脚本直接执行模块导致 `ModuleNotFoundError: projects`，按技术失败保留原目录；唯一修复为改用 `python -m projects.dataset_v3.r0_geometry`，retry1 在 3 秒内 `COMPLETE/exit_code=0`；正式 reward 入口在真实 EasyR1 环境及固定 `NAVSIM_STAT_PATH` 下导入成功，18 项本地/远端 focused tests 通过；
- Random geometry：Raw/CDT-task EffectiveGroupRate=`0.7125/0.715`，exact-zero=`0.2875/0.285`，low-nonzero=`0.5495/0.6235`；CDT-task 的 cross-tier inversion/tie=`0/0`、within-tier inversion/tie=`0`；
- TailMix geometry：Raw/CDT-task EffectiveGroupRate=`0.758/0.768`，exact-zero=`0.242/0.232`，low-nonzero=`0.39/0.4455`；CDT-task 的 cross-tier inversion/tie=`0/0`、within-tier inversion/tie=`0`；
- 科学决策：两项 task metric 对 valid rollout 完整、finite 且值域通过；`R_task` 在两个 selector 上通过全部预注册排序与 EffectiveGroupRate 门槛，因此冻结为唯一 CDT scalar reward；不选 `R_PDMS`，避免将 Collision/DAC/TTC safety 语义在 tier 与连续项中重复计分；invalid 保持在 L0-L3 之外并取得技术零；
- 生产入口：CDT 为 `navsim_reward_text.py:compute_score_cdt_task`，Raw control 为 `navsim_reward_text.py:compute_score_raw_pdms`；旧 `compute_score_fast` 继续返回 scaled-PDMS，只服务已完成的兼容路径，不作为 V3 Raw 正式对照；
- 关键 hash：geometry group CSV=`6d22dad1262857a943f4025bb18f10538c671daef59f527194bff2b3525248cc`，geometry report=`58d5e3693201c26181c85153eddc8b507c2b2009223ba637febdc96d8386bdc9`，正式 reward protocol=`9b8ab3ab21d214406785af428ea761c5a5980c9c43049b30f2a991da68369eb7`；
- 产物路径：`experiments/dataset_v3_controlled_overlap/reward_freeze/v3_r0_geometry_candidates_20260829{,_retry1}/` 与 `reward_freeze/v3_r0_cdt_task_freeze_20260829/`；
- 状态：`COMPLETE / R_TASK_CDT_V3_FROZEN / H0_READY`；
- 下一唯一动作：只用 Random-Raw 与 train-side `hparam_train/train_monitor` 执行 H0，先冻结 training-group budget 和 LR；R0 的高 low-nonzero rate 已触发在选定 LR 上追加 `grpo` vs `std_floor_grpo`，不得同时改 batch、LoRA 或 KL。

### 记录 V3-012：H0 train-only pilot 数据、预算与晋级门槛预冻结

- 时间：2026-08-29 11:22:13–11:22:15（Asia/Shanghai）；
- branch / source commit / source status：`75f1c480be034883534451db56f2e2ce3a184a01`，source status 为空；
- 数据冻结：从冻结 Random 2,000-token manifest 按 stable hash 与原分布确定性截取 512 个 `hparam_train` groups，intent straight/left/right=`341/111/60`；独立沿用 256-token `train_monitor`，optimizer/monitor token overlap=`0`，两者均为 SFT-seen train-side 且不接触 Dev/Final；
- 预算冻结：LR 与 estimator pilot 均固定 `G=4`、4 groups/update、128 updates、512 processed groups、2,048 train rollout queries；monitor 使用 `n=1`，在 step `0/26/51/77/102/128` 评估，对应 processed groups `0/104/204/308/408/512`；batch 8 只有预注册梯度方差门槛触发时才以 64 updates / 512 groups compute-match；
- 参数边界：首轮仅比较 LR `1e-6` / `3e-6`，其余固定为 standard GRPO、PPO epoch 1、LoRA rank 8 attention-only、KL loss `0.01` / `low_var_kl`、shuffle true、seed `20260829`；R0 low-nonzero gate 已触发，选定 LR 后只追加 `grpo` / `std_floor_grpo(floor=0.05)`；
- LR 门槛：候选需满足 final parse≥0.99、clip≤0.01、三项 safety 相对 step0 降幅均≤0.01、`|ppo_kl|≤0.05`、mean clip fraction≤0.05、max grad norm≤5；两者均 admissible 时，只有 `3e-6` 的 final PDMS gain 比 `1e-6` 高至少 0.005 且 post-baseline mean gain 高至少 0.002 才升级，否则保守冻结 `1e-6`；
- estimator 门槛：只有 std-floor admissible、相对 GRPO 的 final/mean PDMS gain 分别≥0.003/0.001，且每项 final safety 不低超过 0.005，才冻结 std-floor；否则保持 standard GRPO；exact-zero group 在两种实现中均为零 advantage；
- batch 8 触发：选定 estimator 仅在 grad-norm CV>0.5、mean clip fraction>0.02 或至少 5% updates 的 grad norm≥0.99 时追加 compute-matched batch 8；否则固定 batch 4；
- 技术结果：远端 23 项 focused tests 通过；std-floor 独立张量测试对 exact-zero group 输出 `[0,0,0,0]`，对 `[0,.01,.02,.03]` 在 floor 0.05 下输出约 `[-.3,-.1,.1,.3]`；protocol `COMPLETE/exit_code=0`；
- 关键 hash：H0 protocol=`f78701f4e5a273784174a19bdd4f77759773522d9ffa62be944c24af48301a07`，hparam manifest/parquet=`f005502fa76b0880ae777e070a9c831088c3381bafa10f5b32d131b64c1c5b6e` / `64e22bd164db3fd5ebf2d04f632de85fb606607397d3d80e19fd0f7e4ae5de4c`，monitor manifest/parquet=`1d02bfae05d8b749ad9a1cc9da9d94147728ee8831ae539fef5290be096c2752` / `967136155d1e3ffa36726c217f75efd8c92d75cf5b12320520905464b16b51a2`；
- 产物路径：`experiments/dataset_v3_controlled_overlap/hparam_freeze/v3_h0_protocol_20260829/`；
- 状态：`COMPLETE / H0_PROTOCOL_FROZEN / LR_PILOTS_READY`；
- 下一唯一动作：顺序执行 H0-LR1 与 H0-LR3；两次均从原始 Stage-2 初始化并只读同一 `hparam_train/train_monitor`，完成后机械应用上述 LR gate。

## 9. 结论边界

- V3 只承认在 V3 数据和 V3 代码上重新产生的指标；
- 旧台账用于解释设计来源和已失败机制，不为 V3 提供可直接复用的 baseline 数值；
- V3 当前保留 `models/sft_stage2`；它是唯一初始化和零更新锚点，不是全模型从零训练；
- GRPO train-side 是对 SFT-seen 数据的受控重用，不能称为 unseen；严格 unseen 只描述 Dev/Final；
- 当前允许的核心结论是“受控 overlap GRPO 在严格 SFT-unseen Dev/Final 上相对 SFT 的 post-training 增量”；
- `Retrain-SFT` 与等待新增 logs 均不是当前执行路线；若未来重开，必须新增决策记录并重新冻结全部 provenance、split 和 baseline；
- 外部数据集、online selector、HLA、SLDR、constraint optimizer 和多 reward sweep 均不属于当前首轮。

## 10. 证据保存与空间闭环

每个正式 run 至少保留：

- `run.env`、resolved config、source commit/status；
- model/data/manifest/input hash；
- train/dev tokens；
- raw train/dev rollout 和 parsed trajectory；
- train diagnosis、final metrics、paired cluster-bootstrap report；
- policy loss、entropy、KL、clip、grad、LR、advantage、reward 分项、response length、timing 和显存曲线；
- LoRA adapter、optimizer/config、代表样本、`COMPLETE`、`exit_code` 和 result hash。

空间规则：

- 正式训练启动前必须验证上述证据路径可写、预计峰值空间充足；任一前置检查失败时不得启动 run；
- run 顺序执行，不并行保留多个 8 GB full actor state；
- 只有 LoRA、评估、rollout、曲线和 hash 全部固化且进程回收后，才允许精确删除该 run 的 full actor model；
- 不删除 Dataset V2、Stage-2、旧实验 ledger/rollout/report；
- 大日志在结果 hash 固化后可以压缩，不以删除原始科学证据换空间；
- 任何删除都必须记录精确路径、大小、时间和不可恢复性。

长任务监控规则：Luna 必须在单次 turn 内启动服务器侧阻塞 watcher，以 `while + sleep` 持续检查终态、错误、磁盘和 GPU/8901；正常快照不返回 final，只有 `COMPLETE`、`FAILED` 或明确异常才结束并完成验收。主进程启动 run 后直接长时间暂停等待 Luna 唤醒，不做固定间隔轮询，也不承担正常进度兜底。
