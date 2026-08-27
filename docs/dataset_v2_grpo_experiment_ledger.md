# Curious-VLA Dataset V2：ADAS/FALS 与 SDR/SLDR 实验闭环台账

> 生效日期：2026-08-25。
> 本文档是 `dataset_v2_20260825` 上比较 ADAS、FALS、SDR、SLDR 的唯一实时计划与结果台账。
> [`grpo_g4_experiment_ledger.md`](grpo_g4_experiment_ledger.md) 只保留旧 5,656 数据和旧 dev 上的历史证据，不与本轮结果合并统计。

## 1. 目标、状态与结论边界

本轮只回答四个问题：

1. 固定 Random-1K 和 `G=4`，SDR 相对 raw-PDMS 是否带来独立收益；
2. 固定 Random-1K 和 `G=4`，SLDR 相对 SDR 是否改善安全且不损害总体驾驶质量；
3. 固定 SDR 和 `G=4`，FALS-1K 相对 Random-1K 是否提高训练效果；
4. 固定 SDR 和 `G=4`，当前可执行的 ADAS-G4 selector 相对 Random-1K 是否提高训练效果。

截至 2026-08-27（Asia/Shanghai）的状态：

- Dataset V2 的 8,000 candidate、2,000 dev、10,000 张 CAM_F0 和 10,000 份 metric cache 已存在；
- `random_1k.txt` 已冻结，并且 1,000 个 token 全部属于 Phase-1 6K candidate；
- FALS/Random manifest 已冻结；ADAS 因 S0 稳定性失败而关闭；
- Dataset V2 已产生 E0、R4-SDR 与 R4-RAW 的正式 dev 结果；两条无 dev 技术 smoke 已产生并固化 10-step GRPO checkpoint；
- `V2-I0` correct-image 首次运行因本地 parquet loader 入口缺失，在模型初始化后、零 rollout 和零 reward query 阶段技术失败；失败证据已保留；
- R0 raw geometry 补齐后，`V2-D0 retry6` 已通过并将正式 source 更新绑定为 `3201f9b7f1601f53f23fcb17962ca7216f132258`，数据、manifest 与 Stage-2 hash 不变；
- 第一次 pre-freeze asset check 虽通过数据校验，但 launcher 仅支持 D0，已保留为 `PREFREEZE_ONLY`，没有发生 GPU inference，也不作为正式 source freeze；
- `V2-I0` 已通过：correct−shuffled PDMS 为 `+0.39872`，paired log-cluster 95% CI `[+0.34306, +0.45539]`，两侧 parse 均为 `100%`；
- `V2-S0` 结果为 FALS 通过、ADAS 失败；block 5 仅作一致性 sensitivity evidence，block 6–8 已按协议修订取消；
- `V2-S1`、正式 `V2-M0 retry1` 与 `V2-R0 retry1` 已完成；R0 关闭 SLDR，SDR/RAW 路径继续；
- `V2-T0-SDR`、`V2-T0-RAW` 均通过；两侧 TensorBoard 均包含 69 个 scalar tags，事件文件、曲线、rollout、报告与 LoRA 已同步到本地；
- `V2-E0` 首次运行在 dev lock 前因 `val_only + rollout.n=1` 被 GRPO 训练校验误拒而技术失败；最小修复后由 `V2-D0 retry9` 将正式 source 绑定为 `69559e305b9524cdd998ae92a4e4734e386f47e1`；
- `V2-E0 retry1` 已通过：2,000/2,000 dev、parse `100%`、PDMS `0.94746413`、PDMS scaled `0.90872652`、Safe `0.99700000`；永久 `V2_E0_DEV_ACCESSED` lock 已创建；
- E0 完整 TensorBoard、rollout、metrics 和 GPU 曲线已同步到本地；final reserve 继续保持 manifest-only；
- `V2-R4-SDR` 的 250-step 训练和 2,000-token dev 已完整完成；原 launcher 因验证器误把末尾合法 `val` 行计入训练 step 而 `exit_code=1`，修复后以独立 CPU-only postprocess run 完成 technical gate 和 paired analysis，未重新训练或访问 dev；
- `V2-R4-SDR` dev PDMS `0.94579755`，相对 E0 为 `-0.00166658`，paired log-cluster 95% CI `[-0.00363282,-0.00001381]`；E0 只作净更新参考，不替代 SDR−RAW 直接 contrast；
- `V2-R4-RAW` 已完整通过技术门控：250/250 steps、4,000 train rollout、2,000/2,000 dev、TensorBoard/曲线/LoRA/报告均已同步本地；dev PDMS `0.94855070`；
- `SDR−RAW` PDMS 为 `-0.00275315`，paired log-cluster 95% CI `[-0.00523983,-0.00048599]`，按预注册门控记为 `NEGATIVE_OR_TRADEOFF`，关闭 SDR matched-seed 分支；
- 为避免把后续 CDT-HLA 改动带入直接 reward contrast，正式 runtime source 已从 R4-SDR source `69559e305b9524cdd998ae92a4e4734e386f47e1` 仅 cherry-pick 验证器修复并重绑定为 `3c9bf2b4d60b8e4fadaef395743bd0c4ae07fb29`；冻结数据、manifest、Stage-2 和 launcher 行为不变；
- `V2-F4-SDR retry1` 已完整通过技术门控：250/250 steps、4,000 train rollout、2,000/2,000 dev，TensorBoard/曲线/LoRA/报告均已同步本地；dev PDMS `0.94819769`；
- `F4−R4` PDMS 为 `+0.00240014`，paired log-cluster 95% CI `[+0.00006349,+0.00491168]`；方向为正但未达到 `+0.010` 效应门槛，记为 `INCONCLUSIVE_SINGLE_SEED`，不运行 matched seeds；
- SDR、FALS 均未通过单 seed 筛选，ADAS/SLDR 已被前置门控关闭；没有方法达到 dev confirmation，final reserve 保持 manifest-only 且从未访问；
- Dataset V2 台账已闭环，当前没有允许继续执行的训练或 final 动作。

本轮结论边界：

- 所有 Dataset V2 token 都来自曾用于 Stage-2 SFT 的 103,288 条源数据；新 dev/final 对新 RL train 是 log-disjoint，但不是 SFT-unseen；
- 因此本轮可以判断不同 RL selector/reward 在同一 SFT 起点上的相对作用，不能把结果表述为对全新驾驶域的最终泛化证明；
- `ADAS-G4-current` 只代表本文冻结的当前 G4 实现，不代表论文使用 32 次 selector rollout 的 paper-style ADAS；若要评估 ADAS-32，必须另建预注册台账；
- `FALS-G4` 当前按 `pdms_scaled` 排序，ADAS 当前按 raw `pdms` 统计；因此二者各自相对 Random 的 contrast 有效，但 `FALS−ADAS` 不能解释为“只改变排序公式”的单变量比较；
- 不测试 ADAS+FALS hybrid，不用多个 trick 的组合结果替代单因素结论。

## 2. 冻结数据快照

### 2.1 已有资产

| 资产 | 路径 | 数量 / SHA-256 | 当前状态 |
| --- | --- | --- | --- |
| 源 RL parquet | `data/hf_dataset/QA_navtrain_poutine_style_full/data/train.parquet` | 103,288 rows；`86db9581c4bf29552822fdcc7c6bc71dee4a5d7f78c0f9c44b262bad4048f5dd` | 只读上游 |
| V2 train parquet | `data/dataset_v2_20260825/hf/train.parquet` | 8,000 rows；`8b4a059063b07aee508894711155ee19402da01ae3bec0a9dd80166830996639` | 已生成 |
| V2 dev parquet | `data/dataset_v2_20260825/hf/test.parquet` | 2,000 rows；`e65f135d623394b7bdf9b419ca08a1f28467a97577a4f3b1243a43d876bb75c6` | 已生成 |
| Candidate | `manifests/dataset_v2_20260825/selector_pool_8000.txt` | 8,000；`74aa677d1d3c59cd07cb6e3a9666ef18b08b384d83a5f76aa35f2cf8f5430f15` | 已冻结 |
| Phase-1 pool | `manifests/dataset_v2_20260825/selector_pool_phase1_6000.txt` | 6,000；`37a689057cdbc863aeaf104c3c2ea6b200e061181e0c42131300d4f5104fc44a` | 已冻结 |
| Extension | `manifests/dataset_v2_20260825/selector_extension_2000.txt` | 2,000；`355b5990fba8392f4b0ecedf26c28d06a589f0c4a15f51dc5ed35a6167727773` | 仅按门控启用 |
| Random-1K | `manifests/dataset_v2_20260825/random_1k.txt` | 1,000；`7bc6b6eb5873f4cf9691d3da5621098201f31a545f0a918d94f731f55966927c` | 已冻结 |
| Dev | `manifests/dataset_v2_20260825/dev_2000.txt` | 2,000；`2ab37f670c8c0e1b2479d7c3ffbb51f47a344bce8b192c8ac47c705512f54f52` | 正式 exploratory dev |
| Final reserve | `manifests/dataset_v2_20260825/final_reserve_1000.txt` | 1,000；`fa779ec1dad41db6412f02c285270fefdcfe6604f3ab0510876bc2b715f8b7db` | 暂不访问 |
| CAM_F0 | `data/dataset_v2_20260825/sensor_blobs/trainval` | 10,000 files，missing=0，unexpected=0 | 已下载 |
| Metric cache | `exp_root/metric_cache_dataset_v2_20260825` | 10,000 `metric_cache.pkl`，约 4.1 GB | 已生成 |
| Cache manifest | `manifests/dataset_v2_20260825/cache_10000.csv` | 10,000 data rows；`063ed7b1d58ed8ef4885d21782fb62733836960b9f3c4287167a6df770226772` | 已冻结 |

### 2.2 Split 事实

| Split | Tokens | Logs | Straight / Left / Right | 每 log 上限 |
| --- | ---: | ---: | --- | ---: |
| Candidate | 8,000 | 793 | 5,073 / 2,011 / 916 | 25 |
| Dev | 2,000 | 196 | 1,268 / 503 / 229 | 15 |
| Final reserve | 1,000 | 161 | 634 / 251 / 115 | 10 |
| Random-1K | 1,000 | 493 | 634 / 251 / 115 | 5 |

必须始终成立：

- candidate、dev、final reserve 的 token overlap 均为 0；
- candidate、dev、final reserve 的 log overlap 均为 0；
- Random-1K 完全属于 Phase-1 6K；
- V2 active 10K 不包含旧 5,656 个精确 token；
- 所有 V2 prompt 使用 4-second 描述，不允许重新引入源 parquet 的 5-second 错误。

### 2.3 `V2-D0`：形式化数据冻结

在任何 GPU inference 前完成以下事项：

1. 更新 `dataset_card.json` 和 `acceptance_report.json`，明确 active 10K image/cache 已完成、final reserve 仍为 manifest-only；
2. 验证 10,000 个 image path 可读、10,000 个 cache token 与 `cache_10000.csv` 完全一致；
3. 移除或归档两个失效 PID 文件，不把 stale PID 当成仍在运行；
4. 将 `projects/dataset_v2/`、metric-cache builder 修改及相应测试提交到一个确定 commit；
5. source worktree 必须 clean，记录 source commit、Stage-2 model hash、数据 hash 和磁盘余量；
6. 新建 Dataset V2 launcher。它必须显式接收 V2 parquet、manifest、cache 和 experiment root，禁止 fallback 到旧 `metric_cache_released_5656` 或旧 566-token dev；
7. 生成 `V2_DATA_FROZEN` 标记。已有标记时禁止覆盖数据或 manifest；任何内容变化都必须升级 dataset version。

通过条件：上述七项全部满足。失败时只修复数据/入口，不启动 selector 或训练。

## 3. 公平比较与唯一变量

### 3.1 Reward 比较

以下三组必须使用字节级相同的 Random-1K、样本顺序、Stage-2、训练 seed 和优化配置：

- Random + raw-PDMS；
- Random + SDR；
- Random + SLDR。

唯一允许变化的是 reward entrypoint。若 manifest 或顺序不同，则 SDR/SLDR 独立作用不可识别。

### 3.2 Selector 比较

以下三组固定使用 SDR：

- Random-1K + SDR；
- ADAS-1K + SDR；
- FALS-1K + SDR。

ADAS/FALS/Random 的 token 身份不同是 selector 的处理变量，不是不公平。公平性由以下共同约束保证：

- 同一个 candidate universe；
- 同一个冻结 Stage-2 selector rollout bank；
- 每个 manifest 恰好 1,000 个唯一 token；
- intent 配额恰好为 straight 634、left 251、right 115；
- 每个 log 最多 5 条，至少覆盖 200 个 log；
- 相同训练 scene、trajectory、optimizer-step 和 reward-query 预算；
- 相同 Stage-2 初始化、训练 seed、生成配置和 dev；
- 不按照 reward、PDMS、variance 或 headroom 做额外“匹配”，因为这些正是 selector 要利用的信号。

允许 ADAS/FALS/Random 自然重叠，不强制三套 manifest 互斥。必须报告两两 intersection、Jaccard、unique/effective logs、intent 和 train-signal geometry。

本轮只分别估计 `FALS−Random` 与 `ADAS−Random`。由于 FALS 与 ADAS 使用的 selector score basis 也不同，不把 `FALS−ADAS` 点差写成两种数学规则的纯因果差异。

### 3.3 Extension 的共同处理规则

先只使用 Phase-1 6K。

- 如果 ADAS 和 FALS 都能在 6K 内满足 1,000 条、固定 intent 配额和 log cap，则沿用现有 Random-1K；
- 如果任一 selector 因为 train-only eligible/cutoff 不足而需要 Extension 2K，则共同 candidate 升级为 8K；必须从 8K 重新生成 Random、ADAS、FALS 三个 manifest，并重新冻结三者 hash；
- 不允许 ADAS 使用 8K、FALS/Random 使用 6K；
- 是否启用 Extension 只能由 train-side manifest 可行性决定，不得读取 dev。

## 4. 方法定义

### 4.1 Reward

- `raw-PDMS`：`compute_score_group_raw_pdms`，训练 scalar 为 NAVSIM `pdms`；
- `SDR`：`compute_score_group_fast`，训练 scalar 为当前生产 `pdms_scaled`；
- `SLDR-current`：`compute_score_sldr`，完整沿用当前生产实现。不得在看过 V2 dev 后调整 `0.5/0.1/0.6` 等系数；
- 所有 run 都持久化 raw PDMS、PDMS scaled、Safe、Collision、DAC、Progress、TTC、Comfort，训练 reward 均值不能充当最终效果指标。

### 4.2 Random

主 seed 使用已冻结的 `random_1k.txt`。它同时是全部 reward 消融的唯一 anchor manifest。

### 4.3 FALS-G4

在冻结 selector rollout bank 上，对每个 token 的四个 `pdms_scaled` 计算：

\[
\text{difficulty}=1-\bar r,
\qquad
\text{headroom}=r_{\max}-\bar r,
\qquad
\text{FALS}=\text{difficulty}\times\text{headroom}.
\]

按 intent 分层后，从高到低贪心选择，同时执行每-log上限 5。完全同分时使用固定 salted SHA-256 tie-break，不使用 token 字典序。不得根据 dev 修改 score、配额或 cutoff。

该 contrast 识别的是“当前 FALS-SDR 场景选择流程在固定 SDR 训练下，相对 Random 的增益”。它不单独识别 FALS 公式与 `pdms_scaled` 作为 selector score basis 的交互；若后续需要该问题，必须额外预注册 `FALS-raw`，不能从本轮结果反推。

### 4.4 ADAS-G4-current

主定义只复现当前已审计的 G4 train-only 规则：

- selector rollouts/group：4；
- `std_threshold=0.01`；
- `p_est=pdms_mean/pdms_range`；
- `diversity_metric=p_est^4+(1-p_est)^4 < 0.20`；
- predicted-std confidence error `<0.10`；
- 所有 eligible `p_est` 必须位于 `[0,1]`；
- gate 后按 intent 配额、每-log上限和固定 hash 顺序选择 1,000 条，不允许 Random 补齐。

该方法必须写作 `ADAS-G4-current`。它评估当前单卡 G4 可执行实现，不把结果外推为 paper-style ADAS-32。

## 5. 前置机制实验

### 5.1 `V2-I0`：图像敏感性门控

目的：排除 Stage-2 基本忽略图像、导致后续 selector/reward 实验无法解释为视觉驾驶优化的可能。

协议：

- 从 Phase-1 6K 按 intent 和 log 固定 256 个 token，生成 `image_sensitivity_256.txt`；
- 使用冻结 Stage-2 和 deterministic decode，分别输入正确图像与跨-log salted-hash 打乱图像；
- prompt、token、decode、cache 与 reward 完全相同；
- 报告 trajectory/response 改变率，以及 correct minus shuffled 的 PDMS、Safe 和六个 NAVSIM 分项；
- 以 log 为 cluster 做 20,000 次 paired bootstrap。

科学通过条件：

- correct image 的 mean PDMS 至少比 shuffled image 高 `0.01000`；
- PDMS paired cluster-bootstrap 95% CI 下界大于 0；
- parse rate 均至少 99.5%，不存在由格式失败制造的差异。

失败分支：暂停全部正式 GRPO。结果记录为“Stage-2 图像敏感性不足”；先决定是否修复 SFT，不把后续结果表述为视觉 selector/reward 的有效性。

### 5.2 `V2-S0`：selector 稳定性 pilot

目的：判断一次 G4 rollout 是否足以把 ADAS/FALS 当作稳定的场景属性，而不是一次随机采样结果。

协议：

- 从 Phase-1 6K 固定 500 个分层 token；
- 冻结 Stage-2、temperature/top-p=`1.0/1.0`、parser、reward server 和 4 个独立 generation seed：`20260825`、`20260826`、`20260827`、`20260828`；
- seed `20260825` 是主 G4 block，可在 `V2-S1` 复用；四个 block 在单卡上严格顺序执行，不并发；
- 每个 token 生成 16 条轨迹，拆成 4 个互不重叠的 G4 block；
- 每个 block 独立计算 ADAS eligibility 和 FALS ranking；
- dev/final 不得访问。

门控：

- ADAS：4 个 block 的 eligible ratio coefficient of variation `<=0.20`，median pairwise membership Jaccard `>=0.50`；
- FALS：4 个 block 的 rank Spearman median `>=0.60`，Top-25% median pairwise Jaccard `>=0.50`；
- 两种方法的所有统计 finite，四 rollout 覆盖完整；
- 失败的方法不生成正式 manifest，也不启动对应训练。稳定性失败本身记为“当前 G4 预算下 selector 不可靠”；不得根据 pilot 临时调整阈值。

协议修订（2026-08-26，用户指令）：原方案为 8 个 block、ADAS `0.15/0.60`、FALS `0.70/0.60`。在 block 1–4 已运行完成但尚未计算稳定性统计时，为降低执行成本，主分析改为前 4 个 shared block 和上述较宽门槛；已启动的 seed `20260829` block 5 等待完成后只作为敏感性证据，不进入主门控，seed `20260830`–`20260832` 不再执行。该修改发生在部分数据生成后，因此 S0 及后续 selector 归因一律标记为 `EXPLORATORY_PROTOCOL_AMENDED`，不写作严格预注册确认。

### 5.3 `V2-S1`：共享 selector rollout bank

只有 `V2-I0` 通过后执行：

- 对 Phase-1 6K 每 token 生成恰好 4 条轨迹，共 24,000 条；
- 500-token pilot 的预注册主 G4 block可以复用，但必须与其余 token 使用相同生成 seed 定义；
- 若 Extension 被共同启用，再对额外 2K 生成 8,000 条，最终 common bank 为 32,000 条；
- 每条保存 raw response、parsed trajectory、parse flag、raw PDMS、SDR 和全部 NAVSIM 分项；
- 保存 token coverage、group-size、seed、model/data/source hash。

### 5.4 `V2-M0`：selector manifest 冻结

按第 3、4 节生成 Random/ADAS/FALS manifest，并在读取任何 V2 dev output 前完成：

- token 数、唯一性、candidate membership；
- intent 精确配额；
- 每-log上限；
- ADAS eligible pool 和逐级 gate 计数；
- FALS cutoff、cutoff tie 数和 hash tie-break；
- 三 selector 两两 overlap/Jaccard；
- mean reward、mean group std、exact-zero ratio、headroom；
- manifest SHA-256 和完整构建配置。

训练信号改善只能说明 selector 按设计改变预算分配，不得据此宣称模型效果。

### 5.5 `V2-R0`：SDR/SLDR advantage-geometry replay

在共享 rollout bank 上使用生产代码复算 raw、SDR、SLDR。SDR 无论 geometry 是否优于 raw，都必须进入正式消融；SLDR 只有全部门控通过才允许正式训练。

SLDR 门控沿用旧 S0，避免根据新数据重新发明标准：

- 数据、字段和生产 reward 复算完整；
- safe 语义不存在未解释的系统性错标；
- 至少 10% 的 G4 group 满足 `mean(|A_SLDR-A_SDR|) >= 0.10`；
- 实质变化 group 中至少 50% 为 mixed-safety；
- exact-zero group 相对 SDR 至少降低 5 个百分点；
- SLDR 新偏好轨迹在 Collision、DAC、Progress 上不存在 cluster-bootstrap CI 完全低于 0 的一致性退化。

任一科学门控失败：`V2-R4-SLDR` 标记为 `SKIPPED_BY_GATE`，不调系数、不改 safe 判定、不做 dev sweep。若要修正 safe 语义，必须命名为新的 `SLDR-v2` 并另行预注册。

## 6. 正式训练矩阵与执行顺序

### 6.1 共同协议

| 项目 | 冻结值 |
| --- | --- |
| 初始化 | 每个 run 从同一 `models/sft_stage2` 独立开始；D0 固化 model hash |
| GRPO group size | `G=4` |
| Train manifest | 恰好 1,000 scene，每 scene 一次 group exposure |
| Steps | 250 optimizer steps，4 scene groups/step |
| Train trajectories | 4,000 = 1,000 × 4 |
| LoRA / optimizer | 继承旧 R4 的 rank 8、target modules、LR、KL、clip、bf16 等 resolved config |
| Train generation | temperature/top-p `1.0/1.0` |
| Dev generation | temperature/top-p `0.6/0.95`，每 token 1 response，固定 seed |
| Checkpoint | 可保存 step125 供恢复，但科学评估只读取预注册 step250；禁止 checkpoint shopping |
| Main seed | `20260825` |
| Confirmation seeds | `20260826`、`20260827`，只对晋级 contrast 执行 |
| Dev | 同一 `dev_2000.txt`，每个正式 model 只生成一次 |
| 统计 | 20,000 次 paired log-cluster bootstrap；禁止把 2,000 token 当作完全独立样本 |

### 6.2 技术 smoke

在正式 dev lock 前运行：

- `V2-T0-SDR`：Random 子集、10 steps、40 train trajectories、无 dev；
- `V2-T0-RAW`：相同 token/步骤，只切换 raw reward；
- `V2-T0-SLDR`：仅在 `V2-R0` 通过时运行；
- ADAS/FALS 不需要单独训练 smoke，只需验证 manifest 能被同一 SDR launcher 加载。

smoke 只判断路径、显存、reward、parser、checkpoint 和资源回收，不产生科学结论。

### 6.3 正式运行顺序

| 顺序 | ID | Selector | Reward | 直接对照 | 状态 |
| ---: | --- | --- | --- | --- | --- |
| 1 | `V2-E0` | 无训练 | Stage-2 | — | `COMPLETE` |
| 2 | `V2-R4-SDR` | Random-1K | SDR | E0 仅作净更新参考 | `COMPLETE / POSTPROCESS_REPAIRED` |
| 3 | `V2-R4-RAW` | 同一 Random-1K | raw-PDMS | R4-SDR | `COMPLETE / SDR_NEGATIVE_OR_TRADEOFF` |
| 4 | `V2-F4-SDR` | FALS-1K | SDR | R4-SDR | `COMPLETE / INCONCLUSIVE_SINGLE_SEED` |
| 5 | `V2-A4-SDR` | ADAS-1K | SDR | R4-SDR | `SKIPPED_BY_S0_GATE` |
| 6 | `V2-R4-SLDR` | 同一 Random-1K | SLDR-current | R4-SDR | `SKIPPED_BY_R0_GATE` |

不得为了矩阵完整而绕过门控。ADAS 或 SLDR 被跳过时，门控报告就是该方法在当前定义下的闭环结果。

## 7. 正式 run 技术门控

每个 run 启动前必须检查：

- source commit 与 D0 冻结值一致且 worktree clean；
- Stage-2 model、train/dev manifest、train/dev parquet、cache manifest hash 一致；
- active manifest 与 dev/final token、log overlap 为 0；
- 路径全部位于 Dataset V2 namespace，不引用旧 5,656 manifest/cache/dev；
- GPU 显存不少于 24 GB，端口 8901 空闲，Ray/Gunicorn/旧 trainer 无残留；
- 磁盘可用至少 25 GB；低于 15 GB 立即停止新 run，不删除尚未固化的证据换空间；
- 目标 run/debug/ADAS 目录不存在，禁止静默续接其他方法。

每个完整 run 必须满足：

- `exit_code=0` 且 `COMPLETE` 存在；
- 1,000 个 train group、每组恰好 4 条，共 4,000 条；
- 2,000 个 dev token、每 token 恰好一条；
- train/dev parse rate 至少 99.5%；
- clipping=0，所有 reward/advantage/loss/metric finite；
- 无 OOM、CUDA error、no-space、killed、traceback；
- trainer、Ray、Gunicorn、8901、GPU compute 全部回收；
- resolved config、source/model/input hash、原始 rollout、诊断、指标、成本证据完整。

技术失败发生在 dev access 前：允许最小修复后以明确 `retryN` 新目录重跑。永久 `DEV_ACCESSED` 锁创建后，不允许根据性能、parse 分布或指标修改方法再访问同一 dev。

## 8. 评价指标、直接 contrast 与科学门控

### 8.1 指标

主效果指标使用底层 NAVSIM raw PDMS，避免用某种训练 reward 自己证明自己。完整报告：

- PDMS；
- PDMS scaled；
- Safe；
- Collision；
- DAC；
- Progress；
- TTC；
- Comfort；
- parse/clipping；
- train exact-zero group、mean group std、headroom；
- GPU wall time、reward query、峰值显存和磁盘成本。

### 8.2 四个直接 contrast

\[
\Delta_{SDR}=V2\text{-}R4\text{-}SDR - V2\text{-}R4\text{-}RAW
\]

\[
\Delta_{SLDR}=V2\text{-}R4\text{-}SLDR - V2\text{-}R4\text{-}SDR
\]

\[
\Delta_{FALS}=V2\text{-}F4\text{-}SDR - V2\text{-}R4\text{-}SDR
\]

\[
\Delta_{ADAS}=V2\text{-}A4\text{-}SDR - V2\text{-}R4\text{-}SDR
\]

所有差值方向固定为 candidate minus baseline，并在相同 2,000 dev token 上计算。现有 `compare_paired_rollouts.py` 是 token bootstrap，正式分析前必须提供按 `master_index.csv` 中 `log_name` 重采样的 cluster-bootstrap 入口；不得混用两种 CI 名称。

### 8.3 单 seed 筛选门控

SDR、FALS、ADAS 的晋级条件：

- `Delta PDMS >= +0.01000`；
- Safe、Collision、DAC 的点差均不低于 `-0.00500`；
- PDMS 的 paired log-cluster CI 不得完全低于或等于 0；
- 全部技术门控通过。

SLDR 的晋级条件：

- `Delta Safe >= +0.01000`；
- PDMS 点差不低于 `-0.00500`；
- Collision、DAC 点差均不低于 `-0.00500`；
- Safe 的 paired log-cluster CI 不得完全低于或等于 0；
- `V2-R0` 与全部技术门控通过。

解释规则：

- 达到上述点估计与安全门控：进入 matched seed confirmation，不立即宣称有效；
- 主指标为正但未达阈值或 CI 跨 0：记为 `INCONCLUSIVE_SINGLE_SEED`，不进行系数/cutoff sweep；
- 主指标 `<=0` 或安全门控失败：记为 `NEGATIVE_OR_TRADEOFF`，关闭该分支；
- train variance、zero-group 或 headroom 改善不能覆盖 dev 门控失败。

### 8.4 三 seed 确认

只对单 seed 晋级的 contrast 运行 seeds `20260826`、`20260827`：

- 对照与 candidate 在每个 seed 内使用相同 seed、同一 manifest/顺序和同一初始化；
- 已存在的 seed20260825 anchor 可以复用；缺失的对应 baseline 必须补跑，不能跨 seed 相减；
- 三个 seed 分别报告差值，再报告 seed 均值和 log-cluster 不确定性；
- 只有 3/3 seed 主指标同方向、平均效果达到门槛且汇总 95% CI 下界大于 0，才标记 `CONFIRMED_ON_V2_DEV`；
- selector 还必须同时满足 `V2-S0` 稳定性门控，才能把结论归因于 ADAS/FALS，而不是一次 rollout 排名偶然性。

## 9. Final reserve 一次性确认

`final_reserve_1000.txt` 在 dev 筛选和三 seed 确认完成前保持 manifest-only，不下载图像、不建 cache、不做推理。

只有至少一个 contrast 达到 `CONFIRMED_ON_V2_DEV` 时：

1. 冻结唯一 promoted method、对应 anchor、checkpoint hash 和全部分析代码；
2. 为 final 1K 补齐 CAM_F0 和 metric cache，验证与 candidate/dev log-disjoint；
3. 创建永久 `FINAL_ACCESSED` 锁；
4. 用同一 decode 协议分别评估 Stage-2、对应 anchor 和 promoted method；
5. 不根据 final 结果重选 checkpoint、阈值、selector 或 reward 系数。

Final 判定沿用对应方法的 effect-size 与安全门控。通过时只能写作“在 SFT-seen、RL log-disjoint 的 V2 final reserve 上得到相对确认”，不能写作完整 unseen 泛化。

如果没有方法通过 dev confirmation，则 final reserve 永不访问，本轮以负向/证据不足结论闭环。

## 10. 结果记录模板

每完成一个阶段，在本节追加记录，不改写预注册门槛。

### 记录 V2-001：V2-D0 retry1 asset freeze 与 entrypoint 技术失败

- ID / 状态：`V2-D0 retry1 / TECH_FAILED_ENTRYPOINT`；
- 假设与唯一变量：仅冻结 Dataset V2 数据、代码和执行入口，不做 GPU inference；
- source/model/data/manifest hash：source `b32e26f67231c611ed3dda81ab8a4224b305e81e`；Stage-2 两个权重分片 SHA-256 为 `870666c2...b10f0f`、`4f264c53...8744`；train/dev parquet、cache manifest、final manifest 分别为 `8b4a0590...66996639`、`e65f135d...bb75c6`、`063ed7b1...226772`、`fa779ec1...8b7db`；
- frozen config 与 seed：Dataset `dataset_v2_20260825`，seed `20260825`；新 launcher 显式接收 train/dev parquet、active/final manifest、V2 cache、model、experiment root 和 TensorBoard 目录，不存在旧 5,656 cache 或 566-token dev fallback；
- coverage：active token/image/cache 均为 `10,000`，image 全部可读，cache token 与 `cache_10000.csv` 完全一致；final reserve `1,000` 条，保持 manifest-only；
- 代码与测试：本地 `41 passed, 9 skipped`；远端 `50 passed`，`bash -n`、compile、`git diff --check` 通过；source worktree clean；
- 资源：RTX 4090 24,564 MiB 空闲、8901 空闲、无 Ray/Gunicorn/trainer 残留；冻结后 `/root/autodl-tmp` 可用 `67 GB`；
- 产物与清理：正式报告位于 `experiments/dataset_v2_20260825/v2_d0_data_freeze_retry1/`；两个 V2 stale PID `2523/1518` 已移动到显式 archive；pre-freeze marker 和说明保留在 `v2_d0_data_freeze/`，未删除数据或科学证据；
- 科学边界：D0 只证明数据和入口完整，不产生模型效果结论；
- 门控结论：数据七项检查通过，但 `V2-I0` 准备器从非仓库 cwd 启动时出现 `ModuleNotFoundError: projects`；发生在模型加载前，GPU/query/dev access 均为 0，因此不接受 retry1 为最终 source freeze；
- 唯一下一动作：只修复该 CLI import boundary，远端回归后执行 `V2-D0 retry2`。

### 记录 V2-002：V2-D0 retry2 正式冻结

- ID / 状态：`V2-D0 retry2 / COMPLETE`；
- source：`fc8c27f07a01eeeae95297ea8d360e5a2d75abc2`，远端 status clean；
- 回归：非仓库 cwd CLI `--help` 通过，远端 `51 passed`，launcher `bash -n` 通过；
- 数据/model hash：与 retry1 的 train/dev/cache/final 和 Stage-2 权重 hash 完全一致；metadata/hash manifest 因 source freeze 更新为 `414b019f...94a7d`、`54c06398...1d8f7`、`021f3c00...64a95`；
- 资源与访问：GPU/query/dev access 均为 0，磁盘可用约 `67 GB`；
- 门控结论：D0 正式通过，后续所有 run 必须使用 source `fc8c27f...abc2`；
- 唯一下一动作：`V2-I0`。

### 记录 V2-003：V2-I0 correct-image 首次运行技术失败

- ID / 状态：`v2_i0_correct_seed20260825 / TECH_FAILED_LOCAL_PARQUET_LOADER`；
- source/config：source `fc8c27f07a01eeeae95297ea8d360e5a2d75abc2`；Stage-2、256-token manifest、seed `20260825`、deterministic decode、SDR reward 和所有 frozen input hash 均与预注册一致；
- 覆盖与访问：模型和 vLLM/CUDA graph 已初始化，但 train parquet 加载前失败；rollout `0/256`、reward query `0`，未访问 V2 dev/final；
- 根因：`RLHFDataset` 把存在且 hash 正确的本地 `.parquet` 文件路径直接传给 `load_dataset(data_path, split=...)`，被误判为 Hugging Face dataset identifier；
- 资源：`exit_code=1`；GPU、Ray、Gunicorn、8901 和 launcher 已全部回收；无 OOM、CUDA error、no-space 或 killed；磁盘可用约 `67 GB`；
- 产物：失败目录 `v2_i0_correct_seed20260825/` 原样保留，包含 `run.env`、source/input/model hash、GPU 曲线、reward server log 和完整 traceback；
- 门控结论：纯执行入口技术失败，不产生图像敏感性结论；允许最小修复后使用新目录 retry；
- 唯一下一动作：修复本地 parquet loader，重新冻结 source binding，再执行 correct-image retry1。

### 记录 V2-004：V2-D0 retry3 loader 修复后重新冻结

- ID / 状态：`V2-D0 retry3 / COMPLETE`；
- source：`683c05cd99569a3a69082dc7a244dafd2aa7b78a`，唯一代码变化是恢复 `RLHFDataset` 的显式本地文件加载分支；远端 worktree clean；
- 回归：Dataset V2 tests `7 passed`、远端仓库 tests `70 passed`、launcher `bash -n` 和 compile 通过；真实 V2 parquet 以 `load_dataset('parquet', data_files=...)` 成功读取 `8,000` rows；
- 数据/model：train/dev/cache/final manifest 和 Stage-2 权重内容未变；旧 retry2 marker、dataset card 与 acceptance report 已完整归档到 retry2 证据目录后再生成新 marker；
- 资源与访问：GPU/query/dev/final access 均为 0，磁盘可用约 `67 GB`；
- 门控结论：D0 重新通过；后续所有 Dataset V2 run 必须绑定 source `683c05c...7b78a`；
- 唯一下一动作：`V2-I0` correct-image retry1。

### 记录 V2-005：V2-I0 图像敏感性门控

- ID / 状态：`V2-I0 / COMPLETE`；correct 为 `v2_i0_correct_seed20260825_retry1`，shuffled 为 `v2_i0_shuffled_seed20260825`；
- source/model/input：source `683c05cd99569a3a69082dc7a244dafd2aa7b78a`；同一 Stage-2、256-token/256-log manifest、prompt、SDR reward、seed `20260825`、temperature/top-p `0.0/1.0`；唯一变量为正确 CAM_F0 与预注册跨-log salted-hash shuffled CAM_F0；
- coverage：两侧均为 256 个 token、每 token 1 条、parse `100%`、clipping `0`、全部数值 finite；response 与 trajectory 改变率均为 `100%`；
- correct 指标：PDMS `0.94445120`、PDMS scaled `0.90531663`、Safe `0.99609375`、Collision `1.00000000`、DAC `0.99609375`、Progress `0.88387039`、TTC `1.00000000`、Comfort `0.98046875`；
- shuffled 指标：PDMS `0.54572637`、PDMS scaled `0.52325939`、Safe `0.61328125`、Collision `0.81250000`、DAC `0.75000000`、Progress `0.83660517`、TTC `0.77343750`、Comfort `0.98046875`；
- paired log-cluster difference：PDMS `+0.39872483`，20,000-bootstrap 95% CI `[+0.34305619, +0.45539132]`；PDMS scaled `+0.38205725`、Safe `+0.38281250`、Collision `+0.18750000`、DAC `+0.24609375`、Progress `+0.04726522`、TTC `+0.22656250`、Comfort `0.00000000`；
- 资源与完整性：correct 峰值显存 `20,360 MiB`；两侧均 `exit_code=0`、`COMPLETE`，无 OOM/CUDA/no-space/killed/traceback，GPU/Ray/Gunicorn/8901 全部回收；磁盘可用约 `67 GB`；
- 分析产物：`v2_i0_analysis/i0_report.json`，SHA-256 `01a230fa7c25bb86d2d327057c36a08f9e88b68d039c73752f8882115b634855`；rollout-only 阶段无训练 TensorBoard scalar，不产生 loss/entropy 曲线；
- 门控结论：四项科学门控全部通过，Stage-2 对图像具有显著且方向正确的敏感性，开放 `V2-S0`；
- 唯一下一动作：执行 `V2-S0`；其后续协议修订与结果见记录 V2-006。

### 记录 V2-006：V2-S0 四-block 协议修订与稳定性门控

- ID / 状态：`V2-S0 / COMPLETE / EXPLORATORY_PROTOCOL_AMENDED`；
- source/config：blocks 1–5 使用 source `683c05cd99569a3a69082dc7a244dafd2aa7b78a`、同一 Stage-2、500-token manifest、temperature/top-p `1.0/1.0`、seeds `20260825`–`20260829`；前 4 blocks 为修订后主门控，block 5 仅 sensitivity；
- coverage：每个 block 均为 500 groups、2,000 rollouts、每组严格 4 条，parse `100%`、clipping `0`、全部数值 finite；无 dev/final access；
- ADAS 主门控：eligible ratios `[0.026, 0.028, 0.026, 0.016]`，CV `0.22566773 > 0.20`；membership Jaccard median `0.26764706 < 0.50`，失败；
- FALS 主门控：rank Spearman median `0.86500604 >= 0.60`；Top-25% Jaccard median `0.63159014 >= 0.50`，通过；
- sensitivity blocks 2–5：ADAS CV/Jaccard `0.26762912/0.22875817`，仍失败；FALS Spearman/Top-25% Jaccard `0.84260681/0.61949686`，仍通过；
- 技术与资源：五个 block 全部 `COMPLETE/exit_code=0`，无 OOM/CUDA/no-space/killed/traceback，GPU/Ray/Gunicorn/8901 均回收；磁盘可用约 `67 GB`；
- 分析产物：主报告 SHA-256 `1224e45d0adf9e44dae3a78cf9f996429678e4d276ee424645a4ce02165c868d`，sensitivity 报告 `498a8e5b193da11287f2888f73a0d81088a8a6afc0d9d22b17a5bf376f173633`；rollout-only 阶段无训练 TensorBoard scalar；
- 门控结论：关闭 ADAS manifest/训练分支；开放 FALS。由于协议在部分数据生成后修订，后续 selector 归因不得写作严格预注册确认；
- source 重新冻结：四-block 分析入口 commit `7a904ea33eab068d8e67a0970ad36dc2031ff693` 已通过 `V2-D0 retry4` 并成为后续正式 source；
- 唯一下一动作：执行 `V2-S1`；复用 block 1 的 500×4，并以相同 seed 对剩余 5,500 token 生成 22,000 条。

### 记录 V2-007：V2-S1 共享 selector rollout bank

- ID / 状态：`V2-S1 / COMPLETE`；remaining run 为 `v2_s1_remaining5500_seed20260825`，合并 bank 为 `v2_s1_shared_bank_seed20260825`；
- source/config：source `7a904ea33eab068d8e67a0970ad36dc2031ff693`、Stage-2、seed `20260825`、temperature/top-p `1.0/1.0`、production SDR；复用 S0 block 1 的 500×4，仅对剩余 5,500 token 新生成；
- coverage：remaining `5,500 groups / 22,000 rollouts`；合并后 `6,000 groups / 24,000 rollouts`，每组严格 4 条；合并 parse `23,990/24,000 = 99.9583%`，clipping `0`，全部 reward/metric finite；
- integrity：remaining rollout SHA-256 `089f53f10dc924bb7348c8a8709ffa9bdaddcb60bcd92e688da675447049cc03`；shared bank SHA-256 `a4f1f9ae8ab015ee2784b304125ac1697a5815ac8bda32db352e1d41f389f4d7`；
- 资源：remaining `COMPLETE/exit_code=0`，峰值显存 `20,868 MiB`；无 OOM/CUDA/no-space/killed/traceback，GPU/Ray/Gunicorn/8901 全部回收；未访问 dev/final；
- 产物：shared bank 的 `rollouts.jsonl`、`diagnosis.json`、`bank_metadata.json` 与 `COMPLETE` 已固化；rollout-only 阶段无训练 TensorBoard scalar；
- 门控结论：S1 技术门控通过，开放 `V2-M0`；
- 唯一下一动作：冻结 Random/FALS manifest，ADAS 保持关闭。

### 记录 V2-008：V2-M0 首次构建技术逻辑失败

- ID / 状态：`v2_m0_manifests / TECH_FAILED_CLOSED_ADAS_REENTERED`；
- 根因：构建器在 S0 已关闭 ADAS 后仍计算其 eligible pool，仅得到 106 eligible token，并错误决定 `enable_common_extension`；
- 影响：该报告不作为正式 M0，不启用 extension、不生成或训练 ADAS；未访问 dev output/final，未启动 GPU；
- 证据：失败目录 `v2_m0_manifests/` 保留；只允许让构建器跳过已关闭 ADAS，不调整 selector 阈值、FALS cutoff 或数据；
- 唯一下一动作：最小修复 `--skip-adas`，重新冻结 source 后在新目录执行 M0 retry1。

### 记录 V2-009：V2-D0 retry5 与正式 V2-M0 retry1

- source freeze：`V2-D0 retry5 / COMPLETE`，正式 source `5f65a070181c43d2be9cd19de018cd374d676841`；远端 Dataset V2 tests `9 passed`，数据、manifest、cache 与 Stage-2 hash 均未变化；
- ID / 状态：`V2-M0 retry1 / COMPLETE / EXPLORATORY_PROTOCOL_AMENDED`；report 为 `v2_m0_retry1_report.json`，manifest 目录为 `v2_m0_manifests_retry1/`；`extension_required=false`；
- manifest：Random-1K SHA-256 `7bc6b6eb5873f4cf9691d3da5621098201f31a545f0a918d94f731f55966927c`；FALS-1K SHA-256 `2ea6ed972a885cf6f785640b83d5745e9d4314faec563b476c52f49bb8d5e719`；FALS intent `634/251/115`、496 logs；
- overlap：Random/FALS intersection `164`，Jaccard `0.08932462`；FALS cutoff `0.0069589056`；ADAS 无正式 manifest；
- signal geometry：Random/FALS mean group std `0.056372/0.262651`，exact-zero `28.4%/0%`，headroom `0.045714/0.206239`；这些只证明预算分配改变，不构成模型效果结论；
- 门控结论：FALS 与 Random manifest 冻结成功，ADAS `SKIPPED_BY_S0_GATE`；开放 `V2-R0`；
- 唯一下一动作：在完整 S1 shared bank 上运行 raw/SDR/SLDR advantage geometry。

### 记录 V2-010：V2-D0 retry6 与 V2-R0 advantage geometry

- source freeze：补齐 raw-PDMS geometry 的 commit `3201f9b7f1601f53f23fcb17962ca7216f132258` 经本地 `35 passed, 9 skipped`、远端 `53 passed`、compile 与 `git diff --check` 后完成 `V2-D0 retry6`；所有冻结输入与 Stage-2 hash 未变；
- 首次技术失败：`v2_r0_advantage_geometry_seed20260825 / TECH_FAILED_IMPORT_PATH`；遗漏正式入口要求的 EasyR1 `PYTHONPATH`，在 0 行分析、0 dev/final access 阶段失败，资源无残留；
- ID / 状态：`V2-R0 retry1 / COMPLETE / SLDR_SKIPPED_BY_GATE`；输入为 6,000 groups / 24,000 shared rollouts，G=4、CPU-only、20,000 bootstrap、seed `20260825`，未访问 dev/final；
- coverage/recompute：所有 NAVSIM 字段 `24,000/24,000`；stored SDR 与 production recompute mismatch `0`；raw/SDR/SLDR 均使用 production scalar 定义与真实 GRPO estimator；
- geometry：raw/SDR/SLDR exact-zero ratio `28.55%/28.60%/28.5667%`；SLDR 相对 SDR 仅降低 `0.0333 pp < 5 pp`；`mean(|A_SLDR-A_SDR|) >= 0.10` 的 material groups 为 `28/6000 = 0.4667% < 10%`；其中 strict mixed-safety `26/28 = 92.8571%`；
- safe semantics：46 条 partial-collision rollout 中 28 条被 production `>0` 规则记为 safe，涉及 18 token，构成未解释的系统性宽松错标；
- 新偏好审计：23 个 unsafe new-preference pairs、13 个独立 groups；Collision CI `[+0.21154,+0.67308]`、DAC `[-0.28205,0]`、Progress `[-0.03383,+0.23721]`，三者均未完全低于 0；
- 产物：正式 report SHA-256 `66c1e5f2909b9e092758d26a977fb3a0933d9b70fbc6387f783d388e177559eb`；R0 为 CPU replay，无训练 TensorBoard scalar；
- 门控结论：data/recompute、mixed-safety 和三项 CI 通过；safe semantics、material ratio、exact-zero reduction 失败。`V2-T0-SLDR` 与 `V2-R4-SLDR` 均为 `SKIPPED_BY_GATE`，不得调系数；SDR/RAW 继续；
- 唯一下一动作：顺序执行 `V2-T0-SDR` 和 `V2-T0-RAW`。

### 记录 V2-011：V2-T0-SDR 与 V2-T0-RAW 技术门控

- ID / 状态：`V2-T0-SDR / COMPLETE`、`V2-T0-RAW / COMPLETE`；两个 run 均为 Random-40、10 steps、40 groups×4、160 train rollouts、无 dev/final access；
- source/model/input：source `3201f9b7f1601f53f23fcb17962ca7216f132258`，同一 Stage-2、同一 40-token manifest、seed `20260825`、temperature/top-p `1.0/1.0`；唯一变量为 production SDR 与 raw PDMS reward function；
- 技术门控：两侧均 parse `100%`、clipping `0`、全部 finite、`exit_code=0`；SDR/RAW technical report SHA-256 均为 `092ef883824aed4027dff489259924767a403a5ee1b81df28614bb7a632220da`；
- train signal：SDR/RAW exact-zero group ratio `20.0%/22.5%`、reward mean `0.87763066/0.92183462`、reward std `0.14424684/0.15488074`、headroom mean `0.04607722/0.04126691`；RAW 的 `160/160` 条 `training_reward == pdms`；这些 smoke 指标不构成方法效果结论；
- TensorBoard：两侧各有 2 个 event files、69 个 scalar tags；完整覆盖 policy loss、entropy、KL、clip fraction、grad norm、LR、reward/advantage/return、六项 NAVSIM reward、response length、timing、throughput 和显存；峰值显存均为 `21,224 MiB`；
- 本地证据：已同步至 `artifacts/dataset_v2_20260825/v2_t0_{sdr,raw}_random40_seed20260825/`；清理前归档 `v2_t0_evidence_before_cleanup.tar.gz` SHA-256 为 `bfb7692970c442942cfe1060c4850cc0ccc1d872fdf0d159d670f6747f7933d6`，包含 TensorBoard、training curves、raw/train rollout、technical report、LoRA、配置与日志；
- 空间清理：2026-08-26 15:30（Asia/Shanghai）验收时，精确删除两侧 `global_step_10/actor/model_world_size_1_rank_0.pt`（各 `8,144,550,392` bytes；SDR/RAW SHA-256 `5de8515f...e132d7`、`01883a66...fd71`）及 `optim_world_size_1_rank_0.pt`（各 `29,831,355` bytes；`be576af7...6020d`、`ceb9450d...f6d38`）；该 full-state 不可恢复，但可由保留的 frozen input 重算，LoRA/TensorBoard/rollout/report 均保留；磁盘由 `51 GiB` 可用恢复为 `66 GiB`；
- 资源回收：两侧结束后 GPU、Ray、Gunicorn、8901 和 trainer 均无残留；无 OOM/CUDA/no-space/killed/traceback；
- 门控结论：SDR/RAW 正式训练技术路径均通过；SLDR 继续 `SKIPPED_BY_GATE`；
- 唯一下一动作：以新的永久 dev-access lock 执行 `V2-E0`。

### 记录 V2-012：V2-D0 retry7 与 dev access lock 重新冻结

- ID / 状态：`V2-D0 retry7 / COMPLETE`；source `e51464045fd32a8b05d1a0904bea4ee5f2c9537f`，远端 source clean；
- 变更边界：trainer 只在真正进入 validation 前以排他创建模式建立永久锁；Dataset V2 launcher 新增 `eval`/`val_only` 路径，正式 dev evaluation 和带 final validation 的训练必须显式传入独立 `DEV_ACCESSED` lock，T0 不创建锁；
- 回归：本地 `45 passed, 9 skipped`，远端 `54 passed`，compile、launcher `bash -n`、`git diff --check` 通过；
- 数据/model：active `10,000` image/cache、train/dev/final manifest、parquet 和 Stage-2 权重 hash 均未变化；final reserve 保持 manifest-only；
- 证据：freeze report SHA-256 `ce81e873102ff1dce74e02d60c7a4e918aa702938b3971b296ae91d1cbd68303`，已同步到本地 `artifacts/dataset_v2_20260825/v2_d0_data_freeze_retry7/`；
- 门控结论：dev access boundary 和正式 source binding 通过；后续 Dataset V2 正式 run 必须使用 source `e514640...c9537f`；
- 唯一下一动作：`V2-E0`。

### 记录 V2-013：V2-E0 首次运行与 D0 retry8 技术失败、D0 retry9 再冻结

- E0 首次运行：`v2_e0_stage2_seed20260825 / TECH_FAILED_PRE_DEV_CONFIG_VALIDATION`；source `e51464045fd32a8b05d1a0904bea4ee5f2c9537f`，在 Ray trainer 初始化时因 `rollout.n=1` 被 group-relative advantage 训练校验拒绝；`rollout=0`、无 metrics/TensorBoard、永久 dev lock 未创建，未访问 dev output/final；
- 根因与修复：`val_only` 在 validation 后立即返回，不会计算 GRPO advantage；因此仅让 `val_only` 跳过该训练专属 `n>1` 校验，不改变训练路径、reward、decode 或数据。修复 commit `69559e305b9524cdd998ae92a4e4734e386f47e1` 经本地 `45 passed, 9 skipped`、远端 `54 passed`、compile、`bash -n` 和 `git diff --check` 通过；
- D0 retry8：`TECH_FAILED_ARCHIVE_INPUT_MOVED`；归档 retry7 元数据时误将冻结器仍需读取的 `dataset_card.json` 和 `acceptance_report.json` 一并移走，冻结器在读取 card 前失败；无 GPU/dev/final access，失败目录保留；
- D0 retry9：从 retry7 归档复制回上述两个输入，在新目录完成冻结；source `69559e305b9524cdd998ae92a4e4734e386f47e1`，active `10,000` image/cache、parquet、manifest 和 Stage-2 hash 均未变化，final reserve 仍为 manifest-only；磁盘可用约 `66 GiB`；
- 本地证据：`v2_d0_data_freeze_retry8/` 与 `v2_d0_data_freeze_retry9/` 均已同步；
- 唯一下一动作：在同一永久 E0 dev lock 路径尚不存在的前提下，以新目录执行 `V2-E0 retry1`。

### 记录 V2-014：V2-E0 Stage-2 dev anchor

- ID / 状态：`v2_e0_stage2_seed20260825_retry1 / COMPLETE / TECHNICAL_GATE_PASS`；
- source/config：source `69559e305b9524cdd998ae92a4e4734e386f47e1`，Stage-2、V2 dev parquet、`dev_2000.txt`、production SDR、seed `20260825`、rollout n=`1`、temperature/top-p `0.6/0.95`；永久 `V2_E0_DEV_ACCESSED` lock 已创建；
- coverage：2,000 rows / 2,000 unique tokens，与 dev manifest 顺序和集合完全一致，missing/extra `0/0`；parse `2,000/2,000=100%`、clipping `0`、所有 rollout/metrics numeric finite；`training_reward == pdms_scaled` 为 `2,000/2,000`；
- dev metrics：PDMS `0.9474641282`、PDMS scaled `0.9087265239`、Safe `0.9970000000`、Collision `0.9995000000`、DAC `0.9975000000`、Progress `0.8860965352`、TTC `0.9975000000`、Comfort `0.9915000000`；
- 产物：rollout SHA-256 `517a9f72ceb7c7b9951c7627926bf470923b1d69cd810111f7e3537fc0f045fe`；`final_dev_metrics.json` SHA-256 `18a0f7fbdc3160208cbe52df9d1a6bc1b4869d2bd0d402a5302cb2c926a5dcbd`；完整结果已同步到本地 `artifacts/dataset_v2_20260825/v2_e0_stage2_seed20260825_retry1/`；
- TensorBoard：2 个 event files，其中一个 lifecycle placeholder，另一个含 21 个 `val/...` scalar，覆盖 NAVSIM 分项、reward、parse、prompt/response length 与 clip ratio；E0 为 eval-only、训练 step 为 0，因此不产生 loss、entropy、KL、grad 或 LR 训练曲线；
- 资源：运行约 `54m40s`，GPU CSV 2,940 samples，峰值显存 `20,868 MiB`、峰值利用率 `100%`；终态 GPU、Ray、Gunicorn、8901、trainer 全部回收，错误扫描为 0，磁盘可用约 `66 GiB`；
- final boundary：final reserve manifest 仍为 1,000 unique，run 中不存在 final/reserve/heldout rollout 或结果文件；
- 科学边界：E0 是 Stage-2 anchor，不包含 RL 更新，不与自身构成方法 contrast；
- 唯一下一动作：`V2-R4-SDR`。

### 记录 V2-015：V2-R4-SDR 正式训练、后处理修复与 E0 净更新参考

- ID / 状态：原 run `v2_r4_sdr_random1k_seed20260825 / TRAIN_AND_DEV_COMPLETE / POSTPROCESS_TECH_FAILED`；独立修复 run `v2_r4_sdr_postprocess_fix_2118555 / COMPLETE / TECHNICAL_GATE_PASS`；原始 `exit_code=1`、缺少 `COMPLETE/technical_report.json` 的证据保持不改，修复 run 为 `exit_code=0/COMPLETE`；
- 假设与唯一变量：Random-1K、G=4、production SDR，从冻结 Stage-2 独立初始化；E0 仅作为无 RL 更新净变化参考，正式 reward contrast 仍为后续 `R4-SDR−R4-RAW`；
- source/model/data/manifest：训练 source `69559e305b9524cdd998ae92a4e4734e386f47e1`；CPU-only postprocess 修复 source 为 `2118555bdbfa4375cccec3badb5717c832806d36`；后续正式训练 runtime 从 `69559e3` 只 cherry-pick 同一验证器修复并重绑定为 `3c9bf2b4d60b8e4fadaef395743bd0c4ae07fb29`，没有带入 CDT-HLA 或 launcher 行为变化；train/dev/cache/final manifest 与 Stage-2 hash 均与 V2-014 相同；最终 D0 rebind report SHA-256 `4a4608351375527e05a45329ac57abd9fbca9a885491bb44bcebb150f2cf243d`；
- 原始技术失败：`experiment_log.jsonl` 前 250 行是合法 training step 1–250，第 251 行是合法 `step=250` validation row；旧 `verify_train` 错误要求整份日志的 step 列表恰为 1–250。修复只把 step 顺序与 training clipping 检查限定到非-`val` training rows，finite 检查仍覆盖整份日志；本地 Dataset V2 `11 passed`、Safe-GRPO `36 passed, 12 skipped`，远端合计 `59 passed`，compile、launcher syntax 和 diff check 通过；
- CPU-only 修复边界：只读取既有 train/dev rollout、experiment log 和 TensorBoard，未加载模型、未生成新 rollout、未重新访问 dev/final；technical report SHA-256 `830ddbe8ab1a9432b2f76059a9596012585008370b82806be1f8a17a5e90d591`；
- coverage：250/250 steps、1,000 train groups、4,000 train rollouts、2,000/2,000 dev rollouts；train/dev token coverage 和顺序完整，checkpoint step125/250 与两份 LoRA 均存在；
- parse/clipping/finite：train parse `99.95%`、dev parse `100%`、clipping `0`、全部 rollout/training-log/TensorBoard 数值 finite；TensorBoard 2 个 event files，主 event 含 89 个 scalar tags，覆盖 loss、entropy、KL、clip fraction、grad、LR、advantage、reward、response length、timing、显存和 final validation；
- dev metrics：PDMS `0.9457975527`、PDMS scaled `0.9071007565`、Safe `0.9955000000`、Collision `0.9990000000`、DAC `0.9965000000`、Progress `0.8866671896`、TTC `0.9960000000`、Comfort `0.9915000000`；
- E0 paired log-cluster 参考：2,000 token、196 logs、20,000 bootstrap、seed `20260825`；PDMS `-0.0016665755`，95% CI `[-0.0036328182,-0.0000138123]`；PDMS scaled `-0.0016257674` `[-0.0036497283,+0.0001379996]`；Safe/Collision/DAC/Progress/TTC/Comfort 点差分别为 `-0.0015/-0.0005/-0.0010/+0.00057065/-0.0015/0`；paired report SHA-256 `6eedf8e3ecf263cd455df4f6c0b972a0bebc1f0d49515e51c9630ac0351779c0`；
- 尾部诊断：PDMS token 级改善/不变/退化为 `342/1351/307`；3 个新增 Safe=0 token 贡献总 PDMS 净退化的 `89.24%`，排除它们后其余 1,997 token mean delta 为 `-0.00017960`，因此是小范围灾难性尾部退化而非广泛漂移；
- train signal geometry：exact-zero group `28.5%`、低非零 std `50.6%`、合计 `79.1%` 低于 `std_floor=0.05`；reward mean/std `0.87676242/0.18075473`、PDMS mean `0.91571056`、Safe `0.96975`、headroom mean `0.04622416`；entropy/KL/grad/response length 稳定，无 collapse、爆炸或长度 clipping；
- 成本与证据：GPU sampled wall `16,244 s`、峰值显存 `21,266 MiB`；train/dev/final-metrics SHA-256 分别为 `f2d210aa...a3718`、`db8587d0...c6f3f6`、`42553c20...aaf05`；本地预修复归档 `v2_r4_sdr_evidence_pre_postprocess_fix.tar.gz` SHA-256 `f16556b8d510a3b5b258fdd6cc578c6314c1d2b204b1ffe2e653affdbdbfb84c`，修复 run 和 D0 rebind 也已同步本地；
- 空间：远端约 `50 GiB` 可用，R4-SDR step125/250 的四个 full actor/optimizer state 暂时保留；LoRA、rollout、metrics、TensorBoard、曲线和 hash 均已固化，但没有在空间充足时把有效 full-state 当作“无效中间产物”删除；
- 科学边界：R4-SDR 相对 E0 为轻微负向净更新参考，不能据此单独判断 SDR reward 是否有效；只有同一 Random-1K 上的 `R4-SDR−R4-RAW` 才是预注册直接 contrast；
- 门控结论：训练、dev 和修复后的技术门控全部通过，开放 `V2-R4-RAW`；
- 唯一下一动作：执行 `V2-R4-RAW`，随后统一计算 `SDR−RAW` paired log-cluster contrast 并应用单 seed 门控。

### 记录 V2-016：V2-R4-RAW 正式训练与 SDR−RAW 直接对照

- ID / 状态：`v2_r4_raw_random1k_seed20260825 / COMPLETE / TECHNICAL_GATE_PASS`；2026-08-27 09:37 启动，14:13 收到终态，`exit_code=0/COMPLETE`；
- 假设与唯一变量：与 V2-R4-SDR 使用同一 Random-1K、G=4、seed、Stage-2 初始化、250 steps、训练顺序和 decode；唯一处理变量为 reward 从 production SDR 改为 raw PDMS；直接差值固定为 `candidate R4-SDR minus baseline R4-RAW`；
- source/model/data/manifest：source `3c9bf2b4d60b8e4fadaef395743bd0c4ae07fb29`，D0 rebind report SHA-256 `4a4608351375527e05a45329ac57abd9fbca9a885491bb44bcebb150f2cf243d`；Random-1K SHA-256 `7bc6b6eb5873f4cf9691d3da5621098201f31a545f0a918d94f731f55966927c`，train/dev parquet、dev manifest、cache、Stage-2 与 V2-R4-SDR 相同；source clean；
- frozen config：train temperature/top-p `1.0/1.0`，dev `0.6/0.95`，rollout n=`4`，seed `20260825`，`data.shuffle=false`，reward `compute_score_group_raw_pdms`；永久 `V2_R4_RAW_DEV_ACCESSED` lock 已创建；
- coverage/technical gate：250/250 steps、1,000 train groups、4,000 train rollout、2,000/2,000 dev；train/dev parse `99.95%/100%`、clipping `0`、所有 rollout/log/TensorBoard 数值 finite；step125/250 checkpoint、两份 LoRA 和 optimizer/config 均存在；
- dev metrics：PDMS `0.9485507037`、PDMS scaled `0.9099744527`、Safe `0.9970000000`、Collision `0.9990000000`、DAC `0.9980000000`、Progress `0.8884333023`、TTC `0.9980000000`、Comfort `0.9910000000`；
- SDR−RAW paired log-cluster：2,000 token、196 logs、20,000 bootstrap、seed `20260825`；PDMS `-0.0027531510`，95% CI `[-0.0052398301,-0.0004859856]`；PDMS scaled `-0.0028736962` `[-0.0053737374,-0.0006100691]`；Safe `-0.0015`、Collision `0`、DAC `-0.0015`、Progress `-0.00176611`、TTC `-0.0020`、Comfort `+0.0005`；paired report SHA-256 `db937622baf8c24ee3b58bd246b4bbc8136489e5a2d15c1a07bcfd195300b4fe`；
- train signal geometry：raw reward exact-zero/低非零/below-0.05 group 为 `28.0%/56.9%/84.9%`，reward mean/std `0.91750112/0.17060215`、headroom `0.03970377`；相较 SDR 的 below-0.05 `79.1%`，raw reward 产生更多低方差组，但该训练几何只作解释证据，不覆盖 dev 门控；
- TensorBoard/曲线：递归核验 2 个 event files，主 event `969,026` bytes，共 89 个 scalar tags，覆盖 policy loss、entropy、KL、clip fraction、grad、LR、advantage、reward 与 NAVSIM 分项、response length、timing、显存和 final validation；`training_history.csv`、`training_curves.svg`、curve summary 与 29 条代表样本已保存本地；
- 成本与证据：GPU sampled wall `16,230 s`、峰值显存 `21,294 MiB`、峰值利用率 `100%`；终态 GPU `0 MiB`、8901 已关闭，磁盘剩余约 `33.7 GiB`；train/dev/final-metrics/technical-report SHA-256 分别为 `e3e7940a...aaf0`、`e629e178...bdd4`、`f828934a...3cb`、`830ddbe8...d591`；本地证据归档 SHA-256 `48b1afa076858260af7eada6c93547cbd38fa6b2e81f19cc230eb62e76a6fb1e`，只排除两份各约 8.1 GB 的可重建 full actor state；
- 科学门控：安全三项点差均不低于 `-0.005`，但主 PDMS 为负且 CI 整体低于 0，未满足 `Delta PDMS >= +0.010`；按预注册规则标记 `NEGATIVE_OR_TRADEOFF`，关闭 SDR matched-seed 分支，不做 reward/cutoff sweep；
- 科学边界：该结论只否定当前 Random-1K/G4/seed 下 production SDR 相对 raw-PDMS 的单 seed 晋级，不把 E0 参考或训练方差解释当作直接 contrast，也不提前否定 FALS selector 的独立对照；final reserve 未访问；
- 唯一下一动作：执行 `V2-F4-SDR`，以 FALS-1K 对 Random-1K 的 R4-SDR 做固定 reward 的 selector contrast。

### 记录 V2-017：V2-F4-SDR 正式训练、FALS−Random 对照与台账终结

- ID / 状态：首次 launcher invocation `v2_f4_sdr_fals1k_seed20260825 / PRE_RUN_CLI_FAILED`；正式 run `v2_f4_sdr_fals1k_seed20260825_retry1 / COMPLETE / TECHNICAL_GATE_PASS / INCONCLUSIVE_SINGLE_SEED`；正式 run `exit_code=0/COMPLETE`；
- 首次技术失败：干净 runtime launcher 不支持冗余 `--adv-estimator` 参数，在创建 run 目录、GPU/8901、rollout 和 dev lock 前退出；失败 launcher log 已保留。标准 GRPO 是该 launcher 固定默认值，retry1 只移除多余 CLI 参数，没有改代码、数据、reward、selector 或训练配置；
- 假设与唯一变量：FALS-1K 对 Random-1K；两侧固定 production SDR、G=4、seed `20260825`、同一 Stage-2 初始化、250 steps、训练 decode 与同一 2,000-token dev；直接差值为 `candidate F4-SDR minus baseline R4-SDR`；selector 结论继续标记 `EXPLORATORY_PROTOCOL_AMENDED`；
- source/model/data/manifest：source `3c9bf2b4d60b8e4fadaef395743bd0c4ae07fb29`，相对 R4-SDR 训练 source 只含不改变训练行为的 postprocess 验证器修复；FALS-1K `1,000` token、496 logs，SHA-256 `2ea6ed972a885cf6f785640b83d5745e9d4314faec563b476c52f49bb8d5e719`，intent `634/251/115`；train/dev/cache/final manifest 和 Stage-2 hash 均未变化；
- frozen config：train temperature/top-p `1.0/1.0`，dev `0.6/0.95`，rollout n=`4`，seed `20260825`，`data.shuffle=false`，reward `compute_score_group_fast`；永久 `V2_F4_SDR_DEV_ACCESSED` lock 已创建；
- coverage/technical gate：250/250 连续 steps、1,000 train groups、4,000 train rollout、2,000/2,000 dev；train/dev parse `99.85%/100%`、clipping `0`、全部 rollout/log/TensorBoard 数值 finite；step125/250 checkpoint、两份 LoRA 和 optimizer/config 均存在；
- dev metrics：PDMS `0.9481976883`、PDMS scaled `0.9094625089`、Safe `0.9970000000`、Collision `0.9990000000`、DAC `0.9980000000`、Progress `0.8873330422`、TTC `0.9980000000`、Comfort `0.9915000000`；
- FALS−Random paired log-cluster：2,000 token、196 logs、20,000 bootstrap、seed `20260825`；PDMS `+0.0024001356`，95% CI `[+0.0000634919,+0.0049116752]`；PDMS scaled `+0.0023617524` `[+0.0000577544,+0.0048324406]`；Safe `+0.0015`、Collision `0`、DAC `+0.0015`、Progress `+0.00066585`、TTC `+0.0020`、Comfort `0`；paired report SHA-256 `a5d36e7c402360cc6026c3b0d87cd5683399a62c0d5522a6f50a684b51edb708`；
- token 变化：PDMS 改善/不变/退化为 `322/1351/327`，最小/最大 token delta `-0.96531070/+1.0`；点估计和 cluster CI 方向为正，但并非没有尾部交换；
- train signal geometry：exact-zero/低非零/below-0.05 group 为 `3.1%/26.5%/29.6%`，reward mean/std `0.73082004/0.31384514`、headroom `0.17533093`；相较 Random R4-SDR 的 below-0.05 `79.1%` 显著改善，说明 FALS 成功把预算集中到更有组内差异的样本，但该证据不能替代 dev effect-size 门槛；
- TensorBoard/曲线：递归核验 2 个 event files，主 event `969,026` bytes，覆盖 250-step policy loss、entropy、KL、clip fraction、grad、LR、advantage、NAVSIM reward 分项、response length、timing、显存与 final validation；entropy mean `0.15553671`、KL mean `0.00012848`、grad norm mean/max `0.02270081/0.07933678`、response clip `0`，无 collapse 或数值异常；完整 event、CSV、SVG、curve summary 与代表样本已保存本地；
- 成本与证据：GPU sampled wall `19,151 s`、峰值显存 `21,294 MiB`、峰值利用率 `100%`；终态 GPU/Ray/Gunicorn/trainer 为 0，8901 已关闭，错误扫描 clean，磁盘剩余约 `17.6 GiB`；train/dev/final-metrics/technical-report SHA-256 分别为 `88101072...6aff`、`ea20eccd...6522`、`d389b6cf...e819`、`8c61cafc...1943`；本地证据归档 SHA-256 `8fe6d4e6f00ab52b30b6d3fbbd9deaed803a01a4fd9bd6c8c0f113d3e5e338a3`，只排除两份各约 8.1 GB 的可重建 full actor state；
- 科学门控：PDMS CI 下界大于 0，Safe/Collision/DAC 点差均不低于 `-0.005`，技术门控通过；但 `Delta PDMS=+0.00240014 < +0.01000`，按预注册规则记为 `INCONCLUSIVE_SINGLE_SEED`，不补 seeds `20260826/20260827`，不调整 FALS cutoff 或训练超参数；
- final/停止边界：SDR 为 `NEGATIVE_OR_TRADEOFF`，FALS 为 `INCONCLUSIVE_SINGLE_SEED`，ADAS/SLDR 分别由 S0/R0 关闭；没有方法进入三 seed confirmation，因此 `FINAL_ACCESSED` 不存在，final reserve 保持 manifest-only，不下载图像、不建 cache、不做推理；
- 最终结论：当前 Dataset V2/G4 预算下，raw-PDMS 优于 production SDR；FALS 明显改善训练信号几何并带来小幅正向 dev 效果，但效应不足以晋级；ADAS-G4-current 不稳定，SLDR-current 与 SDR 几何差异不足。本轮按预注册队列闭环停止，不追加组合或 sweep。

### 记录模板

- ID / 状态：`PENDING | RUNNING | TECH_FAILED | SKIPPED_BY_GATE | COMPLETE`；
- 假设与唯一变量：
- 直接对照：
- source/model/data/manifest hash：
- frozen config 与 seed：
- train/dev rollout coverage：
- parse/clipping/finite：
- PDMS / PDMS scaled / Safe / 六分项：
- paired log-cluster difference 与 95% CI：
- train signal geometry：
- GPU/query/wall-time/disk：
- 产物与清理：
- 科学边界：
- 门控结论：
- 唯一下一动作：

### 主结果表

| ID | 状态 | PDMS | PDMS scaled | Safe | Collision | DAC | 直接差值 / CI | 决策 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| V2-E0 | COMPLETE | 0.947464 | 0.908727 | 0.997000 | 0.999500 | 0.997500 | — | Stage-2 anchor |
| V2-R4-SDR | COMPLETE | 0.945798 | 0.907101 | 0.995500 | 0.999000 | 0.996500 | vs E0 PDMS `-0.001667` `[-0.003633,-0.000014]` | R4-RAW anchor |
| V2-R4-RAW | COMPLETE | 0.948551 | 0.909974 | 0.997000 | 0.999000 | 0.998000 | SDR−RAW PDMS `-0.002753` `[-0.005240,-0.000486]` | SDR negative/tradeoff |
| V2-F4-SDR | COMPLETE | 0.948198 | 0.909463 | 0.997000 | 0.999000 | 0.998000 | FALS−Random PDMS `+0.002400` `[+0.000063,+0.004912]` | inconclusive single seed |
| V2-A4-SDR | SKIPPED_BY_GATE | — | — | — | — | — | ADAS−Random | S0 failed |
| V2-R4-SLDR | SKIPPED_BY_GATE | — | — | — | — | — | SLDR−SDR | R0 failed |

## 11. 证据保存与空间闭环

每个正式 run 至少保留：

- `run.env`、resolved config、source commit/status；
- model/data/manifest/input hash；
- train/dev tokens；
- raw train/dev rollout 和 parsed trajectory；
- train diagnosis、final metrics、paired cluster-bootstrap report；
- policy loss、entropy、KL、clip、grad、LR、advantage、reward 分项、response length、timing 和显存曲线；
- LoRA adapter、optimizer/config、代表样本、`COMPLETE`、`exit_code` 和 result hash。

空间规则：

- run 顺序执行，不并行保留多个 8 GB full actor state；
- 只有 LoRA、评估、rollout、曲线和 hash 全部固化且进程回收后，才允许精确删除该 run 的 full actor model；
- 不删除 Dataset V2、Stage-2、旧实验 ledger/rollout/report；
- 大日志在结果 hash 固化后可以压缩，不以删除原始科学证据换空间；
- 任何删除都必须记录精确路径、大小、时间和不可恢复性。

长任务监控规则：Luna 必须在单次 turn 内启动服务器侧阻塞 watcher，以 `while + sleep` 持续检查终态、错误、磁盘和 GPU/8901；正常快照不返回 final，只有 `COMPLETE`、`FAILED` 或明确异常才结束并完成验收。主进程启动 run 后直接长时间暂停等待 Luna 唤醒，不做固定间隔轮询，也不承担正常进度兜底。

## 12. 当前执行队列

严格按以下顺序推进，每次只开放一个动作：

1. `V2-D0`：完成数据卡、cache/image、代码 commit 和 V2 launcher 的形式化冻结；
2. `V2-I0`：图像敏感性；
3. `V2-S0`：500×32 selector 稳定性 pilot；
4. `V2-S1`：Phase-1 6K×4 共享 rollout bank；
5. `V2-M0`：冻结 Random/ADAS/FALS manifest；
6. `V2-R0`：raw/SDR/SLDR advantage geometry；
7. `V2-T0-*`：无 dev 技术 smoke；
8. `V2-E0 → V2-R4-SDR → V2-R4-RAW → V2-F4-SDR → V2-A4-SDR → V2-R4-SLDR`，条件分支按门控跳过；
9. 统一 paired log-cluster 分析；
10. 只对晋级 contrast 做两组 matched seed；
11. 只对确认方法访问 final reserve；
12. 填写最终结论并停止，不追加未预注册组合。

当前没有允许继续执行的训练或 final 动作；台账已闭环停止。
