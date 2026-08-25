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

截至 2026-08-25 的状态：

- Dataset V2 的 8,000 candidate、2,000 dev、10,000 张 CAM_F0 和 10,000 份 metric cache 已存在；
- `random_1k.txt` 已冻结，并且 1,000 个 token 全部属于 Phase-1 6K candidate；
- ADAS/FALS manifest 尚未生成；
- Dataset V2 尚未产生正式 selector rollout、GRPO checkpoint 或 dev 结果；
- 数据资产已完成，但形式化冻结尚未完成：`dataset_card.json` 和 `acceptance_report.json` 仍把 image/cache 标成 `deferred`，两个 PID 文件已经失效但仍残留，远端数据构建代码尚未提交且 source worktree 非 clean；
- 当前唯一下一动作是 `V2-D0`，不得跳过它启动 GPU 实验。

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
- 冻结 Stage-2、temperature/top-p、parser、reward server 和 8 个独立 generation seed；
- 每个 token 生成 32 条轨迹，拆成 8 个互不重叠的 G4 block；
- 每个 block 独立计算 ADAS eligibility 和 FALS ranking；
- dev/final 不得访问。

门控：

- ADAS：8 个 block 的 eligible ratio coefficient of variation `<=0.15`，median pairwise membership Jaccard `>=0.60`；
- FALS：8 个 block 的 rank Spearman median `>=0.70`，Top-25% median pairwise Jaccard `>=0.60`；
- 两种方法的所有统计 finite，四 rollout 覆盖完整；
- 失败的方法不生成正式 manifest，也不启动对应训练。稳定性失败本身记为“当前 G4 预算下 selector 不可靠”；不得根据 pilot 临时调整阈值。

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
| 1 | `V2-E0` | 无训练 | Stage-2 | — | `PENDING` |
| 2 | `V2-R4-SDR` | Random-1K | SDR | E0 仅作净更新参考 | `PENDING` |
| 3 | `V2-R4-RAW` | 同一 Random-1K | raw-PDMS | R4-SDR | `PENDING` |
| 4 | `V2-F4-SDR` | FALS-1K | SDR | R4-SDR | `BLOCKED_BY_S0_M0` |
| 5 | `V2-A4-SDR` | ADAS-1K | SDR | R4-SDR | `BLOCKED_BY_S0_M0` |
| 6 | `V2-R4-SLDR` | 同一 Random-1K | SLDR-current | R4-SDR | `BLOCKED_BY_R0` |

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
| V2-E0 | PENDING | — | — | — | — | — | — | — |
| V2-R4-SDR | PENDING | — | — | — | — | — | vs E0 | — |
| V2-R4-RAW | PENDING | — | — | — | — | — | SDR−RAW | — |
| V2-F4-SDR | BLOCKED | — | — | — | — | — | FALS−Random | — |
| V2-A4-SDR | BLOCKED | — | — | — | — | — | ADAS−Random | — |
| V2-R4-SLDR | BLOCKED | — | — | — | — | — | SLDR−SDR | — |

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

当前唯一允许执行的动作：`V2-D0`。
