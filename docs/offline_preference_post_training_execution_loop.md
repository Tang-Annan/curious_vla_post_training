# Curious-VLA 资源受限离线偏好后训练执行台账

> 生效日期：2026-08-15。本文档是路线转向后的唯一实时执行台账；
> [`post_training_execution_loop.md`](post_training_execution_loop.md) 只保留为 GRPO 阶段历史证据，不再更新。
> 服务器日志、配置、manifest、模型和指标仍是原始证据，本文档只记录可追溯事实、预注册门控与下一步。

## 1. 当前决策快照

- 当前证据基线：`023139a`；开发分支为 `codex/offline-preference-post-training`，P0 执行 source `c36767a`，P1-S 执行 source `fe6eac6`，原 `codex/post-training-analysis` 冻结为 GRPO 证据分支。
- 路线结论：停止围绕 GRPO estimator、sampling cap、reward coefficient 或 std normalization 继续追分；已完成的 E0–E4、R1–R3 作为前半段证据冻结。
- 新核心问题：在固定 rollout / reward-query 预算下，能否把已有 trajectory-level safety/quality reward 转成离线 preference supervision，并以更低在线成本获得比 RSFT、普通 DPO 和现有 FALS-GRPO 更稳定的策略。
- 当前唯一动作：P0、P1-S 已闭环；进入 P1-M，只按冻结规则构造 PDMS-Pair、Safety-Gap-Pair 与 chosen-only RSFT 数据，先验收 pair 数量、比例、确定性与泄漏，不启动 GPU。
- 冻结开发集：566 token；每个正式方法只允许一次最终 dev 评估，不用 dev 选择 pair 阈值或训练超参数。
- 旧 565-token held-out 已访问 520 条并永久失去 unseen 资格；部分 rollout 已删除，`F1_HELDOUT_ACCESSED` 永久锁保留，禁止补跑剩余 45 条或把它用于最终确认。
- P0 证明当前服务器资产无法建立合格的新 final set：旧 manifest 外 97,632 个 token 的 log 可用，但 CAM_F0 图像可用数为 0。P6 预注册为不执行 final-set 推理，除非用户未来明确扩展数据下载范围并在任何新方法 dev 结果产生前重新立项。
- 明确排除：不下载或评估官方 final checkpoint；不重做完整 FTE；不开展 ELF-VLA teacher feedback；不继续 Dr.GRPO、Dynamic Sampling、SLDR/Std-Floor 组合或 sweep。

| 阶段 | 状态 | GPU | 回答的问题 | 下一动作 |
| --- | --- | ---: | --- | --- |
| P0 | 已完成 | 0 | 如何保留 GRPO 证据、隔离新路线并修正 final-set 边界 | 无新 final set；旧 split 永久封存 |
| P1-S | 已完成 | 0 | 18,100 条 scored rollout 是否具备可训练表示 | 18,088 条候选通过 join/round-trip，允许 P1-M |
| P1-M | 执行中 | 0 | PDMS 与 Safety-Gap 能构成多少可信 pair | 只用 train 构建统计与冻结数据集 |
| P2 | 被 P1-M 阻塞 | 低 | 只学习 chosen trajectory 是否足够 | chosen-only RSFT |
| P3 | 被 P1-M 阻塞 | 中低 | pairwise PDMS supervision 是否优于 RSFT/GRPO | 普通 trajectory DPO |
| P4 | 被 P1-M 阻塞 | 中低 | safety-aware pair mining 是否带来独立增益 | Safety-Gap DPO |
| P5 | 条件执行 | 中 | policy 更新后刷新一次 pair 是否继续提升 | 只允许一轮 refresh |
| P6 | 最终阶段 | 中 | 第二 seed、behavior audit 与新 final set 是否确认结论 | 方法、checkpoint、阈值全部冻结后执行 |

## 2. 分支与代码处理决策

### 2.1 新开分支，但不做破坏性回退

推荐从当前证据完整的 HEAD 新建：

```bash
git switch -c codex/offline-preference-post-training
```

执行边界：

1. `codex/post-training-analysis` 保留为 GRPO 证据分支，不继续开发。
2. 新分支继承证据提交 `023139a` 及本台账；不得使用 `git reset --hard`，也不批量 revert R1–R3。
3. R1/R2/R3 是可复核的科学负结果，不是需要抹掉的历史；其新增开关默认不启用，并不改变普通 GRPO 默认路径。
4. DPO/RSFT 使用独立的 LLaMA-Factory 环境，正常情况下不修改 `EasyR1/verl/trainer/`。
5. 只有旧代码与新路径发生真实冲突时，才在新分支做一个可解释的最小隔离修改；不得以“代码看起来多”为理由清理历史实现。
6. 若最终需要面试展示用的精简分支，在 P6 完成后再从 `main` 建 release 分支并 cherry-pick 已验证的基础设施、FALS 证据和 preference 实现；开发期不提前整理历史。

### 2.2 新路线的最小代码边界

| 位置 | 责任 | 当前状态 |
| --- | --- | --- |
| `projects/safe_preference/build_preference_dataset.py` | schema 审计、join、pair mining、LLaMA-Factory 数据输出 | P1-S schema/join/processor 审计已实现；P1-M 待实现 |
| `projects/safe_preference/analyze_preference_dataset.py` | pair 数量、gap、长度、安全构成与 hash 汇总 | 待实现；只有 builder 过大时才拆分 |
| `sft/preference/` | P2/P3/P4 的冻结 YAML 与 export YAML | 待实现 |
| `scripts/run_safe_preference_experiment.sh` | smoke、正式训练、导出、状态与证据门控 | 待实现 |
| `scripts/run_safe_preference_eval.sh` | 复用冻结 NAVSIM dev 协议评估任意 merged model | 待实现 |
| `tests/test_safe_preference.py` | pair 规则、泄漏、round-trip、确定性与失败语义 | P1-S 5 项测试通过；P1-M 规则测试待补 |
| `projects/safe_grpo/`、`scripts/run_safe_grpo_experiment.sh` | 历史 GRPO 证据 | 只读，不扩展 |

不要先写新的 trainer、reward model 或通用数据框架。第一版只实现一个确定性 builder、三份配置和两个薄启动器。

## 3. 冻结证据与结论边界

### 3.1 同协议 566-token dev

| 方法 | 唯一变化 | PDMS scaled | PDMS | Safe | Collision | DAC | Progress | TTC | Comfort | 冻结结论 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| E0 | Stage-2 | 0.65938 | 0.68361 | 0.72438 | 0.96643 | 0.75265 | 0.91135 | 0.94876 | 0.92049 | 基线 |
| E1 | Random 1k + Vanilla GRPO | 0.64281 | 0.66691 | 0.70671 | 0.95936 | 0.74205 | 0.91071 | 0.94170 | 0.91873 | 随机 GRPO 退化 |
| E2 | FALS 1k + Vanilla GRPO | 0.67230 | 0.69758 | 0.74028 | 0.96908 | 0.76678 | 0.90938 | 0.95406 | 0.92049 | 当前 dev 候选 |
| R1 | FALS 1k + Dr.GRPO | 0.64292 | 0.66711 | 0.70671 | 0.95760 | 0.74558 | 0.90932 | 0.94346 | 0.92049 | 相对 E2 明确负向 |
| R2-G | E2 + Dynamic Sampling | 0.65326 | 0.67718 | 0.71555 | 0.96555 | 0.74205 | 0.91150 | 0.95406 | 0.92049 | 成本门控通过、效果负向 |
| F0/E2-50 | E2 step 50 | 0.65305 | 0.67701 | 0.71555 | 0.96555 | 0.74382 | 0.90999 | 0.94700 | 0.92049 | 不替代 E2 step 250 |

E2 相对 E0 的 PDMS scaled 点估计为 `+0.01292`，但 paired-bootstrap 95% CI 为 `[-0.00924,+0.03516]`；只能称为单 seed 的正向 dev 候选，不能称为稳定超过 Stage-2。R1 相对 E2 为 `-0.02938`，CI 全负；R2-G 相对 E2 为 `-0.01904`，同时使用 4,208 条 train reward query，故不再沿这两条路径调参。

### 3.2 D0 train-only 数据资产

| 项目 | 冻结事实 |
| --- | --- |
| 目录 | `/root/autodl-tmp/curious-vla-workspace/experiments/safe_grpo/d0_stage2_train_n4_seed20260812/` |
| rollout | 4,525 token × 4 = 18,100 行 |
| 文件 | `d0_train_rollouts.jsonl` |
| parse success | 99.9337%，12 条失败、涉及 11 个 token |
| PDMS scaled mean/std | 0.59636 / 0.43850 |
| exact-zero group | 18.14% |
| headroom | 0.26838 |
| Safe | 65.71% |
| pairwise ADE/FDE | 0.67096 / 1.60331 |
| 数据边界 | 只含 train；与 566 dev、565 held-out 重叠均为 0 |

当前 reward logger 的持久化行包含 token、denormalized poses、parse 状态和 NAVSIM 分项，但代码没有持久化原始 model response、prompt 或 images。因此“18,100 条完整 scored trajectory”不等于“18,100 条可直接输入 DPO 的文本样本”。P1-S 必须先补齐表示链，不能绕过。

## 4. 全局冻结协议

### 4.1 数据与泄漏边界

1. train/dev/legacy-held-out 固定为 `4,525 / 566 / 565`，两两重叠为 0；legacy-held-out 已被部分访问，只保留为泄漏审计边界。
2. P1 的 gap 分位数、pair 类型、pair 数量和排序只允许读取 D0 train 与原始 train prompt/image/assistant-template。
3. dev 只用于 P2、P3、P4 和条件 P5 的一次正式效果评估；不得用于选择 `gap quantile`、pair ratio、DPO beta、learning rate、epoch 或 LoRA 配置。
4. legacy-held-out 永久禁止继续推理。新 final manifest 必须在 P0 仅由未推理的数据元信息冻结；P6 前除完整性/hash 检查外不得加载内容，不得用于模型或 checkpoint 选择。
5. 所有生成数据必须保存 token manifest、输入 hash、输出 hash、source commit、LLaMA-Factory commit 和完整 resolved config。
6. 同一 token 最多进入每个正式数据集一个 pair，避免单个 scene 因组合数较多被隐式过采样。

### 4.2 模型、训练与评估边界

| 项目 | 冻结值 |
| --- | --- |
| 所有 P2/P3/P4 base/reference | `models/sft_stage2`；均从同一 Stage-2 独立开始 |
| framework | P2/P3/P4 使用 pinned LLaMA-Factory；不把 DPO 塞进 EasyR1 |
| precision | bf16 |
| LoRA | rank 8、alpha 16、`q/k/v/o_proj`，vision tower 与 projector 冻结 |
| micro batch | 1 |
| gradient accumulation | 16 |
| gradient checkpointing | true |
| cutoff length | 4096；覆盖既有 3,072 prompt + 512 response 边界 |
| image max pixels | 262,144，与现有 SFT 配置一致 |
| train seed | discovery `20260812`；条件确认 `20260813` |
| epochs | 正式 P2/P3/P4 各 3.0；pair 总量相同 |
| RSFT learning rate | `1e-5`，不 sweep |
| DPO | sigmoid loss，beta `0.1`，learning rate `5e-6`，不 sweep |
| dev generation | 566 token × 1，seed `20260812`，temperature 0.6，top-p 0.95，response 512 |
| checkpoint | 每个方法只使用正式训练结束 checkpoint；smoke 权重丢弃 |

若 pinned LLaMA-Factory 的字段名或默认行为与表中不一致，P1-S 必须先把等价字段写入 resolved config；不得静默依赖 clone 当天的 HEAD。配置不兼容属于技术阻塞，不授权改变科学变量。

### 4.3 通用执行闭环

每个阶段严格按以下顺序：

1. 预注册唯一输入、唯一变化、命令、输出目录、技术门控、效果门控和失败分支。
2. 只实现阶段所需代码，先跑 fixture/unit tests，再跑全量 CPU 数据检查。
3. smoke 不读取 dev，不沿用权重；正式训练总是从冻结 parent 重新开始。
4. 正式运行写入新目录，禁止覆盖、复用半成品或从 smoke 续训。
5. 先验收 source/config/data/checkpoint/log/exit/resource，再读取 dev 指标。
6. 结果写回本台账后才进入下一阶段。
7. 技术失败只允许修复一个已定位问题；科学负结果不得通过调参重跑。

## 5. P0：路线切换与证据封存

### 目标

把 GRPO 开发线冻结为历史证据，同时建立不受历史实验污染的新开发分支。

### 执行清单

1. 确认 `git status --short --branch` 中没有 tracked 修改；pytest 临时目录不属于实验输入，也不因本阶段删除。
2. 从证据提交 `023139a` 建立 `codex/offline-preference-post-training`；将本台账作为该分支的首个提交，不在 GRPO 证据分支追加提交。
3. 在服务器为新分支使用新的 source checkout 或 clean fast-forward；不得覆盖 GRPO 实验目录。
4. 新实验根目录固定为：

```text
/root/autodl-tmp/curious-vla-workspace/experiments/safe_preference/
```

5. 保存以下只读输入清单与 hash：D0 rollout、train/dev/legacy-held-out manifest、RL parquet、原始 SFT JSON、trajectory stats、Stage-2 model。
6. 确认 `F1_HELDOUT_ACCESSED` 永久锁存在；不删除、不覆盖、不补跑旧 F1。
7. 只用尚未推理的数据元信息审计新 final-set 候选；若可行，在查看任何 P2/P3/P4 dev 结果前冻结版本化 manifest、来源、数量、互斥性与 hash。若不可行，预注册 P6 无 final-set 推理。

### 通过标准

- 原 GRPO 分支与服务器证据可访问；
- 新分支 source clean；
- D0、原始数据、Stage-2、metric cache 路径存在；
- 未启动 GPU、reward server、legacy-held-out 或新 final-set 推理；
- 旧 F1 永久锁保留，新 final-set 可行性与冻结状态已明确写回。

## 6. P1-S：数据表示与可恢复性审计

### 要回答的问题

D0 的 score/pose 能否与训练时的 prompt、image 和合法 assistant response 模板一一对应，并无损构造成 LLaMA-Factory 的多模态 preference 样本。

### 冻结输入

```text
D0:       experiments/safe_grpo/d0_stage2_train_n4_seed20260812/d0_train_rollouts.jsonl
RL data:  src/curious_vla_post_training/EasyR1/data/QA_navtrain_poutine_style_full/data/train.parquet
SFT data: 待 P1-S 定位并冻结的原始 103k assistant-template 数据
stats:    src/curious_vla_post_training/stats/trajectory_stats_train.json
manifest: manifests/train_tokens.txt
```

路径以服务器实际 checkout 为准，但 resolved path 与 hash 必须写入 P1 输出，不允许模糊搜索后静默取第一个同名文件。

### 表示规则

1. 以 token 为唯一 join key；RL parquet 提供实际 GRPO `problem` 与 `images`，原始 SFT JSON 提供同 token 的合法 assistant JSON 模板。
2. D0 `poses` 是 denormalized 8×3 轨迹；用冻结 mean/std 做逆变换，生成 normalized trajectory。
3. chosen/rejected 复用同一 token 的同一非轨迹字段，只替换 `future_trajectory`。因此本项目训练和声明的对象是 trajectory preference，不声明对 CoT、critical-object reasoning 或 explanation 做了 preference 优化。
4. 数值序列化精度固定为 6 位小数；构建后必须经当前 parser → denormalize round-trip，最大绝对误差 `<= 1e-4`。
5. `parsed_ok=false` 或不是 8×3 的 rollout 只进入统计，不进入正式 pair；不得把格式失败当成 safety negative。
6. 每条 chosen/rejected 都必须是合法 JSON、恰有一个 `future_trajectory`、processor 后长度 `<=512`，且 image 路径存在。
7. 若任一 token 无法同时 join RL prompt/image 与 SFT assistant template，不允许用空 explanation、伪造标签或固定占位文本兜底。

### 审计输出

```text
experiments/safe_preference/p1_d0_pairs_seed20260812/
├── schema_audit.json
├── join_failures.jsonl
├── roundtrip_failures.jsonl
├── input_sha256.txt
└── source_commit.txt
```

### 硬门控

- D0 恰为 18,100 行、4,525 token、每 token 4 条；
- D0 外部 token 为 0，dev/held-out overlap 为 0；
- RL prompt/image join 覆盖 4,525/4,525；
- SFT template join 覆盖 4,525/4,525；
- 进入候选池的每个 rollout round-trip 通过；
- LLaMA-Factory processor 抽查与全量 schema validation 通过；
- 不读取 dev 指标，不启动 GPU。

任一 join 门控失败时，P1-M 阻塞。先明确缺失数据是否仍在服务器或可由原始数据确定性恢复；不得直接开始 DPO，也不得把 synthetic placeholder 当正式数据。

## 7. P1-M：Preference Pair Mining 与数据冻结

### 7.1 共同定义

对每条有效 rollout 定义：

```text
is_safe = parsed_ok
          and no_at_fault_collisions == 1
          and drivable_area_compliance == 1
          and time_to_collision_within_bound == 1
```

这里不复用旧 `safe` 字段，因为旧实现只组合 Collision 与 DAC；Safety-Gap 必须显式包含 TTC。Comfort 与 Progress 只报告，不用于覆盖 safety preference。

先在全部有效、四条均可用的 train scene 上计算：

```text
PDMS gap = max(pdms_scaled) - min(pdms_scaled)
delta = 正 gap 分布的 60th percentile
```

`delta` 只计算一次并写入 `preference_stats.json`；看到任何 dev 结果后不得修改。

### 7.2 P3 对照：PDMS-Pair

每个 scene 只生成一对：

- chosen：`pdms_scaled` 最大的有效 rollout；
- rejected：`pdms_scaled` 最小的有效 rollout；
- 仅保留 `gap >= delta`；
- 同分依次按完整 safety tuple、token 内 rollout index 确定，禁止随机漂移。

该对照不使用 safety override。即使综合分最高轨迹存在安全问题，也按 PDMS 排序，以便 P4 检验 safety-aware construction 的独立价值。

### 7.3 P4 方法：Safety-Gap-Pair

每个 scene 最多一对，按互斥 tier 构造：

- Tier A（Safety pair）：四条均解析成功，且同时存在 `is_safe=true` 与 `is_safe=false`。chosen 为安全 rollout 中 PDMS 最高者；rejected 为不安全 rollout 中 PDMS 最高者，形成 hard unsafe negative。Safety preference 可以覆盖综合 PDMS 排序。
- Tier B（Safe-quality pair）：四条均安全；chosen/rejected 为 PDMS 最大/最小者，且 `gap >= delta`。
- Tier C：四条均不安全、无足够差异或含 parse failure；不进入正式训练。

正式 P4 数据固定为 `60% Tier A + 40% Tier B`。设 PDMS 可用数为 `N_P`、Tier A/B 数为 `N_A/N_B`，则：

```text
B = 不超过 1,000 的最大 5 的倍数，且
B <= N_P
0.6B <= N_A
0.4B <= N_B
```

P3 与 P4 都使用恰好 `B` 个 pair；P2 使用 P4 的同一 `B` 个 chosen。P3 按 gap 降序取前 B；P4 的 Tier A 按 rejected PDMS 降序、Tier B 按 gap 降序取数，最终均以 token 稳定打破同分。

### 7.4 输出与统计

预期实现命令：

```bash
python projects/safe_preference/build_preference_dataset.py \
  --rollouts /root/autodl-tmp/curious-vla-workspace/experiments/safe_grpo/d0_stage2_train_n4_seed20260812/d0_train_rollouts.jsonl \
  --rl-data /root/autodl-tmp/curious-vla-workspace/src/curious_vla_post_training/EasyR1/data/QA_navtrain_poutine_style_full/data/train.parquet \
  --sft-data /root/autodl-tmp/curious-vla-workspace/src/curious_vla_post_training/datasets/QA_sft_navsim_train_cot_1view_103k_baseline_norm.json \
  --train-manifest /root/autodl-tmp/curious-vla-workspace/manifests/train_tokens.txt \
  --gap-quantile 0.60 --max-pairs 1000 --safety-ratio 0.60 \
  --seed 20260812 \
  --output-dir /root/autodl-tmp/curious-vla-workspace/experiments/safe_preference/p1_d0_pairs_seed20260812
```

输出：

```text
preference_stats.json
pdms_pair_dataset.json
safety_gap_pair_dataset.json
rejection_sft_dataset.json
pdms_pair_tokens.txt
safety_gap_pair_tokens.txt
dataset_sha256.txt
dataset_examples_redacted.json
```

`preference_stats.json` 至少包含：valid scene、parse failure、join failure、`N_P/N_A/N_B/N_C/B`、gap quantiles、chosen/rejected token length、safe-vs-unsafe 与 safe-vs-safe 数量、chosen/rejected 各 NAVSIM 分项、每 scene 可构造 pair 数和所有过滤原因。

### P1-M 晋级线

- `B >= 500`；
- Tier A/Tier B 比例精确为 60/40，P3/P4/P2 数量相同；
- 0 个 dev/held-out token，0 个重复 token，0 个失效 image path；
- chosen/rejected round-trip、JSON 和 processor validation 100% 通过；
- 相同输入重复执行得到字节相同的数据集与统计；
- 人工只读抽查 30 对：10 个 P3、10 个 Tier A、10 个 Tier B，规则错误为 0。

若 `B < 500`，当前 D0 不足以支持预注册的平衡消融，停止 GPU 路线并记录“pair 数据条件不足”；不得降低门槛、改变 60/40 或读取 dev 后重挖数据。

## 8. P2：Safety-Gap Chosen-only RSFT

### 假设与唯一变量

只把 P4 数据中的 chosen response 当 SFT 目标，检验收益是否主要来自高质量轨迹暴露，而不是 pairwise objective。Base 固定为 Stage-2，不能从 E2 或 P3 继续训练。

### 20-step smoke

```bash
conda activate llamafactory
llamafactory-cli train sft/preference/p2_rsft_smoke.yaml
```

smoke 只验证：20 optimizer steps 完成、loss/grad finite、峰值显存不超过物理显存、image 与 response 正确进入 batch、adapter 可保存。不得运行 dev；smoke 输出不得作为正式 checkpoint。

### 正式训练与评估

```bash
bash scripts/run_safe_preference_experiment.sh p2
bash scripts/run_safe_preference_eval.sh p2
```

冻结配置：`stage=sft`、3 epochs、LoRA 与全局协议一致。正式运行从 Stage-2 重新开始，训练集恰为 P1-M 冻结的 B 个 chosen。

### 验收与结论

- 技术：run `COMPLETE`、exit code 0、adapter/export 完整、无 NaN/OOM、训练样本覆盖正确、source clean、GPU 回收。
- dev：报告完整九项 NAVSIM 指标、parse success、clipping、墙钟、峰值显存与训练样本数。
- 解释：若 P2 提升，说明数据质量/behavior cloning 已贡献收益；若 P3/P4 后续超过 P2，差值才可归因于 pairwise supervision。无论 P2 正负都继续 P3/P4，因为三者是预注册消融。

## 9. P3：PDMS-Pair Multimodal DPO

### 假设与唯一变量

相对 P2，训练数据换为相同数量的 PDMS chosen/rejected pair，objective 换为 DPO；base/reference 仍为 Stage-2。P3 不使用 safety override。

### 执行

```bash
conda activate llamafactory
llamafactory-cli train sft/preference/p3_pdms_dpo_smoke.yaml
bash scripts/run_safe_preference_experiment.sh p3
bash scripts/run_safe_preference_eval.sh p3
```

20-step smoke 的权重丢弃；正式配置为 sigmoid DPO、beta 0.1、learning rate `5e-6`、3 epochs，其余与全局协议一致。

### 验收与结论

- 除 P2 技术项外，保存 chosen/rejected policy-reference log-prob margin、DPO loss 和 reward accuracy；数值必须 finite。
- 只把 P3−P2 解释为“PDMS pairwise objective + pair construction”的联合差异；不把训练 margin 当 dev 泛化证据。
- 无论 P3 是否超过 P2/E2，都继续 P4；不得因 P3 结果修改 P4 数据或 beta。

## 10. P4：Safety-Gap Multimodal DPO

### 核心贡献与唯一变量

P4 与 P3 使用同一 Stage-2 base/reference、同一 pair 总量 B、同一训练 seed、同一 DPO YAML 和同一 dev 协议；唯一变化是 preference construction 从 PDMS-Pair 切换到 60/40 Safety-Gap-Pair。

因此可声明的贡献是：针对 autonomous-driving reward 的结构设计 safety-aware trajectory preference mining；不声明发明新的 DPO loss。

### 执行

```bash
conda activate llamafactory
llamafactory-cli train sft/preference/p4_safety_gap_dpo_smoke.yaml
bash scripts/run_safe_preference_experiment.sh p4
bash scripts/run_safe_preference_eval.sh p4
```

### 工程通过

- P1 数据 hash 与训练 resolved config 完全匹配；
- 3 epochs 完成，adapter/export 与日志完整；
- loss、margin、grad finite；无 OOM、覆盖缺失或 image failure；
- dev 566×1 覆盖完整，parse success `>=99.5%`，clipping `<=0.5%`；
- 资源回收且输出目录不可覆盖。

### 科学晋级线

P4 只有同时满足以下条件才晋级 P5/第二 seed：

1. 相对 P3 的 PDMS scaled `>= +0.01000`；
2. 相对 E2 的 PDMS scaled `>= +0.01000`；
3. Safe、Collision、TTC 点估计均不低于 P3 和 E2；
4. 20,000 次 token-paired bootstrap 中，P4−P3 的 PDMS scaled 95% CI 下界 `> 0`；
5. parse success 与 clipping 达到工程线。

未满足时，P4 仍作为完整消融保留，但跳过 P5，不调 gap quantile、Tier ratio、beta、learning rate 或 epoch 追分。

## 11. P5：一次 Iterative Pair Refresh（条件执行）

只有 P4 达到全部科学晋级线才执行。

### 冻结流程

```text
P4-v1 frozen model
  -> 同一 4,525 train token × 4 rollout，首次即保存 raw response
  -> 使用完全相同的 P1-S/P1-M 规则重建 pair
  -> P4-v1 作为新 base/reference，训练一轮 P4-v2
  -> 一次 dev 评估
```

只允许一次 refresh；不得循环 v3/v4。新 rollout 使用 train，不能读取 dev/held-out；pair ratio、gap quantile、B 上限、DPO beta 和 3-epoch 预算保持冻结。v2 只有相对 v1 达到 `PDMS scaled >= +0.01000` 且 Safe/Collision/TTC 不降才成为最终候选，否则回退 P4-v1。

## 12. P6：确认 seed、Behavioral Audit 与新 final set

### 12.1 匹配第二 training seed

只有 preference 方法通过 P4/P5 晋级线时才执行第二 training seed。若最终 preference 候选为 P4，则用 seed `20260813` 同时重跑 P3 与 P4；若候选为 P5，则同时重跑 P4 与 P5。数据集、base/reference、epoch、batch 和所有超参数不变。

确认标准：

- 两个 seed 上候选相对直接 comparator 的 PDMS scaled 差值均为正；
- 两 seed 平均差值 `>= +0.01000`；
- Safe、Collision、TTC 不出现方向一致的退化。

不满足时，preference 方法只能称为单 seed exploratory result，最终稳定候选回退 E2；不得选择表现更好的 seed 权重。

### 12.2 Final dev behavioral audit

冻结 Stage-2、E2、P3 与最终 preference 候选后，只在 dev 上补一次共同的 sampled audit。生成协议在执行前从 Curious-VLA 官方实现中核对并写入 resolved config；禁止根据结果调整 temperature 或 N。

至少报告：

| 类别 | 指标 |
| --- | --- |
| Overall | PDMS / PDMS scaled |
| Safety | Collision / DAC / TTC / Safe |
| Efficiency / Comfort | Progress / Comfort |
| Output | parse success / clipping / response length |
| Preference | chosen-rejected log-prob margin（train-only 解释） |
| Diversity | pairwise ADE / FDE @8 |
| Best-of-N | PDMS@1/2/4/8 |
| Cost | GPU wall time / rollout count / reward query / peak memory |

若单次 PDMS 上升但 Diversity@8 和 Best-of-8 明显下降，结论必须写成 exploitation–exploration trade-off，不能只报告主指标。

### 12.3 新 final set（仅在 P0 成功冻结时执行）

旧 565-token split 已在上一条路线中生成 520 条部分 rollout，永久降级为 accessed analysis split。其剩余 45 条不得补跑，旧 manifest、部分运行记录或永久锁均不得被包装成新 final set。

启动前必须同时冻结：

- 唯一方法集合：Stage-2、E2，以及存在时已通过第二 seed 确认的 discovery-seed preference checkpoint；
- 每个模型的 source、config、adapter/merged-model hash；
- P0 预注册的新 final manifest、数据来源、数量、与 train/dev/legacy-held-out 的零重叠及 hash；
- 一个不可覆盖的运行目录、永久 access lock、失败恢复语义与后处理命令。

新 final set 只作为一次 final panel 运行。无论结果好坏，都不得再修改模型、pair、threshold、seed 或 checkpoint。若模型生成未完整，不重跑缺失推理；若生成完整但后处理失败，只允许复用原 rollout 恢复后处理。若 P0 无法建立合格的新 final manifest，本项目最终确认只报告 frozen dev、第二 seed 与 behavioral audit，并明确没有 unseen final-set 证据。

## 13. 结果到下一步的唯一映射

| 最新结果 | 下一步 |
| --- | --- |
| P1-S join/round-trip 失败 | 阻塞训练；只修数据来源或表示问题 |
| P1-M `B < 500` | 关闭当前离线 DPO 路线，记录数据不足，不降低门槛 |
| P1-M 通过 | 冻结数据 hash，依次执行 P2 → P3 → P4 |
| P2/P3 正向或负向 | 均继续预注册的下一项，不调参 |
| P4 未过科学晋级线 | 跳过 P5 与第二 seed；稳定候选回退 E2，仍可在项目正式收口时进入 P6 final audit |
| P4 过线 | 执行一次 P5；随后对最终候选做匹配第二 seed |
| P5 未超过 P4 | 最终 preference 候选回退 P4-v1 |
| 第二 seed 不确认 | preference 降级 exploratory；稳定候选回退 E2 |
| 第二 seed 确认 | 执行 final dev behavioral audit |
| 全部模型、阈值、checkpoint 冻结，且 P0 已冻结新 final set | 执行一次新 final-set panel，只汇总不再开发 |
| P0 未能冻结新 final set | P6 不做 final-set 推理；明确报告证据缺口 |

## 14. 永久停止项

除非出现新的项目目标或外部证据改变当前问题，以下不再进入本轮执行队列：

- Dr.GRPO learning-rate/clip/reward 调参；
- Dynamic Sampling cap、filter threshold 或并发 sweep；
- SLDR、Std-Floor 与 FALS 的继续组合；
- 为 `+0.005` 量级 dev 波动追加 GRPO 变体；
- 当前阶段的 structured teacher feedback / recovery；
- 完整重做 Curious-VLA FTE 或官方 benchmark 复现；
- 继续运行旧 565-token held-out 或补跑剩余 45 条；
- 在 P6 前运行新 final set；
- 在 P1 数据门控前安装或实现新的 DPO trainer。

## 15. 实时记录模板

每个阶段完成或失败后只在本文件追加一条记录，并同步第 1 节状态：

```text
### 记录 NNN：<阶段与事件>

- 状态：计划中 / 运行中 / 技术通过 / 科学正向 / 科学负向 / 证据不足 / 失败 / 按门控跳过
- 假设与唯一变量：<本阶段回答什么，只改什么>
- 预注册门控：<进入条件、技术验收、科学晋级线>
- 代码与环境：<source commit/status、LLaMA-Factory commit、resolved config、seed>
- 原始证据：<服务器目录、日志、模型、数据和 hash>
- 数据边界：<manifest、覆盖、train/dev/held-out overlap>
- 技术结果：<退出码、loss、显存、墙钟、异常、资源回收>
- 效果结果：<PDMS、安全、parse、paired bootstrap、cost；如适用则列两个 seed>
- 分析边界：<能说明什么、不能说明什么>
- 决策：<推进 / 回退 / 跳过 / 最小修复重试 / 结束>
- 下一动作：<唯一允许动作及启动门控>
```

## 16. 外部动机与证据边界

这些工作只用于提出假设，不替代本项目自身实验：

1. [Curious-VLA](https://arxiv.org/abs/2603.06049)：说明 narrow policy、FTE/ADAS/SDR 与 behavioral diagnostics 的问题背景。
2. [DriveDPO](https://arxiv.org/abs/2509.17940)：提供 safety-aware trajectory preference 与 iterative DPO 的外部动机。
3. [VL-DPO](https://arxiv.org/abs/2605.20082)：提供从 rollout 自动构造 preferred/rejected trajectory 的外部动机。
4. [LLaMA-Factory preference data format](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md)：只作为 pinned trainer 的格式参考；正式运行以服务器实际 commit 为准。

本项目最终只能声明由 P2/P3/P4/P5 自身同协议证据支持的结论；外部论文报告的 benchmark 数值不得与本项目 566-dev 数值直接横比。

## 17. 实时执行记录

### 记录 001：P0 分支、资产与 final-set 边界闭环

- 状态：技术通过；新路线分支与既有证据封存完成。新 final set 在当前资产下不可行，P6 不执行 final-set 推理。
- 假设与唯一变量：只审计本地/服务器 source、保留资产、旧 F1 永久锁和旧 manifest 外数据可用性；不训练、不运行模型推理、不访问 dev 或 legacy-held-out 内容。
- 代码与环境：从 `023139a` 建立 `codex/offline-preference-post-training`；台账首提交 `e54ae57`。确定性候选 builder 与测试 source `c36767a`，本地/服务器均为 `4 passed`，compile 与 diff check 通过。该 builder 完成 P0 审计使命后从活跃树移除，历史由 Git 保留。
- 服务器同步：source fast-forward 至 `c36767a` 且 status clean；GPU 无 compute PID，8901 无监听，`/root/autodl-tmp` 为 `58/120 GB`、可用约 `63 GB`。E0、D0、R0 retry1、E2、F0、三个旧 split 和 `F1_HELDOUT_ACCESSED` 全部存在。
- 冻结输入：D0 rollout SHA-256 `2ededee1d08d754c251a1f1777d2df4e44e52f4a859e884afeed95521e6ef9d6`；RL parquet `86db9581c4bf29552822fdcc7c6bc71dee4a5d7f78c0f9c44b262bad4048f5dd`；trajectory stats `3f272a89b634def0f5cee65175e45cad288b1b6c85e7a1a708505fe38958ec49`；train/dev/legacy-held-out manifest 分别为 `4a19947abd86d4265e055a6408fc8a6d579fcc083cb5bc4c207159d5c60d8168`、`49dd1fae7f8e77589a27af832835bce8f705c0c5b9062145e180890bf3934cfd`、`6972791333181f03143f636ab565771c970c01a54b5920df3c8c5645dc2085ef`；旧 F1 锁 hash 为 `a994d13c76e0630b388ca066345045f135f7d4ef28597e984bb7c447eb83c6b5`。
- 数据审计：RL parquet 为 103,288 行/唯一 token；旧 train/dev/legacy-held-out union 为 5,656，parquet 外剩余 97,632。第一次按 salted SHA-256 选择 566 个候选时，首个候选缺少 CAM_F0 图像，builder 在创建输出目录前按门控失败。随后全量审计确认 97,632 个候选的 trainval log 可用数为 97,632，但 image 可用数为 0；两个候选输出目录均不存在，没有留下半成品。
- final-set 结论：当前 17 GB NAVSIM 资产只覆盖旧 5,656 split 的 sensor blobs。下载额外大规模 sensor shards 属于新的数据与预算范围，P0 不擅自扩展；因此不构造伪 unseen set，不补跑旧 F1 剩余 45 条。最终报告必须明确没有 unseen final-set 证据。
- P1 前置缺口：D0 18,100 行确认不含 `response`；服务器存在 103,288 行 RL parquet prompt/image/token，但预设的原始 103k SFT JSON 不存在。实际 trajectory stats 路径修正为 `stats/trajectory_stats_train.json`。
- 决策：P0 完成；关闭当前 final-set 分支。允许进入 P1-S，只定位可验证的 assistant-template 来源并实现 train-only 表示审计，仍不启动 GPU。
- 下一动作：检查官方已下载数据或 Hugging Face 发布文件是否能恢复与 RL token 一一对应的原始 assistant JSON；找不到则按 P1-S 门控阻塞，不生成占位 response。

### 记录 002：P1-S assistant-template、join、round-trip 与 processor 门控闭环

- 状态：技术通过；P1-S 全部硬门控通过，允许进入 P1-M，仍未启动 GPU、reward server、训练或任何 dev/legacy-held-out 推理。
- 假设与唯一变量：只回答 D0 的 score/denormalized pose 能否与实际 GRPO prompt、image 和官方合法 assistant JSON template 以 token 一一对应，并无损生成 LLaMA-Factory 多模态 preference response；不选择 pair、不读取 dev 指标、不改变模型。
- 预注册门控：D0 必须为 18,100 行、4,525 token、每 token 4 条；RL/SFT join 均为 4,525/4,525；与 dev/legacy-held-out overlap 为 0；所有候选 trajectory 经 6 位小数序列化、当前 parser 与 denormalize 后误差 `<=1e-4`；图像存在；官方 assistant JSON schema 全量合法；确定性抽取 30 个样本经 pinned LLaMA-Factory `qwen2_vl` processor 后 response 长度 `<=512`。
- 代码与环境：实现与测试 source `fe6eac6bc60c254fd41054610781c70ece0df0bd`，本地/服务器均为 `5 passed`，compile、`pip check` 与 diff check 通过，source status clean。LLaMA-Factory 冻结在 `f28afaf6355af515454dfb16c97d728307c93897`；processor-only Python 3.11 环境为 LLaMA-Factory `0.9.6.dev0`、Torch `2.8.0+cpu`、Transformers `5.8.0`、Datasets `4.0.0`、Accelerate `1.11.0`、PEFT `0.18.1`、TRL `0.24.0`。该 CPU 环境只用于 P1 schema/processor 验证；正式 P2–P4 GPU trainer 环境仍须在 P1-M 通过后单独冻结与 smoke 验收。
- assistant-template 来源：官方发布地址 `MashiroLn/Curious-VLA-dev` 实际是 Hugging Face model repo，而仓库 `docs/train_sft.md` 的 `--repo-type dataset` 已失效。只下载 `CuriousVLA_data/QA_sft_navsim_train_cot_1view_103k_baseline_norm.json`，大小 553,008,318 bytes，LFS 与本地 SHA-256 均为 `0c5b1e689c259d007d2fdb8735ee10dfd4a93bd80ec977f66be9682e8736fcf5`；未下载整库或额外 sensor shard。
- 原始证据：正式输出目录 `/root/autodl-tmp/curious-vla-workspace/experiments/safe_preference/p1_d0_pairs_seed20260812/`；`schema_audit.json` 的 `all_core_gates_passed/all_gates_passed` 均为 `true`，`join_failures.jsonl` 与 `roundtrip_failures.jsonl` 均为 0 bytes，`source_commit.txt` 与执行 source 一致。其余输入 hash 与记录 001 一致；新增 SFT hash 如上。Stage-2 processor 文件 `tokenizer.json`、`tokenizer_config.json`、`preprocessor_config.json`、`chat_template.json` 的 hash 已写入 `schema_audit.json`。
- 数据边界：D0 为 18,100 行、4,525 个唯一 train token、每 token 4 条；train manifest 精确相等，dev/legacy-held-out overlap 均为 0。RL parquet 与 SFT JSON 都是 103,288 行/唯一 token，D0 join 均为 4,525/4,525；SFT/RL image 对应 4,525/4,525，实际文件存在 4,525/4,525；4,525 个 assistant response 全部是无重复 key 的合法四字段 JSON 与 8×3 trajectory。
- prompt 版本差异：SFT prompt 与实际 RL/GRPO prompt 的 byte-exact match 为 0/4,525，但全部且只存在同一处官方版本差异：Task 2 的 `optimal future 5-second trajectory` 在 SFT 中修正为 `4-second`；两者的 Task 4 均要求 4 秒、8 点 trajectory。正式 preference 表示原样保留 RL prompt，不改写 prompt；SFT 只提供同 token/image 的合法非 trajectory assistant 字段，因此未生成占位 explanation、伪标签或第二 prompt 真值源。
- 技术结果：12 条 `parsed_ok=false` 或非 8×3 rollout 按预注册规则只计入统计、不进入候选池；其余 18,088 条全部 round-trip 通过，最大绝对误差 `0.0`，失败 0。30 个 salted-hash 确定性 processor 样本全部通过 `Qwen2_5_VLProcessor`，chosen/rejected response 为 394–419 tokens，完整多模态输入为 2,111–2,134 tokens，既满足 response `<=512`，也低于全局 cutoff 4,096。正式命令退出码 0；结束后 GPU 无 compute PID、8901 无监听、无残留 builder/pip 进程，磁盘可用约 60 GB。
- 效果结果：不适用；P1-S 只证明表示、join、parser 与 processor 技术可行，未生成 preference dataset、未训练模型、未读取任何 dev 效果。
- 分析边界：可以声明 D0 的 18,088 条合法 trajectory rollout 能在冻结来源下无占位地恢复为 trajectory-only preference response；不能据此声明可形成足够多的 Safety-Gap pair，更不能声明 DPO/RSFT 有收益。SFT/GRPO prompt 的系统性单行差异必须在后续数据与报告中继续保留审计记录。
- 决策：P1-S 完成，按门控推进 P1-M；不修改 `5-second → 4-second` 的实际 RL prompt，不降低 P1-M 的 `B>=500`、60/40 或确定性门槛。
- 下一动作：只实现并执行冻结的 P1-M pair mining；计算一次 `delta=PDMS gap 60th percentile`，验收 `N_P/N_A/N_B/N_C/B`、60/40、P2/P3/P4 等量、零泄漏、processor/round-trip 与字节确定性。若 `B<500`，关闭当前离线 DPO 路线，不启动 GPU。
