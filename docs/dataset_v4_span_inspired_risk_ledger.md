# Dataset V4 Span-Inspired Risk 执行台账

> 状态日期：2026-09-01
> 当前状态：Risk50 + Raw-PDMS（GPU-A）训练完成，在同源 matched Dev 上相对 RR 通过方向门但 Risk 主指标 CI 跨 0；Safety-Continuous（GPU-B）完成 500 updates 后因 Monitor 记录 bug 保留 `FAILED/1`，final checkpoint 经独立恢复门进入 exploratory Dev，但相对 GPU-A 明确退化并关闭 reward 晋级；Final 未访问；第 16–17 节 selector 仍为未执行的独立后续方案
> 数据边界：读取冻结 Train Screen 与完整 Dev；`Final accessed = false`

## 1. 结论摘要

1. 旧的 `Dev Tail` 并非完全无效：它显著富集了严格近距交互，`critical proximity=51/206`，高于 `Dev Natural=25/210`。
2. 旧 Tail 没有同时富集 expert-response complexity：Tail 为 `36/206`，Natural 为 `40/210`。因此旧 Tail 只对“近距”语义对齐，不能代表完整的风险/恢复语义。
3. 首轮宽口径 `event risk` 覆盖 Train `6,767/8,000`、Dev `368/416`，过宽，只能作为复杂上下文 inventory，不能冻结为 V4 风险集。
4. V4 Tier-1 将评价侧风险改为 `critical proximity ∪ response complexity`；训练侧再要求 critical proximity 具有当前同类型前视输入支持。
5. 冻结定义下，Train Tier-1 learnable pool 为 `3,289`；positive / policy-negative / paired-recovery 容量为 `3,015 / 176 / 193`。
6. 按 2,000-group Span 比例模板，需求为 `1,667 / 166 / 167`。容量形式上可行，但 negative 只余 `10`，recovery 只余 `26`，状态为 `FEASIBLE_TIGHT_MARGIN`。
7. 本轮只冻结语义和候选 manifests，不启动 GPU 训练。下一门是确定性角色采样与 disjoint family quota；不能直接把 3,289 个 Tier-1 token 全部混训。

## 2. 与 V3 的关系

- V3 台账、原始 Screen/Dev/Final 划分及既有实验结论保持不变；V4 是追加协议，不回写历史结果。
- V4 不重新划分 Dev 日志，也不读取或重标注 Final。
- V4 复用的冻结集合：
  - Train Screen：8,000 个 `sft_seen` token；
  - Dev Natural：210 个 `sft_unseen` token；
  - Dev Tail：206 个 `sft_unseen` token；
  - Train 与 Dev token 严格不相交。
- V4 的目标不是证明 selector 已经产生 policy-level 增益，而是把“GT 风险上下文”“当前输入可学习性”“policy failure/recovery”拆成三个门，避免继续用同一个 Tail 标签混合三种语义。

## 3. SpanVLA 迁移边界

参考：[SpanVLA: Efficient Action Bridging and Learning from Negative-Recovery Samples for Vision-Language-Action Model](https://arxiv.org/abs/2604.19710)。论文的 negative-recovery 数据来自真实道路早期探索测试中的 suboptimal ego trajectories 与对应 expert corrections。

当前项目没有真实 takeover/correction pair，因此只能做如下受限迁移：

- GT expert trajectory 与 actor/traffic/construction annotation 用于构造 policy-independent 场景门；
- 当前 `CAM_F0` 对应的前视上下文用于构造 input-support / learnability 门；
- 两个独立 G4 rollout block 的稳定性用于构造 policy-derived negative/recovery proxy；
- `stable_severe` 不等于真实负轨迹，`stable_mixed_recoverable` 不等于人工纠正轨迹；所有文档与实验名必须保留 `Span-inspired` 或 `proxy` 限定。

本台账沿用 2,000-group 比例模板：

| 阶段/角色 | 数量 |
|---|---:|
| warmup positive | 667 |
| mixed positive | 1,000 |
| mixed negative | 166 |
| mixed recovery | 167 |
| 总计 | 2,000 |

## 4. 输入与不可触碰边界

### 4.1 正式输入 SHA256

| 输入 | SHA256 |
|---|---|
| `grpo_screen_8000.txt` | `0df963c45c06f0e7590d9e698cc086e5317532672b6031158636ac4ff8b50f00` |
| `dev_natural.txt` | `f8200afae6a29954fc41cbc126f9bfc2909d668cd593d9ea5c68a8812348b5a5` |
| `dev_tail.txt` | `dca81d1dca0d45415b0e3040bb6f834a60fcb4714c170c3b3a969aa9c513b35b` |
| `master_index.csv` | `40b3a1fb4a9c12a7a4cce26497aa0058128c7370870477033a2a7e523a90280b` |
| `stability_capacity.csv` | `ebb90ea58bc7e8605f00eaa85ab4247a8f6264b027deb176af2a5a89fe254f41` |

### 4.2 运行边界

- `workers=1`
- `CUDA_VISIBLE_DEVICES` 为空
- cgroup：`cpu.max=50000 100000`，即 0.5 vCPU
- cgroup：`memory.max=2147483648`，即 2 GiB
- `Dev accessed=true`
- `Final accessed=false`
- 不读取 Dev/Final policy outcome 来调阈值；Dev 只用于冻结标签分布审计。

## 5. 两层标签协议

### 5.1 宽口径 context inventory，不作为最终风险集

首轮扫描使用以下宽口径：

- vehicle expert-path horizon separation `<=5m`；
- pedestrian/bicycle expert-path horizon separation `<=10m`；
- 20m 内 construction context 与 expert turn/lateral/brake/stop-to-go 同现；
- traffic-control context 与 expert brake/stop-to-go 同现。

该口径得到：

| 集合 | event context | 当前输入支持 | learnable proxy |
|---|---:|---:|---:|
| Train Screen | 6,767 / 8,000 | 5,773 | 4,880 |
| Dev all | 368 / 416 | 354 | 309 |

覆盖率 Train `84.59%`、Dev `88.46%`，不能解释为稀缺真实风险。V4 决策明确设置 `broad_event_is_context_only=true`。

### 5.2 冻结的 V4 Tier-1

`critical proximity`：GT expert-path horizon 中满足以下任一条件：

- vehicle minimum separation `<=3m`；
- pedestrian/bicycle minimum separation `<=5m`。

`response complexity`：满足以下任一条件：

- 当前前视 20m、±45° 内存在 construction object，且 expert trajectory 出现 turn、lateral、braking 或 stop-to-go response；
- 当前存在 traffic-control context，且 expert trajectory 出现 braking 或 stop-to-go response。

评价侧：

```text
Eval Tier-1 = critical proximity OR response complexity
```

训练侧：

```text
Train Tier-1 =
  matching-current-front-visible critical proximity
  OR response complexity
```

其中 critical vehicle 必须有当前前视 vehicle context，critical VRU 必须有当前前视 VRU context。这样不会把未来才进入视野的 actor 误算为当前单帧 `CAM_F0` 可学习信号。

## 6. 正式容量结果

### 6.1 Train 结构

| 标签 | 场景数 |
|---|---:|
| strict critical proximity | 746 |
| 当前前视可见 critical proximity | 538 |
| front construction response | 1,027 |
| current signal hard response | 2,160 |
| response complexity union | 2,969 |
| Eval-style Tier-1 union | 3,434 |
| Train Tier-1 learnable union | 3,289 |

Train Tier-1 明显由 signal/construction response 主导，因此下一步必须按互斥 family 做 quota，不能在 3,289 内直接均匀随机采样。

### 6.2 角色容量与 2K 模板

| 角色 | 可用 | 需求 | 余量 |
|---|---:|---:|---:|
| positive | 3,015 | 1,667 | +1,348 |
| policy negative proxy | 176 | 166 | +10 |
| paired recovery proxy | 193 | 167 | +26 |

- negative/recovery 有 154 个共享 token；这表示同一场景可提供不同 rollout 角色，是 pair 语义，不应误报为互斥独立场景。
- 只要进一步增加任何严格过滤，2K 模板都可能因 negative 容量首先失效。
- `recipe_status=FEASIBLE_TIGHT_MARGIN` 只表示 manifest 容量足够，不表示训练有效，也不构成启动 GPU 的授权。

## 7. Dev 评价组成与 Tail 复盘

完整 Dev 416 个 token 全部保留，主报告改为 Tier-1 / Control 及子类型分层；不创建只含 139 个风险 token 的替代 benchmark。

| 冻结旧 split | 场景数 | critical proximity | response complexity | V4 Eval Tier-1 |
|---|---:|---:|---:|---:|
| Dev Tail | 206 | 51 | 36 | 80 |
| Dev Natural | 210 | 25 | 40 | 59 |
| 合计 | 416 | 76 | 76 | 139 |

解释：

- 旧 Tail 对 critical proximity 有效，51 对 25，说明原静态 interaction ranking 捕获了接近性。
- 旧 Tail 对 response complexity 不富集，36 对 40；它没有覆盖“交通控制/施工上下文下的 expert response”语义。
- V4 Tier-1 后 Tail 仍较高，80 对 59，但差异主要来自 proximity，而不是完整 recovery demand。
- 评价时应同时报告 `All Dev`、`Tier-1=139`、`Control=277`、`critical proximity=76`、`response complexity=76`；旧 Tail/Natural 只保留为历史辅助切片。

## 8. 正式运行记录

### 8.1 保留的失败 run

- run id：`v4_span_inspired_risk_capacity_20260831`
- source commit：`35a50d8340dd6d7b0e2d98b7d1543f5b6c7b82c3`
- 状态：`FAILED`，exit code 1，耗时 172 秒
- 原因：错误复用 Dev 非重叠 14-frame chunk 规则定位 SFT Screen token；目标日志中存在非 chunk-center token。
- 处置：失败目录原样保留；未覆盖、未删除。

修复改为只对冻结目标 token 执行滑动 14-frame 定位；没有将邻近帧加入 Screen/Dev 集合。新增 non-chunk-aligned target 回归测试。

### 8.2 正式 capacity run

- run id：`v4_span_inspired_risk_capacity_20260831_r1`
- source commit：`3b38b8d0dfe713c4df2c05819a78dae21124905d`
- 状态：`COMPLETE`，exit code 0
- 覆盖：Train 8,000 行，Dev 416 行，相关日志 1,091 个
- wall time：222 秒
- 磁盘：3.5 MiB
- 资源：0.5 vCPU / 2 GiB，`oom=0`，`oom_kill=0`
- tests：相关测试 21 passed

关键输出 SHA256：

| 输出 | SHA256 |
|---|---|
| `train_scene_labels.csv` | `5a4dd9faa3b0ca780f67b59e427e2000b1c1a175534027ada41e2e412feaa0b3` |
| `dev_scene_labels.csv` | `058c5de76286401fe7bdb752fc89138f4f693ed6fb5eae77d7b11472e6e52bc8` |
| `span_risk_capacity_report.json` | `c287bad4f1a0e0abab777b3bf41da44d8f4bc306f98c3d74ee589ee913ca1784` |

### 8.3 正式 decision run

- run id：`v4_span_inspired_risk_decision_20260831`
- source commit：`49d0dc8d21579f822cbda5a087e8c0d8c582f482`
- 状态：`COMPLETE`，exit code 0
- wall time：3 秒
- 磁盘：712 KiB
- tests：相关测试 23 passed
- 所有输入/输出 `sha256sum -c` 通过
- 所有 manifest 无重复；Train role manifests 均为 Train Tier-1 子集；Dev Tier-1 与 Control 互斥且并集恰为 416。

关键输出 SHA256：

| 输出 | 行数 | SHA256 |
|---|---:|---|
| `train_v4_tier1_learnable.txt` | 3,289 | `1d5e129abb58f49b7d67adcab15d2145cfe7e444ac223422d99f61b627146f8f` |
| `train_v4_positive.txt` | 3,015 | `3f6ede781f69f444157dd4e3deb996dc43afbb93888c561e22ed957ac0dac1b5` |
| `train_v4_policy_negative.txt` | 176 | `b5569b68a25dab239358a88f28fd5f485156748e4e2db268e35312be2052a8af` |
| `train_v4_paired_recovery.txt` | 193 | `f87a7536f7bcbfda52a010d2f5ae9176f37c9b715d4e042463d4ce4cf3de39e6` |
| `dev_v4_tier1.txt` | 139 | `57cfd1a04d261af4d9f3d69421d0709289ff999f3f304584ebb0a17725854a95` |
| `dev_v4_control.txt` | 277 | `cb31f3b7139a8a47c56d9cd50afcaccccf2c7e4ec5999bd5b5bb5c550c2c5afe` |
| `v4_span_risk_decision_report.json` | — | `db1ee092f326dcc49241a8fac600ce15b0df1270b29c8a2d24098da23ef9ad5a` |

服务器目录：

```text
/root/autodl-tmp/curious-vla-workspace/experiments/dataset_v3_controlled_overlap/semantic_audit/
  v4_span_inspired_risk_capacity_20260831/       # preserved FAILED
  v4_span_inspired_risk_capacity_20260831_r1/    # COMPLETE
  v4_span_inspired_risk_decision_20260831/       # COMPLETE
```

## 9. 已知限制

1. `critical proximity` 使用 expert-path 上的 actor-center separation，不是碰撞框 overlap、TTC，也不是候选 policy 的 counterfactual risk。
2. construction/traffic response 表示复杂度，不自动等于危险；因此保留独立子标签，不与 proximity 混成单一强结论。
3. traffic-control metadata 不保证信号灯像素在 `CAM_F0` 中清晰可辨；当前只是 input-support proxy。
4. policy negative/recovery 来自 rollout reward tier 稳定性，仍受序列级 credit assignment、探索 support 和 reward calibration 影响。
5. Train `map_location` 元数据为空；Dev 有 280 个 `intent=unknown`。在修复上游 metadata 前，不允许按 Train map 或 Dev intent 冻结 quota。
6. 当前 2K 配方 negative/recovery 余量很小；任何 family quota、去重、日志 cap 或更严格视觉可见性要求都必须重新做容量门。

## 10. 下一执行门

在任何 GPU 训练前，必须完成以下事项：

1. 将 Train Tier-1 family 按互斥优先级冻结：`critical proximity > construction response > signal response`；
2. 在 positive / negative / recovery 各角色内分别做 family、intent、log-cap 的确定性容量检查；
3. 明确 negative 与 recovery 共享 token 时对应的不同 rollout/reference 记录，不能仅重复同一监督目标；
4. 生成精确 667/1,000/166/167 manifests 与 SHA256，并验证 train-only；
5. 冻结评价报告协议：All Dev + Tier-1 + Control + 两个 Tier-1 子类型；
6. 通过上述门后再决定是否执行最后一次 GPU 尝试。

当时冻结决策：`V4 Tier-1 semantics accepted; training launch = false`。该判断已被第 12 节的历史模型难度门结果替代；原 Tier-1 不再是 GPU-A 的 primary risk 候选。

## 11. V4 单变量实验闭环计划

### 11.1 只回答三个问题

1. 按真实驾驶场景组织训练数据，是否优于相同规模的随机数据；
2. 数据固定后，CDT 奖励是否相对原驾驶评分产生额外增益；
3. 数据与奖励结论明确后，policy-derived failure/recovery proxy 是否还有额外价值。

三项不得在同一次实验中同时改变。正式因果链固定为：

```text
随机数据 + 原奖励
  → 风险均衡数据 + 原奖励
  → 同一风险均衡数据 + CDT
  → 同一风险均衡数据 + CDT + failure/recovery proxy
```

### 11.2 阶段与停止门

| 阶段 | 唯一变化 | 首要问题 | 进入下一阶段的条件 |
|---|---|---|---|
| CPU-A | 无训练；重做场景互斥分类和容量审计 | 能否得到可复现且不被信号灯主导的 2,000 条候选集 | exact family/intent/log-cap 约束可满足 |
| CPU-B | 无训练；只重算已有模型的 Dev 切片 | 新标签是否稳定定位模型薄弱场景 | Tier-1 与 critical proximity 在历史模型上方向性低于 Control |
| GPU-A | Random 2K 改为风险均衡 2K；其余全部相同 | 数据选择是否有效 | Tier-1/critical 改善，Control 不明显退化 |
| GPU-B | 固定 GPU-A 的同一 2K，只替换为 CDT | 奖励是否有额外价值 | 相对 GPU-A 有增量且 Control/safety 不退化 |
| GPU-C | 固定数据与奖励，只加入少量 failure/recovery proxy | proxy pairing 是否有额外价值 | 相对 GPU-B 有增量；结论始终保留 proxy 限定 |

任一阶段未通过，其后的 GPU 阶段自动关闭；不得通过同时修改数据、奖励、PPO 强度或训练预算来补救。

### 11.3 CPU-A 预注册规则

- Train Tier-1 的互斥优先级固定为：`近距离交互 > 施工响应 > 信号灯响应`；
- 近距离交互只使用当前同类型前视可见的 critical proximity；
- 全局 intent quota 与 V3 Random 基线保持一致：straight/left/right=`1333/434/233`；
- 审计 per-log cap=`2/4/6/8`，采用能够满足 2,000 条精确约束的最小 cap；
- 目标 family quota 在容量审计后机械冻结为：近距离交互至少 25%，信号灯响应不超过 50%，其余给施工响应；
- 选样使用固定 seed，manifest 必须 2,000 unique tokens、train-only、与 Dev/Final 无 overlap；
- CPU-A 产出的 2,000 条在 CPU-B 通过前只称 `provisional candidate`，不得送入 optimizer。

### 11.4 CPU-B 固定报告切片

对既有 SFT、Random+原奖励、TailMix+原奖励、TailMix+CDT、TailMix+CDT+PPO2 五组模型，统一重算：

- All Dev：416；
- V4 Tier-1：139；
- Control：277；
- critical proximity：76；
- response complexity：76；
- 旧 current-interaction comparator：用于检查新标签是否比已知近距定义更能定位失败。

每个切片至少报告 StrictClear、PDMS-scaled、Collision 与 TTC；以 `Tier-1 − Control` 和 `critical proximity − Control` 的跨模型方向一致性作为训练门。Dev 结果只用于判定标签是否有效，不参与 token 选择或 family quota 调参。

### 11.5 GPU-A 的冻结对照原则

若 CPU-B 通过，GPU-A 只能改变 2,000-token optimizer manifest：

- 初始化模型、seed、G、batch、学习率、LoRA、PPO epochs、总 update、rollout 预算不变；
- 奖励继续使用原 Raw-PDMS；
- primary comparison 为“风险均衡 2K + Raw-PDMS”相对“Random 2K + Raw-PDMS”；
- 必须同时报告 Tier-1、critical proximity、Control 与 All Dev；
- 期望方向为 Tier-1 上升、critical proximity 上升、Control 近似不变、All Dev 不下降。

本计划写入时仍保持：`GPU training authorized = false`。

## 12. CPU 闭环正式结果：原标签失败与 current-visible 修订

### 12.1 首次互斥容量与历史模型难度 run

- run id：`v4_experiment_closure_cpu_20260831`
- source commit：`f382cd10dc94cbcb9d2e3415ab0ebd2b8b5a1d3b`
- 状态：`COMPLETE`，exit code 0，wall time 6 秒
- 环境：0.5 vCPU / 2 GiB，`CUDA_VISIBLE_DEVICES` 为空
- tests：26 passed
- 数据边界：Dev accessed=true，Final accessed=false
- 资源终态：`oom=0`、`oom_kill=0`

原 3,289 个候选按 `近距离 > 施工 > 信号` 互斥后为：

| family | 场景数 | unique logs |
|---|---:|---:|
| strict horizon proximity | 538 | 285 |
| construction response | 949 | 467 |
| signal response | 1,802 | 648 |

per-log 容量审计：

| cap | 最多可用场景 | 2,000 exact 状态 |
|---:|---:|---|
| 2 | 1,680 | upper bound insufficient |
| 4 | 2,730 | exact feasible |
| 6 | 3,194 | 未继续求解；cap=4 已是最小可行 |
| 8 | 3,289 | 未继续求解；cap=4 已是最小可行 |

MILP 在 cap=4 下生成了精确 2,000 条 provisional candidate：

- family：近距离/施工/信号=`500/500/1000`；
- intent：straight/left/right=`1333/434/233`；
- unique logs=`820`；
- max per log=`4`；
- manifest SHA256：`2e619615f3c97b91f5ee24f42f19f946fa4300dcf8b05bf6c23402f2b2047310`；
- 状态：`PROVISIONAL_NOT_TRAINING_AUTHORIZED`。

### 12.2 原 139-scene Tier-1 难度门失败

五组既有模型全部重新按 All Dev / Tier-1 / Control / critical proximity / response complexity / current interaction 统计，并按 `log_name` 做 20,000 次 cluster bootstrap。

结果方向与预期相反：

- `Tier-1 − Control` StrictClear 在五个模型上全部为正，范围 `+6.56～+8.00 pp`；
- `Tier-1 − Control` PDMS-scaled 全部为正，范围 `+0.0610～+0.0715`；
- critical proximity 76 个场景相对 Control 的 StrictClear point delta 为 `+1.10～+3.14 pp`，没有稳定定位困难；
- response complexity 76 个场景的 StrictClear 为 `90.79%～92.11%`，相对 Control 高约 `+14.62～+16.65 pp`，主要是模型已经容易处理的红灯/停车/起步上下文；
- 原 V3 current-interaction 265 个场景相对 noninteraction 151 个场景的 StrictClear 在五个模型上为 `−11.09～−13.45 pp`，PDMS-scaled 为 `−0.0986～−0.1125`，全部 bootstrap CI 上界小于 0。

机械结论：

```text
LABEL_DIFFICULTY_GATE_FAILED
original_tier1_training_ready=false
GPU training authorized=false
```

这证明“expert 做了明显响应”不能直接当作“模型困难风险”；它更适合作为 context family，而不是 primary risk label。

### 12.3 current-visible 追加修订

修订没有在 Dev 上搜索新阈值，而是复用早已存在的 current interaction 阈值，并补上当前输入支持：

```text
primary risk =
  当前 vehicle distance <= 5m 且当前前视存在 vehicle context
  OR
  当前 VRU distance <= 10m 且当前前视存在 VRU context
```

construction 与 signal response 降为训练数据的 context quota family，不再并入评价侧 primary risk。

正式修订 run：

- run id：`v4_experiment_closure_cpu_20260831_r1`
- source commit：`4a19f0b76a3f9c35405c0897cfe127486f0af952`
- 状态：`COMPLETE`，exit code 0，wall time 9 秒
- tests：27 passed
- 数据边界：Dev accessed=true，Final accessed=false
- 资源终态：`oom=0`、`oom_kill=0`

修订后的 Train 互斥容量：

| family | 场景数 | unique logs |
|---|---:|---:|
| current-visible proximity | 1,871 | 691 |
| construction context | 790 | 431 |
| signal context | 1,344 | 562 |
| union | 4,005 | 975 |

cap=2 的理论上限为 1,825，仍不足 2,000；cap=4 的上限为 3,104 且 exact feasible。修订版 provisional 2K 精确满足：

- family=`500/500/1000`；
- intent=`1333/434/233`；
- unique logs=`834`；
- max per log=`4`；
- manifest SHA256：`3bcb6f32febe5f743a9e400ec750cf59f94c6256159ed8023a4cc4440f6f3e25`。

修订后的 Dev primary risk / Control=`214/202`。五模型历史结果如下：

| 模型 | Risk StrictClear | Control StrictClear | delta | Risk PDMS-scaled | Control PDMS-scaled | delta |
|---|---:|---:|---:|---:|---:|---:|
| SFT | 72.43% | 84.16% | -11.73 pp | 0.6948 | 0.8151 | -0.1203 |
| RR | 72.90% | 84.16% | -11.26 pp | 0.7010 | 0.8181 | -0.1172 |
| TR | 72.43% | 85.64% | -13.21 pp | 0.6978 | 0.8266 | -0.1289 |
| TC | 72.43% | 84.65% | -12.22 pp | 0.6971 | 0.8206 | -0.1235 |
| TC-PPO2 | 71.96% | 83.66% | -11.70 pp | 0.6902 | 0.8114 | -0.1212 |

统计门：

- 五模型 StrictClear point delta 全部小于 0，95% bootstrap CI 上界范围为 `-3.25～-4.72 pp`；
- 五模型 PDMS-scaled point delta 全部小于 0，95% bootstrap CI 上界范围为 `-0.0432～-0.0512`；
- `LABEL_DIFFICULTY_GATE_PASS`；
- 状态限定为 `HISTORICAL_MODEL_SUPPORTED_PROVISIONAL_REVISION`，不是独立 holdout 证明。

### 12.4 正式产物与当前门

`v4_experiment_closure_cpu_20260831_r1` 关键输出：

| 输出 | SHA256 |
|---|---|
| `provisional_current_visible_risk_balanced_2000.txt` | `3bcb6f32febe5f743a9e400ec750cf59f94c6256159ed8023a4cc4440f6f3e25` |
| `train_current_visible_exclusive_labels.csv` | `328de268945d0c306c52220a2d36d7cfc88c263a6bbad1fea2aa69a9b6bc3ee9` |
| `dev_current_visible_model_slices.csv` | `28395ffd073db52528c89e2d1119a9b7eeb3285fc7f693464142a9aa1aaabf8b` |
| `v4_cpu_experiment_closure_report.json` | `7143049ca9815380fbcb5970556bb5514abd8f557f6346dda8b3580544bec206` |

当前决策：

1. 原 139-scene Tier-1 退回历史诊断标签，不得用于 GPU-A primary gate；
2. current-visible 214-scene 修订冻结为下一候选，不再继续读取 Dev 调阈值；
3. 2,000 条修订 manifest 已精确生成，但仍保留 `provisional` 名称；
4. 当前无卡服务器不启动训练；`gpu_training_authorized=false`；
5. 下一步只允许把该 exact manifest 物化为与 RR 完全匹配的 optimizer parquet/config，完成输入哈希和成本门后，再切换 GPU 执行“只改数据”的 GPU-A。

## 13. 近距离风险占比容量门与 Random 语义差异

### 13.1 对上一版计划的修订

GPU-A 前增加一个纯 Train、CPU-only 的比例门。此门不读取 Dev/Final，也不训练模型；唯一目标是证明新的 2K optimizer manifest 相对既有 Random 2K 在风险语义上具有足够大的差异。

固定规则：

- primary risk 沿用第 12 节 current-visible 定义，不再调阈值；
- 总量固定 2,000，intent 固定 straight/left/right=`1333/434/233`，每日志最多 4 条；
- 对 40%/50%/60% 三档分别求 exact feasible 解；剩余配额在 construction/signal context 间等分；
- 另求上述总量、intent、log-cap 约束下的 primary-risk 最大值，只作为容量上界，不作为训练候选；
- 首选冻结 50% primary risk + 25% construction + 25% signal；
- 冻结门预注册为：50% exact feasible、intent 精确、max-per-log≤4，并且相对既有 Random 2K 的 primary-risk 占比提高至少 20 个百分点；
- 若未过门，GPU 继续关闭；不得通过修改奖励、PPO 强度或训练预算掩盖数据语义差异不足。

正式对比表固定报告 mutually exclusive 的 proximity/construction/signal/control、intent、unique logs 与 max-per-log。既有 Random 清单保持原样，不为改善表格而重采样；因此它是否满足 max-per-log=4 也作为审计结果如实记录。

计划 run id：`v4_risk_ratio_audit_20260831`；执行环境固定为无卡服务器，`CUDA_VISIBLE_DEVICES` 为空，`dev_accessed=false`，`final_accessed=false`，`gpu_training_authorized=false`。

### 13.2 正式容量结果

- run id：`v4_risk_ratio_audit_20260831`
- source commit：`31fbb069aab4652db1f586212bb0e0e072b56c8f`
- 状态：`COMPLETE`，exit code 0，wall time 6 秒
- 环境：0.5 vCPU / 2 GiB，`CUDA_VISIBLE_DEVICES` 为空
- tests：14 passed
- 数据边界：Train only，Dev accessed=false，Final accessed=false
- 资源终态：`oom=0`、`oom_kill=0`、训练进程数 0，run 目录 224 KiB

40%/50%/60% 三档均可在 intent=`1333/434/233`、max-per-log=4 下精确求解：

| 方案 | proximity | construction | signal | unique logs | max/log | 状态 |
|---|---:|---:|---:|---:|---:|---|
| Risk40 | 800 | 600 | 600 | 862 | 4 | exact feasible |
| Risk50 | 1,000 | 500 | 500 | 880 | 4 | exact feasible |
| Risk60 | 1,200 | 400 | 400 | 854 | 4 | exact feasible |

在只固定总量、intent 与 max-per-log=4、其余 context family 不设配额时，MILP 的 primary-risk 最大解为：

- proximity=`1,685/2,000=84.25%`；
- construction/signal=`216/99`；
- unique logs=`803`，max per log=`4`。

该 84.25% 仅是约束下容量上界。它会把 context diversity 压缩到 15.75%，因此不进入训练；不能因为容量允许就选择最大化方案。

### 13.3 Random 与冻结 Risk50 的语义差异

| 数据 | proximity | construction | signal | control | intent S/L/R | unique logs | max/log |
|---|---:|---:|---:|---:|---|---:|---:|
| Random 2K | 464（23.20%） | 203（10.15%） | 334（16.70%） | 999（49.95%） | 1333/434/233 | 944 | 6 |
| Frozen Risk50 2K | 1,000（50.00%） | 500（25.00%） | 500（25.00%） | 0 | 1333/434/233 | 880 | 4 |

结论：

1. primary-risk 占比从 23.2% 提高到 50.0%，差值 `+26.8 pp`，超过预注册的 `+20 pp` 门；
2. 两个清单仅重叠 495 条，Jaccard=`14.12%`，不是对 Random 的轻微重排；
3. Frozen Risk50 精确满足 intent 和 max-per-log=4；正式状态为 `FROZEN_RISK50_CPU_GATE_PASS`；
4. 旧 Random 清单的 max-per-log 实测为 6。后续可以把 GPU-A 解释为“整个风险导向数据组织相对旧 Random 的效果”，但不能把差异严格归因于风险占比一个变量；若要做纯比例因果实验，需另建同样 max-per-log=4 的 matched Random，并重新训练对应基线；
5. 当前选择冻结 50%，而不是 60% 或 84.25%，原因是它已经制造足够大的可检验语义差异，同时保留 1,000 条 construction/signal context diversity，实验含义最直接。

冻结清单：`frozen_current_visible_risk50_2000.txt`，SHA256=`28bff1c503377b94eff0adb5415db3696c74f8c463b528b98d5e44009eeaade1`。

关键输出 SHA256：

| 输出 | SHA256 |
|---|---|
| `candidate_risk_ratio_40_2000.txt` | `f81751fedbee7c755b3db7e75766aca83ee0b4af657680874a42f601a3aa66cd` |
| `candidate_risk_ratio_60_2000.txt` | `ef3ec92cb8f452e1e3b6a5d3638e07fa379fa050be2fcb8189e60273f39d989c` |
| `capacity_max_primary_risk_2000.txt` | `f7902c8dc6e0eebbb2a89df41e6050cc74310f07d1e58e8e3a648661eac82f12` |
| `random_vs_frozen_risk_composition.csv` | `43a53e479da550a29137458a6b11294986ebea0d2da278d14ae61b9ed893ee31` |
| `v4_risk_ratio_audit_report.json` | `cca5a3f46ad74898c74bda3f666dc701241761ce54ca5db22304f6b334c2bfe6` |

当前执行门：Risk50 数据语义门已通过并冻结；GPU 仍未启动，`gpu_training_authorized=false`。下一步只准备与 RR 对齐的 optimizer parquet/config、训练时长与成本门；待 GPU 实例可用后再执行“Risk50 + Raw-PDMS”，不同时改奖励或 PPO 参数。

## 14. Risk50 optimizer 物化与 RR 对齐计划

### 14.1 唯一允许变化的训练输入

以 RR 实际保存的 `checkpoints/experiment_config.json` 为配置权威，而不是重新解释旧台账或依赖当前 YAML 默认值。Risk50 配置相对 RR 只允许三个路径发生变化：

1. `data.train_files`：Random parquet → Risk50 parquet；
2. `trainer.experiment_name`：RR run id → `v4_risk50_raw_g4_b4_seed20260827`；
3. `trainer.save_checkpoint_path`：RR checkpoint 目录 → V4 checkpoint 目录。

其余完整 resolved config 必须逐字段相等，包括 seed=`20260827`、Raw-PDMS、G=4、4 groups/update、500 updates、PPO epoch=1、LR=`1e-6`、constant scheduler、LoRA rank 8 attention-only、KL=`0.01/low_var_kl`、shuffle=true、train sampling=`1.0/1.0` 与 Monitor steps=`0/100/200/300/400/500`。本次不继承 TC-PPO2 的 `ppo_epochs=2`。

### 14.2 CPU-only 训练前门

物化 run 必须完成：

- 从冻结 Screen 8K parquet 按 Risk50 manifest 顺序抽取 2,000 行；
- 输出 parquet schema 与 RR Random parquet 完全一致，answer token 顺序与 manifest 完全一致；
- 2,000 unique tokens、所有图像路径存在、与 Train Monitor 256 overlap=0；
- 当前 runtime 合并历史 RR config 后不得新增或改变字段；
- SFT Stage-2 两个模型分片、config 和 index 的 SHA256 重新通过 RR `model_sha256.txt`；
- 使用真实 tokenizer/processor 对 Risk50 parquet 做四点 CPU dataloader smoke；
- 冻结正式启动入口，使启动前再次验证 prep hashes、source commit、GPU idle、8901 端口、30 GiB 磁盘和目标目录不存在。

RR 参考成本来自既有正式 run：wall time=`27,358 秒（7 小时 35 分 58 秒）`，峰值显存=`21,266 MiB / 24,564 MiB`。Risk50 与 RR 的序列预算完全相同，因此训练时长应按同量级预留；正式成本门以至少 24 GiB GPU、至少 30 GiB 可用磁盘和约 8 小时连续窗口为准。

RR 训练 source=`fafda8a771753653a098582b15cce7b17f603037`。当前 source 相对 RR 增加了 PPO epoch telemetry/iterator 修复；PPO epoch=1 时 optimizer update 数与损失路径不变，但源码不是 byte-identical。正式对齐表述限定为“数据以外的 resolved config 完全相同，当前训练实现包含额外只读 telemetry”；不得写成源码逐字节相同。

计划 run id：`v4_risk50_rr_aligned_prepare_20260831`。本阶段保持 `CUDA_VISIBLE_DEVICES=empty`、Dev/Final 均不读取且不启动训练。

### 14.3 首次准备失败与最小修复

首次 run `v4_risk50_rr_aligned_prepare_20260831` 在配置语义门失败并完整保留，状态=`FAILED`。模型哈希和 parquet 抽取已经通过，失败原因是：RR 保存的 `experiment_config.json` 是 post-init resolved config，reward 路径与函数名已拆为：

```text
reward_function=/.../navsim_reward_text.py
reward_function_name=compute_score_raw_pdms
```

该 resolved JSON 不能直接作为启动配置回灌；框架会把无冒号的路径重新解释为默认函数 `main`。最小修复只在生成 runnable config 时重组为 `path:compute_score_raw_pdms`，然后再次执行 post-init，并要求最终 resolved config 相对 RR 仍然只有预注册的三个路径差异。没有放宽字段比较，也没有覆盖失败目录。

### 14.4 正式 r1 训练准备结果

- run id：`v4_risk50_rr_aligned_prepare_20260831_r1`
- source commit：`5844792a809dd15a550687b720da914fad99b6e9`
- 状态：`COMPLETE`，exit code 0，wall time 348 秒
- tests：17 passed；两个 bash 启动脚本 `bash -n` 通过
- 环境：0.5 vCPU / 2 GiB，`CUDA_VISIBLE_DEVICES` 为空
- 数据边界：Train only，Dev accessed=false，Final accessed=false
- 资源终态：`oom=0`、`oom_kill=0`，run 目录 848 KiB

物化结果：

- `risk50_train_2000.txt`：2,000 unique tokens，字节级等于冻结 Risk50 manifest；
- `risk50_train_2000.parquet`：2,000 rows，列为 `images/problem/answer`，schema 与 RR Random parquet 完全一致；
- answer token 顺序与 manifest 完全一致；2,000 个图像引用全部存在；
- 与冻结 Train Monitor 256 overlap=`0`；
- 当前 runtime 重新解析 RR config 的 drift=`[]`；V4 resolved config 相对 RR 只变化 `data.train_files`、`trainer.experiment_name`、`trainer.save_checkpoint_path`；
- SFT Stage-2 两个 safetensors 分片、`config.json`、index 全部通过 RR 原 `model_sha256.txt`；
- dataloader smoke 使用实际 tokenizer/processor 读取索引 `0/666/1333/1999`，batch=`4`，`input_ids shape=[4,3072]`，状态=`V4_RISK50_DATALOADER_SMOKE_PASS`。

关键输出 SHA256：

| 输出 | SHA256 |
|---|---|
| `risk50_train_2000.txt` | `28bff1c503377b94eff0adb5415db3696c74f8c463b528b98d5e44009eeaade1` |
| `risk50_train_2000.parquet` | `e2933afd27fe3a8df2ba52b3586f8ff4efcbf1092d455a2ca92c16a5c0e1a378` |
| `risk50_rr_aligned_config.json` | `08b20da15a270a641f685fb0b40d19e0bfa95a63507515b275d612f127b5c4ae` |
| `dataloader_smoke_report.json` | `a6c979cd6fe74b816e2cdd534bbb7bbcf70731195383fff6e0815b598ece6f40` |
| `v4_risk50_training_prepare_report.json` | `5e6c9019084a5f8778d0c8259fe12a776e8b3fc3352513b638d3d7da1047a4a4` |

CPU 侧启动前复核：source clean；可用磁盘约 36.1 GiB；8901 端口使用者 0；目标 formal run 和 debug 目录均不存在；Raw-PDMS 函数存在。正式训练入口已经冻结为：

```text
bash scripts/run_dataset_v3_formal_cell.sh --cell V4-RISK50 --seed 20260827
```

启动器会在创建 run 前再次验证 prep 全部 SHA256、source commit=`5844792...`、GPU idle、8901 空闲、磁盘≥30 GiB、目标目录不存在。当前持久化准备状态=`V4_RISK50_RR_ALIGNED_READY`；无卡实例不启动训练，`gpu_training_authorized=false`。切换 GPU 后仍需通过启动瞬间的两个易失门，预计训练参考时长 7 小时 36 分，建议预留约 8 小时连续 GPU 窗口。

## 15. V4 Reward：安全优先连续奖励设计与 CPU 审计冻结

### 15.1 冻结定义

设计原则来自用户 2026-08-31 的 reward 意见（Risk50 已冻结前提下不再以 CDT 为主 reward）：先判安全，安全程度连续化，安全之后才允许进度/舒适加分。冻结公式：

```text
R = (S + 0.25 * H * Q) / 1.25
S = 0.55 * H + 0.30 * R_TTC + 0.15 * R_distance
Q = 0.70 * ego_progress + 0.30 * history_comfort
H = 1  iff  no_at_fault_collisions == 1.0 and drivable_area_compliance == 1.0, else 0
R_TTC = min(t_inf / 4.0s, 1.0), t_inf = min(time_to_at_fault_collision, time_to_ttc_infraction)
        (无碰撞且无 TTC 违规时 t_inf = inf → R_TTC = 1.0)
R_distance = min(min_distance_to_actors / 5.0m, 1.0)
```

关键性质（结构性，审计复核）：`H=1` 时 `R >= 0.55`，`H=0` 时 `R <= 0.45`，硬安全通过/失败永不相交；危险轨迹只能通过 `S` 竞争，不会因进度/舒适反超安全轨迹。

训练侧入口：`compute_score_safety_continuous`（`EasyR1/verl/utils/reward_score/navsim/navsim_reward_text.py`）。JSON 语义约定：无风险时刻/无参与者距离在 server 响应中为 `null`，reward 函数解释为 `inf`（`navsim_reward_text.simulator_reward` 转 `inf`；`cdt_scalar_reward.time_to_infraction` 跳过 `null`；审计层 `None → inf`）。

### 15.2 实现与可审计字段

NAVSIM scorer 新增三个逐候选字段，统一在 human-penalty 分支用同一 scorer 重打分之前捕获：

| 字段 | 定义 |
|---|---|
| `time_to_at_fault_collision` | 首次 at-fault 碰撞时刻（s），无碰撞为 `null`（即 inf） |
| `time_to_ttc_infraction` | 首次 TTC 违规时刻（s），无违规为 `null` |
| `min_distance_to_actors` | 4s 水平线上自车 footprint 与任意 `AGENT_TYPES` 动态参与者的最小多边形净距（m），无参与者为 `null` |

代码：`navsim_eval/navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py` 增加 `_calculate_min_distance_to_actors`；`navsim_eval/navsim/evaluate/pdm_score.py` 在 human-penalty 前写入 `pdm_result`；`EasyR1/.../cdt_scalar_reward.py` 增加 `safety_continuous_reward`/`safety_hard_gate`/`time_to_infraction`；`navsim_reward_text.py` 增加 `compute_score_safety_continuous` 与 `SAVED_METRICS` 新字段。实现与修复 commit：`cc46c4c`（设计+审计框架）、`07b8116`（human-penalty 覆盖修复）、`4346f21`（JSON null）、`c7cf3f7`（py3.9 兼容）、`5daf802`（CLI dispatch）、`6d58e23`（replay 按 manifest 过滤）、`6e852c5`（audit 标签按 manifest 过滤）。本地与服务器测试 `37 passed`，服务器 `git diff --check` 与 source clean 通过。

### 15.3 CPU 审计协议（预注册）

只使用冻结 Risk50 2,000 token（8,000 条候选轨迹）与既有 Train-only 标签，不读取 Dev/Final：

1. **区分度门**：新 reward 的零扩散组数 ≤ Raw-PDMS，且 Raw-PDMS 全平、新 reward 有差异的组数 > 0；
2. **安全反转门**：同一 G=4 组内同时存在 `H=1` 与 `H=0` 候选时，`min(R|H=1) > max(R|H=0)`，违规数必须为 0；
3. **family 区分度门**：proximity family 的组内中位 range ≥ construction 且 ≥ signal；
4. **非 GT 模仿门**：reward 输入不含任何 GT/expert 轨迹距离项（构造性验证）+ 报告相关性。

运行环境：`v4_reward_audit_20260831_r4`（replay，16 vCPU / 120 GiB 实例，gunicorn `-w 8`、client `--workers 8`，2,000/2,000 groups 完成，7,996/8,000 metric replayed，4 parse failures，PDMS 一致性 1e-8 通过）；`_r4` 的 audit 步骤因标签集合校验 bug 失败并保留；修复后 `_r5` 复用 `_r4` replay 产物完成 audit，`COMPLETE/exit_code=0`。`dev_accessed=false`、`final_accessed=false`。

### 15.4 审计结果

| 门 | 结果 | 证据 |
|---|---|---|
| 区分度 | **PASS** | 零扩散组 Raw-PDMS 670 → safety 218；Raw 全平组 670 中 452 组被新 reward 区分；distinct≥2 组占比 0.665 → 0.891（CDT 0.668） |
| 安全反转 | **PASS** | 140 个混合组检查，违规 0；`safe_min=0.7662 > unsafe_max=0.3600` |
| family 区分度 | **PARTIAL** | proximity 组内中位 range `0.01040`，construction `0.01072`，signal `0.01042`（门槛未达）；但 distinct≥2 组占比 proximity `0.912` > construction `0.818`，signal `0.922`；差异约 3%，属银行级噪声，无语义反转 |
| 非 GT 模仿 | **PASS** | 构造性无 GT 输入；Pearson(reward, ego_progress)=`0.096`，Pearson(reward, pdms)=`0.921`（保留 benchmark 锚定） |

其他关键统计（8,000 条候选）：`hard_safe=7,757`、`H=0=243`（3.04%）；`time_to_at_fault_collision` 有限值 72 条（中位 3.7s）；`time_to_ttc_infraction` 有限值 119 条（中位 2.6s）；`min_distance_to_actors` 中位 1.52m、q75 3.42m（5m 封顶未饱和，距离项有梯度）；reward 整体中位 0.907、q95 0.9997；proximity family 均值 reward `0.882`（最低，与“近距离场景更难”一致）。

### 15.5 决定与后续

1. **冻结** 安全优先连续奖励为 GPU-B 候选；权重 `0.55/0.30/0.15/0.25`、`TTC_SAFE=4.0s`、`DISTANCE_SAFE=5.0m` 一并冻结，不再用 Dev 调参；
2. family 门记录为 `PARTIAL` 观察项而非阻断：新 reward 在全部 family 都提供连续区分（0.891），proximity 的 distinct 比例最高，中位 range 差异约 3% 且无反转；GPU-B 解释时把 proximity 切片作为次要指标报告；
3. GPU-A 顺序不变：`Risk50 + Raw-PDMS`（已 READY）先行，回答数据本身价值；GPU-B 才把 reward 换成 `compute_score_safety_continuous`，其余 resolved config 与 GPU-A 逐字段一致；
4. 审计产物：`v4_reward_audit_20260831_r5/results/reward_audit_report.json` SHA256=`6a65d087c84a14fdb2b7715a8f1fe7cd9b11908962dc7244215526216f80b569`，`candidate_reward_rows.csv`（8,000 rows）SHA256=`59f3c321b2ed9fc901ee7c5da211f97ae8b54b563fb15d3d89a54c76f7f1ffd5`；replay 产物 `_r4/enriched_risk50_reward.jsonl` SHA256=`00df412d7aa610f70d6203e15d96f8fd018ddb138c45dd70cc130d05a6fe1b87`；manifest/labels SHA 与第 13.3/12.4 节一致。

## 18. GPU-A / GPU-B 训练、matched Dev 与科学闭环

### 18.1 GPU-A：Risk50 + Raw-PDMS 技术终态

- run id：`v4_risk50_raw_g4_b4_seed20260827`
- source commit：`5844792a809dd15a550687b720da914fad99b6e9`
- 状态：`COMPLETE`，exit code 0；`500/500 updates`、`2,000/2,000 groups`、`8,000/8,000 train rollouts`
- Train Monitor：`0/100/200/300/400/500` 各 `256/256`；parse、clip、non-finite gates 全部通过
- wall time：`27,981s（7:46:21）`；峰值显存 `21,266 MiB`
- final LoRA：adapter SHA256=`8e9985f47987a8aa7ff5ca301e9396a6011b827d28f61148b1b93584d42b70fe`，config SHA256=`2323f39159be33a475a0a393741b08952f088aae7f27217e1e6185ad5abea2cc`
- full actor 清理前已验证 Stage-2 base + LoRA 与原 1,113 个 tensors 逐项一致，其中 LoRA tensors=`288`，状态=`BASE_PLUS_LORA_EXACT_RECOVERY_PASS`。为 matched Dev 临时物化后再次验证 tensor 语义，Dev 完成后删除临时 full actor `8,144,554,868 bytes`；LoRA、训练/Dev rollouts、report、log、hash 与恢复证据保留。

### 18.2 GPU-B：训练主体完成、原 formal run 失败与独立恢复门

- run id：`v4_risk50_safety_g4_b4_seed20260827`
- source commit：`1fb29e476d8f3a2c7d35fb901f8779146830b8ac`，source status 为空
- 原 formal 状态：`FAILED`，exit code 1；原 marker 不修改、不补写 `COMPLETE`
- 训练主体：experiment log 明确覆盖 step `1..500`，共 `500/500 updates`、`2,000/2,000 groups`、`8,000/8,000 train rollouts`；wall=`27,996s（7:46:36）`，峰值显存约 `24,081 MiB`
- 失败点：六个 Monitor step 均只有 `253/256`，稳定缺少 token `5555e20bdf6c53ba`、`a148c0eb102a527f`、`e4d4b35a03025182`。三者都是无动态参与者场景；server 的 `null` 距离被 client 规范化为 `+inf`，旧 reward 又错误要求距离 finite，产生 `18=3×6` 次 `min_distance_to_actors ... inf` 异常。该问题没有出现在 8,000 条训练 rollout 中，因此未改变 optimizer 所见训练 reward，但使历史六点 Monitor 曲线不完整。
- 最小修复只允许非负 `+inf` 表示“无参与者/安全距离封顶”，新增回归测试；没有放宽 formal verifier。
- 独立 recovery run：`v4_risk50_safety_final_monitor_recovery_20260901`，source=`e43a7d7cb9249dc9be22a1d93634fe193b4680b8`，`COMPLETE/0`，不访问 Dev/Final。final checkpoint 重跑 Monitor=`256/256`，parse=`1.0`、clip=`0`、non-finite=`0`，PDMS-scaled=`0.901163`、StrictClear=`0.980469`；状态=`CHECKPOINT_USABLE_FOR_EXPLORATORY_DEV`。
- recovery 不能重建历史 `0/100/200/300/400/500` 的三条缺失轨迹；它只证明 final checkpoint 的完整训练日志、final Monitor health 与进入 matched exploratory Dev 的技术可用性。
- GPU-B full actor 删除前，Stage-2 base + LoRA 与原 1,113 个 tensors 逐项一致，LoRA tensors=`288`，状态=`BASE_PLUS_LORA_EXACT_RECOVERY_PASS`。adapter SHA256=`e3761316bc8566866d6603b897d7c6f5f0a584b35cb8128f8c3bd74fa289c0f6`，config SHA256=`e443e99582463f2037fb2e820886f819c3c5c8cbb42bbb6d556bb08d17ac9c7f`。

### 18.3 三模型同协议 matched Dev 416

RR、GPU-A、GPU-B 全部在 source=`e43a7d7cb9249dc9be22a1d93634fe193b4680b8` 下重新执行：

- 冻结 Dev=`416`，Natural/Tail=`210/206`；evaluation seed=`20260827`
- `n=1`、temperature=`0.6`、top-p=`0.95`、max response length=`512`
- 同一 Stage-2 base、parser、metric cache、输入 hashes 与 evaluator
- current-visible Risk/Control=`214/202`；response-complexity=`76`
- 三个 run 均 `COMPLETE/0`、416 unique tokens、clip=`0`、指标 finite；Final accessed=`false`
- RR full actor 早于本轮被删除，无法追溯证明与原 full actor 逐 tensor 同一；本轮验证保留 adapter 与组合模型 LoRA tensors 一致并成功物化，adapter SHA256=`7836880df0b5b36de80f05fa133136df875218720bfef74c5024e325ebe7132e`。该边界不影响三者均按当前 source 从各自保留 LoRA 进行 matched inference，但不得改写成历史 full-state byte identity 证明。

主指标：

| 模型 | All Dev PDMS-scaled | All StrictClear | Risk PDMS-scaled | Risk StrictClear | Control PDMS-scaled | Control StrictClear |
|---|---:|---:|---:|---:|---:|---:|
| RR | 0.757413 | 0.783654 | 0.700921 | 0.728972 | 0.817262 | 0.841584 |
| GPU-A | 0.769043 | 0.798077 | 0.711790 | 0.738318 | 0.829698 | 0.861386 |
| GPU-B | 0.753000 | 0.781250 | 0.691946 | 0.719626 | 0.817681 | 0.846535 |

Risk 214 的 candidate safety 均值：

| 模型 | Collision | DAC | Direction | Traffic-light | TTC |
|---|---:|---:|---:|---:|---:|
| RR | 0.735981 | 0.976636 | 0.978972 | 0.887850 | 0.747664 |
| GPU-A | 0.745327 | 0.976636 | 0.974299 | 0.892523 | 0.747664 |
| GPU-B | 0.726636 | 0.971963 | 0.971963 | 0.892523 | 0.738318 |

### 18.4 paired log-cluster bootstrap 与机械结论

统计协议为 fixed seed=`20260901`、20,000 次 paired `log_name` cluster bootstrap。方向门在读取三个新 Dev 结果前冻结：Risk PDMS-scaled/StrictClear 点估计均上升、All Dev PDMS-scaled 不下降、Control PDMS-scaled `>=-0.01`、Risk/Control safety component drop 不超过 `0.005`；GPU-B 另要求 Risk 至少一个 safety component 上升。Risk 两个主指标 CI lower 都 `>0` 才允许写“统计支持的提升”。

GPU-A − RR：

- Risk PDMS-scaled：`+0.010869`，95% CI=`[-0.003804,+0.025646]`
- Risk StrictClear：`+0.009346`，95% CI=`[-0.009390,+0.027149]`
- All Dev PDMS-scaled：`+0.011630`，95% CI=`[+0.001549,+0.022801]`
- Control PDMS-scaled：`+0.012436`，95% CI=`[+0.000603,+0.029892]`
- Risk candidate safety delta：Collision `+0.009346`、DAC `0`、Direction `-0.004673`、Traffic-light `+0.004673`、TTC `0`
- 全部方向/非退化门通过，但 Risk 两个主 CI 均跨 0；机械状态=`DIRECTIONAL_EXPLORATORY_PASS`。结论限定为：Risk50 数据选择在单 seed、已访问 Dev 上方向性优于 RR，且 All Dev/Control 的 PDMS-scaled paired CI 为正；证据不足以声称 Risk slice 上统计稳定提升或训练 seed 稳定性。

GPU-B − GPU-A：

- Risk PDMS-scaled：`-0.019844`，95% CI=`[-0.037533,-0.000123]`
- Risk StrictClear：`-0.018692`，95% CI=`[-0.037634,0]`
- All Dev PDMS-scaled：`-0.016044`，95% CI=`[-0.029382,-0.005242]`
- Control PDMS-scaled：`-0.012017`，95% CI=`[-0.029079,-0.000189]`
- Risk candidate safety delta：Collision `-0.018692`、DAC `-0.004673`、Direction `-0.002336`、Traffic-light `0`、TTC `-0.009346`
- 所有预注册方向门失败，Risk 没有任何 safety component 上升；机械状态=`NO_IMPROVEMENT_GATE`。Safety-Continuous 的训练侧区分度与无反转审计成立，但没有转化为 final policy 增益，反而在 Risk、All Dev、Control 与安全分项上退化。GPU-B 不晋级，按第 11.2 节顺序停止门关闭 GPU-C；不得用 selector、PPO、seed 或阈值变更补救本对照。

### 18.5 产物、hash 与空间闭环

正式 science run：`v4_post_training_science_20260901`，source=`e43a7d7cb9249dc9be22a1d93634fe193b4680b8`，`COMPLETE/0`，source clean、资源回收、磁盘剩余约 `34 GiB`。

| 输出 | SHA256 |
|---|---|
| `model_slice_metrics.csv` | `4f0f164b7a3aeec9773276e07522eaf739a8033840f2137ce45c60737007a50f` |
| `paired_deltas.csv` | `18b239af6931dc0b338d34926b2770c3f09775491ea572b594bae87dee922e51` |
| `v4_post_training_science_report.json` | `04c29c05fbdd7c78fcade401e8bdbf6bcd017713f633b056c731cc0bc04d5291` |

空间清理仅删除已复制、hash 固化且有显式恢复边界的文件：

- GPU-B full actor `8,144,550,392 bytes`，精确可由 Stage-2 + LoRA 恢复；连同重复 debug/ADAS 共回收 `7,978,692 KiB`
- GPU-A 临时物化 actor `8,144,554,868 bytes`，连同重复 debug/ADAS 共回收 `7,975,168 KiB`
- RR 临时物化 actor `8,144,554,868 bytes`，连同本轮重复 ADAS/debug 共回收 `7,954,692 KiB`

Stage-2 base、三个 LoRA、formal run、原 GPU-B `FAILED` 现场、recovery、三套 matched Dev、science report、access records、rollouts、scene metrics、logs、hash 与台账全部保留。GPU、Ray、Gunicorn/8901 均已回收；`Final accessed=false`。
