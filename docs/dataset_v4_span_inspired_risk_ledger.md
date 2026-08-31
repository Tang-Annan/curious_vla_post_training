# Dataset V4 Span-Inspired Risk 执行台账

> 状态日期：2026-08-31
> 当前状态：CPU-only 场景重标注、互斥容量与五模型难度闭环已完成；原 Tier-1 难度门失败，current-visible 修订获得历史模型支持；未启动训练
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
