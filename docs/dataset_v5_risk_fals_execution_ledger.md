# Dataset V5 Risk / FALS 闭环台账

> 当前阶段：`DATA_PREPARATION_ONLY`。执行资源为 `0.5 vCPU / no GPU`；本阶段只生成与核验两套 2K 训练数据，不启动 GRPO、不加载 policy checkpoint、不访问 Dev/Final。未来训练顺序与证据要求在本台账冻结，但须等待独立 GPU 阶段显式启动。

## 1. 本轮目标与执行边界

本轮物化两个只改变 selector 的数据集：

1. `V5-RISK50`：保留 Risk50 结构，候选内部只用固定 seed stable hash 确定性选择，不读取 FALS 排名；
2. `V5-RISK50-FALS`：保留相同结构约束，使用方案 B 的 lexicographic FALS 目标尽量减少无学习信号 anchor。

两套数据都只消费冻结 Train Screen 8K、当前帧 NAVSIM state、冻结 SFT `G=4` raw-PDMS rollout 和既有 Train-only scene labels。禁止读取 GPU-A/GPU-B policy rollout、训练后 collision、Dev 或 Final；`gpu_used=false`、`training_launched=false` 是本轮硬门。

## 2. V5 Primary Interaction Risk

### 2.1 同 actor 身份约束

每个风险触发必须由同一个当前帧 `track_token` 同时提供 actor type、position、velocity 和 reason。实现逐 actor 计算并输出触发 token；不同 actor 的可见性、距离和速度不得拼接成一条风险证据。annotation arrays 必须对齐，`track_token` 必须非空且当前帧唯一。

### 2.2 当前状态定义

Primary Risk 是以下当前状态信号的并集：

```text
Primary Risk = immediate proximity OR projected closest-approach conflict
```

冻结参数：

| 信号 | 冻结条件 |
|---|---|
| 支持区域 | ego 前方 `±45°`；immediate 最远 20 m，projected 最远 40 m |
| immediate vehicle | 当前距离 `≤5 m` |
| immediate VRU | pedestrian/bicycle 当前距离 `≤10 m` |
| relative velocity | `current actor velocity - current ego velocity` |
| projected conflict | 线性当前运动学进入 vehicle/VRU `3/5 m` safety radius，horizon `≤4 s`，radial closing speed `≥0.5 m/s` |
| lateral convergence | projected conflict 的解释子标签：actor 当前在 safety radius 外，横向速度向 ego 收敛且 `|v_y|≥0.5 m/s`；不独立放宽 Primary |

这里的 TTC/closest approach 只由当前 actor state 与 ego state 推导，不读取 future GT collision 或 policy rollout。因此它补足纯距离 risk，同时不引入 model-dependent risk。

### 2.3 family 优先级

互斥 family 固定为：

```text
risk > construction > signal > control
```

Primary Risk 命中后直接进入 `risk`。未命中风险时，construction 要求当前施工前方上下文加 expert 强反应；signal 要求当前 traffic control 加 expert braking/stop-to-go。expert 行为只用于 construction/signal 的 Train-only 语义补充，不参与 Primary Risk。

## 3. FALS 与方案 B

每个 Screen token 使用冻结 SFT `G=4` raw-PDMS：

```text
Mean       = mean(raw_pdms_G4)
Best       = max(raw_pdms_G4)
Headroom   = Best - Mean
Difficulty = 1 - Mean
FALS       = Difficulty * Headroom
```

`FALS-positive` 定义为 `FALS>0`。方案 B 在全部结构硬约束下按以下顺序求精确 MILP：

1. 最大化 FALS-positive risk；
2. 固定第 1 项最优值后，最大化全部 FALS-positive；
3. 固定前两项后，最大化 StrictClear-mixed；
4. 固定前三项后，最大化 FALS 总和，以 seed=`20260904` stable hash 打破并列。

非 FALS-positive 但因 exact quota 必须保留的 token 记为 anchor。FALS 只影响第二套数据的 membership，不改变 risk 定义和 raw-PDMS reward。

## 4. 两套数据的共同硬约束

| 维度 | 冻结值 |
|---|---:|
| total | 2,000 |
| family risk/construction/signal | 1,000 / 500 / 500 |
| intent straight/left/right | 1,333 / 434 / 233 |
| max scenes per log | 4 |
| Train Monitor overlap | 0 |
| Dev/Final access | false / false |

物化输出固定为：

```text
v5_risk50_2000.txt
v5_risk50_2000.parquet
v5_risk50_fals_2000.txt
v5_risk50_fals_2000.parquet
v5_scene_fals_membership.csv
v5_risk_fals_dataset_report.json
```

parquet 必须写后回读为 2,000 rows 且 schema 与 frozen Screen parquet 一致；manifest、membership、输入和输出文件均记录 SHA-256。

## 5. Train-only 全量预审计

Screen 8K 的预注册容量为：

| 指标 | 数量 |
|---|---:|
| Primary Risk | 1,373 |
| immediate proximity | 361 |
| projected conflict | 1,205 |
| dynamic addition（projected 且非 immediate） | 1,012 |
| lateral convergence | 118 |
| 全 Screen FALS-positive | 5,617 |
| FALS-positive Primary Risk | 953 |

FALS-positive Primary 虽有 953 条，但 family、intent 和 `max-per-log=4` 联合约束下，方案 B 对 1,000 个 risk slot 的严格最优值是 940。因此“真正可学习 risk 达到 1,000”在冻结约束下不成立：第二套数据必须包含 60 个 risk anchors。

方案 B 的预注册 exact 2K 最优组成：

| family | FALS-positive | anchors | total |
|---|---:|---:|---:|
| risk | 940 | 60 | 1,000 |
| construction | 455 | 45 | 500 |
| signal | 500 | 0 | 500 |
| 合计 | 1,895 | 105 | 2,000 |

另有 862 个 raw risk token 满足 `Headroom≥0.005`。该 sensitivity 只描述更严格信号容量，不替换 `FALS>0` 的正式方案 B 定义。

## 6. CPU 数据准备运行

- run id：`v5_risk_fals_datasets_20260904`
- remote source：`/root/autodl-tmp/curious-vla-workspace/src/curious_vla_v3`
- remote result：`/root/autodl-tmp/curious-vla-workspace/experiments/dataset_v3_controlled_overlap/semantic_audit/v5_risk_fals_datasets_20260904/`
- runner：`scripts/run_dataset_v5_risk_fals_prepare_cpu.sh`
- workers：`1`
- 当前状态：`COMPLETE/exit_code=0`
- source commit：`c9e1a8f7c93eccfef26e2e3f7a11b84801221584`
- source status：空；wall time=`205 s`
- `training_launched=false`、`gpu_used=false`、`dev_accessed=false`、`final_accessed=false`

正式 runner 已先执行 focused tests（`4 passed`）、compile、shell syntax 和 `git diff --check`，再扫描 frozen Screen 的 1,063 个原始 Train logs。输出目录存在时拒绝覆盖，正式 run 已生成 `COMPLETE` marker 和 `exit_code=0`。

## 7. 实测数据分布与完整性

| 数据集 | exact total | family R/C/S | intent S/L/R | max/log | FALS-positive | anchors | 状态 |
|---|---:|---|---|---:|---:|---:|---|
| V5-RISK50 | 2,000 | 1,000/500/500 | 1,333/434/233 | 4 | 1,375 | 不适用 | `READY` |
| V5-RISK50-FALS | 2,000 | 1,000/500/500 | 1,333/434/233 | 4 | 1,895 | 105 | `READY` |

方案 B 的 family 分解与预注册值精确一致：risk=`940 FALS + 60 anchors`，construction=`455 + 45`，signal=`500 + 0`。这确认 FALS 约束把可学习 risk 从普通 Risk50 的 706 提升到 940，但在现有 family/intent/log-cap 下仍不能达到 1,000。

两套数据的诊断分布：

| 指标 | V5-RISK50 | V5-RISK50-FALS |
|---|---:|---:|
| unique logs | 873 | 850 |
| Mean raw-PDMS | 0.914837 | 0.880628 |
| Mean Headroom | 0.043886 | 0.067029 |
| Mean FALS | 0.013929 | 0.021478 |
| StrictClear-mixed | 200 | 297 |
| Headroom≥0.0025 | 1,341 | 1,852 |
| Headroom≥0.005 | 1,258 | 1,782 |
| Headroom≥0.01 | 1,052 | 1,558 |

Primary Risk 的 current-state reason 分解：

| reason | V5-RISK50 | V5-RISK50-FALS |
|---|---:|---:|
| immediate proximity | 255 | 249 |
| projected conflict | 882 | 886 |
| dynamic addition | 745 | 751 |
| lateral convergence | 81 | 80 |
| immediate vehicle / VRU | 39 / 220 | 32 / 219 |
| projected vehicle / VRU | 549 / 383 | 565 / 372 |

两套 manifest overlap=`1,254`，Jaccard=`45.6664%`。两个 parquet 均为 2,000 rows、`images/problem/answer` schema、2,000 个 image references、missing images=`0`、manifest order exact。Train Monitor overlap=`0`；Dev/Final 均未访问。

### 7.1 输出 SHA-256

| 输出 | SHA-256 |
|---|---|
| `v5_risk50_2000.txt` | `73c184ef2aac954f40ac9a3b87c7e335f5a554b4885ac9bf351e3562e2e9db9b` |
| `v5_risk50_2000.parquet` | `3e1a6ff35bd97bdcadf93ad574e9c5b91c02e9368ab2c227dc4a3317162650c0` |
| `v5_risk50_fals_2000.txt` | `3372c73f65154260c8c3305597fc1d723e542d6e73bbfcc732f39fda7897112f` |
| `v5_risk50_fals_2000.parquet` | `3df9e3a57a2656328b5abe540842e32d006781cfd366d6fc1400b25128243329` |
| `v5_scene_fals_membership.csv` | `09a1fd63a01dbfc62ff70dd9516caa1075f9af4778433cce99c9fc47cb828343` |
| `v5_risk_fals_dataset_report.json` | `55fca20bae054699f77c1b1b8f20292e4fbfd3e7c78384c3b9f33cf12fd61648` |

完整 run 已同步到本地：

```text
D:\Desktop\curious_vla\artifacts\dataset_v5_risk_fals_20260904\
  v5_risk_fals_datasets_20260904\
```

本地逐文件 SHA-256 与远程 `result_sha256.txt` 一致；两个 manifest 均为 2,000 rows/2,000 unique tokens，实测 overlap=1,254。本地 PyArrow `19.0.0` 读取 `answer.gt: list<null>` 时仍触发已知的 `Repetition level histogram size mismatch`，远程 PyArrow `25.0.1` 已对两个文件完成写后回读。因此该现象是旧 reader 兼容边界，不是 parquet 损坏；后续远程训练可直接使用，若迁移到本地训练环境须先使用兼容 reader。

## 8. 未来两轮 GRPO 冻结顺序

当前不启动训练。未来获得 GPU 并显式进入训练阶段后，只允许按顺序执行：

1. `V5-RISK50 + raw-PDMS`；
2. `V5-RISK50-FALS + raw-PDMS`。

第二轮不得早于第一轮完成；两轮使用相同训练协议，selector 是主要变量。每轮原始训练日志、Monitor 数据与曲线证据需同步到本地：

```text
artifacts/dataset_v5_risk_fals_20260904/<run_id>/training_evidence/
```

至少保留 `training_history.csv`、`training_curves.svg`、`training_curve_summary.json`、`training_evidence_manifest.json` 及其 SHA-256。曲线是展示证据，结论仍以原始训练 JSONL/CSV 和固定评测为准。本轮 CPU 数据准备不会产生这些训练曲线，也不得伪造占位曲线。
