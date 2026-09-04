# Dataset V5 Risk / FALS 闭环台账

> 当前阶段：`GPU_PREPARATION_READY`。执行资源仍为 `0.5 vCPU / no GPU`；corrected `Risk50+FALS` 已重新物化，两套 2K 数据、raw-PDMS 配置、顺序门、真实 dataloader smoke 与训练曲线导出入口均已冻结。当前未启动 GRPO、未加载 policy checkpoint、未访问 Dev/Final；切换 GPU 实例后可从第 1 轮正常训练入口开始。

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
4. 固定前三项后，最大化整数化 FALS 总和，其中 `FALS_q=round(FALS×10^12)`；
5. 固定前四项全部最优值后，才以 seed=`20260904` stable-hash rank 打破并列。

第 4、5 层是两个独立 MILP。stable hash 不再作为 FALS objective 的浮点扰动项，因此不会为了 hash 牺牲任何 `FALS_q`。

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

## 6. 初版 CPU 数据准备运行（已由 corrected r1 取代）

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

## 7. 初版实测数据分布与完整性（仅保留历史审计）

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

## 9. 训练前修复闭环

训练前复查确认并修复两项阻断问题：

1. 原实现把 `-FALS + 1e-9×stable_hash` 写在同一目标中，无法保证严格 lexicographic。现改为五阶段求解；第 4 层最大化 `FALS_q`，第 5 层通过 exact weighted constraint 固定 `max_fals_quantized` 后再做 hash tie-break，且 `mip_rel_gap=0`。
2. 原实现会在建字典时静默覆盖重复的 Screen master token、scene-label token 或 raw-log stem。现均在建索引前显式检查并直接失败；current-frame annotation arrays 仍要求对齐，track token 仍要求非空且唯一。冻结 rollout schema 没有独立 generation/sample ID，因此只声明并验证每个 token 恰好 4 rows，不把内容相同的独立采样误判为重复。

回归覆盖包括：receding TTC、超过 4 秒 horizon、lateral crossing、annotation 对齐、重复 track token、重复 master/labels/raw stem、binding log cap、不可避免 anchors、risk-FALS 对 total-FALS 的优先级、StrictClear-mixed 优先级、旧浮点扰动尺度反例，以及第 5 层不改变前四层最优值。正式 corrected data runner 为 `16 passed`；最终 GPU-preflight runner 为 `23 passed`。本地连同 formal/config/exporter 相关测试为 `31 passed`。

## 10. Corrected 数据物化结果

- run id：`v5_risk_fals_datasets_20260904_r1`
- source commit：`44dafb512928967588253916c0969f42f2313402`
- 状态：`COMPLETE/exit_code=0`
- wall time：`204 s`
- raw logs：完整扫描 `1,063/1,063`
- 边界：`gpu_used=false`、`training_launched=false`、`dev_accessed=false`、`final_accessed=false`

严格最优值：

| 层级 | corrected 最优值 |
|---|---:|
| risk FALS-positive | 940 |
| total FALS-positive | 1,895 |
| StrictClear-mixed | 297 |
| `FALS_q` sum，scale=`10^12` | 42,956,168,591,389 |
| raw FALS sum | 42.956168591389265 |

corrected `Risk50` 与初版 byte-identical，2,000/2,000 token 不变。corrected `Risk50+FALS` 与初版 overlap=`1,940/2,000`，两侧各替换 60 个 token，证明旧混合 objective 的 hash 扰动确实改变了应由 FALS 决定的 membership。两套 corrected 数据之间 overlap=`1,255`。

| 指标 | corrected V5-RISK50 | corrected V5-RISK50-FALS |
|---|---:|---:|
| total | 2,000 | 2,000 |
| family R/C/S | 1,000/500/500 | 1,000/500/500 |
| intent S/L/R | 1,333/434/233 | 1,333/434/233 |
| max/log | 4 | 4 |
| FALS-positive | 1,375 | 1,895 |
| risk FALS-positive | 706 | 940 |
| StrictClear-mixed | 200 | 297 |
| unique logs | 873 | 865 |
| mean raw-PDMS | 0.914837 | 0.880044 |
| mean FALS | 0.013929 | 0.021478 |

corrected 输出 SHA-256：

| 输出 | SHA-256 |
|---|---|
| `v5_risk50_2000.txt` | `73c184ef2aac954f40ac9a3b87c7e335f5a554b4885ac9bf351e3562e2e9db9b` |
| `v5_risk50_2000.parquet` | `3e1a6ff35bd97bdcadf93ad574e9c5b91c02e9368ab2c227dc4a3317162650c0` |
| `v5_risk50_fals_2000.txt` | `eafbc47a5977808d77635f43da80159da90eacafef712e5d6843ec34592c1400` |
| `v5_risk50_fals_2000.parquet` | `adcf63a38f8723a3c611b1f298574b978d9f77cbf595817cf30d06beb6ed461b` |
| `v5_scene_fals_membership.csv` | `b595455cd939c7202f1cf4eec3e4ee314d604a0654f80b58045d98f433992c46` |
| `v5_risk_fals_dataset_report.json` | `84974e57050b8116d8d27911f9c107d1d0275f0d47ccd1cb300c6a6db783e785` |

远程与本地 `result_sha256.txt` 已逐文件一致。本地证据目录：

```text
D:\Desktop\curious_vla\artifacts\dataset_v5_risk_fals_20260904\v5_risk_fals_datasets_20260904_r1\
```

## 11. GPU 前最终准备状态

- run id：`v5_risk_fals_gpu_prepare_20260904_r2`
- source commit：`687531fe12ffd0921febbde3d92bb61d9f98b6a6`
- 状态：`V5_GPU_PREPARATION_READY`，`COMPLETE/exit_code=0`
- wall time：`416 s`
- 可用磁盘：`36,448,317,440 bytes`，满足 30 GiB 门
- SFT Stage-2：两个 model shards、`config.json`、index 共 4 项 SHA 全部 `OK`
- 两套 parquet：各 2,000 rows，schema 与 RR reference 一致，manifest order exact，图像 `2,000/2,000` 存在
- Train Monitor：256 rows，schema/order exact，图像 `256/256` 存在；两套 optimizer data overlap 均为 0
- 真实 loader smoke：两套均为 2,000 rows，抽样 `0/666/1333/1999`，batch=`4`，`input_ids=[4,3072]`
- future formal/debug dirs：全部不存在
- GPU idle 与 reward port 8901：延迟到 GPU 启动瞬间检查

两份 runnable config 相对历史 RR resolved config 只允许并实测只改变：

```text
data.train_files
trainer.experiment_name
trainer.save_checkpoint_path
```

其余冻结为：相同 Stage-2 独立初始化、Raw-PDMS、standard GRPO、seed=`20260827`、`G=4`、4 groups/update、LR=`1e-6`、PPO epoch=`1`、LoRA rank 8 attention-only、KL=`0.01/low_var_kl`、2,000 groups、8,000 train rollouts、500 updates、Monitor=`0/100/200/300/400/500`。

| 配置 | SHA-256 |
|---|---|
| `v5_risk50_raw_config.json` | `b96c1a225a00e42f8ff3840b7c17bc9f5315089055553a88ce025c9fcb3a1f21` |
| `v5_risk50_fals_raw_config.json` | `0b3603bc795cd1ab93421d68fb817bcb1a2d2241c7a21f62c22899d1847aad08` |
| `dataloader_smoke_report.json` | `129aec3839ab542e2c88ed58507e41fd48ae35a1b8fe870a8c57282296e705c9` |
| `v5_training_prepare_report.json` | `c9376d385f10f267cc8bacc25d3c43cccf45c4e3d13e2f8d2abc599949fac4ac` |

本地预检证据目录：

```text
D:\Desktop\curious_vla\artifacts\dataset_v5_risk_fals_20260904\v5_risk_fals_gpu_prepare_20260904_r2\
```

正常训练入口固定为：

```bash
bash scripts/run_dataset_v3_formal_cell.sh --cell V5-RISK50 --seed 20260827
bash scripts/run_dataset_v3_formal_cell.sh --cell V5-RISK50-FALS --seed 20260827
```

必须顺序执行。第二条命令会在创建 run 前验证第一轮 `COMPLETE`、`exit_code=0`、`training_report.status=COMPLETE`、`training_report.cell=V5-RISK50`，并逐文件执行第一轮 `result_sha256.txt`。两轮均从同一个 Stage-2 初始化，第二轮不继承第一轮 checkpoint。

每轮训练成功后自动导出并纳入 `result_sha256.txt`：`training_history.csv`、`training_curves.svg`、`training_curve_summary.json`、`representative_train_samples.jsonl`、`training_evidence_manifest.json`。训练完成后按既有 artifacts 约定完整下载到本地，不使用占位曲线。
