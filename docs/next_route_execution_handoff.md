# Curious-VLA 下一技术路线执行交接

> 状态更新（2026-08-24）：本文冻结为旧路线交接材料。用户随后明确重新开启有前置门控的 `G=4` GRPO 探索，因此本文关于“不得继续 GRPO”的规划性限制已被新指令替代；服务器安全、旧 held-out、访问锁和不可恢复产物边界继续有效。当前路线以 [`grpo_g4_experiment_ledger.md`](grpo_g4_experiment_ledger.md) 为准。
>
> 本文只交接执行方式和不可违反的状态边界，不为下一路线预选算法。解释默认使用简体中文；代码、技术术语和学术引用保持准确英文。

## 1. 旧路线冻结时不可变状态

- 本地工作区：`D:\Desktop\curious_vla`
- 当前分支：`codex/post-training-analysis`
- 可写远端：`post-training` → `Tang-Annan/curious_vla_post_training`
- 服务器仓库：`/root/autodl-tmp/curious-vla-workspace/src/curious_vla_post_training`
- 服务器实验根目录：`/root/autodl-tmp/curious-vla-workspace/experiments/safe_grpo`
- 旧 GRPO、Dr.GRPO、Dynamic Sampling 和 Recovery 路线已终止；不得继续旧 R1/R2/R3、不得重启 F1。新 `G=4` 实验只能按新台账建立独立命名空间和门控。
- F1 已在用户终止前生成 520/565 条 held-out 部分 rollout。部分结果和运行目录已删除，但 `F1_HELDOUT_ACCESSED` 永久锁保留；原 held-out 已被访问，下一路线不得把它表述为未见终测集，也不得补跑剩余 45 条。
- 服务器只保留 E0、D0、R0 retry1、E2、F0、三个 split manifest、E2 step 50/250 和 F1 访问锁。E2 step 250 只是新路线的可复用基线或初始化点，不代表 GRPO 贡献已被确认。
- 完整科学结论与清理事实见 `docs/post_training_execution_loop.md`；已删除原始产物不可恢复。

## 2. SSH 与远程命令

直接连接：

```text
ssh -p 47507 root@connect.nmb1.seetacloud.com
```

密码由用户单独提供，禁止写入代码、Git、文档或日志。Windows 本地主要通过 `tools/remote_step.py` 执行非交互命令；每个新 shell 都要重新设置环境变量：

```powershell
$env:REMOTE_HOST='connect.nmb1.seetacloud.com'
$env:REMOTE_PORT='47507'
$env:REMOTE_USER='root'
$env:REMOTE_PASSWORD='<向用户索取，不落盘>'
python tools/remote_step.py "<remote command>"
```

`remote_step.py` 基于 Paramiko；未设置上述变量会直接报 `KeyError`。长命令要避免 PowerShell 在本地展开远端的 `$var`、`$!` 或 `$(...)`；优先不用远端 shell 变量，必要时使用 base64 参数或把命令写成无本地插值的形式。退出码文件必须读取内容，例如 `cat run/exit_code`，不能把 `stat size=2 bytes` 误认为退出码 2。

## 3. Git 同步与 GitHub 超时恢复

本地只推当前工作分支：

```text
git push post-training codex/post-training-analysis
```

不要向 `origin`（`Mashiroln/curious_vla`）推送；当前账号对它返回 HTTP 403，这是权限问题，不是网络问题，网络加速不能修复。

服务器 `origin` 指向 `Tang-Annan/curious_vla_post_training`。每次运行前要求 source clean，并使用 fast-forward：

```text
cd /root/autodl-tmp/curious-vla-workspace/src/curious_vla_post_training
GIT_TERMINAL_PROMPT=0 timeout 45 git fetch origin codex/post-training-analysis
git merge --ff-only FETCH_HEAD
git status --short --branch
git rev-parse HEAD
```

GitHub 恢复顺序固定：

1. 先直连一次，必须带有限 timeout；若本地 SSH 等待被终止，要再查服务器是否残留 `git fetch` 进程。
2. 直连失败时，只在单次远端 shell 中执行 `source /etc/network_turbo` 后重试 fetch；该加速只用于 GitHub/Hugging Face，不要写进 `.bashrc`。
3. 仍失败才使用增量 Git bundle：本地以服务器旧 commit 为排除点创建并验证 bundle，上传到 `/tmp`，服务器 `git fetch <bundle> codex/post-training-analysis` 后 `git merge --ff-only FETCH_HEAD`。不得用文件覆盖代替 Git 历史同步。

## 4. 测试方式

本地先跑最小相关检查：

```text
python -m pytest tests/test_safe_grpo.py -q --basetemp=.pytest_tmp_<stage>
python -m compileall -q projects/safe_grpo EasyR1/verl/trainer
git diff --check
```

Windows 本机没有 Bash；shell launcher 的 `bash -n` 放到服务器执行。服务器标准检查：

```text
cd /root/autodl-tmp/curious-vla-workspace/src/curious_vla_post_training
bash -n scripts/<launcher>.sh
/root/autodl-tmp/curious-vla-workspace/envs/curious/bin/python -m compileall -q <changed_python_paths>
/root/autodl-tmp/curious-vla-workspace/envs/curious/bin/python -m pytest tests/test_safe_grpo.py -q
git diff --check
```

本地依赖缺失、执行很慢或 GPU 路径无法覆盖时，直接在远端做一次有意义的测试，不在本地重复等待同一失败。正式运行前还要检查目标目录不存在、source clean、GPU 无 compute PID、8901 无监听、输入 manifest/checkpoint/hash 完整。

本地根目录当前可能显示既有 `.pytest_tmp*` 与 `.tmp_pytest_fsdp` 未跟踪目录；不要暂存它们，也不要把它们当作 source dirty 的实验变更。递归删除这些旧测试目录需要用户单独明确授权。

## 5. 长任务启动与 Luna 子代理监控

> 2026-08-26 补充：本节固化 R4-RAW 对话中实际使用的 `luna_monitor_audit` 设计，作为跨路线可复用的长任务监控协议。

通过 Git 同步的 launcher 采用独立目录和后台日志：

```text
nohup bash scripts/<launcher>.sh > /root/autodl-tmp/curious-vla-workspace/logs/<run>.launcher.log 2>&1 < /dev/null &
```

正常启动的最低证据是 `RUNNING`、`run.env`、`source_commit.txt`、空的 `source_status.txt`、主进程、GPU 占用和 reward server 8901 健康。实验目录禁止覆盖；技术失败保留目录并只允许在明确规则授权时用新目录 retry。


### 5.1 主代理与 Luna 的职责分离

每个正式 run 只创建一个专用 Luna 子代理，任务名固定为 `luna_monitor_<run>`，便于主代理继续等待或发送补充指令。在支持 agent type 的 Codex 环境中使用 `luna_worker`，并继承包含当前运行上下文的 turns；不要为每次轮询重新创建子代理。

本次对话的调度参数可按下列形式复用：

```text
task_name: "luna_monitor_audit"          # 或 luna_monitor_<run>
agent_type: "luna_worker"
fork_turns: "all"                        # 继承运行边界和当前临时凭据
message: <5.2 的完整任务模板>
```

- 主代理独占所有写操作：代码与预注册、preflight、启动、停止/retry 决策、结果分析、台账回填和产物清理。
- Luna 只做读操作：查状态文件、日志、rollout/metric 行数、GPU、端口、进程和磁盘，并在终态做完整验收。
- Luna 禁止执行 `kill/rm/mv`、启动/retry launcher、修改 Git/文件、补写 `COMPLETE/FAILED`、清理 checkpoint，也不得自行改变科学门控。
- 子代理不得再拆分下级代理。主代理在 Luna 运行期间使用长时 `wait_agent`；等待超时时只继续等待子代理，不再建第二套 SSH 正常进度轮询，也避免自动续转等消耗额度的额外行为。

SSH 的 host/port/user、精确的 run/log 路径、预期覆盖和终态条件必须在创建任务时再明确一次，不能只说“帮我监控训练”。临时 SSH 密码通过当前对话的继承上下文提供，Luna 仅注入自己的 `REMOTE_PASSWORD` 进程环境；禁止把密码重写到任务名、watcher 输出、服务器文件、文档或 Git。如子代理没有继承凭据，应回报主代理重新以完整上下文派发，不猜测、不落盘。

### 5.2 可复用的子代理任务模板

```text
你负责只读监控 <RUN_NAME>，你不是仓库中唯一代理。
连接：<HOST>:<PORT> / <USER>；使用当前上下文中的临时凭据，只注入进程环境，不落盘、不回显。
路径：RUN_DIR=<ABS_RUN_DIR>，LAUNCHER_LOG=<ABS_LOG>，SOURCE_DIR=<ABS_SOURCE_DIR>。
预期：<MAX_STEPS>，<TRAIN_COUNT/GROUP_SIZE>，<STEP125_DEV_COUNT>，<FINAL_DEV_COUNT>，<REQUIRED_ARTIFACTS>。
只读边界：不 kill/rm/mv，不启动或 retry，不修改文件/Git/状态标记，不做科学决策。
实现：在单个 turn 中启动服务器侧阻塞 `while + sleep` watcher，直到 COMPLETE、FAILED 或明确异常；正常快照只留在你自己的 turn 内。
回传：常态不通知主代理；只在终态/异常时发一条结构化消息，然后完成验收。
```

### 5.3 单 turn 阻塞 watcher

Luna 在自己的单个 turn 内通过 `tools/remote_step.py` 打开一个持续 SSH 命令，服务器端使用 `while + sleep`。如本地执行器回传 session id，只继续等待同一 session，不断开后反复新建 SSH。主代理则直接等待 Luna 的终态消息。

```text
while run 未进入终态:
    读取 RUNNING / COMPLETE / FAILED / exit_code 内容
    从原始日志和 JSONL 读取阶段、step/batch/覆盖，计算完成比例和 ETA
    只读检查 GPU、8901、trainer/Ray/Gunicorn、磁盘和错误关键字
    在 Luna turn 内输出时间、阶段、进度、ETA、健康和下一间隔
    按 ETA 选择 sleep 时长
终态或明确异常后:
    运行完整只读验收
    向主代理只回传一次结构化结果
```

进度必须从事实源推导，不仅看 launcher 是否存活：

| 阶段 | 主要事实源 | 对 R4-RAW 的判定示例 |
| --- | --- | --- |
| startup | `RUNNING`、`run.env`、`source_commit.txt`、`source_status.txt`、进程/GPU/8901 | source clean，trainer 与 reward server 健康 |
| train | `checkpoints/experiment_log.jsonl`、debug/raw train rollout | `step/max_steps`；最终 250 step、4,000 train rollout |
| intermediate dev | checkpoint tracker、`step125_dev_rollouts.jsonl` | step125 checkpoint 和 566/566 dev |
| final dev | `dev_rollouts.jsonl`、`final_dev_metrics.json` | step250 的 566/566 dev |
| postprocess | diagnosis、paired report、training evidence、result hash | 曲线、代表样本、三份 paired report 与 hash 完整 |
| terminal | `COMPLETE/FAILED`、`exit_code` 文件内容、资源状态 | `COMPLETE`、内容为 `0`，GPU/Ray/Gunicorn/8901 回收 |

剩余 ETA 的检查间隔为：

- 大于 60 分钟：每 60 分钟；
- 60–30 分钟：每 30 分钟；
- 30–10 分钟：每 10 分钟；
- 不超过 10 分钟：每 5 分钟。


### 5.4 异常升级与终态回传

以下任一条成立时 Luna 才主动通知主代理：`COMPLETE`；`FAILED`；日志明确出现 OOM/traceback/CUDA/no-space/killed/non-finite；主进程和 `RUNNING` 状态矛盾；GPU/8901 与当前阶段明确矛盾；磁盘余量已威胁当前 checkpoint 或证据落盘。单次日志暂无新行、评估阶段 GPU 利用率下降或原始轮转文件消失，本身不等于异常；必须与进程、状态文件和日志合并判断。

终态报告固定包含：

1. `status`、`COMPLETE/RUNNING/FAILED`、`exit_code` **文件内容**；禁止再用文件 size 代替内容。
2. 训练 step/group/rollout 和各评估 split 的预期/实际/唯一数、parse/clipping/non-finite。
3. checkpoint/LoRA、metrics、paired report、曲线/代表样本、manifest/hash 的完整性。
4. 关键指标和成本摘要，但不替主代理做科学晋级决策。
5. 峰值显存、墙钟时间、磁盘，以及 GPU、trainer、Ray、Gunicorn/8901 是否全部回收。
6. 错误扫描命中和“未执行任何删除/修改”的明确声明。

主代理收到回传后才进行 paired analysis、科学门控、台账回填或精确清理。如果 Luna 报异常，主代理先做只读复核，再决定是否需要停止或 retry；子代理不得为了“自愈”而改变服务器状态。

## 6. 服务器资源与清理边界

常用只读检查：

```text
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits
fuser 8901/tcp
ps -eo pid,ppid,pgid,stat,etime,args
df -h /root/autodl-tmp
git status --short --branch
```

清理前必须先列出目标的绝对路径、大小和对应结论，确认所有进程已停止，再用同一个远端 Linux shell 删除显式路径；禁止用宽泛 glob、环境变量展开或递归删除 workspace/root。当前清理后 `/root/autodl-tmp` 约剩余 63 GB。保留集合不得删除：

```text
experiments/safe_grpo/e0_stage2_dev_seed20260812
experiments/safe_grpo/d0_stage2_train_n4_seed20260812
experiments/safe_grpo/r0_difficulty_bias_seed20260812_retry1
experiments/safe_grpo/e2_fals_lora_1k_seed20260812
experiments/safe_grpo/f0_e2_step50_dev_seed20260812
experiments/safe_grpo/F1_HELDOUT_ACCESSED
manifests/train_tokens.txt
manifests/dev_tokens.txt
manifests/heldout_tokens.txt
```

## 7. 下一路线启动前

先写新的科学问题、唯一变量、对照、预算、数据边界、技术门控、晋级线和失败分支，再实现和执行。新路线应把 E0/Stage-2 与 E2 作为已知基线，但不能复用旧 held-out 做一次性确认；需要建立新的、版本化且此前未用于任何模型推理的最终评估集，或明确把旧 held-out 降级为已访问分析集。不得根据新 dev 结果回头恢复 GRPO 超参数搜索。
