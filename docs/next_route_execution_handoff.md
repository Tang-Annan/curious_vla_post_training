# Curious-VLA 下一技术路线执行交接

> 本文只交接执行方式和不可违反的状态边界，不为下一路线预选算法。解释默认使用简体中文；代码、技术术语和学术引用保持准确英文。

## 1. 当前不可变状态

- 本地工作区：`D:\Desktop\curious_vla`
- 当前分支：`codex/post-training-analysis`
- 可写远端：`post-training` → `Tang-Annan/curious_vla_post_training`
- 服务器仓库：`/root/autodl-tmp/curious-vla-workspace/src/curious_vla_post_training`
- 服务器实验根目录：`/root/autodl-tmp/curious-vla-workspace/experiments/safe_grpo`
- GRPO、Dr.GRPO、Dynamic Sampling 和 Recovery 路线已终止；不得继续 R1/R2/R3、不得重启 F1。
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

## 5. 长任务启动与 Luna 监控

通过 Git 同步的 launcher 采用独立目录和后台日志：

```text
nohup bash scripts/<launcher>.sh > /root/autodl-tmp/curious-vla-workspace/logs/<run>.launcher.log 2>&1 < /dev/null &
```

正常启动的最低证据是 `RUNNING`、`run.env`、`source_commit.txt`、空的 `source_status.txt`、主进程、GPU 占用和 reward server 8901 健康。实验目录禁止覆盖；技术失败保留目录并只允许在明确规则授权时用新目录 retry。

长任务正常启动后，复用 Luna 子代理只读监控。子代理每次在自身内部显示：当前 step/batch、完成比例、ETA、GPU/8901/错误健康和下一次间隔；正常进展不回传主对话，只在 `COMPLETE` 或明确异常时通知主进程。剩余 ETA 的检查间隔为：

- 大于 60 分钟：每 60 分钟；
- 60–30 分钟：每 30 分钟；
- 30–10 分钟：每 10 分钟；
- 不超过 10 分钟：每 5 分钟。

在 Codex 目标模式下，监控期应暂停主目标，避免主线程频繁唤醒。完成验收要读取 `exit_code` 内容，并检查覆盖、manifest 互斥、checkpoint/metrics、OOM/traceback/CUDA/no-space/killed、GPU、Ray、Gunicorn/8901 和残留进程。

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
