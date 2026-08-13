# E2 FALS manifest provenance

正式 E2 的唯一输入为服务器文件：

`/root/autodl-tmp/curious-vla-workspace/manifests/fals_d0_seed20260812/fals_top_1000.txt`

- 来源：D0 `d0_train_rollouts.jsonl`，4,525 个冻结 train token，每个 4 个 rollout。
- 排序：`(1 - mean_reward) * (max_reward - mean_reward)`，降序；同分按 token 升序。
- 预算：top 1,000。
- 边界：1,000 个唯一 token，train 外、dev、held-out 重叠均为 0。
- SHA-256：`fd62a6f204806beff51fa7e1fb0f853027655b4b47f00f9633c787b04e0ffed0`。
- 完整 ranking SHA-256：`e1fdc70eb5818348dc6df0747965a7fb712bc16d132496fea317fefe6c0df9ed`。

原始 ranking、manifest 和验证摘要保存在服务器同目录；本文件只记录可追溯标识，不复制第二份正式 manifest。
