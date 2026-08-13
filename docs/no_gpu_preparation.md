# No-GPU Preparation

Workspace: `/root/autodl-tmp/curious-vla-workspace`

The no-GPU phase is complete when all 5,656 manifest tokens have one metric cache file and the metadata CSV indexes exactly those files. The cache builder is resumable and safe to restart with the same command:

```bash
base=/root/autodl-tmp/curious-vla-workspace
export OPENSCENE_DATA_ROOT="$base/data/navsim"
export NUPLAN_MAPS_ROOT="$base/data/navsim/maps"
nohup "$base/envs/navsim/bin/python" \
  "$base/src/curious_vla/projects/safe_grpo/cache_manifest.py" \
  --manifest "$base/manifests/released_5656_log_group_split.csv" \
  --logs "$base/data/navsim/navsim_logs/trainval/trainval" \
  --maps "$base/data/navsim/maps" \
  --output "$base/exp_root/metric_cache_released_5656" \
  >"$base/logs/metric_cache_released_5656.log" 2>&1 </dev/null &
```

Final acceptance checks:

```bash
base=/root/autodl-tmp/curious-vla-workspace
find "$base/exp_root/metric_cache_released_5656" -name metric_cache.pkl -type f | wc -l
tail -1 "$base/logs/metric_cache_released_5656.log"
wc -l "$base/exp_root/metric_cache_released_5656/metadata/metric_cache_released_5656_metadata_node_0.csv"
df -h /root/autodl-tmp
```

Expected values are 5,656 cache files, final log coverage `5656/5656`, and 5,657 metadata CSV lines including the header. Data acceptance evidence is stored at `manifests/data_acceptance.json`.

Do not run model inference, reward-server deployment, vLLM, or LoRA-GRPO validation in no-GPU mode. Run those after switching to the GPU instance.
