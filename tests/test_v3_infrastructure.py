from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_local_dataset_files_use_explicit_file_loader() -> None:
    dataset = (ROOT / "EasyR1/verl/utils/dataset.py").read_text(encoding="utf-8")

    assert "if os.path.isfile(data_path):" in dataset
    assert "load_dataset(file_type, data_files=data_path, split=data_split)" in dataset


def test_validation_supports_single_rollout_and_exclusive_lock() -> None:
    config = (ROOT / "EasyR1/verl/trainer/config.py").read_text(encoding="utf-8")
    trainer = (ROOT / "EasyR1/verl/trainer/ray_trainer.py").read_text(encoding="utf-8")

    assert "dev_access_lock_path: Optional[str] = None" in config
    assert "not config.trainer.val_only" in trainer
    assert 'with open(lock_path, "x", encoding="utf-8") as handle:' in trainer
    assert trainer.index('with open(lock_path, "x"') < trainer.index('print("Start validation...")')


def test_multi_epoch_ppo_recreates_mini_batch_iterator() -> None:
    actor = (ROOT / "EasyR1/verl/workers/actor/dp_actor.py").read_text(encoding="utf-8")

    assert "for epoch in range(1, self.config.ppo_epochs + 1):" in actor
    assert "mini_batch_iter = tqdm(mini_batches, desc=\"Train mini-batches\", position=1)" in actor
    assert "for mini_batch in mini_batch_iter:" in actor
    assert "mini_batches = tqdm(mini_batches" not in actor
    assert "actor/epoch{epoch}/log_ratio_mean" in actor
    assert "actor/epoch{epoch}/log_ratio_p95" in actor
    assert "actor/epoch{epoch}/log_ratio_p99" in actor
    assert "actor/epoch{epoch}/ppo_kl" in actor
    assert "actor/epoch{epoch}/pg_clipfrac_higher" in actor
    assert "actor/epoch{epoch}/pg_clipfrac_lower" in actor
    assert "actor/epoch{epoch}/pg_loss" in actor
    assert "actor/epoch{epoch}/grad_norm" in actor
