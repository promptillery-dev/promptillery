import pytest

from promptillery.ablation import AblationStudyRunner
from promptillery.config import ExperimentConfig


def test_ablation_runner_rejects_partial_shard_configuration():
    config = ExperimentConfig(name="shard-test")

    with pytest.raises(ValueError, match="provided together"):
        AblationStudyRunner(config, shard_index=0)

    with pytest.raises(ValueError, match="provided together"):
        AblationStudyRunner(config, shard_count=2)


def test_ablation_runner_rejects_invalid_shard_bounds():
    config = ExperimentConfig(name="shard-test")

    with pytest.raises(ValueError, match="at least 1"):
        AblationStudyRunner(config, shard_index=0, shard_count=0)

    with pytest.raises(ValueError, match="between 0 and shard_count"):
        AblationStudyRunner(config, shard_index=2, shard_count=2)


def test_ablation_runner_splits_generated_grid_deterministically():
    config = ExperimentConfig(
        name="shard-test",
        policy_name=["student_only", "cost_heuristic", "frugalkd_p"],
        token_budget=[25, 100],
    )
    configs = config.generate_ablation_configs()

    shard_zero = AblationStudyRunner(
        config,
        shard_index=0,
        shard_count=2,
    )._select_shard_configs(configs)
    shard_one = AblationStudyRunner(
        config,
        shard_index=1,
        shard_count=2,
    )._select_shard_configs(configs)

    assert [cfg.name for cfg in shard_zero] == [
        configs[0].name,
        configs[2].name,
        configs[4].name,
    ]
    assert [cfg.name for cfg in shard_one] == [
        configs[1].name,
        configs[3].name,
        configs[5].name,
    ]
    assert sorted(cfg.name for cfg in shard_zero + shard_one) == sorted(
        cfg.name for cfg in configs
    )
