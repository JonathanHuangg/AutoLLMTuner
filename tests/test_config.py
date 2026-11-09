"""Config loading: every shipped run config parses, and paths resolve
relative to the config file no matter where the process runs from."""

import os
from pathlib import Path

import pytest

from evolutionary.config import LatencyConfig, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"


@pytest.mark.parametrize(
    "name",
    [
        "qwen2.5_run_config.yaml",
        "mistral_run_config.yaml",
        "trinity_run_config.yaml",
        "llama-run-config.yaml",
        "smoke_run_config.yaml",
    ],
)
def test_all_shipped_configs_load(name):
    cfg = load_config(str(CONFIGS / name))
    assert cfg.run.population_size > 0
    assert cfg.run.generations > 0
    assert 0 < cfg.run.mutation_rate < 1
    assert 0 < cfg.run.crossover_probability <= 1
    assert 0 < cfg.run.elitism_frac <= 1
    assert cfg.evaluation.measure_latency in LatencyConfig
    assert cfg.model.default_fp_bits == 16


def test_smoke_config_is_mock_only():
    cfg = load_config(str(CONFIGS / "smoke_run_config.yaml"))
    assert cfg.evaluation.mock_backend is True
    assert cfg.evaluation.measure_latency is LatencyConfig.NONE


def test_real_run_configs_are_not_mock():
    for name in (
        "qwen2.5_run_config.yaml",
        "mistral_run_config.yaml",
        "trinity_run_config.yaml",
        "llama-run-config.yaml",
    ):
        cfg = load_config(str(CONFIGS / name))
        assert cfg.evaluation.mock_backend is False


def test_paths_resolve_relative_to_config_file(tmp_path, monkeypatch):
    # Run from an unrelated working directory: config-relative paths must
    # still point at the real files, not at $CWD.
    monkeypatch.chdir(tmp_path)
    cfg = load_config(str(CONFIGS / "qwen2.5_run_config.yaml"))
    assert os.path.exists(cfg.units.spec_file)
    assert os.path.exists(cfg.units.layer_spec_file)
    assert os.path.exists(cfg.model.param_counts_file)
    assert os.path.exists(cfg.hardware.lut_file)
    # out_dir resolves under the repo, not under tmp_path
    assert Path(cfg.io.out_dir).resolve().parent == REPO_ROOT


def test_val_tokens_schedule_splits_generations():
    from evolutionary.main_search import val_tokens_for_epoch

    schedule = (512, 2048)
    assert [val_tokens_for_epoch(e, 10, schedule) for e in range(10)] == (
        [512] * 5 + [2048] * 5
    )
