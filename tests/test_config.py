"""
Тесты загрузки конфига из .toml
"""
import pytest
from pathlib import Path
from patchclamp.config import ExperimentConfig


TOML = Path(__file__).parents[1] / "configs" / "gramicidin_a.toml"


def test_load_from_toml():
    cfg = ExperimentConfig.from_toml(TOML)
    assert cfg.name == "Gramicidin A"
    assert cfg.pdb_id == "1MAG"


def test_target_voltages():
    cfg = ExperimentConfig.from_toml(TOML)
    assert cfg.target_voltages == {-150, -100, -50, 50, 100, 150}


def test_manual_exclude_keys():
    cfg = ExperimentConfig.from_toml(TOML)
    assert "26629001.abf" in cfg.manual_exclude
    assert "26629003.abf" in cfg.manual_exclude


def test_manual_exclude_values():
    cfg = ExperimentConfig.from_toml(TOML)
    # 26629003 должен иметь 2 зоны исключения
    zones = cfg.manual_exclude["26629003.abf"]
    assert len(zones) == 2
    assert zones[0] == (978, 1628)
    assert zones[1] == (4556, 4747)


def test_defaults_are_sane():
    cfg = ExperimentConfig.from_toml(TOML)
    assert cfg.lowpass_hz == 400.0
    assert cfg.min_duration_sec == 30.0
    assert cfg.max_channels == 6
    assert cfg.noise_threshold == 4.0


def test_summary_runs():
    cfg = ExperimentConfig.from_toml(TOML)
    s = cfg.summary()
    assert "Gramicidin A" in s
    assert "1MAG" in s
