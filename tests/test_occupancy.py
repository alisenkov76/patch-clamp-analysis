"""
Тесты квантования и occupancy-метрик.
"""
import numpy as np
import pytest
from patchclamp.config import ExperimentConfig
from patchclamp.occupancy import (
    quantize_open_channels,
    compute_occupancy,
    build_occupancy_table,
)
import pandas as pd

TOML = __import__("pathlib").Path(__file__).parents[1] / "configs" / "gramicidin_a.toml"


@pytest.fixture
def cfg():
    return ExperimentConfig.from_toml(TOML)


# ── quantize ─────────────────────────────────────────────

def test_quantize_closed_state(cfg):
    """Сигнал на уровне closed → все нули."""
    y = np.zeros(1000)
    n = quantize_open_channels(y, I_closed=0.0, delta_I=2.0, cfg=cfg)
    assert np.all(n == 0)


def test_quantize_one_channel(cfg):
    """Сигнал на уровне 1 канала → все единицы."""
    y = np.full(1000, 2.0)
    n = quantize_open_channels(y, I_closed=0.0, delta_I=2.0, cfg=cfg)
    assert np.all(n == 1)


def test_quantize_two_channels(cfg):
    """Сигнал на уровне 2 каналов → все двойки."""
    y = np.full(1000, 4.0)
    n = quantize_open_channels(y, I_closed=0.0, delta_I=2.0, cfg=cfg)
    assert np.all(n == 2)


def test_quantize_clip_max(cfg):
    """Сигнал выше max_channels → клипуется в max_channels."""
    y = np.full(1000, 100.0)
    n = quantize_open_channels(y, I_closed=0.0, delta_I=2.0, cfg=cfg)
    assert np.all(n == cfg.max_channels)


def test_quantize_nan_delta_returns_zeros(cfg):
    """delta_I=NaN → возвращаем нули."""
    y = np.ones(1000)
    n = quantize_open_channels(y, I_closed=0.0, delta_I=np.nan, cfg=cfg)
    assert np.all(n == 0)


def test_quantize_negative_delta(cfg):
    """Отрицательный delta_I (ток идёт в минус при -V)."""
    y = np.full(1000, -2.0)
    n = quantize_open_channels(y, I_closed=0.0, delta_I=-2.0, cfg=cfg)
    assert np.all(n == 1)


# ── compute_occupancy ────────────────────────────────────

def test_popen_all_closed(cfg):
    n_open = np.zeros(10000, dtype=int)
    occ = compute_occupancy(n_open, duration_sec=2.0)
    assert occ["Popen"] == 0.0
    assert occ["Nmax"] == 0
    assert occ["open_events"] == 0


def test_popen_always_open(cfg):
    n_open = np.ones(10000, dtype=int)
    occ = compute_occupancy(n_open, duration_sec=2.0)
    assert occ["Popen"] == 1.0
    assert occ["Nmax"] == 1


def test_popen_half(cfg):
    n_open = np.array([0, 1] * 5000, dtype=int)
    occ = compute_occupancy(n_open, duration_sec=2.0)
    assert abs(occ["Popen"] - 0.5) < 0.01


def test_open_events_count(cfg):
    """0→1 переходы: [0,1,0,1,0] → 2 события."""
    n_open = np.array([0, 1, 0, 1, 0], dtype=int)
    occ = compute_occupancy(n_open, duration_sec=1.0)
    assert occ["open_events"] == 2


# ── build_occupancy_table ────────────────────────────────

def test_build_occupancy_table_shape(cfg):
    df = pd.DataFrame({
        "voltage_mV":   [-100, -100, 100, 100],
        "duration_sec": [300.0, 200.0, 250.0, 150.0],
        "Nmax":         [2, 1, 2, 2],
        "Nmean":        [0.5, 0.3, 0.6, 0.4],
        "Popen":        [0.4, 0.2, 0.5, 0.3],
        "open_events":  [120, 80, 150, 90],
        "events_per_s": [0.4, 0.4, 0.6, 0.6],
    })
    out = build_occupancy_table(df)
    assert len(out) == 2
    assert set(out["voltage_mV"]) == {-100, 100}


def test_build_occupancy_table_weighted_popen(cfg):
    """Popen_wavg должен быть взвешен по длительности."""
    df = pd.DataFrame({
        "voltage_mV":   [100, 100],
        "duration_sec": [100.0, 900.0],   # 100 сек Popen=1.0, 900 сек Popen=0.0
        "Nmax":         [1, 0],
        "Nmean":        [1.0, 0.0],
        "Popen":        [1.0, 0.0],
        "open_events":  [10, 0],
        "events_per_s": [0.1, 0.0],
    })
    out = build_occupancy_table(df)
    # ожидаем 100/1000 = 0.1
    assert abs(out.loc[0, "Popen_wavg"] - 0.1) < 0.01
