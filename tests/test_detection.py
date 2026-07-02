"""
Тесты детекции одноканального тока.
"""
import numpy as np
import pytest
from patchclamp.config import ExperimentConfig
from patchclamp.detection import (
    detect_by_histogram,
    detect_by_transitions,
    detect_single_channel_current,
)

TOML = __import__("pathlib").Path(__file__).parents[1] / "configs" / "gramicidin_a.toml"


@pytest.fixture
def cfg():
    return ExperimentConfig.from_toml(TOML)


@pytest.fixture
def two_level_signal():
    """
    Синтетика: closed=0 pA, open=2 pA, Popen~30%, шум 0.3 pA.
    """
    rng = np.random.default_rng(42)
    fs  = 5000.0
    n   = int(fs * 60)
    state = rng.random(n) < 0.3
    y = np.where(state, 2.0, 0.0) + rng.normal(0, 0.3, n)
    return y, fs


# ── histogram ────────────────────────────────────────────

def test_histogram_detects_two_peaks(cfg, two_level_signal):
    y, fs = two_level_signal
    res = detect_by_histogram(y, voltage_mV=100, cfg=cfg)
    assert res["n_peaks"] >= 2


def test_histogram_I_closed_near_zero(cfg, two_level_signal):
    y, fs = two_level_signal
    res = detect_by_histogram(y, voltage_mV=100, cfg=cfg)
    assert abs(res["I_closed"]) < 0.2, f"I_closed={res['I_closed']:.3f}"


def test_histogram_I_open_near_two(cfg, two_level_signal):
    y, fs = two_level_signal
    res = detect_by_histogram(y, voltage_mV=100, cfg=cfg)
    assert abs(res["I_open"] - 2.0) < 0.2, f"I_open={res['I_open']:.3f}"


def test_histogram_delta_I_positive_for_positive_V(cfg, two_level_signal):
    y, fs = two_level_signal
    res = detect_by_histogram(y, voltage_mV=100, cfg=cfg)
    assert res["delta_I"] > 0


def test_histogram_delta_I_negative_for_negative_V(cfg):
    """При отрицательном V ток открытия идёт в минус."""
    rng = np.random.default_rng(7)
    n   = int(5000 * 60)
    state = rng.random(n) < 0.3
    y = np.where(state, -2.0, 0.0) + rng.normal(0, 0.3, n)
    res = detect_by_histogram(y, voltage_mV=-100, cfg=cfg)
    assert res["delta_I"] < 0


def test_histogram_method_field(cfg, two_level_signal):
    y, fs = two_level_signal
    res = detect_by_histogram(y, voltage_mV=100, cfg=cfg)
    assert "method" in res
    assert "histogram" in res["method"]


# ── transitions fallback ──────────────────────────────────

def test_transitions_detects_step(cfg):
    """Сигнал с редкими ступеньками → transitions находит ~2 pA."""
    rng = np.random.default_rng(99)
    fs  = 5000.0
    n   = int(fs * 120)
    y   = np.zeros(n)

    # вставляем ступеньки каждые 5 сек
    for k in range(0, n, int(fs * 5)):
        end = min(k + int(fs * 0.5), n)
        y[k:end] = 2.0

    y += rng.normal(0, 0.15, n)
    step = detect_by_transitions(y, voltage_mV=100, fs=fs)
    assert step is not None
    assert np.isfinite(step)
    assert abs(step - 2.0) < 0.5, f"step={step:.3f}"


def test_transitions_returns_nan_for_flat_signal(cfg):
    """Плоский сигнал → нет переходов → nan."""
    y  = np.ones(50000) * 1.5
    step = detect_by_transitions(y, voltage_mV=100, fs=5000.0)
    assert step is None or np.isnan(step)


# ── detect_single_channel_current ────────────────────────

def test_full_pipeline_returns_all_keys(cfg, two_level_signal):
    y, fs = two_level_signal
    res = detect_single_channel_current(y, voltage_mV=100, fs=fs, cfg=cfg)
    for key in ("I_closed", "I_open", "delta_I", "sem_delta",
                "n_peaks", "method", "hist"):
        assert key in res, f"Нет ключа: {key}"


def test_full_pipeline_accuracy(cfg, two_level_signal):
    y, fs = two_level_signal
    res = detect_single_channel_current(y, voltage_mV=100, fs=fs, cfg=cfg)
    assert abs(res["delta_I"] - 2.0) < 0.3, f"delta_I={res['delta_I']:.3f}"
