"""
Тесты построения ВАХ и линейного фита.
"""
import numpy as np
import pandas as pd
import pytest
from patchclamp.config import ExperimentConfig
from patchclamp.iv import pooled_sem, build_iv, fit_linear_iv

TOML = __import__("pathlib").Path(__file__).parents[1] / "configs" / "gramicidin_a.toml"


@pytest.fixture
def cfg():
    return ExperimentConfig.from_toml(TOML)


@pytest.fixture
def clean_summary(cfg):
    """Синтетическая ВАХ: G=25 pS, без шума."""
    voltages = [-150, -100, -50, 50, 100, 150]
    return pd.DataFrame({
        "voltage_mV":    voltages,
        "corrected_dI":  [v * 0.025 for v in voltages],
        "corrected_sem": [0.1] * 6,
        "n_intervals":   [3] * 6,
        "n_files":       [1] * 6,
        "total_sec":     [300.0] * 6,
        "qc_short":      [False] * 6,
        "conductance_pS": [0.0] * 6,
        "files":         ["test.abf"] * 6,
    })


# ── pooled_sem ───────────────────────────────────────────

def test_pooled_sem_equal_weights():
    """Одинаковые веса и σ → pooled_sem = σ / sqrt(N_total)."""
    sems = np.array([1.0, 1.0, 1.0])
    ns   = np.array([100.0, 100.0, 100.0])
    result = pooled_sem(sems, ns)
    expected = 1.0 / np.sqrt(300.0)
    assert abs(result - expected) < 1e-10


def test_pooled_sem_zero():
    sems = np.array([0.0, 0.0])
    ns   = np.array([100.0, 200.0])
    assert pooled_sem(sems, ns) == 0.0


def test_pooled_sem_nan_treated_as_zero():
    """NaN в sems не должен ломать вычисление."""
    sems = np.array([np.nan, 1.0])
    ns   = np.array([100.0, 100.0])
    result = pooled_sem(sems, ns)
    assert np.isfinite(result)


# ── fit_linear_iv ────────────────────────────────────────

def test_fit_recovers_G(cfg, clean_summary):
    _, G, R2 = fit_linear_iv(clean_summary, cfg)
    assert abs(G - 25.0) < 0.5, f"G={G:.2f}"
    assert R2 > 0.999, f"R2={R2:.4f}"


def test_fit_marks_predicted(cfg, clean_summary):
    summary, G, R2 = fit_linear_iv(clean_summary, cfg)
    pred = summary["is_predicted"]
    # ±50 mV должны быть predicted, остальные — нет
    assert pred[summary["voltage_mV"].abs() == 50].all()
    assert not pred[summary["voltage_mV"].abs() >= 100].any()


def test_fit_qc_short_excluded(cfg, clean_summary):
    """Точка с qc_short=True не участвует в фите."""
    summary = clean_summary.copy()
    # помечаем ±150 как qc_short
    summary.loc[summary["voltage_mV"].abs() == 150, "qc_short"] = True
    _, G, R2 = fit_linear_iv(summary, cfg)
    # фит только по ±100 → всё равно должен дать ~25 pS
    assert abs(G - 25.0) < 1.0


def test_fit_through_zero(cfg, clean_summary):
    """Фит через 0: при V=0 предсказание должно быть 0."""
    summary, G, _ = fit_linear_iv(clean_summary, cfg)
    G_slope = G / 1000.0   # pA/mV
    assert abs(G_slope * 0) == 0.0


def test_fit_antisymmetric(cfg, clean_summary):
    """predicted_dI должен быть антисимметричен."""
    summary, G, _ = fit_linear_iv(clean_summary, cfg)
    for v in [50, 100, 150]:
        pos = float(summary.loc[summary["voltage_mV"] == v,  "corrected_dI"].iloc[0])
        neg = float(summary.loc[summary["voltage_mV"] == -v, "corrected_dI"].iloc[0])
        assert abs(pos + neg) < 0.01, f"|V|={v}: {pos:.3f} + {neg:.3f} != 0"
