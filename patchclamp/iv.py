"""
Построение ВАХ и оценка проводимости одиночного канала.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from patchclamp.config import ExperimentConfig


# ─────────────────────────────────────────────
# Утилиты
# ─────────────────────────────────────────────

def pooled_sem(sems: np.ndarray, ns: np.ndarray) -> float:
    """
    Pooled SEM, взвешенный по числу точек.

    σ_pool = sqrt( Σ(n_i * σ_i²) / Σn_i ) / sqrt(Σn_i)
    """
    ns   = np.asarray(ns,   dtype=float)
    sems = np.asarray(sems, dtype=float)

    # заменяем NaN нулями (интервалы без оценки σ не вносят вклад)
    sems = np.where(np.isfinite(sems), sems, 0.0)

    total_n    = ns.sum()
    pooled_var = np.sum(ns * sems**2) / total_n
    return float(np.sqrt(pooled_var / total_n))


# ─────────────────────────────────────────────
# Построение ВАХ
# ─────────────────────────────────────────────

def build_iv(
    exp_df: pd.DataFrame,
    ctrl_df: Optional[pd.DataFrame],
    cfg: ExperimentConfig,
) -> pd.DataFrame:
    """
    Усредняет single-channel ΔI по напряжению.

    ВАЖНО:
    ΔI = I_open - I_closed уже убирает baseline внутри интервала.
    Контроль НЕ вычитаем — добавляем только для справочного графика.

    QC-флаг qc_short: total_sec < cfg.min_total_sec_for_fit.

    Возвращает DataFrame:
        voltage_mV, mean_delta_I, pooled_sem, corrected_dI,
        corrected_sem, n_intervals, n_files, total_sec,
        conductance_pS, qc_short, files,
        (опционально) control_I, control_noise
    """
    exp = exp_df.dropna(subset=["delta_I", "voltage_mV"]).copy()
    exp = exp[exp["voltage_mV"] != 0]

    rows = []
    for voltage, grp0 in exp.groupby("voltage_mV"):
        grp = grp0.copy()

        w       = grp["n_points"].astype(float).values
        y       = grp["delta_I"].astype(float).values
        mean_dI = float(np.average(y, weights=w))

        p_sem = pooled_sem(
            sems=grp["sem_delta"].astype(float).fillna(0).values,
            ns=grp["n_points"].astype(float).values,
        )

        dur_col   = "duration_sec" if "duration_sec" in grp.columns else "duration"
        total_sec = float(grp[dur_col].sum()) if dur_col in grp.columns else np.nan

        rows.append({
            "voltage_mV":   int(voltage),
            "mean_delta_I": mean_dI,
            "pooled_sem":   float(p_sem),
            "corrected_dI": mean_dI,       # без leak-коррекции
            "corrected_sem": float(p_sem),
            "n_intervals":  int(len(grp)),
            "n_files":      int(grp["file"].nunique()),
            "total_sec":    total_sec,
            "files":        ", ".join(grp["file"].unique()),
        })

    summary = (
        pd.DataFrame(rows)
        .sort_values("voltage_mV")
        .reset_index(drop=True)
    )

    # QC-флаг: мало данных → не доверяем точке в фите
    summary["qc_short"] = summary["total_sec"] < cfg.min_total_sec_for_fit

    # Контроль — только baseline для справочного графика
    if ctrl_df is not None and not ctrl_df.empty:
        ctrl_mean = (
            ctrl_df.groupby("voltage_mV")
            .agg(
                control_I=("I_closed", "mean"),
                control_noise=(
                    "sem_delta",
                    lambda x: float(
                        np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2))
                    ),
                ),
            )
            .reset_index()
        )
        summary = summary.merge(ctrl_mean, on="voltage_mV", how="left")

    # Проводимость: g = 1000 * |ΔI / V|  (pS)
    v = summary["voltage_mV"].astype(float)
    summary["conductance_pS"] = (
        1000.0 * summary["corrected_dI"].astype(float) / v
    ).abs()

    return summary


# ─────────────────────────────────────────────
# Линейный фит
# ─────────────────────────────────────────────

def fit_linear_iv(
    summary: pd.DataFrame,
    cfg: ExperimentConfig,
) -> Tuple[pd.DataFrame, float, float]:
    """
    Линейный фит I = G * V через 0 по надёжным точкам.

    Надёжные точки: |V| >= 100 mV И qc_short == False.
    Ненадёжные точки (|V| < 100 mV) заменяются предсказанием фита
    и помечаются is_predicted = True.

    Возвращает: (summary, G_pS, R2)
    """
    summary = summary.copy()
    summary["corrected_dI_meas"]  = summary["corrected_dI"]
    summary["corrected_sem_meas"] = summary["corrected_sem"]
    summary["is_predicted"] = False

    # надёжные точки для фита
    reliable_mask = (
        (summary["voltage_mV"].abs() >= 100) &
        (~summary["qc_short"].fillna(False))
    )
    reliable = summary[reliable_mask]

    if len(reliable) < 2:
        print("  ⚠️  Мало надёжных точек для фита (нужно ≥ 2)")
        return summary, np.nan, np.nan

    v = reliable["voltage_mV"].astype(float).values
    i = reliable["corrected_dI"].astype(float).values

    # фит через 0: G = (v·i) / (v·v)
    slope0   = float((v @ i) / (v @ v))   # pA/mV
    G_fit_pS = slope0 * 1000.0            # pS

    # ошибка slope0
    resid    = i - slope0 * v
    RSS      = float(np.sum(resid**2))
    dof      = max(1, len(v) - 1)
    sigma2   = RSS / dof
    se_slope = float(np.sqrt(sigma2 / np.sum(v**2)))

    # R² (через 0: TSS = Σi²)
    TSS = float(np.sum(i**2))
    R2  = 1.0 - RSS / TSS if TSS > 0 else np.nan

    print(f"\n  Линейный фит через 0 (|V|≥100, qc_ok):")
    print(f"  G = {G_fit_pS:.2f} pS  |  R² = {R2:.4f}  |  n = {len(v)}")

    # предсказание и замена ненадёжных точек
    summary["predicted_dI"]        = slope0 * summary["voltage_mV"].astype(float)
    summary["conductance_fit_pS"]  = G_fit_pS

    unreliable = summary["voltage_mV"].abs() < 100
    if unreliable.any():
        print(f"\n  Замена |V|<100 mV предсказанием фита:")
        for _, row in summary[unreliable].iterrows():
            pred = slope0 * float(row["voltage_mV"])
            print(f"    {int(row['voltage_mV']):+d} mV: "
                  f"measured={row['corrected_dI']:.2f} → predicted={pred:.2f} pA")

        summary.loc[unreliable, "corrected_dI"]  = (
            slope0 * summary.loc[unreliable, "voltage_mV"].astype(float)
        )
        summary.loc[unreliable, "corrected_sem"] = (
            se_slope * summary.loc[unreliable, "voltage_mV"].abs().astype(float)
        )
        summary.loc[unreliable, "is_predicted"] = True

    # пересчёт проводимости
    v_all = summary["voltage_mV"].astype(float)
    summary["conductance_pS"] = (
        1000.0 * summary["corrected_dI"].astype(float) / v_all
    ).abs()

    return summary, G_fit_pS, R2
