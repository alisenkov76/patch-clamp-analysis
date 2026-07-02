"""
Квантование тока в число открытых каналов + occupancy-метрики.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from patchclamp.config import ExperimentConfig


def quantize_open_channels(
    current: np.ndarray,
    I_closed: float,
    delta_I: float,
    cfg: ExperimentConfig,
) -> np.ndarray:
    """
    Квантование тока в число открытых каналов 0, 1, 2, ...

    current  : очищенный ток (pA)
    I_closed : уровень закрытого состояния (pA)
    delta_I  : шаг одного канала (pA), со знаком
    cfg      : берём cfg.max_channels

    Возвращает int-массив той же длины, что current.
    """
    if delta_I is None or not np.isfinite(delta_I) or delta_I == 0:
        return np.zeros(len(current), dtype=int)

    step      = abs(delta_I)
    direction = np.sign(delta_I)

    # проекция на направление открытия
    proj = (current - I_closed) * direction

    # округление к ближайшему уровню
    n = np.floor((proj + 0.5 * step) / step).astype(int)
    n = np.clip(n, 0, cfg.max_channels)
    return n


def compute_occupancy(
    n_open: np.ndarray,
    duration_sec: float,
) -> dict:
    """
    Считает occupancy-метрики по массиву n_open.

    Возвращает dict:
        Nmax         : максимальное число одновременно открытых каналов
        Nmean        : среднее число открытых каналов
        Popen        : доля времени хотя бы 1 канал открыт
        open_events  : число переходов 0→>0
        events_per_s : open_events / duration_sec
    """
    Nmax  = int(n_open.max())
    Nmean = float(n_open.mean())
    Popen = float((n_open >= 1).mean())

    # переходы closed→open: 0 → >0
    open_events = int(np.sum((n_open[:-1] == 0) & (n_open[1:] > 0)))
    events_per_s = open_events / duration_sec if duration_sec > 0 else np.nan

    return {
        "Nmax":         Nmax,
        "Nmean":        round(Nmean, 4),
        "Popen":        round(Popen, 4),
        "open_events":  open_events,
        "events_per_s": round(events_per_s, 3),
    }


def build_occupancy_table(df_intervals: pd.DataFrame) -> pd.DataFrame:
    """
    Агрегирует occupancy-метрики по напряжению.

    df_intervals должен содержать колонки:
        voltage_mV, duration_sec, Nmax, Nmean, Popen,
        open_events, events_per_s

    Возвращает DataFrame с одной строкой на напряжение.
    """
    required = {"voltage_mV", "duration_sec", "Nmax", "Nmean",
                "Popen", "open_events", "events_per_s"}
    missing = required - set(df_intervals.columns)
    if missing:
        raise ValueError(f"build_occupancy_table: нет колонок {missing}")

    records = []

    for v, grp in df_intervals.groupby("voltage_mV"):
        grp = grp.dropna(subset=["Nmax", "Nmean", "Popen"])
        if grp.empty:
            continue

        n_intervals  = len(grp)
        total_sec    = float(grp["duration_sec"].sum())
        weights      = grp["duration_sec"].values

        Nmax_obs     = int(grp["Nmax"].max())

        # взвешенные средние по длительности
        Nmean_w = float(np.average(grp["Nmean"].values, weights=weights))
        Popen_w = float(np.average(grp["Popen"].values, weights=weights))

        # SEM для Popen между интервалами
        Popen_sem = float(grp["Popen"].sem()) if n_intervals > 1 else np.nan

        total_events   = int(grp["open_events"].sum())
        mean_events_s  = float(
            np.average(grp["events_per_s"].values, weights=weights)
        )

        records.append({
            "voltage_mV":    int(v),
            "n_intervals":   n_intervals,
            "total_sec":     round(total_sec, 1),
            "Nmax_obs":      Nmax_obs,
            "Nmean_wavg":    round(Nmean_w, 4),
            "Popen_wavg":    round(Popen_w, 4),
            "Popen_sem":     round(Popen_sem, 4) if np.isfinite(Popen_sem) else np.nan,
            "total_events":  total_events,
            "mean_events_s": round(mean_events_s, 3),
        })

    return (
        pd.DataFrame(records)
        .sort_values("voltage_mV")
        .reset_index(drop=True)
    )
