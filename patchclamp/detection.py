"""
Детекция одноканального тока:
- Гистограммный метод (основной)
- Метод переходов (fallback для редких открытий)
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from patchclamp.config import ExperimentConfig


DetectionResult = Dict[str, object]


def detect_by_histogram(
    current: np.ndarray,
    voltage_mV: int,
    cfg: ExperimentConfig,
) -> DetectionResult:
    """
    Детекция ΔI через гистограмму амплитуд.

    Алгоритм:
    1. Строим гистограмму тока
    2. Сглаживаем gaussian_filter1d
    3. Находим пики → closed = самый "закрытый" по направлению V
    4. Open = ближайший пик в сторону открытия

    Возвращает dict:
        I_closed, I_open, delta_I, sem_delta, n_peaks,
        method, hist=(smoothed_hist, centers)
    """
    direction = np.sign(voltage_mV) if voltage_mV != 0 else 1.0

    hist, edges = np.histogram(current, bins=cfg.hist_bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    hist_s = gaussian_filter1d(hist.astype(float), sigma=2)

    peaks, _ = find_peaks(
        hist_s,
        height=hist_s.max() * cfg.peak_min_height,
        distance=cfg.peak_min_dist,
    )

    # closed = пик с минимальной проекцией на direction
    if len(peaks) > 0:
        peaks_arr = np.array(peaks, dtype=int)
        closed_idx = peaks_arr[np.argmin(direction * centers[peaks_arr])]
    else:
        closed_idx = int(np.argmax(hist_s))

    I_closed = float(centers[closed_idx])

    # кандидаты на open: в сторону direction, дальше min_step_pA
    cand = []
    for p in peaks:
        d = (centers[p] - I_closed) * direction
        if d > cfg.min_step_pA:
            cand.append(p)

    if len(cand) == 0:
        # fallback: квантиль (редкие открытия)
        q = 0.995 if direction > 0 else 0.005
        I_open = float(np.quantile(current, q))
        return {
            "I_closed":  I_closed,
            "I_open":    I_open,
            "delta_I":   float(I_open - I_closed),
            "sem_delta": np.nan,
            "n_peaks":   int(len(peaks)),
            "method":    "histogram_quantile_fallback",
            "hist":      (hist_s, centers),
        }

    # выбираем ближайший open-пик (одноканальный)
    cand_arr = np.array(cand, dtype=int)
    chosen   = cand_arr[np.argmin(np.abs(centers[cand_arr] - I_closed))]
    I_open   = float(centers[chosen])

    return {
        "I_closed":  I_closed,
        "I_open":    I_open,
        "delta_I":   float(I_open - I_closed),
        "sem_delta": np.nan,
        "n_peaks":   int(len(peaks)),
        "method":    "histogram",
        "hist":      (hist_s, centers),
    }


def detect_by_transitions(
    current: np.ndarray,
    voltage_mV: int,
    fs: float,
    smooth_ms: float = 1.0,
    avg_ms: float = 5.0,
    thr_sigma: float = 6.0,
    min_step_pA: float = 0.2,
) -> float:
    """
    Fallback: оценка ΔI по резким переходам (ступенькам).

    Возвращает ΔI (со знаком) или np.nan если переходов не найдено.
    """
    direction = np.sign(voltage_mV) if voltage_mV != 0 else 1.0

    # скользящее среднее
    n_smooth = max(3, int(fs * smooth_ms / 1000.0))
    kernel = np.ones(n_smooth) / n_smooth
    y = np.convolve(current, kernel, mode="same")

    dy = np.diff(y)
    mad = 1.4826 * np.median(np.abs(dy - np.median(dy)))
    if mad == 0:
        return np.nan

    # индексы резких переходов
    idx = np.where(np.abs(dy) > thr_sigma * mad)[0]
    if len(idx) == 0:
        return np.nan

    # дедупликация
    min_gap = max(1, int(fs * avg_ms / 1000.0))
    keep: list[int] = []
    last = -10**9
    for i in idx:
        if i - last >= min_gap:
            keep.append(i)
            last = i

    w = max(5, int(fs * avg_ms / 1000.0))
    steps: list[float] = []

    for i in keep:
        if i - w < 0 or i + w >= len(y):
            continue
        pre  = float(np.mean(y[i - w : i]))
        post = float(np.mean(y[i : i + w]))
        d_open = direction * (post - pre)
        if d_open > min_step_pA:
            steps.append(d_open)

    if len(steps) < 5:
        return np.nan

    steps_arr = np.array(steps)
    hist, edges = np.histogram(steps_arr, bins=40)
    centers = (edges[:-1] + edges[1:]) / 2.0
    step_mag = float(centers[np.argmax(hist)])

    return direction * step_mag


def detect_single_channel_current(
    current: np.ndarray,
    voltage_mV: int,
    fs: float,
    cfg: ExperimentConfig,
) -> DetectionResult:
    """
    Главная функция детекции.

    Пробует histogram → если n_peaks < 2 или delta_I = NaN,
    fallback на transitions.

    Всегда возвращает dict с полями:
        I_closed, I_open, delta_I, sem_delta, n_peaks, method, hist
    """
    res = detect_by_histogram(current, voltage_mV, cfg)

    need_fallback = (
        res["n_peaks"] < 2
        or res["delta_I"] is None
        or (isinstance(res["delta_I"], float) and np.isnan(res["delta_I"]))
    )

    if need_fallback:
        step = detect_by_transitions(
            current, voltage_mV, fs,
            min_step_pA=cfg.min_step_pA,
        )
        if step is not None and np.isfinite(step):
            res = {
                **res,
                "I_open":    res["I_closed"] + step,
                "delta_I":   step,
                "sem_delta": np.nan,
                "method":    "transitions_fallback",
            }
            print(f"      fallback (transitions): ΔI={step:.2f} pA")
        else:
            res = {
                **res,
                "delta_I":   np.nan,
                "sem_delta": np.nan,
                "method":    "failed",
            }
            print(f"      ⚠️  ΔI не определён")

    return res
