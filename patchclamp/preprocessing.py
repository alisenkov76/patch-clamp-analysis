"""
Предобработка patch-clamp трасс:
- Фильтр Баттерворта (lowpass)
- Автомаска шума через MAD
"""
from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal

from patchclamp.config import ExperimentConfig


def lowpass_filter(
    data: np.ndarray,
    fs: float,
    cfg: ExperimentConfig,
) -> np.ndarray:
    """
    Фильтр Баттерворта 4-го порядка (lowpass).

    data : сигнал (pA)
    fs   : частота дискретизации (Гц)
    cfg  : берём cfg.lowpass_hz
    """
    nyq = 0.5 * fs
    cutoff = cfg.lowpass_hz

    if cutoff >= nyq:
        raise ValueError(
            f"lowpass_hz={cutoff} >= Nyquist={nyq}. "
            f"Уменьши lowpass_hz в конфиге."
        )

    b, a = scipy_signal.butter(4, cutoff / nyq, btype="low")
    return scipy_signal.filtfilt(b, a, data)


def mask_noisy_segments(
    current: np.ndarray,
    fs: float,
    cfg: ExperimentConfig,
) -> np.ndarray:
    """
    Автомаска шумных участков через MAD.

    Алгоритм:
    1. Считаем глобальный MAD (робастный σ)
    2. Скользящим окном (50 мс) проверяем std каждого чанка
    3. Чанки с std > threshold * MAD → маскируем (False)

    Возвращает булев массив той же длины, что current.
    True  = чистый сигнал
    False = шум, исключить
    """
    window = max(2, int((cfg.noise_window_ms / 1000.0) * fs))
    global_mad = 1.4826 * np.median(
        np.abs(current - np.median(current))
    )

    mask = np.ones(len(current), dtype=bool)
    step = max(1, window // 2)

    for i in range(0, len(current) - window, step):
        chunk = current[i : i + window]
        if np.std(chunk) > cfg.noise_threshold * global_mad:
            mask[i : i + window] = False

    frac_bad = 1.0 - mask.mean()
    if frac_bad > 0.01:
        print(f"      шум: исключено {frac_bad * 100:.1f}% точек")

    return mask


def preprocess(
    current: np.ndarray,
    fs: float,
    cfg: ExperimentConfig,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Полный пайплайн предобработки одного интервала.

    Возвращает:
        current_filt  : отфильтрованный сигнал (полная длина)
        noise_mask    : булев массив (True = чистый)

    Использование:
        filt, mask = preprocess(current, fs, cfg)
        current_clean = filt[mask]
    """
    # 1. Lowpass
    filt = lowpass_filter(current, fs, cfg)

    # 2. Маска шума
    mask = mask_noisy_segments(filt, fs, cfg)

    if verbose:
        n_clean = mask.sum()
        n_total = len(mask)
        print(f"      preprocess: {n_clean}/{n_total} точек чистых "
              f"({n_clean/n_total*100:.1f}%)")

    return filt, mask
