"""
Загрузка ABF файлов и парсинг тегов напряжения.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Set

import numpy as np
import pandas as pd
import pyabf

from patchclamp.config import ExperimentConfig


# ─────────────────────────────────────────────
# Низкоуровневые утилиты
# ─────────────────────────────────────────────

def parse_voltage_from_tag(comment: str) -> float:
    """
    Извлекает напряжение из строки тега Clampex.

    Примеры:
        "Holding on 'Cmd 0' => -50 mV"  →  -50.0
        "Holding on 'Cmd 0' => 100 mV"  →  100.0

    Raises:
        ValueError если паттерн не найден.
    """
    m = re.search(r'=>\s*(-?\d+(?:\.\d+)?)\s*mV', comment)
    if m:
        return float(m.group(1))
    raise ValueError(f"Не удалось распарсить тег: '{comment}'")


def load_sweep(path: str | Path, sweep: int = 0):
    """
    Загружает один sweep из ABF.

    Возвращает (t, y, fs):
        t  : np.ndarray времён (сек)
        y  : np.ndarray тока (pA)
        fs : частота дискретизации (Гц)
    """
    abf = pyabf.ABF(str(path))
    abf.setSweep(sweep)
    return abf.sweepX.copy(), abf.sweepY.copy(), float(abf.dataRate)


# ─────────────────────────────────────────────
# Парсинг интервалов из тегов
# ─────────────────────────────────────────────

def get_voltage_intervals(
    path: str | Path,
    cfg: ExperimentConfig,
) -> pd.DataFrame:
    """
    Строит таблицу интервалов напряжения из тегов gap-free ABF.

    Логика:
    - Тег i задаёт напряжение от t_tag[i] до t_tag[i+1]
    - Последний тег → интервал до конца файла
    - Фильтры: длительность, нулевое напряжение, target_voltages

    Возвращает DataFrame:
        file, voltage_mV, t_start, t_end, duration_sec
    """
    path = Path(path)
    if not path.exists():
        print(f"  ⚠️  Файл не найден: {path}")
        return pd.DataFrame()

    abf      = pyabf.ABF(str(path))
    name     = path.name
    comments = abf.tagComments
    times    = abf.tagTimesSec

    if not comments:
        print(f"  ⚠️  {name}: теги отсутствуют")
        return pd.DataFrame()

    # парсим теги
    voltages: list[float] = []
    valid_times: list[float] = []

    for comment, t in zip(comments, times):
        try:
            v = parse_voltage_from_tag(comment)
            voltages.append(v)
            valid_times.append(t)
        except ValueError as e:
            print(f"    ⚠️  {e}")

    if not voltages:
        print(f"  ⚠️  {name}: теги не распарсились")
        return pd.DataFrame()

    total_dur = float(abf.sweepLengthSec)
    rows: list[dict] = []
    skipped: list[str] = []

    for i, (v, t_start) in enumerate(zip(voltages, valid_times)):
        t_end    = valid_times[i + 1] if i + 1 < len(valid_times) else total_dur
        duration = t_end - t_start
        v_int    = int(v)

        # фильтр: слишком короткий
        if duration < cfg.min_duration_sec:
            skipped.append(
                f"    пропуск {v:+.0f} mV "
                f"[{t_start:.1f}–{t_end:.1f}s] "
                f"длина={duration:.1f}s < {cfg.min_duration_sec}s"
            )
            continue

        # фильтр: нулевое напряжение
        if cfg.skip_zero and v == 0.0:
            skipped.append(
                f"    пропуск  0 mV [{t_start:.1f}–{t_end:.1f}s] (holding)"
            )
            continue

        # фильтр: не в target_voltages
        if cfg.target_voltages and v_int not in cfg.target_voltages:
            skipped.append(
                f"    пропуск {v:+.0f} mV "
                f"[{t_start:.1f}–{t_end:.1f}s] (не в target)"
            )
            continue

        rows.append({
            "file":        name,
            "voltage_mV":  v_int,
            "t_start":     round(t_start, 3),
            "t_end":       round(t_end, 3),
            "duration_sec": round(duration, 1),
        })

    # печать
    print(f"\n  {'─'*52}")
    print(f"  {name}")
    print(f"  {'─'*52}")
    if skipped:
        print("  Пропущено:")
        for s in skipped:
            print(s)

    df = pd.DataFrame(rows)
    if df.empty:
        print("  ❌ Нет подходящих интервалов")
        return df

    print(f"\n  Принято ({len(df)} интервалов):")
    print(df[["voltage_mV","t_start","t_end","duration_sec"]].to_string(index=False))

    found   = set(df["voltage_mV"].unique())
    missing = cfg.target_voltages - found
    if missing:
        print(f"\n  ⚠️  Отсутствуют: {sorted(missing)} mV")
    else:
        print(f"\n  ✅ Все целевые напряжения найдены")

    return df


# ─────────────────────────────────────────────
# Ручные маски
# ─────────────────────────────────────────────

def apply_manual_exclude(
    intervals: pd.DataFrame,
    file_name: str,
    cfg: ExperimentConfig,
) -> pd.DataFrame:
    """
    Исключает или разбивает интервалы по ручным маскам из cfg.

    Возвращает обновлённый DataFrame.
    """
    if file_name not in cfg.manual_exclude:
        return intervals

    exclude_zones = cfg.manual_exclude[file_name]
    MIN_DUR = cfg.min_duration_sec

    def split_one(t_s: float, t_e: float, voltage: int) -> list[tuple[float,float]]:
        segments = [(t_s, t_e)]

        for (ex_s, ex_e) in exclude_zones:
            new_segs: list[tuple[float,float]] = []
            for (s, e) in segments:
                # нет пересечения
                if ex_e <= s or ex_s >= e:
                    new_segs.append((s, e))
                    continue
                # полное покрытие
                if ex_s <= s and ex_e >= e:
                    print(f"    исключён:  {voltage:+d} mV [{s:.0f}–{e:.0f}s]")
                    continue
                # зона в начале
                if ex_s <= s < ex_e < e:
                    new_s = ex_e
                    if e - new_s >= MIN_DUR:
                        print(f"    обрезан (начало): {voltage:+d} mV {s:.0f}→{new_s:.0f}s")
                        new_segs.append((new_s, e))
                    continue
                # зона в конце
                if s < ex_s < e <= ex_e:
                    new_e = ex_s
                    if new_e - s >= MIN_DUR:
                        print(f"    обрезан (конец): {voltage:+d} mV {new_e:.0f}←{e:.0f}s")
                        new_segs.append((s, new_e))
                    continue
                # зона внутри → два куска
                if s < ex_s and ex_e < e:
                    left  = (s, ex_s)
                    right = (ex_e, e)          # ← правильно: после зоны
                    print(f"    разбит: {voltage:+d} mV [{s:.0f}–{e:.0f}s] шум [{ex_s:.0f}–{ex_e:.0f}s]")
                    if left[1]  - left[0]  >= MIN_DUR:
                        new_segs.append(left)
                    if right[1] - right[0] >= MIN_DUR:
                        new_segs.append(right)
                    continue
                new_segs.append((s, e))
            segments = new_segs

        return segments

    rows_out: list[pd.Series] = []
    for _, row in intervals.iterrows():
        segs = split_one(row["t_start"], row["t_end"], int(row["voltage_mV"]))
        for (s, e) in segs:
            r = row.copy()
            r["t_start"]     = round(s, 3)
            r["t_end"]       = round(e, 3)
            r["duration_sec"] = round(e - s, 1)
            rows_out.append(r)

    if not rows_out:
        return pd.DataFrame()

    return pd.DataFrame(rows_out).reset_index(drop=True)
