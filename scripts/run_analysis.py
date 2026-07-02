"""
Точка входа пайплайна patch-clamp анализа.

Использование:
    python scripts/run_analysis.py --config configs/gramicidin_a.toml
    python scripts/run_analysis.py --config configs/gramicidin_a.toml --results results/gA
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyabf

# чтобы запускать из корня репо
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from patchclamp.config       import ExperimentConfig
from patchclamp.io           import get_voltage_intervals, apply_manual_exclude
from patchclamp.preprocessing import preprocess
from patchclamp.detection    import detect_single_channel_current
from patchclamp.occupancy    import (quantize_open_channels,
                                     compute_occupancy,
                                     build_occupancy_table)
from patchclamp.iv           import build_iv, fit_linear_iv
from patchclamp.plotting     import (plot_iv, plot_g_vs_v,
                                     plot_g_absV, plot_occupancy,
                                     plot_diagnostics)


# ─────────────────────────────────────────────
# Анализ одного интервала
# ─────────────────────────────────────────────

def analyze_interval(
    abf: pyabf.ABF,
    t_start: float,
    t_end: float,
    voltage_mV: int,
    cfg: ExperimentConfig,
    is_control: bool = False,
) -> dict | None:
    abf.setSweep(0)
    t = abf.sweepX
    y = abf.sweepY.copy()

    mask_t  = (t >= t_start) & (t <= t_end)
    current = y[mask_t]

    if len(current) < 500:
        print(f"      ⚠️  мало точек: {len(current)}")
        return None

    fs = float(abf.dataRate)

    # предобработка
    filt, noise_mask = preprocess(current, fs, cfg)
    current_clean    = filt[noise_mask]

    if len(current_clean) < 200:
        print(f"      ⚠️  мало точек после маски: {len(current_clean)}")
        return None

    # ── контроль ──────────────────────────────
    if is_control:
        I_cl = float(np.median(current_clean))
        mad  = 1.4826 * float(np.median(
            np.abs(current_clean - np.median(current_clean))
        ))
        print(f"      CONTROL {voltage_mV:+5d} mV  "
              f"[{t_start:.0f}–{t_end:.0f}s]  "
              f"I_cl={I_cl:.2f} pA  noise(MAD)={mad:.2f} pA")
        return {
            "voltage_mV":  voltage_mV,
            "t_start":     t_start,
            "t_end":       t_end,
            "duration_sec": t_end - t_start,
            "I_closed":    I_cl,
            "I_open":      np.nan,
            "delta_I":     0.0,
            "sem_delta":   mad,
            "n_peaks":     1,
            "n_points":    len(current_clean),
            "Nmax": None, "Nmean": None, "Popen": None,
            "open_events": None, "events_per_s": None,
            "hist":    (None, None),
            "current": current_clean,
        }

    # ── детекция ΔI ───────────────────────────
    res = detect_single_channel_current(current_clean, voltage_mV, fs, cfg)

    # ── occupancy ─────────────────────────────
    delta_I_val = res["delta_I"]
    if delta_I_val is None or not np.isfinite(delta_I_val):
        Nmax = Nmean = Popen = open_events = events_per_s = None
    else:
        n_open = quantize_open_channels(
            current_clean, res["I_closed"], delta_I_val, cfg
        )
        occ        = compute_occupancy(n_open, t_end - t_start)
        Nmax       = occ["Nmax"]
        Nmean      = occ["Nmean"]
        Popen      = occ["Popen"]
        open_events  = occ["open_events"]
        events_per_s = occ["events_per_s"]
        print(f"      {voltage_mV:+5d} mV  "
              f"[{t_start:.0f}–{t_end:.0f}s]  "
              f"ΔI={delta_I_val:.2f} pA  "
              f"Popen={Popen:.3f}  Nmax={Nmax}  "
              f"method={res['method']}")

    return {
        "voltage_mV":   voltage_mV,
        "t_start":      t_start,
        "t_end":        t_end,
        "duration_sec": t_end - t_start,
        "I_closed":     res["I_closed"],
        "I_open":       res["I_open"],
        "delta_I":      res["delta_I"],
        "sem_delta":    res["sem_delta"],
        "n_peaks":      res["n_peaks"],
        "n_points":     len(current_clean),
        "Nmax":         Nmax,
        "Nmean":        Nmean,
        "Popen":        Popen,
        "open_events":  open_events,
        "events_per_s": events_per_s,
        "hist":         res["hist"],
        "current":      current_clean,
    }


# ─────────────────────────────────────────────
# Анализ одного файла
# ─────────────────────────────────────────────

def analyze_file(
    path: str,
    cfg: ExperimentConfig,
    results_dir: Path,
    is_control: bool = False,
) -> pd.DataFrame:
    name = Path(path).name
    print(f"\n{'='*60}")
    print(f"{'КОНТРОЛЬ' if is_control else 'ЭКСПЕРИМЕНТ'}: {name}")

    intervals = get_voltage_intervals(path, cfg)
    if intervals.empty:
        return pd.DataFrame()

    intervals = apply_manual_exclude(intervals, name, cfg)
    if intervals.empty:
        print("  Все интервалы исключены")
        return pd.DataFrame()

    abf      = pyabf.ABF(path)
    results  = []
    hist_data = []

    for _, row in intervals.iterrows():
        r = analyze_interval(
            abf,
            float(row["t_start"]),
            float(row["t_end"]),
            int(row["voltage_mV"]),
            cfg,
            is_control=is_control,
        )
        if r is None:
            continue

        r["file"]       = name
        r["is_control"] = is_control
        hist_data.append(r)
        results.append({k: v for k, v in r.items()
                        if k not in ("hist", "current")})

    if hist_data and not is_control:
        plot_diagnostics(hist_data, name, cfg, results_dir)

    return pd.DataFrame(results)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single-channel patch-clamp analysis pipeline"
    )
    parser.add_argument(
        "--config", required=True,
        help="Путь к .toml конфигу (например configs/gramicidin_a.toml)"
    )
    parser.add_argument(
        "--results", default="results",
        help="Папка для результатов (default: results/)"
    )
    args = parser.parse_args()

    cfg         = ExperimentConfig.from_toml(args.config)
    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"ПАЙПЛАЙН: {cfg.name}")
    print("=" * 60)
    print(cfg.summary())

    # ── контроль ──────────────────────────────
    ctrl_df = pd.DataFrame()
    if cfg.control_file and Path(cfg.control_file).exists():
        ctrl_df = analyze_file(cfg.control_file, cfg, results_dir,
                               is_control=True)
    else:
        print(f"⚠️  Контроль не найден: {cfg.control_file}")

    # ── эксперимент ───────────────────────────
    exp_frames = []
    for f in cfg.exp_files:
        if not Path(f).exists():
            print(f"⚠️  Не найден: {f}")
            continue
        df = analyze_file(f, cfg, results_dir, is_control=False)
        if not df.empty:
            exp_frames.append(df)

    if not exp_frames:
        print("❌ Нет экспериментальных данных")
        sys.exit(1)

    exp_df = pd.concat(exp_frames, ignore_index=True)

    # ── сохранение сырых результатов ──────────
    raw_path = results_dir / "iv_raw_intervals.csv"
    exp_df.to_csv(raw_path, index=False)
    print(f"\n→ {raw_path}")

    # ── occupancy ─────────────────────────────
    occ_cols = {"voltage_mV","duration_sec","Nmax","Nmean",
                "Popen","open_events","events_per_s"}
    if occ_cols.issubset(exp_df.columns):
        df_occ = build_occupancy_table(exp_df)
        occ_path = results_dir / "occupancy_by_voltage.csv"
        df_occ.to_csv(occ_path, index=False)
        print(f"→ {occ_path}")
        print(df_occ.to_string(index=False))
        plot_occupancy(df_occ, cfg, results_dir)

    # ── ВАХ ───────────────────────────────────
    summary = build_iv(exp_df, ctrl_df, cfg)
    summary, G_fit, R2 = fit_linear_iv(summary, cfg)

    print("\n" + "=" * 60)
    print("ИТОГОВАЯ ТАБЛИЦА:")
    cols = ["voltage_mV","corrected_dI","corrected_sem",
            "conductance_pS","n_intervals","total_sec","qc_short"]
    print(summary[cols].to_string(index=False))

    summary_path = results_dir / "iv_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\n→ {summary_path}")

    # ── графики ───────────────────────────────
    plot_iv(summary, cfg, G_fit_pS=G_fit, results_dir=results_dir)
    plot_g_vs_v(summary, cfg, G_fit_pS=G_fit, results_dir=results_dir)
    plot_g_absV(summary, cfg, G_fit_pS=G_fit, results_dir=results_dir)

    # ── финальная статистика ──────────────────
    g_vals = summary["conductance_pS"].dropna()
    print("\n" + "=" * 60)
    print(f"G (linear fit) : {G_fit:.2f} pS  |  R² = {R2:.4f}")
    print(f"G (mean ± std) : {g_vals.mean():.1f} ± {g_vals.std():.1f} pS")
    print(f"Литература gA  : 20–28 pS (2M KCl, DOPhC)")
    print("=" * 60)


if __name__ == "__main__":
    main()
