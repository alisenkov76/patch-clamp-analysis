"""
Визуализация результатов patch-clamp анализа.

Функции:
    plot_iv          — I–V кривая + линейный фит
    plot_g_vs_v      — проводимость g(V)
    plot_g_absV      — проводимость g(|V|), усреднение ±V
    plot_occupancy   — Nmax(V) и Popen(V)
    plot_diagnostics — трассы + гистограммы для каждого интервала
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from patchclamp.config import ExperimentConfig


# ─────────────────────────────────────────────
# Вспомогательные
# ─────────────────────────────────────────────

def _out(name: str, results_dir: str | Path) -> Path:
    p = Path(results_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p / name


def _title(cfg: ExperimentConfig) -> str:
    return f"{cfg.name}  |  {cfg.description}"


# ─────────────────────────────────────────────
# I–V кривая
# ─────────────────────────────────────────────

def plot_iv(
    summary: pd.DataFrame,
    cfg: ExperimentConfig,
    G_fit_pS: Optional[float] = None,
    results_dir: str | Path = "results",
) -> None:
    """
    I–V кривая: ΔI(V) + линейный фит.
    Measured и predicted точки разделены визуально.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    is_pred = summary.get("is_predicted", pd.Series(False, index=summary.index))
    qc_s    = summary.get("qc_short",    pd.Series(False, index=summary.index))

    # measured
    m = ~is_pred & ~qc_s
    ax.errorbar(
        summary.loc[m, "voltage_mV"],
        summary.loc[m, "corrected_dI"],
        yerr=summary.loc[m, "corrected_sem"],
        fmt="o-", color="steelblue",
        capsize=5, capthick=2, markersize=9, linewidth=2,
        label="ΔI measured",
        zorder=3,
    )

    # predicted (полые маркеры)
    p = is_pred
    if p.any():
        ax.errorbar(
            summary.loc[p, "voltage_mV"],
            summary.loc[p, "corrected_dI"],
            yerr=summary.loc[p, "corrected_sem"],
            fmt="o", mfc="white", mec="steelblue", color="steelblue",
            capsize=5, markersize=9, linewidth=1.5,
            label="ΔI predicted (|V|<100)",
            zorder=3,
        )

    # qc_short (крестики)
    if qc_s.any():
        ax.scatter(
            summary.loc[qc_s, "voltage_mV"],
            summary.loc[qc_s, "corrected_dI"],
            marker="x", s=120, color="black", zorder=4,
            label="QC: short interval",
        )

    # линейный фит
    if G_fit_pS is not None and np.isfinite(G_fit_pS):
        v_arr  = summary["voltage_mV"].values
        v_line = np.linspace(v_arr.min(), v_arr.max(), 200)
        ax.plot(
            v_line, G_fit_pS / 1000.0 * v_line,
            color="red", linewidth=1.8, linestyle="--",
            label=f"Linear fit: G = {G_fit_pS:.1f} pS",
        )

    # контроль (baseline)
    if "control_I" in summary.columns and summary["control_I"].notna().any():
        ax.plot(
            summary["voltage_mV"], summary["control_I"],
            "s--", color="gray", alpha=0.7, markersize=7,
            label="Control (baseline I)",
        )

    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Voltage (mV)", fontsize=12)
    ax.set_ylabel("ΔI single channel (pA)", fontsize=12)
    ax.set_title(f"I–V curve\n{_title(cfg)}", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    out = _out("iv_curve_final.png", results_dir)
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"  → {out}")


# ─────────────────────────────────────────────
# g(V)
# ─────────────────────────────────────────────

def plot_g_vs_v(
    summary: pd.DataFrame,
    cfg: ExperimentConfig,
    G_fit_pS: Optional[float] = None,
    results_dir: str | Path = "results",
) -> None:
    """
    Проводимость g(V) = 1000·|ΔI/V|.
    Measured / predicted / qc_short разделены визуально.
    """
    s = summary[summary["voltage_mV"] != 0].copy()
    v = s["voltage_mV"].astype(float)

    # измеренный g (до подмены)
    dI_meas = s.get("corrected_dI_meas", s["corrected_dI"]).astype(float)
    g_meas  = (1000.0 * dI_meas / v).abs()

    # финальный g
    g_used     = (1000.0 * s["corrected_dI"].astype(float) / v).abs()
    g_used_sem = (1000.0 * s["corrected_sem"].astype(float) / v.abs())

    is_pred = s.get("is_predicted", pd.Series(False, index=s.index)).fillna(False)
    qc_s    = s.get("qc_short",    pd.Series(False, index=s.index)).fillna(False)

    fig, ax = plt.subplots(figsize=(8, 6))

    # raw measured (серые точки, справочно)
    ax.scatter(v, g_meas, s=50, color="gray", alpha=0.5,
               label="g_raw (before fit substitution)", zorder=2)

    # measured финальный
    m = ~is_pred
    ax.errorbar(
        v[m], g_used[m], yerr=g_used_sem[m],
        fmt="o", color="darkorange",
        capsize=5, markersize=9, linewidth=2,
        label="g (measured)", zorder=3,
    )

    # predicted
    if is_pred.any():
        ax.errorbar(
            v[is_pred], g_used[is_pred], yerr=g_used_sem[is_pred],
            fmt="o", mfc="white", mec="darkorange", color="darkorange",
            capsize=5, markersize=9, linewidth=1.5,
            label="g (predicted for |V|<100)", zorder=3,
        )

    # qc_short
    if qc_s.any():
        ax.scatter(v[qc_s], g_used[qc_s],
                   marker="x", s=120, color="black", zorder=4,
                   label="QC: short interval")

    if G_fit_pS is not None and np.isfinite(G_fit_pS):
        ax.axhline(G_fit_pS, color="red", linestyle="--", linewidth=2,
                   label=f"G (linear fit) = {G_fit_pS:.2f} pS")

    ax.set_xlabel("Voltage (mV)", fontsize=12)
    ax.set_ylabel("Conductance (pS)", fontsize=12)
    ax.set_title(f"g(V)\n{_title(cfg)}", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    out = _out("g_vs_v.png", results_dir)
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"  → {out}")


# ─────────────────────────────────────────────
# g(|V|)
# ─────────────────────────────────────────────

def plot_g_absV(
    summary: pd.DataFrame,
    cfg: ExperimentConfig,
    G_fit_pS: Optional[float] = None,
    results_dir: str | Path = "results",
) -> None:
    """
    Проводимость g(|V|): усредняем +V и -V.
    """
    tmp = summary[summary["voltage_mV"] != 0].copy()
    tmp["absV"] = tmp["voltage_mV"].abs().astype(int)
    tmp["g"]    = (1000.0 * tmp["corrected_dI"] / tmp["voltage_mV"]).abs()

    g_abs = (
        tmp.groupby("absV")["g"]
        .agg(g_mean="mean", g_std="std", n="count")
        .reset_index()
        .sort_values("absV")
    )
    g_abs["g_sem"] = g_abs["g_std"] / np.sqrt(g_abs["n"])
    g_abs.loc[g_abs["n"] == 1, "g_sem"] = 0.0

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(
        g_abs["absV"], g_abs["g_mean"], yerr=g_abs["g_sem"],
        fmt="o-", color="darkorange",
        capsize=5, capthick=2, markersize=9, linewidth=2,
        label="g(|V|)  (±V averaged)",
    )

    if G_fit_pS is not None and np.isfinite(G_fit_pS):
        ax.axhline(G_fit_pS, color="red", linestyle="--", linewidth=2,
                   label=f"G (linear fit) = {G_fit_pS:.2f} pS")

    ax.set_xlabel("|Voltage| (mV)", fontsize=12)
    ax.set_ylabel("Conductance (pS)", fontsize=12)
    ax.set_title(f"g(|V|)\n{_title(cfg)}", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    out = _out("g_vs_absV.png", results_dir)
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"  → {out}")


# ─────────────────────────────────────────────
# Occupancy
# ─────────────────────────────────────────────

def plot_occupancy(
    df_occ: pd.DataFrame,
    cfg: ExperimentConfig,
    results_dir: str | Path = "results",
) -> None:
    """
    Nmax(V) — bar chart и Popen(V) — errorbar.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Channel occupancy\n{_title(cfg)}", fontsize=11)

    v = df_occ["voltage_mV"].values

    # Nmax bar
    ax = axes[0]
    ax.bar(v, df_occ["Nmax_obs"].values,
           width=18, color="steelblue", alpha=0.75, edgecolor="navy")
    ax.set_xlabel("Voltage (mV)", fontsize=12)
    ax.set_ylabel("Nmax observed", fontsize=12)
    ax.set_title("Max simultaneous open channels")
    ax.set_xticks(v)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.2, axis="y")

    # Popen errorbar
    ax = axes[1]
    yerr = df_occ["Popen_sem"].fillna(0).values
    ax.errorbar(
        v, df_occ["Popen_wavg"].values, yerr=yerr,
        fmt="o-", color="crimson",
        capsize=5, linewidth=2, markersize=8,
        label="Popen ± SEM",
    )
    ax.set_xlabel("Voltage (mV)", fontsize=12)
    ax.set_ylabel("Popen (weighted mean)", fontsize=12)
    ax.set_title("Open probability vs Voltage")
    ax.set_xticks(v)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    out = _out("occupancy_vs_voltage.png", results_dir)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  → {out}")


# ─────────────────────────────────────────────
# Диагностика интервалов
# ─────────────────────────────────────────────

def plot_diagnostics(
    results: List[dict],
    file_name: str,
    cfg: ExperimentConfig,
    results_dir: str | Path = "results",
) -> None:
    """
    Трасса + гистограмма для каждого интервала файла.
    """
    n = len(results)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 2, figsize=(14, 3.5 * n), squeeze=False)

    for row_idx, r in enumerate(results):
        v       = r["voltage_mV"]
        current = r["current"]
        hist_s, centers = r["hist"]

        # трасса
        ax_t = axes[row_idx, 0]
        ax_t.plot(current, color="steelblue", linewidth=0.4, alpha=0.8)

        if np.isfinite(r.get("I_closed", np.nan)):
            ax_t.axhline(r["I_closed"], color="green", lw=1.5,
                         linestyle="--", label=f"closed={r['I_closed']:.1f} pA")
        if np.isfinite(r.get("I_open", np.nan)):
            ax_t.axhline(r["I_open"], color="red", lw=1.5,
                         linestyle="--", label=f"open={r['I_open']:.1f} pA")

        ax_t.set_title(f"{v:+d} mV  [{r['t_start']:.0f}–{r['t_end']:.0f} s]")
        ax_t.set_ylabel("pA")
        ax_t.legend(fontsize=8)
        ax_t.grid(True, alpha=0.25)

        # гистограмма
        ax_h = axes[row_idx, 1]
        if hist_s is not None and centers is not None:
            ax_h.bar(centers, hist_s,
                     width=(centers[1] - centers[0]),
                     color="steelblue", alpha=0.75, edgecolor="none")

        if np.isfinite(r.get("I_closed", np.nan)):
            ax_h.axvline(r["I_closed"], color="green", lw=2,
                         label=f"closed={r['I_closed']:.1f}")
        if np.isfinite(r.get("I_open", np.nan)):
            ax_h.axvline(r["I_open"], color="red", lw=2,
                         label=f"open={r['I_open']:.1f}")

        dI  = r.get("delta_I", np.nan)
        sem = r.get("sem_delta", np.nan)
        ax_h.set_title(
            f"ΔI = {dI:.2f} pA" if np.isfinite(dI) else "ΔI = N/A"
        )
        ax_h.set_xlabel("pA")
        ax_h.set_ylabel("counts")
        ax_h.legend(fontsize=8)
        ax_h.grid(True, alpha=0.25)

    plt.suptitle(f"Diagnostics: {file_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    out = _out(f"diag_{Path(file_name).stem}.png", results_dir)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  → {out}")
