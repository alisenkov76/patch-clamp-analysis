# patch-clamp-analysis

![tests](https://github.com/alisenkov76/patch-clamp-analysis/actions/workflows/tests.yml/badge.svg)


Universal single-channel patch-clamp analysis pipeline.

Designed for gap-free ABF recordings (Axon Instruments / pCLAMP).  
Config-driven: one `.toml` file per experiment, no code changes needed.

---

## Features

- **Automatic tag parsing** — extracts voltage intervals from Clampex tags
- **Manual exclusion** — mask noise / artifact regions per file
- **Signal preprocessing** — Butterworth lowpass + MAD-based noise mask
- **Single-channel detection** — histogram method + transition fallback
- **Channel occupancy** — Nmax, Nmean, Popen, open events/s
- **I-V curve + conductance** — weighted pooled SEM, linear fit through 0
- **QC flags** — short intervals excluded from fit, predicted points marked
- **Publication-quality plots** — I-V, g(V), g(|V|), occupancy, diagnostics

---

## Validated on

| Channel | Membrane | Solution | G measured | G literature |
|---|---|---|---|---|
| Gramicidin A (1MAG) | DOPC | 2M KCl + 10mM HEPES | **26.1 +/- 2.4 pS** | 20-28 pS |

---

## Installation

```bash
conda create -n patchclamp python=3.11
conda activate patchclamp
git clone git@github.com:alisenkov76/patch-clamp-analysis.git
cd patch-clamp-analysis
pip install -e .
```

---

## Usage

```bash
python scripts/run_analysis.py --config configs/gramicidin_a.toml
```

With custom results directory:

```bash
python scripts/run_analysis.py \
    --config configs/gramicidin_a.toml \
    --results results/gramicidin_a
```

---

## Config format

```toml
[experiment]
name        = "Your Channel Name"
description = "Membrane, solution, conditions"
pdb_id      = "XXXX"

[files]
control     = "data/raw/control.abf"
experiments = ["data/raw/exp1.abf", "data/raw/exp2.abf"]

[recording]
sample_rate_hz     = 5000
target_voltages_mV = [-150, -100, -50, 50, 100, 150]
min_duration_sec   = 30.0
skip_zero_mV       = true

[preprocessing]
lowpass_hz      = 400.0
noise_window_ms = 50.0
noise_threshold = 4.0

[detection]
hist_bins       = 400
peak_min_height = 0.02
peak_min_dist   = 20
min_step_pA     = 0.3
max_channels    = 6

[qc]
min_total_sec_for_fit = 200.0

[manual_exclude]
"noisy_file.abf"  = [[978, 1628], [4556, 4747]]
"no_channels.abf" = [[0, 99999]]
```

---

## Output files

| File | Description |
|---|---|
| `iv_raw_intervals.csv` | Per-interval delta-I, occupancy metrics |
| `iv_summary.csv` | Per-voltage averaged I-V + conductance |
| `occupancy_by_voltage.csv` | Nmax, Popen, events/s by voltage |
| `iv_curve_final.png` | I-V curve with linear fit |
| `g_vs_v.png` | Conductance g(V) |
| `g_vs_absV.png` | Conductance g(|V|), +/-V averaged |
| `occupancy_vs_voltage.png` | Nmax and Popen vs voltage |
| `diag_*.png` | Per-file trace + histogram diagnostics |

---

## Project structure

```
patch-clamp-analysis/
├── patchclamp/
│   ├── config.py          # ExperimentConfig dataclass, .toml loader
│   ├── io.py              # ABF loader, tag parser, manual exclude
│   ├── preprocessing.py   # Butterworth filter, MAD noise mask
│   ├── detection.py       # Histogram + transitions delta-I detection
│   ├── occupancy.py       # Quantization, Popen, Nmax
│   ├── iv.py              # build_iv, fit_linear_iv, pooled_sem
│   └── plotting.py        # All publication-quality plots
├── scripts/
│   └── run_analysis.py    # CLI entry point
├── configs/
│   └── gramicidin_a.toml  # Gramicidin A experiment config
└── tests/
    └── test_detection.py  # Unit tests
```

---

## Method notes

**delta-I detection:**  
Primary: amplitude histogram → Gaussian smoothing → peak finding.  
Fallback: derivative-based step detection for low Popen recordings.

**Conductance estimate:**  
Linear fit `I = G*V` through origin, reliable points only  
(`|V| >= 100 mV`, `total_sec >= 200 s`).  
Points at `|V| < 100 mV` replaced by fit prediction (`is_predicted=True`).

**Leak correction:**  
Not applied to delta-I. Single-channel `delta-I = I_open - I_closed`  
subtracts baseline within each interval.  
Control recording shown on plots for reference only.

---

## Requirements

```
numpy >= 1.26
scipy >= 1.12
pandas >= 2.2
matplotlib >= 3.8
pyabf >= 2.3
```

---

## Author

Developed as part of a computational biology / CADD portfolio.  
Lisenkov ALexey alisenkov2005@gmail.com/lisenkov_as@edu.spbau.ru Alferov Federal State Budgetary Institution of Higher Education and Science Saint Petersburg National Research Academic University of the Russian Academy of Sciences, Saint Petersburg.
