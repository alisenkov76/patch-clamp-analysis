"""
Загрузка параметров эксперимента из .toml файла.
"""
from __future__ import annotations
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple


@dataclass
class ExperimentConfig:
    # --- meta ---
    name: str
    description: str = ""
    pdb_id: str = ""

    # --- files ---
    control_file: str = ""
    exp_files: List[str] = field(default_factory=list)

    # --- recording ---
    sample_rate_hz: int = 5000
    target_voltages: Set[int] = field(default_factory=lambda: {-150,-100,-50,50,100,150})
    min_duration_sec: float = 30.0
    skip_zero: bool = True

    # --- preprocessing ---
    lowpass_hz: float = 400.0
    noise_window_ms: float = 50.0
    noise_threshold: float = 4.0

    # --- detection ---
    hist_bins: int = 400
    peak_min_height: float = 0.02
    peak_min_dist: int = 20
    min_step_pA: float = 0.3
    max_channels: int = 6

    # --- qc ---
    min_total_sec_for_fit: float = 200.0

    # --- manual exclude: {filename: [(t_start, t_end), ...]} ---
    manual_exclude: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)

    @classmethod
    def from_toml(cls, path: str | Path) -> "ExperimentConfig":
        with open(path, "rb") as f:
            d = tomllib.load(f)

        excl_raw = d.get("manual_exclude", {})
        excl: Dict[str, List[Tuple[float, float]]] = {
            fname: [tuple(iv) for iv in intervals]   # type: ignore[misc]
            for fname, intervals in excl_raw.items()
        }

        rec  = d.get("recording", {})
        pre  = d.get("preprocessing", {})
        det  = d.get("detection", {})
        qc   = d.get("qc", {})
        exp  = d.get("experiment", {})
        fils = d.get("files", {})

        return cls(
            name=exp.get("name", "Unknown"),
            description=exp.get("description", ""),
            pdb_id=exp.get("pdb_id", ""),
            control_file=fils.get("control", ""),
            exp_files=fils.get("experiments", []),
            sample_rate_hz=rec.get("sample_rate_hz", 5000),
            target_voltages=set(rec.get("target_voltages_mV",
                                        [-150,-100,-50,50,100,150])),
            min_duration_sec=rec.get("min_duration_sec", 30.0),
            skip_zero=rec.get("skip_zero_mV", True),
            lowpass_hz=pre.get("lowpass_hz", 400.0),
            noise_window_ms=pre.get("noise_window_ms", 50.0),
            noise_threshold=pre.get("noise_threshold", 4.0),
            hist_bins=det.get("hist_bins", 400),
            peak_min_height=det.get("peak_min_height", 0.02),
            peak_min_dist=det.get("peak_min_dist", 20),
            min_step_pA=det.get("min_step_pA", 0.3),
            max_channels=det.get("max_channels", 6),
            min_total_sec_for_fit=qc.get("min_total_sec_for_fit", 200.0),
            manual_exclude=excl,
        )

    def summary(self) -> str:
        lines = [
            f"Experiment : {self.name}",
            f"PDB        : {self.pdb_id}",
            f"Control    : {self.control_file}",
            f"Exp files  : {len(self.exp_files)}",
            f"Voltages   : {sorted(self.target_voltages)} mV",
            f"Lowpass    : {self.lowpass_hz} Hz",
            f"Min dur    : {self.min_duration_sec} s",
            f"Manual exc : {len(self.manual_exclude)} files",
        ]
        return "\n".join(lines)
