"""ArduPilot Rover steering + EK3 review.

Dispatched from `analysis.generate_review` when the reviewed log is an ArduPilot
dataflash BIN. It reuses the shared plotting/formatting helpers from `analysis`
and focuses on the steering-rate controller and EKF3 health:

- **Steering control gains** — pulled from the log's ``PARM`` records
  (`ATC_STR_RAT_FF` / `_P` / `_I` / `_D` ...). ArduPilot has no bare
  ``ATC_STR_P``/``ATC_STR_I``; those map to the rate-controller params and we
  fall back to the literal names just in case a custom build logs them.
- **Turn-rate tracking & overshoot** — from the ``STER`` message
  (``DesTurnRate`` vs ``TurnRate``). `turn_rate_overshoot` quantifies how far the
  achieved turn rate exceeds the commanded rate.
- **Steering-rate PID terms** — from ``PIDS`` (target/actual + P/I/D/FF).
- **EK3 health** — innovations (``XKF3``) and normalized test ratios (``XKF4``).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .analysis import (
    arr,
    bool_spans,
    format_spans,
    latex_escape,
    save_fig,
    setup_axis,
    t_rel,
)
from .config import ReviewConfig
from .logsource import LogSource

# Dataflash message types the Rover review consumes; used to bound parsing of
# large logs (PARM is always retained by the source for parameter access).
MESSAGE_TYPES = ("STER", "PIDS", "XKF3", "XKF4")

# Steering-rate controller gains, by role -> candidate PARM names (first wins).
_GAIN_PARAMS: dict[str, tuple[str, ...]] = {
    "ff": ("ATC_STR_RAT_FF",),
    "p": ("ATC_STR_RAT_P", "ATC_STR_P"),
    "i": ("ATC_STR_RAT_I", "ATC_STR_I"),
    "d": ("ATC_STR_RAT_D",),
    "imax": ("ATC_STR_RAT_IMAX",),
    "fltt": ("ATC_STR_RAT_FLTT",),
}

# Below this commanded turn rate (deg/s) we treat the command as ~zero and skip
# overshoot accounting, so sensor noise around a null setpoint is not counted.
COMMAND_THRESHOLD_DPS = 3.0
# Achieved rate must exceed the command by this fraction to count as overshoot.
OVERSHOOT_THRESHOLD = 0.10


def steering_gains(params: dict[str, float]) -> dict[str, float | None]:
    """Resolve the steering-rate controller gains from onboard parameters."""
    gains: dict[str, float | None] = {}
    for role, candidates in _GAIN_PARAMS.items():
        gains[role] = next((params[name] for name in candidates if name in params), None)
    return gains


def _mask_max(ds, key: str) -> int:
    """Max value of an integer bitmask field, or 0 if the field is absent."""
    if key not in ds.data:
        return 0
    values = arr(ds, key)
    finite = values[np.isfinite(values)]
    return int(np.max(finite)) if finite.size else 0


def assess_nav_health(
    ratio_max: dict[str, float],
    ratio_frac: dict[str, float],
    fault_max: int,
    timeout_max: int,
    cores: list[int],
) -> tuple[str, str]:
    """Classify EKF3 navigation health from XKF4 status.

    Returns ``(verdict, rationale)`` where verdict is HEALTHY / MARGINAL /
    DEGRADED. A filter fault (``FS``) is treated as degraded; aiding timeouts
    (``TS``), primary-core switches, or sustained/large test-ratio gate exceedances
    are marginal; brief minor exceedances stay healthy with a note.
    """
    reasons: list[str] = []
    degraded = False
    marginal = False

    if fault_max > 0:
        degraded = True
        reasons.append(f"EKF filter-fault mask reached {fault_max}")
    if timeout_max > 0:
        marginal = True
        reasons.append(f"aiding-timeout mask reached {timeout_max}")
    if len(cores) > 1:
        marginal = True
        reasons.append(f"primary core switched among {cores}")

    for label in ("vel", "pos", "hgt", "mag"):
        mx = ratio_max.get(label)
        if mx is None or mx <= 1.0:
            continue
        pct = ratio_frac.get(label, 0.0) * 100.0
        if mx >= 5.0 or pct >= 5.0:
            marginal = True
            reasons.append(f"{label} test ratio exceeded gate (max {mx:.1f}, {pct:.2f}% of samples)")
        else:
            reasons.append(f"{label} test ratio briefly over gate (max {mx:.1f}, {pct:.2f}%)")

    if degraded:
        return "DEGRADED", "; ".join(reasons)
    if marginal:
        return "MARGINAL", "; ".join(reasons)
    if reasons:
        return "HEALTHY", "only minor transient rejections: " + "; ".join(reasons)
    return "HEALTHY", "all innovation test ratios within gate; no faults or core switches"


def turn_rate_overshoot(
    t: np.ndarray,
    des: np.ndarray,
    act: np.ndarray,
    command_threshold: float = COMMAND_THRESHOLD_DPS,
    overshoot_threshold: float = OVERSHOOT_THRESHOLD,
) -> dict[str, object]:
    """Quantify how far the achieved turn rate overshoots the commanded rate.

    ``t``/``des``/``act`` are aligned samples (the ``STER`` message logs all three
    together, so no resampling is needed). Overshoot is only scored while a
    meaningful turn is commanded (``|des| > command_threshold``) and the achieved
    rate is in the same direction but larger in magnitude. Returns peak/typical
    overshoot, the fraction of commanded time spent overshooting, and the
    active-interval spans for the report.
    """
    t = np.asarray(t, dtype=float)
    des = np.asarray(des, dtype=float)
    act = np.asarray(act, dtype=float)

    finite = np.isfinite(t) & np.isfinite(des) & np.isfinite(act)
    empty = {
        "rms_err": float("nan"),
        "max_abs_err": float("nan"),
        "peak_overshoot_pct": 0.0,
        "frac_overshoot": 0.0,
        "n_events": 0,
        "spans": [],
    }
    if finite.sum() == 0:
        return empty
    t, des, act = t[finite], des[finite], act[finite]

    cmd_mask = np.abs(des) > command_threshold
    if not cmd_mask.any():
        return empty

    err = act - des
    same_dir = np.sign(act) == np.sign(des)
    excess = np.abs(act) - np.abs(des)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(cmd_mask & same_dir & (excess > 0), excess / np.abs(des), 0.0)
    over_mask = (ratio > overshoot_threshold) & cmd_mask
    spans = bool_spans(t, over_mask)

    cmd_err = err[cmd_mask]
    return {
        "rms_err": float(np.sqrt(np.mean(cmd_err**2))),
        "max_abs_err": float(np.max(np.abs(cmd_err))),
        "peak_overshoot_pct": float(np.max(ratio) * 100.0),
        "frac_overshoot": float(np.mean(over_mask[cmd_mask])),
        "n_events": len(spans),
        "spans": spans,
    }


def plot_steering_tracking(source: LogSource, base_timestamp: int, fig_dir: Path) -> Path:
    ster = source.get_dataset("STER")
    fig, axes = plt.subplots(4, 1, figsize=(10.5, 8.4), sharex=True)

    if ster:
        t = t_rel(ster, base_timestamp)
        if "DesTurnRate" in ster.data and "TurnRate" in ster.data:
            des = arr(ster, "DesTurnRate")
            act = arr(ster, "TurnRate")
            axes[0].plot(t, des, label="desired turn rate", lw=1.1)
            axes[0].plot(t, act, label="achieved turn rate", lw=1.0)
            axes[1].plot(t, act - des, label="turn rate error", lw=0.9, color="tab:red")
            axes[1].axhline(0.0, color="black", lw=0.6)
            ov = turn_rate_overshoot(t, des, act)
            for start, end in ov["spans"]:
                axes[0].axvspan(start, end, color="tab:orange", alpha=0.15)
        for key, label in [("DesLatAcc", "desired lat accel"), ("LatAcc", "lat accel")]:
            if key in ster.data:
                axes[2].plot(t, arr(ster, key), label=label, lw=0.9)
        for key, label in [("SteerIn", "steering in"), ("SteerOut", "steering out")]:
            if key in ster.data:
                axes[3].plot(t, arr(ster, key), label=label, lw=0.9)

    setup_axis(axes[0], "Turn rate: desired vs achieved (overshoot shaded)", "deg/s")
    setup_axis(axes[1], "Turn rate error (achieved - desired)", "deg/s")
    setup_axis(axes[2], "Lateral acceleration", "m/s^2")
    setup_axis(axes[3], "Steering command")
    axes[3].set_xlabel("flight-log relative time [s]")
    for ax in axes:
        ax.legend(loc="best", fontsize=8)

    return save_fig(fig, fig_dir, "rover_steering_tracking")


def plot_steering_pid(source: LogSource, base_timestamp: int, fig_dir: Path) -> Path:
    pids = source.get_dataset("PIDS")
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 7.2), sharex=True)

    if pids:
        t = t_rel(pids, base_timestamp)
        for key, label in [("Tar", "target"), ("Act", "actual")]:
            if key in pids.data:
                axes[0].plot(t, arr(pids, key), label=label, lw=1.0)
        for key in ["P", "I", "D", "FF"]:
            if key in pids.data:
                axes[1].plot(t, arr(pids, key), label=key, lw=0.9)
        if "Err" in pids.data:
            axes[2].plot(t, arr(pids, "Err"), label="error", lw=0.9, color="tab:red")
            axes[2].axhline(0.0, color="black", lw=0.6)

    setup_axis(axes[0], "Steering-rate PID: target vs actual")
    setup_axis(axes[1], "PID term contributions")
    setup_axis(axes[2], "PID error")
    axes[2].set_xlabel("flight-log relative time [s]")
    for ax in axes:
        ax.legend(loc="best", fontsize=8)

    return save_fig(fig, fig_dir, "rover_steering_pid")


def plot_ek3_innovations(source: LogSource, base_timestamp: int, fig_dir: Path) -> Path:
    xkf3 = source.get_dataset("XKF3")
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 7.2), sharex=True)

    if xkf3:
        t = t_rel(xkf3, base_timestamp)
        groups = [
            (axes[0], ["IVN", "IVE", "IVD"], "Velocity innovations", "m/s"),
            (axes[1], ["IPN", "IPE", "IPD"], "Position innovations", "m"),
            (axes[2], ["IMX", "IMY", "IMZ", "IYAW"], "Magnetometer / yaw innovations", None),
        ]
        for ax, keys, title, ylabel in groups:
            for key in keys:
                if key in xkf3.data:
                    ax.plot(t, arr(xkf3, key), lw=0.9, label=key)
            setup_axis(ax, title, ylabel)

    axes[-1].set_xlabel("flight-log relative time [s]")
    for ax in axes:
        ax.legend(loc="best", fontsize=7, ncols=2)

    return save_fig(fig, fig_dir, "rover_ek3_innovations")


def plot_ek3_test_ratios(source: LogSource, base_timestamp: int, fig_dir: Path) -> Path:
    xkf4 = source.get_dataset("XKF4")
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.0), sharex=True)

    if xkf4:
        t = t_rel(xkf4, base_timestamp)
        for key, label in [
            ("SV", "velocity"),
            ("SP", "position"),
            ("SH", "height"),
            ("SM", "magnetometer"),
        ]:
            if key in xkf4.data:
                axes[0].plot(t, arr(xkf4, key), lw=0.9, label=label)
        axes[0].axhline(1.0, color="black", lw=0.9, ls="--", label="gate")
        # Test ratios are non-negative and occasionally spike far past the gate;
        # symlog keeps the 0..1 region readable while still showing the outliers.
        axes[0].set_yscale("symlog", linthresh=1.0)
        for key in ["FS", "TS", "SS", "GPS", "PI"]:
            if key in xkf4.data:
                axes[1].step(t, arr(xkf4, key), where="post", lw=0.8, label=key)

    setup_axis(axes[0], "EK3 normalized innovation test ratios (1.0 = gate)", "test ratio")
    setup_axis(axes[1], "EK3 fault/timeout/solution-status masks")
    axes[1].set_xlabel("flight-log relative time [s]")
    for ax in axes:
        ax.legend(loc="best", fontsize=7, ncols=2)

    return save_fig(fig, fig_dir, "rover_ek3_test_ratios")


def _fmt(value: float | None, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:.4g}{suffix}"


def rover_metrics(source: LogSource, base_timestamp: int) -> dict[str, str]:
    """Steering/EK3 metrics for `summary.txt` and the report tables."""
    out: dict[str, str] = {}

    gains = steering_gains(source.params)
    out["str_rat_ff"] = _fmt(gains["ff"])
    out["str_rat_p"] = _fmt(gains["p"])
    out["str_rat_i"] = _fmt(gains["i"])
    out["str_rat_d"] = _fmt(gains["d"])
    out["str_rat_imax"] = _fmt(gains["imax"])

    ster = source.get_dataset("STER")
    if ster and "DesTurnRate" in ster.data and "TurnRate" in ster.data:
        t = t_rel(ster, base_timestamp)
        ov = turn_rate_overshoot(t, arr(ster, "DesTurnRate"), arr(ster, "TurnRate"))
        out["turn_rate_rms_err"] = _fmt(ov["rms_err"], " deg/s")
        out["turn_rate_max_abs_err"] = _fmt(ov["max_abs_err"], " deg/s")
        out["turn_rate_peak_overshoot"] = _fmt(ov["peak_overshoot_pct"], "%")
        out["turn_rate_overshoot_time"] = _fmt(ov["frac_overshoot"] * 100.0, "%")
        out["turn_rate_overshoot_events"] = f"{ov['n_events']}; {format_spans(ov['spans'], limit=3)}"

    xkf4 = source.get_dataset("XKF4")
    if xkf4:
        ratio_max: dict[str, float] = {}
        ratio_frac: dict[str, float] = {}
        for key, label in [("SV", "vel"), ("SP", "pos"), ("SH", "hgt"), ("SM", "mag")]:
            if key in xkf4.data:
                values = arr(xkf4, key)
                finite = values[np.isfinite(values)]
                if finite.size:
                    ratio_max[label] = float(np.max(finite))
                    ratio_frac[label] = float((finite > 1.0).mean())
                    out[f"ek3_test_ratio_{label}_max"] = f"{ratio_max[label]:.2f}"
                    out[f"ek3_test_ratio_{label}_gt1"] = f"{int((finite > 1.0).sum())}/{finite.size}"

        fault_max = _mask_max(xkf4, "FS")
        timeout_max = _mask_max(xkf4, "TS")
        cores = sorted({int(c) for c in arr(xkf4, "PI")}) if "PI" in xkf4.data else []
        out["ek3_filter_fault_max"] = str(fault_max)
        out["ek3_aiding_timeout_max"] = str(timeout_max)
        if cores:
            out["ek3_primary_cores"] = ",".join(map(str, cores))

        verdict, note = assess_nav_health(ratio_max, ratio_frac, fault_max, timeout_max, cores)
        out["nav_health"] = verdict
        out["nav_health_note"] = note

    return out


# Metrics tabulated in the report, as (label, summary key) in display order.
_METRIC_ROWS: tuple[tuple[str, str], ...] = (
    (r"\texttt{ATC\_STR\_RAT\_FF}", "str_rat_ff"),
    (r"\texttt{ATC\_STR\_RAT\_P}", "str_rat_p"),
    (r"\texttt{ATC\_STR\_RAT\_I}", "str_rat_i"),
    (r"\texttt{ATC\_STR\_RAT\_D}", "str_rat_d"),
    (r"\texttt{ATC\_STR\_RAT\_IMAX}", "str_rat_imax"),
    ("Turn-rate RMS error (while commanded)", "turn_rate_rms_err"),
    ("Turn-rate max absolute error", "turn_rate_max_abs_err"),
    ("Peak turn-rate overshoot", "turn_rate_peak_overshoot"),
    ("Time overshooting (of commanded time)", "turn_rate_overshoot_time"),
    ("Overshoot events", "turn_rate_overshoot_events"),
    ("EK3 velocity test ratio (max)", "ek3_test_ratio_vel_max"),
    ("EK3 position test ratio (max)", "ek3_test_ratio_pos_max"),
    ("EK3 height test ratio (max)", "ek3_test_ratio_hgt_max"),
    ("EK3 magnetometer test ratio (max)", "ek3_test_ratio_mag_max"),
    ("EK3 primary cores seen", "ek3_primary_cores"),
)


def build_latex_rover(
    config: ReviewConfig,
    figures: dict[str, Path],
    summary: dict[str, str],
) -> str:
    log_title = latex_escape(config.log.title)

    rows = [
        f"{label} & {latex_escape(summary[key])}" + r" \\"
        for label, key in _METRIC_ROWS
        if key in summary
    ]

    rel_figures = {k: p.relative_to(config.output_dir).as_posix() for k, p in figures.items()}

    return rf"""\documentclass[10pt]{{article}}
\usepackage[margin=0.7in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{float}}
\usepackage{{hyperref}}
\usepackage{{siunitx}}
\usepackage{{caption}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{0.5em}}

\title{{{latex_escape(config.title)}}}
\author{{Generated from ArduPilot dataflash (.BIN) artifacts}}
\date{{}}

\begin{{document}}
\maketitle

\section{{Scope}}
This report reviews ArduPilot Rover steering performance and EKF3 health from a
dataflash log, with emphasis on the steering-rate controller gains
(\texttt{{ATC\_STR\_RAT\_FF}}, \texttt{{ATC\_STR\_RAT\_P}}, \texttt{{ATC\_STR\_RAT\_I}}),
desired-versus-achieved turn rate, and turn-rate overshoot.  The log under review is
\texttt{{{log_title}}}.

\section{{Steering Control Gains and Tracking}}
Steering-rate controller gains: FF
{latex_escape(summary.get("str_rat_ff", "n/a"))}, P
{latex_escape(summary.get("str_rat_p", "n/a"))}, I
{latex_escape(summary.get("str_rat_i", "n/a"))}, D
{latex_escape(summary.get("str_rat_d", "n/a"))}, IMAX
{latex_escape(summary.get("str_rat_imax", "n/a"))}.  Turn-rate tracking RMS error is
{latex_escape(summary.get("turn_rate_rms_err", "n/a"))} with peak overshoot
{latex_escape(summary.get("turn_rate_peak_overshoot", "n/a"))} over
{latex_escape(summary.get("turn_rate_overshoot_events", "n/a"))}.

\section{{Key Metrics}}
\begin{{table}}[H]
\centering
\scriptsize
\begin{{tabular}}{{ll}}
\toprule
Metric & Value \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\caption{{Steering-rate gains, turn-rate tracking, and EKF3 test-ratio peaks for
\texttt{{{log_title}}}.  Metrics whose source message is absent from the log are
omitted.}}
\end{{table}}

\section{{Turn-Rate Tracking and Overshoot}}
\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{{rel_figures["steering_tracking"]}}}
\caption{{Desired vs achieved turn rate (overshoot intervals shaded), turn-rate
error, lateral acceleration, and steering command.}}
\end{{figure}}

\section{{Steering-Rate PID}}
\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{{rel_figures["steering_pid"]}}}
\caption{{Steering-rate PID target vs actual and the P/I/D/FF term contributions
driven by the configured gains.}}
\end{{figure}}

\section{{EKF3 Innovations}}
\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{{rel_figures["ek3_innovations"]}}}
\caption{{EKF3 velocity, position, and magnetometer/yaw innovations (XKF3).}}
\end{{figure}}

\section{{EKF3 Test Ratios}}
\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{{rel_figures["ek3_test_ratios"]}}}
\caption{{EKF3 normalized innovation test ratios (XKF4, symlog axis); 1.0 is the
rejection gate.}}
\end{{figure}}

\section{{State Estimation Health}}
Overall navigation assessment:
\textbf{{{latex_escape(summary.get("nav_health", "n/a"))}}}.
{latex_escape(summary.get("nav_health_note", "no EKF3 status available"))}.

This verdict is read from the EKF3 status (XKF4): a non-zero filter-fault mask is
treated as degraded; aiding timeouts, primary-core switches, or large/sustained
innovation test-ratio gate exceedances are marginal.  EKF filter-fault mask peaked
at {latex_escape(summary.get("ek3_filter_fault_max", "n/a"))} and the aiding-timeout
mask at {latex_escape(summary.get("ek3_aiding_timeout_max", "n/a"))}; the per-source
test-ratio behaviour is plotted above (EKF3 Test Ratios) and the corresponding raw
innovations in the EKF3 Innovations figure.

\end{{document}}
"""


def build_review(
    config: ReviewConfig,
    source: LogSource,
    base_timestamp: int,
    fig_dir: Path,
) -> tuple[dict[str, Path], dict[str, str], str]:
    """Build the Rover figures, metrics, and LaTeX body for one dataflash log."""
    figures = {
        "steering_tracking": plot_steering_tracking(source, base_timestamp, fig_dir),
        "steering_pid": plot_steering_pid(source, base_timestamp, fig_dir),
        "ek3_innovations": plot_ek3_innovations(source, base_timestamp, fig_dir),
        "ek3_test_ratios": plot_ek3_test_ratios(source, base_timestamp, fig_dir),
    }
    summary = rover_metrics(source, base_timestamp)
    return figures, summary, build_latex_rover(config, figures, summary)
