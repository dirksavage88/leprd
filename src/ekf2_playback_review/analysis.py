from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from pyulog import ULog

from .config import LogSpec, ReviewConfig


@dataclass(frozen=True)
class ReviewArtifacts:
    output_dir: Path
    tex_path: Path
    summary_path: Path
    pdf_path: Path | None
    figures: dict[str, Path]


def get_ds(ulog: ULog, name: str, multi_id: int = 0):
    matches = [d for d in ulog.data_list if d.name == name and d.multi_id == multi_id]
    return matches[0] if matches else None


def t_rel(ds, base_timestamp: int) -> np.ndarray:
    return (np.asarray(ds.data["timestamp"], dtype=np.int64) - int(base_timestamp)) / 1e6


def arr(ds, field: str, dtype=float) -> np.ndarray:
    return np.asarray(ds.data[field], dtype=dtype)


def interp_at(src_t: np.ndarray, src_y: np.ndarray, dst_t: np.ndarray) -> np.ndarray:
    finite = np.isfinite(src_t) & np.isfinite(src_y)
    if finite.sum() < 2:
        return np.full_like(dst_t, np.nan, dtype=float)
    return np.interp(dst_t, src_t[finite], src_y[finite], left=np.nan, right=np.nan)


def bool_spans(t: np.ndarray, values: np.ndarray) -> list[tuple[float, float]]:
    vals = np.asarray(values).astype(bool)
    spans: list[tuple[float, float]] = []
    start = None

    for idx, val in enumerate(vals):
        if val and start is None:
            start = float(t[idx])
        if start is not None and not val:
            spans.append((start, float(t[idx - 1])))
            start = None

    if start is not None and len(t):
        spans.append((start, float(t[-1])))

    return spans


def format_spans(spans: list[tuple[float, float]], limit: int = 5) -> str:
    if not spans:
        return "none"

    parts = []
    for start, end in spans[:limit]:
        if abs(start - end) < 1e-3:
            parts.append(f"{start:.3f}s")
        else:
            parts.append(f"{start:.3f}--{end:.3f}s")

    if len(spans) > limit:
        parts.append(f"+{len(spans) - limit} more")

    return ", ".join(parts)


def save_fig(fig: plt.Figure, fig_dir: Path, name: str) -> Path:
    png = fig_dir / f"{name}.png"
    pdf = fig_dir / f"{name}.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png


def setup_axis(ax, title: str, ylabel: str | None = None):
    ax.set_title(title, loc="left", fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)


def plot_position_overview(ulog: ULog, base_timestamp: int, fig_dir: Path) -> Path:
    gps = get_ds(ulog, "vehicle_gps_position")
    air = get_ds(ulog, "vehicle_air_data")
    lpos = get_ds(ulog, "vehicle_local_position")
    gpos = get_ds(ulog, "vehicle_global_position")

    fig, axes = plt.subplots(4, 1, figsize=(10.5, 8.0), sharex=True)

    if gps:
        tg = t_rel(gps, base_timestamp)
        axes[0].plot(tg, arr(gps, "altitude_msl_m"), label="GPS MSL altitude", lw=1.2)
        axes[1].plot(tg, arr(gps, "vel_d_m_s"), label="GPS D velocity", lw=1.0)

    if air:
        ta = t_rel(air, base_timestamp)
        axes[0].plot(ta, arr(air, "baro_alt_meter"), label="baro altitude", lw=1.2)

    if gpos:
        tp = t_rel(gpos, base_timestamp)
        axes[0].plot(tp, arr(gpos, "alt"), label="EKF global alt", lw=1.1)

    if lpos:
        tl = t_rel(lpos, base_timestamp)
        axes[1].plot(tl, arr(lpos, "vz"), label="EKF vz", lw=1.0)
        axes[2].plot(tl, arr(lpos, "x"), label="local x", lw=0.9)
        axes[2].plot(tl, arr(lpos, "y"), label="local y", lw=0.9)
        axes[2].plot(tl, arr(lpos, "z"), label="local z", lw=0.9)
        axes[3].plot(tl, arr(lpos, "vx"), label="vx", lw=0.9)
        axes[3].plot(tl, arr(lpos, "vy"), label="vy", lw=0.9)
        axes[3].plot(tl, arr(lpos, "vz"), label="vz", lw=0.9)

    setup_axis(axes[0], "Altitude disagreement", "m")
    setup_axis(axes[1], "Vertical velocity", "m/s")
    setup_axis(axes[2], "Local position", "m")
    setup_axis(axes[3], "Local velocity", "m/s")
    axes[3].set_xlabel("flight-log relative time [s]")
    for ax in axes:
        ax.legend(loc="best", fontsize=8)

    return save_fig(fig, fig_dir, "latest_position_overview")


def plot_gps_quality(ulog: ULog, base_timestamp: int, fig_dir: Path) -> Path:
    gps = get_ds(ulog, "vehicle_gps_position")
    gps_status = get_ds(ulog, "estimator_gps_status")
    fig, axes = plt.subplots(5, 1, figsize=(10.5, 9.0), sharex=True)

    if gps:
        t = t_rel(gps, base_timestamp)
        for key in ["fix_type", "satellites_used"]:
            if key in gps.data:
                axes[0].plot(t, arr(gps, key), label=key)
        for key in ["hdop", "vdop"]:
            if key in gps.data:
                axes[1].plot(t, arr(gps, key), label=key.upper())
        for key in ["eph", "epv"]:
            if key in gps.data:
                axes[2].plot(t, arr(gps, key), label=key.upper())
        for key, label in [("s_variance_m_s", "speed accuracy"), ("vel_d_m_s", "GPS vel D")]:
            if key in gps.data:
                axes[3].plot(t, arr(gps, key), label=label)
        for key in ["jamming_state", "jamming_indicator", "spoofing_state"]:
            if key in gps.data:
                axes[4].plot(t, arr(gps, key), label=key)

    if gps_status:
        t = t_rel(gps_status, base_timestamp)
        checks = [
            "check_fail_gps_fix",
            "check_fail_min_sat_count",
            "check_fail_max_pdop",
            "check_fail_max_vert_err",
            "check_fail_max_spd_err",
            "check_fail_max_vert_spd_err",
        ]
        offset = 0.0
        for key in checks:
            if key in gps_status.data:
                axes[4].step(t, arr(gps_status, key, int) + offset, where="post", lw=0.8, label=key)
                offset += 1.1

    setup_axis(axes[0], "Fix and satellites")
    setup_axis(axes[1], "DOP")
    setup_axis(axes[2], "Reported position accuracy", "m")
    setup_axis(axes[3], "Speed metrics")
    setup_axis(axes[4], "Jamming/spoofing and GPS check flags")
    axes[4].set_xlabel("flight-log relative time [s]")
    axes[1].set_ylim(-0.5, 10.0)
    for ax in axes:
        ax.legend(loc="best", fontsize=7, ncols=2)

    return save_fig(fig, fig_dir, "latest_gps_quality")


def plot_fusion_flags_and_resets(ulog: ULog, base_timestamp: int, fig_dir: Path) -> Path:
    flags = get_ds(ulog, "estimator_status_flags")
    status = get_ds(ulog, "estimator_status")
    event = get_ds(ulog, "estimator_event_flags")
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.2), sharex=True)

    if flags:
        t = t_rel(flags, base_timestamp)
        flag_fields = [
            ("cs_gps", 0),
            ("cs_gnss_vel", 1),
            ("cs_gps_hgt", 2),
            ("cs_baro_hgt", 3),
            ("cs_inertial_dead_reckoning", 4),
            ("fs_bad_acc_vertical", 5),
        ]
        for key, offset in flag_fields:
            if key in flags.data:
                axes[0].step(t, arr(flags, key, int) + offset, where="post", label=key)

    if status:
        t = t_rel(status, base_timestamp)
        for key in [
            "reset_count_vel_ne",
            "reset_count_vel_d",
            "reset_count_pos_ne",
            "reset_count_pod_d",
            "reset_count_quat",
        ]:
            if key in status.data:
                axes[1].step(t, arr(status, key, int), where="post", label=key)
        for key in ["filter_fault_flags", "innovation_check_flags", "gps_check_fail_flags"]:
            if key in status.data:
                axes[2].step(t, arr(status, key, int), where="post", label=key)

    if event:
        t = t_rel(event, base_timestamp)
        event_fields = [
            ("reset_vel_to_gps", 0),
            ("reset_pos_to_gps", 1),
            ("reset_hgt_to_gps", 2),
            ("reset_hgt_to_baro", 3),
            ("gps_data_stopped", 4),
        ]
        for key, offset in event_fields:
            if key in event.data:
                axes[2].step(t, arr(event, key, int) + 5 * offset, where="post", lw=0.9, label=key)

    setup_axis(axes[0], "Fusion and fault flags")
    setup_axis(axes[1], "State reset counters")
    setup_axis(axes[2], "Status/event masks")
    axes[2].set_xlabel("flight-log relative time [s]")
    for ax in axes:
        ax.legend(loc="best", fontsize=7, ncols=2)

    return save_fig(fig, fig_dir, "latest_fusion_flags_resets")


def plot_innovation_ratios(ulog: ULog, base_timestamp: int, fig_dir: Path) -> Path:
    ratios = get_ds(ulog, "estimator_innovation_test_ratios")
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.2), sharex=True)

    if ratios:
        t = t_rel(ratios, base_timestamp)
        groups = [
            (axes[0], ["gps_hvel[0]", "gps_hvel[1]", "gps_vvel"], "GPS velocity innovation test ratios"),
            (axes[1], ["gps_hpos[0]", "gps_hpos[1]", "gps_vpos"], "GPS position innovation test ratios"),
            (axes[2], ["baro_vpos", "heading", "mag_field[0]", "mag_field[1]", "mag_field[2]"], "Other relevant test ratios"),
        ]
        for ax, keys, title in groups:
            for key in keys:
                if key in ratios.data:
                    y = arr(ratios, key)
                    y = np.where(np.isfinite(y), y, np.nan)
                    ax.plot(t, y, lw=0.9, label=key)
            ax.axhline(1.0, color="black", lw=0.9, ls="--", label="gate")
            ax.axhline(0.36, color="gray", lw=0.7, ls=":", label="3 sigma equiv")
            setup_axis(ax, title, "test ratio")
            ax.set_yscale("symlog", linthresh=0.1)

    axes[-1].set_xlabel("flight-log relative time [s]")
    for ax in axes:
        ax.legend(loc="best", fontsize=7, ncols=2)

    return save_fig(fig, fig_dir, "latest_innovation_test_ratios")


def plot_residuals(
    ulog: ULog,
    base_timestamp: int,
    fig_dir: Path,
    keys: list[str],
    name: str,
    title_prefix: str,
) -> Path:
    innov = get_ds(ulog, "estimator_innovations")
    vari = get_ds(ulog, "estimator_innovation_variances")
    fig, axes = plt.subplots(len(keys), 1, figsize=(10.5, 2.2 * len(keys) + 1.2), sharex=True)
    if len(keys) == 1:
        axes = [axes]

    if innov and vari:
        t_i = t_rel(innov, base_timestamp)
        t_v = t_rel(vari, base_timestamp)
        for ax, key in zip(axes, keys):
            if key not in innov.data or key not in vari.data:
                setup_axis(ax, f"{key} missing")
                continue
            residual = arr(innov, key)
            variance = interp_at(t_v, arr(vari, key), t_i)
            threshold = 5.0 * np.sqrt(np.maximum(variance, 0.0))
            ax.plot(t_i, residual, label=f"{key} innovation", lw=0.9)
            ax.plot(t_i, threshold, color="tab:red", lw=0.8, ls="--", label="+5 sigma gate")
            ax.plot(t_i, -threshold, color="tab:red", lw=0.8, ls="--", label="-5 sigma gate")
            setup_axis(ax, f"{title_prefix}: {key}", "m or m/s")
            ax.legend(loc="best", fontsize=7)

    axes[-1].set_xlabel("flight-log relative time [s]")
    return save_fig(fig, fig_dir, name)


def plot_replay_comparison(
    loaded: dict[str, ULog],
    log_specs: tuple[LogSpec, ...],
    base_timestamp: int,
    fig_dir: Path,
) -> Path:
    fig, axes = plt.subplots(4, 1, figsize=(10.5, 9.0), sharex=True)
    cmap = plt.get_cmap("tab10")
    specs = [spec for spec in log_specs if spec.key in loaded and spec.key != "orig"]
    offsets = {spec.key: idx * 1.5 for idx, spec in enumerate(specs)}

    for idx, spec in enumerate(specs):
        ulog = loaded[spec.key]
        status = get_ds(ulog, "estimator_status")
        flags = get_ds(ulog, "estimator_status_flags")
        lpos = get_ds(ulog, "vehicle_local_position")
        color = cmap(idx % 10)
        label = spec.key
        if status:
            t = t_rel(status, base_timestamp)
            if "reset_count_pod_d" in status.data:
                axes[0].step(t, arr(status, "reset_count_pod_d", int), where="post", color=color, label=f"{label} z reset")
            if "reset_count_vel_d" in status.data:
                axes[1].step(t, arr(status, "reset_count_vel_d", int), where="post", color=color, label=f"{label} vz reset")
        if flags:
            t = t_rel(flags, base_timestamp)
            if "fs_bad_acc_vertical" in flags.data:
                axes[2].step(t, arr(flags, "fs_bad_acc_vertical", int) + offsets[spec.key], where="post", color=color, label=f"{label} bad vert accel")
            if "cs_gnss_vel" in flags.data:
                axes[2].step(t, arr(flags, "cs_gnss_vel", int) + offsets[spec.key] + 0.5, where="post", color=color, ls="--", label=f"{label} gnss vel")
        if lpos:
            t = t_rel(lpos, base_timestamp)
            axes[3].plot(t, arr(lpos, "z"), color=color, lw=0.8, label=f"{label} local z")

    setup_axis(axes[0], "Vertical position reset count")
    setup_axis(axes[1], "Vertical velocity reset count")
    setup_axis(axes[2], "Bad vertical accel and GNSS velocity flags")
    setup_axis(axes[3], "Local z comparison", "m")
    axes[3].set_xlabel("flight-log relative time [s]")
    for ax in axes:
        ax.legend(loc="best", fontsize=7, ncols=2)

    return save_fig(fig, fig_dir, "replay_comparison")


def metric_summary(ulog: ULog, base_timestamp: int) -> dict[str, str]:
    gps = get_ds(ulog, "vehicle_gps_position")
    air = get_ds(ulog, "vehicle_air_data")
    flags = get_ds(ulog, "estimator_status_flags")
    status = get_ds(ulog, "estimator_status")
    ratios = get_ds(ulog, "estimator_innovation_test_ratios")
    gps_status = get_ds(ulog, "estimator_gps_status")
    lpos = get_ds(ulog, "vehicle_local_position")
    gpos = get_ds(ulog, "vehicle_global_position")

    out: dict[str, str] = {}
    if gps:
        gps_alt = arr(gps, "altitude_msl_m")
        out["gps_alt_delta"] = f"{gps_alt[-1] - gps_alt[0]:.1f} m"
        out["gps_alt_last"] = f"{gps_alt[-1]:.1f} m"
        out["gps_quality_typical"] = (
            f"{np.nanmedian(arr(gps, 'satellites_used')):.0f} sats, "
            f"HDOP {np.nanmedian(arr(gps, 'hdop')):.2f}, "
            f"VDOP {np.nanmedian(arr(gps, 'vdop')):.2f}, "
            f"EPV {np.nanmedian(arr(gps, 'epv')):.2f} m"
        )
    if air:
        baro_alt = arr(air, "baro_alt_meter")
        out["baro_alt_delta"] = f"{baro_alt[-1] - baro_alt[0]:.1f} m"
    if lpos:
        z = arr(lpos, "z")
        vz = arr(lpos, "vz")
        out["local_z_range"] = f"{np.nanmin(z):.1f} to {np.nanmax(z):.1f} m"
        out["local_z_last"] = f"{z[-1]:.1f} m"
        out["local_vz_range"] = f"{np.nanmin(vz):.1f} to {np.nanmax(vz):.1f} m/s"
    if gpos:
        alt = arr(gpos, "alt")
        out["global_alt_last"] = f"{alt[-1]:.1f} m"
    if status:
        for key in ["reset_count_vel_ne", "reset_count_vel_d", "reset_count_pos_ne", "reset_count_pod_d", "reset_count_quat"]:
            if key in status.data:
                values = arr(status, key, int)
                out[key] = f"{values[0]}--{values[-1]}"
    if flags:
        t = t_rel(flags, base_timestamp)
        for key in ["cs_gps", "cs_gnss_vel", "cs_gps_hgt", "cs_baro_hgt", "fs_bad_acc_vertical", "cs_inertial_dead_reckoning"]:
            if key in flags.data:
                out[key] = format_spans(bool_spans(t, arr(flags, key, int)), limit=3)
    if ratios:
        for key in ["gps_vpos", "gps_vvel", "gps_hvel[0]", "gps_hvel[1]", "gps_hpos[0]", "gps_hpos[1]", "baro_vpos"]:
            if key in ratios.data:
                values = arr(ratios, key)
                finite = values[np.isfinite(values)]
                if finite.size:
                    out[f"{key}_max"] = f"{np.nanmax(finite):.2f}"
                    out[f"{key}_gt1"] = f"{int((finite > 1.0).sum())}/{finite.size}"
    if gps_status:
        t = t_rel(gps_status, base_timestamp)
        for key in ["check_fail_max_vert_spd_err", "check_fail_max_vert_err", "check_fail_gps_fix"]:
            if key in gps_status.data:
                values = arr(gps_status, key, int).astype(bool)
                out[key] = f"{int(values.sum())}/{values.size}; {format_spans(bool_spans(t, values), limit=3)}"

    return out


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def build_latex(
    config: ReviewConfig,
    figures: dict[str, Path],
    summaries: dict[str, dict[str, str]],
) -> str:
    latest = summaries[config.latest_key]
    spec_by_key = {spec.key: spec for spec in config.logs}
    rows = []
    for spec in config.logs:
        if spec.key == config.original_key:
            continue
        s = summaries[spec.key]
        title = latex_escape(spec.title)
        rows.append(
            " & ".join(
                [
                    title,
                    latex_escape(s.get("gps_alt_delta", "n/a")),
                    latex_escape(s.get("reset_count_pod_d", "n/a")),
                    latex_escape(s.get("reset_count_vel_d", "n/a")),
                    latex_escape(s.get("fs_bad_acc_vertical", "n/a")),
                    latex_escape(s.get("cs_gnss_vel", "n/a")),
                ]
            )
            + r" \\"
        )

    rel_figures = {key: path.relative_to(config.output_dir).as_posix() for key, path in figures.items()}
    latest_title = latex_escape(spec_by_key[config.latest_key].title)

    return rf"""\documentclass[10pt]{{article}}
\usepackage[margin=0.7in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{float}}
\usepackage{{hyperref}}
\usepackage{{siunitx}}
\usepackage{{caption}}
\usepackage{{subcaption}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{0.5em}}

\title{{{latex_escape(config.title)}}}
\author{{Generated from ULog replay artifacts}}
\date{{}}

\begin{{document}}
\maketitle

\section{{Scope}}
This report reviews PX4 EKF2 replay logs with emphasis on height-source selection, GNSS innovation gates, GPS quality checks, reset counters, and dead-reckoning intervals.  The primary log is \texttt{{{latest_title}}}.

\section{{High-Level Read}}
Reported GPS altitude changes by {latex_escape(latest.get("gps_alt_delta", "n/a"))}.  GNSS quality indicators are summarized as {latex_escape(latest.get("gps_quality_typical", "n/a"))}.  The latest local-z output ends at {latex_escape(latest.get("local_z_last", "n/a"))}; EKF global altitude ends at {latex_escape(latest.get("global_alt_last", "n/a"))}.

GNSS velocity fusion is interpreted from \texttt{{estimator\_status\_flags.cs\_gnss\_vel}} because dedicated \texttt{{estimator\_aid\_src\_gnss\_vel}} topics are not always logged in replay output.  Innovation test ratio 1.0 is the configured gate, not one sigma.

\section{{Summary Table}}
\begin{{table}}[H]
\centering
\scriptsize
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{llllll}}
\toprule
Replay & GPS alt delta & z resets & vz resets & bad vertical accel & GNSS vel active \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}%
}}
\caption{{Replay comparison. Reset ranges show first--last counter values.}}
\end{{table}}

\section{{Latest Replay: Position and Raw Altitude}}
\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{{rel_figures["position"]}}}
\caption{{Raw GPS altitude and EKF local/global position outputs for the latest replay.}}
\end{{figure}}

\section{{GPS Quality Indicators}}
\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{{rel_figures["quality"]}}}
\caption{{GPS fix, satellites, DOP, reported accuracy, jamming/spoofing fields, and EKF GPS check failures. GDOP is not logged directly; HDOP, VDOP, and EKF PDOP checks are the available proxies.}}
\end{{figure}}

\section{{Fusion Flags and Reset Counters}}
\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{{rel_figures["flags"]}}}
\caption{{Fusion-state flags, reset counters, and event/status masks.}}
\end{{figure}}

\section{{Innovation Test Ratios}}
\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{{rel_figures["ratios"]}}}
\caption{{Innovation test ratios. A ratio of 1.0 is the configured innovation gate. The 0.36 line is a rough 3-sigma equivalent when using a 5-sigma gate.}}
\end{{figure}}

\section{{Residuals Versus Thresholds}}
\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{{rel_figures["vertical_residuals"]}}}
\caption{{Vertical residuals and approximate $\pm5\sigma$ gate thresholds from logged innovation variances.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{{rel_figures["horizontal_residuals"]}}}
\caption{{Horizontal GPS velocity/position residuals and approximate $\pm5\sigma$ gate thresholds.}}
\end{{figure}}

\section{{Replay Comparison}}
\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{{rel_figures["comparison"]}}}
\caption{{Comparison of replay configurations.}}
\end{{figure}}

\section{{Latest Metrics}}
\begin{{itemize}}
\item \textbf{{GNSS height active:}} {latex_escape(latest.get("cs_gps_hgt", "n/a"))}
\item \textbf{{GNSS velocity active:}} {latex_escape(latest.get("cs_gnss_vel", "n/a"))}
\item \textbf{{Bad vertical accel:}} {latex_escape(latest.get("fs_bad_acc_vertical", "n/a"))}
\item \textbf{{GPS vertical position ratio max:}} {latex_escape(latest.get("gps_vpos_max", "n/a"))}
\item \textbf{{GPS vertical velocity ratio max:}} {latex_escape(latest.get("gps_vvel_max", "n/a"))}
\item \textbf{{GPS vertical speed check failures:}} {latex_escape(latest.get("check_fail_max_vert_spd_err", "n/a"))}
\end{{itemize}}

\end{{document}}
"""


def write_summary(output_dir: Path, summaries: dict[str, dict[str, str]]) -> Path:
    summary_path = output_dir / "summary.txt"
    lines = []
    for key, summary in summaries.items():
        lines.append(f"[{key}]")
        for metric, value in sorted(summary.items()):
            lines.append(f"{metric}: {value}")
        lines.append("")
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def compile_latex(tex_path: Path) -> Path | None:
    if shutil.which("latexmk") is None:
        return None
    subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", tex_path.name],
        cwd=tex_path.parent,
        check=True,
    )
    pdf_path = tex_path.with_suffix(".pdf")
    return pdf_path if pdf_path.exists() else None


def generate_review(config: ReviewConfig, compile_pdf: bool | None = None) -> ReviewArtifacts:
    output_dir = config.output_dir
    fig_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    missing = [spec.path for spec in config.logs if not spec.path.exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"missing configured log file(s):\n{formatted}")

    loaded: dict[str, ULog] = {spec.key: ULog(str(spec.path)) for spec in config.logs}
    base_timestamp = loaded[config.original_key].start_timestamp
    latest = loaded[config.latest_key]

    figures = {
        "position": plot_position_overview(latest, base_timestamp, fig_dir),
        "quality": plot_gps_quality(latest, base_timestamp, fig_dir),
        "flags": plot_fusion_flags_and_resets(latest, base_timestamp, fig_dir),
        "ratios": plot_innovation_ratios(latest, base_timestamp, fig_dir),
        "vertical_residuals": plot_residuals(
            latest,
            base_timestamp,
            fig_dir,
            ["gps_vpos", "gps_vvel", "baro_vpos"],
            "latest_vertical_residual_thresholds",
            "vertical residual",
        ),
        "horizontal_residuals": plot_residuals(
            latest,
            base_timestamp,
            fig_dir,
            ["gps_hvel[0]", "gps_hvel[1]", "gps_hpos[0]", "gps_hpos[1]"],
            "latest_horizontal_residual_thresholds",
            "horizontal residual",
        ),
        "comparison": plot_replay_comparison(loaded, config.logs, base_timestamp, fig_dir),
    }

    summaries = {key: metric_summary(ulog, base_timestamp) for key, ulog in loaded.items()}
    summary_path = write_summary(output_dir, summaries)

    tex_path = output_dir / "ekf2_replay_review.tex"
    tex_path.write_text(build_latex(config, figures, summaries), encoding="utf-8")

    should_compile = config.compile_pdf if compile_pdf is None else compile_pdf
    pdf_path = compile_latex(tex_path) if should_compile else None

    return ReviewArtifacts(
        output_dir=output_dir,
        tex_path=tex_path,
        summary_path=summary_path,
        pdf_path=pdf_path,
        figures=figures,
    )
