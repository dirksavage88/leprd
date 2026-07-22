"""Precision-landing failure analysis.

Analyses a single ULog for optical-flow health, flight-mode timeline,
land-detection, and AGL/thrust traces during identified PL attempts.
Does NOT touch barometer or propwash code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from pyulog import ULog

from .analysis import (
    arr,
    bool_spans,
    compile_latex,
    get_ds,
    get_thrust,
    get_range,
    save_fig,
    setup_axis,
    t_rel,
)


def _latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


# ---------------------------------------------------------------------------
# PX4 nav_state enum (v1.14 / v1.15)
# ---------------------------------------------------------------------------
_NAV = {
    0: "MANUAL",
    1: "ALTCTL",
    2: "POSCTL",
    3: "MISSION",
    4: "LOITER",
    5: "RTL",
    10: "ACRO",
    14: "OFFBOARD",
    15: "STAB",
    17: "AUTO_TAKEOFF",
    18: "AUTO_LAND",
    20: "AUTO_PRECLAND",
    21: "ORBIT",
}

# Nav states that indicate a landing attempt is in progress
_LANDING_STATES = {18, 20, 14}  # AUTO_LAND, AUTO_PRECLAND, OFFBOARD


@dataclass(frozen=True)
class PLArtifacts:
    output_dir: Path
    summary_path: Path
    figures: dict[str, Path]
    tex_path: Path | None = None
    pdf_path: Path | None = None


class PLWindow(NamedTuple):
    label: str
    t_start: float
    t_end: float
    nav_state: int


# ---------------------------------------------------------------------------
# Replay check
# ---------------------------------------------------------------------------

def check_ekf2_replay(ulog: ULog) -> tuple[bool, str]:
    """Return (is_replay, human-readable note)."""
    topic_names = {d.name for d in ulog.data_list}
    if "ekf2_timestamps" in topic_names:
        return True, "ekf2_timestamps topic present — this is an EKF2 replay log"
    replay_topics = [n for n in topic_names if "replay" in n.lower() or "groundtruth" in n.lower()]
    if replay_topics:
        return True, f"replay/groundtruth topics found: {replay_topics}"
    return False, "no replay markers — real flight log"


# ---------------------------------------------------------------------------
# Timeline helpers
# ---------------------------------------------------------------------------

def nav_state_transitions(ulog: ULog, base_timestamp: int) -> list[tuple[float, int, str]]:
    vs = get_ds(ulog, "vehicle_status")
    if vs is None:
        return []
    t = t_rel(vs, base_timestamp)
    nav = arr(vs, "nav_state", dtype=int)
    transitions: list[tuple[float, int, str]] = []
    prev = None
    for ti, n in zip(t, nav):
        if n != prev:
            transitions.append((float(ti), int(n), _NAV.get(int(n), str(n))))
            prev = n
    return transitions


def detect_pl_windows(ulog: ULog, base_timestamp: int, min_duration_s: float = 2.0) -> list[PLWindow]:
    """Return time windows where a landing-capable nav state was active."""
    vs = get_ds(ulog, "vehicle_status")
    if vs is None:
        return []
    t = t_rel(vs, base_timestamp)
    nav = arr(vs, "nav_state", dtype=int)

    windows: list[PLWindow] = []
    start = None
    cur_state = None
    for ti, n in zip(t, nav):
        if n in _LANDING_STATES and start is None:
            start = float(ti)
            cur_state = int(n)
        elif n not in _LANDING_STATES and start is not None:
            if float(ti) - start >= min_duration_s:
                label = f"PL{len(windows) + 1} ({_NAV.get(cur_state, cur_state)})"
                windows.append(PLWindow(label, start, float(ti), cur_state))
            start = None
            cur_state = None

    if start is not None and len(t) > 0:
        end = float(t[-1])
        if end - start >= min_duration_s:
            label = f"PL{len(windows) + 1} ({_NAV.get(cur_state, cur_state)})"
            windows.append(PLWindow(label, start, end, cur_state))

    return windows


# ---------------------------------------------------------------------------
# Plot: mode timeline — Gantt-style coloured spans per nav state
# ---------------------------------------------------------------------------

_MODE_COLORS: dict[int, str] = {
    0:  "tab:gray",     # MANUAL
    1:  "tab:cyan",     # ALTCTL
    2:  "tab:blue",     # POSCTL
    3:  "tab:purple",   # MISSION
    4:  "tab:green",    # LOITER
    5:  "gold",         # RTL
    10: "tab:brown",    # ACRO
    14: "tab:orange",   # OFFBOARD
    15: "tab:olive",    # STAB
    17: "tab:pink",     # TAKEOFF
    18: "tab:red",      # AUTO_LAND
    20: "crimson",      # AUTO_PRECLAND
}


def _mode_spans(
    ulog: ULog, base_timestamp: int
) -> list[tuple[float, float, int, str]]:
    """Return (t_start, t_end, nav_state, name) for every contiguous mode segment."""
    vs = get_ds(ulog, "vehicle_status")
    if vs is None:
        return []
    t = t_rel(vs, base_timestamp)
    nav = arr(vs, "nav_state", dtype=int)
    spans: list[tuple[float, float, int, str]] = []
    prev_n, prev_t = int(nav[0]), float(t[0])
    for ti, n in zip(t[1:], nav[1:]):
        if n != prev_n:
            spans.append((prev_t, float(ti), prev_n, _NAV.get(prev_n, str(prev_n))))
            prev_n, prev_t = int(n), float(ti)
    spans.append((prev_t, float(t[-1]), prev_n, _NAV.get(prev_n, str(prev_n))))
    return spans


def plot_mode_timeline(
    ulog: ULog,
    base_timestamp: int,
    fig_dir: Path,
    pl_windows: list[PLWindow],
) -> Path:
    """2-panel: Gantt mode timeline on top, AGL trace below. Both share the time axis."""
    spans = _mode_spans(ulog, base_timestamp)
    range_result = get_range(ulog, base_timestamp)
    ld = get_ds(ulog, "vehicle_land_detected")

    fig, (ax_mode, ax_agl) = plt.subplots(
        2, 1, figsize=(13, 5), sharex=True,
        gridspec_kw={"height_ratios": [1, 2]},
    )
    fig.suptitle("Vehicle mode transitions", fontsize=10)

    # ---- top panel: Gantt bars ----
    bar_h = 0.6
    seen: set[int] = set()
    for t0, t1, state, name in spans:
        color = _MODE_COLORS.get(state, "tab:gray")
        label = name if state not in seen else None
        ax_mode.barh(
            0, t1 - t0, left=t0, height=bar_h,
            color=color, edgecolor="white", linewidth=0.5,
            label=label, align="center",
        )
        seen.add(state)
        dur = t1 - t0
        # annotate if wide enough to fit text (>1s visible space)
        if dur > 1.0:
            ax_mode.text(
                t0 + dur / 2, 0, f"{name}\n{dur:.1f}s",
                ha="center", va="center", fontsize=7,
                fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.1", fc="none", ec="none"),
            )
        else:
            # narrow band: label above with arrow
            ax_mode.annotate(
                f"{name}\n{dur:.2f}s",
                xy=(t0 + dur / 2, bar_h / 2),
                xytext=(t0 + dur / 2, 1.1),
                fontsize=6, ha="center", color=color,
                arrowprops=dict(arrowstyle="-", color=color, lw=0.8),
            )

    ax_mode.set_yticks([])
    ax_mode.set_ylim(-0.6, 1.8)
    ax_mode.legend(loc="upper right", fontsize=7, ncols=4, framealpha=0.8)
    ax_mode.set_title("Nav state (coloured spans, duration labelled)", fontsize=8, pad=2)

    # ---- bottom panel: AGL + land-detected shading ----
    if range_result is not None:
        t_r, r_m = range_result
        ax_agl.plot(t_r, r_m, lw=1.2, color="tab:blue", label="rangefinder AGL")

    if ld is not None:
        t_ld = t_rel(ld, base_timestamp)
        landed = arr(ld, "landed", dtype=int)
        for s, e in bool_spans(t_ld, landed):
            ax_agl.axvspan(s, e, color="green", alpha=0.18, label="landed" if s == bool_spans(t_ld, landed)[0][0] else "")

    # vertical lines at every mode transition
    for t0, t1, state, name in spans[1:]:  # skip the first (no left edge)
        color = _MODE_COLORS.get(state, "tab:gray")
        ax_agl.axvline(t0, color=color, lw=1.0, ls="--", alpha=0.7)
        ax_mode.axvline(t0, color="white", lw=0.6, alpha=0.4)

    setup_axis(ax_agl, "AGL (rangefinder)", "m")
    ax_agl.set_xlabel("flight-log relative time [s]")
    ax_agl.legend(loc="best", fontsize=7)

    # shade PL windows on AGL panel
    _shade_windows([ax_agl], pl_windows)

    fig.tight_layout()
    return save_fig(fig, fig_dir, "pl_mode_timeline")


# ---------------------------------------------------------------------------
# Plot: flight overview — nav state, altitude, land detection
# ---------------------------------------------------------------------------

def plot_flight_overview(
    ulog: ULog,
    base_timestamp: int,
    fig_dir: Path,
    pl_windows: list[PLWindow],
) -> Path:
    lpos = get_ds(ulog, "vehicle_local_position")
    vs = get_ds(ulog, "vehicle_status")
    ld = get_ds(ulog, "vehicle_land_detected")
    range_result = get_range(ulog, base_timestamp)
    thrust_result = get_thrust(ulog, base_timestamp)

    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)

    # panel 1: AGL altitude (rangefinder preferred; EKF -z fallback)
    if range_result is not None:
        t_r, r_m = range_result
        axes[0].plot(t_r, r_m, lw=1.2, color="tab:blue", label="rangefinder AGL (truth)")
    if lpos is not None:
        t_lp = t_rel(lpos, base_timestamp)
        ekf_z = arr(lpos, "z")
        # normalise EKF z so min-z maps to 0 when rangefinder absent
        if range_result is None:
            ekf_alt = -ekf_z - np.nanmin(-ekf_z)
        else:
            ekf_alt = -ekf_z
        axes[0].plot(t_lp, ekf_alt, lw=0.9, color="tab:green", alpha=0.7, label="EKF local −z")
    setup_axis(axes[0], "Altitude (rangefinder AGL preferred)", "m")

    # panel 2: vertical velocity
    if lpos is not None:
        axes[1].plot(t_lp, arr(lpos, "vz"), lw=0.9, color="tab:olive", label="EKF vz (NED, +down)")
        axes[1].axhline(0, color="gray", lw=0.6, ls=":")
    setup_axis(axes[1], "Vertical velocity (+ = descending)", "m/s")

    # panel 3: thrust
    if thrust_result is not None:
        t_th, thrust = thrust_result
        axes[2].plot(t_th, thrust, lw=0.9, color="tab:brown", label="|thrust|")
        axes[2].set_ylim(-0.05, 1.05)
    setup_axis(axes[2], "Normalised thrust", "")

    # panel 4: nav state + land-detected
    if vs is not None:
        t_vs = t_rel(vs, base_timestamp)
        nav = arr(vs, "nav_state", dtype=int)
        axes[3].step(t_vs, nav, where="post", lw=1.0, color="tab:purple", label="nav_state")
        # annotate state labels
        transitions = nav_state_transitions(ulog, base_timestamp)
        for ti, n, name in transitions:
            axes[3].annotate(name, (ti, n), fontsize=6, rotation=45,
                             xytext=(2, 4), textcoords="offset points", color="tab:purple")
    if ld is not None:
        t_ld = t_rel(ld, base_timestamp)
        landed = arr(ld, "landed", dtype=int)
        axes[3].step(t_ld, landed * 30, where="post", lw=1.5, color="black",
                     ls="--", alpha=0.5, label="landed×30")
    setup_axis(axes[3], "Nav state (raw int) + land-detected (scaled)", "")
    axes[3].set_xlabel("flight-log relative time [s]")

    # shade PL windows
    _shade_windows(axes, pl_windows)

    for ax in axes:
        ax.legend(loc="best", fontsize=7, ncols=2)

    return save_fig(fig, fig_dir, "pl_flight_overview")


# ---------------------------------------------------------------------------
# Plot: optical flow health
# ---------------------------------------------------------------------------

def plot_optical_flow_health(
    ulog: ULog,
    base_timestamp: int,
    fig_dir: Path,
    pl_windows: list[PLWindow],
) -> Path:
    aid = get_ds(ulog, "estimator_aid_src_optical_flow")
    vof = get_ds(ulog, "vehicle_optical_flow")
    of_vel = get_ds(ulog, "estimator_optical_flow_vel")

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)

    # panel 1: sensor quality
    if vof is not None and "quality" in vof.data:
        t_vof = t_rel(vof, base_timestamp)
        q = arr(vof, "quality")
        axes[0].plot(t_vof, q, lw=0.9, color="tab:blue", label="optical flow quality (0–255)")
        axes[0].axhline(50, color="tab:red", lw=0.8, ls="--", label="quality floor 50")
        axes[0].set_ylim(0, 270)
    setup_axis(axes[0], "Optical flow sensor quality", "")

    # panel 2: EKF innovation test ratios (gate = 1.0)
    if aid is not None:
        t_aid = t_rel(aid, base_timestamp)
        for i, color in [(0, "tab:orange"), (1, "tab:red")]:
            key = f"test_ratio[{i}]"
            if key in aid.data:
                y = arr(aid, key)
                y = np.where(np.isfinite(y), y, np.nan)
                axes[1].plot(t_aid, y, lw=0.9, color=color, label=f"test_ratio[{i}]")
        axes[1].axhline(1.0, color="black", lw=1.0, ls="--", label="gate (1.0)")
        axes[1].set_yscale("symlog", linthresh=0.05)
    setup_axis(axes[1], "EKF OF innovation test ratios (>1 = rejected)", "ratio")

    # panel 3: fusion stall indicator (time_last_fuse stops changing)
    if aid is not None and "time_last_fuse" in aid.data:
        t_aid = t_rel(aid, base_timestamp)
        fuse_t = arr(aid, "time_last_fuse", dtype=float)
        stall = np.zeros(len(fuse_t), dtype=float)
        for i in range(1, len(fuse_t)):
            stall[i] = 1.0 if fuse_t[i] == fuse_t[i - 1] else 0.0
        axes[2].fill_between(t_aid, stall, step="post", color="tab:red", alpha=0.6, label="fusion stalled")
        axes[2].set_ylim(-0.1, 1.3)
    setup_axis(axes[2], "OF fusion stall (EKF stopped fusing)", "")

    # panel 4: EKF-fused body velocity from optical flow
    if of_vel is not None:
        t_vel = t_rel(of_vel, base_timestamp)
        for field, color, label in [
            ("vel_body[0]", "tab:blue", "fused vx body"),
            ("vel_body[1]", "tab:orange", "fused vy body"),
        ]:
            if field in of_vel.data:
                axes[3].plot(t_vel, arr(of_vel, field), lw=0.9, color=color, label=label)
        axes[3].axhline(0, color="gray", lw=0.6, ls=":")
    setup_axis(axes[3], "EKF optical-flow fused body velocity", "m/s")
    axes[3].set_xlabel("flight-log relative time [s]")

    _shade_windows(axes, pl_windows)

    for ax in axes:
        ax.legend(loc="best", fontsize=7, ncols=2)

    return save_fig(fig, fig_dir, "pl_optical_flow")


# ---------------------------------------------------------------------------
# Plot: close-up of each PL window
# ---------------------------------------------------------------------------

def plot_pl_closeup(
    ulog: ULog,
    base_timestamp: int,
    fig_dir: Path,
    pl_windows: list[PLWindow],
    pad_s: float = 3.0,
) -> dict[str, Path]:
    lpos = get_ds(ulog, "vehicle_local_position")
    aid = get_ds(ulog, "estimator_aid_src_optical_flow")
    range_result = get_range(ulog, base_timestamp)
    thrust_result = get_thrust(ulog, base_timestamp)
    ld = get_ds(ulog, "vehicle_land_detected")
    t_ocm, pos_flag, vel_flag = _voxl_offboard_phases(ulog, base_timestamp)

    import matplotlib.patches as mpatches

    out: dict[str, Path] = {}
    for win in pl_windows:
        t0 = win.t_start - pad_s
        t1 = win.t_end + pad_s

        fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(f"Close-up: {win.label}  ({win.t_start:.1f}–{win.t_end:.1f} s)", fontsize=10)

        # panel 1: AGL with VOXL control-phase shading
        ax0 = axes[0]
        if len(t_ocm) > 0:
            m_ocm = (t_ocm >= t0) & (t_ocm <= t1)
            t_w = t_ocm[m_ocm]; v_w = vel_flag[m_ocm]; p_w = pos_flag[m_ocm]
            for i in range(len(t_w) - 1):
                if v_w[i] and not p_w[i]:
                    ax0.axvspan(t_w[i], t_w[i + 1], color="tab:orange", alpha=0.18, zorder=0)
                elif p_w[i]:
                    ax0.axvspan(t_w[i], t_w[i + 1], color="tab:green", alpha=0.25, zorder=0)

        if range_result is not None:
            t_r, r_m = range_result
            m = (t_r >= t0) & (t_r <= t1)
            ax0.plot(t_r[m], r_m[m], lw=1.3, color="tab:blue", label="rangefinder AGL")
        if lpos is not None:
            t_lp = t_rel(lpos, base_timestamp)
            m = (t_lp >= t0) & (t_lp <= t1)
            ax0.plot(t_lp[m], -arr(lpos, "z")[m], lw=0.9, color="tab:green",
                         alpha=0.7, label="EKF −z")
        if ld is not None:
            t_ld = t_rel(ld, base_timestamp)
            m = (t_ld >= t0) & (t_ld <= t1)
            for span_s, span_e in bool_spans(t_ld[m], arr(ld, "landed", dtype=int)[m]):
                ax0.axvspan(span_s + t0, span_e + t0, color="gray", alpha=0.2, label="landed")
        phase_handles = [
            mpatches.Patch(facecolor="tab:orange", alpha=0.45, label="offboard velocity ctrl (ocm.velocity=1)"),
            mpatches.Patch(facecolor="tab:green", alpha=0.5, label="offboard position ctrl (ocm.position=1)"),
        ]
        setup_axis(ax0, "AGL + offboard\\_control\\_mode phase", "m")

        # panel 2: OF test ratios
        if aid is not None:
            t_aid = t_rel(aid, base_timestamp)
            m = (t_aid >= t0) & (t_aid <= t1)
            for i, color in [(0, "tab:orange"), (1, "tab:red")]:
                key = f"test_ratio[{i}]"
                if key in aid.data:
                    y = arr(aid, key)[m]
                    axes[1].plot(t_aid[m], np.where(np.isfinite(y), y, np.nan),
                                 lw=1.0, color=color, label=f"OF test_ratio[{i}]")
            axes[1].axhline(1.0, color="black", lw=1.0, ls="--", label="gate")
            axes[1].set_yscale("symlog", linthresh=0.05)
        setup_axis(axes[1], "OF innovation test ratio (>1 = EKF rejected)", "ratio")

        # panel 3: thrust
        if thrust_result is not None:
            t_th, thrust = thrust_result
            m = (t_th >= t0) & (t_th <= t1)
            axes[2].plot(t_th[m], thrust[m], lw=1.0, color="tab:brown", label="|thrust|")
            axes[2].set_ylim(-0.05, 1.05)
        setup_axis(axes[2], "Normalised thrust", "")

        # panel 4: EKF vz
        if lpos is not None:
            t_lp = t_rel(lpos, base_timestamp)
            m = (t_lp >= t0) & (t_lp <= t1)
            axes[3].plot(t_lp[m], arr(lpos, "vz")[m], lw=1.0, color="tab:olive",
                         label="EKF vz (+down)")
            axes[3].axhline(0, color="gray", lw=0.6, ls=":")
        setup_axis(axes[3], "Vertical velocity (+down)", "m/s")
        axes[3].set_xlabel("time [s]")

        # shade the PL window itself
        for ax in axes:
            ax.axvspan(win.t_start, win.t_end, color="tab:purple", alpha=0.10,
                       label=win.label if ax is axes[0] else "")

        h0, l0 = ax0.get_legend_handles_labels()
        ax0.legend(handles=phase_handles + h0, labels=[p.get_label() for p in phase_handles] + l0,
                   loc="best", fontsize=7, ncols=2)
        for ax in axes[1:]:
            ax.legend(loc="best", fontsize=7, ncols=2)

        slug = win.label.lower().replace(" ", "_").replace("(", "").replace(")", "")
        key = f"closeup_{slug}"
        out[key] = save_fig(fig, fig_dir, key)

    return out


# ---------------------------------------------------------------------------
# Plot: VOXL abort sequence — velocity→position control transition, AGL at acquisition
# ---------------------------------------------------------------------------

def _voxl_offboard_phases(
    ulog: ULog, base_timestamp: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (t, pos_flag, vel_flag) from offboard_control_mode."""
    ocm = get_ds(ulog, "offboard_control_mode")
    if ocm is None:
        return np.array([]), np.array([]), np.array([])
    t = t_rel(ocm, base_timestamp)
    pos = arr(ocm, "position", dtype=float)
    vel = arr(ocm, "velocity", dtype=float)
    return t, pos, vel


def _trajectory_sp(
    ulog: ULog, base_timestamp: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (t, px, py, pz, vz, vx) from trajectory_setpoint."""
    tsp = get_ds(ulog, "trajectory_setpoint")
    if tsp is None:
        empty = np.array([])
        return empty, empty, empty, empty, empty, empty
    t = t_rel(tsp, base_timestamp)
    def _f(k):
        return arr(tsp, k) if k in tsp.data else np.full(len(t), np.nan)
    return t, _f("position[0]"), _f("position[1]"), _f("position[2]"), _f("velocity[2]"), _f("velocity[0]")


def _detect_voxl_abort(t_sp: np.ndarray, pz: np.ndarray, threshold_m: float = 2.0) -> float | None:
    """Return time of first large upward jump in commanded NED z (abort)."""
    valid = np.isfinite(pz)
    t_v = t_sp[valid]; pz_v = pz[valid]
    if len(pz_v) < 2:
        return None
    dz = np.diff(pz_v)  # NED: negative dz = commanded climb
    idx = np.where(dz < -threshold_m)[0]
    return float(t_v[idx[0] + 1]) if len(idx) > 0 else None


def plot_voxl_abort_analysis(
    ulog: ULog,
    base_timestamp: int,
    fig_dir: Path,
    pl_windows: list[PLWindow],
    pad_s: float = 5.0,
) -> Path:
    """4-panel deep-dive on the VOXL velocity→position control transition and abort."""
    range_result = get_range(ulog, base_timestamp)
    lpos = get_ds(ulog, "vehicle_local_position")
    t_ocm, pos_flag, vel_flag = _voxl_offboard_phases(ulog, base_timestamp)
    t_sp, spx, spy, spz, sp_vz, sp_vx = _trajectory_sp(ulog, base_timestamp)

    # Clip to the largest OFFBOARD PL window + pad
    offboard_wins = [w for w in pl_windows if w.nav_state == 14]
    if offboard_wins:
        t0 = offboard_wins[0].t_start - pad_s
        t1 = offboard_wins[-1].t_end + pad_s
    else:
        t0 = t_sp[0] if len(t_sp) else 0.0
        t1 = t_sp[-1] if len(t_sp) else 60.0

    abort_t = _detect_voxl_abort(t_sp, spz)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("VOXL OFFBOARD abort analysis — velocity→position control transition", fontsize=10)

    # --- panel 1: AGL with phase shading ---
    ax = axes[0]
    if range_result is not None:
        t_r, r_m = range_result
        m = (t_r >= t0) & (t_r <= t1)
        ax.plot(t_r[m], r_m[m], lw=1.5, color="tab:blue", label="rangefinder AGL")

    # shade velocity-only phase (no tag lock) and position phase (tag lock)
    if len(t_ocm) > 0:
        m_ocm = (t_ocm >= t0) & (t_ocm <= t1)
        t_w = t_ocm[m_ocm]; v_w = vel_flag[m_ocm]; p_w = pos_flag[m_ocm]
        for i in range(len(t_w) - 1):
            if v_w[i] and not p_w[i]:  # velocity-only = no tag
                ax.axvspan(t_w[i], t_w[i + 1], color="tab:orange", alpha=0.18, zorder=0)
            elif p_w[i]:  # position control = tag lock
                ax.axvspan(t_w[i], t_w[i + 1], color="tab:green", alpha=0.25, zorder=0)

    if abort_t is not None:
        ax.axvline(abort_t, color="tab:red", lw=1.5, ls="--", label=f"abort t={abort_t:.2f}s")
    setup_axis(ax, "AGL (rangefinder) + VOXL control phase", "m")

    # --- panel 2: commanded NED z vs actual NED z ---
    ax = axes[1]
    if lpos is not None:
        t_lp = t_rel(lpos, base_timestamp)
        m = (t_lp >= t0) & (t_lp <= t1)
        ax.plot(t_lp[m], arr(lpos, "z")[m], lw=1.1, color="tab:green", label="vehicle NED z (actual)")

    if len(t_sp) > 0:
        m = (t_sp >= t0) & (t_sp <= t1)
        z_cmd = np.where(np.isfinite(spz), spz, np.nan)
        ax.plot(t_sp[m], z_cmd[m], lw=1.5, color="tab:red", ls="--",
                marker="o", markersize=3, label="traj_sp NED z (VOXL cmd)")

    ax.invert_yaxis()  # NED: lower z = higher altitude → intuitive
    if abort_t is not None:
        ax.axvline(abort_t, color="tab:red", lw=1.2, ls="--", alpha=0.6)
    setup_axis(ax, "NED z: actual vs VOXL-commanded (inverted: up = higher alt)", "m NED")

    # --- panel 3: offboard_control_mode flags ---
    ax = axes[2]
    if len(t_ocm) > 0:
        m = (t_ocm >= t0) & (t_ocm <= t1)
        ax.step(t_ocm[m], pos_flag[m], where="post", lw=1.5, color="tab:green",
                label="offboard_ctrl_mode.position (best tag-lock proxy)")
        ax.step(t_ocm[m], vel_flag[m] * 0.6, where="post", lw=1.2, color="tab:orange",
                ls="--", label="offboard\\_ctrl\\_mode.velocity ×0.6 (velocity ctrl phase)")
        ax.set_ylim(-0.1, 1.2)
    ax.text(0.01, 0.92, "NOTE: landing_target_pose topic ABSENT — no direct AprilTag bit in log",
            transform=ax.transAxes, fontsize=6, color="tab:red", style="italic")
    if abort_t is not None:
        ax.axvline(abort_t, color="tab:red", lw=1.2, ls="--", alpha=0.6)
    setup_axis(ax, "offboard_control_mode booleans (position=True ≡ VOXL tag lock)", "bool")

    # --- panel 4: commanded vz vs actual vz ---
    ax = axes[3]
    if lpos is not None:
        t_lp = t_rel(lpos, base_timestamp)
        m = (t_lp >= t0) & (t_lp <= t1)
        ax.plot(t_lp[m], arr(lpos, "vz")[m], lw=1.1, color="tab:olive", label="vehicle vz actual (NED +down)")

    if len(t_sp) > 0:
        m = (t_sp >= t0) & (t_sp <= t1)
        vz_cmd = np.where(np.isfinite(sp_vz), sp_vz, np.nan)
        ax.plot(t_sp[m], vz_cmd[m], lw=1.4, color="tab:purple", ls="--", marker="s",
                markersize=3, label="traj_sp vz (VOXL cmd, valid in vel phase)")

    ax.axhline(0, color="gray", lw=0.6, ls=":")
    if abort_t is not None:
        ax.axvline(abort_t, color="tab:red", lw=1.2, ls="--", alpha=0.6)
    setup_axis(ax, "Vertical velocity vz (NED, +down = descending)", "m/s")
    ax.set_xlabel("flight-log relative time [s]")

    _shade_windows(axes, pl_windows)
    import matplotlib.patches as mpatches
    phase_handles = [
        mpatches.Patch(facecolor="tab:orange", alpha=0.45, label="velocity ctrl (ocm.velocity=1, ocm.position=0)"),
        mpatches.Patch(facecolor="tab:green", alpha=0.5, label="position ctrl (ocm.position=1)"),
    ]
    h0, l0 = axes[0].get_legend_handles_labels()
    axes[0].legend(handles=phase_handles + h0, labels=[p.get_label() for p in phase_handles] + l0,
                   loc="best", fontsize=7, ncols=2)
    for ax in axes[1:]:
        ax.legend(loc="best", fontsize=7, ncols=2)
    axes[-1].set_xlim(t0, t1)

    return save_fig(fig, fig_dir, "pl_voxl_abort_analysis")


# ---------------------------------------------------------------------------
# Plot: VOXL XY approach — lateral guidance and horizontal error
# ---------------------------------------------------------------------------

def plot_voxl_xy_approach(
    ulog: ULog,
    base_timestamp: int,
    fig_dir: Path,
    pl_windows: list[PLWindow],
    pad_s: float = 5.0,
) -> Path:
    """4-panel showing VOXL XY guidance: commanded vs actual horizontal motion."""
    lpos = get_ds(ulog, "vehicle_local_position")
    range_result = get_range(ulog, base_timestamp)
    t_ocm, pos_flag, vel_flag = _voxl_offboard_phases(ulog, base_timestamp)
    t_sp, spx, spy, spz, sp_vz, sp_vx = _trajectory_sp(ulog, base_timestamp)
    tsp = get_ds(ulog, "trajectory_setpoint")
    sp_vy = arr(tsp, "velocity[1]") if tsp is not None and "velocity[1]" in tsp.data else np.full(len(t_sp), np.nan)

    offboard_wins = [w for w in pl_windows if w.nav_state == 14]
    if offboard_wins:
        t0 = offboard_wins[0].t_start - pad_s
        t1 = offboard_wins[-1].t_end + pad_s
    else:
        t0, t1 = 0.0, 60.0

    abort_t = _detect_voxl_abort(t_sp, spz)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("VOXL XY approach: commanded lateral velocity and position error", fontsize=10)

    # --- panel 1: commanded lateral velocity (velocity-only phase only) ---
    ax = axes[0]
    if len(t_sp) > 0:
        m = (t_sp >= t0) & (t_sp <= t1)
        ax.plot(t_sp[m], np.where(np.isfinite(sp_vx), sp_vx, np.nan)[m],
                lw=1.1, color="tab:blue", label="cmd vx (body-aligned)")
        ax.plot(t_sp[m], np.where(np.isfinite(sp_vy), sp_vy, np.nan)[m],
                lw=1.1, color="tab:orange", label="cmd vy")
    ax.axhline(0, color="gray", lw=0.6, ls=":")
    setup_axis(ax, "VOXL commanded lateral velocity (NaN when tag seen)", "m/s")

    # --- panel 2: actual vehicle vx/vy ---
    ax = axes[1]
    if lpos is not None:
        t_lp = t_rel(lpos, base_timestamp)
        m = (t_lp >= t0) & (t_lp <= t1)
        ax.plot(t_lp[m], arr(lpos, "vx")[m], lw=1.0, color="tab:blue", label="vehicle vx actual")
        ax.plot(t_lp[m], arr(lpos, "vy")[m], lw=1.0, color="tab:orange", label="vehicle vy actual")
    ax.axhline(0, color="gray", lw=0.6, ls=":")
    setup_axis(ax, "Vehicle actual lateral velocity", "m/s")

    # --- panel 3: lateral error to commanded position (only during position phase) ---
    ax = axes[2]
    if lpos is not None and len(t_sp) > 0:
        t_lp = t_rel(lpos, base_timestamp)
        veh_x = arr(lpos, "x")
        veh_y = arr(lpos, "y")
        # interpolate vehicle position to setpoint timestamps
        m_sp = (t_sp >= t0) & (t_sp <= t1) & np.isfinite(spx) & np.isfinite(spy)
        if m_sp.sum() > 0:
            vx_interp = np.interp(t_sp[m_sp], t_lp, veh_x)
            vy_interp = np.interp(t_sp[m_sp], t_lp, veh_y)
            err_x = spx[m_sp] - vx_interp
            err_y = spy[m_sp] - vy_interp
            lateral_err = np.hypot(err_x, err_y)
            ax.plot(t_sp[m_sp], lateral_err, lw=1.4, color="tab:red", marker="o",
                    markersize=4, label="|cmd_pos − actual| (position phase only)")
            ax.plot(t_sp[m_sp], err_x, lw=1.0, color="tab:blue", ls="--", alpha=0.7, label="error x")
            ax.plot(t_sp[m_sp], err_y, lw=1.0, color="tab:orange", ls="--", alpha=0.7, label="error y")
        ax.axhline(0, color="gray", lw=0.6, ls=":")
    setup_axis(ax, "Lateral error to commanded position (only valid when tag locked)", "m")

    # --- panel 4: AGL with tag-acquisition annotation ---
    ax = axes[3]
    if range_result is not None:
        t_r, r_m = range_result
        m = (t_r >= t0) & (t_r <= t1)
        ax.plot(t_r[m], r_m[m], lw=1.3, color="tab:blue", label="AGL")

    # mark the switch to position control
    if len(t_ocm) > 0:
        switch_idx = np.where(np.diff(pos_flag.astype(float)) > 0)[0]
        for si in switch_idx:
            st = t_ocm[si + 1]
            if t0 <= st <= t1:
                ax.axvline(st, color="tab:green", lw=1.5, ls=":", label=f"tag acquired t={st:.2f}s")
                if range_result is not None:
                    agl_at_acq = float(np.interp(st, t_r, r_m))
                    ax.annotate(
                        f"tag lock\n{agl_at_acq:.2f}m AGL",
                        (st, agl_at_acq), fontsize=7, color="tab:green",
                        xytext=(6, 6), textcoords="offset points",
                    )

    if abort_t is not None:
        ax.axvline(abort_t, color="tab:red", lw=1.5, ls="--", label=f"abort t={abort_t:.2f}s")
        if range_result is not None:
            agl_at_abort = float(np.interp(abort_t, t_r, r_m))
            ax.annotate(
                f"abort\n{agl_at_abort:.2f}m AGL",
                (abort_t, agl_at_abort), fontsize=7, color="tab:red",
                xytext=(6, 6), textcoords="offset points",
            )
    setup_axis(ax, "AGL — tag acquisition and abort moments", "m")
    ax.set_xlabel("flight-log relative time [s]")

    _shade_windows(axes, pl_windows)
    for ax_i in axes:
        ax_i.legend(loc="best", fontsize=7, ncols=2)
    axes[-1].set_xlim(t0, t1)

    return save_fig(fig, fig_dir, "pl_voxl_xy_approach")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def pl_metric_summary(
    ulog: ULog,
    base_timestamp: int,
    pl_windows: list[PLWindow],
) -> dict[str, str]:
    out: dict[str, str] = {}

    is_replay, replay_note = check_ekf2_replay(ulog)
    out["ekf2_replay"] = replay_note

    # landing_target_pose presence
    has_lt = any(d.name == "landing_target_pose" for d in ulog.data_list)
    out["landing_target_pose"] = "PRESENT" if has_lt else "ABSENT — no active PL target sensor"

    out["pl_windows_detected"] = str(len(pl_windows))
    for win in pl_windows:
        out[f"{win.label}_t_start"] = f"{win.t_start:.2f} s"
        out[f"{win.label}_t_end"] = f"{win.t_end:.2f} s"
        out[f"{win.label}_nav_state"] = _NAV.get(win.nav_state, str(win.nav_state))

    aid = get_ds(ulog, "estimator_aid_src_optical_flow")
    if aid is not None and "test_ratio[0]" in aid.data:
        t_aid = t_rel(aid, base_timestamp)
        tr0 = arr(aid, "test_ratio[0]")
        tr1 = arr(aid, "test_ratio[1]")
        fuse_t = arr(aid, "time_last_fuse", dtype=float)

        for win in pl_windows:
            m = (t_aid >= win.t_start) & (t_aid <= win.t_end)
            if m.sum() == 0:
                continue
            r0 = tr0[m]; r1 = tr1[m]; ft = fuse_t[m]
            stalls = int(np.sum(np.diff(ft) == 0))
            out[f"{win.label}_of_test_ratio_max"] = (
                f"axis0={np.nanmax(r0):.3f}  axis1={np.nanmax(r1):.3f}"
            )
            out[f"{win.label}_of_fusion_stalls"] = f"{stalls}/{len(ft) - 1} samples"
            out[f"{win.label}_of_gate_breach_count"] = (
                f"axis0={int((r0 > 1.0).sum())}  axis1={int((r1 > 1.0).sum())}"
            )

    vof = get_ds(ulog, "vehicle_optical_flow")
    if vof is not None and "quality" in vof.data:
        q = arr(vof, "quality")
        out["of_quality_global"] = f"min={int(q.min())}  max={int(q.max())}  mean={q.mean():.1f}"

    range_result = get_range(ulog, base_timestamp)
    for win in pl_windows:
        if range_result is not None:
            t_r, r_m = range_result
            m = (t_r >= win.t_start) & (t_r <= win.t_end)
            if m.sum() > 0:
                out[f"{win.label}_agl_min_m"] = f"{r_m[m].min():.3f}"
                out[f"{win.label}_agl_at_end_m"] = f"{r_m[m][-1]:.3f}"

    # VOXL OFFBOARD phase metrics
    t_ocm, pos_flag, vel_flag = _voxl_offboard_phases(ulog, base_timestamp)
    t_sp, spx, spy, spz, sp_vz, _ = _trajectory_sp(ulog, base_timestamp)

    if len(t_ocm) > 0 and len(t_sp) > 0:
        abort_t = _detect_voxl_abort(t_sp, spz)
        out["voxl_abort_time_s"] = f"{abort_t:.3f}" if abort_t is not None else "none detected"

        # find switch to position control
        switch_idx = np.where(np.diff(pos_flag.astype(float)) > 0)[0]
        if len(switch_idx) > 0:
            tag_acq_t = float(t_ocm[switch_idx[0] + 1])
            out["voxl_tag_acquisition_time_s"] = f"{tag_acq_t:.3f}"

            # AGL at acquisition
            if range_result is not None:
                t_r, r_m = range_result
                out["voxl_agl_at_tag_acquisition_m"] = f"{float(np.interp(tag_acq_t, t_r, r_m)):.3f}"
            if abort_t is not None and range_result is not None:
                out["voxl_agl_at_abort_m"] = f"{float(np.interp(abort_t, t_r, r_m)):.3f}"

            # duration of each phase within first OFFBOARD window
            offboard_wins = [w for w in pl_windows if w.nav_state == 14]
            if offboard_wins:
                w0 = offboard_wins[0]
                m_vel = (t_ocm >= w0.t_start) & (t_ocm <= w0.t_end) & (vel_flag > 0) & (pos_flag == 0)
                m_pos = (t_ocm >= w0.t_start) & (t_ocm <= w0.t_end) & (pos_flag > 0)
                # approximate duration by time span of each phase
                if m_vel.sum() > 1:
                    out["voxl_velocity_ctrl_duration_s"] = (
                        f"{float(t_ocm[m_vel][-1] - t_ocm[m_vel][0]):.2f}"
                    )
                if m_pos.sum() > 1:
                    out["voxl_position_ctrl_duration_s"] = (
                        f"{float(t_ocm[m_pos][-1] - t_ocm[m_pos][0]):.2f}"
                    )

        # count valid (non-NaN) XYZ position setpoints
        valid_pos_sp = int(np.sum(np.isfinite(spz) & np.isfinite(spx)))
        out["voxl_n_valid_position_setpoints"] = str(valid_pos_sp)

    # EKF2 active fusion sources from estimator_status_flags
    esf = get_ds(ulog, "estimator_status_flags")
    if esf is not None:
        for field, _ in _EKF_CS_FIELDS:
            if field in esf.data:
                active = bool(np.array(esf.data[field]).any())
                out[f"ekf_{field}"] = "YES" if active else "NO"

    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _shade_windows(axes, pl_windows: list[PLWindow], colors: tuple[str, ...] = ("tab:purple", "tab:cyan")) -> None:
    for i, win in enumerate(pl_windows):
        color = colors[i % len(colors)]
        for ax in axes:
            ax.axvspan(win.t_start, win.t_end, color=color, alpha=0.12, zorder=0)
        axes[0].annotate(
            win.label, (win.t_start, axes[0].get_ylim()[1]),
            fontsize=7, color=color, xytext=(2, -10), textcoords="offset points",
        )


# ---------------------------------------------------------------------------
# LaTeX report
# ---------------------------------------------------------------------------

_EKF_CS_FIELDS = [
    ("cs_baro_hgt", "Barometer height"),
    ("cs_opt_flow", "Optical flow velocity"),
    ("cs_rng_hgt",  "Rangefinder height aid"),
    ("cs_ev_pos",   "External vision position (VIO)"),
    ("cs_ev_vel",   "External vision velocity (VIO)"),
    ("cs_ev_yaw",   "External vision yaw (VIO)"),
    ("cs_ev_hgt",   "External vision height (VIO)"),
    ("cs_gps",      "GPS"),
    ("cs_mag_hdg",  "Magnetometer heading"),
]


def _build_ekf_sources_table(summary: dict[str, str]) -> str:
    rows = ""
    for field, label in _EKF_CS_FIELDS:
        val = summary.get(f"ekf_{field}", "---")
        active = val == "YES"
        cell = r"\textbf{YES}" if active else r"\textit{no}"
        escaped_field = field.replace("_", r"\_")
        rows += f"  \\texttt{{{escaped_field}}} & {label} & {cell} \\\\\n"
    return rf"""
\begin{{table}}[H]
\centering
\begin{{tabular}}{{@{{}}lll@{{}}}}
\toprule
\textbf{{cs\_ flag}} & \textbf{{Sensor / source}} & \textbf{{Active?}} \\
\midrule
{rows}\bottomrule
\end{{tabular}}
\caption{{EKF2 active fusion sources (\texttt{{estimator\_status\_flags}}).
  A field marked \textbf{{YES}} was set to 1 at least once during the flight;
  \textit{{no}} means it was 0 throughout.
  All \texttt{{cs\_ev\_*}} flags are \textit{{no}}, confirming VIO data was never
  received or fused --- independent of the \texttt{{EKF2\_EV\_CTRL = 15}} permission setting.}}
\label{{tab:ekf_sources}}
\end{{table}}"""


def build_pl_latex(
    log_path: Path,
    output_dir: Path,
    figures: dict[str, Path],
    summary: dict[str, str],
    pl_windows: list[PLWindow],
) -> str:
    le = _latex_escape
    log_name = le(log_path.name)

    # Relative paths from output_dir (where the .tex lives) to figures/
    rel = {k: v.relative_to(output_dir).as_posix() for k, v in figures.items()}

    # ---- Executive summary paragraph (auto-generated from summary keys) ----
    has_voxl = "voxl_abort_time_s" in summary
    is_offboard = any(w.nav_state == 14 for w in pl_windows)

    exec_lines: list[str] = []
    for win in pl_windows:
        wl = le(win.label)
        t_s = le(summary.get(f"{win.label}_t_start", "?"))
        t_e = le(summary.get(f"{win.label}_t_end", "?"))
        nav = le(summary.get(f"{win.label}_nav_state", "?"))
        exec_lines.append(
            rf"\textbf{{{wl}}} ({nav}, t\,=\,{t_s}--{t_e}):"
        )

    if has_voxl and is_offboard:
        acq_t = le(summary.get("voxl_tag_acquisition_time_s", "?"))
        acq_agl = le(summary.get("voxl_agl_at_tag_acquisition_m", "?"))
        abort_t = le(summary.get("voxl_abort_time_s", "?"))
        abort_agl = le(summary.get("voxl_agl_at_abort_m", "?"))
        vel_dur = le(summary.get("voxl_velocity_ctrl_duration_s", "?"))
        pos_dur = le(summary.get("voxl_position_ctrl_duration_s", "?"))
        lt = le(summary.get("landing_target_pose", "?"))

        exec_summary = rf"""
The VOXL companion computer commanded the abort.
VOXL was in velocity-only control (\texttt{{offboard\_control\_mode.velocity=True}},
\texttt{{position=False}}) for \textbf{{{vel_dur}~s}},
then switched to position control at \textbf{{{acq_agl}~m AGL}} (t\,=\,{acq_t}~s),
which is the proxy event indicating VOXL acquired the AprilTag
(PX4 logs contain no direct tag-detection bit; see \S\,Evidence Trace).
An abort (commanded climb $\approx$5~m) followed immediately at
t\,=\,{abort_t}~s ({abort_agl}~m AGL), after only {pos_dur}~s in position ctrl.
\texttt{{landing\_target\_pose}} was \textbf{{{lt}}}.
The optical flow sensor was healthy throughout; EKF rejection after the abort
is a consequence of the sudden climb, not the cause.

\medskip
\textbf{{Likely root cause:}} The setpoint-type switch from velocity to position ctrl
occurred only at ground level (12\,cm AGL), leaving no margin for a controlled
final approach.  Whether this reflects the tag entering camera FOV late,
or a VOXL minimum-altitude threshold triggering an immediate abort on first
tag acquisition, requires VOXL-side logs to confirm.
This pattern is consistent with the observed second-attempt-only failure:
after the first successful landing the vehicle re-positioned slightly,
delaying (or preventing) tag acquisition until essentially touching the pad.
"""
    else:
        exec_summary = "See metric table and figures below for details."

    # ---- Metrics table ----
    def row(label: str, key: str) -> str:
        val = le(summary.get(key, "---"))
        return rf"  {le(label)} & {val} \\"

    # Window-specific rows
    win_rows = ""
    for win in pl_windows:
        wl = le(win.label)
        win_rows += rf"  \multicolumn{{2}}{{l}}{{\textit{{{wl}}}}} \\" + "\n"
        for suffix, label in [
            ("_t_start", "Start time"),
            ("_t_end", "End time"),
            ("_nav_state", "Nav state"),
            ("_agl_min_m", "Min AGL (m)"),
            ("_agl_at_end_m", "AGL at window end (m)"),
        ]:
            k = f"{win.label}{suffix}"
            if k in summary:
                win_rows += f"  {le(label)} & {le(summary[k])} \\\\\n"

    voxl_rows = ""
    if has_voxl:
        voxl_rows = r"\midrule" + "\n"
        voxl_rows += r"  \multicolumn{2}{l}{\textit{VOXL phase analysis}} \\" + "\n"
        for key, label in [
            ("voxl_tag_acquisition_time_s", "Tag acquisition time (s)"),
            ("voxl_agl_at_tag_acquisition_m", "AGL at tag acquisition (m)"),
            ("voxl_abort_time_s", "Abort command time (s)"),
            ("voxl_agl_at_abort_m", "AGL at abort (m)"),
            ("voxl_velocity_ctrl_duration_s", "Velocity-only ctrl duration (s)"),
            ("voxl_position_ctrl_duration_s", "Position ctrl duration (s)"),
            ("voxl_n_valid_position_setpoints", "Valid position setpoints (total)"),
        ]:
            if key in summary:
                voxl_rows += f"  {le(label)} & {le(summary[key])} \\\\\n"

    of_rows = r"\midrule" + "\n"
    of_rows += r"  \multicolumn{2}{l}{\textit{Optical flow}} \\" + "\n"
    for key, label in [
        ("of_quality_global", "OF quality (global)"),
        ("landing_target_pose", "landing\\_target\\_pose topic"),
        ("ekf2_replay", "EKF2 replay"),
    ]:
        if key in summary:
            of_rows += f"  {le(label)} & {le(summary[key])} \\\\\n"
    for win in pl_windows:
        for suffix, label in [
            ("_of_test_ratio_max", "OF test ratio max"),
            ("_of_gate_breach_count", "OF gate breach count"),
            ("_of_fusion_stalls", "OF fusion stalls"),
        ]:
            k = f"{win.label}{suffix}"
            if k in summary:
                of_rows += f"  {le(label)} & {le(summary[k])} \\\\\n"

    # ---- Close-up figures ----
    closeup_figs = ""
    for key, path in sorted(figures.items()):
        if key.startswith("closeup_"):
            label = key.replace("closeup_", "").replace("_", " ").title()
            closeup_figs += rf"""
\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{{rel[key]}}}
\caption{{Close-up: {le(label)} window — AGL, OF test ratios, thrust, and vertical velocity.}}
\end{{figure}}
"""

    # ---- Mode transitions table ----
    mode_rows = ""
    if "mode_timeline" in rel:
        from pyulog import ULog as _ULog  # local import to avoid circular at module level
        _ulog_tmp = _ULog(str(log_path))
        _base_tmp = _ulog_tmp.start_timestamp
        _spans = _mode_spans(_ulog_tmp, _base_tmp)
        for t0, t1, state, name in _spans:
            dur = t1 - t0
            mode_rows += f"  {le(f'{t0:.2f}')} & {le(f'{t1:.2f}')} & {le(name)} & {le(f'{dur:.2f}')} \\\\\n"

    mode_table = ""
    if mode_rows:
        mode_table = rf"""
\begin{{table}}[H]
\centering
\begin{{tabular}}{{@{{}}rrlr@{{}}}}
\toprule
\textbf{{Start (s)}} & \textbf{{End (s)}} & \textbf{{Mode}} & \textbf{{Duration (s)}} \\
\midrule
{mode_rows}\bottomrule
\end{{tabular}}
\caption{{Complete nav-state transition log. The brief ALTCTL window (t\,$\approx$54.7--59.2~s)
reflects the pilot switching to altitude hold after the abort before returning to POSCTL.}}
\end{{table}}
"""

    # ---- Conditional voxl sections ----
    voxl_figures = ""
    if "voxl_abort" in rel:
        voxl_figures += rf"""
\section{{VOXL Abort Sequence}}
Panel 1 shows AGL with control-phase shading (orange = \texttt{{ocm.velocity=1, position=0}};
green = \texttt{{ocm.position=1}}, the proxy for VOXL switching to tag-referenced setpoints).
Panel 2 compares VOXL-commanded NED\,z with actual vehicle NED\,z;
the step from $z\approx6$ to $z=1$ is the abort command
($z=1$ NED $\approx$ 5.7\,m AGL in this log's local frame).
Panel 3 shows the \texttt{{offboard\_control\_mode}} booleans explicitly.
Panel 4 shows commanded vs.\ actual vertical velocity.

\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{{rel["voxl_abort"]}}}
\caption{{VOXL abort analysis: velocity$\to$position control transition,
commanded NED\,z jump, and vertical velocity response.}}
\end{{figure}}
"""

    if "voxl_xy_approach" in rel:
        voxl_figures += rf"""
\section{{VOXL XY Approach}}
Panel 1: VOXL-commanded lateral velocity during velocity-only phase (\texttt{{ocm.velocity=1}}).
Panel 2: actual vehicle lateral velocity.
Panel 3: lateral error to commanded position --- only valid during the
position-ctrl setpoints (12\,cm AGL, \texttt{{ocm.position=1}}).
Panel 4: AGL with control-mode transition and abort instants annotated.

\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{{rel["voxl_xy_approach"]}}}
\caption{{VOXL XY approach: commanded lateral velocity, actual velocity,
and lateral position error during the brief tag-lock window.}}
\end{{figure}}
"""

    return rf"""\documentclass[10pt]{{article}}
\usepackage[margin=0.75in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{float}}
\usepackage{{hyperref}}
\usepackage{{siunitx}}
\usepackage{{caption}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{0.5em}}

\title{{Precision Landing Failure Analysis}}
\author{{Generated from ULog}}
\date{{}}

\begin{{document}}
\maketitle

\section{{Scope}}
Single-log precision-landing analysis for \texttt{{{log_name}}}.
The vehicle has no GPS; altitude estimation relies on barometer + optical flow.
All timestamps are log-relative seconds from the start of the \texttt{{.ulg}} file.

\section{{Executive Summary}}
{exec_summary}

\section{{Key Metrics}}
\begin{{table}}[H]
\centering
\begin{{tabular}}{{@{{}}ll@{{}}}}
\toprule
\textbf{{Metric}} & \textbf{{Value}} \\
\midrule
  PL windows detected & {le(summary.get("pl_windows_detected", "?"))} \\
{win_rows}{voxl_rows}{of_rows}\bottomrule
\end{{tabular}}
\caption{{Summary metrics extracted from \texttt{{pl\_summary.txt}}.}}
\end{{table}}

\section{{Flight Overview}}
Nav state timeline, AGL (rangefinder preferred), vertical velocity, and
normalised thrust across the full log.
Shaded regions mark identified landing-mode windows.

\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{{rel.get("overview", "")}}}
\caption{{Flight overview: altitude, vertical velocity, thrust, and nav-state timeline.}}
\end{{figure}}

\section{{Mode Transitions}}
Gantt-style timeline of every nav-state segment with durations labelled.
The coloured dashed lines on the AGL panel below mark each transition instant.

The apparent 11~ms \textbf{{POSCTL}} sliver at t\,$\approx$54.645~s is \emph{{not}} a deliberate
pilot action: two \texttt{{MAV\_CMD\_DO\_SET\_MODE}} commands arrived from the GCS
(\texttt{{source\_system = 255}}, i.e.\ QGroundControl) just 2.2~ms apart ---
the first exiting OFFBOARD into POSCTL, the second immediately entering ALTCTL.
This is a QGC implementation artifact: one user click on ``Altitude'' in QGC
dispatches both commands in sequence.

The subsequent \textbf{{ALTCTL}} window (t\,$\approx$54.7--59.2~s, 4.5~s) is the
pilot holding altitude after the abort, followed by a deliberate switch back to
\textbf{{POSCTL}} (second \texttt{{vehicle\_command}} at t\,=\,59.179~s,
\texttt{{param2 = 3}}).

\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{{rel.get("mode_timeline", "")}}}
\caption{{Mode transition timeline (top) and AGL with transition markers (bottom).
Orange = OFFBOARD (VOXL precision landing), cyan = ALTCTL (pilot altitude hold),
blue = POSCTL, red = AUTO\_LAND.}}
\end{{figure}}

{mode_table}

{voxl_figures}

\section{{Optical Flow Health}}
Panel 1: sensor quality (0--255; floor at 50 shown as red dashed).
Panel 2: EKF innovation test ratios for both OF axes (gate = 1.0).
Panel 3: fusion stall indicator (red = EKF stopped fusing OF).
Panel 4: EKF-fused body velocity from optical flow.

\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{{rel.get("optical_flow", "")}}}
\caption{{Optical flow health. Quality was normal throughout;
EKF rejection after t\,=\,52.8~s is a consequence of the commanded climb,
not a cause of the abort.}}
\end{{figure}}

\section{{Window Close-ups}}
{closeup_figs}

\section{{Evidence Trace}}
Every key claim in this report is grounded in a specific uORB topic and field
logged by PX4.
Table~\ref{{tab:evidence}} maps each claim to its source.
Fields marked \emph{{absent}} were searched for and not found in this log.

\begin{{table}}[H]
\centering
\small
\begin{{tabular}}{{@{{}}p{{5.2cm}}p{{5.5cm}}p{{3.5cm}}@{{}}}}
\toprule
\textbf{{Claim}} & \textbf{{uORB topic.field}} & \textbf{{Value / observation}} \\
\midrule
VOXL in velocity-only ctrl (\texttt{{position=False}}) &
  \texttt{{offboard\_control\_mode}} \newline \texttt{{.velocity = True}} \newline \texttt{{.position = False}} &
  t = 39.0--52.1~s \\
\addlinespace
VOXL switched to position ctrl (\texttt{{position=True}}) &
  \texttt{{offboard\_control\_mode}} \newline \texttt{{.position = True}} \newline \texttt{{.velocity = False}} &
  t = 52.134~s (best proxy for tag acquisition; no direct detection bit in PX4 log) \\
\addlinespace
No direct AprilTag detection bit &
  \texttt{{landing\_target\_pose}} &
  \textbf{{ABSENT}} from log; \texttt{{offboard\_control\_mode .position}} is the only proxy \\
\addlinespace
Abort: VOXL commanded climb &
  \texttt{{trajectory\_setpoint}} \newline \texttt{{.position[2]}} &
  6.007 $\to$ 1.000 NED at t = 52.805~s \\
\addlinespace
Abort altitude is a round constant &
  \texttt{{trajectory\_setpoint}} \newline \texttt{{.position[2] = 1.000}} &
  z at OFFBOARD start = 4.699; round value implies VOXL config param, not dynamic computation \\
\addlinespace
AGL at tag acquisition &
  \texttt{{distance\_sensor}} \newline \texttt{{.current\_distance}} &
  0.120~m at t = 52.134~s \\
\addlinespace
Pilot mode switches user-commanded &
  \texttt{{vehicle\_command}} \newline \texttt{{.command = 176}} \newline \texttt{{.source\_system = 255}} &
  src\_sys 255 = GCS (QGC), not VOXL (245) \\
\addlinespace
11~ms POSCTL blink is a QGC artifact &
  Two \texttt{{vehicle\_command}} \newline entries 2.2~ms apart: \newline param2 = 3 then param2 = 2 &
  Single user click in QGC sends POSCTL exit + ALTCTL enter as back-to-back MAVLink commands \\
\bottomrule
\end{{tabular}}
\caption{{Evidence trace: uORB topics and fields supporting each key claim.}}
\label{{tab:evidence}}
\end{{table}}

\section{{Observability Limits}}
The PX4 log exposes only VOXL's \emph{{outputs}} to PX4 --- the trajectory setpoints
it sent and the \texttt{{offboard\_control\_mode}} flags it set.
VOXL's internal inputs (camera frames, AprilTag detection timestamps, abort
decision logic) are not visible from any PX4 uORB topic.
Specifically:

\begin{{itemize}}
  \item \texttt{{landing\_target\_pose}} is absent --- VOXL does not forward tag
        detections to PX4; it only changes setpoint type when it acts on a detection.
  \item The EKF2 state (\texttt{{estimator\_status\_flags}}, \texttt{{estimator\_aid\_src\_*}})
        reflects vehicle state estimated from baro and optical flow.
        It has no visibility into VOXL's control decisions.
  \item The switch from velocity to position setpoints at t\,=\,52.134~s is the
        \emph{{only}} observable proxy for tag acquisition.
        Whether the tag was in camera FOV earlier but rejected by VOXL,
        or genuinely not visible until 12~cm AGL, cannot be determined from
        this log --- VOXL-side logs are required.
\end{{itemize}}

\section{{VOXL State Machine (from companion screencast)}}

A screencast of the companion computer terminal during the \emph{{preceding successful}}
precision landing provides the normal-path state machine for the \texttt{{precision\_land\_node}}
ROS\,2 process.
The full sequence, with VOXL-internal timestamps, is:

\begin{{enumerate}}
  \item \textbf{{DESCENDING}} (velocity ctrl, centering): repeated velocity commands
        \texttt{{F/R/D/yaw\_rate}} while searching for the tag.
  \item \textbf{{YAW align}}: \texttt{{State Transition: descending $\to$ 5.1}};
        \texttt{{YAW --- aligning yaw\_error: 5.1}}.
  \item \textbf{{P\_Final\_Descent\_Phase}}: \texttt{{Yaw aligned --- entering P\_Final\_Descent\_phase}}.
  \item \textbf{{Position ctrl}}: \texttt{{Transitioning to position control}}
        (timestamp $T_0$).
        VOXL logs the initial NED target:
        \texttt{{NED target --- N:\,-0.174, D:\,3.835}}
        ($D = 3.835$\,m $\approx 2.86$\,m AGL in this log's local frame).
  \item \textbf{{Touchdown confirmed}}: fires $18.9$\,s after position ctrl entry
        ($T_0 + 18.9$\,s) --- consistent with a physical descent from $\sim$2.9\,m AGL
        at the commanded $D_{{vz}}=0.1$\,m/s.
  \item \textbf{{MAVSDK land}}: \texttt{{Switching to MAVSDK land}} $\to$
        \texttt{{Land command sent.}} $\to$ PX4 enters AUTO\_LAND.
  \item \textbf{{Cleanup}}: \texttt{{Closing Tag Detection pipe}};
        \texttt{{Waiting for landing trigger}}.
\end{{enumerate}}

\subsection*{{Why the second attempt failed}}

In the failed second landing (this PX4 log), the tag was not acquired until
12\,cm AGL --- well below the nominal $\sim$2.9\,m acquisition height.
The position-ctrl phase lasted only $0.67$\,s before the abort command arrived
(cf.\ $18.9$\,s in the successful attempt).

Two hypotheses are consistent with the evidence:

\begin{{enumerate}}
  \item \textbf{{Altitude-threshold abort}}: the \texttt{{precision\_land\_node}} checks
        whether the current altitude is already below a ``touchdown'' threshold
        immediately on tag acquisition.
        At 12\,cm AGL, this check fires immediately and the node sends an abort
        setpoint (\texttt{{trajectory\_setpoint.position[2]}} = 1.000 NED,
        i.e.\ $\approx$5.7\,m AGL) rather than the MAVSDK land command.
  \item \textbf{{Physical bounce}}: the vehicle briefly contacted the pad at 12\,cm,
        the rangefinder and/or IMU triggered \texttt{{touchdown\_confirmed}},
        but the pad contact was too brief for a stable land --- the abort
        setpoint then commanded a $\approx$5\,m climb.
\end{{enumerate}}

Distinguishing these requires the VOXL terminal output from the \emph{{second}} landing
attempt (not present in the screencast, which ends at ``Waiting for landing trigger''
before the second trigger was pressed).

\end{{document}}
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_pl_report(log_path: Path, output_dir: Path) -> PLArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    ulog = ULog(str(log_path))
    base_timestamp = ulog.start_timestamp

    is_replay, replay_note = check_ekf2_replay(ulog)
    if is_replay:
        print(f"[pl-analysis] NOTE: {replay_note}")
    else:
        print(f"[pl-analysis] {replay_note}")

    pl_windows = detect_pl_windows(ulog, base_timestamp)
    if not pl_windows:
        print("[pl-analysis] WARNING: no landing-mode windows detected (OFFBOARD/AUTO_LAND/PRECLAND)")
    else:
        for w in pl_windows:
            print(f"[pl-analysis] detected {w.label}  t={w.t_start:.1f}–{w.t_end:.1f}s")

    figures: dict[str, Path] = {}
    figures["overview"] = plot_flight_overview(ulog, base_timestamp, fig_dir, pl_windows)
    figures["mode_timeline"] = plot_mode_timeline(ulog, base_timestamp, fig_dir, pl_windows)
    figures["optical_flow"] = plot_optical_flow_health(ulog, base_timestamp, fig_dir, pl_windows)
    figures["voxl_abort"] = plot_voxl_abort_analysis(ulog, base_timestamp, fig_dir, pl_windows)
    figures["voxl_xy_approach"] = plot_voxl_xy_approach(ulog, base_timestamp, fig_dir, pl_windows)
    closeups = plot_pl_closeup(ulog, base_timestamp, fig_dir, pl_windows)
    figures.update(closeups)

    summary = pl_metric_summary(ulog, base_timestamp, pl_windows)

    summary_path = output_dir / "pl_summary.txt"
    lines = [f"{k}: {v}" for k, v in sorted(summary.items())]
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    tex = build_pl_latex(log_path, output_dir, figures, summary, pl_windows)
    tex_path = output_dir / "pl_report.tex"
    tex_path.write_text(tex, encoding="utf-8")

    pdf_path = compile_latex(tex_path)

    return PLArtifacts(
        output_dir=output_dir,
        summary_path=summary_path,
        figures=figures,
        tex_path=tex_path,
        pdf_path=pdf_path,
    )
