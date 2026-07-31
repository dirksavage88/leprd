from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
from pyulog import ULog

from .config import ReviewConfig
from .logsource import ARDUPILOT, open_log, resolve_log_type


@dataclass(frozen=True)
class ReviewArtifacts:
    output_dir: Path
    tex_path: Path
    summary_path: Path
    pdf_path: Path
    figures: dict[str, Path]


@dataclass(frozen=True)
class MetricSeries:
    t: np.ndarray
    values: np.ndarray
    source: str


@dataclass(frozen=True)
class FusionStateRow:
    field: str
    meaning: str
    duration_s: float
    percent: float
    spans: str


@dataclass(frozen=True)
class EstimatorExceptionRow:
    topic: str
    field: str
    meaning: str
    duration_s: float
    spans: str


@dataclass(frozen=True)
class ResetEventRow:
    source: str
    time_s: float
    old_count: int
    new_count: int
    delta: str


@dataclass(frozen=True)
class RangefinderConfig:
    has_topic: bool
    topic_summary: str
    configured_hardware: str
    param_summary: str
    report_summary: str


@dataclass(frozen=True)
class ParameterRow:
    prefix: str
    name: str
    value: str


@dataclass(frozen=True)
class NavStateRow:
    value: int
    name: str
    duration_s: float
    spans: str


@dataclass(frozen=True)
class NavStateInfo:
    enum: str
    description: str
    color: str


@dataclass(frozen=True)
class ModeSpan:
    start_s: float
    end_s: float
    value: int
    name: str


_HEIGHT_AID_SOURCES = {
    "baro_vpos": "estimator_aid_src_baro_hgt",
    "rng_vpos": "estimator_aid_src_rng_hgt",
    "gps_vpos": "estimator_aid_src_gnss_hgt",
}

_AGGREGATE_METRIC_TOPICS = {
    "innovation": "estimator_innovations",
    "innovation_variance": "estimator_innovation_variances",
    "test_ratio": "estimator_innovation_test_ratios",
}

_BOOLEAN_PANEL_HEIGHT_RATIO = 0.28
_MODE_SHADING_ALPHA = 0.075

# Copied from PX4 msg/versioned/VehicleStatus.msg (MESSAGE_VERSION = 4).
PX4_NAV_STATES = {
    0: NavStateInfo("NAVIGATION_STATE_MANUAL", "Manual mode", "#4E79A7"),
    1: NavStateInfo("NAVIGATION_STATE_ALTCTL", "Altitude control mode", "#F28E2B"),
    2: NavStateInfo("NAVIGATION_STATE_POSCTL", "Position control mode", "#59A14F"),
    3: NavStateInfo("NAVIGATION_STATE_AUTO_MISSION", "Auto mission mode", "#E15759"),
    4: NavStateInfo("NAVIGATION_STATE_AUTO_LOITER", "Auto loiter mode", "#B07AA1"),
    5: NavStateInfo("NAVIGATION_STATE_AUTO_RTL", "Auto return to launch mode", "#EDC948"),
    6: NavStateInfo("NAVIGATION_STATE_POSITION_SLOW", "Position slow", "#76B7B2"),
    7: NavStateInfo("NAVIGATION_STATE_GUIDED_COURSE", "Guided Course mode (FW: maintain course/alt/speed)", "#FF9DA7"),
    8: NavStateInfo("NAVIGATION_STATE_ALTITUDE_CRUISE", "Altitude with Cruise mode", "#9C755F"),
    9: NavStateInfo("NAVIGATION_STATE_FREE3", "Reserved/free state", "#BAB0AC"),
    10: NavStateInfo("NAVIGATION_STATE_ACRO", "Acro mode", "#1F77B4"),
    11: NavStateInfo("NAVIGATION_STATE_FREE2", "Reserved/free state", "#FF7F0E"),
    12: NavStateInfo("NAVIGATION_STATE_DESCEND", "Descend mode (no position control)", "#2CA02C"),
    13: NavStateInfo("NAVIGATION_STATE_TERMINATION", "Termination mode", "#D62728"),
    14: NavStateInfo("NAVIGATION_STATE_OFFBOARD", "Offboard mode", "#9467BD"),
    15: NavStateInfo("NAVIGATION_STATE_STAB", "Stabilized mode", "#8C564B"),
    16: NavStateInfo("NAVIGATION_STATE_FREE1", "Reserved/free state", "#E377C2"),
    17: NavStateInfo("NAVIGATION_STATE_AUTO_TAKEOFF", "Takeoff", "#7F7F7F"),
    18: NavStateInfo("NAVIGATION_STATE_AUTO_LAND", "Land", "#BCBD22"),
    19: NavStateInfo("NAVIGATION_STATE_AUTO_FOLLOW_TARGET", "Auto Follow", "#17BECF"),
    20: NavStateInfo("NAVIGATION_STATE_AUTO_PRECLAND", "Precision land with landing target", "#AEC7E8"),
    21: NavStateInfo("NAVIGATION_STATE_ORBIT", "Orbit in a circle", "#FFBB78"),
    22: NavStateInfo("NAVIGATION_STATE_AUTO_VTOL_TAKEOFF", "Takeoff, transition, establish loiter", "#98DF8A"),
    23: NavStateInfo("NAVIGATION_STATE_EXTERNAL1", "External mode 1", "#FF9896"),
    24: NavStateInfo("NAVIGATION_STATE_EXTERNAL2", "External mode 2", "#C5B0D5"),
    25: NavStateInfo("NAVIGATION_STATE_EXTERNAL3", "External mode 3", "#C49C94"),
    26: NavStateInfo("NAVIGATION_STATE_EXTERNAL4", "External mode 4", "#F7B6D2"),
    27: NavStateInfo("NAVIGATION_STATE_EXTERNAL5", "External mode 5", "#C7C7C7"),
    28: NavStateInfo("NAVIGATION_STATE_EXTERNAL6", "External mode 6", "#DBDB8D"),
    29: NavStateInfo("NAVIGATION_STATE_EXTERNAL7", "External mode 7", "#9EDAE5"),
    30: NavStateInfo("NAVIGATION_STATE_EXTERNAL8", "External mode 8", "#86BCB6"),
    31: NavStateInfo("NAVIGATION_STATE_MAX", "Sentinel maximum value", "#D4A6C8"),
}
NAV_STATE_NAMES = {value: info.enum for value, info in PX4_NAV_STATES.items()}

_FUSION_STATE_FIELDS = [
    ("cs_gps", "GPS fusion intended"),
    ("cs_gnss_pos", "GNSS position fusion intended"),
    ("cs_gnss_vel", "GNSS velocity fusion intended"),
    ("cs_baro_hgt", "baro data fused"),
    ("cs_gps_hgt", "GPS altitude fused"),
    ("cs_rng_hgt", "range finder height fused"),
    ("cs_mag_hdg", "magnetic yaw heading fusion intended"),
    ("cs_mag_3d", "3-axis magnetometer fusion intended"),
    ("cs_mag_dec", "synthetic magnetic declination fusion intended"),
    ("cs_mag", "3-axis magnetometer mag-state fusion intended"),
    ("cs_opt_flow", "optical flow fusion intended"),
    ("cs_ev_pos", "external vision position fusion intended"),
    ("cs_ev_vel", "external vision velocity fusion intended"),
    ("cs_ev_yaw", "external vision yaw fusion intended"),
    ("cs_ev_hgt", "external vision height fused"),
    ("cs_wind", "wind velocity estimated"),
]

_ESTIMATOR_STATUS_MEANINGS = {
    "filter_fault_flags": "EKF internal faults bitmask",
    "innovation_check_flags": "sensor innovation pass/fail bitmask",
    "timeout_flags": "timeout flags (vel, pos, hgt)",
    "health_flags": "legacy LPE sensor health states (vel, pos, hgt)",
    "gps_check_fail_flags": "GPS check failure bitmask",
}

_PLOTTED_FUSION_FLAG_FIELDS = [
    "cs_gps",
    "cs_gnss_vel",
    "cs_gps_hgt",
    "cs_baro_hgt",
    "cs_rng_hgt",
    "cs_rng_kin_consistent",
    "cs_opt_flow",
    "cs_gnd_effect",
    "cs_in_air",
    "cs_vehicle_at_rest",
    "cs_inertial_dead_reckoning",
    "fs_bad_acc_vertical",
    "reject_hagl",
    "reject_optflow_x",
    "reject_optflow_y",
]

_RESET_COUNTER_FIELDS = [
    "reset_count_vel_ne",
    "reset_count_vel_d",
    "reset_count_pos_ne",
    "reset_count_pod_d",
    "reset_count_quat",
]

_STATUS_MASK_FIELDS = [
    "control_mode_flags",
    "filter_fault_flags",
    "innovation_check_flags",
    "gps_check_fail_flags",
    "solution_status_flags",
    "health_flags",
    "timeout_flags",
]

# Flight Review decodes estimator_status.innovation_check_flags with this grouping.
_INNOVATION_CHECK_FLAG_GROUPS = [
    ("Velocity Check Bit", 0, 0x1),
    ("Horizontal Position Check Bit", 1, 0x1),
    ("Vertical Position Check Bit", 2, 0x1),
    ("Mag X, Y, Z Check Bits", 3, 0x7),
    ("Yaw Check Bit", 6, 0x1),
    ("Airspeed Check Bit", 7, 0x1),
    ("Synthetic Sideslip Check Bit", 8, 0x1),
    ("Height to Ground Check Bit", 9, 0x1),
    ("Optical Flow X, Y Check Bits", 10, 0x3),
]

_ESTIMATOR_EVENT_FLAG_FIELDS = [
    "gps_checks_passed",
    "reset_vel_to_gps",
    "reset_vel_to_flow",
    "reset_vel_to_vision",
    "reset_vel_to_zero",
    "reset_pos_to_last_known",
    "reset_pos_to_gps",
    "reset_pos_to_vision",
    "starting_gps_fusion",
    "starting_vision_pos_fusion",
    "starting_vision_vel_fusion",
    "starting_vision_yaw_fusion",
    "yaw_aligned_to_imu_gps",
    "reset_hgt_to_baro",
    "reset_hgt_to_gps",
    "reset_hgt_to_rng",
    "reset_hgt_to_ev",
    "gps_quality_poor",
    "gps_fusion_timout",
    "gps_data_stopped",
    "gps_data_stopped_using_alternate",
    "height_sensor_timeout",
    "stopping_navigation",
    "invalid_accel_bias_cov_reset",
    "bad_yaw_using_gps_course",
    "stopping_mag_use",
    "vision_data_stopped",
    "emergency_yaw_reset_mag_stopped",
    "emergency_yaw_reset_gps_yaw_stopped",
]

_ESTIMATOR_STATUS_FLAG_MEANINGS = {
    "cs_mag_field_disturbed": "mag field does not match expected strength",
    "cs_mag_fault": "magnetometer declared faulty and no longer used",
    "cs_gnss_yaw_fault": "GNSS heading declared faulty and no longer used",
    "cs_gps_yaw_fault": "GPS heading declared faulty and no longer used",
    "cs_rng_fault": "range finder declared faulty and no longer used",
    "cs_inertial_dead_reckoning": "no measurements constraining horizontal velocity drift",
    "cs_wind_dead_reckoning": "navigation reliant on wind-relative measurements",
    "cs_gnss_fault": "GNSS measurements declared faulty",
    "cs_gnss_hgt_fault": "GNSS altitude measurements declared faulty",
    "fs_bad_mag_x": "magnetometer X-axis fusion numerical error",
    "fs_bad_mag_y": "magnetometer Y-axis fusion numerical error",
    "fs_bad_mag_z": "magnetometer Z-axis fusion numerical error",
    "fs_bad_hdg": "heading fusion numerical error",
    "fs_bad_mag_decl": "magnetic declination fusion numerical error",
    "fs_bad_airspeed": "airspeed fusion numerical error",
    "fs_bad_sideslip": "synthetic sideslip fusion numerical error",
    "fs_bad_optflow_x": "optical flow X-axis fusion numerical error",
    "fs_bad_optflow_y": "optical flow Y-axis fusion numerical error",
    "fs_bad_vel_n": "North velocity fusion numerical error",
    "fs_bad_vel_e": "East velocity fusion numerical error",
    "fs_bad_vel_d": "Down velocity fusion numerical error",
    "fs_bad_pos_n": "North position fusion numerical error",
    "fs_bad_pos_e": "East position fusion numerical error",
    "fs_bad_pos_d": "Down position fusion numerical error",
    "fs_bad_acc_bias": "bad delta velocity bias estimates detected",
    "fs_bad_acc_vertical": "bad vertical accelerometer data detected",
    "fs_bad_acc_clipping": "delta velocity data contains clipping",
    "reject_hor_vel": "horizontal velocity innovation rejected",
    "reject_ver_vel": "vertical velocity innovation rejected",
    "reject_hor_pos": "horizontal position innovation rejected",
    "reject_ver_pos": "vertical position innovation rejected",
    "reject_yaw": "yaw innovation rejected",
    "reject_airspeed": "airspeed innovation rejected",
    "reject_sideslip": "sideslip innovation rejected",
    "reject_hagl": "height-above-ground innovation rejected",
    "reject_optflow_x": "optical-flow X innovation rejected",
    "reject_optflow_y": "optical-flow Y innovation rejected",
}

_ESTIMATOR_EVENT_WARNING_MEANINGS = {
    "gps_quality_poor": "GPS quality poor",
    "gps_fusion_timout": "GPS fusion timeout",
    "gps_data_stopped": "GPS data stopped",
    "gps_data_stopped_using_alternate": "GPS data stopped; using alternate",
    "height_sensor_timeout": "height sensor timeout",
    "stopping_navigation": "stopping navigation",
    "invalid_accel_bias_cov_reset": "invalid accel bias covariance reset",
    "bad_yaw_using_gps_course": "bad yaw using GPS course",
    "stopping_mag_use": "stopping mag use",
    "vision_data_stopped": "vision data stopped",
    "emergency_yaw_reset_mag_stopped": "emergency yaw reset; mag stopped",
    "emergency_yaw_reset_gps_yaw_stopped": "emergency yaw reset; GPS yaw stopped",
}

_GPS_CHECK_FAIL_MEANINGS = {
    "check_fail_gps_fix": "insufficient fix type",
    "check_fail_min_sat_count": "minimum required satellite count fail",
    "check_fail_max_pdop": "maximum allowed PDOP fail",
    "check_fail_max_horz_err": "maximum allowed horizontal position error fail",
    "check_fail_max_vert_err": "maximum allowed vertical position error fail",
    "check_fail_max_spd_err": "maximum allowed speed error fail",
    "check_fail_max_horz_drift": "maximum allowed horizontal position drift fail",
    "check_fail_max_vert_drift": "maximum allowed vertical position drift fail",
    "check_fail_max_horz_spd_err": "maximum allowed horizontal speed fail",
    "check_fail_max_vert_spd_err": "maximum allowed vertical velocity discrepancy fail",
}

_RANGEFINDER_ENABLE_PARAMS = {
    "SENS_EN_LL40LS": "Lidar-Lite / LL40LS",
    "SENS_EN_SF0X": "Lightware SF0X family",
    "SENS_EN_SF1XX": "Lightware SF1XX family",
    "SENS_EN_TF02PRO": "Benewake TF02-Pro",
    "SENS_EN_TFMINI": "Benewake TFmini",
    "SENS_EN_VL53L0X": "VL53L0X",
    "SENS_EN_VL53L1X": "VL53L1X",
    "SENS_ULAND_CFG": "uLanding",
    "UAVCAN_SUB_RNG": "UAVCAN rangefinder subscription",
}

_RANGEFINDER_DETAIL_PARAMS = [
    "SYS_HAS_NUM_DIST",
    "EKF2_RNG_CTRL",
    "EKF2_HGT_REF",
    "SENS_SF0X_CFG",
    "SENS_TFMINI_CFG",
    "SENS_TFMINI_HW",
    "SF1XX_MODE",
    "SF1XX_ROT",
    "UAVCAN_RNG_MIN",
    "UAVCAN_RNG_MAX",
]

_DISTANCE_SENSOR_TYPES = {
    0: "laser",
    1: "ultrasound",
    2: "infrared",
    3: "radar",
    4: "unknown",
}

_DISTANCE_SENSOR_ORIENTATIONS = {
    24: "upward-facing",
    25: "downward-facing",
}

_MPC_ALT_MODE_VALUES = {
    0: (
        "Altitude following",
        "Controls height relative to the local earth-frame origin. This is not terrain following.",
    ),
    1: (
        "Terrain following",
        "Controls height relative to ground using the EKF distance-to-ground estimate when valid.",
    ),
    2: (
        "Terrain hold",
        "Uses ground-relative height while horizontally stationary, then switches back to earth-frame altitude while moving.",
    ),
}

def get_ds(ulog: ULog, name: str, multi_id: int = 0):
    matches = [d for d in ulog.data_list if d.name == name and d.multi_id == multi_id]
    return matches[0] if matches else None


def t_rel(ds, base_timestamp: int) -> np.ndarray:
    return (np.asarray(ds.data["timestamp"], dtype=np.int64) - int(base_timestamp)) / 1e6


def log_start_utc_us(ulog: ULog, base_timestamp: int) -> int | None:
    info = getattr(ulog, "msg_info_dict", {})
    if not isinstance(info, dict):
        return None

    boot_time = info.get("boot_time_utc_us")
    if boot_time in (None, 0, "0"):
        return None

    try:
        return int(boot_time) + int(base_timestamp)
    except (TypeError, ValueError):
        return None


def system_time_note(ulog: ULog, base_timestamp: int) -> str:
    start_utc_us = log_start_utc_us(ulog, base_timestamp)
    if start_utc_us is None:
        return "UTC system time unavailable: ULog metadata does not contain boot_time_utc_us."
    start_time = datetime.fromtimestamp(start_utc_us / 1e6, timezone.utc)
    return f"UTC system time available from boot_time_utc_us; log start {start_time.isoformat()}."


def add_system_time_axis(ax, ulog: ULog, base_timestamp: int) -> bool:
    start_utc_us = log_start_utc_us(ulog, base_timestamp)
    if start_utc_us is None:
        return False

    ticks = ax.get_xticks()
    lo, hi = ax.get_xlim()
    ticks = [float(tick) for tick in ticks if lo <= tick <= hi]
    if not ticks:
        return False

    datetimes = [
        datetime.fromtimestamp((start_utc_us + int(round(tick * 1e6))) / 1e6, timezone.utc)
        for tick in ticks
    ]
    labels = [dt.strftime("%H:%M:%S") for dt in datetimes]
    date_label = datetimes[0].strftime("%Y-%m-%d UTC")

    top = ax.twiny()
    top.set_xlim(ax.get_xlim())
    top.set_xticks(ticks)
    top.set_xticklabels(labels, fontsize=8)
    top.set_xlabel(f"system time ({date_label})")
    return True


def arr(ds, field: str, dtype=float) -> np.ndarray:
    return np.asarray(ds.data[field], dtype=dtype)


def get_metric_series(ulog: ULog, base_timestamp: int, key: str, metric: str) -> MetricSeries | None:
    """Return a scalar EKF innovation metric.

    Prefer newer estimator_aid_src_* topics for scalar height sources, then fall back
    to the aggregate estimator_innovations-style topics used by older/replay logs.
    """
    aid_topic = _HEIGHT_AID_SOURCES.get(key)
    if aid_topic is not None:
        aid = get_ds(ulog, aid_topic)
        if aid is not None and metric in aid.data:
            return MetricSeries(t_rel(aid, base_timestamp), arr(aid, metric), f"{aid_topic}.{metric}")

    aggregate_topic = _AGGREGATE_METRIC_TOPICS[metric]
    aggregate = get_ds(ulog, aggregate_topic)
    if aggregate is not None and key in aggregate.data:
        return MetricSeries(
            t_rel(aggregate, base_timestamp),
            arr(aggregate, key),
            f"{aggregate_topic}.{key}",
        )

    return None


def gps_alt_m(ds) -> np.ndarray:
    # newer PX4: float metres; older PX4 (≤1.14 ModalAI): int32 millimetres
    if "altitude_msl_m" in ds.data:
        return np.asarray(ds.data["altitude_msl_m"], dtype=float)
    return np.asarray(ds.data["alt"], dtype=float) * 1e-3


def initial_parameters(ulog: ULog) -> dict[str, object]:
    params = getattr(ulog, "initial_parameters", None)
    return params if isinstance(params, dict) else {}


def format_param_value(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(numeric - round(numeric)) < 1e-6:
        return str(int(round(numeric)))
    return f"{numeric:.3g}"


def param_enabled(value: object) -> bool:
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return False


def unique_int_values(ds, field: str) -> list[int]:
    if field not in ds.data:
        return []
    values = np.asarray(ds.data[field])
    if values.size == 0:
        return []
    finite = values[np.isfinite(values.astype(float))]
    return sorted({int(v) for v in finite})


def unique_float_values(ds, field: str) -> list[float]:
    if field not in ds.data:
        return []
    values = np.asarray(ds.data[field], dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return []
    return sorted({round(float(v), 3) for v in finite})


def describe_distance_sensor_topic(ulog: ULog) -> tuple[bool, str]:
    sensors = [d for d in ulog.data_list if d.name == "distance_sensor"]
    if not sensors:
        return False, "no distance_sensor topic"

    parts: list[str] = []
    for ds in sorted(sensors, key=lambda item: item.multi_id):
        fields: list[str] = [f"multi_id {ds.multi_id}"]
        type_values = unique_int_values(ds, "type")
        if type_values:
            types = ", ".join(_DISTANCE_SENSOR_TYPES.get(v, f"type {v}") for v in type_values)
            fields.append(f"type {types}")
        orientation_values = unique_int_values(ds, "orientation")
        if orientation_values:
            orientations = ", ".join(
                _DISTANCE_SENSOR_ORIENTATIONS.get(v, f"orientation {v}") for v in orientation_values
            )
            fields.append(orientations)
        min_values = unique_float_values(ds, "min_distance")
        max_values = unique_float_values(ds, "max_distance")
        if min_values or max_values:
            min_text = "/".join(f"{v:g}" for v in min_values) if min_values else "?"
            max_text = "/".join(f"{v:g}" for v in max_values) if max_values else "?"
            fields.append(f"range {min_text}-{max_text} m")
        device_ids = unique_int_values(ds, "device_id")
        if device_ids:
            fields.append("device_id " + "/".join(str(v) for v in device_ids[:4]))
        parts.append("; ".join(fields))

    return True, "distance_sensor logged: " + " | ".join(parts)


def describe_rangefinder_config(ulog: ULog) -> RangefinderConfig:
    params = initial_parameters(ulog)
    has_topic, topic_summary = describe_distance_sensor_topic(ulog)

    enabled = [
        f"{label} ({name}={format_param_value(params[name])})"
        for name, label in _RANGEFINDER_ENABLE_PARAMS.items()
        if name in params and param_enabled(params[name])
    ]
    configured_hardware = ", ".join(enabled) if enabled else "no enabled rangefinder driver params found"

    details = [
        f"{name}={format_param_value(params[name])}"
        for name in _RANGEFINDER_DETAIL_PARAMS
        if name in params
    ]
    param_summary = "; ".join(details) if details else "no rangefinder-specific params found"

    if has_topic:
        report_summary = f"{topic_summary}; configured hardware: {configured_hardware}; params: {param_summary}"
    else:
        report_summary = f"{topic_summary}; configured hardware: {configured_hardware}; params: {param_summary}"

    return RangefinderConfig(has_topic, topic_summary, configured_hardware, param_summary, report_summary)


def describe_mpc_alt_mode(ulog: ULog) -> dict[str, str]:
    params = initial_parameters(ulog)
    if "MPC_ALT_MODE" not in params:
        return {
            "mpc_alt_mode": "not logged",
            "mpc_alt_mode_label": "unknown",
            "mpc_alt_mode_detail": "MPC_ALT_MODE was not present in ULog initial_parameters.",
            "mpc_alt_mode_related_params": "n/a",
        }

    try:
        value = int(round(float(params["MPC_ALT_MODE"])))
    except (TypeError, ValueError):
        value = -1

    label, detail = _MPC_ALT_MODE_VALUES.get(
        value,
        ("unknown", "Unrecognized MPC_ALT_MODE value in this log."),
    )

    related_names = ["MPC_HOLD_MAX_XY", "MPC_HOLD_MAX_Z", "EKF2_HGT_REF", "EKF2_RNG_CTRL"]
    related = [
        f"{name}={format_param_value(params[name])}"
        for name in related_names
        if name in params
    ]

    return {
        "mpc_alt_mode": str(value),
        "mpc_alt_mode_label": label,
        "mpc_alt_mode_detail": detail,
        "mpc_alt_mode_related_params": "; ".join(related) if related else "none logged",
    }


def gps_disabled_label(ulog: ULog) -> str | None:
    params = initial_parameters(ulog)
    for name in ("SYS_HAS_GPS", "HAS_GPS"):
        if name not in params:
            continue
        try:
            if int(round(float(params[name]))) == 0:
                return f"{name}=0"
        except (TypeError, ValueError):
            continue
    return None


def logged_parameter_rows(ulog: ULog, prefixes: tuple[str, ...] = ("EKF2_", "MPC_")) -> list[ParameterRow]:
    rows: list[ParameterRow] = []
    for name, value in sorted(initial_parameters(ulog).items()):
        prefix = next((item for item in prefixes if name.startswith(item)), None)
        if prefix is None:
            continue
        rows.append(ParameterRow(prefix=prefix.rstrip("_"), name=name, value=format_param_value(value)))
    return rows


def nav_state_name(value: int) -> str:
    return NAV_STATE_NAMES.get(value, f"UNKNOWN_{value}")


def nav_state_color(value: int) -> str:
    info = PX4_NAV_STATES.get(value)
    return info.color if info is not None else "#BDBDBD"


def nav_state_spans(ulog: ULog, base_timestamp: int) -> list[ModeSpan]:
    status = get_ds(ulog, "vehicle_status")
    if status is None or "nav_state" not in status.data:
        return []

    t = t_rel(status, base_timestamp)
    nav = arr(status, "nav_state", int)
    if len(t) == 0 or len(nav) == 0:
        return []

    count = min(len(t), len(nav))
    t = t[:count]
    nav = nav[:count]
    spans: list[ModeSpan] = []
    start_idx = 0
    for idx in range(1, count):
        if int(nav[idx]) == int(nav[start_idx]):
            continue
        value = int(nav[start_idx])
        spans.append(ModeSpan(float(t[start_idx]), float(t[idx]), value, nav_state_name(value)))
        start_idx = idx

    value = int(nav[start_idx])
    spans.append(ModeSpan(float(t[start_idx]), float(t[-1]), value, nav_state_name(value)))
    return spans


def shade_mode_background(ax_or_axes, ulog: ULog, base_timestamp: int, enabled: bool = True):
    if not enabled:
        return

    spans = nav_state_spans(ulog, base_timestamp)
    if not spans:
        return

    axes = np.ravel(np.atleast_1d(ax_or_axes))
    for ax in axes:
        has_data = ax.has_data()
        xlim = ax.get_xlim()
        if not has_data:
            xlim = (spans[0].start_s, spans[-1].end_s)
        for span in spans:
            start = max(span.start_s, xlim[0])
            end = min(span.end_s, xlim[1])
            if end <= start:
                continue
            ax.axvspan(start, end, color=nav_state_color(span.value), alpha=_MODE_SHADING_ALPHA, lw=0, zorder=0)
        ax.set_xlim(xlim)


def logged_nav_state_rows(ulog: ULog, base_timestamp: int) -> list[NavStateRow]:
    status = get_ds(ulog, "vehicle_status")
    if status is None or "nav_state" not in status.data:
        return []

    t = t_rel(status, base_timestamp)
    nav = arr(status, "nav_state", int)
    rows: list[NavStateRow] = []
    for value in sorted({int(item) for item in nav}):
        mask = nav == value
        rows.append(
            NavStateRow(
                value=value,
                name=nav_state_name(value),
                duration_s=true_duration(t, mask),
                spans=format_spans(bool_spans(t, mask), limit=4),
            )
        )
    return rows


def interp_step_previous(src_t: np.ndarray, src_y: np.ndarray, dst_t: np.ndarray) -> np.ndarray:
    src_t = np.asarray(src_t, dtype=float)
    src_y = np.asarray(src_y, dtype=float)
    dst_t = np.asarray(dst_t, dtype=float)
    finite = np.isfinite(src_t) & np.isfinite(src_y)
    if finite.sum() == 0:
        return np.full_like(dst_t, np.nan, dtype=float)

    src_t = src_t[finite]
    src_y = src_y[finite]
    order = np.argsort(src_t)
    src_t = src_t[order]
    src_y = src_y[order]

    idx = np.searchsorted(src_t, dst_t, side="right") - 1
    out = np.full_like(dst_t, np.nan, dtype=float)
    valid = idx >= 0
    out[valid] = src_y[idx[valid]]
    return out


def terrain_estimate_summary(ulog: ULog, base_timestamp: int) -> dict[str, str]:
    lpos = get_ds(ulog, "vehicle_local_position")
    flags = get_ds(ulog, "estimator_status_flags")
    out: dict[str, str] = {}

    if lpos is None or "dist_bottom" not in lpos.data:
        out["terrain_dist_bottom"] = "not logged"
        return out

    t = t_rel(lpos, base_timestamp)
    dist_bottom = arr(lpos, "dist_bottom")
    finite = dist_bottom[np.isfinite(dist_bottom)]
    if finite.size:
        out["terrain_dist_bottom"] = f"{np.nanmin(finite):.2f} to {np.nanmax(finite):.2f} m"

    if "dist_bottom_valid" in lpos.data:
        valid = arr(lpos, "dist_bottom_valid", int).astype(bool)
        out["terrain_dist_bottom_valid"] = format_spans(bool_spans(t, valid), limit=4)
    else:
        out["terrain_dist_bottom_valid"] = "not logged"

    if "dist_bottom_reset_counter" in lpos.data:
        reset_count = arr(lpos, "dist_bottom_reset_counter", int)
        out["terrain_dist_bottom_reset_counter"] = f"{reset_count[0]}--{reset_count[-1]}"

    if "dist_bottom_sensor_bitfield" in lpos.data:
        bitfield = arr(lpos, "dist_bottom_sensor_bitfield", int)
        observed = ", ".join(str(value) for value in sorted({int(value) for value in bitfield}))
        out["terrain_dist_bottom_sensor_bitfield"] = observed

    if flags is not None:
        t_flags = t_rel(flags, base_timestamp)
        for key in ["cs_rng_terrain", "cs_opt_flow_terrain", "cs_rng_hgt", "cs_rng_kin_consistent"]:
            if key in flags.data:
                values = arr(flags, key, int).astype(bool)
                out[key] = format_spans(bool_spans(t_flags, values), limit=4)

    return out


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


def exceedance_start_times(t: np.ndarray, values: np.ndarray, threshold: float = 1.0) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    t = np.asarray(t, dtype=float)
    if values.size == 0 or t.size == 0:
        return np.asarray([], dtype=float)

    count = min(values.size, t.size)
    values = values[:count]
    t = t[:count]
    exceeded = np.isfinite(values) & (values > threshold)
    if exceeded.size == 0:
        return np.asarray([], dtype=float)

    starts = exceeded & np.concatenate(([True], ~exceeded[:-1]))
    return t[starts]


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


def sample_durations(t: np.ndarray) -> np.ndarray:
    if len(t) == 0:
        return np.array([], dtype=float)
    if len(t) == 1:
        return np.array([0.0], dtype=float)
    durations = np.diff(t, append=t[-1])
    durations[-1] = np.median(durations[:-1])
    return durations.astype(float)


def true_duration(t: np.ndarray, values: np.ndarray) -> float:
    vals = np.asarray(values).astype(bool)
    if len(t) != len(vals):
        raise ValueError("time and value arrays must be the same length")
    return float(np.sum(sample_durations(t)[vals]))


def counter_step_indices(values: np.ndarray) -> np.ndarray:
    counters = np.asarray(values, dtype=int)
    if counters.size == 0:
        return np.array([], dtype=int)
    return np.where(np.diff(counters, prepend=counters[0]) > 0)[0]


def quat_delta_text(ds, index: int) -> str:
    fields = [f"delta_q_reset[{idx}]" for idx in range(4)]
    if not all(field in ds.data for field in fields):
        return "delta quaternion not logged"

    quat = np.array([float(ds.data[field][index]) for field in fields], dtype=float)
    norm = float(np.linalg.norm(quat))
    if norm <= 0.0:
        return "delta quaternion zero"

    quat /= norm
    angle_deg = np.degrees(2.0 * np.arccos(np.clip(abs(quat[0]), -1.0, 1.0)))
    yaw_deg = np.degrees(
        np.arctan2(
            2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
            1.0 - 2.0 * (quat[2] ** 2 + quat[3] ** 2),
        )
    )
    return f"{angle_deg:.2f} deg quaternion, {yaw_deg:.2f} deg yaw"


def resampled_armed_mask(ulog: ULog, base_timestamp: int, t: np.ndarray) -> np.ndarray:
    status = get_ds(ulog, "vehicle_status")
    if status is None or "arming_state" not in status.data:
        return np.ones(len(t), dtype=bool)

    ts = t_rel(status, base_timestamp)
    armed = arr(status, "arming_state", int) == 2
    if len(ts) == 0:
        return np.ones(len(t), dtype=bool)

    indices = np.searchsorted(ts, t, side="right") - 1
    indices = np.clip(indices, 0, len(ts) - 1)
    return armed[indices]


def field_duration_row(
    ulog: ULog,
    base_timestamp: int,
    ds,
    field: str,
) -> tuple[float, float, str] | None:
    if ds is None or field not in ds.data:
        return None

    t = t_rel(ds, base_timestamp)
    armed = resampled_armed_mask(ulog, base_timestamp, t)
    armed_duration = true_duration(t, armed)
    values = arr(ds, field, int).astype(bool) & armed
    duration = true_duration(t, values)
    percent = 0.0 if armed_duration <= 0.0 else 100.0 * duration / armed_duration
    spans = format_spans(bool_spans(t, values), limit=4)
    return duration, percent, spans


def fields_or_duration_row(
    ulog: ULog,
    base_timestamp: int,
    ds,
    fields: list[str],
) -> tuple[float, float, str] | None:
    if ds is None or not fields or not all(field in ds.data for field in fields):
        return None

    t = t_rel(ds, base_timestamp)
    armed = resampled_armed_mask(ulog, base_timestamp, t)
    armed_duration = true_duration(t, armed)
    values = np.zeros(len(t), dtype=bool)
    for field in fields:
        values |= arr(ds, field, int).astype(bool)
    values &= armed

    duration = true_duration(t, values)
    percent = 0.0 if armed_duration <= 0.0 else 100.0 * duration / armed_duration
    spans = format_spans(bool_spans(t, values), limit=4)
    return duration, percent, spans


def active_fusion_state_rows(ulog: ULog, base_timestamp: int) -> list[FusionStateRow]:
    flags = get_ds(ulog, "estimator_status_flags")
    rows: list[FusionStateRow] = []
    if flags is None:
        return rows

    mag_aiding = fields_or_duration_row(ulog, base_timestamp, flags, ["cs_mag_hdg", "cs_mag_3d"])
    if mag_aiding is not None:
        duration, percent, spans = mag_aiding
        if duration > 0.0:
            rows.append(
                FusionStateRow(
                    "cs_mag_hdg_or_3d",
                    "magnetic yaw aiding via heading or 3-axis fusion",
                    duration,
                    percent,
                    spans,
                )
            )

    for field, meaning in _FUSION_STATE_FIELDS:
        result = field_duration_row(ulog, base_timestamp, flags, field)
        if result is None:
            continue
        duration, percent, spans = result
        if duration > 0.0:
            rows.append(FusionStateRow(field, meaning, duration, percent, spans))
    return rows


def reset_event_rows(ulog: ULog, base_timestamp: int) -> list[ResetEventRow]:
    rows: list[ResetEventRow] = []

    attitude = get_ds(ulog, "vehicle_attitude")
    if attitude is not None and "quat_reset_counter" in attitude.data:
        t = t_rel(attitude, base_timestamp)
        counter = arr(attitude, "quat_reset_counter", int)
        for index in counter_step_indices(counter):
            rows.append(
                ResetEventRow(
                    "vehicle_attitude.quat_reset_counter",
                    float(t[index]),
                    int(counter[index - 1]) if index > 0 else int(counter[index]),
                    int(counter[index]),
                    quat_delta_text(attitude, int(index)),
                )
            )

    local_position = get_ds(ulog, "vehicle_local_position")
    if local_position is not None and "heading_reset_counter" in local_position.data:
        t = t_rel(local_position, base_timestamp)
        counter = arr(local_position, "heading_reset_counter", int)
        for index in counter_step_indices(counter):
            if "delta_heading" in local_position.data:
                delta = f"{np.degrees(float(local_position.data['delta_heading'][index])):.2f} deg heading"
            else:
                delta = "delta heading not logged"
            rows.append(
                ResetEventRow(
                    "vehicle_local_position.heading_reset_counter",
                    float(t[index]),
                    int(counter[index - 1]) if index > 0 else int(counter[index]),
                    int(counter[index]),
                    delta,
                )
            )

    return sorted(rows, key=lambda row: row.time_s)


def estimator_exception_rows(ulog: ULog, base_timestamp: int) -> list[EstimatorExceptionRow]:
    rows: list[EstimatorExceptionRow] = []

    status = get_ds(ulog, "estimator_status")
    if status is not None:
        for field, meaning in _ESTIMATOR_STATUS_MEANINGS.items():
            result = field_duration_row(ulog, base_timestamp, status, field)
            if result is None:
                continue
            duration, _percent, spans = result
            if duration > 0.0:
                rows.append(EstimatorExceptionRow("estimator_status", field, meaning, duration, spans))

    flags = get_ds(ulog, "estimator_status_flags")
    if flags is not None:
        for field in flags.data:
            if not (
                field.startswith("fs_")
                or field.startswith("reject_")
                or field in _ESTIMATOR_STATUS_FLAG_MEANINGS
            ):
                continue
            meaning = _ESTIMATOR_STATUS_FLAG_MEANINGS.get(field, field.replace("_", " "))
            result = field_duration_row(ulog, base_timestamp, flags, field)
            if result is None:
                continue
            duration, _percent, spans = result
            if duration > 0.0:
                rows.append(EstimatorExceptionRow("estimator_status_flags", field, meaning, duration, spans))

    events = get_ds(ulog, "estimator_event_flags")
    if events is not None:
        for field, meaning in _ESTIMATOR_EVENT_WARNING_MEANINGS.items():
            result = field_duration_row(ulog, base_timestamp, events, field)
            if result is None:
                continue
            duration, _percent, spans = result
            if duration > 0.0:
                rows.append(EstimatorExceptionRow("estimator_event_flags", field, meaning, duration, spans))

    gps_status = get_ds(ulog, "estimator_gps_status")
    if gps_status is not None:
        for field, meaning in _GPS_CHECK_FAIL_MEANINGS.items():
            result = field_duration_row(ulog, base_timestamp, gps_status, field)
            if result is None:
                continue
            duration, _percent, spans = result
            if duration > 0.0:
                rows.append(EstimatorExceptionRow("estimator_gps_status", field, meaning, duration, spans))

    return rows


def save_fig(fig, fig_dir: Path, name: str) -> Path:
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


def setup_boolean_axis(ax, title: str):
    setup_axis(ax, title, "0/1")
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 1])


def setup_integer_value_axis(ax, values: np.ndarray):
    finite = np.asarray(values)[np.isfinite(values)]
    if not finite.size:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
        return

    y_min = int(np.nanmin(finite))
    y_max = int(np.nanmax(finite))
    if y_min == y_max:
        ax.set_ylim(y_min - 0.5, y_max + 0.5)
        ax.set_yticks([y_min])
        return

    ax.set_ylim(y_min - 0.15, y_max + 0.15)
    if y_max - y_min <= 6:
        ax.set_yticks(np.arange(y_min, y_max + 1))
    else:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))


def available_fields(ds, candidate_fields: list[str]) -> list[str]:
    if ds is None:
        return []
    return [field for field in candidate_fields if field in ds.data]


def innovation_check_flag_group_values(values: np.ndarray, bit_shift: int, mask: int) -> np.ndarray:
    return (np.asarray(values, dtype=np.uint64) >> int(bit_shift)) & int(mask)


def get_thrust(ulog: ULog, base_timestamp: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (t, thrust_magnitude) from vehicle_thrust_setpoint or actuator_controls_0."""
    thr = get_ds(ulog, "vehicle_thrust_setpoint")
    if thr is not None and "xyz[2]" in thr.data:
        t = t_rel(thr, base_timestamp)
        # xyz[2] is negative for upward thrust in NED; take absolute value
        return t, np.abs(arr(thr, "xyz[2]"))

    actuator = get_ds(ulog, "actuator_controls_0")
    if actuator is not None and "control[3]" in actuator.data:
        t = t_rel(actuator, base_timestamp)
        return t, arr(actuator, "control[3]")

    return None


def get_range(ulog: ULog, base_timestamp: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (t, range_m) from distance_sensor, filtered to valid readings.

    signal_quality == 0 is explicitly bad; -1 means 'not provided'.
    Both are valid for the purpose of excluding explicit hardware failures.
    """
    dist = get_ds(ulog, "distance_sensor")
    if dist is None:
        return None

    t = t_rel(dist, base_timestamp)
    current_distance = arr(dist, "current_distance")

    if "signal_quality" in dist.data:
        sig = arr(dist, "signal_quality", dtype=int)
        valid = sig != 0  # exclude only explicit bad (0); -1 (not provided) is ok
    else:
        valid = np.ones(len(t), dtype=bool)

    if valid.sum() < 2:
        return None

    return t[valid], current_distance[valid]


def compute_baro_range_divergence(
    ulog: ULog,
    base_timestamp: int,
    align_window_s: float = 30.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    """Return (t_baro, baro_alt_m, aligned_divergence, offset).

    divergence = (baro_alt - range_agl_interpolated) - median_offset_at_start
    Uses distance_sensor as AGL truth. The offset absorbs the MSL-to-AGL
    reference difference so only dynamic baro error relative to the laser is shown.
    Returns None when rangefinder data is unavailable.
    """
    range_result = get_range(ulog, base_timestamp)
    air = get_ds(ulog, "vehicle_air_data")
    if range_result is None or air is None:
        return None

    t_range, range_m = range_result
    t_baro = t_rel(air, base_timestamp)
    baro_alt = arr(air, "baro_alt_meter")

    range_i = interp_at(t_range, range_m, t_baro)

    raw_diff = baro_alt - range_i
    early = (t_baro >= 0) & (t_baro <= align_window_s) & np.isfinite(raw_diff)
    if early.sum() < 2:
        early = np.isfinite(raw_diff)
    offset = float(np.median(raw_diff[early])) if early.sum() >= 1 else 0.0

    return t_baro, baro_alt, raw_diff - offset, offset


def compute_baro_gps_divergence(
    ulog: ULog,
    base_timestamp: int,
    align_window_s: float = 30.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    """Return baro divergence against GPS altitude as a legacy fallback.

    Rangefinder data is the preferred altitude truth source. GPS altitude is only
    useful as a rough diagnostic fallback and is not used for propwash scoring.
    """
    air = get_ds(ulog, "vehicle_air_data")
    gps = get_ds(ulog, "vehicle_gps_position")
    if air is None or gps is None:
        return None

    t_baro = t_rel(air, base_timestamp)
    t_gps = t_rel(gps, base_timestamp)
    baro_alt = arr(air, "baro_alt_meter")
    gps_alt = gps_alt_m(gps)
    gps_i = interp_at(t_gps, gps_alt, t_baro)

    raw_diff = baro_alt - gps_i
    early = (t_baro >= 0) & (t_baro <= align_window_s) & np.isfinite(raw_diff)
    if early.sum() < 2:
        early = np.isfinite(raw_diff)
    offset = float(np.median(raw_diff[early])) if early.sum() >= 1 else 0.0

    return t_baro, baro_alt, raw_diff - offset, offset


def detect_propwash_events(
    ulog: ULog,
    base_timestamp: int,
    divergence_threshold_m: float = 0.5,
    throttle_threshold: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return (t, propwash_mask, divergence) on the baro time axis.

    propwash = |aligned_divergence| > threshold AND thrust > threshold.
    Requires rangefinder-based divergence. GPS altitude is not used as truth.
    """
    result = compute_baro_range_divergence(ulog, base_timestamp)
    if result is None:
        return None

    t_baro, _baro_alt, divergence, _offset = result

    thrust_result = get_thrust(ulog, base_timestamp)
    if thrust_result is not None:
        t_thr, thrust = thrust_result
        thrust_i = interp_at(t_thr, thrust, t_baro)
        throttle_mask = thrust_i > throttle_threshold
    else:
        throttle_mask = np.ones(len(t_baro), dtype=bool)

    propwash = (np.abs(divergence) > divergence_threshold_m) & throttle_mask & np.isfinite(divergence)
    return t_baro, propwash, divergence


def _hover_mask(
    ulog: ULog,
    base_timestamp: int,
    t_ref: np.ndarray,
    max_speed_ms: float = 1.5,
) -> np.ndarray:
    lpos = get_ds(ulog, "vehicle_local_position")
    if lpos is None:
        return np.ones(len(t_ref), dtype=bool)
    t_lp = t_rel(lpos, base_timestamp)
    h_speed = np.sqrt(arr(lpos, "vx") ** 2 + arr(lpos, "vy") ** 2)
    h_speed_i = interp_at(t_lp, h_speed, t_ref)
    return h_speed_i < max_speed_ms


def compute_tilt_deg(ulog: ULog, base_timestamp: int) -> tuple[np.ndarray, np.ndarray] | None:
    att = get_ds(ulog, "vehicle_attitude")
    if att is None:
        return None
    t = t_rel(att, base_timestamp)
    qx = arr(att, "q[1]")
    qy = arr(att, "q[2]")
    cos_tilt = np.clip(1.0 - 2.0 * (qx**2 + qy**2), -1.0, 1.0)
    return t, np.degrees(np.arccos(cos_tilt))


def plot_position_overview(ulog: ULog, base_timestamp: int, fig_dir: Path, shade_modes: bool = True) -> Path:
    """Altitude overview using EKF local-z, baro, optional rangefinder, and GPS."""
    air = get_ds(ulog, "vehicle_air_data")
    lpos = get_ds(ulog, "vehicle_local_position")
    gps = get_ds(ulog, "vehicle_gps_position")
    range_result = get_range(ulog, base_timestamp)

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 7.0), sharex=True)

    # Panel 1: AGL altitude comparison
    if lpos:
        tl = t_rel(lpos, base_timestamp)
        axes[0].plot(tl, -arr(lpos, "z"), label="EKF local alt (−z)", lw=1.1, color="tab:green")
    if range_result is not None:
        t_r, r_m = range_result
        axes[0].plot(t_r, r_m, label="Rangefinder AGL", lw=1.2, color="tab:blue")
    if air:
        ta = t_rel(air, base_timestamp)
        baro_agl = arr(air, "baro_alt_meter")
        # align baro to range for the first 30s
        if range_result is not None:
            t_r, r_m = range_result
            baro_i = interp_at(t_r, r_m, ta)
            early = (ta >= 0) & (ta <= 30.0) & np.isfinite(baro_i)
            offset = float(np.median(baro_agl[early] - baro_i[early])) if early.sum() > 0 else 0.0
            axes[0].plot(ta, baro_agl - offset, label="Baro (aligned to rangefinder)", lw=1.0,
                         color="tab:orange", alpha=0.8)
        else:
            axes[0].plot(ta, baro_agl, label="Baro MSL", lw=1.0, color="tab:orange", alpha=0.8)
    if gps:
        tg = t_rel(gps, base_timestamp)
        axes[0].plot(tg, gps_alt_m(gps), label="GPS MSL (ref only)", lw=0.8,
                     color="gray", ls=":", alpha=0.5)
    altitude_title = (
        "Altitude — EKF, rangefinder, baro (GPS shown for reference only)"
        if range_result is not None
        else "Altitude — EKF, baro, GPS reference (no distance_sensor topic)"
    )
    setup_axis(axes[0], altitude_title, "m AGL / offset-aligned")

    # Panel 2: EKF vertical velocity
    if lpos:
        tl = t_rel(lpos, base_timestamp)
        axes[1].plot(tl, arr(lpos, "vz"), label="EKF vz (NED down)", lw=1.0, color="tab:green")
        axes[1].axhline(0, color="gray", lw=0.6, ls=":")
    setup_axis(axes[1], "Vertical velocity", "m/s")

    # Panel 3: EKF horizontal position
    if lpos:
        tl = t_rel(lpos, base_timestamp)
        axes[2].plot(tl, arr(lpos, "x"), label="local x", lw=0.9)
        axes[2].plot(tl, arr(lpos, "y"), label="local y", lw=0.9)
    setup_axis(axes[2], "Local horizontal position", "m")
    axes[2].set_xlabel("flight-log relative time [s]")
    shade_mode_background(axes, ulog, base_timestamp, enabled=shade_modes)

    for ax in axes:
        ax.legend(loc="best", fontsize=8)

    return save_fig(fig, fig_dir, "position_overview")


def plot_local_xy_position(ulog: ULog, base_timestamp: int, fig_dir: Path, shade_modes: bool = True) -> Path:
    """Local horizontal EKF position: XY track plus x(t) and y(t)."""
    lpos = get_ds(ulog, "vehicle_local_position")
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10.5, 8.8),
        gridspec_kw={"height_ratios": [1.55, 1, 1]},
        constrained_layout=True,
    )

    if lpos and all(key in lpos.data for key in ["x", "y"]):
        t = t_rel(lpos, base_timestamp)
        x = arr(lpos, "x")
        y = arr(lpos, "y")

        finite_xy = np.isfinite(x) & np.isfinite(y)
        if finite_xy.any():
            axes[0].plot(y[finite_xy], x[finite_xy], lw=1.0, color="tab:blue", label="local XY")
            axes[0].scatter(y[finite_xy][0], x[finite_xy][0], s=28, color="tab:green", label="start", zorder=3)
            axes[0].scatter(y[finite_xy][-1], x[finite_xy][-1], s=28, color="tab:red", label="end", zorder=3)
            axes[0].set_aspect("equal", adjustable="box")

        axes[1].plot(t, x, lw=0.9, color="tab:blue", label="local x")
        axes[2].plot(t, y, lw=0.9, color="tab:orange", label="local y")
    else:
        for ax in axes:
            ax.text(0.5, 0.5, "vehicle_local_position x/y not logged", ha="center", va="center",
                    transform=ax.transAxes)

    setup_axis(axes[0], "Local XY position", "x [m]")
    axes[0].set_xlabel("y [m]")
    setup_axis(axes[1], "Local x as a function of time", "x [m]")
    setup_axis(axes[2], "Local y as a function of time", "y [m]")
    axes[2].set_xlabel("flight-log relative time [s]")
    shade_mode_background(axes[1:], ulog, base_timestamp, enabled=shade_modes)

    for ax in axes:
        handles, _labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best", fontsize=8)

    return save_fig(fig, fig_dir, "local_xy_position")


def plot_nav_state_timeline(ulog: ULog, base_timestamp: int, fig_dir: Path, shade_modes: bool = True) -> Path:
    """Plot vehicle_status.nav_state using PX4 enum labels for observed states."""
    status = get_ds(ulog, "vehicle_status")
    fig, ax = plt.subplots(figsize=(10.5, 4.2), constrained_layout=True)

    if status is None or "nav_state" not in status.data:
        ax.text(
            0.5,
            0.5,
            "vehicle_status.nav_state not logged",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        setup_axis(ax, "Flight mode timeline", "nav_state")
        ax.set_xlabel("flight-log relative time [s]")
        shade_mode_background(ax, ulog, base_timestamp, enabled=shade_modes)
        if not add_system_time_axis(ax, ulog, base_timestamp):
            ax.text(0.01, 0.97, "UTC system time unavailable", transform=ax.transAxes,
                    fontsize=8, color="tab:red", va="top")
        return save_fig(fig, fig_dir, "nav_state_timeline")

    t = t_rel(status, base_timestamp)
    nav = arr(status, "nav_state", int)
    observed = sorted({int(value) for value in nav})
    tick_labels = [f"{value} {nav_state_name(value)}" for value in observed]

    ax.step(t, nav, where="post", lw=1.2, color="tab:purple", label="vehicle_status.nav_state")
    ax.scatter(t[0], nav[0], s=18, color="tab:green", label="first sample", zorder=3)
    ax.scatter(t[-1], nav[-1], s=18, color="tab:red", label="last sample", zorder=3)
    ax.set_yticks(observed)
    ax.set_yticklabels(tick_labels)
    if observed:
        ax.set_ylim(min(observed) - 0.75, max(observed) + 0.75)
    setup_axis(ax, "Flight mode from ULog vehicle_status.nav_state", "PX4 nav_state enum")
    ax.set_xlabel("flight-log relative time [s]")
    shade_mode_background(ax, ulog, base_timestamp, enabled=shade_modes)
    mode_handles = [
        Patch(
            facecolor=nav_state_color(value),
            edgecolor="black",
            linewidth=0.3,
            alpha=0.35,
            label=f"{value} {nav_state_name(value)}",
        )
        for value in observed
    ]
    if mode_handles:
        ax.legend(handles=mode_handles, loc="best", fontsize=7, title="Observed mode colors")
    if not add_system_time_axis(ax, ulog, base_timestamp):
        ax.text(0.01, 0.97, "UTC system time unavailable", transform=ax.transAxes,
                fontsize=8, color="tab:red", va="top")

    return save_fig(fig, fig_dir, "nav_state_timeline")


def plot_tracking_components(
    ulog: ULog,
    base_timestamp: int,
    fig_dir: Path,
    fields: list[str],
    axis_labels: list[str],
    quantity_label: str,
    units_label: str,
    output_name: str,
    shade_modes: bool = True,
) -> Path:
    """Flight Review-style local-position/setpoint tracking plus estimate-setpoint deltas."""
    lpos = get_ds(ulog, "vehicle_local_position")
    sp = get_ds(ulog, "vehicle_local_position_setpoint")
    fig, axes = plt.subplots(6, 1, figsize=(10.5, 9.8), sharex=True, constrained_layout=True)
    setpoint_missing = sp is None

    if lpos is None:
        for ax in axes:
            ax.text(0.5, 0.5, "vehicle_local_position not logged", ha="center", va="center",
                    transform=ax.transAxes)
        for idx, (field, label) in enumerate(zip(fields, axis_labels)):
            setup_axis(axes[idx], f"{quantity_label} {label}", f"{field} [{units_label}]")
            setup_axis(axes[idx + 3], f"{quantity_label} {label} delta", units_label)
        axes[-1].set_xlabel("flight-log relative time [s]")
        shade_mode_background(axes, ulog, base_timestamp, enabled=shade_modes)
        return save_fig(fig, fig_dir, output_name)

    t = t_rel(lpos, base_timestamp)
    tsp = t_rel(sp, base_timestamp) if sp is not None else None
    colors = ["tab:blue", "tab:green", "tab:purple"]
    setpoint_colors = ["tab:orange", "tab:red", "tab:brown"]

    for idx, (field, label, color, sp_color) in enumerate(zip(fields, axis_labels, colors, setpoint_colors)):
        estimate = arr(lpos, field) if field in lpos.data else None
        setpoint = arr(sp, field) if sp is not None and field in sp.data else None
        ax_value = axes[idx]
        ax_delta = axes[idx + 3]

        if estimate is not None:
            ax_value.plot(t, estimate, lw=1.0, color=color, label=f"{field} estimate")
        else:
            ax_value.text(0.5, 0.5, f"vehicle_local_position.{field} not logged",
                          ha="center", va="center", transform=ax_value.transAxes)

        if setpoint is not None and tsp is not None:
            ax_value.step(tsp, setpoint, where="post", lw=1.0, color=sp_color, label=f"{field} setpoint")
        elif setpoint_missing:
            ax_value.text(0.01, 0.90, "vehicle_local_position_setpoint not logged",
                          transform=ax_value.transAxes, fontsize=8, color="tab:red")

        if estimate is not None and setpoint is not None and tsp is not None:
            delta = estimate - interp_step_previous(tsp, setpoint, t)
            ax_delta.plot(t, delta, lw=0.9, color=color, label=f"{field} estimate - setpoint")
            ax_delta.axhline(0.0, color="gray", lw=0.7, ls=":")
        else:
            ax_delta.text(0.5, 0.5, f"{field} delta unavailable", ha="center", va="center",
                          transform=ax_delta.transAxes)
        setup_axis(ax_value, f"{quantity_label} {label}", f"{field} [{units_label}]")
        setup_axis(ax_delta, f"{quantity_label} {label} delta (estimate - setpoint)", units_label)

    axes[-1].set_xlabel("flight-log relative time [s]")
    shade_mode_background(axes, ulog, base_timestamp, enabled=shade_modes)

    for ax in axes:
        handles, _labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best", fontsize=7, ncols=2)

    add_system_time_axis(axes[-1], ulog, base_timestamp)
    return save_fig(fig, fig_dir, output_name)


def plot_position_tracking(ulog: ULog, base_timestamp: int, fig_dir: Path, shade_modes: bool = True) -> Path:
    return plot_tracking_components(
        ulog,
        base_timestamp,
        fig_dir,
        fields=["x", "y", "z"],
        axis_labels=["X", "Y", "Z"],
        quantity_label="Local Position",
        units_label="m",
        output_name="position_tracking_xyz",
        shade_modes=shade_modes,
    )


def plot_velocity_tracking(ulog: ULog, base_timestamp: int, fig_dir: Path, shade_modes: bool = True) -> Path:
    return plot_tracking_components(
        ulog,
        base_timestamp,
        fig_dir,
        fields=["vx", "vy", "vz"],
        axis_labels=["X", "Y", "Z"],
        quantity_label="Local Velocity",
        units_label="m/s",
        output_name="velocity_tracking_xyz",
        shade_modes=shade_modes,
    )


def plot_manual_control_inputs(ulog: ULog, base_timestamp: int, fig_dir: Path, shade_modes: bool = True) -> Path:
    manual = get_ds(ulog, "manual_control_setpoint")
    controls = [
        ("roll", "Roll input"),
        ("pitch", "Pitch input"),
        ("yaw", "Yaw input"),
        ("throttle", "Throttle input"),
    ]
    if manual is not None and not any(field in manual.data for field, _label in controls):
        controls = [
            ("y", "Y / roll input"),
            ("x", "X / pitch input"),
            ("r", "Yaw input"),
            ("z", "Throttle input"),
        ]

    boolean_fields = [
        ("sticks_moving", "sticks_moving"),
        ("valid", "valid"),
    ]
    plotted_controls = available_fields(manual, [field for field, _label in controls])
    plotted_booleans = available_fields(manual, [field for field, _label in boolean_fields])
    nrows = max(1, len(plotted_controls) + len(plotted_booleans))
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(10.5, max(4.0, 0.85 + 1.72 * nrows)),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.ravel(np.atleast_1d(axes))

    if manual is None:
        axes[0].text(0.5, 0.5, "manual_control_setpoint not logged", ha="center", va="center",
                     transform=axes[0].transAxes)
        setup_axis(axes[0], "Manual Control Inputs unavailable", "input")
        axes[0].set_xlabel("flight-log relative time [s]")
        shade_mode_background(axes, ulog, base_timestamp, enabled=shade_modes)
        return save_fig(fig, fig_dir, "manual_control_inputs")

    t = t_rel(manual, base_timestamp)
    axis_idx = 0
    color_cycle = ["tab:blue", "tab:green", "tab:purple", "tab:orange"]
    control_labels = dict(controls)
    for color, field in zip(color_cycle, plotted_controls):
        ax = axes[axis_idx]
        ax.plot(t, arr(manual, field), lw=1.0, color=color, label=field)
        ax.axhline(0.0, color="gray", lw=0.7, ls=":")
        ax.set_ylim(-1.1, 1.1)
        setup_axis(ax, control_labels.get(field, field), "normalized input")
        ax.legend(loc="best", fontsize=7)
        axis_idx += 1

    boolean_labels = dict(boolean_fields)
    for field in plotted_booleans:
        ax = axes[axis_idx]
        ax.step(t, arr(manual, field, int), where="post", lw=1.0, color="tab:red", label=field)
        setup_boolean_axis(ax, boolean_labels.get(field, field))
        ax.legend(loc="best", fontsize=7)
        axis_idx += 1

    if axis_idx == 0:
        axes[0].text(0.5, 0.5, "manual_control_setpoint input fields not logged",
                     ha="center", va="center", transform=axes[0].transAxes)
        setup_axis(axes[0], "Manual Control Inputs unavailable", "input")

    axes[-1].set_xlabel("flight-log relative time [s]")
    shade_mode_background(axes, ulog, base_timestamp, enabled=shade_modes)
    add_system_time_axis(axes[-1], ulog, base_timestamp)
    return save_fig(fig, fig_dir, "manual_control_inputs")


def plot_terrain_estimate(ulog: ULog, base_timestamp: int, fig_dir: Path, shade_modes: bool = True) -> Path:
    """Plot published terrain/HAGL estimate from vehicle_local_position."""
    lpos = get_ds(ulog, "vehicle_local_position")
    flags = get_ds(ulog, "estimator_status_flags")
    gpos = get_ds(ulog, "vehicle_global_position")
    fig, axes = plt.subplots(
        5,
        1,
        figsize=(10.5, 7.4),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={
            "height_ratios": [1.35] + [_BOOLEAN_PANEL_HEIGHT_RATIO] * 4,
        },
    )

    if lpos is None or "dist_bottom" not in lpos.data:
        for ax in axes:
            ax.text(
                0.5,
                0.5,
                "vehicle_local_position.dist_bottom not logged",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        setup_axis(axes[0], "Terrain / HAGL estimate unavailable", "m")
        setup_boolean_axis(axes[1], "dist_bottom_valid")
        setup_boolean_axis(axes[2], "dist_bottom_sensor_bitfield != 0")
        setup_boolean_axis(axes[3], "cs_rng_hgt")
        setup_boolean_axis(axes[4], "cs_rng_kin_consistent")
        axes[4].set_xlabel("flight-log relative time [s]")
        shade_mode_background(axes, ulog, base_timestamp, enabled=shade_modes)
        return save_fig(fig, fig_dir, "terrain_estimate")

    t = t_rel(lpos, base_timestamp)
    dist_bottom = arr(lpos, "dist_bottom")
    valid = arr(lpos, "dist_bottom_valid", int).astype(bool) if "dist_bottom_valid" in lpos.data else None

    axes[0].plot(t, dist_bottom, lw=1.0, color="tab:blue", alpha=0.75, label="dist_bottom (HAGL estimate)")

    for key, label, color in [
        ("hagl_min", "hagl_min", "tab:red"),
        ("hagl_max", "hagl_max", "tab:orange"),
        ("hagl_max_z", "hagl_max_z", "tab:orange"),
        ("hagl_max_xy", "hagl_max_xy", "tab:purple"),
    ]:
        if key in lpos.data:
            values = arr(lpos, key)
            finite = np.isfinite(values)
            if finite.any():
                axes[0].plot(t[finite], values[finite], lw=0.8, ls="--", color=color, label=label)

    setup_axis(axes[0], "Published HAGL / terrain estimate", "m")

    if "z" in lpos.data:
        local_terrain_z = arr(lpos, "z") + dist_bottom
        axes[0].plot(t, local_terrain_z, lw=1.0, color="tab:brown", label="z + dist_bottom (terrain, NED down)")
        axes[0].plot(t, arr(lpos, "z"), lw=0.8, color="gray", ls=":", alpha=0.8, label="vehicle local z")
    if gpos is not None and "terrain_alt" in gpos.data:
        tg = t_rel(gpos, base_timestamp)
        axes[0].plot(tg, arr(gpos, "terrain_alt"), lw=0.9, color="tab:cyan",
                     label="vehicle_global_position.terrain_alt")
    setup_axis(axes[0], "Published HAGL and local terrain vertical position", "m")

    if valid is not None:
        axes[1].step(t, valid.astype(int), where="post", lw=1.0, color="tab:green", label="dist_bottom_valid")
    else:
        axes[1].text(0.5, 0.5, "not logged", ha="center", va="center", transform=axes[1].transAxes)
    setup_boolean_axis(axes[1], "HAGL estimate valid")

    if "dist_bottom_sensor_bitfield" in lpos.data:
        source_active = arr(lpos, "dist_bottom_sensor_bitfield", int) != 0
        axes[2].step(t, source_active.astype(int), where="post", lw=1.0, color="tab:purple",
                     label="dist_bottom_sensor_bitfield != 0")
    else:
        axes[2].text(0.5, 0.5, "not logged", ha="center", va="center", transform=axes[2].transAxes)
    setup_boolean_axis(axes[2], "HAGL source bitfield active")

    if flags is not None:
        t_flags = t_rel(flags, base_timestamp)
        if "cs_rng_hgt" in flags.data:
            axes[3].step(t_flags, arr(flags, "cs_rng_hgt", int), where="post", lw=1.0,
                         color="tab:red", label="cs_rng_hgt")
        else:
            axes[3].text(0.5, 0.5, "not logged", ha="center", va="center", transform=axes[3].transAxes)
    else:
        axes[3].text(0.5, 0.5, "estimator_status_flags not logged",
                     ha="center", va="center", transform=axes[3].transAxes)
    setup_boolean_axis(axes[3], "Range height fusion active")

    if flags is not None:
        t_flags = t_rel(flags, base_timestamp)
        if "cs_rng_kin_consistent" in flags.data:
            axes[4].step(t_flags, arr(flags, "cs_rng_kin_consistent", int), where="post", lw=1.0,
                         color="tab:blue", label="cs_rng_kin_consistent")
        else:
            axes[4].text(0.5, 0.5, "not logged", ha="center", va="center", transform=axes[4].transAxes)
    else:
        axes[4].text(0.5, 0.5, "estimator_status_flags not logged",
                     ha="center", va="center", transform=axes[4].transAxes)
    setup_boolean_axis(axes[4], "Range kinematic consistency")
    axes[4].set_xlabel("flight-log relative time [s]")
    shade_mode_background(axes, ulog, base_timestamp, enabled=shade_modes)

    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="best", fontsize=7)

    return save_fig(fig, fig_dir, "terrain_estimate")


def plot_baro_propwash(
    ulog: ULog,
    base_timestamp: int,
    fig_dir: Path,
    divergence_threshold_m: float = 0.5,
    throttle_threshold: float = 0.3,
    hover_max_speed_ms: float = 1.5,
    shade_modes: bool = True,
) -> Path:
    """Five-panel barometer / propwash analysis using distance_sensor as truth.

    Panel 1: EKF local-z, optional rangefinder AGL, baro.
    Panel 2: baro−rangefinder divergence with ±threshold bands when available.
    Panel 3: normalized thrust from vehicle_thrust_setpoint (or actuator_controls_0).
    Panel 4: vehicle tilt from vertical.
    Panel 5: EKF baro height innovation test ratio.
    Propwash event spans highlighted in translucent red.

    GPS altitude is excluded from scoring because it can drift several metres.
    """
    air = get_ds(ulog, "vehicle_air_data")
    lpos = get_ds(ulog, "vehicle_local_position")
    range_result = get_range(ulog, base_timestamp)
    thrust_result = get_thrust(ulog, base_timestamp)
    ratios = get_ds(ulog, "estimator_innovation_test_ratios")

    fig, axes = plt.subplots(5, 1, figsize=(10.5, 11.0), sharex=True)

    # Propwash detection
    pw_result = detect_propwash_events(ulog, base_timestamp, divergence_threshold_m, throttle_threshold)
    propwash_spans: list[tuple[float, float]] = []
    divergence_arr: np.ndarray | None = None
    t_div: np.ndarray | None = None

    if pw_result is not None:
        t_div, pw_mask, divergence_arr = pw_result
        propwash_spans = bool_spans(t_div, pw_mask)

    # Compute baro offset relative to rangefinder, when present.
    baro_result = compute_baro_range_divergence(ulog, base_timestamp)
    offset = 0.0
    if baro_result is not None:
        _tb, _ba, _div, offset = baro_result

    # --- panel 1: altitude comparison ---
    if lpos:
        tl = t_rel(lpos, base_timestamp)
        axes[0].plot(tl, -arr(lpos, "z"), label="EKF local alt (−z)", lw=1.1, color="tab:green", zorder=3)
    if range_result is not None:
        t_r, r_m = range_result
        axes[0].plot(t_r, r_m, label="Rangefinder AGL (truth)", lw=1.3, color="tab:blue", zorder=4)
    if air:
        ta = t_rel(air, base_timestamp)
        if range_result is not None:
            baro_agl = arr(air, "baro_alt_meter") - offset
            baro_label = "Baro (offset-aligned to rangefinder)"
        else:
            baro_agl = arr(air, "baro_alt_meter")
            baro_label = "Baro altitude"
        axes[0].plot(ta, baro_agl, label=baro_label, lw=1.0,
                     color="tab:orange", alpha=0.85)
    altitude_title = (
        "Altitude comparison — rangefinder is truth; GPS excluded"
        if range_result is not None
        else "Altitude comparison — no distance_sensor topic; no range truth"
    )
    setup_axis(axes[0], altitude_title, "m AGL / MSL")

    # --- panel 2: baro-rangefinder divergence ---
    if t_div is not None and divergence_arr is not None:
        axes[1].plot(t_div, divergence_arr, lw=1.0, color="tab:purple", label="baro − rangefinder (aligned)")
        axes[1].axhline(divergence_threshold_m, color="tab:red", lw=0.9, ls="--",
                        label=f"+{divergence_threshold_m:.2f} m threshold")
        axes[1].axhline(-divergence_threshold_m, color="tab:red", lw=0.9, ls="--",
                        label=f"−{divergence_threshold_m:.2f} m threshold")
        axes[1].axhline(0.0, color="gray", lw=0.6, ls=":")
        setup_axis(axes[1], "Baro − rangefinder divergence (prop wash signal)", "m")
    else:
        axes[1].text(
            0.5, 0.5,
            "No distance_sensor topic; rangefinder baro divergence not computed",
            ha="center", va="center", transform=axes[1].transAxes,
        )
        setup_axis(axes[1], "Rangefinder divergence unavailable", "m")

    # --- panel 3: thrust ---
    if thrust_result is not None:
        t_thr, thrust = thrust_result
        axes[2].plot(t_thr, thrust, lw=0.9, color="tab:brown", label="|thrust| (normalized)")
        axes[2].axhline(throttle_threshold, color="gray", lw=0.8, ls="--",
                        label=f"threshold {throttle_threshold:.2f}")
        axes[2].set_ylim(-0.05, 1.05)
    setup_axis(axes[2], "Normalized thrust (vehicle_thrust_setpoint)", "normalized")

    # --- panel 4: tilt ---
    tilt = compute_tilt_deg(ulog, base_timestamp)
    if tilt is not None:
        t_att, tilt_deg = tilt
        axes[3].plot(t_att, tilt_deg, lw=0.9, color="tab:cyan", label="tilt from vertical")
        axes[3].set_ylim(bottom=0)
    setup_axis(axes[3], "Vehicle tilt from vertical", "deg")

    # --- panel 5: baro innovation ratio ---
    baro_ratio = get_metric_series(ulog, base_timestamp, "baro_vpos", "test_ratio")
    if baro_ratio is not None:
        tr = baro_ratio.t
        bvp = baro_ratio.values
        bvp = np.where(np.isfinite(bvp), bvp, np.nan)
        axes[4].plot(tr, bvp, lw=0.9, color="tab:red", label=baro_ratio.source)
        axes[4].axhline(1.0, color="black", lw=0.9, ls="--", label="gate (1.0)")
        axes[4].set_yscale("symlog", linthresh=0.1)
    setup_axis(axes[4], "EKF baro height innovation test ratio", "ratio")
    axes[4].set_xlabel("flight-log relative time [s]")

    shade_mode_background(axes, ulog, base_timestamp, enabled=shade_modes)

    # --- overlay propwash spans ---
    for start, end in propwash_spans:
        for ax in axes:
            ax.axvspan(start, end, color="red", alpha=0.15, zorder=0)

    if propwash_spans:
        axes[0].axvspan(propwash_spans[0][0], propwash_spans[0][0], color="red",
                        alpha=0.4, label="propwash event")

    for ax in axes:
        handles, _labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best", fontsize=7, ncols=2)

    return save_fig(fig, fig_dir, "baro_propwash")


def plot_gps_quality(ulog: ULog, base_timestamp: int, fig_dir: Path, shade_modes: bool = True) -> Path:
    gps = get_ds(ulog, "vehicle_gps_position")
    gps_status = get_ds(ulog, "estimator_gps_status")
    disabled_label = gps_disabled_label(ulog)
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
    setup_axis(axes[2], "Reported position accuracy (EPH/EPV)", "m")
    setup_axis(axes[3], "Speed metrics")
    setup_axis(axes[4], "Jamming/spoofing and GPS check flags")
    axes[4].set_xlabel("flight-log relative time [s]")
    axes[1].set_ylim(-0.5, 10.0)
    shade_mode_background(axes, ulog, base_timestamp, enabled=shade_modes)

    if disabled_label is not None:
        for ax in axes:
            ax.text(
                0.99,
                0.92,
                disabled_label,
                ha="right",
                va="top",
                transform=ax.transAxes,
                fontsize=9,
                color="tab:red",
                fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 1.5},
            )

    fig.text(0.5, 0.01,
             "Note: GPS altitude is NOT used as truth for baro divergence. "
             "Rangefinder-based divergence requires a logged distance_sensor topic.",
             ha="center", fontsize=8, style="italic", color="gray")

    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="best", fontsize=7, ncols=2)

    return save_fig(fig, fig_dir, "gps_quality")


def plot_fusion_flags(ulog: ULog, base_timestamp: int, fig_dir: Path, shade_modes: bool = True) -> Path:
    flags = get_ds(ulog, "estimator_status_flags")
    fields = available_fields(flags, _PLOTTED_FUSION_FLAG_FIELDS)
    nrows = max(1, len(fields))
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(10.5, max(2.0, 0.72 + 0.50 * nrows)),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.ravel(np.atleast_1d(axes))

    if not fields or flags is None:
        axes[0].text(0.5, 0.5, "estimator_status_flags not logged", ha="center", va="center",
                     transform=axes[0].transAxes)
        setup_boolean_axis(axes[0], "Fusion flags unavailable")
    else:
        t = t_rel(flags, base_timestamp)
        for ax, key in zip(axes, fields):
            ax.step(t, arr(flags, key, int), where="post", lw=1.0, color="tab:blue", label=key)
            setup_boolean_axis(ax, key)
            ax.legend(loc="best", fontsize=7)

    axes[-1].set_xlabel("flight-log relative time [s]")
    shade_mode_background(axes, ulog, base_timestamp, enabled=shade_modes)
    return save_fig(fig, fig_dir, "fusion_flags")


def plot_reset_counters(ulog: ULog, base_timestamp: int, fig_dir: Path, shade_modes: bool = True) -> Path:
    status = get_ds(ulog, "estimator_status")
    reset_rows = reset_event_rows(ulog, base_timestamp)
    fields = available_fields(status, _RESET_COUNTER_FIELDS)
    nrows = max(1, len(fields))
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(10.5, max(2.0, 0.85 + 0.72 * nrows)),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.ravel(np.atleast_1d(axes))

    if not fields or status is None:
        axes[0].text(0.5, 0.5, "estimator_status reset counters not logged", ha="center", va="center",
                     transform=axes[0].transAxes)
        setup_axis(axes[0], "Reset counters unavailable", "count")
    else:
        t = t_rel(status, base_timestamp)
        for ax, key in zip(axes, fields):
            values = arr(status, key, int)
            ax.step(t, values, where="post", lw=1.0, color="tab:purple", label=key)
            for row_idx, row in enumerate(reset_rows):
                ax.axvline(
                    row.time_s,
                    color="tab:red",
                    lw=0.9,
                    ls=":",
                    alpha=0.75,
                    label="attitude/heading reset" if row_idx == 0 else None,
                )
            setup_axis(ax, key, "count")
            setup_integer_value_axis(ax, values)
            ax.legend(loc="best", fontsize=7)

    axes[-1].set_xlabel("flight-log relative time [s]")
    shade_mode_background(axes, ulog, base_timestamp, enabled=shade_modes)
    return save_fig(fig, fig_dir, "reset_counters")


def plot_status_masks(ulog: ULog, base_timestamp: int, fig_dir: Path, shade_modes: bool = True) -> Path:
    status = get_ds(ulog, "estimator_status")
    fields = available_fields(status, _STATUS_MASK_FIELDS)
    nrows = max(1, len(fields))
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(10.5, max(2.0, 0.85 + 0.72 * nrows)),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.ravel(np.atleast_1d(axes))

    if not fields or status is None:
        axes[0].text(0.5, 0.5, "estimator_status masks not logged", ha="center", va="center",
                     transform=axes[0].transAxes)
        setup_axis(axes[0], "Status masks unavailable", "mask")
    else:
        t = t_rel(status, base_timestamp)
        for ax, key in zip(axes, fields):
            values = arr(status, key, int)
            ax.step(t, values, where="post", lw=1.0, color="tab:blue", label=key)
            setup_axis(ax, key, "raw mask")
            setup_integer_value_axis(ax, values)
            ax.legend(loc="best", fontsize=7)

    axes[-1].set_xlabel("flight-log relative time [s]")
    shade_mode_background(axes, ulog, base_timestamp, enabled=shade_modes)
    return save_fig(fig, fig_dir, "status_masks")


def plot_innovation_check_flags(ulog: ULog, base_timestamp: int, fig_dir: Path, shade_modes: bool = True) -> Path:
    status = get_ds(ulog, "estimator_status")
    nrows = len(_INNOVATION_CHECK_FLAG_GROUPS)
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(10.5, max(2.4, 0.72 + 0.50 * nrows)),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.ravel(np.atleast_1d(axes))

    if status is None or "innovation_check_flags" not in status.data:
        for ax in axes:
            ax.text(0.5, 0.5, "estimator_status.innovation_check_flags not logged",
                    ha="center", va="center", transform=ax.transAxes)
        setup_axis(axes[0], "Innovation check flags unavailable", "bit value")
    else:
        t = t_rel(status, base_timestamp)
        raw = arr(status, "innovation_check_flags", int)
        for ax, (label, bit_shift, mask) in zip(axes, _INNOVATION_CHECK_FLAG_GROUPS):
            values = innovation_check_flag_group_values(raw, bit_shift, mask)
            ax.step(t, values, where="post", lw=1.0, color="tab:red", label=label)
            ylabel = "0/1" if mask == 1 else f"0-{mask}"
            setup_axis(ax, label, ylabel)
            setup_integer_value_axis(ax, values)
            ax.legend(loc="best", fontsize=7)

    axes[-1].set_xlabel("flight-log relative time [s]")
    shade_mode_background(axes, ulog, base_timestamp, enabled=shade_modes)
    return save_fig(fig, fig_dir, "innovation_check_flags")


def plot_estimator_event_flags(ulog: ULog, base_timestamp: int, fig_dir: Path, shade_modes: bool = True) -> Path:
    event = get_ds(ulog, "estimator_event_flags")
    fields = []
    if event is not None:
        for key in available_fields(event, _ESTIMATOR_EVENT_FLAG_FIELDS):
            values = arr(event, key, int)
            if np.any(values != 0) or len(np.unique(values)) > 1:
                fields.append(key)
    nrows = max(1, len(fields))
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(10.5, max(2.0, 0.72 + 0.50 * nrows)),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.ravel(np.atleast_1d(axes))

    if not fields or event is None:
        axes[0].text(0.5, 0.5, "no active estimator_event_flags booleans logged", ha="center", va="center",
                     transform=axes[0].transAxes)
        setup_boolean_axis(axes[0], "Estimator event flags")
    else:
        t = t_rel(event, base_timestamp)
        for ax, key in zip(axes, fields):
            ax.step(t, arr(event, key, int), where="post", lw=1.0, color="tab:green", label=key)
            setup_boolean_axis(ax, key)
            ax.legend(loc="best", fontsize=7)

    axes[-1].set_xlabel("flight-log relative time [s]")
    shade_mode_background(axes, ulog, base_timestamp, enabled=shade_modes)
    return save_fig(fig, fig_dir, "estimator_event_flags")


def plot_innovation_ratios(ulog: ULog, base_timestamp: int, fig_dir: Path, shade_modes: bool = True) -> Path:
    fig, axes = plt.subplots(5, 1, figsize=(10.5, 10.2), sharex=True, constrained_layout=True)

    groups = [
        (axes[0], ["gps_hvel[0]", "gps_hvel[1]", "gps_vvel"], "GPS velocity innovation test ratios"),
        (axes[1], ["gps_hpos[0]", "gps_hpos[1]", "gps_vpos"], "GPS position innovation test ratios"),
        (axes[2], ["baro_vpos"], "Baro vertical position innovation test ratio"),
        (axes[3], ["rng_vpos"], "Range vertical position innovation test ratio"),
        (axes[4], ["heading"], "Heading innovation test ratio"),
    ]
    for ax, keys, title in groups:
        for key in keys:
            series = get_metric_series(ulog, base_timestamp, key, "test_ratio")
            if series is not None:
                y = np.where(np.isfinite(series.values), series.values, np.nan)
                ax.plot(series.t, y, lw=0.9, label=series.source)
        ax.axhline(1.0, color="black", lw=0.9, ls="--", label="gate")
        ax.axhline(0.36, color="gray", lw=0.7, ls=":", label="3 sigma equiv")
        setup_axis(ax, title, "test ratio")
        ax.set_yscale("symlog", linthresh=0.1)

    axes[-1].set_xlabel("flight-log relative time [s]")
    shade_mode_background(axes, ulog, base_timestamp, enabled=shade_modes)
    for ax in axes:
        ax.legend(loc="best", fontsize=7, ncols=2)

    return save_fig(fig, fig_dir, "innovation_test_ratios")


def plot_height_innovation_test_ratios(ulog: ULog, base_timestamp: int, fig_dir: Path, shade_modes: bool = True) -> Path:
    fig, axes = plt.subplots(4, 1, figsize=(10.5, 8.8), sharex=True)
    channels = [
        ("hagl", "HAGL innovation test ratio"),
        ("hagl_rate", "HAGL rate innovation test ratio"),
        ("baro_vpos", "Baro vertical position innovation test ratio"),
        ("rng_vpos", "Range vertical position innovation test ratio"),
    ]

    for ax, (key, title) in zip(axes, channels):
        series = get_metric_series(ulog, base_timestamp, key, "test_ratio")
        ax.axhline(1.0, color="tab:red", lw=0.9, ls="--", label="test ratio gate 1.0")
        if series is None:
            setup_axis(ax, f"{title} missing", "ratio")
            ax.legend(loc="best", fontsize=7)
            continue

        y = np.where(np.isfinite(series.values), series.values, np.nan)
        ax.plot(series.t, y, lw=0.9, color="tab:blue", label=series.source)
        starts = exceedance_start_times(series.t, y, 1.0)
        for idx, start in enumerate(starts):
            ax.axvline(
                start,
                color="tab:red",
                lw=0.8,
                ls=":",
                alpha=0.7,
                label="exceeds 1.0" if idx == 0 else None,
            )
        setup_axis(ax, title, "ratio")
        finite = y[np.isfinite(y)]
        y_max = max(1.2, float(np.nanmax(finite)) * 1.08) if finite.size else 1.2
        ax.set_ylim(0.0, y_max)
        ax.legend(loc="best", fontsize=7, ncols=2)

    axes[-1].set_xlabel("flight-log relative time [s]")
    shade_mode_background(axes, ulog, base_timestamp, enabled=shade_modes)
    return save_fig(fig, fig_dir, "height_innovation_test_ratios")


def plot_residuals(
    ulog: ULog,
    base_timestamp: int,
    fig_dir: Path,
    keys: list[str],
    name: str,
    title_prefix: str,
    shade_modes: bool = True,
) -> Path:
    fig, axes = plt.subplots(len(keys), 1, figsize=(10.5, 2.2 * len(keys) + 1.2), sharex=True)
    if len(keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, keys):
        innovation = get_metric_series(ulog, base_timestamp, key, "innovation")
        variance_series = get_metric_series(ulog, base_timestamp, key, "innovation_variance")
        if innovation is None or variance_series is None:
            setup_axis(ax, f"{key} missing")
            continue

        variance = interp_at(variance_series.t, variance_series.values, innovation.t)
        threshold = 5.0 * np.sqrt(np.maximum(variance, 0.0))
        ax.plot(innovation.t, innovation.values, label=innovation.source, lw=0.9)
        ax.plot(innovation.t, threshold, color="tab:red", lw=0.8, ls="--", label="+5 sigma gate")
        ax.plot(innovation.t, -threshold, color="tab:red", lw=0.8, ls="--", label="-5 sigma gate")
        setup_axis(ax, f"{title_prefix}: {key}", "m or m/s")
        ax.legend(loc="best", fontsize=7)

    axes[-1].set_xlabel("flight-log relative time [s]")
    shade_mode_background(axes, ulog, base_timestamp, enabled=shade_modes)
    return save_fig(fig, fig_dir, name)


def baro_metric_summary(
    ulog: ULog,
    base_timestamp: int,
    divergence_threshold_m: float = 0.5,
    throttle_threshold: float = 0.3,
    hover_max_speed_ms: float = 1.5,
) -> dict[str, str]:
    rangefinder_config = describe_rangefinder_config(ulog)
    out: dict[str, str] = {
        "distance_sensor_topic": "present" if rangefinder_config.has_topic else "absent",
        "rangefinder_hardware": rangefinder_config.configured_hardware,
        "rangefinder_config": rangefinder_config.report_summary,
    }

    baro_result = compute_baro_range_divergence(ulog, base_timestamp)
    if baro_result is None:
        out["baro_truth_source"] = "no rangefinder — baro divergence not computed"
        return out

    out["baro_truth_source"] = "rangefinder"
    t_div, _baro_alt, divergence, offset = baro_result

    finite_div = divergence[np.isfinite(divergence)]
    if finite_div.size:
        out["baro_range_offset_m"] = f"{offset:.2f}"
        out["baro_range_max_divergence_m"] = f"{np.max(np.abs(finite_div)):.2f}"
        out["baro_range_rms_divergence_m"] = f"{np.sqrt(np.mean(finite_div**2)):.2f}"

    pw_result = detect_propwash_events(ulog, base_timestamp, divergence_threshold_m, throttle_threshold)
    if pw_result is None:
        return out
    t_baro, pw_mask, _div = pw_result

    pw_spans = bool_spans(t_baro, pw_mask)
    out["baro_propwash_events"] = str(len(pw_spans))
    total_pw_s = sum(end - start for start, end in pw_spans)
    out["baro_propwash_total_s"] = f"{total_pw_s:.1f}"
    out["baro_propwash_spans"] = format_spans(pw_spans, limit=4)

    hover = _hover_mask(ulog, base_timestamp, t_baro, hover_max_speed_ms)
    hover_pw = pw_mask & hover
    hover_pw_spans = bool_spans(t_baro, hover_pw)
    total_hover_pw_s = sum(end - start for start, end in hover_pw_spans)
    out["baro_propwash_hover_events"] = str(len(hover_pw_spans))
    out["baro_propwash_hover_total_s"] = f"{total_hover_pw_s:.1f}"

    hover_div = np.abs(divergence[hover & np.isfinite(divergence)])
    if hover_div.size:
        out["baro_range_max_divergence_hover_m"] = f"{np.max(hover_div):.2f}"

    return out


def metric_summary(ulog: ULog, base_timestamp: int) -> dict[str, str]:
    air = get_ds(ulog, "vehicle_air_data")
    flags = get_ds(ulog, "estimator_status_flags")
    status = get_ds(ulog, "estimator_status")
    gps_status = get_ds(ulog, "estimator_gps_status")
    lpos = get_ds(ulog, "vehicle_local_position")
    gps = get_ds(ulog, "vehicle_gps_position")

    out: dict[str, str] = describe_mpc_alt_mode(ulog)

    if gps:
        gps_alt = gps_alt_m(gps)
        out["gps_alt_drift_m"] = f"{gps_alt[-1] - gps_alt[0]:.1f} (informational only — not used as truth)"
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
        x = arr(lpos, "x")
        y = arr(lpos, "y")
        z = arr(lpos, "z")
        vz = arr(lpos, "vz")
        out["local_x_range"] = f"{np.nanmin(x):.1f} to {np.nanmax(x):.1f} m"
        out["local_y_range"] = f"{np.nanmin(y):.1f} to {np.nanmax(y):.1f} m"
        out["local_z_range"] = f"{np.nanmin(z):.1f} to {np.nanmax(z):.1f} m"
        out["local_z_last"] = f"{z[-1]:.1f} m"
        out["local_vz_range"] = f"{np.nanmin(vz):.1f} to {np.nanmax(vz):.1f} m/s"

    if status:
        for key in ["reset_count_vel_ne", "reset_count_vel_d", "reset_count_pos_ne",
                    "reset_count_pod_d", "reset_count_quat"]:
            if key in status.data:
                values = arr(status, key, int)
                out[key] = f"{values[0]}--{values[-1]}"

    if flags:
        t = t_rel(flags, base_timestamp)
        for key in ["cs_gps", "cs_gnss_vel", "cs_gps_hgt", "cs_baro_hgt",
                    "cs_rng_hgt", "fs_bad_acc_vertical", "cs_inertial_dead_reckoning"]:
            if key in flags.data:
                out[key] = format_spans(bool_spans(t, arr(flags, key, int)), limit=3)

    for key in ["gps_vpos", "gps_vvel", "gps_hvel[0]", "gps_hvel[1]",
                "gps_hpos[0]", "gps_hpos[1]", "baro_vpos", "rng_vpos"]:
        series = get_metric_series(ulog, base_timestamp, key, "test_ratio")
        if series is not None:
            finite = series.values[np.isfinite(series.values)]
            if finite.size:
                out[f"{key}_source"] = series.source
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


def latex_breakable_text(text: str) -> str:
    escaped = latex_escape(text)
    for marker in [r"\_", ";", ",", "="]:
        escaped = escaped.replace(marker, marker + r"\allowbreak{}")
    return escaped


def latex_html_color(color: str) -> str:
    return color.strip().lstrip("#").upper()


def fusion_state_latex_table(rows: list[FusionStateRow]) -> str:
    if not rows:
        return "No active estimator fusion-state fields were logged during armed time."

    body = "\n".join(
        " & ".join(
            [
                latex_escape(row.field),
                latex_escape(row.meaning),
                f"{row.duration_s:.1f}",
                f"{row.percent:.1f}\\%",
                latex_escape(row.spans),
            ]
        )
        + r" \\"
        + "\n\\hline"
        for row in rows
    )

    return rf"""\begin{{table}}[H]
\centering
\scriptsize
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{|l|l|r|r|l|}}
\hline
Field & Literal meaning & Duration s & Armed \% & Spans \\
\hline
{body}
\end{{tabular}}%
}}
\caption{{Active estimator fusion state from \texttt{{estimator\_status\_flags}}. Derived combined rows use OR logic; individual rows still show the underlying EKF mode split.}}
\end{{table}}"""


def reset_event_latex_table(rows: list[ResetEventRow]) -> str:
    if not rows:
        return "No attitude or local-position heading reset counter increments were logged."

    body = "\n".join(
        " & ".join(
            [
                latex_escape(row.source),
                f"{row.time_s:.3f}",
                f"{row.old_count}--{row.new_count}",
                latex_escape(row.delta),
            ]
        )
        + r" \\"
        + "\n\\hline"
        for row in rows
    )

    return rf"""\begin{{table}}[H]
\centering
\scriptsize
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{|l|r|l|p{{0.38\textwidth}}|}}
\hline
Source & Time s & Count & Delta \\
\hline
{body}
\end{{tabular}}%
}}
\caption{{Logged attitude and heading reset counter increments. Times are flight-log relative seconds.}}
\end{{table}}"""


def estimator_exception_latex_table(rows: list[EstimatorExceptionRow]) -> str:
    if not rows:
        return (
            "No non-healthy estimator status, fault, rejection, timeout, dead-reckoning, "
            "or GPS check-failure fields occurred during armed time."
        )

    body = "\n".join(
        " & ".join(
            [
                latex_escape(row.topic),
                latex_escape(row.field),
                latex_escape(row.meaning),
                f"{row.duration_s:.1f}",
                latex_escape(row.spans),
            ]
        )
        + r" \\"
        + "\n\\hline"
        for row in rows
    )

    return rf"""\begin{{table}}[H]
\centering
\scriptsize
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{|l|l|l|r|l|}}
\hline
Topic & Field & Literal meaning & Duration s & Spans \\
\hline
{body}
\end{{tabular}}%
}}
\caption{{Only non-healthy estimator status fields that occurred during armed time.}}
\end{{table}}"""


def nav_state_latex_table(rows: list[NavStateRow]) -> str:
    if not rows:
        return "No \\texttt{vehicle\\_status.nav\\_state} samples were logged."

    body = "\n".join(
        " & ".join(
            [
                str(row.value),
                latex_escape(row.name),
                f"{row.duration_s:.1f}",
                latex_escape(row.spans),
            ]
        )
        + r" \\"
        + "\n\\hline"
        for row in rows
    )

    return rf"""\begin{{table}}[H]
\centering
\scriptsize
\begin{{tabular}}{{|r|l|r|l|}}
\hline
Enum value & PX4 nav state & Duration s & Spans \\
\hline
{body}
\end{{tabular}}
\caption{{Observed flight modes from the logged \texttt{{vehicle\_status.nav\_state}} topic field. This is not a PX4 parameter from \texttt{{initial\_parameters}}.}}
\end{{table}}"""


def nav_state_color_key_latex_table() -> str:
    body = "\n".join(
        " & ".join(
            [
                str(value),
                latex_breakable_text(info.enum),
                latex_breakable_text(info.description),
                rf"\cellcolor[HTML]{{{latex_html_color(info.color)}}} ",
            ]
        )
        + r" \\"
        + "\n\\hline"
        for value, info in PX4_NAV_STATES.items()
    )

    return rf"""\scriptsize
\begin{{longtable}}{{|r|p{{0.33\textwidth}}|p{{0.43\textwidth}}|p{{0.08\textwidth}}|}}
\caption{{Saved PX4 navigation-state enum list copied from \texttt{{msg/versioned/VehicleStatus.msg}}. Plot background colors use these values with low opacity.}}\\
\hline
Value & Enum & PX4 description & Plot color \\
\hline
\endfirsthead
\hline
Value & Enum & PX4 description & Plot color \\
\hline
\endhead
\hline
\multicolumn{{4}}{{|r|}}{{continued on next page}}\\
\hline
\endfoot
\hline
\endlastfoot
{body}
\end{{longtable}}
\normalsize"""


def logged_parameter_latex_table(rows: list[ParameterRow]) -> str:
    if not rows:
        return "No \\texttt{EKF2\\_*} or \\texttt{MPC\\_*} initial parameters were logged."

    tables = []
    for prefix in ["EKF2", "MPC"]:
        group_rows = [row for row in rows if row.prefix == prefix]
        if not group_rows:
            continue

        body = "\n".join(
            " & ".join(
                [
                    latex_escape(row.name),
                    latex_escape(row.value),
                ]
            )
            + r" \\"
            + "\n\\hline"
            for row in group_rows
        )

        tables.append(
            rf"""\subsection*{{\texttt{{{prefix}\_*}}}}
\scriptsize
\setlength{{\tabcolsep}}{{4pt}}
\setlength{{\LTleft}}{{0pt}}
\setlength{{\LTright}}{{\fill}}
\begin{{longtable}}{{|p{{0.42\textwidth}}|p{{0.22\textwidth}}|}}
\caption{{Logged \texttt{{{prefix}\_*}} ULog initial parameters ({len(group_rows)} rows). These are the values saved in the log; firmware-default comparison is not applied in this table.}}\\
\hline
Parameter & Logged value \\
\hline
\endfirsthead
\hline
Parameter & Logged value \\
\hline
\endhead
\hline
\multicolumn{{2}}{{|r|}}{{continued on next page}}\\
\hline
\endfoot
\hline
\endlastfoot
{body}
\end{{longtable}}
\normalsize"""
        )

    return "\n\n".join(tables)


def build_latex(
    config: ReviewConfig,
    figures: dict[str, Path],
    summary: dict[str, str],
    fusion_rows: list[FusionStateRow],
    reset_rows: list[ResetEventRow],
    exception_rows: list[EstimatorExceptionRow],
    nav_state_rows: list[NavStateRow],
    parameter_rows: list[ParameterRow],
) -> str:
    rel_figures = {key: path.relative_to(config.output_dir).as_posix() for key, path in figures.items()}
    log_title = latex_escape(config.log.title)

    baro_truth = latex_escape(summary.get("baro_truth_source", "unknown"))
    rangefinder_config = latex_breakable_text(summary.get("rangefinder_config", "unknown"))
    baro_rms = latex_escape(summary.get("baro_range_rms_divergence_m", "n/a"))
    baro_max = latex_escape(summary.get("baro_range_max_divergence_m", "n/a"))
    pw_events = latex_escape(summary.get("baro_propwash_events", "n/a"))
    pw_total = latex_escape(summary.get("baro_propwash_total_s", "n/a"))
    fusion_table = fusion_state_latex_table(fusion_rows)
    reset_table = reset_event_latex_table(reset_rows)
    exception_table = estimator_exception_latex_table(exception_rows)
    nav_state_table = nav_state_latex_table(nav_state_rows)
    nav_state_color_key = nav_state_color_key_latex_table()
    parameter_table = logged_parameter_latex_table(parameter_rows)
    has_rangefinder = summary.get("baro_truth_source") == "rangefinder"
    if has_rangefinder:
        rangefinder_scope = (
            "GPS altitude is shown in the GPS quality panel for informational context only; "
            "it is not used as truth for rangefinder-based baro divergence."
        )
        baro_summary_items = rf"""\item \textbf{{Baro--rangefinder RMS divergence:}} {baro_rms}~m
\item \textbf{{Baro--rangefinder max divergence:}} {baro_max}~m
\item \textbf{{Propwash events (baro--rangefinder $>{latex_escape(f"{config.divergence_threshold_m:.2f}")}$~m while thrust $>{latex_escape(f"{config.throttle_threshold:.2f}")}$):}} {pw_events} events, {pw_total}~s"""
        altitude_caption = (
            "EKF local altitude ($-z$), rangefinder AGL truth, and baro aligned to rangefinder. "
            "GPS MSL shown as dotted reference only."
        )
        propwash_intro = (
            rf"Propwash events (red spans) occur when $|\text{{baro}} - \text{{rangefinder}}| > "
            rf"{latex_escape(f'{config.divergence_threshold_m:.2f}')}$~m and thrust $> "
            rf"{latex_escape(f'{config.throttle_threshold:.2f}')}$."
        )
        propwash_caption = (
            "Panel 1: altitude comparison using rangefinder as truth. "
            "Panel 2: baro$-$rangefinder divergence with threshold bands. "
            "Panel 3: normalized thrust. Panel 4: vehicle tilt. "
            "Panel 5: EKF baro height innovation ratio. Red spans are propwash events."
        )
    else:
        rangefinder_scope = (
            "No rangefinder truth source is logged. GPS altitude is shown for context only, "
            "and rangefinder-based baro divergence/propwash scoring is not computed."
        )
        baro_summary_items = (
            r"\item \textbf{Rangefinder baro divergence:} unavailable; no \texttt{distance\_sensor} topic"
            "\n"
            r"\item \textbf{Propwash events:} not computed without rangefinder truth"
        )
        altitude_caption = (
            "EKF local altitude, baro altitude, and GPS MSL reference. "
            "No \\texttt{distance\\_sensor} topic was logged."
        )
        propwash_intro = (
            r"Rangefinder-based propwash scoring is unavailable because this log has no "
            r"\texttt{distance\_sensor} topic."
        )
        propwash_caption = (
            "Diagnostic altitude/baro plot. No rangefinder divergence is computed because "
            "the log does not contain \\texttt{distance\\_sensor}."
        )
    gps_caption = (
        "GPS quality indicators. GPS altitude is informational only; rangefinder-based "
        "baro divergence requires a logged \\texttt{distance\\_sensor} topic."
    )
    time_note = latex_escape(summary.get("system_time_note", "UTC system time status unknown."))
    terrain_dist_bottom = latex_escape(summary.get("terrain_dist_bottom", "not logged"))
    terrain_valid = latex_escape(summary.get("terrain_dist_bottom_valid", "not logged"))
    terrain_bitfield = latex_escape(summary.get("terrain_dist_bottom_sensor_bitfield", "not logged"))
    mpc_alt_mode = latex_escape(summary.get("mpc_alt_mode", "not logged"))
    mpc_alt_label = latex_escape(summary.get("mpc_alt_mode_label", "unknown"))
    mpc_alt_detail = latex_escape(summary.get("mpc_alt_mode_detail", "unknown"))
    mpc_alt_related = latex_breakable_text(summary.get("mpc_alt_mode_related_params", "none logged"))

    return rf"""\documentclass[10pt]{{article}}
\usepackage[margin=0.7in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{float}}
\usepackage{{hyperref}}
\usepackage{{longtable}}
\usepackage{{siunitx}}
\usepackage{{caption}}
\usepackage[table]{{xcolor}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{0.5em}}
\renewcommand{{\arraystretch}}{{1.12}}

\title{{{latex_escape(config.title)}}}
\author{{Generated from ULog}}
\date{{}}

\begin{{document}}
\maketitle

\section{{Scope}}
Single-log EKF2 analysis for \texttt{{{log_title}}}.
Altitude truth source: \textbf{{{baro_truth}}}.
Rangefinder evidence: {rangefinder_config}.
{rangefinder_scope}
Time-series plots use very light background shading for
\texttt{{vehicle\_status.nav\_state}} when mode shading is enabled. The complete
PX4 enum and color key are saved in the appendix.

\section{{High-Level Read}}
\begin{{itemize}}
{baro_summary_items}
\item \textbf{{Rangefinder evidence:}} {rangefinder_config}
\item \textbf{{EKF local-z range:}} {latex_escape(summary.get("local_z_range", "n/a"))}~m
\item \textbf{{Baro height active:}} {latex_escape(summary.get("cs_baro_hgt", "n/a"))}
\item \textbf{{Range height active:}} {latex_escape(summary.get("cs_rng_hgt", "n/a"))}
\item \textbf{{Dead reckoning:}} {latex_escape(summary.get("cs_inertial_dead_reckoning", "none"))}
\item \textbf{{Bad vertical accel:}} {latex_escape(summary.get("fs_bad_acc_vertical", "none"))}
\item \textbf{{GPS quality:}} {latex_escape(summary.get("gps_quality_typical", "n/a"))}
\end{{itemize}}

\section{{Active Estimator Fusion State}}
{fusion_table}

\section{{Estimator Reset Events}}
{reset_table}

\section{{Estimator Health Exceptions}}
{exception_table}

\section{{MPC\_ALT\_MODE}}
\begin{{itemize}}
\item \textbf{{Logged value:}} \texttt{{{mpc_alt_mode}}} ({mpc_alt_label})
\item \textbf{{Controller behavior:}} {mpc_alt_detail}
\item \textbf{{Related logged parameters:}} {mpc_alt_related}
\item \textbf{{Estimator distinction:}} \texttt{{MPC\_ALT\_MODE}} changes the multicopter
position controller altitude reference behavior. It does not select the EKF height
reference; that is reported separately through \texttt{{EKF2\_HGT\_REF}} and the
active fusion flags such as \texttt{{cs\_baro\_hgt}} and \texttt{{cs\_rng\_hgt}}.
\end{{itemize}}

\section{{Flight Mode Timeline}}
Flight Review reports mode from the logged topic field
\texttt{{vehicle\_status.nav\_state}}, not from a PX4 parameter in
\texttt{{initial\_parameters}}.
{time_note}
{nav_state_table}

\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{{rel_figures["nav_state"]}}}
\caption{{Flight mode timeline from \texttt{{vehicle\_status.nav\_state}}. The
y-axis tick labels include the numeric enum value observed in the log and the
matching PX4 navigation-state name. When \texttt{{boot\_time\_utc\_us}} is
available in the ULog metadata, the top x-axis shows UTC system time. The
background color bands are the same mode shading used on the other time-series
plots.}}
\end{{figure}}

\section{{Terrain / HAGL Estimate}}
\begin{{itemize}}
\item \textbf{{Source field:}} \texttt{{vehicle\_local\_position.dist\_bottom}}
\item \textbf{{Logged HAGL range:}} {terrain_dist_bottom}
\item \textbf{{Valid spans:}} {terrain_valid}
\item \textbf{{Source bitfield values:}} {terrain_bitfield}
\item \textbf{{Range height fusion:}} {latex_escape(summary.get("cs_rng_hgt", "not logged"))}
\item \textbf{{Range kinematically consistent:}} {latex_escape(summary.get("cs_rng_kin_consistent", "not logged"))}
\end{{itemize}}

\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{{rel_figures["terrain"]}}}
\caption{{Terrain/HAGL estimate from \texttt{{vehicle\_local\_position}}. The
first panel combines \texttt{{dist\_bottom}} with logged HAGL min/max bounds and
the implied local terrain vertical position \texttt{{z + dist\_bottom}} when
available. The remaining panels show separate 0/1 status traces for
\texttt{{dist\_bottom\_valid}},
\texttt{{dist\_bottom\_sensor\_bitfield != 0}}, \texttt{{cs\_rng\_hgt}}, and
\texttt{{cs\_rng\_kin\_consistent}}.}}
\end{{figure}}

\section{{Altitude Overview}}
\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{{rel_figures["position"]}}}
\caption{{{altitude_caption}}}
\end{{figure}}

\section{{Local XYZ Position Tracking}}
\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{{rel_figures["position_tracking"]}}}
\caption{{Flight Review-style local position tracking. The first three panels
show \texttt{{vehicle\_local\_position}} x, y, and z against step-held
\texttt{{vehicle\_local\_position\_setpoint}} values. The bottom three panels
show estimate minus step-held setpoint for each axis. PX4 local z is NED-down.}}
\end{{figure}}

\clearpage
\section{{Local XYZ Velocity Tracking}}
\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{{rel_figures["velocity_tracking"]}}}
\caption{{Flight Review-style local velocity tracking. The first three panels
show \texttt{{vehicle\_local\_position}} vx, vy, and vz against step-held
\texttt{{vehicle\_local\_position\_setpoint}} values. The bottom three panels
show estimate minus step-held setpoint for each axis. PX4 local vz is NED-down,
so positive vz is downward/descent. For user input context, Flight Review uses
\texttt{{manual\_control\_setpoint}}; this log contains that topic.}}
\end{{figure}}

\clearpage
\section{{Manual Control Inputs}}
\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{{rel_figures["manual_control"]}}}
\caption{{Logged joystick/manual inputs from \texttt{{manual\_control\_setpoint}}.
The normalized roll, pitch, yaw, and throttle channels show the commanded user
inputs that can produce position-control setpoint changes. The boolean panels
show \texttt{{sticks\_moving}} and \texttt{{valid}} when logged.}}
\end{{figure}}

\section{{Local XY Position}}
\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{{rel_figures["local_xy"]}}}
\caption{{EKF local horizontal position. The first panel is the local XY track with
start/end markers; the second and third panels show local x and local y as
functions of flight-log relative time.}}
\end{{figure}}

\section{{Barometer / Propwash Analysis}}
{propwash_intro}

\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{{rel_figures["baro_propwash"]}}}
\caption{{{propwash_caption}}}
\end{{figure}}

\section{{GPS Quality (Informational)}}
\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{{rel_figures["quality"]}}}
\caption{{{gps_caption}}}
\end{{figure}}

\section{{Fusion Flags}}
\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{{rel_figures["fusion_flags"]}}}
\caption{{Estimator fusion and rejection flags from \texttt{{estimator\_status\_flags}}.
Each field is plotted as its own 0/1 trace with no visual offsets.}}
\end{{figure}}

\section{{Reset Counters}}
\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{{rel_figures["reset_counters"]}}}
\caption{{State reset counters from \texttt{{estimator\_status}}. Each counter is
shown in its own subplot; red dotted lines mark attitude or heading reset events
from \texttt{{vehicle\_attitude}} and \texttt{{vehicle\_local\_position}}.}}
\end{{figure}}

\section{{Estimator Status Masks}}
\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{{rel_figures["status_masks"]}}}
\caption{{Raw estimator status masks from \texttt{{estimator\_status}}, plotted
separately with no offsets. The decoded \texttt{{innovation\_check\_flags}}
groups are shown in the next section.}}
\end{{figure}}

\section{{Innovation Check Flags}}
\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{{rel_figures["innovation_check_flags"]}}}
\caption{{Decoded \texttt{{estimator\_status.innovation\_check\_flags}} using the
same bit grouping Flight Review uses for velocity, horizontal position, vertical
position, magnetometer, yaw, airspeed, synthetic sideslip, HAGL, and optical-flow
checks. Multi-bit groups show the grouped integer value.}}
\end{{figure}}

\section{{Estimator Event Flags}}
\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{{rel_figures["event_flags"]}}}
\caption{{Active boolean fields from \texttt{{estimator\_event\_flags}}. All-zero
event fields are omitted so persistent reset-source flags remain readable.}}
\end{{figure}}

\section{{Innovation Test Ratios}}
\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{{rel_figures["ratios"]}}}
\caption{{Innovation test ratios. 1.0 is the configured gate; 0.36 is approximately 3-sigma with a 5-sigma gate.
\texttt{{baro\_vpos}} and \texttt{{rng\_vpos}} are the key channels for baro compensation evaluation.}}
\end{{figure}}

\section{{Height Innovation Test Ratios}}
\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{{rel_figures["height_ratios"]}}}
\caption{{Height-related innovation test ratios for \texttt{{hagl}},
\texttt{{hagl\_rate}}, \texttt{{baro\_vpos}}, and \texttt{{rng\_vpos}}. The
horizontal red dashed line marks the 1.0 rejection gate; vertical red dotted
lines mark upward crossings where the logged test ratio first exceeds 1.0.}}
\end{{figure}}

\section{{Residuals Versus Gates}}
\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{{rel_figures["vertical_residuals"]}}}
\caption{{Vertical innovations vs $\pm5\sigma$ gates from logged innovation variances.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{{rel_figures["horizontal_residuals"]}}}
\caption{{Horizontal GPS residuals vs $\pm5\sigma$ gates.}}
\end{{figure}}

\clearpage
\appendix
\section{{PX4 Navigation-State Color Key}}
{nav_state_color_key}

\section{{Logged EKF2 and MPC Parameters}}
{parameter_table}

\end{{document}}
"""


def write_summary(output_dir: Path, summary: dict[str, str]) -> Path:
    summary_path = output_dir / "summary.txt"
    lines = []
    for metric, value in sorted(summary.items()):
        lines.append(f"{metric}: {value}")
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def compile_latex(tex_path: Path) -> Path:
    if shutil.which("latexmk") is None:
        raise RuntimeError(
            "latexmk not found. Install a LaTeX distribution first "
            "(macOS: MacTeX or BasicTeX; Linux: latexmk plus a TeX Live package)."
        )
    subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", tex_path.name],
        cwd=tex_path.parent,
        check=True,
    )
    pdf_path = tex_path.with_suffix(".pdf")
    if not pdf_path.exists():
        raise FileNotFoundError(f"latexmk completed but did not produce {pdf_path}")
    return pdf_path


def _px4_review(
    config: ReviewConfig, fig_dir: Path
) -> tuple[dict[str, Path], dict[str, str], str]:
    """Build the PX4 EKF2 figures, metrics, and LaTeX body from a ULog."""
    # Unwrapped to the pyulog handle: the PX4 plots/metrics below read ULog
    # attributes that are outside the LogSource interface.
    ulog = open_log(config.log.path, config.log.log_type).raw
    base_timestamp = ulog.start_timestamp

    figures = {
        "position": plot_position_overview(ulog, base_timestamp, fig_dir, shade_modes=config.shade_flight_modes),
        "local_xy": plot_local_xy_position(ulog, base_timestamp, fig_dir, shade_modes=config.shade_flight_modes),
        "nav_state": plot_nav_state_timeline(ulog, base_timestamp, fig_dir, shade_modes=config.shade_flight_modes),
        "position_tracking": plot_position_tracking(
            ulog, base_timestamp, fig_dir, shade_modes=config.shade_flight_modes
        ),
        "velocity_tracking": plot_velocity_tracking(
            ulog, base_timestamp, fig_dir, shade_modes=config.shade_flight_modes
        ),
        "manual_control": plot_manual_control_inputs(
            ulog, base_timestamp, fig_dir, shade_modes=config.shade_flight_modes
        ),
        "terrain": plot_terrain_estimate(ulog, base_timestamp, fig_dir, shade_modes=config.shade_flight_modes),
        "baro_propwash": plot_baro_propwash(
            ulog, base_timestamp, fig_dir,
            divergence_threshold_m=config.divergence_threshold_m,
            throttle_threshold=config.throttle_threshold,
            hover_max_speed_ms=config.hover_max_speed_ms,
            shade_modes=config.shade_flight_modes,
        ),
        "quality": plot_gps_quality(ulog, base_timestamp, fig_dir, shade_modes=config.shade_flight_modes),
        "fusion_flags": plot_fusion_flags(ulog, base_timestamp, fig_dir, shade_modes=config.shade_flight_modes),
        "reset_counters": plot_reset_counters(ulog, base_timestamp, fig_dir, shade_modes=config.shade_flight_modes),
        "status_masks": plot_status_masks(ulog, base_timestamp, fig_dir, shade_modes=config.shade_flight_modes),
        "innovation_check_flags": plot_innovation_check_flags(
            ulog, base_timestamp, fig_dir, shade_modes=config.shade_flight_modes
        ),
        "event_flags": plot_estimator_event_flags(ulog, base_timestamp, fig_dir, shade_modes=config.shade_flight_modes),
        "ratios": plot_innovation_ratios(ulog, base_timestamp, fig_dir, shade_modes=config.shade_flight_modes),
        "height_ratios": plot_height_innovation_test_ratios(
            ulog, base_timestamp, fig_dir, shade_modes=config.shade_flight_modes
        ),
        "vertical_residuals": plot_residuals(
            ulog, base_timestamp, fig_dir,
            keys=["baro_vpos", "rng_vpos", "gps_vpos", "gps_vvel"],
            name="vertical_residual_thresholds",
            title_prefix="Vertical",
            shade_modes=config.shade_flight_modes,
        ),
        "horizontal_residuals": plot_residuals(
            ulog, base_timestamp, fig_dir,
            keys=["gps_hpos[0]", "gps_hpos[1]", "gps_hvel[0]", "gps_hvel[1]"],
            name="horizontal_residual_thresholds",
            title_prefix="Horizontal GPS",
            shade_modes=config.shade_flight_modes,
        ),
    }

    summary = {
        **metric_summary(ulog, base_timestamp),
        "system_time_note": system_time_note(ulog, base_timestamp),
        **terrain_estimate_summary(ulog, base_timestamp),
        **baro_metric_summary(
            ulog, base_timestamp,
            divergence_threshold_m=config.divergence_threshold_m,
            throttle_threshold=config.throttle_threshold,
            hover_max_speed_ms=config.hover_max_speed_ms,
        ),
    }
    fusion_rows = active_fusion_state_rows(ulog, base_timestamp)
    reset_rows = reset_event_rows(ulog, base_timestamp)
    exception_rows = estimator_exception_rows(ulog, base_timestamp)
    nav_state_rows = logged_nav_state_rows(ulog, base_timestamp)
    parameter_rows = logged_parameter_rows(ulog)

    tex = build_latex(
        config,
        figures,
        summary,
        fusion_rows,
        reset_rows,
        exception_rows,
        nav_state_rows,
        parameter_rows,
    )
    return figures, summary, tex


def _rover_review(
    config: ReviewConfig, fig_dir: Path
) -> tuple[dict[str, Path], dict[str, str], str]:
    """Build the ArduPilot Rover figures, metrics, and LaTeX body from a BIN."""
    # Imported here (not at module scope) because rover imports the shared helpers
    # from this module. open_log pulls in pymavlink lazily too.
    from . import rover

    source = open_log(config.log.path, config.log.log_type, keep_types=rover.MESSAGE_TYPES)
    return rover.build_review(config, source, source.start_timestamp, fig_dir)


def generate_review(config: ReviewConfig) -> ReviewArtifacts:
    output_dir = config.output_dir
    fig_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not config.log.path.exists():
        raise FileNotFoundError(f"log file not found: {config.log.path}")

    # The log's format selects the analysis: ArduPilot dataflash gets the Rover
    # steering/EK3 review, everything else the PX4 EKF2 review.
    is_rover = resolve_log_type(config.log.path, config.log.log_type) == ARDUPILOT
    if is_rover:
        figures, summary, tex = _rover_review(config, fig_dir)
    else:
        figures, summary, tex = _px4_review(config, fig_dir)

    summary_path = write_summary(output_dir, summary)

    tex_path = output_dir / ("rover_review.tex" if is_rover else "ekf2_review.tex")
    tex_path.write_text(tex, encoding="utf-8")

    pdf_path = compile_latex(tex_path)

    return ReviewArtifacts(
        output_dir=output_dir,
        tex_path=tex_path,
        summary_path=summary_path,
        pdf_path=pdf_path,
        figures=figures,
    )
