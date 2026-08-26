from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from pyulog import ULog

from .config import ReviewConfig


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
class SignalSeries:
    label: str
    t: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class ActuatorOutputSet:
    title: str
    ylabel: str
    source: str
    series: list[SignalSeries]


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
class StatusMaskRow:
    field: str
    observed_values: str
    active_bits: str


@dataclass(frozen=True)
class SensorStatusRow:
    sensor: str
    name: str
    value: str
    status: str


@dataclass(frozen=True)
class ParameterBitmaskRow:
    parameter: str
    raw_value: str
    bit: int
    status: str
    meaning: str


@dataclass(frozen=True)
class ParameterEnumRow:
    parameter: str
    raw_value: str
    value: int
    selected: bool
    status: str
    meaning: str


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

_DIRECT_AID_METRIC_FIELDS = {
    "flow[0]": (
        "estimator_aid_src_optical_flow",
        "innovation[0]",
        "innovation_variance[0]",
        "test_ratio[0]",
    ),
    "flow[1]": (
        "estimator_aid_src_optical_flow",
        "innovation[1]",
        "innovation_variance[1]",
        "test_ratio[1]",
    ),
}

_AGGREGATE_METRIC_TOPICS = {
    "innovation": "estimator_innovations",
    "innovation_variance": "estimator_innovation_variances",
    "test_ratio": "estimator_innovation_test_ratios",
}

_BOOLEAN_PANEL_HEIGHT_RATIO = 0.28
_MODE_SHADING_ALPHA = 0.075
_REPORT_LINE_WIDTH_SCALE = 2.0
_ACTUATOR_SATURATION_LABELS = {
    -2: "ACTUATOR_SATURATION_LOWER",
    -1: "ACTUATOR_SATURATION_LOWER_DYN",
    0: "ACTUATOR_SATURATION_OK",
    1: "ACTUATOR_SATURATION_UPPER_DYN",
    2: "ACTUATOR_SATURATION_UPPER",
}

# Copied from PX4 msg/versioned/VehicleStatus.msg (MESSAGE_VERSION = 4).
PX4_NAV_STATES = {
    0: NavStateInfo("NAVIGATION_STATE_MANUAL", "Manual mode", "#4E79A7"),
    1: NavStateInfo("NAVIGATION_STATE_ALTCTL", "Altitude control mode", "#F28E2B"),
    2: NavStateInfo("NAVIGATION_STATE_POSCTL", "Position control mode", "#59A14F"),
    3: NavStateInfo("NAVIGATION_STATE_AUTO_MISSION", "Auto mission mode", "#E15759"),
    4: NavStateInfo("NAVIGATION_STATE_AUTO_LOITER", "Auto loiter mode", "#B07AA1"),
    5: NavStateInfo("NAVIGATION_STATE_AUTO_RTL", "Auto return to launch mode", "#EDC948"),
    6: NavStateInfo("NAVIGATION_STATE_POSITION_SLOW", "Position slow", "#76B7B2"),
    7: NavStateInfo(
        "NAVIGATION_STATE_GUIDED_COURSE",
        "Guided Course mode (FW: maintain course/alt/speed)",
        "#FF9DA7",
    ),
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
    20: NavStateInfo(
        "NAVIGATION_STATE_AUTO_PRECLAND", "Precision land with landing target", "#AEC7E8"
    ),
    21: NavStateInfo("NAVIGATION_STATE_ORBIT", "Orbit in a circle", "#FFBB78"),
    22: NavStateInfo(
        "NAVIGATION_STATE_AUTO_VTOL_TAKEOFF", "Takeoff, transition, establish loiter", "#98DF8A"
    ),
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

_CONTROL_STATUS_GROUPS = [
    (
        "Alignment and vehicle state",
        [
            "cs_tilt_align",
            "cs_yaw_align",
            "cs_in_air",
            "cs_fixed_wing",
            "cs_in_transition",
            "cs_vehicle_at_rest",
            "cs_gnd_effect",
            "cs_heading_observable",
            "cs_yaw_manual",
        ],
    ),
    (
        "GNSS and global aiding",
        [
            "cs_gps",
            "cs_gnss_pos",
            "cs_gnss_vel",
            "cs_gps_hgt",
            "cs_gnss_yaw",
            "cs_aux_gpos",
        ],
    ),
    (
        "Height and terrain aiding",
        [
            "cs_baro_hgt",
            "cs_rng_hgt",
            "cs_ev_hgt",
            "cs_fake_hgt",
            "cs_rng_terrain",
            "cs_opt_flow_terrain",
            "cs_rng_kin_consistent",
            "cs_rng_stuck",
            "cs_baro_fault",
            "cs_rng_fault",
            "cs_gnss_hgt_fault",
        ],
    ),
    (
        "Vision and optical flow",
        ["cs_opt_flow", "cs_ev_pos", "cs_ev_vel", "cs_ev_yaw", "cs_ev_yaw_fault"],
    ),
    (
        "Magnetometer and heading",
        [
            "cs_mag_hdg",
            "cs_mag_3d",
            "cs_mag_dec",
            "cs_mag",
            "cs_synthetic_mag_z",
            "cs_mag_aligned_in_flight",
            "cs_mag_heading_consistent",
            "cs_mag_field_disturbed",
            "cs_mag_fault",
            "cs_gnss_yaw_fault",
            "cs_gps_yaw_fault",
        ],
    ),
    (
        "Air data, wind, and auxiliary",
        ["cs_wind", "cs_wind_dead_reckoning", "cs_fuse_beta", "cs_fuse_aspd", "cs_gravity_vector"],
    ),
    (
        "Dead reckoning and fake aiding",
        ["cs_inertial_dead_reckoning", "cs_fake_pos", "cs_valid_fake_pos", "cs_constant_pos"],
    ),
]

_RESET_COUNTER_FIELDS = [
    "reset_count_vel_ne",
    "reset_count_vel_d",
    "reset_count_pos_ne",
    "reset_count_pod_d",
    "reset_count_quat",
]

_LOCAL_POSITION_RESET_COUNTER_FIELDS = [
    "xy_reset_counter",
    "z_reset_counter",
    "vxy_reset_counter",
    "vz_reset_counter",
    "heading_reset_counter",
    "dist_bottom_reset_counter",
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

_PARAMETER_BITMASK_DEFINITIONS = {
    "EKF2_IMU_CTRL": [
        (0, "Gyro bias"),
        (1, "Accel bias"),
        (2, "Gravity vector fusion"),
    ],
    "EKF2_GPS_CTRL": [
        (0, "Longitude and latitude fusion"),
        (1, "Altitude fusion"),
        (2, "3D velocity fusion"),
        (3, "Dual antenna heading fusion"),
    ],
    "EKF2_MAG_CHECK": [
        (0, "Strength (EKF2_MAG_CHK_STR)"),
        (1, "Inclination (EKF2_MAG_CHK_INC)"),
        (2, "Wait for WMM"),
    ],
    "EKF2_GPS_CHECK": [
        (0, "Satellite count (EKF2_REQ_NSATS)"),
        (1, "PDOP (EKF2_REQ_PDOP)"),
        (2, "Horizontal position error / EPH (EKF2_REQ_EPH)"),
        (3, "Vertical position error / EPV (EKF2_REQ_EPV)"),
        (4, "Speed accuracy (EKF2_REQ_SACC)"),
        (5, "Horizontal position drift (EKF2_REQ_HDRIFT)"),
        (6, "Vertical position drift (EKF2_REQ_VDRIFT)"),
        (7, "Horizontal speed offset (EKF2_REQ_HDRIFT)"),
        (8, "Vertical speed offset/discrepancy (EKF2_REQ_VDRIFT)"),
        (9, "Spoofing"),
        (10, "GPS fix type (EKF2_REQ_FIX)"),
        (11, "Jamming"),
    ],
}

_PARAMETER_ENUM_DEFINITIONS = {
    "EKF2_RNG_CTRL": [
        (0, "Disable range fusion"),
        (1, "Enabled - conditional mode"),
        (2, "Enabled"),
    ],
}

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

_EXPECTED_SENSOR_STATUS_PARAMS = [
    ("IMU", "SYS_HAS_ACC", "accelerometer availability status bit"),
    ("IMU", "SYS_HAS_GYRO", "gyroscope availability status bit"),
    ("Magnetometer", "SYS_HAS_MAG", "magnetometer availability status bit"),
    ("Barometer", "SYS_HAS_BARO", "barometer availability status bit"),
    ("GPS / GNSS", "SYS_HAS_GPS", "GNSS availability status bit"),
    ("Range finder", "SYS_HAS_NUM_DIST", "logged number of distance sensors"),
    ("Optical flow", "SYS_HAS_OF", "optical-flow availability status bit"),
    ("Barometer", "EKF2_BARO_CTRL", "EKF barometer height-aiding control"),
    ("GPS / GNSS", "EKF2_GPS_CTRL", "EKF GNSS aiding control"),
    ("Range finder", "EKF2_RNG_CTRL", "EKF range height-aiding control"),
    ("Optical flow", "EKF2_OF_CTRL", "EKF optical-flow aiding control"),
    ("External vision", "EKF2_EV_CTRL", "EKF external-vision aiding control"),
    ("IMU", "EKF2_IMU_CTRL", "EKF IMU control bitmask"),
]

_SENSOR_PARAM_ORDER = {
    name: index for index, (_sensor, name, _note) in enumerate(_EXPECTED_SENSOR_STATUS_PARAMS)
}


def sensor_label_for_param(name: str) -> str:
    explicit = {param: sensor for sensor, param, _note in _EXPECTED_SENSOR_STATUS_PARAMS}
    if name in explicit:
        return explicit[name]

    stripped = name
    for prefix in ("SYS_HAS_NUM_", "SYS_HAS_", "EKF2_"):
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix) :]
            break
    stripped = stripped.removesuffix("_CTRL")
    return stripped.replace("_", " ").title()


def interpret_sensor_param(name: str, value: object | None) -> str:
    if value is None:
        return "not logged"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "logged"

    if name.startswith("SYS_HAS_"):
        if name == "SYS_HAS_NUM_DIST":
            return "DISABLED" if numeric <= 0 else "ENABLED"
        return "ENABLED" if numeric > 0 else "DISABLED"

    if name.startswith("EKF2_") and name.endswith("_CTRL"):
        return "DISABLED" if numeric <= 0 else "ENABLED"

    return "logged"


def sensor_status_rows(ulog: ULog) -> list[SensorStatusRow]:
    params = initial_parameters(ulog)
    names = {
        name
        for name in params
        if name.startswith("SYS_HAS_") or (name.startswith("EKF2_") and name.endswith("_CTRL"))
    }
    names.update(name for _sensor, name, _note in _EXPECTED_SENSOR_STATUS_PARAMS)

    def sort_key(name: str) -> tuple[int, str]:
        return (_SENSOR_PARAM_ORDER.get(name, 10_000), name)

    rows: list[SensorStatusRow] = []
    for name in sorted(names, key=sort_key):
        value = params.get(name)
        rows.append(
            SensorStatusRow(
                sensor=sensor_label_for_param(name),
                name=name,
                value="not logged" if value is None else format_param_value(value),
                status=interpret_sensor_param(name, value),
            )
        )
    return rows


def parameter_bitmask_rows(ulog: ULog) -> list[ParameterBitmaskRow]:
    params = initial_parameters(ulog)
    rows: list[ParameterBitmaskRow] = []

    for name, bit_defs in _PARAMETER_BITMASK_DEFINITIONS.items():
        value = params.get(name)
        if value is None:
            for bit, meaning in bit_defs:
                rows.append(
                    ParameterBitmaskRow(
                        parameter=name,
                        raw_value="not logged",
                        bit=bit,
                        status="NOT LOGGED",
                        meaning=meaning,
                    )
                )
            continue

        try:
            mask = round(float(value))
        except (TypeError, ValueError):
            mask = 0
            raw_value = str(value)
        else:
            raw_value = str(mask)

        for bit, meaning in bit_defs:
            enabled = bool(mask & (1 << bit))
            rows.append(
                ParameterBitmaskRow(
                    parameter=name,
                    raw_value=raw_value,
                    bit=bit,
                    status="ENABLED" if enabled else "DISABLED",
                    meaning=meaning,
                )
            )
    return rows


def parameter_enum_rows(ulog: ULog) -> list[ParameterEnumRow]:
    params = initial_parameters(ulog)
    rows: list[ParameterEnumRow] = []

    for name, value_defs in _PARAMETER_ENUM_DEFINITIONS.items():
        value = params.get(name)
        if value is None:
            for option, meaning in value_defs:
                rows.append(
                    ParameterEnumRow(
                        parameter=name,
                        raw_value="not logged",
                        value=option,
                        selected=False,
                        status="NOT LOGGED",
                        meaning=meaning,
                    )
                )
            continue

        try:
            selected_value = round(float(value))
        except (TypeError, ValueError):
            selected_value = -1
            raw_value = str(value)
        else:
            raw_value = str(selected_value)

        for option, meaning in value_defs:
            selected = option == selected_value
            if selected and option == 0:
                status = "DISABLED"
            elif selected:
                status = "ENABLED"
            else:
                status = "not selected"

            rows.append(
                ParameterEnumRow(
                    parameter=name,
                    raw_value=raw_value,
                    value=option,
                    selected=selected,
                    status=status,
                    meaning=meaning,
                )
            )
    return rows


def get_ds(ulog: ULog, name: str, multi_id: int = 0):
    matches = [d for d in ulog.data_list if d.name == name and d.multi_id == multi_id]
    return matches[0] if matches else None


def get_all_ds(ulog: ULog, name: str) -> list:
    return sorted(
        [d for d in ulog.data_list if d.name == name],
        key=lambda item: getattr(item, "multi_id", 0),
    )


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
        return ""
    start_time = datetime.fromtimestamp(start_utc_us / 1e6, UTC)
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
        datetime.fromtimestamp((start_utc_us + round(tick * 1e6)) / 1e6, UTC) for tick in ticks
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


def time_since_last_fuse_s(ds, base_timestamp: int) -> tuple[np.ndarray, np.ndarray] | None:
    if ds is None or "time_last_fuse" not in ds.data or "timestamp" not in ds.data:
        return None

    timestamp = arr(ds, "timestamp", np.uint64).astype(float)
    time_last_fuse = arr(ds, "time_last_fuse", np.uint64).astype(float)
    count = min(len(timestamp), len(time_last_fuse))
    if count == 0:
        return None

    timestamp = timestamp[:count]
    time_last_fuse = time_last_fuse[:count]
    valid = (time_last_fuse > 0) & (timestamp >= time_last_fuse)
    delta_s = np.full(count, np.nan)
    delta_s[valid] = (timestamp[valid] - time_last_fuse[valid]) / 1e6
    return (timestamp - int(base_timestamp)) / 1e6, delta_s


def measurement_interval_s(
    ds, base_timestamp: int, sample_timestamp_field: str = "timestamp_sample"
) -> tuple[np.ndarray, np.ndarray, str] | None:
    if ds is None or "timestamp" not in ds.data:
        return None

    source_field = sample_timestamp_field if sample_timestamp_field in ds.data else "timestamp"
    timestamp = arr(ds, "timestamp", np.uint64).astype(float)
    sample_timestamp = arr(ds, source_field, np.uint64).astype(float)
    count = min(len(timestamp), len(sample_timestamp))
    if count == 0:
        return None

    timestamp = timestamp[:count]
    sample_timestamp = sample_timestamp[:count]
    interval_s = np.full(count, np.nan)
    if count > 1:
        delta_us = np.diff(sample_timestamp)
        valid = delta_us > 0
        interval_s[np.nonzero(valid)[0] + 1] = delta_us[valid] / 1e6
    return (timestamp - int(base_timestamp)) / 1e6, interval_s, source_field


def measurement_latency_s(
    ds, base_timestamp: int, sample_timestamp_field: str = "timestamp_sample"
) -> tuple[np.ndarray, np.ndarray, str] | None:
    if ds is None or "timestamp" not in ds.data or sample_timestamp_field not in ds.data:
        return None

    timestamp = arr(ds, "timestamp", np.uint64).astype(float)
    sample_timestamp = arr(ds, sample_timestamp_field, np.uint64).astype(float)
    count = min(len(timestamp), len(sample_timestamp))
    if count == 0:
        return None

    timestamp = timestamp[:count]
    sample_timestamp = sample_timestamp[:count]
    latency_s = np.full(count, np.nan)
    valid = sample_timestamp > 0
    latency_s[valid] = (timestamp[valid] - sample_timestamp[valid]) / 1e6
    return (timestamp - int(base_timestamp)) / 1e6, latency_s, sample_timestamp_field


def measurement_count_s(ds, base_timestamp: int) -> tuple[np.ndarray, np.ndarray] | None:
    if ds is None or "timestamp" not in ds.data:
        return None

    timestamp = arr(ds, "timestamp", np.uint64).astype(float)
    if not len(timestamp):
        return None

    return (timestamp - int(base_timestamp)) / 1e6, np.arange(1, len(timestamp) + 1)


def get_metric_series(
    ulog: ULog, base_timestamp: int, key: str, metric: str
) -> MetricSeries | None:
    """Return a scalar EKF innovation metric.

    Prefer newer estimator_aid_src_* topics for scalar height sources, then fall back
    to the aggregate estimator_innovations-style topics used by older/replay logs.
    """
    direct_fields = _DIRECT_AID_METRIC_FIELDS.get(key)
    if direct_fields is not None:
        aid_topic, innovation_field, variance_field, ratio_field = direct_fields
        metric_field = {
            "innovation": innovation_field,
            "innovation_variance": variance_field,
            "test_ratio": ratio_field,
        }[metric]
        aid = get_ds(ulog, aid_topic)
        if aid is not None and metric_field in aid.data:
            return MetricSeries(
                t_rel(aid, base_timestamp),
                arr(aid, metric_field),
                f"{aid_topic}.{metric_field}",
            )

    aid_topic = _HEIGHT_AID_SOURCES.get(key)
    if aid_topic is not None:
        aid = get_ds(ulog, aid_topic)
        if aid is not None and metric in aid.data:
            return MetricSeries(
                t_rel(aid, base_timestamp), arr(aid, metric), f"{aid_topic}.{metric}"
            )

    aggregate_topic = _AGGREGATE_METRIC_TOPICS[metric]
    aggregate = get_ds(ulog, aggregate_topic)
    if aggregate is not None and key in aggregate.data:
        return MetricSeries(
            t_rel(aggregate, base_timestamp),
            arr(aggregate, key),
            f"{aggregate_topic}.{key}",
        )

    return None


def rangefinder_test_ratio_series(
    ulog: ULog, base_timestamp: int
) -> tuple[MetricSeries | None, MetricSeries | None]:
    aid = get_ds(ulog, "estimator_aid_src_rng_hgt")
    if aid is not None and "test_ratio" in aid.data:
        range_height = MetricSeries(
            t_rel(aid, base_timestamp),
            arr(aid, "test_ratio"),
            "estimator_aid_src_rng_hgt.test_ratio",
        )
    else:
        aggregate = get_ds(ulog, "estimator_innovation_test_ratios")
        range_height = None
        if aggregate is not None:
            fallback_field = "hagl" if "hagl" in aggregate.data else "rng_vpos"
            if fallback_field in aggregate.data:
                range_height = MetricSeries(
                    t_rel(aggregate, base_timestamp),
                    arr(aggregate, fallback_field),
                    f"estimator_innovation_test_ratios.{fallback_field}",
                )

    return range_height, get_metric_series(ulog, base_timestamp, "hagl_rate", "test_ratio")


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
        return str(round(numeric))
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
    configured_hardware = (
        ", ".join(enabled) if enabled else "no enabled rangefinder driver params found"
    )

    details = [
        f"{name}={format_param_value(params[name])}"
        for name in _RANGEFINDER_DETAIL_PARAMS
        if name in params
    ]
    param_summary = "; ".join(details) if details else "no rangefinder-specific params found"

    if has_topic:
        report_summary = (
            f"{topic_summary}; configured hardware: {configured_hardware}; params: {param_summary}"
        )
    else:
        report_summary = (
            f"{topic_summary}; configured hardware: {configured_hardware}; params: {param_summary}"
        )

    return RangefinderConfig(
        has_topic, topic_summary, configured_hardware, param_summary, report_summary
    )


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
        value = round(float(params["MPC_ALT_MODE"]))
    except (TypeError, ValueError):
        value = -1

    label, detail = _MPC_ALT_MODE_VALUES.get(
        value,
        ("unknown", "Unrecognized MPC_ALT_MODE value in this log."),
    )

    related_names = ["MPC_HOLD_MAX_XY", "MPC_HOLD_MAX_Z", "EKF2_HGT_REF", "EKF2_RNG_CTRL"]
    related = [
        f"{name}={format_param_value(params[name])}" for name in related_names if name in params
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
            if round(float(params[name])) == 0:
                return f"{name}=0"
        except (TypeError, ValueError):
            continue
    return None


def logged_parameter_rows(
    ulog: ULog, prefixes: tuple[str, ...] = ("EKF2_", "MPC_", "SENS_", "SDLOG_")
) -> list[ParameterRow]:
    rows: list[ParameterRow] = []
    for name, value in sorted(initial_parameters(ulog).items()):
        prefix = next((item for item in prefixes if name.startswith(item)), None)
        if prefix is None:
            continue
        rows.append(
            ParameterRow(prefix=prefix.rstrip("_"), name=name, value=format_param_value(value))
        )
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


def shade_mode_background(
    ax_or_axes,
    ulog: ULog,
    base_timestamp: int,
    enabled: bool = True,
    alpha: float = _MODE_SHADING_ALPHA,
    highlight_values: set[int] | None = None,
    highlight_alpha: float = 0.16,
):
    if not enabled:
        return

    spans = nav_state_spans(ulog, base_timestamp)
    if not spans:
        return

    highlighted = {14} if highlight_values is None else highlight_values
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
            ax.axvspan(
                start,
                end,
                color=nav_state_color(span.value),
                alpha=alpha,
                lw=0,
                zorder=0,
            )
        for span in spans:
            if span.value not in highlighted:
                continue
            start = max(span.start_s, xlim[0])
            end = min(span.end_s, xlim[1])
            if end <= start:
                continue
            color = nav_state_color(span.value)
            ax.axvspan(start, end, color=color, alpha=highlight_alpha, lw=0, zorder=0.1)
            ax.axvline(start, color=color, lw=1.0, ls=":", alpha=0.9, zorder=0.2)
            ax.axvline(end, color=color, lw=1.0, ls=":", alpha=0.9, zorder=0.2)
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


def local_minus_ev_position_delta(
    odom,
    local_position,
    base_timestamp: int,
    ev_position_field: str,
    local_position_field: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    if (
        odom is None
        or local_position is None
        or ev_position_field not in odom.data
        or local_position_field not in local_position.data
    ):
        return None

    t_ev = t_rel(odom, base_timestamp)
    ev_position = arr(odom, ev_position_field)
    t_local = t_rel(local_position, base_timestamp)
    local_value = arr(local_position, local_position_field)
    count = min(len(t_ev), len(ev_position))
    if count == 0:
        return None

    t_ev = t_ev[:count]
    ev_position = ev_position[:count]
    local_at_ev = interp_at(t_local, local_value, t_ev)
    return t_ev, local_at_ev - ev_position


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


def boolean_time_percentage_label(t: np.ndarray, values: np.ndarray) -> str | None:
    count = min(len(t), len(values))
    if count == 0:
        return None

    durations = sample_durations(np.asarray(t[:count], dtype=float))
    finite = np.isfinite(durations)
    total_s = float(np.sum(durations[finite]))
    if total_s <= 0.0:
        return None

    vals = np.asarray(values[:count]).astype(bool)
    enabled_s = float(np.sum(durations[finite & vals]))
    return f"{100.0 * enabled_s / total_s:.1f}% enabled time"


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


def local_position_reset_delta_text(ds, field: str, index: int) -> str:
    if field == "xy_reset_counter":
        if "delta_xy[0]" in ds.data and "delta_xy[1]" in ds.data:
            dx = float(ds.data["delta_xy[0]"][index])
            dy = float(ds.data["delta_xy[1]"][index])
            return f"dx={dx:.3f} m, dy={dy:.3f} m"
        return "delta xy not logged"

    if field == "z_reset_counter":
        if "delta_z" in ds.data:
            return f"{float(ds.data['delta_z'][index]):.3f} m z"
        return "delta z not logged"

    if field == "vxy_reset_counter":
        if "delta_vxy[0]" in ds.data and "delta_vxy[1]" in ds.data:
            dvx = float(ds.data["delta_vxy[0]"][index])
            dvy = float(ds.data["delta_vxy[1]"][index])
            return f"dvx={dvx:.3f} m/s, dvy={dvy:.3f} m/s"
        return "delta vxy not logged"

    if field == "vz_reset_counter":
        if "delta_vz" in ds.data:
            return f"{float(ds.data['delta_vz'][index]):.3f} m/s vz"
        return "delta vz not logged"

    if field == "heading_reset_counter":
        if "delta_heading" in ds.data:
            return f"{np.degrees(float(ds.data['delta_heading'][index])):.2f} deg heading"
        return "delta heading not logged"

    if field == "dist_bottom_reset_counter":
        if "delta_dist_bottom" in ds.data:
            return f"{float(ds.data['delta_dist_bottom'][index]):.3f} m HAGL"
        return "delta dist_bottom not logged"

    return "delta not logged"


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
    if local_position is not None:
        t = t_rel(local_position, base_timestamp)
        for field in _LOCAL_POSITION_RESET_COUNTER_FIELDS:
            if field not in local_position.data:
                continue

            counter = arr(local_position, field, int)
            for index in counter_step_indices(counter):
                rows.append(
                    ResetEventRow(
                        f"vehicle_local_position.{field}",
                        float(t[index]),
                        int(counter[index - 1]) if index > 0 else int(counter[index]),
                        int(counter[index]),
                        local_position_reset_delta_text(local_position, field, int(index)),
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
                rows.append(
                    EstimatorExceptionRow("estimator_status", field, meaning, duration, spans)
                )

    flags = get_ds(ulog, "estimator_status_flags")
    if flags is not None:
        for field in flags.data:
            if not (
                field.startswith(("fs_", "reject_")) or field in _ESTIMATOR_STATUS_FLAG_MEANINGS
            ):
                continue
            meaning = _ESTIMATOR_STATUS_FLAG_MEANINGS.get(field, field.replace("_", " "))
            result = field_duration_row(ulog, base_timestamp, flags, field)
            if result is None:
                continue
            duration, _percent, spans = result
            if duration > 0.0:
                rows.append(
                    EstimatorExceptionRow("estimator_status_flags", field, meaning, duration, spans)
                )

    events = get_ds(ulog, "estimator_event_flags")
    if events is not None:
        for field, meaning in _ESTIMATOR_EVENT_WARNING_MEANINGS.items():
            result = field_duration_row(ulog, base_timestamp, events, field)
            if result is None:
                continue
            duration, _percent, spans = result
            if duration > 0.0:
                rows.append(
                    EstimatorExceptionRow("estimator_event_flags", field, meaning, duration, spans)
                )

    gps_status = get_ds(ulog, "estimator_gps_status")
    if gps_status is not None:
        for field, meaning in _GPS_CHECK_FAIL_MEANINGS.items():
            result = field_duration_row(ulog, base_timestamp, gps_status, field)
            if result is None:
                continue
            duration, _percent, spans = result
            if duration > 0.0:
                rows.append(
                    EstimatorExceptionRow("estimator_gps_status", field, meaning, duration, spans)
                )

    return rows


def active_bit_indices(values: np.ndarray) -> list[int]:
    raw_values = np.asarray(values, dtype=np.uint64)
    if raw_values.size == 0:
        return []
    observed = int(np.bitwise_or.reduce(raw_values))
    return [bit for bit in range(observed.bit_length()) if observed & (1 << bit)]


def status_mask_rows(ulog: ULog) -> list[StatusMaskRow]:
    status = get_ds(ulog, "estimator_status")
    rows: list[StatusMaskRow] = []
    if status is None:
        return rows

    for field in available_fields(status, _STATUS_MASK_FIELDS):
        values = arr(status, field, int)
        observed_values = ", ".join(
            str(value) for value in sorted({int(value) for value in values})
        )
        active_bits = active_bit_indices(values)
        rows.append(
            StatusMaskRow(
                field=field,
                observed_values=observed_values if observed_values else "none",
                active_bits=", ".join(f"bit {bit}" for bit in active_bits)
                if active_bits
                else "none",
            )
        )
    return rows


def save_fig(fig, fig_dir: Path, name: str) -> Path:
    png = fig_dir / f"{name}.png"
    pdf = fig_dir / f"{name}.pdf"
    for ax in fig.axes:
        for line in ax.lines:
            line.set_linewidth(line.get_linewidth() * _REPORT_LINE_WIDTH_SCALE)
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png


def setup_axis(ax, title: str, ylabel: str | None = None):
    ax.set_title(title, loc="left", fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)


def setup_latency_axis(ax, values_ms: np.ndarray):
    finite = np.asarray(values_ms, dtype=float)
    finite = finite[np.isfinite(finite)]
    ax.axhline(0, color="gray", lw=0.7, ls=":")

    if not finite.size:
        return

    y_min = float(np.nanmin(finite))
    y_max = float(np.nanmax(finite))
    if max(abs(y_min), abs(y_max)) > 1_000.0:
        ax.set_yscale("symlog", linthresh=1.0)
        return

    span = y_max - y_min
    margin = max(0.1, span * 0.15)
    ax.set_ylim(min(0.0, y_min - margin), max(0.0, y_max + margin))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))


def setup_boolean_axis(ax, title: str):
    setup_axis(ax, title, "0/1")
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 1])


def plot_scalar_trace(
    ax, t: np.ndarray, values: np.ndarray, title: str, ylabel: str, color: str = "tab:blue"
):
    ax.plot(t, values, lw=0.9, color=color)
    setup_axis(ax, title, ylabel)


def plot_boolean_trace(ax, t: np.ndarray, values: np.ndarray, title: str, color: str = "tab:blue"):
    count = min(len(t), len(values))
    if count:
        ax.step(t[:count], np.asarray(values[:count], dtype=int), where="post", lw=1.0, color=color)
    else:
        plot_unavailable(ax, "not logged")
    setup_boolean_axis(ax, title)


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


def plot_optical_flow_measurement_counts(ax, ulog: ULog, base_timestamp: int):
    plotted = False
    all_counts = []
    for topic, color in [
        ("sensor_optical_flow", "tab:green"),
        ("vehicle_optical_flow", "tab:purple"),
    ]:
        datasets = get_all_ds(ulog, topic)
        include_instance = len(datasets) > 1
        for ds in datasets:
            count_result = measurement_count_s(ds, base_timestamp)
            if count_result is None:
                continue

            t_count, counts = count_result
            instance = f"[{ds.multi_id}]" if include_instance else ""
            label = f"{topic}{instance} ({int(counts[-1])} samples)"
            ax.step(t_count, counts, where="post", lw=0.9, color=color, label=label)
            all_counts.append(counts)
            plotted = True

    if plotted:
        setup_integer_value_axis(ax, np.concatenate(all_counts))
    else:
        plot_unavailable(ax, "sensor_optical_flow/vehicle_optical_flow not logged")
    setup_axis(ax, "Optical-flow measurement count", "count")


def plot_unavailable(ax, message: str):
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)


def add_vibration_threshold_bands(ax, green_limit: float = 5.0, yellow_limit: float = 10.0):
    y_min, y_max = ax.get_ylim()
    upper = max(y_max, yellow_limit * 1.15)
    ax.axhspan(0.0, green_limit, color="tab:green", alpha=0.08, lw=0, zorder=0)
    ax.axhspan(green_limit, yellow_limit, color="gold", alpha=0.11, lw=0, zorder=0)
    ax.axhspan(yellow_limit, upper, color="tab:red", alpha=0.08, lw=0, zorder=0)
    ax.axhline(green_limit, color="tab:green", lw=0.8, ls="--", alpha=0.65, label="<5")
    ax.axhline(yellow_limit, color="tab:red", lw=0.8, ls="--", alpha=0.65, label="<10")
    ax.set_ylim(min(y_min, 0.0), upper)


def plot_boolean_group(
    ax,
    t: np.ndarray,
    series: list[tuple[str, np.ndarray, str]],
    title: str,
):
    plot_boolean_group_sources(
        ax,
        [(label, t, values, color) for label, values, color in series],
        title,
    )


def plot_boolean_group_sources(
    ax,
    series: list[tuple[str, np.ndarray, np.ndarray, str]],
    title: str,
):
    plotted = False
    yticks = []
    yticklabels = []
    for index, (label, t, values, color) in enumerate(series):
        values = np.asarray(values, dtype=int)
        t = np.asarray(t, dtype=float)
        count = min(values.size, t.size)
        if count == 0:
            continue
        offset = index * 1.35
        ax.step(t[:count], values[:count] + offset, where="post", lw=1.0, color=color, label=label)
        yticks.append(offset + 0.5)
        yticklabels.append(label)
        plotted = True

    if not plotted:
        plot_unavailable(ax, "no requested status fields logged")
    if yticks:
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, fontsize=14)
        ax.set_ylim(-0.15, yticks[-1] + 0.85)
    setup_axis(ax, title)


def status_flag_series(flags, fields: list[str]) -> list[tuple[str, np.ndarray, str]]:
    colors = [
        "tab:blue",
        "tab:orange",
        "tab:red",
        "tab:green",
        "tab:purple",
        "tab:brown",
        "tab:pink",
    ]
    if flags is None:
        return []
    out = []
    for index, field in enumerate(fields):
        if field in flags.data:
            out.append((field, arr(flags, field, int), colors[index % len(colors)]))
    return out


def control_status_flag_groups(flags) -> list[tuple[str, list[str]]]:
    if flags is None:
        return []

    logged_cs_fields = [field for field in flags.data if field.startswith("cs_")]
    used: set[str] = set()
    groups: list[tuple[str, list[str]]] = []
    for title, requested_fields in _CONTROL_STATUS_GROUPS:
        fields = [field for field in requested_fields if field in flags.data]
        if fields:
            groups.append((title, fields))
            used.update(fields)

    remaining = [field for field in logged_cs_fields if field not in used]
    if remaining:
        groups.append(("Other control status fields", remaining))

    return groups


def aid_test_ratio_fields(ds) -> list[str]:
    if ds is None:
        return []
    return sorted(
        [field for field in ds.data if field == "test_ratio" or field.startswith("test_ratio[")],
        key=lambda field: (len(field), field),
    )


def plot_direct_test_ratios(ax, ds, base_timestamp: int, title: str, prefix: str):
    plotted = False
    if ds is not None:
        t = t_rel(ds, base_timestamp)
        colors = ["tab:blue", "tab:orange", "tab:red", "tab:green"]
        for index, field in enumerate(aid_test_ratio_fields(ds)):
            y = np.where(np.isfinite(arr(ds, field)), arr(ds, field), np.nan)
            ax.plot(t, y, lw=0.9, color=colors[index % len(colors)], label=f"{prefix}.{field}")
            plotted = True
    ax.axhline(1.0, color="black", lw=0.9, ls="--", label="gate (1.0)")
    if plotted:
        ax.set_yscale("symlog", linthresh=0.1)
    else:
        plot_unavailable(ax, f"{prefix}.test_ratio not logged")
    setup_axis(ax, title, "test ratio")
    ax.legend(loc="best", fontsize=7, ncols=2)


def logged_gate_label(ulog: ULog, parameter: str) -> str:
    params = initial_parameters(ulog)
    if parameter in params:
        return f"{parameter}={format_param_value(params[parameter])}"
    return f"{parameter} not logged"


def ev_test_ratio_panel_specs(ulog: ULog) -> list[tuple[str, str, list[str], str]]:
    """Return EV aid-source test-ratio panels available in the log."""
    candidates = [
        (
            "estimator_aid_src_ev_pos",
            f"EV horizontal position innovation test ratios ({logged_gate_label(ulog, 'EKF2_EVP_GATE')})",
            ["test_ratio[0]", "test_ratio[1]"],
            "EV position",
        ),
        (
            "estimator_aid_src_ev_vel",
            f"EV velocity innovation test ratios ({logged_gate_label(ulog, 'EKF2_EVV_GATE')})",
            ["test_ratio[0]", "test_ratio[1]", "test_ratio[2]"],
            "EV velocity",
        ),
        (
            "estimator_aid_src_ev_hgt",
            f"EV height innovation test ratio ({logged_gate_label(ulog, 'EKF2_EVP_GATE')})",
            ["test_ratio"],
            "EV height",
        ),
        (
            "estimator_aid_src_ev_yaw",
            f"EV yaw innovation test ratio ({logged_gate_label(ulog, 'EKF2_HDG_GATE')})",
            ["test_ratio"],
            "EV yaw",
        ),
    ]

    specs = []
    for topic, title, fields, label_prefix in candidates:
        ds = get_ds(ulog, topic)
        if ds is None:
            continue
        available = [field for field in fields if field in ds.data]
        if available:
            specs.append((topic, title, available, label_prefix))
    return specs


def plot_ev_test_ratio_panel(
    ax, ds, base_timestamp: int, title: str, fields: list[str], label_prefix: str
):
    t = t_rel(ds, base_timestamp)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for index, field in enumerate(fields):
        values = np.where(np.isfinite(arr(ds, field)), arr(ds, field), np.nan)
        ax.plot(
            t, values, lw=0.9, color=colors[index % len(colors)], label=f"{label_prefix} {field}"
        )
    ax.axhline(1.0, color="black", lw=0.9, ls="--", label="rejection gate (1.0)")
    ax.set_yscale("symlog", linthresh=0.1)
    setup_axis(ax, title, "test ratio")


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


def finite_signal(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float)


def has_nonconstant_signal(values: np.ndarray) -> bool:
    y = finite_signal(values)
    finite = y[np.isfinite(y)]
    if finite.size == 0:
        return False
    return bool(np.nanmax(finite) - np.nanmin(finite) > 1e-9)


def get_motor_output_set(ulog: ULog, base_timestamp: int) -> ActuatorOutputSet | None:
    motors = get_ds(ulog, "actuator_motors")
    if motors is not None:
        t = t_rel(motors, base_timestamp)
        series: list[SignalSeries] = []
        for index in range(12):
            field = f"control[{index}]"
            if field not in motors.data:
                break
            values = finite_signal(motors.data[field])
            if np.isnan(values).all():
                break
            series.append(SignalSeries(f"Motor {index + 1}", t, values))
        if series:
            return ActuatorOutputSet(
                "Motor outputs (actuator_motors)",
                "normalized",
                "actuator_motors.control[]",
                series,
            )

    outputs = get_ds(ulog, "actuator_outputs")
    if outputs is None:
        return None
    count = 16
    if "noutputs" in outputs.data:
        finite_counts = np.asarray(outputs.data["noutputs"], dtype=float)
        finite_counts = finite_counts[np.isfinite(finite_counts)]
        if finite_counts.size:
            count = min(int(np.nanmax(finite_counts)), count)

    t = t_rel(outputs, base_timestamp)
    series = []
    for index in range(count):
        field = f"output[{index}]"
        if field not in outputs.data:
            break
        values = finite_signal(outputs.data[field])
        if has_nonconstant_signal(values):
            series.append(SignalSeries(f"Output {index}", t, values))

    if not series:
        return None
    return ActuatorOutputSet(
        "Actuator outputs",
        "output",
        "actuator_outputs.output[]",
        series,
    )


def get_esc_rpm_series(ulog: ULog, base_timestamp: int) -> list[SignalSeries]:
    esc = get_ds(ulog, "esc_status")
    if esc is None:
        return []

    fields: list[tuple[int, str]] = []
    if "esc_count" in esc.data:
        count_values = np.asarray(esc.data["esc_count"], dtype=float)
        finite_counts = count_values[np.isfinite(count_values)]
        esc_count = int(np.nanmax(finite_counts)) if finite_counts.size else 0
        fields = [(index, f"esc[{index}].esc_rpm") for index in range(esc_count)]
    else:
        for field in sorted(esc.data):
            if field.startswith("esc[") and field.endswith("].esc_rpm"):
                try:
                    index = int(field.split("[", 1)[1].split("]", 1)[0])
                except (IndexError, ValueError):
                    continue
                fields.append((index, field))

    t = t_rel(esc, base_timestamp)
    series: list[SignalSeries] = []
    for index, field in fields:
        if field not in esc.data:
            continue
        rpm = finite_signal(esc.data[field])
        finite = rpm[np.isfinite(rpm)]
        if finite.size:
            series.append(SignalSeries(f"ESC {index + 1} RPM", t, rpm))
    return series


def actuator_saturation_label(value: int) -> str:
    return _ACTUATOR_SATURATION_LABELS.get(int(value), f"unknown ({int(value)})")


def get_actuator_saturation_series(ulog: ULog, base_timestamp: int) -> list[SignalSeries]:
    status_sets = get_all_ds(ulog, "control_allocator_status")
    if not status_sets:
        return []

    series: list[SignalSeries] = []
    include_instance = len(status_sets) > 1
    for status in status_sets:
        t = t_rel(status, base_timestamp)
        multi_id = getattr(status, "multi_id", 0)
        prefix = f"control_allocator_status[{multi_id}]." if include_instance else ""

        for index in range(16):
            field = f"actuator_saturation[{index}]"
            if field not in status.data:
                continue
            values = finite_signal(status.data[field])
            finite = values[np.isfinite(values)]
            if finite.size:
                series.append(SignalSeries(f"{prefix}{field}", t, values))

    return series


def setup_actuator_saturation_axis(ax, title: str):
    setup_axis(ax, title, "state")
    ticks = sorted(_ACTUATOR_SATURATION_LABELS)
    ax.set_yticks(ticks)
    ax.set_yticklabels([_ACTUATOR_SATURATION_LABELS[tick] for tick in ticks], fontsize=6)
    ax.set_ylim(min(ticks) - 0.35, max(ticks) + 0.35)


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
    Uses distance_sensor as the AGL reference. The offset absorbs the MSL-to-AGL
    reference difference so only dynamic baro error relative to the rangefinder is shown.
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

    Rangefinder data is the preferred altitude reference source. GPS altitude is only
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
    Requires rangefinder-based divergence. GPS altitude is not used for scoring.
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

    propwash = (
        (np.abs(divergence) > divergence_threshold_m) & throttle_mask & np.isfinite(divergence)
    )
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


def quaternion_to_euler_deg(
    ds, prefix: str = "q"
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    fields = [f"{prefix}[{index}]" for index in range(4)]
    if any(field not in ds.data for field in fields):
        return None

    qw, qx, qy, qz = [arr(ds, field) for field in fields]
    norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    norm = np.where(norm > 1e-12, norm, np.nan)
    qw = qw / norm
    qx = qx / norm
    qy = qy / norm
    qz = qz / norm

    roll = np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx**2 + qy**2))
    pitch = np.arcsin(np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy**2 + qz**2))
    return np.degrees(roll), np.degrees(pitch), np.degrees(np.unwrap(yaw))


def attitude_setpoint_euler_deg(ds) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if all(field in ds.data for field in ("roll_body", "pitch_body", "yaw_body")):
        return (
            np.degrees(arr(ds, "roll_body")),
            np.degrees(arr(ds, "pitch_body")),
            np.degrees(np.unwrap(arr(ds, "yaw_body"))),
        )
    return quaternion_to_euler_deg(ds, prefix="q_d")


def plot_position_overview(
    ulog: ULog, base_timestamp: int, fig_dir: Path, shade_modes: bool = True
) -> Path:
    """Altitude overview using EKF local-z, baro, optional rangefinder, and GPS."""
    air = get_ds(ulog, "vehicle_air_data")
    lpos = get_ds(ulog, "vehicle_local_position")
    gps = get_ds(ulog, "vehicle_gps_position")
    range_result = get_range(ulog, base_timestamp)

    fig, axes = plt.subplots(4, 1, figsize=(10.5, 8.8), sharex=True)

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
            axes[0].plot(
                ta,
                baro_agl - offset,
                label="Baro (aligned to rangefinder)",
                lw=1.0,
                color="tab:orange",
                alpha=0.8,
            )
        else:
            axes[0].plot(ta, baro_agl, label="Baro MSL", lw=1.0, color="tab:orange", alpha=0.8)
    if gps:
        tg = t_rel(gps, base_timestamp)
        axes[0].plot(
            tg, gps_alt_m(gps), label="GPS MSL (ref only)", lw=0.8, color="gray", ls=":", alpha=0.5
        )
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

    # Panel 4: terrain/HAGL estimator states
    terrain_hagl_series = terrain_hagl_state_series(ulog, base_timestamp)
    for series in terrain_hagl_series:
        axes[3].plot(series.t, series.values, label=series.label, lw=1.0, alpha=0.85)
    if not terrain_hagl_series:
        plot_unavailable(
            axes[3],
            "vehicle_local_position.dist_bottom / vehicle_global_position terrain not logged",
        )
    setup_axis(axes[3], "Terrain and HAGL state comparison", "m")
    axes[3].set_xlabel("flight-log relative time [s]")
    shade_mode_background(axes, ulog, base_timestamp, enabled=shade_modes)

    for ax in axes:
        handles, _labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best", fontsize=8)

    return save_fig(fig, fig_dir, "position_overview")


def terrain_hagl_state_series(ulog: ULog, base_timestamp: int) -> list[SignalSeries]:
    """Return terrain and HAGL states in log-relative metres when available."""
    lpos = get_ds(ulog, "vehicle_local_position")
    gpos = get_ds(ulog, "vehicle_global_position")
    estimator_states = get_ds(ulog, "estimator_states")
    series: list[SignalSeries] = []

    if estimator_states is not None and "states[24]" in estimator_states.data:
        series.append(
            SignalSeries(
                "terrain state: estimator_states.states[24] (_state.terrain)",
                t_rel(estimator_states, base_timestamp),
                arr(estimator_states, "states[24]"),
            )
        )

    if lpos is not None and "dist_bottom" in lpos.data:
        t = t_rel(lpos, base_timestamp)
        dist_bottom = arr(lpos, "dist_bottom")
        series.append(SignalSeries("HAGL: vehicle_local_position.dist_bottom", t, dist_bottom))

        if "z" in lpos.data:
            series.append(
                SignalSeries(
                    "terrain local NED: z + dist_bottom",
                    t,
                    arr(lpos, "z") + dist_bottom,
                )
            )

    if gpos is None:
        return series

    tg = t_rel(gpos, base_timestamp)
    terrain_alt = None
    if "terrain_alt" in gpos.data:
        terrain_alt = arr(gpos, "terrain_alt")
        if "terrain_alt_valid" in gpos.data:
            terrain_alt = np.where(
                arr(gpos, "terrain_alt_valid", int).astype(bool), terrain_alt, np.nan
            )

        finite = np.isfinite(terrain_alt)
        if finite.any():
            terrain_ref = float(terrain_alt[finite][0])
            series.append(
                SignalSeries(
                    "terrain global NED rel start: -delta terrain_alt",
                    tg,
                    terrain_ref - terrain_alt,
                )
            )

    if terrain_alt is not None and "alt" in gpos.data:
        alt = arr(gpos, "alt")
        if "alt_valid" in gpos.data:
            alt = np.where(arr(gpos, "alt_valid", int).astype(bool), alt, np.nan)
        series.append(
            SignalSeries(
                "HAGL: vehicle_global_position.alt - terrain_alt",
                tg,
                alt - terrain_alt,
            )
        )

    return series


def plot_local_xy_position(
    ulog: ULog, base_timestamp: int, fig_dir: Path, shade_modes: bool = True
) -> Path:
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
            axes[0].scatter(
                y[finite_xy][0], x[finite_xy][0], s=28, color="tab:green", label="start", zorder=3
            )
            axes[0].scatter(
                y[finite_xy][-1], x[finite_xy][-1], s=28, color="tab:red", label="end", zorder=3
            )
            axes[0].set_aspect("equal", adjustable="box")

        axes[1].plot(t, x, lw=0.9, color="tab:blue", label="local x")
        axes[2].plot(t, y, lw=0.9, color="tab:orange", label="local y")
    else:
        for ax in axes:
            ax.text(
                0.5,
                0.5,
                "vehicle_local_position x/y not logged",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

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


def plot_nav_state_timeline(
    ulog: ULog, base_timestamp: int, fig_dir: Path, shade_modes: bool = True
) -> Path:
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
        add_system_time_axis(ax, ulog, base_timestamp)
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
    add_system_time_axis(ax, ulog, base_timestamp)

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
            ax.text(
                0.5,
                0.5,
                "vehicle_local_position not logged",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
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

    for idx, (field, label, color, sp_color) in enumerate(
        zip(fields, axis_labels, colors, setpoint_colors)
    ):
        estimate = arr(lpos, field) if field in lpos.data else None
        setpoint = arr(sp, field) if sp is not None and field in sp.data else None
        ax_value = axes[idx]
        ax_delta = axes[idx + 3]

        if estimate is not None:
            ax_value.plot(t, estimate, lw=1.0, color=color, label=f"{field} estimate")
        else:
            ax_value.text(
                0.5,
                0.5,
                f"vehicle_local_position.{field} not logged",
                ha="center",
                va="center",
                transform=ax_value.transAxes,
            )

        if setpoint is not None and tsp is not None:
            ax_value.step(
                tsp, setpoint, where="post", lw=1.0, color=sp_color, label=f"{field} setpoint"
            )
        elif setpoint_missing:
            ax_value.text(
                0.01,
                0.90,
                "vehicle_local_position_setpoint not logged",
                transform=ax_value.transAxes,
                fontsize=8,
                color="tab:red",
            )

        if estimate is not None and setpoint is not None and tsp is not None:
            delta = estimate - interp_step_previous(tsp, setpoint, t)
            ax_delta.plot(t, delta, lw=0.9, color=color, label=f"{field} estimate - setpoint")
            ax_delta.axhline(0.0, color="gray", lw=0.7, ls=":")
        else:
            ax_delta.text(
                0.5,
                0.5,
                f"{field} delta unavailable",
                ha="center",
                va="center",
                transform=ax_delta.transAxes,
            )
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


def plot_position_tracking(
    ulog: ULog, base_timestamp: int, fig_dir: Path, shade_modes: bool = True
) -> Path:
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


def plot_velocity_tracking(
    ulog: ULog, base_timestamp: int, fig_dir: Path, shade_modes: bool = True
) -> Path:
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


def finish_px4_setpoint_axes(
    axes: np.ndarray, ulog: ULog, base_timestamp: int, shade_modes: bool = True
):
    axes[-1].set_xlabel("flight-log relative time [s]")
    spans = nav_state_spans(ulog, base_timestamp)
    if spans:
        for ax in axes:
            ax.set_xlim(spans[0].start_s, spans[-1].end_s)
    shade_mode_background(axes, ulog, base_timestamp, enabled=shade_modes, alpha=0.12)
    for ax in axes:
        handles, _labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best", fontsize=7, ncols=2)
    add_system_time_axis(axes[-1], ulog, base_timestamp)


def plot_px4_setpoint_commands(
    ulog: ULog, base_timestamp: int, fig_dir: Path, shade_modes: bool = True
) -> Path:
    trajectory = get_ds(ulog, "trajectory_setpoint")
    offboard = get_ds(ulog, "offboard_control_mode")
    lpos_sp = get_ds(ulog, "vehicle_local_position_setpoint")

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(10.5, 12.0),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2.2, 1.35, 1.7, 2.2, 1.35]},
    )
    axes = np.ravel(np.atleast_1d(axes))
    colors = [
        "tab:blue",
        "tab:orange",
        "tab:red",
        "tab:green",
        "tab:purple",
        "tab:brown",
        "tab:pink",
    ]

    if trajectory is not None:
        t_traj = t_rel(trajectory, base_timestamp)
        for index, color in enumerate(colors):
            field = f"velocity[{index}]"
            if field in trajectory.data:
                axes[0].plot(
                    t_traj,
                    arr(trajectory, field),
                    lw=0.9,
                    color=color,
                    label=f"trajectory_setpoint.{field}",
                )
        for index, color in enumerate(colors):
            field = f"position[{index}]"
            if field in trajectory.data:
                axes[1].step(
                    t_traj,
                    arr(trajectory, field),
                    where="post",
                    lw=0.9,
                    color=color,
                    label=f"trajectory_setpoint.{field}",
                )
    if not axes[0].has_data():
        plot_unavailable(axes[0], "trajectory_setpoint.velocity[] not logged")
    axes[0].axhline(0.0, color="gray", lw=0.7, ls=":")
    setup_axis(axes[0], "Trajectory setpoint velocity", "m/s")
    if not axes[1].has_data():
        plot_unavailable(axes[1], "trajectory_setpoint.position[] not logged")
    setup_axis(axes[1], "Trajectory setpoint position", "m")

    if offboard is not None:
        t_offboard = t_rel(offboard, base_timestamp)
        fields = [
            "position",
            "velocity",
            "acceleration",
            "attitude",
            "body_rate",
            "thrust_and_torque",
            "direct_actuator",
        ]
        series = [
            (field, t_offboard, arr(offboard, field, int), colors[index % len(colors)])
            for index, field in enumerate(fields)
            if field in offboard.data
        ]
        plot_boolean_group_sources(axes[2], series, "Offboard control mode")
    else:
        plot_unavailable(axes[2], "offboard_control_mode not logged")
        setup_boolean_axis(axes[2], "Offboard control mode")

    if lpos_sp is not None:
        t_lpos_sp = t_rel(lpos_sp, base_timestamp)
        for field, color in [("vx", "tab:blue"), ("vy", "tab:orange"), ("vz", "tab:red")]:
            if field in lpos_sp.data:
                axes[3].step(
                    t_lpos_sp,
                    arr(lpos_sp, field),
                    where="post",
                    lw=0.9,
                    color=color,
                    label=f"vehicle_local_position_setpoint.{field}",
                )
        for field, color in [("x", "tab:blue"), ("y", "tab:orange"), ("z", "tab:red")]:
            if field in lpos_sp.data:
                axes[4].step(
                    t_lpos_sp,
                    arr(lpos_sp, field),
                    where="post",
                    lw=0.9,
                    color=color,
                    label=f"vehicle_local_position_setpoint.{field}",
                )
    if not axes[3].has_data():
        plot_unavailable(axes[3], "vehicle_local_position_setpoint.vx/vy/vz not logged")
    axes[3].axhline(0.0, color="gray", lw=0.7, ls=":")
    setup_axis(axes[3], "Vehicle local position setpoint velocity", "m/s")
    if not axes[4].has_data():
        plot_unavailable(axes[4], "vehicle_local_position_setpoint.x/y/z not logged")
    setup_axis(axes[4], "Vehicle local position setpoint position", "m")

    finish_px4_setpoint_axes(axes, ulog, base_timestamp, shade_modes=shade_modes)
    return save_fig(fig, fig_dir, "px4_setpoint_commands")


def plot_px4_setpoint_response(
    ulog: ULog, base_timestamp: int, fig_dir: Path, shade_modes: bool = True
) -> Path:
    trajectory = get_ds(ulog, "trajectory_setpoint")
    lpos_sp = get_ds(ulog, "vehicle_local_position_setpoint")
    lpos = get_ds(ulog, "vehicle_local_position")
    dist = get_ds(ulog, "distance_sensor")
    estimator_states = get_ds(ulog, "estimator_states")
    attitude_sp = get_ds(ulog, "vehicle_attitude_setpoint")

    fig, axes = plt.subplots(
        6,
        1,
        figsize=(10.5, 13.2),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.5, 2.2, 1.7, 1.25, 1.35, 1.35]},
    )
    axes = np.ravel(np.atleast_1d(axes))
    colors = [
        "tab:blue",
        "tab:orange",
        "tab:red",
        "tab:green",
        "tab:purple",
        "tab:brown",
        "tab:pink",
    ]

    if lpos is not None:
        t_lpos = t_rel(lpos, base_timestamp)
        if "z" in lpos.data:
            axes[0].plot(
                t_lpos,
                arr(lpos, "z"),
                lw=0.9,
                color="tab:blue",
                label="vehicle_local_position.z",
            )
        if "vz" in lpos.data:
            axes[1].plot(
                t_lpos,
                arr(lpos, "vz"),
                lw=0.9,
                color="tab:blue",
                label="vehicle_local_position.vz",
            )
        if "z_deriv" in lpos.data:
            axes[1].plot(
                t_lpos,
                arr(lpos, "z_deriv"),
                lw=0.9,
                color="tab:green",
                label="vehicle_local_position.z_deriv",
            )
        if "dist_bottom" in lpos.data:
            axes[2].plot(
                t_lpos,
                arr(lpos, "dist_bottom"),
                lw=0.9,
                color="tab:orange",
                label="vehicle_local_position.dist_bottom",
            )
        if "z" in lpos.data and "dist_bottom" in lpos.data:
            axes[2].plot(
                t_lpos,
                arr(lpos, "z") + arr(lpos, "dist_bottom"),
                lw=0.9,
                color="tab:blue",
                label="vehicle_local_position.z + dist_bottom",
            )
        reset_fields = [
            "xy_reset_counter",
            "z_reset_counter",
            "vxy_reset_counter",
            "vz_reset_counter",
            "heading_reset_counter",
            "dist_bottom_reset_counter",
        ]
        for index, field in enumerate(reset_fields):
            if field in lpos.data:
                axes[3].step(
                    t_lpos,
                    arr(lpos, field, int),
                    where="post",
                    lw=0.8,
                    color=colors[index % len(colors)],
                    label=f"vehicle_local_position.{field}",
                )
    if lpos_sp is not None:
        t_lpos_sp = t_rel(lpos_sp, base_timestamp)
        if "z" in lpos_sp.data:
            axes[0].step(
                t_lpos_sp,
                arr(lpos_sp, "z"),
                where="post",
                lw=0.9,
                color="tab:red",
                label="vehicle_local_position_setpoint.z",
            )
        if "vz" in lpos_sp.data:
            axes[1].step(
                t_lpos_sp,
                arr(lpos_sp, "vz"),
                where="post",
                lw=0.9,
                color="tab:red",
                label="vehicle_local_position_setpoint.vz",
            )
    if trajectory is not None and "velocity[2]" in trajectory.data:
        axes[1].step(
            t_rel(trajectory, base_timestamp),
            arr(trajectory, "velocity[2]"),
            where="post",
            lw=0.9,
            ls="--",
            color="tab:purple",
            label="trajectory_setpoint.velocity[2]",
        )
    if dist is not None and "current_distance" in dist.data:
        axes[2].plot(
            t_rel(dist, base_timestamp),
            arr(dist, "current_distance"),
            lw=0.8,
            color="tab:red",
            label="distance_sensor.current_distance",
        )
    if estimator_states is not None and "states[24]" in estimator_states.data:
        axes[2].step(
            t_rel(estimator_states, base_timestamp),
            arr(estimator_states, "states[24]"),
            where="post",
            lw=0.9,
            ls="--",
            color="tab:purple",
            label="estimator_states.states[24] (_state.terrain)",
        )
    if not axes[0].has_data():
        plot_unavailable(axes[0], "vehicle_local_position.z not logged")
    setup_axis(axes[0], "Vertical position and setpoint (NED z)", "m")
    if not axes[1].has_data():
        plot_unavailable(axes[1], "vehicle_local_position.vz not logged")
    axes[1].axhline(0.0, color="gray", lw=0.7, ls=":")
    setup_axis(axes[1], "Vertical velocity state and setpoints (NED vz)", "m/s")
    if not axes[2].has_data():
        plot_unavailable(axes[2], "range/HAGL/terrain fields not logged")
    axes[2].axhline(0.0, color="gray", lw=0.7, ls=":")
    setup_axis(axes[2], "Range height and terrain vertical state", "m")
    if not axes[3].has_data():
        plot_unavailable(axes[3], "vehicle_local_position reset counters not logged")
    setup_axis(axes[3], "Vehicle local position reset counters", "count")
    setup_integer_value_axis(
        axes[3],
        np.concatenate([np.asarray(line.get_ydata()) for line in axes[3].lines])
        if axes[3].lines
        else np.array([]),
    )

    sp_euler = attitude_setpoint_euler_deg(attitude_sp) if attitude_sp is not None else None
    if attitude_sp is not None:
        t_att_sp = t_rel(attitude_sp, base_timestamp)
        if sp_euler is not None:
            for index, (label, color) in enumerate(
                [("roll", "tab:blue"), ("pitch", "tab:orange"), ("yaw", "tab:red")]
            ):
                axes[4].step(
                    t_att_sp,
                    sp_euler[index],
                    where="post",
                    lw=0.9,
                    color=color,
                    label=f"vehicle_attitude_setpoint {label}",
                )
    if not axes[4].has_data():
        plot_unavailable(axes[4], "vehicle_attitude_setpoint.q_d[] not logged")
    setup_axis(axes[4], "Vehicle attitude setpoint Euler angles", "deg")

    if attitude_sp is not None:
        t_att_sp = t_rel(attitude_sp, base_timestamp)
        for index, color in enumerate(colors[:3]):
            field = f"thrust_body[{index}]"
            if field in attitude_sp.data:
                axes[5].step(
                    t_att_sp,
                    arr(attitude_sp, field),
                    where="post",
                    lw=0.9,
                    color=color,
                    label=f"vehicle_attitude_setpoint.{field}",
                )
    if not axes[5].has_data():
        plot_unavailable(axes[5], "vehicle_attitude_setpoint.thrust_body[] not logged")
    setup_axis(axes[5], "Vehicle attitude setpoint thrust body", "normalized")

    finish_px4_setpoint_axes(axes, ulog, base_timestamp, shade_modes=shade_modes)
    return save_fig(fig, fig_dir, "px4_setpoint_response")


def plot_manual_control_inputs(
    ulog: ULog, base_timestamp: int, fig_dir: Path, shade_modes: bool = True
) -> Path:
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
        axes[0].text(
            0.5,
            0.5,
            "manual_control_setpoint not logged",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
        )
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
        axes[0].text(
            0.5,
            0.5,
            "manual_control_setpoint input fields not logged",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
        )
        setup_axis(axes[0], "Manual Control Inputs unavailable", "input")

    axes[-1].set_xlabel("flight-log relative time [s]")
    shade_mode_background(axes, ulog, base_timestamp, enabled=shade_modes)
    add_system_time_axis(axes[-1], ulog, base_timestamp)
    return save_fig(fig, fig_dir, "manual_control_inputs")


def plot_attitude_overview(
    ulog: ULog, base_timestamp: int, fig_dir: Path, shade_modes: bool = True
) -> Path:
    attitude = get_ds(ulog, "vehicle_attitude")
    attitude_sp = get_ds(ulog, "vehicle_attitude_setpoint")
    angular_velocity = get_ds(ulog, "vehicle_angular_velocity")
    lpos = get_ds(ulog, "vehicle_local_position")

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(10.5, 9.2),
        sharex=True,
        constrained_layout=True,
    )

    euler = quaternion_to_euler_deg(attitude) if attitude is not None else None
    t_att = t_rel(attitude, base_timestamp) if attitude is not None else None
    sp_euler = attitude_setpoint_euler_deg(attitude_sp) if attitude_sp is not None else None
    t_sp = t_rel(attitude_sp, base_timestamp) if attitude_sp is not None else None
    labels = [("Roll", "tab:blue"), ("Pitch", "tab:green"), ("Yaw", "tab:purple")]
    for index, (label, color) in enumerate(labels):
        ax = axes[index]
        if euler is not None and t_att is not None:
            ax.plot(t_att, euler[index], lw=0.9, color=color, label=f"{label.lower()} estimate")
        if sp_euler is not None and t_sp is not None:
            ax.step(
                t_sp,
                sp_euler[index],
                where="post",
                lw=0.9,
                ls="--",
                color="tab:orange",
                label=f"{label.lower()} setpoint",
            )
        if not ax.has_data():
            plot_unavailable(ax, f"{label.lower()} attitude not logged")
        setup_axis(ax, f"Attitude {label}", "deg")

    if angular_velocity is not None:
        t_rate = t_rel(angular_velocity, base_timestamp)
        for index, (label, color) in enumerate(
            [("roll rate", "tab:blue"), ("pitch rate", "tab:green"), ("yaw rate", "tab:purple")]
        ):
            field = f"xyz[{index}]"
            if field in angular_velocity.data:
                axes[3].plot(
                    t_rate,
                    np.degrees(arr(angular_velocity, field)),
                    lw=0.8,
                    color=color,
                    label=label,
                )
    if not axes[3].has_data():
        plot_unavailable(axes[3], "vehicle_angular_velocity.xyz[] not logged")
    setup_axis(axes[3], "Body angular velocity", "deg/s")

    if lpos is not None:
        t_lpos = t_rel(lpos, base_timestamp)
        plotted_motion = False
        if all(field in lpos.data for field in ("vx", "vy")):
            h_speed = np.sqrt(arr(lpos, "vx") ** 2 + arr(lpos, "vy") ** 2)
            axes[4].plot(t_lpos, h_speed, lw=0.9, color="tab:blue", label="horizontal speed")
            plotted_motion = True
        if all(field in lpos.data for field in ("ax", "ay")):
            h_accel = np.sqrt(arr(lpos, "ax") ** 2 + arr(lpos, "ay") ** 2)
            axes[4].plot(t_lpos, h_accel, lw=0.9, color="tab:red", label="horizontal acceleration")
            plotted_motion = True
        if not plotted_motion:
            plot_unavailable(axes[4], "vehicle_local_position horizontal motion fields not logged")
    else:
        plot_unavailable(axes[4], "vehicle_local_position not logged")
    setup_axis(axes[4], "Horizontal motion context", "m/s or m/s^2")

    axes[-1].set_xlabel("flight-log relative time [s]")
    shade_mode_background(axes, ulog, base_timestamp, enabled=shade_modes)
    for ax in axes:
        handles, _labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best", fontsize=7, ncols=2)
    add_system_time_axis(axes[-1], ulog, base_timestamp)
    return save_fig(fig, fig_dir, "attitude_overview")


def front_rear_motor_indices(ulog: ULog, motor_count: int) -> tuple[list[int], list[int]]:
    params = getattr(ulog, "initial_parameters", {})
    try:
        configured_count = int(params.get("CA_ROTOR_COUNT", motor_count))
    except (TypeError, ValueError):
        configured_count = motor_count

    front: list[int] = []
    rear: list[int] = []
    for index in range(min(max(configured_count, 0), motor_count)):
        try:
            x_position = float(params[f"CA_ROTOR{index}_PX"])
        except (KeyError, TypeError, ValueError):
            continue
        if x_position > 1e-4:
            front.append(index)
        elif x_position < -1e-4:
            rear.append(index)
    return front, rear


def plot_forward_lurch_diagnostics(
    ulog: ULog, base_timestamp: int, fig_dir: Path, shade_modes: bool = True
) -> Path:
    attitude = get_ds(ulog, "vehicle_attitude")
    attitude_sp = get_ds(ulog, "vehicle_attitude_setpoint")
    angular_velocity = get_ds(ulog, "vehicle_angular_velocity")
    rates_sp = get_ds(ulog, "vehicle_rates_setpoint")
    torque_sp = get_ds(ulog, "vehicle_torque_setpoint")
    allocator = get_ds(ulog, "control_allocator_status")
    motors = get_ds(ulog, "actuator_motors")
    bias = get_ds(ulog, "estimator_sensor_bias")

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(10.5, 11.8),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.45, 1.45, 1.25, 1.5, 1.35]},
    )
    axes = np.ravel(np.atleast_1d(axes))

    measured_euler = quaternion_to_euler_deg(attitude) if attitude is not None else None
    setpoint_euler = attitude_setpoint_euler_deg(attitude_sp) if attitude_sp is not None else None
    if measured_euler is not None:
        axes[0].plot(
            t_rel(attitude, base_timestamp),
            measured_euler[1],
            lw=1.0,
            color="tab:blue",
            label="measured pitch",
        )
    if setpoint_euler is not None:
        axes[0].step(
            t_rel(attitude_sp, base_timestamp),
            setpoint_euler[1],
            where="post",
            lw=1.0,
            ls="--",
            color="tab:orange",
            label="pitch setpoint",
        )
    if not axes[0].has_data():
        plot_unavailable(axes[0], "pitch attitude/setpoint not logged")
    axes[0].axhline(0.0, color="gray", lw=0.7, ls=":")
    setup_axis(axes[0], "Pitch attitude tracking", "deg")

    if angular_velocity is not None and "xyz[1]" in angular_velocity.data:
        axes[1].plot(
            t_rel(angular_velocity, base_timestamp),
            np.degrees(arr(angular_velocity, "xyz[1]")),
            lw=1.0,
            color="tab:blue",
            label="measured pitch rate",
        )
    if rates_sp is not None and "pitch" in rates_sp.data:
        axes[1].step(
            t_rel(rates_sp, base_timestamp),
            np.degrees(arr(rates_sp, "pitch")),
            where="post",
            lw=1.0,
            ls="--",
            color="tab:orange",
            label="pitch-rate setpoint",
        )
    if not axes[1].has_data():
        plot_unavailable(axes[1], "pitch-rate measurement/setpoint not logged")
    axes[1].axhline(0.0, color="gray", lw=0.7, ls=":")
    setup_axis(axes[1], "Pitch-rate tracking", "deg/s")

    if torque_sp is not None and "xyz[1]" in torque_sp.data:
        axes[2].step(
            t_rel(torque_sp, base_timestamp),
            arr(torque_sp, "xyz[1]"),
            where="post",
            lw=1.0,
            color="tab:purple",
            label="commanded pitch torque",
        )
    if allocator is not None and "unallocated_torque[1]" in allocator.data:
        axes[2].step(
            t_rel(allocator, base_timestamp),
            arr(allocator, "unallocated_torque[1]"),
            where="post",
            lw=0.85,
            ls="--",
            color="tab:red",
            label="unallocated pitch torque",
        )
    if not axes[2].has_data():
        plot_unavailable(axes[2], "pitch torque setpoint/allocation residual not logged")
    axes[2].axhline(0.0, color="gray", lw=0.7, ls=":")
    setup_axis(axes[2], "Pitch torque command and allocation residual", "normalized")

    motor_fields = []
    if motors is not None:
        for index in range(12):
            field = f"control[{index}]"
            if field not in motors.data or np.isnan(arr(motors, field)).all():
                break
            motor_fields.append(field)
    front_indices, rear_indices = front_rear_motor_indices(ulog, len(motor_fields))
    if motors is not None and front_indices and rear_indices:
        t_motors = t_rel(motors, base_timestamp)
        front_mean = np.nanmean(
            np.vstack([arr(motors, motor_fields[index]) for index in front_indices]), axis=0
        )
        rear_mean = np.nanmean(
            np.vstack([arr(motors, motor_fields[index]) for index in rear_indices]), axis=0
        )
        axes[3].plot(
            t_motors,
            front_mean,
            lw=0.95,
            color="tab:blue",
            label=f"front mean (motors {', '.join(str(i + 1) for i in front_indices)})",
        )
        axes[3].plot(
            t_motors,
            rear_mean,
            lw=0.95,
            color="tab:orange",
            label=f"rear mean (motors {', '.join(str(i + 1) for i in rear_indices)})",
        )
        axes[3].plot(
            t_motors,
            front_mean - rear_mean,
            lw=1.0,
            color="tab:green",
            label="front - rear differential",
        )
    else:
        plot_unavailable(axes[3], "actuator_motors and signed CA_ROTOR*_PX geometry not logged")
    axes[3].axhline(0.0, color="gray", lw=0.7, ls=":")
    setup_axis(axes[3], "Front/rear motor command comparison", "normalized")

    if bias is not None:
        t_bias = t_rel(bias, base_timestamp)
        for index, color in enumerate(["tab:blue", "tab:orange", "tab:green"]):
            field = f"accel_bias[{index}]"
            if field in bias.data:
                axes[4].plot(
                    t_bias,
                    arr(bias, field),
                    lw=1.0,
                    color=color,
                    label=f"accel bias {'XYZ'[index]}",
                )
    if not axes[4].has_data():
        plot_unavailable(axes[4], "estimator_sensor_bias.accel_bias[] not logged")
    axes[4].axhline(0.0, color="gray", lw=0.7, ls=":")
    setup_axis(axes[4], "EKF accelerometer in-run bias", "m/s$^2$")

    axes[-1].set_xlabel("flight-log relative time [s]")
    shade_mode_background(axes, ulog, base_timestamp, enabled=shade_modes)
    for ax in axes:
        handles, _labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best", fontsize=7, ncols=2)
    add_system_time_axis(axes[-1], ulog, base_timestamp)
    return save_fig(fig, fig_dir, "forward_lurch_diagnostics")


def plot_actuator_outputs(
    ulog: ULog, base_timestamp: int, fig_dir: Path, shade_modes: bool = True
) -> Path:
    outputs = get_motor_output_set(ulog, base_timestamp)
    saturation_series = get_actuator_saturation_series(ulog, base_timestamp)
    thrust_result = get_thrust(ulog, base_timestamp)
    rpm_series = get_esc_rpm_series(ulog, base_timestamp)

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(10.5, 8.4),
        sharex=True,
        constrained_layout=True,
    )

    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
    ]

    if outputs is not None:
        for index, signal in enumerate(outputs.series):
            count = min(signal.t.size, signal.values.size)
            if count:
                axes[0].plot(
                    signal.t[:count],
                    signal.values[:count],
                    lw=0.75,
                    color=colors[index % len(colors)],
                    label=signal.label,
                )
        if outputs.ylabel == "normalized":
            axes[0].set_ylim(-1.05, 1.05)
        setup_axis(axes[0], outputs.title, outputs.ylabel)
    else:
        plot_unavailable(axes[0], "actuator_motors or varying actuator_outputs not logged")
        setup_axis(axes[0], "Motor / actuator outputs unavailable", "output")

    if saturation_series:
        for index, signal in enumerate(saturation_series):
            count = min(signal.t.size, signal.values.size)
            if count:
                axes[1].step(
                    signal.t[:count],
                    signal.values[:count],
                    where="post",
                    lw=0.75,
                    color=colors[index % len(colors)],
                    label=signal.label,
                )
    else:
        plot_unavailable(axes[1], "control_allocator_status.actuator_saturation[] not logged")
    setup_actuator_saturation_axis(axes[1], "Actuator saturation state (control_allocator_status)")

    if thrust_result is not None:
        t_thr, thrust = thrust_result
        axes[2].plot(t_thr, thrust, lw=0.9, color="tab:brown", label="thrust")
        finite = thrust[np.isfinite(thrust)]
        if finite.size and np.nanmin(finite) >= -0.05 and np.nanmax(finite) <= 1.05:
            axes[2].set_ylim(-0.05, 1.05)
    else:
        plot_unavailable(
            axes[2], "vehicle_thrust_setpoint.xyz[2] or actuator_controls_0.control[3] not logged"
        )
    setup_axis(axes[2], "Thrust setpoint", "normalized")

    if rpm_series:
        for index, signal in enumerate(rpm_series):
            count = min(signal.t.size, signal.values.size)
            if count:
                axes[3].plot(
                    signal.t[:count],
                    signal.values[:count],
                    lw=0.8,
                    color=colors[index % len(colors)],
                    label=signal.label,
                )
    else:
        plot_unavailable(axes[3], "esc_status.esc[i].esc_rpm not logged")
    setup_axis(axes[3], "ESC RPM", "RPM")
    axes[-1].set_xlabel("flight-log relative time [s]")

    shade_mode_background(axes, ulog, base_timestamp, enabled=shade_modes)
    for axis_index, ax in enumerate(axes):
        handles, _labels = ax.get_legend_handles_labels()
        if handles:
            ncols = 4 if axis_index == 1 else 2
            ax.legend(loc="best", fontsize=6 if axis_index == 1 else 7, ncols=ncols)
    add_system_time_axis(axes[-1], ulog, base_timestamp)
    return save_fig(fig, fig_dir, "actuator_outputs")


def plot_terrain_estimate(
    ulog: ULog, base_timestamp: int, fig_dir: Path, shade_modes: bool = True
) -> Path:
    """Plot published terrain/HAGL estimate from vehicle_local_position."""
    lpos = get_ds(ulog, "vehicle_local_position")
    lpos_sp = get_ds(ulog, "vehicle_local_position_setpoint")
    trajectory_sp = get_ds(ulog, "trajectory_setpoint")
    gpos = get_ds(ulog, "vehicle_global_position")
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(10.5, 7.2),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={
            "height_ratios": [1.35, 1.0] + [_BOOLEAN_PANEL_HEIGHT_RATIO] * 2,
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
        setup_axis(axes[1], "Terrain-hold HAGL setpoint unavailable", "m")
        setup_boolean_axis(axes[2], "dist_bottom_valid")
        setup_boolean_axis(axes[3], "dist_bottom_sensor_bitfield != 0")
        axes[3].set_xlabel("flight-log relative time [s]")
        shade_mode_background(axes, ulog, base_timestamp, enabled=shade_modes)
        return save_fig(fig, fig_dir, "terrain_estimate")

    t = t_rel(lpos, base_timestamp)
    dist_bottom = arr(lpos, "dist_bottom")
    valid = (
        arr(lpos, "dist_bottom_valid", int).astype(bool)
        if "dist_bottom_valid" in lpos.data
        else None
    )

    axes[0].plot(
        t, dist_bottom, lw=1.0, color="tab:blue", alpha=0.75, label="dist_bottom (HAGL estimate)"
    )

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

    local_terrain_z = None
    if "z" in lpos.data:
        local_z = arr(lpos, "z")
        local_terrain_z = local_z + dist_bottom
        axes[0].plot(
            t,
            local_terrain_z,
            lw=1.0,
            color="tab:brown",
            label="z + dist_bottom (terrain, NED down)",
        )
        axes[0].plot(t, local_z, lw=0.8, color="gray", ls=":", alpha=0.8, label="vehicle local z")
    if gpos is not None and "terrain_alt" in gpos.data:
        tg = t_rel(gpos, base_timestamp)
        axes[0].plot(
            tg,
            arr(gpos, "terrain_alt"),
            lw=0.9,
            color="tab:cyan",
            label="vehicle_global_position.terrain_alt",
        )
    setup_axis(axes[0], "Published HAGL and local terrain vertical position", "m")

    axes[1].plot(
        t,
        dist_bottom,
        lw=1.0,
        color="tab:blue",
        label="actual HAGL (dist_bottom)",
    )
    if local_terrain_z is not None and lpos_sp is not None and "z" in lpos_sp.data:
        t_sp = t_rel(lpos_sp, base_timestamp)
        sp_z = interp_step_previous(t_sp, arr(lpos_sp, "z"), t)
        sp_hagl = local_terrain_z - sp_z
        finite = np.isfinite(sp_hagl)
        if finite.any():
            axes[1].plot(
                t[finite],
                sp_hagl[finite],
                lw=1.0,
                color="tab:orange",
                label="HAGL implied by vehicle_local_position_setpoint.z",
            )
    if (
        local_terrain_z is not None
        and trajectory_sp is not None
        and "position[2]" in trajectory_sp.data
    ):
        t_sp = t_rel(trajectory_sp, base_timestamp)
        sp_z = interp_step_previous(t_sp, arr(trajectory_sp, "position[2]"), t)
        sp_hagl = local_terrain_z - sp_z
        finite = np.isfinite(sp_hagl)
        if finite.any():
            axes[1].plot(
                t[finite],
                sp_hagl[finite],
                lw=0.9,
                color="tab:green",
                ls="--",
                label="HAGL implied by trajectory_setpoint.position[2]",
            )
    if not axes[1].has_data():
        plot_unavailable(axes[1], "z setpoint or local terrain estimate not logged")
    setup_axis(axes[1], "Actual HAGL versus z-setpoint implied HAGL", "m")

    if valid is not None:
        axes[2].step(
            t, valid.astype(int), where="post", lw=1.0, color="tab:green", label="dist_bottom_valid"
        )
    else:
        axes[2].text(0.5, 0.5, "not logged", ha="center", va="center", transform=axes[2].transAxes)
    setup_boolean_axis(axes[2], "HAGL estimate valid")

    if "dist_bottom_sensor_bitfield" in lpos.data:
        source_active = arr(lpos, "dist_bottom_sensor_bitfield", int) != 0
        axes[3].step(
            t,
            source_active.astype(int),
            where="post",
            lw=1.0,
            color="tab:purple",
            label="dist_bottom_sensor_bitfield != 0",
        )
    else:
        axes[3].text(0.5, 0.5, "not logged", ha="center", va="center", transform=axes[3].transAxes)
    setup_boolean_axis(axes[3], "HAGL source bitfield active")
    axes[3].set_xlabel("flight-log relative time [s]")
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
    """Five-panel barometer / propwash analysis using distance_sensor as reference.

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

    fig, axes = plt.subplots(5, 1, figsize=(10.5, 11.0), sharex=True)

    # Propwash detection
    pw_result = detect_propwash_events(
        ulog, base_timestamp, divergence_threshold_m, throttle_threshold
    )
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
        axes[0].plot(
            tl, -arr(lpos, "z"), label="EKF local alt (−z)", lw=1.1, color="tab:green", zorder=3
        )
    if range_result is not None:
        t_r, r_m = range_result
        axes[0].plot(t_r, r_m, label="Rangefinder AGL", lw=1.3, color="tab:blue", zorder=4)
    if air:
        ta = t_rel(air, base_timestamp)
        if range_result is not None:
            baro_agl = arr(air, "baro_alt_meter") - offset
            baro_label = "Baro (offset-aligned to rangefinder)"
        else:
            baro_agl = arr(air, "baro_alt_meter")
            baro_label = "Baro altitude"
        axes[0].plot(ta, baro_agl, label=baro_label, lw=1.0, color="tab:orange", alpha=0.85)
    altitude_title = (
        "Altitude comparison — rangefinder reference; GPS excluded"
        if range_result is not None
        else "Altitude comparison — no distance_sensor topic; no range reference"
    )
    setup_axis(axes[0], altitude_title, "m AGL / MSL")

    # --- panel 2: baro-rangefinder divergence ---
    if t_div is not None and divergence_arr is not None:
        axes[1].plot(
            t_div, divergence_arr, lw=1.0, color="tab:purple", label="baro − rangefinder (aligned)"
        )
        axes[1].axhline(
            divergence_threshold_m,
            color="tab:red",
            lw=0.9,
            ls="--",
            label=f"+{divergence_threshold_m:.2f} m threshold",
        )
        axes[1].axhline(
            -divergence_threshold_m,
            color="tab:red",
            lw=0.9,
            ls="--",
            label=f"−{divergence_threshold_m:.2f} m threshold",
        )
        axes[1].axhline(0.0, color="gray", lw=0.6, ls=":")
        setup_axis(axes[1], "Baro − rangefinder divergence (prop wash signal)", "m")
    else:
        axes[1].text(
            0.5,
            0.5,
            "No distance_sensor topic; rangefinder baro divergence not computed",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
        setup_axis(axes[1], "Rangefinder divergence unavailable", "m")

    # --- panel 3: thrust ---
    if thrust_result is not None:
        t_thr, thrust = thrust_result
        axes[2].plot(t_thr, thrust, lw=0.9, color="tab:brown", label="|thrust| (normalized)")
        axes[2].axhline(
            throttle_threshold,
            color="gray",
            lw=0.8,
            ls="--",
            label=f"threshold {throttle_threshold:.2f}",
        )
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
        axes[0].axvspan(
            propwash_spans[0][0],
            propwash_spans[0][0],
            color="red",
            alpha=0.4,
            label="propwash event",
        )

    for ax in axes:
        handles, _labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best", fontsize=7, ncols=2)

    return save_fig(fig, fig_dir, "baro_propwash")


def plot_gps_quality(
    ulog: ULog, base_timestamp: int, fig_dir: Path, shade_modes: bool = True
) -> Path:
    gps = get_ds(ulog, "vehicle_gps_position")
    disabled_label = gps_disabled_label(ulog)
    fields = [
        ("fix_type", "Fix type", "enum", "tab:blue"),
        ("satellites_used", "Satellites used", "count", "tab:orange"),
        ("hdop", "HDOP", "DOP", "tab:blue"),
        ("vdop", "VDOP", "DOP", "tab:orange"),
        ("eph", "EPH reported horizontal accuracy", "m", "tab:blue"),
        ("epv", "EPV reported vertical accuracy", "m", "tab:orange"),
        ("s_variance_m_s", "Speed accuracy", "m/s", "tab:green"),
        ("vel_d_m_s", "GPS vertical velocity D", "m/s", "tab:purple"),
        ("jamming_state", "Jamming state", "enum", "tab:red"),
        ("jamming_indicator", "Jamming indicator", "indicator", "tab:red"),
        ("spoofing_state", "Spoofing state", "enum", "tab:brown"),
    ]
    plotted_fields = [field for field in fields if gps is not None and field[0] in gps.data]
    nrows = max(1, len(plotted_fields))
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(10.5, max(4.0, 0.7 + 0.9 * nrows)),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.ravel(np.atleast_1d(axes))

    if not plotted_fields or gps is None:
        plot_unavailable(axes[0], "vehicle_gps_position not logged")
        setup_axis(axes[0], "GPS quality unavailable", "value")
    else:
        t = t_rel(gps, base_timestamp)
        for ax, (field, title, ylabel, color) in zip(axes, plotted_fields):
            plot_scalar_trace(ax, t, arr(gps, field), title, ylabel, color=color)
            if field in {"fix_type", "satellites_used", "jamming_state", "spoofing_state"}:
                setup_integer_value_axis(ax, arr(gps, field, int))

    axes[-1].set_xlabel("flight-log relative time [s]")
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

    fig.text(
        0.5,
        0.01,
        "Note: GPS altitude is NOT used for baro divergence scoring. "
        "Rangefinder-based divergence requires a logged distance_sensor topic.",
        ha="center",
        fontsize=8,
        style="italic",
        color="gray",
    )

    return save_fig(fig, fig_dir, "gps_quality")


def plot_gps_checks(
    ulog: ULog, base_timestamp: int, fig_dir: Path, shade_modes: bool = True
) -> Path:
    gps_status = get_ds(ulog, "estimator_gps_status")
    checks = [
        ("check_fail_gps_fix", "GPS fix check fail"),
        ("check_fail_min_sat_count", "Minimum satellite count check fail"),
        ("check_fail_max_pdop", "Maximum PDOP check fail"),
        ("check_fail_max_horz_err", "Maximum horizontal error check fail"),
        ("check_fail_max_vert_err", "Maximum vertical error check fail"),
        ("check_fail_max_spd_err", "Maximum speed error check fail"),
        ("check_fail_max_horz_drift", "Maximum horizontal drift check fail"),
        ("check_fail_max_vert_drift", "Maximum vertical drift check fail"),
        ("check_fail_max_horz_spd_err", "Maximum horizontal speed check fail"),
        ("check_fail_max_vert_spd_err", "Maximum vertical speed check fail"),
    ]
    plotted = [item for item in checks if gps_status is not None and item[0] in gps_status.data]
    nrows = max(1, len(plotted))
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(10.5, max(4.0, 0.7 + 0.95 * nrows)),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.ravel(np.atleast_1d(axes))

    if not plotted or gps_status is None:
        plot_unavailable(axes[0], "estimator_gps_status check fields not logged")
        setup_boolean_axis(axes[0], "GPS checks unavailable")
    else:
        t = t_rel(gps_status, base_timestamp)
        for ax, (field, title) in zip(axes, plotted):
            plot_boolean_trace(ax, t, arr(gps_status, field, int), title, color="tab:red")

    axes[-1].set_xlabel("flight-log relative time [s]")
    shade_mode_background(axes, ulog, base_timestamp, enabled=shade_modes)
    add_system_time_axis(axes[-1], ulog, base_timestamp)
    return save_fig(fig, fig_dir, "gps_checks")


_SENSOR_TIMING_TOPIC_LABELS = [
    ("vehicle_imu", "IMU"),
    ("vehicle_optical_flow", "Optical flow"),
    ("vehicle_air_data", "Barometer"),
    ("vehicle_magnetometer", "Magnetometer"),
    ("sensor_mag", "Raw magnetometer"),
    ("vehicle_gps_position", "GPS / GNSS"),
    ("distance_sensor", "Range finder"),
    ("vehicle_visual_odometry", "External vision"),
]


def _timing_series_label(label: str, ds, include_instance: bool) -> str:
    multi_id = getattr(ds, "multi_id", 0)
    return f"{label}[{multi_id}]" if include_instance or multi_id else label


def sensor_latency_series(ulog: ULog, base_timestamp: int) -> list[SignalSeries]:
    series: list[SignalSeries] = []
    for topic, label in _SENSOR_TIMING_TOPIC_LABELS:
        datasets = get_all_ds(ulog, topic)
        include_instance = len(datasets) > 1
        for ds in datasets:
            latency_result = measurement_latency_s(ds, base_timestamp)
            if latency_result is None or not np.any(np.isfinite(latency_result[1])):
                continue
            t, latency_s, source_field = latency_result
            series.append(
                SignalSeries(
                    label=(
                        f"{_timing_series_label(label, ds, include_instance)} "
                        f"({topic}.timestamp - {source_field})"
                    ),
                    t=t,
                    values=latency_s * 1e3,
                )
            )
    return series


def plot_sensor_latency(
    ulog: ULog, base_timestamp: int, fig_dir: Path, shade_modes: bool = True
) -> Path:
    latency_series = sensor_latency_series(ulog, base_timestamp)
    nrows = max(1, len(latency_series))

    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(10.5, max(5.5, 0.9 + 1.2 * nrows)),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.ravel(np.atleast_1d(axes))

    if latency_series:
        for ax, sensor_series in zip(axes, latency_series):
            ax.plot(sensor_series.t, sensor_series.values, lw=0.9, color="tab:purple")
            setup_latency_axis(ax, sensor_series.values)
            setup_axis(ax, sensor_series.label, "ms")
    else:
        plot_unavailable(axes[0], "timestamp_sample latency fields not logged")
        setup_axis(axes[0], "Sensor sample-to-publication latency unavailable", "ms")
    axes[-1].set_xlabel("flight-log relative time [s]")

    shade_mode_background(axes, ulog, base_timestamp, enabled=shade_modes)
    add_system_time_axis(axes[-1], ulog, base_timestamp)
    return save_fig(fig, fig_dir, "sensor_latency")


def plot_imu_health(
    ulog: ULog, base_timestamp: int, fig_dir: Path, shade_modes: bool = True
) -> Path:
    imus = get_all_ds(ulog, "vehicle_imu_status")
    imu = imus[0] if imus else None
    sensor_combined = get_ds(ulog, "sensor_combined")
    status = get_ds(ulog, "estimator_status")
    flags = get_ds(ulog, "estimator_status_flags")
    bool_fields = []
    if flags is not None:
        t_flags = t_rel(flags, base_timestamp)
        for field in ["fs_bad_acc_vertical", "fs_bad_acc_bias", "fs_bad_acc_clipping"]:
            if field in flags.data:
                bool_fields.append((field, t_flags, arr(flags, field, int)))
    nrows = 7 + max(1, len(bool_fields))
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(10.5, max(9.8, 0.7 + 1.0 * nrows)),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.ravel(np.atleast_1d(axes))
    axis_idx = 0

    raw_accel_plotted = False
    if sensor_combined is not None:
        t_sc = t_rel(sensor_combined, base_timestamp)
        for index, color in enumerate(["tab:blue", "tab:orange", "tab:green"]):
            field = f"accelerometer_m_s2[{index}]"
            if field in sensor_combined.data:
                axes[axis_idx].plot(
                    t_sc,
                    arr(sensor_combined, field),
                    lw=0.7,
                    color=color,
                    label=f"sensor_combined.{field}",
                )
                raw_accel_plotted = True
    elif imu is not None and "delta_velocity_dt" in imu.data:
        t_imu = t_rel(imu, base_timestamp)
        dt = arr(imu, "delta_velocity_dt") * 1e-6
        dt = np.where(dt > 0.0, dt, np.nan)
        for index, color in enumerate(["tab:blue", "tab:orange", "tab:green"]):
            field = f"delta_velocity[{index}]"
            if field in imu.data:
                axes[axis_idx].plot(
                    t_imu,
                    arr(imu, field) / dt,
                    lw=0.7,
                    color=color,
                    label=f"vehicle_imu.{field} / delta_velocity_dt",
                )
                raw_accel_plotted = True
    if not raw_accel_plotted:
        plot_unavailable(axes[axis_idx], "raw accelerometer measurements not logged")
    setup_axis(axes[axis_idx], "Raw accelerometer measurements", "m/s^2")
    axis_idx += 1

    raw_gyro_plotted = False
    if sensor_combined is not None:
        t_sc = t_rel(sensor_combined, base_timestamp)
        for index, color in enumerate(["tab:blue", "tab:orange", "tab:green"]):
            field = f"gyro_rad[{index}]"
            if field in sensor_combined.data:
                axes[axis_idx].plot(
                    t_sc,
                    arr(sensor_combined, field),
                    lw=0.7,
                    color=color,
                    label=f"sensor_combined.{field}",
                )
                raw_gyro_plotted = True
    elif imu is not None and "delta_angle_dt" in imu.data:
        t_imu = t_rel(imu, base_timestamp)
        dt = arr(imu, "delta_angle_dt") * 1e-6
        dt = np.where(dt > 0.0, dt, np.nan)
        for index, color in enumerate(["tab:blue", "tab:orange", "tab:green"]):
            field = f"delta_angle[{index}]"
            if field in imu.data:
                axes[axis_idx].plot(
                    t_imu,
                    arr(imu, field) / dt,
                    lw=0.7,
                    color=color,
                    label=f"vehicle_imu.{field} / delta_angle_dt",
                )
                raw_gyro_plotted = True
    if not raw_gyro_plotted:
        plot_unavailable(axes[axis_idx], "raw gyroscope measurements not logged")
    setup_axis(axes[axis_idx], "Raw gyroscope measurements", "rad/s")
    axis_idx += 1

    accel_vibe_plotted = False
    for index, ds in enumerate(imus):
        if "accel_vibration_metric" not in ds.data:
            continue
        t_imu = t_rel(ds, base_timestamp)
        axes[axis_idx].plot(
            t_imu,
            arr(ds, "accel_vibration_metric"),
            lw=0.9,
            label=f"Accel {getattr(ds, 'multi_id', index)} vibration",
        )
        accel_vibe_plotted = True
    if status is not None:
        t_status = t_rel(status, base_timestamp)
        for index, field in enumerate(["vibe[0]", "vibe[1]", "vibe[2]"]):
            if field in status.data:
                axes[axis_idx].plot(
                    t_status,
                    arr(status, field),
                    lw=0.8,
                    ls=":",
                    label=f"estimator_status.{field}",
                )
                accel_vibe_plotted = True
    if accel_vibe_plotted:
        add_vibration_threshold_bands(axes[axis_idx])
    else:
        plot_unavailable(axes[axis_idx], "IMU accel vibration metrics not logged")
    setup_axis(axes[axis_idx], "IMU accel vibration metrics", "m/s^2")
    axis_idx += 1

    coning_plotted = False
    for index, ds in enumerate(imus):
        if "delta_angle_coning_metric" in ds.data:
            axes[axis_idx].plot(
                t_rel(ds, base_timestamp),
                arr(ds, "delta_angle_coning_metric"),
                lw=0.9,
                label=f"IMU {getattr(ds, 'multi_id', index)}",
            )
            coning_plotted = True
    if not coning_plotted:
        plot_unavailable(axes[axis_idx], "vehicle_imu_status.delta_angle_coning_metric not logged")
    setup_axis(axes[axis_idx], "IMU delta-angle coning metric", "metric")
    axis_idx += 1

    if imu is not None:
        t_imu = t_rel(imu, base_timestamp)
        for index, color in enumerate(["tab:blue", "tab:orange", "tab:green"]):
            field = f"accel_clipping[{index}]"
            if field in imu.data:
                axes[axis_idx].step(
                    t_imu, arr(imu, field, int), where="post", lw=0.9, color=color, label=field
                )
    if sensor_combined is not None and "accelerometer_clipping" in sensor_combined.data:
        t_sc = t_rel(sensor_combined, base_timestamp)
        axes[axis_idx].step(
            t_sc,
            arr(sensor_combined, "accelerometer_clipping", int),
            where="post",
            lw=0.8,
            color="tab:red",
            label="sensor_combined.accelerometer_clipping",
        )
    if not axes[axis_idx].has_data():
        plot_unavailable(axes[axis_idx], "accelerometer clipping counters not logged")
    setup_axis(axes[axis_idx], "Accelerometer clipping", "count / mask")
    setup_integer_value_axis(
        axes[axis_idx],
        np.asarray(axes[axis_idx].lines[0].get_ydata()) if axes[axis_idx].lines else np.array([]),
    )
    axis_idx += 1

    if imu is not None:
        t_imu = t_rel(imu, base_timestamp)
        for index, color in enumerate(["tab:blue", "tab:orange", "tab:green"]):
            field = f"gyro_clipping[{index}]"
            if field in imu.data:
                axes[axis_idx].step(
                    t_imu, arr(imu, field, int), where="post", lw=0.9, color=color, label=field
                )
    if not axes[axis_idx].has_data():
        plot_unavailable(axes[axis_idx], "gyroscope clipping counters not logged")
    setup_axis(axes[axis_idx], "Gyroscope clipping", "count")
    setup_integer_value_axis(
        axes[axis_idx],
        np.asarray(axes[axis_idx].lines[0].get_ydata()) if axes[axis_idx].lines else np.array([]),
    )
    axis_idx += 1

    if bool_fields:
        for field, t, values in bool_fields:
            plot_boolean_trace(axes[axis_idx], t, values, f"Estimator IMU quality flag: {field}")
            axis_idx += 1
    else:
        plot_unavailable(axes[axis_idx], "estimator_status_flags IMU quality flags not logged")
        setup_boolean_axis(axes[axis_idx], "Estimator IMU quality flags")
        axis_idx += 1

    if status is not None:
        t_status = t_rel(status, base_timestamp)
        plotted = False
        for field, color in [
            ("filter_fault_flags", "tab:red"),
            ("solution_status_flags", "tab:purple"),
            ("health_flags", "tab:green"),
        ]:
            if field in status.data:
                axes[axis_idx].step(
                    t_status,
                    arr(status, field, int),
                    where="post",
                    lw=0.9,
                    color=color,
                    label=field,
                )
                plotted = True
        if not plotted:
            plot_unavailable(axes[axis_idx], "estimator_status health masks not logged")
    else:
        plot_unavailable(axes[axis_idx], "estimator_status not logged")
    setup_axis(axes[axis_idx], "Estimator health / fault masks", "raw mask")
    axes[axis_idx].set_xlabel("flight-log relative time [s]")

    shade_mode_background(axes, ulog, base_timestamp, enabled=shade_modes)
    for ax in axes:
        handles, _labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best", fontsize=7, ncols=2)
    add_system_time_axis(axes[-1], ulog, base_timestamp)
    return save_fig(fig, fig_dir, "imu_health")


def _plot_imu_bias_adjustment(
    ulog: ULog,
    base_timestamp: int,
    fig_dir: Path,
    *,
    raw_topic: str,
    raw_fields: list[str],
    combined_fields: list[str],
    adjusted_topic: str,
    adjusted_fields: list[str],
    bias_fields: list[str],
    variance_fields: list[str],
    bias_limit_field: str,
    bias_valid_field: str,
    bias_stable_field: str,
    quantity: str,
    units: str,
    filename: str,
    shade_modes: bool,
) -> Path:
    raw = get_ds(ulog, raw_topic)
    combined = get_ds(ulog, "sensor_combined")
    adjusted = get_ds(ulog, adjusted_topic)
    bias = get_ds(ulog, "estimator_sensor_bias")

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(10.5, 12.5),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.5, 1.5, 1.5, 1.4, 1.0]},
    )
    axes = np.ravel(np.atleast_1d(axes))
    axis_names = ["X", "Y", "Z"]

    for index, axis_name in enumerate(axis_names):
        ax = axes[index]

        if raw is not None and raw_fields[index] in raw.data:
            ax.plot(
                t_rel(raw, base_timestamp),
                arr(raw, raw_fields[index]),
                lw=0.0,
                ls="none",
                marker="o",
                ms=3.2,
                color="0.25",
                alpha=0.8,
                label=f"{raw_topic}.{raw_fields[index]} (sensor/board FRD)",
            )

        if combined is not None and combined_fields[index] in combined.data:
            ax.plot(
                t_rel(combined, base_timestamp),
                arr(combined, combined_fields[index]),
                lw=0.65,
                color="tab:blue",
                alpha=0.75,
                label=f"sensor_combined.{combined_fields[index]} (calibrated body FRD)",
            )

        if adjusted is not None and adjusted_fields[index] in adjusted.data:
            ax.plot(
                t_rel(adjusted, base_timestamp),
                arr(adjusted, adjusted_fields[index]),
                lw=1.0,
                color="tab:green",
                label=f"{adjusted_topic}.{adjusted_fields[index]} (bias-adjusted body FRD)",
            )
        elif (
            combined is not None
            and bias is not None
            and combined_fields[index] in combined.data
            and bias_fields[index] in bias.data
        ):
            t_combined = t_rel(combined, base_timestamp)
            estimated_bias = interp_step_previous(
                t_rel(bias, base_timestamp), arr(bias, bias_fields[index]), t_combined
            )
            ax.plot(
                t_combined,
                arr(combined, combined_fields[index]) - estimated_bias,
                lw=0.9,
                ls="--",
                color="tab:green",
                label=f"sensor_combined - {bias_fields[index]} (derived fallback)",
            )

        if not ax.has_data():
            plot_unavailable(ax, f"{quantity.lower()} axis {axis_name} measurements not logged")
        ax.axhline(0.0, color="gray", lw=0.7, ls=":")
        setup_axis(ax, f"{quantity} {axis_name}: sensor through bias adjustment", units)
        if ax.has_data():
            ax.legend(loc="best", fontsize=6.5, ncols=1)

    bias_plotted = False
    if bias is not None:
        t_bias = t_rel(bias, base_timestamp)
        for index, (bias_field, variance_field, color) in enumerate(
            zip(bias_fields, variance_fields, ["tab:blue", "tab:orange", "tab:green"])
        ):
            if bias_field not in bias.data:
                continue
            values = arr(bias, bias_field)
            axes[3].plot(
                t_bias,
                values,
                lw=1.0,
                color=color,
                label=f"{axis_names[index]} bias",
            )
            if variance_field in bias.data:
                sigma = np.sqrt(np.maximum(arr(bias, variance_field), 0.0))
                axes[3].fill_between(
                    t_bias,
                    values - sigma,
                    values + sigma,
                    color=color,
                    alpha=0.12,
                    linewidth=0,
                    label=f"{axis_names[index]} +/-1 sigma",
                )
            bias_plotted = True

        if bias_limit_field in bias.data:
            limit = arr(bias, bias_limit_field)
            axes[3].plot(
                t_bias,
                limit,
                lw=0.8,
                ls="--",
                color="tab:red",
                label="bias limit",
            )
            axes[3].plot(t_bias, -limit, lw=0.8, ls="--", color="tab:red")
    if not bias_plotted:
        plot_unavailable(axes[3], f"estimator_sensor_bias {quantity.lower()} bias not logged")
    axes[3].axhline(0.0, color="gray", lw=0.7, ls=":")
    setup_axis(axes[3], f"EKF in-run {quantity.lower()} bias estimate", units)
    if axes[3].has_data():
        axes[3].legend(loc="best", fontsize=6.5, ncols=4)

    if bias is not None:
        t_bias = t_rel(bias, base_timestamp)
        status_series = [
            (field, arr(bias, field, int), color)
            for field, color in [
                (bias_valid_field, "tab:blue"),
                (bias_stable_field, "tab:green"),
            ]
            if field in bias.data
        ]
        plot_boolean_group(
            axes[4],
            t_bias,
            status_series,
            f"EKF {quantity.lower()} bias status",
        )
    else:
        plot_unavailable(axes[4], "estimator_sensor_bias not logged")
        setup_boolean_axis(axes[4], f"EKF {quantity.lower()} bias status")

    axes[-1].set_xlabel("flight-log relative time [s]")
    shade_mode_background(axes, ulog, base_timestamp, enabled=shade_modes)
    add_system_time_axis(axes[-1], ulog, base_timestamp)
    return save_fig(fig, fig_dir, filename)


def plot_accel_bias_adjustment(
    ulog: ULog, base_timestamp: int, fig_dir: Path, shade_modes: bool = True
) -> Path:
    return _plot_imu_bias_adjustment(
        ulog,
        base_timestamp,
        fig_dir,
        raw_topic="sensor_accel",
        raw_fields=["x", "y", "z"],
        combined_fields=[f"accelerometer_m_s2[{index}]" for index in range(3)],
        adjusted_topic="vehicle_acceleration",
        adjusted_fields=[f"xyz[{index}]" for index in range(3)],
        bias_fields=[f"accel_bias[{index}]" for index in range(3)],
        variance_fields=[f"accel_bias_variance[{index}]" for index in range(3)],
        bias_limit_field="accel_bias_limit",
        bias_valid_field="accel_bias_valid",
        bias_stable_field="accel_bias_stable",
        quantity="Acceleration",
        units="m/s$^2$",
        filename="accel_bias_adjustment",
        shade_modes=shade_modes,
    )


def plot_gyro_bias_adjustment(
    ulog: ULog, base_timestamp: int, fig_dir: Path, shade_modes: bool = True
) -> Path:
    return _plot_imu_bias_adjustment(
        ulog,
        base_timestamp,
        fig_dir,
        raw_topic="sensor_gyro",
        raw_fields=["x", "y", "z"],
        combined_fields=[f"gyro_rad[{index}]" for index in range(3)],
        adjusted_topic="vehicle_angular_velocity",
        adjusted_fields=[f"xyz[{index}]" for index in range(3)],
        bias_fields=[f"gyro_bias[{index}]" for index in range(3)],
        variance_fields=[f"gyro_bias_variance[{index}]" for index in range(3)],
        bias_limit_field="gyro_bias_limit",
        bias_valid_field="gyro_bias_valid",
        bias_stable_field="gyro_bias_stable",
        quantity="Angular velocity",
        units="rad/s",
        filename="gyro_bias_adjustment",
        shade_modes=shade_modes,
    )


def plot_barometer_health(
    ulog: ULog, base_timestamp: int, fig_dir: Path, shade_modes: bool = True
) -> Path:
    air = get_ds(ulog, "vehicle_air_data")
    events = get_ds(ulog, "estimator_event_flags")
    raw_fields = []
    if air is not None:
        t_air = t_rel(air, base_timestamp)
        for field, ylabel, color in [
            ("baro_pressure_pa", "Pa", "tab:blue"),
            ("baro_temp_celcius", "deg C", "tab:red"),
            ("rho", "kg/m^3", "tab:green"),
        ]:
            if field in air.data:
                raw_fields.append((field, t_air, arr(air, field), ylabel, color))
    bool_fields = []
    if events is not None:
        t_events = t_rel(events, base_timestamp)
        for field, color in [
            ("reset_hgt_to_baro", "tab:green"),
            ("height_sensor_timeout", "tab:red"),
        ]:
            if field in events.data:
                bool_fields.append((field, t_events, arr(events, field, int), color))
    nrows = 2 + max(1, len(raw_fields)) + max(1, len(bool_fields))
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(10.5, max(9.2, 0.7 + 1.0 * nrows)),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.ravel(np.atleast_1d(axes))
    axis_idx = 0

    if air is not None:
        t_air = t_rel(air, base_timestamp)
        if "baro_alt_meter" in air.data:
            axes[axis_idx].plot(
                t_air,
                arr(air, "baro_alt_meter"),
                lw=0.9,
                color="tab:orange",
                label="vehicle_air_data.baro_alt_meter",
            )
    if not axes[axis_idx].has_data():
        plot_unavailable(axes[axis_idx], "vehicle_air_data.baro_alt_meter not logged")
    setup_axis(axes[axis_idx], "Barometer altitude", "m")
    axis_idx += 1

    if raw_fields:
        for field, t, values, ylabel, color in raw_fields:
            plot_scalar_trace(
                axes[axis_idx],
                t,
                values,
                f"Barometer raw health: {field}",
                ylabel,
                color=color,
            )
            axis_idx += 1
    else:
        plot_unavailable(axes[axis_idx], "barometer pressure/temperature fields not logged")
        setup_axis(axes[axis_idx], "Barometer raw health fields", "logged units")
        axis_idx += 1

    baro_ratio = get_metric_series(ulog, base_timestamp, "baro_vpos", "test_ratio")
    axes[axis_idx].axhline(1.0, color="black", lw=0.9, ls="--", label="gate (1.0)")
    if baro_ratio is not None:
        axes[axis_idx].plot(
            baro_ratio.t,
            np.where(np.isfinite(baro_ratio.values), baro_ratio.values, np.nan),
            lw=0.9,
            color="tab:red",
            label=baro_ratio.source,
        )
        axes[axis_idx].set_yscale("symlog", linthresh=0.1)
    else:
        plot_unavailable(axes[axis_idx], "barometer height test ratio not logged")
    setup_axis(axes[axis_idx], "Barometer height innovation test ratio", "test ratio")
    axis_idx += 1

    if bool_fields:
        for field, t, values, color in bool_fields:
            plot_boolean_trace(axes[axis_idx], t, values, f"Barometer status: {field}", color=color)
            axis_idx += 1
    else:
        plot_unavailable(axes[axis_idx], "barometer fusion/reset/timeout flags not logged")
        setup_boolean_axis(axes[axis_idx], "Barometer fusion and timeout/reset flags")
        axis_idx += 1
    axes[axis_idx - 1].set_xlabel("flight-log relative time [s]")

    shade_mode_background(axes, ulog, base_timestamp, enabled=shade_modes)
    for ax in axes:
        handles, _labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best", fontsize=7, ncols=2)
    add_system_time_axis(axes[-1], ulog, base_timestamp)
    return save_fig(fig, fig_dir, "barometer_health")


def plot_magnetometer_health(
    ulog: ULog, base_timestamp: int, fig_dir: Path, shade_modes: bool = True
) -> Path:
    mag = get_ds(ulog, "vehicle_magnetometer") or get_ds(ulog, "sensor_mag")
    flags = get_ds(ulog, "estimator_status_flags")
    bool_fields = []
    if flags is not None:
        t_flags = t_rel(flags, base_timestamp)
        for field, color in [("reject_yaw", "tab:red")]:
            if field in flags.data:
                bool_fields.append((field, t_flags, arr(flags, field, int), color))
    nrows = 2 + max(1, len(bool_fields))
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(10.5, max(9.2, 0.7 + 0.95 * nrows)),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.ravel(np.atleast_1d(axes))
    axis_idx = 0

    if mag is not None:
        t_mag = t_rel(mag, base_timestamp)
        for field, color in [
            ("magnetometer_ga[0]", "tab:blue"),
            ("magnetometer_ga[1]", "tab:orange"),
            ("magnetometer_ga[2]", "tab:green"),
            ("x", "tab:blue"),
            ("y", "tab:orange"),
            ("z", "tab:green"),
        ]:
            if field in mag.data:
                axes[axis_idx].plot(
                    t_mag, arr(mag, field), lw=0.8, color=color, label=f"{mag.name}.{field}"
                )
    if not axes[axis_idx].has_data():
        plot_unavailable(axes[axis_idx], "vehicle_magnetometer / sensor_mag components not logged")
    setup_axis(axes[axis_idx], "Magnetometer components", "gauss / raw")
    axis_idx += 1

    heading_ratio = get_metric_series(ulog, base_timestamp, "heading", "test_ratio")
    axes[axis_idx].axhline(1.0, color="black", lw=0.9, ls="--", label="gate (1.0)")
    if heading_ratio is not None:
        axes[axis_idx].plot(
            heading_ratio.t,
            np.where(np.isfinite(heading_ratio.values), heading_ratio.values, np.nan),
            lw=0.9,
            color="tab:red",
            label=heading_ratio.source,
        )
        axes[axis_idx].set_yscale("symlog", linthresh=0.1)
    else:
        plot_unavailable(axes[axis_idx], "heading innovation test ratio not logged")
    setup_axis(axes[axis_idx], "Heading innovation test ratio", "test ratio")
    axis_idx += 1

    if bool_fields:
        for field, t, values, color in bool_fields:
            plot_boolean_trace(
                axes[axis_idx], t, values, f"Magnetometer status: {field}", color=color
            )
            axis_idx += 1
    else:
        plot_unavailable(axes[axis_idx], "magnetometer rejection flags not logged")
        setup_boolean_axis(axes[axis_idx], "Magnetometer rejection flags")
        axis_idx += 1
    axes[axis_idx - 1].set_xlabel("flight-log relative time [s]")

    shade_mode_background(axes, ulog, base_timestamp, enabled=shade_modes)
    for ax in axes:
        handles, _labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best", fontsize=7, ncols=2)
    add_system_time_axis(axes[-1], ulog, base_timestamp)
    return save_fig(fig, fig_dir, "magnetometer_health")


def plot_optical_flow_sensor_health(
    ulog: ULog, base_timestamp: int, fig_dir: Path, shade_modes: bool = True
) -> Path:
    aid = get_ds(ulog, "estimator_aid_src_optical_flow")
    vof = get_ds(ulog, "vehicle_optical_flow")
    of_vel = get_ds(ulog, "estimator_optical_flow_vel")
    flags = get_ds(ulog, "estimator_status_flags")
    bool_fields = []
    if aid is not None:
        t_aid = t_rel(aid, base_timestamp)
        for field, color in [("fusion_enabled", "tab:green"), ("innovation_rejected", "tab:red")]:
            if field in aid.data:
                values = arr(aid, field, int)
                label = f"aid {field}"
                if field == "fusion_enabled":
                    percentage = boolean_time_percentage_label(t_aid, values)
                    if percentage is not None:
                        label = f"{label} ({percentage})"
                bool_fields.append((label, t_aid, values, color))
    if flags is not None:
        t_flags = t_rel(flags, base_timestamp)
        for field, color in [
            ("reject_optflow_x", "tab:orange"),
            ("reject_optflow_y", "tab:red"),
            ("fs_bad_optflow_x", "tab:green"),
            ("fs_bad_optflow_y", "tab:purple"),
        ]:
            if field in flags.data:
                bool_fields.append((field, t_flags, arr(flags, field, int), color))
    nrows = 5 + max(1, len(bool_fields))
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(10.5, max(9.8, 0.7 + 0.95 * nrows)),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.ravel(np.atleast_1d(axes))
    axis_idx = 0

    if vof is not None:
        t_vof = t_rel(vof, base_timestamp)
        if "quality" in vof.data:
            axes[axis_idx].plot(
                t_vof,
                arr(vof, "quality"),
                lw=0.9,
                color="tab:blue",
                label="vehicle_optical_flow.quality",
            )
            axes[axis_idx].axhline(50, color="tab:red", lw=0.8, ls="--", label="quality floor 50")
            axes[axis_idx].set_ylim(0, 270)
    if not axes[axis_idx].has_data():
        plot_unavailable(axes[axis_idx], "vehicle_optical_flow.quality not logged")
    setup_axis(axes[axis_idx], "Optical-flow sensor quality", "0-255")
    axis_idx += 1

    plot_optical_flow_measurement_counts(axes[axis_idx], ulog, base_timestamp)
    axis_idx += 1

    if vof is not None:
        t_vof = t_rel(vof, base_timestamp)
        for field, color in [("pixel_flow[0]", "tab:blue"), ("pixel_flow[1]", "tab:orange")]:
            if field in vof.data:
                axes[axis_idx].plot(
                    t_vof,
                    arr(vof, field),
                    lw=0.9,
                    color=color,
                    label=f"vehicle_optical_flow.{field}",
                )
        axes[axis_idx].axhline(0, color="gray", lw=0.6, ls=":")
    if not axes[axis_idx].has_data():
        plot_unavailable(axes[axis_idx], "vehicle_optical_flow pixel flow not logged")
    setup_axis(axes[axis_idx], "Optical-flow pixel flow", "rad")
    axis_idx += 1

    plot_direct_test_ratios(
        axes[axis_idx],
        aid,
        base_timestamp,
        "Optical-flow innovation test ratios",
        "estimator_aid_src_optical_flow",
    )
    axis_idx += 1

    if bool_fields:
        for field, t, values, color in bool_fields:
            plot_boolean_trace(
                axes[axis_idx], t, values, f"Optical-flow status: {field}", color=color
            )
            axis_idx += 1
    else:
        plot_unavailable(axes[axis_idx], "optical-flow fusion/rejection flags not logged")
        setup_boolean_axis(axes[axis_idx], "Optical-flow status flags")
        axis_idx += 1

    if of_vel is not None:
        t_vel = t_rel(of_vel, base_timestamp)
        for field, color in [("vel_body[0]", "tab:blue"), ("vel_body[1]", "tab:orange")]:
            if field in of_vel.data:
                axes[axis_idx].plot(
                    t_vel,
                    arr(of_vel, field),
                    lw=0.9,
                    color=color,
                    label=f"estimator_optical_flow_vel.{field}",
                )
        axes[axis_idx].axhline(0, color="gray", lw=0.6, ls=":")
    if not axes[axis_idx].has_data():
        plot_unavailable(axes[axis_idx], "estimator_optical_flow_vel body velocity not logged")
    setup_axis(axes[axis_idx], "EKF-fused optical-flow body velocity", "m/s")
    axes[axis_idx].set_xlabel("flight-log relative time [s]")

    shade_mode_background(axes, ulog, base_timestamp, enabled=shade_modes)
    for ax in axes:
        handles, _labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best", fontsize=7, ncols=2)
    add_system_time_axis(axes[-1], ulog, base_timestamp)
    return save_fig(fig, fig_dir, "optical_flow_health")


def plot_optical_flow_fusion_gating(
    ulog: ULog, base_timestamp: int, fig_dir: Path, shade_modes: bool = True
) -> Path:
    aid = get_ds(ulog, "estimator_aid_src_optical_flow")
    flags = get_ds(ulog, "estimator_status_flags")
    bool_fields = []
    if aid is not None:
        t_aid = t_rel(aid, base_timestamp)
        for field, color in [
            ("fusion_enabled", "tab:green"),
            ("innovation_rejected", "tab:red"),
        ]:
            if field in aid.data:
                values = arr(aid, field, int)
                label = f"aid {field}"
                if field == "fusion_enabled":
                    percentage = boolean_time_percentage_label(t_aid, values)
                    if percentage is not None:
                        label = f"{label} ({percentage})"
                bool_fields.append((label, t_aid, values, color))
    if flags is not None:
        t_flags = t_rel(flags, base_timestamp)
        for field, color in [
            ("reject_optflow_x", "tab:orange"),
            ("reject_optflow_y", "tab:red"),
            ("fs_bad_optflow_x", "tab:green"),
            ("fs_bad_optflow_y", "tab:purple"),
        ]:
            if field in flags.data:
                bool_fields.append((field, t_flags, arr(flags, field, int), color))

    nrows = 7 + max(1, len(bool_fields))
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(10.5, max(9.8, 0.75 + 1.0 * nrows)),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.ravel(np.atleast_1d(axes))
    axis_idx = 0

    for key, title in [
        ("flow[0]", "Optical-flow X innovation test ratio"),
        ("flow[1]", "Optical-flow Y innovation test ratio"),
    ]:
        series = get_metric_series(ulog, base_timestamp, key, "test_ratio")
        if series is not None:
            y = np.where(np.isfinite(series.values), series.values, np.nan)
            axes[axis_idx].plot(series.t, y, lw=0.9, color="tab:blue", label=series.source)
            axes[axis_idx].set_yscale("symlog", linthresh=0.1)
        else:
            plot_unavailable(axes[axis_idx], f"{key} optical-flow test ratio not logged")
        axes[axis_idx].axhline(1.0, color="black", lw=0.9, ls="--", label="gate (1.0)")
        axes[axis_idx].axhline(0.36, color="gray", lw=0.7, ls=":", label="3 sigma equiv")
        setup_axis(axes[axis_idx], title, "test ratio")
        axis_idx += 1

    for key, title in [
        ("flow[0]", "Optical-flow X innovation versus 5 sigma gate"),
        ("flow[1]", "Optical-flow Y innovation versus 5 sigma gate"),
    ]:
        innovation = get_metric_series(ulog, base_timestamp, key, "innovation")
        variance_series = get_metric_series(ulog, base_timestamp, key, "innovation_variance")
        if innovation is not None and variance_series is not None:
            variance = interp_at(variance_series.t, variance_series.values, innovation.t)
            threshold = 5.0 * np.sqrt(np.maximum(variance, 0.0))
            axes[axis_idx].plot(
                innovation.t,
                innovation.values,
                lw=0.9,
                color="tab:blue",
                label=innovation.source,
            )
            axes[axis_idx].plot(
                innovation.t,
                threshold,
                color="tab:red",
                lw=0.8,
                ls="--",
                label="+5 sigma gate",
            )
            axes[axis_idx].plot(
                innovation.t,
                -threshold,
                color="tab:red",
                lw=0.8,
                ls="--",
                label="-5 sigma gate",
            )
        else:
            plot_unavailable(axes[axis_idx], f"{key} innovation or variance not logged")
        setup_axis(axes[axis_idx], title, "rad")
        axis_idx += 1

    plot_optical_flow_measurement_counts(axes[axis_idx], ulog, base_timestamp)
    axis_idx += 1

    delta_result = time_since_last_fuse_s(aid, base_timestamp)
    if delta_result is not None and np.any(np.isfinite(delta_result[1])):
        delta_t, delta_s = delta_result
        axes[axis_idx].plot(delta_t, delta_s, lw=0.9, color="tab:purple")
    else:
        plot_unavailable(
            axes[axis_idx],
            "estimator_aid_src_optical_flow.time_last_fuse not logged",
        )
    setup_axis(
        axes[axis_idx],
        "Optical-flow time since last successful fusion",
        "s",
    )
    axis_idx += 1

    if aid is not None and "fused" in aid.data:
        plot_boolean_trace(
            axes[axis_idx],
            t_rel(aid, base_timestamp),
            arr(aid, "fused", int),
            "Optical-flow gating status: aid fused",
            color="tab:blue",
        )
    else:
        plot_unavailable(axes[axis_idx], "estimator_aid_src_optical_flow.fused not logged")
        setup_boolean_axis(axes[axis_idx], "Optical-flow gating status: aid fused")
    axis_idx += 1

    if bool_fields:
        for field, t, values, color in bool_fields:
            plot_boolean_trace(
                axes[axis_idx], t, values, f"Optical-flow gating status: {field}", color
            )
            axis_idx += 1
    else:
        plot_unavailable(axes[axis_idx], "optical-flow fusion/rejection flags not logged")
        setup_boolean_axis(axes[axis_idx], "Optical-flow fusion/rejection flags")
        axis_idx += 1

    axes[-1].set_xlabel("flight-log relative time [s]")
    shade_mode_background(axes, ulog, base_timestamp, enabled=shade_modes)
    for ax in axes:
        handles, _labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best", fontsize=7, ncols=2)
    add_system_time_axis(axes[-1], ulog, base_timestamp)
    return save_fig(fig, fig_dir, "optical_flow_fusion_gating")


def plot_rangefinder_health(
    ulog: ULog, base_timestamp: int, fig_dir: Path, shade_modes: bool = True
) -> Path:
    dist = get_ds(ulog, "distance_sensor")
    lpos = get_ds(ulog, "vehicle_local_position")
    aid = get_ds(ulog, "estimator_aid_src_rng_hgt")
    flags = get_ds(ulog, "estimator_status_flags")
    events = get_ds(ulog, "estimator_event_flags")
    bool_fields = []
    if aid is not None:
        t_aid = t_rel(aid, base_timestamp)
        for field, color in [("fusion_enabled", "tab:green"), ("innovation_rejected", "tab:red")]:
            if field in aid.data:
                bool_fields.append((f"aid {field}", t_aid, arr(aid, field, int), color))
    if flags is not None:
        t_flags = t_rel(flags, base_timestamp)
        for field, color in [("reject_hagl", "tab:red")]:
            if field in flags.data:
                bool_fields.append((field, t_flags, arr(flags, field, int), color))
    if events is not None:
        t_events = t_rel(events, base_timestamp)
        for field, color in [
            ("height_sensor_timeout", "tab:red"),
            ("reset_hgt_to_rng", "tab:green"),
        ]:
            if field in events.data:
                bool_fields.append((field, t_events, arr(events, field, int), color))
    range_ratio, hagl_rate_ratio = rangefinder_test_ratio_series(ulog, base_timestamp)
    nrows = 3 + int(hagl_rate_ratio is not None) + max(1, len(bool_fields))
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(10.5, max(9.8, 0.7 + 0.95 * nrows)),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.ravel(np.atleast_1d(axes))
    axis_idx = 0

    range_result = get_range(ulog, base_timestamp)
    if range_result is not None:
        t_range, range_m = range_result
        axes[axis_idx].plot(
            t_range, range_m, lw=0.9, color="tab:blue", label="distance_sensor.current_distance"
        )
    if lpos is not None and "dist_bottom" in lpos.data:
        t_lpos = t_rel(lpos, base_timestamp)
        axes[axis_idx].plot(
            t_lpos,
            arr(lpos, "dist_bottom"),
            lw=0.8,
            color="tab:green",
            label="vehicle_local_position.dist_bottom",
        )
    if not axes[axis_idx].has_data():
        plot_unavailable(
            axes[axis_idx], "distance_sensor.current_distance / dist_bottom not logged"
        )
    setup_axis(axes[axis_idx], "Range finder and EKF HAGL estimate", "m")
    axis_idx += 1

    if dist is not None:
        t_dist = t_rel(dist, base_timestamp)
        for field, color in [("signal_quality", "tab:blue"), ("variance", "tab:orange")]:
            if field in dist.data:
                axes[axis_idx].plot(
                    t_dist, arr(dist, field), lw=0.9, color=color, label=f"distance_sensor.{field}"
                )
    if not axes[axis_idx].has_data():
        plot_unavailable(axes[axis_idx], "distance_sensor quality fields not logged")
    setup_axis(axes[axis_idx], "Range finder signal quality / variance", "logged units")
    axis_idx += 1

    axes[axis_idx].axhline(1.0, color="black", lw=0.9, ls="--", label="gate (1.0)")
    if range_ratio is not None:
        axes[axis_idx].plot(
            range_ratio.t,
            np.where(np.isfinite(range_ratio.values), range_ratio.values, np.nan),
            lw=0.9,
            color="tab:red",
            label=range_ratio.source,
        )
        axes[axis_idx].set_yscale("symlog", linthresh=0.1)
    else:
        plot_unavailable(axes[axis_idx], "range height innovation test ratio not logged")
    setup_axis(axes[axis_idx], "Range height innovation test ratio", "test ratio")
    axis_idx += 1

    if hagl_rate_ratio is not None:
        axes[axis_idx].axhline(1.0, color="black", lw=0.9, ls="--", label="gate (1.0)")
        axes[axis_idx].plot(
            hagl_rate_ratio.t,
            np.where(np.isfinite(hagl_rate_ratio.values), hagl_rate_ratio.values, np.nan),
            lw=0.9,
            color="tab:purple",
            label=hagl_rate_ratio.source,
        )
        axes[axis_idx].set_yscale("symlog", linthresh=0.1)
        setup_axis(axes[axis_idx], "HAGL rate innovation test ratio", "test ratio")
        axis_idx += 1

    if bool_fields:
        for field, t, values, color in bool_fields:
            plot_boolean_trace(
                axes[axis_idx], t, values, f"Range finder status: {field}", color=color
            )
            axis_idx += 1
    else:
        plot_unavailable(axes[axis_idx], "range finder aid/event/rejection flags not logged")
        setup_boolean_axis(axes[axis_idx], "Range finder aid/event/rejection flags")
        axis_idx += 1
    axes[axis_idx - 1].set_xlabel("flight-log relative time [s]")

    shade_mode_background(axes, ulog, base_timestamp, enabled=shade_modes)
    for ax in axes:
        handles, _labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best", fontsize=7, ncols=2)
    add_system_time_axis(axes[-1], ulog, base_timestamp)
    return save_fig(fig, fig_dir, "rangefinder_health")


def plot_external_vision_health(
    ulog: ULog, base_timestamp: int, fig_dir: Path, shade_modes: bool = True
) -> Path:
    odom = get_ds(ulog, "vehicle_visual_odometry")
    events = get_ds(ulog, "estimator_event_flags")
    ev_ratio_specs = ev_test_ratio_panel_specs(ulog)
    bool_fields = []
    if events is not None:
        t_events = t_rel(events, base_timestamp)
        for field, color in [
            ("vision_data_stopped", "tab:red"),
            ("starting_vision_pos_fusion", "tab:blue"),
            ("starting_vision_vel_fusion", "tab:orange"),
            ("starting_vision_yaw_fusion", "tab:green"),
        ]:
            if field in events.data:
                bool_fields.append((field, t_events, arr(events, field, int), color))
    nrows = 1 + max(1, len(ev_ratio_specs)) + max(1, len(bool_fields))
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(10.5, max(8.8, 0.7 + 0.9 * nrows)),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.ravel(np.atleast_1d(axes))
    axis_idx = 0

    if odom is not None:
        t_odom = t_rel(odom, base_timestamp)
        for field, color in [
            ("quality", "tab:blue"),
            ("pose_frame", "tab:orange"),
            ("velocity_frame", "tab:green"),
        ]:
            if field in odom.data:
                axes[axis_idx].plot(
                    t_odom,
                    arr(odom, field),
                    lw=0.8,
                    color=color,
                    label=f"vehicle_visual_odometry.{field}",
                )
    if not axes[axis_idx].has_data():
        plot_unavailable(axes[axis_idx], "vehicle_visual_odometry quality/frame fields not logged")
    setup_axis(axes[axis_idx], "External-vision odometry quality / frames", "logged units")
    axis_idx += 1

    if ev_ratio_specs:
        for topic, title, fields, label_prefix in ev_ratio_specs:
            plot_ev_test_ratio_panel(
                axes[axis_idx],
                get_ds(ulog, topic),
                base_timestamp,
                title,
                fields,
                label_prefix,
            )
            axis_idx += 1
    else:
        plot_unavailable(axes[axis_idx], "external-vision aid-source test ratios not logged")
        setup_axis(axes[axis_idx], "External-vision innovation test ratios", "test ratio")
        axis_idx += 1

    if bool_fields:
        for field, t, values, color in bool_fields:
            plot_boolean_trace(
                axes[axis_idx], t, values, f"External-vision status: {field}", color=color
            )
            axis_idx += 1
    else:
        plot_unavailable(axes[axis_idx], "external-vision event flags not logged")
        setup_boolean_axis(axes[axis_idx], "External-vision event flags")
        axis_idx += 1
    axes[axis_idx - 1].set_xlabel("flight-log relative time [s]")

    shade_mode_background(axes, ulog, base_timestamp, enabled=shade_modes)
    for ax in axes:
        handles, _labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best", fontsize=7, ncols=2)
    add_system_time_axis(axes[-1], ulog, base_timestamp)
    return save_fig(fig, fig_dir, "external_vision_health")


def plot_external_vision_position_comparison(
    ulog: ULog, base_timestamp: int, fig_dir: Path, shade_modes: bool = True
) -> Path:
    odom = get_ds(ulog, "vehicle_visual_odometry")
    local_position = get_ds(ulog, "vehicle_local_position")
    fig, axes = plt.subplots(6, 1, figsize=(10.5, 9.8), sharex=True, constrained_layout=True)
    axes = np.ravel(np.atleast_1d(axes))
    axis_specs = [
        ("X", "position[0]", "x", "tab:blue"),
        ("Y", "position[1]", "y", "tab:green"),
        ("Z", "position[2]", "z", "tab:purple"),
    ]
    t_odom = t_rel(odom, base_timestamp) if odom is not None else None
    t_local = t_rel(local_position, base_timestamp) if local_position is not None else None

    for idx, (axis_label, ev_field, local_field, color) in enumerate(axis_specs):
        ax_value = axes[idx]
        if (
            local_position is not None
            and local_field in local_position.data
            and t_local is not None
        ):
            ax_value.plot(
                t_local,
                arr(local_position, local_field),
                lw=1.0,
                color=color,
                label=f"vehicle_local_position.{local_field}",
            )
        else:
            ax_value.text(
                0.01,
                0.90,
                f"vehicle_local_position.{local_field} not logged",
                transform=ax_value.transAxes,
                fontsize=8,
                color="tab:red",
            )

        if odom is not None and ev_field in odom.data and t_odom is not None:
            ax_value.plot(
                t_odom,
                arr(odom, ev_field),
                lw=0.8,
                marker=".",
                markersize=2.0,
                color="tab:orange",
                label=f"vehicle_visual_odometry.{ev_field}",
            )
        else:
            ax_value.text(
                0.01,
                0.78,
                f"vehicle_visual_odometry.{ev_field} not logged",
                transform=ax_value.transAxes,
                fontsize=8,
                color="tab:red",
            )
        setup_axis(ax_value, f"Local position estimate vs EV {axis_label}", f"{axis_label} [m]")

        ax_delta = axes[idx + 3]
        delta = local_minus_ev_position_delta(
            odom, local_position, base_timestamp, ev_field, local_field
        )
        if delta is not None and np.any(np.isfinite(delta[1])):
            t_delta, values = delta
            ax_delta.plot(t_delta, values, lw=0.9, color=color)
            ax_delta.axhline(0.0, color="gray", lw=0.7, ls=":")
        else:
            ax_delta.text(
                0.5,
                0.5,
                f"{axis_label} local - EV delta unavailable",
                ha="center",
                va="center",
                transform=ax_delta.transAxes,
            )
        setup_axis(ax_delta, f"Local minus EV {axis_label}", "m")

    axes[-1].set_xlabel("flight-log relative time [s]")
    shade_mode_background(axes, ulog, base_timestamp, enabled=shade_modes)
    for ax in axes:
        handles, _labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best", fontsize=7, ncols=2)
    add_system_time_axis(axes[-1], ulog, base_timestamp)
    return save_fig(fig, fig_dir, "external_vision_position_comparison")


def plot_control_status_flags(
    ulog: ULog, base_timestamp: int, fig_dir: Path, shade_modes: bool = True
) -> Path:
    flags = get_ds(ulog, "estimator_status_flags")
    groups = control_status_flag_groups(flags)
    nrows = max(1, len(groups))
    height_ratios = [max(1.35, 0.34 * len(fields)) for _title, fields in groups] or [1.0]
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(10.5, max(4.0, 0.9 + sum(height_ratios))),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    axes = np.ravel(np.atleast_1d(axes))

    if not groups or flags is None:
        plot_unavailable(axes[0], "estimator_status_flags.cs_* fields not logged")
        setup_boolean_axis(axes[0], "Control statuses unavailable")
    else:
        t = t_rel(flags, base_timestamp)
        for ax, (title, fields) in zip(axes, groups):
            plot_boolean_group(ax, t, status_flag_series(flags, fields), title)

    axes[-1].set_xlabel("flight-log relative time [s]")
    shade_mode_background(axes, ulog, base_timestamp, enabled=shade_modes)
    add_system_time_axis(axes[-1], ulog, base_timestamp)
    return save_fig(fig, fig_dir, "control_status_flags")


def plot_inertial_dead_reckoning(
    ulog: ULog, base_timestamp: int, fig_dir: Path, shade_modes: bool = True
) -> Path:
    flags = get_ds(ulog, "estimator_status_flags")
    aid_flow = get_ds(ulog, "estimator_aid_src_optical_flow")
    local_position = get_ds(ulog, "vehicle_local_position")
    status = get_ds(ulog, "estimator_status")
    events = get_ds(ulog, "estimator_event_flags")

    fig, axes = plt.subplots(
        7,
        1,
        figsize=(10.5, 15.5),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.4, 1.4, 2.1, 2.8, 2.2, 2.1, 1.7]},
    )
    axes = np.ravel(np.atleast_1d(axes))
    colors = ["tab:blue", "tab:orange", "tab:green"]

    if local_position is not None:
        t_local = t_rel(local_position, base_timestamp)
        for field, color in zip(["vx", "vy", "vz"], colors):
            if field in local_position.data:
                axes[0].plot(
                    t_local,
                    arr(local_position, field),
                    lw=0.9,
                    color=color,
                    label=f"vehicle_local_position.{field}",
                )
        for field, color in zip(["ax", "ay", "az"], colors):
            if field in local_position.data:
                axes[1].plot(
                    t_local,
                    arr(local_position, field),
                    lw=0.9,
                    color=color,
                    label=f"vehicle_local_position.{field}",
                )
    if not axes[0].has_data():
        plot_unavailable(axes[0], "vehicle_local_position.vx/vy/vz not logged")
    if not axes[1].has_data():
        plot_unavailable(axes[1], "vehicle_local_position.ax/ay/az not logged")
    for ax, title, ylabel in [
        (axes[0], "EKF local velocity", "m/s"),
        (axes[1], "EKF local acceleration", "m/s$^2$"),
    ]:
        ax.axhline(0.0, color="gray", lw=0.7, ls=":")
        setup_axis(ax, title, ylabel)
        if ax.has_data():
            ax.legend(loc="best", fontsize=7, ncols=3)

    if flags is not None:
        t_flags = t_rel(flags, base_timestamp)
        plot_boolean_group(
            axes[2],
            t_flags,
            status_flag_series(
                flags,
                [
                    "cs_inertial_dead_reckoning",
                    "cs_wind_dead_reckoning",
                    "cs_in_air",
                    "cs_vehicle_at_rest",
                ],
            ),
            "Dead-reckoning and vehicle state",
        )
        plot_boolean_group(
            axes[3],
            t_flags,
            status_flag_series(
                flags,
                [
                    "cs_opt_flow",
                    "cs_ev_pos",
                    "cs_ev_vel",
                    "cs_gps",
                    "cs_gnss_pos",
                    "cs_gnss_vel",
                    "cs_fake_pos",
                    "cs_valid_fake_pos",
                    "cs_constant_pos",
                ],
            ),
            "Horizontal aiding control status",
        )
        plot_boolean_group(
            axes[4],
            t_flags,
            status_flag_series(
                flags,
                [
                    "cs_rng_hgt",
                    "cs_ev_hgt",
                    "cs_baro_hgt",
                    "cs_gps_hgt",
                    "cs_gnss_hgt",
                    "cs_fake_hgt",
                ],
            ),
            "Height aiding control status",
        )
    else:
        for ax, title in [
            (axes[2], "Dead-reckoning and vehicle state"),
            (axes[3], "Horizontal aiding control status"),
            (axes[4], "Height aiding control status"),
        ]:
            plot_unavailable(ax, "estimator_status_flags not logged")
            setup_axis(ax, title)

    if aid_flow is not None:
        t_flow = t_rel(aid_flow, base_timestamp)
        plot_boolean_group(
            axes[5],
            t_flow,
            [
                (field, arr(aid_flow, field, int), color)
                for field, color in [
                    ("fusion_enabled", "tab:green"),
                    ("fused", "tab:blue"),
                    ("innovation_rejected", "tab:red"),
                ]
                if field in aid_flow.data
            ],
            "Optical-flow aid-source sample status",
        )
    else:
        plot_unavailable(axes[5], "estimator_aid_src_optical_flow not logged")
        setup_axis(axes[5], "Optical-flow aid-source sample status")

    reset_series = []
    if local_position is not None:
        t_local = t_rel(local_position, base_timestamp)
        for field, color in [
            ("vxy_reset_counter", "tab:blue"),
            ("xy_reset_counter", "tab:orange"),
        ]:
            if field in local_position.data:
                values = arr(local_position, field, int)
                axes[6].step(
                    t_local,
                    values,
                    where="post",
                    lw=1.0,
                    color=color,
                    label=f"vehicle_local_position.{field}",
                )
                reset_series.append(values)
    if not reset_series and status is not None:
        t_status = t_rel(status, base_timestamp)
        for field, color in [
            ("reset_count_vel_ne", "tab:blue"),
            ("reset_count_pos_ne", "tab:orange"),
        ]:
            if field in status.data:
                values = arr(status, field, int)
                axes[6].step(
                    t_status,
                    values,
                    where="post",
                    lw=1.0,
                    color=color,
                    label=f"estimator_status.{field}",
                )
                reset_series.append(values)

    if events is not None:
        t_events = t_rel(events, base_timestamp)
        for field, color in [
            ("reset_vel_to_flow", "tab:blue"),
            ("reset_pos_to_last_known", "tab:red"),
        ]:
            if field not in events.data:
                continue
            active_times = t_events[arr(events, field, int).astype(bool)]
            for index, event_time in enumerate(active_times):
                axes[6].axvline(
                    event_time,
                    color=color,
                    lw=0.8,
                    ls=":",
                    alpha=0.8,
                    label=field if index == 0 else None,
                )

    if reset_series:
        setup_integer_value_axis(axes[6], np.concatenate(reset_series))
    elif not axes[6].has_data():
        plot_unavailable(axes[6], "horizontal reset counters/events not logged")
    setup_axis(axes[6], "Horizontal state resets", "count")
    if axes[6].has_data():
        axes[6].legend(loc="best", fontsize=7, ncols=2)

    axes[-1].set_xlabel("flight-log relative time [s]")
    shade_mode_background(axes, ulog, base_timestamp, enabled=shade_modes)
    add_system_time_axis(axes[-1], ulog, base_timestamp)
    return save_fig(fig, fig_dir, "inertial_dead_reckoning")


def plot_reset_counters(
    ulog: ULog, base_timestamp: int, fig_dir: Path, shade_modes: bool = True
) -> Path:
    status = get_ds(ulog, "estimator_status")
    local_position = get_ds(ulog, "vehicle_local_position")
    reset_rows = reset_event_rows(ulog, base_timestamp)
    series = [
        ("estimator_status", status, field)
        for field in available_fields(status, _RESET_COUNTER_FIELDS)
    ] + [
        ("vehicle_local_position", local_position, field)
        for field in available_fields(local_position, _LOCAL_POSITION_RESET_COUNTER_FIELDS)
    ]
    nrows = max(1, len(series))
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(10.5, max(2.0, 0.85 + 0.72 * nrows)),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.ravel(np.atleast_1d(axes))

    if not series:
        axes[0].text(
            0.5,
            0.5,
            "estimator_status/vehicle_local_position reset counters not logged",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
        )
        setup_axis(axes[0], "Reset counters unavailable", "count")
    else:
        for ax, (source, ds, key) in zip(axes, series):
            t = t_rel(ds, base_timestamp)
            values = arr(ds, key, int)
            label = f"{source}.{key}"
            ax.step(t, values, where="post", lw=1.0, color="tab:purple", label=label)
            for row_idx, row in enumerate(reset_rows):
                ax.axvline(
                    row.time_s,
                    color="tab:red",
                    lw=0.9,
                    ls=":",
                    alpha=0.75,
                    label="logged reset event" if row_idx == 0 else None,
                )
            setup_axis(ax, label, "count")
            setup_integer_value_axis(ax, values)
            ax.legend(loc="best", fontsize=7)

    axes[-1].set_xlabel("flight-log relative time [s]")
    shade_mode_background(axes, ulog, base_timestamp, enabled=shade_modes)
    return save_fig(fig, fig_dir, "reset_counters")


def plot_status_mask_field(
    ulog: ULog,
    base_timestamp: int,
    fig_dir: Path,
    field: str,
    title: str,
    output_name: str,
    shade_modes: bool = True,
) -> Path:
    status = get_ds(ulog, "estimator_status")
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(10.5, 4.6),
        constrained_layout=True,
    )

    if status is None or field not in status.data:
        plot_unavailable(ax, f"estimator_status.{field} not logged")
        setup_boolean_axis(ax, title)
    else:
        t = t_rel(status, base_timestamp)
        values = arr(status, field, int)
        observed = ", ".join(str(value) for value in sorted({int(value) for value in values}))
        bits = active_bit_indices(values)
        if bits:
            colors = [
                "tab:blue",
                "tab:orange",
                "tab:green",
                "tab:red",
                "tab:purple",
                "tab:brown",
                "tab:pink",
            ]
            series = [
                (f"bit {bit}", t, ((values >> bit) & 1).astype(int), colors[index % len(colors)])
                for index, bit in enumerate(bits)
            ]
            plot_boolean_group_sources(ax, series, title)
        else:
            ax.step(t, np.zeros(len(values), dtype=int), where="post", lw=1.0, color="tab:blue")
            setup_boolean_axis(ax, title)
            ax.text(
                0.5,
                0.52,
                "logged; no active bits observed",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        ax.text(
            0.01,
            0.98,
            f"observed raw values: {observed if observed else 'none'}",
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.65, "lw": 0},
        )

    ax.set_xlabel("flight-log relative time [s]")
    shade_mode_background(ax, ulog, base_timestamp, enabled=shade_modes)
    add_system_time_axis(ax, ulog, base_timestamp)
    return save_fig(fig, fig_dir, output_name)


def plot_status_masks(
    ulog: ULog, base_timestamp: int, fig_dir: Path, shade_modes: bool = True
) -> Path:
    status = get_ds(ulog, "estimator_status")
    fields = [
        ("control_mode_flags", "Control mode flags", "tab:blue"),
        ("filter_fault_flags", "Filter fault flags", "tab:red"),
        ("solution_status_flags", "Solution status flags", "tab:purple"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.0), sharex=True, constrained_layout=True)
    axes = np.ravel(np.atleast_1d(axes))

    for ax, (field, title, color) in zip(axes, fields):
        if status is None or field not in status.data:
            plot_unavailable(ax, f"estimator_status.{field} not logged")
            setup_axis(ax, title, "raw mask")
            continue
        t = t_rel(status, base_timestamp)
        values = arr(status, field, int)
        ax.step(t, values, where="post", lw=1.0, color=color, label=field)
        observed = ", ".join(str(value) for value in sorted({int(value) for value in values}))
        ax.text(
            0.01,
            0.98,
            f"observed: {observed if observed else 'none'}",
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.65, "lw": 0},
        )
        setup_axis(ax, title, "raw mask")
        setup_integer_value_axis(ax, values)
        ax.legend(loc="best", fontsize=7)

    axes[-1].set_xlabel("flight-log relative time [s]")
    shade_mode_background(axes, ulog, base_timestamp, enabled=shade_modes)
    return save_fig(fig, fig_dir, "status_masks")


def plot_status_mask_bits_legacy(
    ulog: ULog, base_timestamp: int, fig_dir: Path, shade_modes: bool = True
) -> Path:
    status = get_ds(ulog, "estimator_status")
    bit_fields = []
    if status is not None:
        t = t_rel(status, base_timestamp)
        for field in available_fields(status, _STATUS_MASK_FIELDS):
            values = arr(status, field, int)
            bits = active_bit_indices(values)
            if not bits:
                bit_fields.append((f"{field}: no active bits", t, np.zeros(len(values), dtype=int)))
                continue
            for bit in bits:
                bit_fields.append((f"{field}: bit {bit}", t, ((values >> bit) & 1).astype(int)))
    nrows = max(1, len(bit_fields))
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(10.5, max(3.0, 0.85 + 0.9 * nrows)),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.ravel(np.atleast_1d(axes))

    if not bit_fields or status is None:
        axes[0].text(
            0.5,
            0.5,
            "estimator_status masks not logged",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
        )
        setup_boolean_axis(axes[0], "Status masks unavailable")
    else:
        for ax, (label, t, values) in zip(axes, bit_fields):
            plot_boolean_trace(ax, t, values, label, color="tab:blue")

    axes[-1].set_xlabel("flight-log relative time [s]")
    shade_mode_background(axes, ulog, base_timestamp, enabled=shade_modes)
    return save_fig(fig, fig_dir, "status_mask_bits_legacy")


def plot_innovation_check_flags(
    ulog: ULog, base_timestamp: int, fig_dir: Path, shade_modes: bool = True
) -> Path:
    status = get_ds(ulog, "estimator_status")
    nrows = len(_INNOVATION_CHECK_FLAG_GROUPS)
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(10.5, max(3.0, 0.72 + 1.0 * nrows)),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.ravel(np.atleast_1d(axes))

    if status is None or "innovation_check_flags" not in status.data:
        for ax in axes:
            ax.text(
                0.5,
                0.5,
                "estimator_status.innovation_check_flags not logged",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        setup_axis(axes[0], "Innovation check flags unavailable", "bit value")
    else:
        t = t_rel(status, base_timestamp)
        raw = arr(status, "innovation_check_flags", int)
        for ax, (label, bit_shift, mask) in zip(axes, _INNOVATION_CHECK_FLAG_GROUPS):
            values = innovation_check_flag_group_values(raw, bit_shift, mask)
            ax.step(t, values, where="post", lw=1.0, color="tab:red")
            if mask == 1:
                setup_boolean_axis(ax, label)
            else:
                setup_axis(ax, label, f"0-{mask}")
                setup_integer_value_axis(ax, values)

    axes[-1].set_xlabel("flight-log relative time [s]")
    shade_mode_background(axes, ulog, base_timestamp, enabled=shade_modes)
    return save_fig(fig, fig_dir, "innovation_check_flags")


def plot_estimator_event_flags(
    ulog: ULog, base_timestamp: int, fig_dir: Path, shade_modes: bool = True
) -> Path:
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
        axes[0].text(
            0.5,
            0.5,
            "no active estimator_event_flags booleans logged",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
        )
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


def plot_innovation_ratios(
    ulog: ULog, base_timestamp: int, fig_dir: Path, shade_modes: bool = True
) -> Path:
    channels = [
        ("gps_hvel[0]", "GPS horizontal velocity innovation test ratio 0"),
        ("gps_hvel[1]", "GPS horizontal velocity innovation test ratio 1"),
        ("gps_vvel", "GPS vertical velocity innovation test ratio"),
        ("gps_hpos[0]", "GPS horizontal position innovation test ratio 0"),
        ("gps_hpos[1]", "GPS horizontal position innovation test ratio 1"),
        ("gps_vpos", "GPS vertical position innovation test ratio"),
        ("baro_vpos", "Baro vertical position innovation test ratio"),
        ("rng_vpos", "Range vertical position innovation test ratio"),
        ("heading", "Heading innovation test ratio"),
    ]
    fig, axes = plt.subplots(
        len(channels), 1, figsize=(10.5, 11.6), sharex=True, constrained_layout=True
    )

    for ax, (key, title) in zip(axes, channels):
        series = get_metric_series(ulog, base_timestamp, key, "test_ratio")
        if series is not None:
            y = np.where(np.isfinite(series.values), series.values, np.nan)
            ax.plot(series.t, y, lw=0.9, color="tab:blue")
        else:
            plot_unavailable(ax, f"{key} test ratio not logged")
        ax.axhline(1.0, color="black", lw=0.9, ls="--", label="gate")
        ax.axhline(0.36, color="gray", lw=0.7, ls=":", label="3 sigma equiv")
        setup_axis(ax, title, "test ratio")
        ax.set_yscale("symlog", linthresh=0.1)

    axes[-1].set_xlabel("flight-log relative time [s]")
    shade_mode_background(axes, ulog, base_timestamp, enabled=shade_modes)

    return save_fig(fig, fig_dir, "innovation_test_ratios")


def plot_height_innovation_test_ratios(
    ulog: ULog, base_timestamp: int, fig_dir: Path, shade_modes: bool = True
) -> Path:
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
            plot_unavailable(
                ax,
                f"{key} test ratio not logged; report does not compute this channel",
            )
            setup_axis(ax, f"{title} missing", "ratio")
            ax.legend(loc="best", fontsize=7)
            continue

        y = np.where(np.isfinite(series.values), series.values, np.nan)
        ax.plot(series.t, y, lw=0.9, color="tab:blue", label=series.source)
        ax.axhline(0.0, color="gray", lw=0.7, ls=":", label="zero")
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
        ax.set_ylim(-0.03 * y_max, y_max)
        if finite.size and np.nanmax(np.abs(finite)) <= 1e-12:
            ax.text(
                0.5,
                0.2,
                f"{key} is logged but all samples are zero",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
            )
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
        out["baro_reference_source"] = "no rangefinder — baro divergence not computed"
        return out

    out["baro_reference_source"] = "rangefinder"
    _t_div, _baro_alt, divergence, offset = baro_result

    finite_div = divergence[np.isfinite(divergence)]
    if finite_div.size:
        out["baro_range_offset_m"] = f"{offset:.2f}"
        out["baro_range_max_divergence_m"] = f"{np.max(np.abs(finite_div)):.2f}"
        out["baro_range_rms_divergence_m"] = f"{np.sqrt(np.mean(finite_div**2)):.2f}"

    pw_result = detect_propwash_events(
        ulog, base_timestamp, divergence_threshold_m, throttle_threshold
    )
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
        out["gps_alt_drift_m"] = f"{gps_alt[-1] - gps_alt[0]:.1f} (informational only)"
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
        for key in [
            "reset_count_vel_ne",
            "reset_count_vel_d",
            "reset_count_pos_ne",
            "reset_count_pod_d",
            "reset_count_quat",
        ]:
            if key in status.data:
                values = arr(status, key, int)
                out[key] = f"{values[0]}--{values[-1]}"

    if flags:
        t = t_rel(flags, base_timestamp)
        for key in [
            "cs_gps",
            "cs_gnss_vel",
            "cs_gps_hgt",
            "cs_baro_hgt",
            "cs_rng_hgt",
            "fs_bad_acc_vertical",
            "cs_inertial_dead_reckoning",
        ]:
            if key in flags.data:
                out[key] = format_spans(bool_spans(t, arr(flags, key, int)), limit=3)

    for key in [
        "gps_vpos",
        "gps_vvel",
        "gps_hvel[0]",
        "gps_hvel[1]",
        "gps_hpos[0]",
        "gps_hpos[1]",
        "baro_vpos",
        "rng_vpos",
    ]:
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
                out[key] = (
                    f"{int(values.sum())}/{values.size}; {format_spans(bool_spans(t, values), limit=3)}"
                )

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


def latex_breakable_path(path: Path) -> str:
    escaped = latex_breakable_text(path.as_posix())
    for marker in ["/", " ", "-", "(", ")"]:
        escaped = escaped.replace(marker, marker + r"\allowbreak{}")
    return escaped


def source_log_latex_line(config: ReviewConfig) -> str:
    return rf"\noindent\textbf{{Source ULog:}} \texttt{{{latex_breakable_path(config.log.path)}}}"


def latex_html_color(color: str) -> str:
    return color.strip().lstrip("#").upper()


def status_latex_row_prefix(status: str) -> str:
    normalized = status.strip().upper()
    if normalized == "ENABLED":
        return r"\rowcolor[HTML]{DFF0D8}"
    if normalized == "DISABLED":
        return r"\rowcolor[HTML]{F2DEDE}"
    if normalized == "NOT LOGGED":
        return r"\rowcolor[HTML]{EFEFEF}"
    return ""


def status_latex_text(status: str) -> str:
    normalized = status.strip().upper()
    if normalized == "ENABLED":
        return r"\textbf{ENABLED}"
    if normalized == "DISABLED":
        return r"\textbf{DISABLED}"
    if normalized == "NOT LOGGED":
        return r"\texttt{not logged}"
    return latex_escape(status)


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
\begin{{tabular}}{{|l|l|r|r|}}
\hline
Field & Literal meaning & Duration s & Armed \% \\
\hline
{body}
\end{{tabular}}%
}}
\caption{{Active estimator fusion state from \texttt{{estimator\_status\_flags}}.
Derived combined rows use OR logic; individual rows still show the underlying
EKF mode split. Durations are accumulated over armed time.}}
\end{{table}}"""


def sensor_status_latex_table(rows: list[SensorStatusRow]) -> str:
    if not rows:
        return "No sensor availability or EKF control parameters were logged."

    lines = []
    for row in rows:
        row_prefix = status_latex_row_prefix(row.status)
        cells = [
            latex_escape(row.sensor),
            latex_escape(row.name),
            latex_escape(row.value),
            status_latex_text(row.status),
        ]
        prefix = f"{row_prefix}\n" if row_prefix else ""
        lines.append(prefix + " & ".join(cells) + r" \\")
        lines.append(r"\hline")
    body = "\n".join(lines)

    return rf"""\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{longtable}}{{|p{{0.20\textwidth}}|p{{0.31\textwidth}}|p{{0.18\textwidth}}|p{{0.21\textwidth}}|}}
\caption{{Logged sensor availability bits and EKF sensor-control parameters.
\texttt{{not logged}} means the value was absent from ULog
\texttt{{initial\_parameters}}, not necessarily absent from firmware. Decoded
bitmask details are shown inside the relevant sensor subsections.}}\\
\hline
Sensor & Parameter & Logged value & Interpreted status \\
\hline
\endfirsthead
\hline
Sensor & Parameter & Logged value & Interpreted status \\
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


def parameter_bitmask_latex_table(rows: list[ParameterBitmaskRow], parameter: str) -> str:
    parameter_rows = [row for row in rows if row.parameter == parameter]
    if not parameter_rows:
        return rf"No decoded \texttt{{{latex_escape(parameter)}}} rows were requested."

    raw_value = parameter_rows[0].raw_value

    body = "\n".join(
        (f"{status_latex_row_prefix(row.status)}\n" if status_latex_row_prefix(row.status) else "")
        + " & ".join([str(row.bit), status_latex_text(row.status), latex_escape(row.meaning)])
        + r" \\"
        + "\n\\hline"
        for row in parameter_rows
    )

    return rf"""\begin{{table}}[H]
\centering
\scriptsize
\begin{{tabular}}{{|r|p{{0.18\textwidth}}|p{{0.62\textwidth}}|}}
\hline
Bit & Status & Meaning \\
\hline
{body}
\end{{tabular}}
\caption{{Decoded \texttt{{{latex_escape(parameter)}}} bitmask from ULog
\texttt{{initial\_parameters}}. Logged raw value: \texttt{{{latex_escape(raw_value)}}}.}}
\end{{table}}"""


def parameter_enum_latex_table(rows: list[ParameterEnumRow], parameter: str) -> str:
    parameter_rows = [row for row in rows if row.parameter == parameter]
    if not parameter_rows:
        return rf"No decoded \texttt{{{latex_escape(parameter)}}} rows were requested."

    raw_value = parameter_rows[0].raw_value
    body = "\n".join(
        (f"{status_latex_row_prefix(row.status)}\n" if status_latex_row_prefix(row.status) else "")
        + " & ".join(
            [
                str(row.value),
                r"\textbf{yes}" if row.selected else "no",
                status_latex_text(row.status),
                latex_escape(row.meaning),
            ]
        )
        + r" \\"
        + "\n\\hline"
        for row in parameter_rows
    )

    return rf"""\begin{{table}}[H]
\centering
\scriptsize
\begin{{tabular}}{{|r|p{{0.13\textwidth}}|p{{0.18\textwidth}}|p{{0.52\textwidth}}|}}
\hline
Value & Selected & Status & Meaning \\
\hline
{body}
\end{{tabular}}
\caption{{Decoded \texttt{{{latex_escape(parameter)}}} enum from ULog
\texttt{{initial\_parameters}}. Logged raw value: \texttt{{{latex_escape(raw_value)}}}.}}
\end{{table}}"""


def status_mask_latex_table(rows: list[StatusMaskRow]) -> str:
    if not rows:
        return "No \\texttt{estimator\\_status} mask fields were logged."

    body = "\n".join(
        " & ".join(
            [
                latex_escape(row.field),
                latex_breakable_text(row.observed_values),
                latex_escape(row.active_bits),
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
\begin{{tabular}}{{|l|l|l|}}
\hline
Field & Observed raw values & Active bit rows plotted \\
\hline
{body}
\end{{tabular}}%
}}
\caption{{Observed raw values and active bit rows for \texttt{{estimator\_status}}
mask fields. These fields are bitmasks, not simple enum values.}}
\end{{table}}"""


def mpc_alt_mode_latex_table(summary: dict[str, str]) -> str:
    rows = [
        ("MPC_ALT_MODE", summary.get("mpc_alt_mode", "not logged")),
        ("MPC_ALT_MODE label", summary.get("mpc_alt_mode_label", "unknown")),
        ("Controller behavior", summary.get("mpc_alt_mode_detail", "unknown")),
    ]
    related = summary.get("mpc_alt_mode_related_params", "none logged")
    for item in related.split(";"):
        name, sep, value = item.strip().partition("=")
        if sep:
            rows.append((name.strip(), value.strip()))
        elif item.strip() and item.strip() != "none logged":
            rows.append(("Related parameter", item.strip()))

    body = "\n".join(
        " & ".join([latex_escape(name), latex_breakable_text(value)]) + r" \\" + "\n\\hline"
        for name, value in rows
    )

    return rf"""\begin{{table}}[H]
\centering
\scriptsize
\begin{{tabular}}{{|p{{0.30\textwidth}}|p{{0.62\textwidth}}|}}
\hline
Logged parameter or derived field & Value \\
\hline
{body}
\end{{tabular}}
\caption{{Logged \texttt{{MPC\_ALT\_MODE}} value and related altitude-hold
parameters from ULog \texttt{{initial\_parameters}}.}}
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
        return (
            "No \\texttt{EKF2\\_*}, \\texttt{MPC\\_*}, \\texttt{SENS\\_*}, "
            "or \\texttt{SDLOG\\_*} initial parameters were logged."
        )

    tables = []
    for prefix in ["EKF2", "MPC", "SENS", "SDLOG"]:
        group_rows = [row for row in rows if row.prefix == prefix]
        if not group_rows:
            continue

        body_rows = []
        for index in range(0, len(group_rows), 2):
            left = group_rows[index]
            right = group_rows[index + 1] if index + 1 < len(group_rows) else None
            body_rows.append(
                " & ".join(
                    [
                        latex_breakable_text(left.name),
                        latex_breakable_text(left.value),
                        latex_breakable_text(right.name) if right is not None else "",
                        latex_breakable_text(right.value) if right is not None else "",
                    ]
                )
                + r" \\"
                + "\n\\hline"
            )
        body = "\n".join(body_rows)

        tables.append(
            rf"""\subsection*{{\texttt{{{prefix}\_*}}}}
\fontsize{{11}}{{13}}\selectfont
\setlength{{\tabcolsep}}{{3pt}}
\renewcommand{{\arraystretch}}{{0.92}}
\setlength{{\LTleft}}{{0pt}}
\setlength{{\LTright}}{{0pt}}
\begin{{longtable}}{{|p{{0.27\textwidth}}|p{{0.17\textwidth}}|p{{0.27\textwidth}}|p{{0.17\textwidth}}|}}
\caption{{Logged \texttt{{{prefix}\_*}} ULog initial parameters ({len(group_rows)} rows). These are the values saved in the log; firmware-default comparison is not applied in this table.}}\\
\hline
Parameter & Logged value & Parameter & Logged value \\
\hline
\endfirsthead
\hline
Parameter & Logged value & Parameter & Logged value \\
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
\renewcommand{{\arraystretch}}{{1.12}}
\normalsize"""
        )

    return "\n\n".join(tables)


def build_latex(
    config: ReviewConfig,
    figures: dict[str, Path],
    summary: dict[str, str],
    sensor_rows: list[SensorStatusRow],
    bitmask_rows: list[ParameterBitmaskRow],
    enum_rows: list[ParameterEnumRow],
    fusion_rows: list[FusionStateRow],
    reset_rows: list[ResetEventRow],
    exception_rows: list[EstimatorExceptionRow],
    nav_state_rows: list[NavStateRow],
    parameter_rows: list[ParameterRow],
) -> str:
    rel_figures = {
        key: path.relative_to(config.output_dir).as_posix() for key, path in figures.items()
    }
    sensor_table = sensor_status_latex_table(sensor_rows)
    imu_ctrl_table = parameter_bitmask_latex_table(bitmask_rows, "EKF2_IMU_CTRL")
    gps_ctrl_table = parameter_bitmask_latex_table(bitmask_rows, "EKF2_GPS_CTRL")
    mag_check_table = parameter_bitmask_latex_table(bitmask_rows, "EKF2_MAG_CHECK")
    gps_check_table = parameter_bitmask_latex_table(bitmask_rows, "EKF2_GPS_CHECK")
    rng_ctrl_table = parameter_enum_latex_table(enum_rows, "EKF2_RNG_CTRL")
    fusion_table = fusion_state_latex_table(fusion_rows)
    reset_table = reset_event_latex_table(reset_rows)
    exception_table = estimator_exception_latex_table(exception_rows)
    nav_state_table = nav_state_latex_table(nav_state_rows)
    nav_state_color_key = nav_state_color_key_latex_table()
    parameter_table = logged_parameter_latex_table(parameter_rows)
    mpc_alt_table = mpc_alt_mode_latex_table(summary)
    source_log_line = source_log_latex_line(config)
    has_rangefinder = summary.get("baro_reference_source") == "rangefinder"
    if has_rangefinder:
        altitude_caption = (
            "EKF local altitude ($-z$), rangefinder AGL reference, and baro aligned to rangefinder. "
            "GPS MSL shown as dotted reference only. The bottom panel compares the EKF HAGL "
            "state from \\texttt{vehicle\\_local\\_position.dist\\_bottom} with derived terrain "
            "state traces when logged."
        )
        propwash_intro = (
            rf"Propwash events (red spans) occur when $|\text{{baro}} - \text{{rangefinder}}| > "
            rf"{latex_escape(f'{config.divergence_threshold_m:.2f}')}$~m and thrust $> "
            rf"{latex_escape(f'{config.throttle_threshold:.2f}')}$."
        )
        propwash_caption = (
            "Panel 1: altitude comparison using the rangefinder as an AGL reference. "
            "Panel 2: baro$-$rangefinder divergence with threshold bands. "
            "Panel 3: normalized thrust. Panel 4: vehicle tilt. "
            "Panel 5: EKF baro height innovation ratio. Red spans are propwash events."
        )
    else:
        altitude_caption = (
            "EKF local altitude, baro altitude, and GPS MSL reference. "
            "No \\texttt{distance\\_sensor} topic was logged. The bottom panel compares the EKF "
            "HAGL state from \\texttt{vehicle\\_local\\_position.dist\\_bottom} with derived "
            "terrain state traces when logged."
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
    terrain_bitfield = latex_escape(
        summary.get("terrain_dist_bottom_sensor_bitfield", "not logged")
    )
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
\hypersetup{{hidelinks}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{0.5em}}
\renewcommand{{\arraystretch}}{{1.12}}

\title{{{latex_escape(config.title)}}}
\author{{Generated from ULog}}
\date{{}}

\begin{{document}}
\maketitle
{source_log_line}

\tableofcontents

\clearpage
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

\clearpage
\section{{Sensor Health}}
{sensor_table}

\subsection{{IMU Integrity}}
\subsubsection*{{\texttt{{EKF2\_IMU\_CTRL}}}}
{imu_ctrl_table}

\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth,height=0.84\textheight,keepaspectratio]{{{rel_figures["imu_health"]}}}
\caption{{IMU health and integrity checks. The panels show logged vibration/coning
metrics as separate vibration and coning traces, accelerometer and gyroscope
clipping counters or masks, EKF IMU quality bits including
\texttt{{fs\_bad\_acc\_vertical}} and
\texttt{{fs\_bad\_acc\_clipping}}, and raw estimator health/fault masks.}}
\end{{figure}}

\clearpage
\subsection{{IMU Raw and Bias-Adjusted Measurements}}
\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth,height=0.82\textheight,keepaspectratio]{{{rel_figures["accel_bias_adjustment"]}}}
\caption{{Accelerometer processing and bias adjustment by axis. Each axis
compares sensor/board-frame FRD \texttt{{sensor\_accel}} points, calibrated body
FRD \texttt{{sensor\_combined}} measurements, and the bias-adjusted filtered
\texttt{{vehicle\_acceleration}} output. Sparse sensor-level points reflect the
ULog profile rather than the hardware sample rate. The final panels show EKF
in-run bias, one-standard-deviation uncertainty, limit, validity, and
stability.}}
\end{{figure}}

\clearpage
\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth,height=0.86\textheight,keepaspectratio]{{{rel_figures["gyro_bias_adjustment"]}}}
\caption{{Gyroscope processing and bias adjustment by axis. Each axis compares
sensor/board-frame FRD \texttt{{sensor\_gyro}} points, calibrated body FRD
\texttt{{sensor\_combined}} measurements, and the bias-adjusted filtered
\texttt{{vehicle\_angular\_velocity}} output. Sparse sensor-level points reflect
the ULog profile rather than the hardware sample rate. The final panels show EKF
gyro bias, uncertainty, limit, validity, and stability.}}
\end{{figure}}

\clearpage
\subsection{{Barometer Integrity}}
\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth,height=0.84\textheight,keepaspectratio]{{{rel_figures["barometer_health"]}}}
\caption{{Barometer health and fusion checks. The panels show baro altitude,
logged pressure/temperature-style fields as separate traces when present, baro
height innovation test ratio, and baro height fusion/timeout/reset status.
Health is shown even when \texttt{{EKF2\_BARO\_CTRL}} is off or absent.}}
\end{{figure}}

{propwash_intro}

\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{{rel_figures["baro_propwash"]}}}
\caption{{{propwash_caption}}}
\end{{figure}}

\clearpage
\subsection{{Magnetometer Integrity}}
\subsubsection*{{\texttt{{EKF2\_MAG\_CHECK}}}}
{mag_check_table}

\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth,height=0.84\textheight,keepaspectratio]{{{rel_figures["magnetometer_health"]}}}
\caption{{Magnetometer health and integrity checks. The panels show logged mag
components when available, heading innovation test ratio, and yaw rejection
flags. Magnetometer control-status bits are collected in the Control Statuses
section.}}
\end{{figure}}

\clearpage
\subsection{{Attitude Overview}}
\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth,height=0.84\textheight,keepaspectratio]{{{rel_figures["attitude_overview"]}}}
\caption{{Attitude overview from \texttt{{vehicle\_attitude}},
\texttt{{vehicle\_attitude\_setpoint}}, and
\texttt{{vehicle\_angular\_velocity}} when logged. The horizontal-motion panel
provides context for attitude and heading observability.}}
\end{{figure}}

\clearpage
\subsection{{Pitch and Forward-Lurch Diagnostics}}
\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth,height=0.88\textheight,keepaspectratio]{{{rel_figures["forward_lurch_diagnostics"]}}}
\caption{{Pitch-response chain for diagnosing an uncommanded forward lurch.
The panels compare pitch setpoint with measured pitch, pitch-rate setpoint with
measured body pitch rate, commanded with unallocated pitch torque, front and
rear motor command means derived from signed \texttt{{CA\_ROTOR*\_PX}}
geometry, and EKF accelerometer bias. A positive front-minus-rear differential
means the mixer commanded more output at the front rotors.}}
\end{{figure}}

\clearpage
\subsection{{GPS / GNSS Integrity}}
\subsubsection*{{\texttt{{EKF2\_GPS\_CTRL}}}}
{gps_ctrl_table}

\subsubsection*{{\texttt{{EKF2\_GPS\_CHECK}}}}
{gps_check_table}

\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth,height=0.84\textheight,keepaspectratio]{{{rel_figures["quality"]}}}
\caption{{{gps_caption}}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth,height=0.84\textheight,keepaspectratio]{{{rel_figures["gps_checks"]}}}
\caption{{GPS acceptance checks from \texttt{{estimator\_gps\_status}}. Each
available check-failure field is plotted as its own 0/1 trace so \texttt{{max\_pdop}},
speed, drift, and position checks remain readable.}}
\end{{figure}}

\clearpage
\subsection{{Optical Flow Integrity}}
\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth,height=0.84\textheight,keepaspectratio]{{{rel_figures["optical_flow_health"]}}}
\caption{{Optical-flow health and integrity checks. The panels show sensor
quality, raw \texttt{{vehicle\_optical\_flow.pixel\_flow}} axes, EKF
aid-source innovation test ratios, aid-source fusion/rejection state when
logged, optical-flow fault/rejection flags, and the EKF-fused optical-flow body
velocity.
\texttt{{cs\_opt\_flow}} is collected in the Control Statuses section.}}
\end{{figure}}

\clearpage
\subsection{{Range Finder Integrity}}
\subsubsection*{{\texttt{{EKF2\_RNG\_CTRL}}}}
{rng_ctrl_table}

\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth,height=0.84\textheight,keepaspectratio]{{{rel_figures["rangefinder_health"]}}}
\caption{{Range finder health and integrity checks.}}
\end{{figure}}

\clearpage
\subsection{{External Vision Integrity}}
\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth,height=0.84\textheight,keepaspectratio]{{{rel_figures["external_vision_position"]}}}
\caption{{External-vision XYZ position measurements compared with the EKF local
position estimate.}}
\end{{figure}}

\clearpage
\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth,height=0.84\textheight,keepaspectratio]{{{rel_figures["external_vision_health"]}}}
\caption{{External-vision health and integrity checks.}}
\end{{figure}}

\clearpage
\section{{Sensor Latency}}
\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth,height=0.88\textheight,keepaspectratio]{{{rel_figures["sensor_latency"]}}}
\caption{{Sensor sample-to-publication latency.}}
\end{{figure}}

\clearpage
\section{{Active Estimator Fusion State}}
{fusion_table}

\section{{Estimator Reset Events}}
{reset_table}

\section{{Estimator Health Exceptions}}
{exception_table}

\section{{MPC\_ALT\_MODE}}
{mpc_alt_table}
\textbf{{Estimator distinction:}} \texttt{{MPC\_ALT\_MODE}} changes the multicopter
position controller altitude reference behavior. It does not select the EKF height
reference; that is reported separately through \texttt{{EKF2\_HGT\_REF}} and the
control-status fields such as \texttt{{cs\_baro\_hgt}} and
\texttt{{cs\_rng\_hgt}}, shown in the Control Statuses section.

\section{{Terrain / HAGL Estimate}}
\begin{{itemize}}
\item \textbf{{Source field:}} \texttt{{vehicle\_local\_position.dist\_bottom}}
\item \textbf{{Logged HAGL range:}} {terrain_dist_bottom}
\item \textbf{{Valid spans:}} {terrain_valid}
\item \textbf{{Source bitfield values:}} {terrain_bitfield}
\end{{itemize}}

\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{{rel_figures["terrain"]}}}
\caption{{Terrain/HAGL estimate from \texttt{{vehicle\_local\_position}}. The
first panel combines \texttt{{dist\_bottom}} with logged HAGL min/max bounds and
the implied local terrain vertical position \texttt{{z + dist\_bottom}} when
available. The remaining panels show \texttt{{dist\_bottom\_valid}} and
\texttt{{dist\_bottom\_sensor\_bitfield != 0}}. EKF control-status bits are
collected in the Control Statuses section.}}
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
\section{{PX4 Setpoint Diagnostics}}
\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth,height=0.90\textheight,keepaspectratio]{{{rel_figures["px4_setpoint_commands"]}}}
\caption{{PX4 setpoint command diagnostics from trajectory, offboard-control, and
local-position-setpoint topics.}}
\end{{figure}}

\clearpage
\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth,height=0.90\textheight,keepaspectratio]{{{rel_figures["px4_setpoint_response"]}}}
\caption{{PX4 setpoint response diagnostics from local-position and
attitude-setpoint topics.}}
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

\section{{Motor / Actuator / ESC Outputs}}
\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{{rel_figures["actuator_outputs"]}}}
\caption{{Motor, actuator, and ESC output overview. The first panel prefers
\texttt{{actuator\_motors.control[]}} when dynamic control allocation is logged,
then falls back to varying \texttt{{actuator\_outputs.output[]}} channels. The
second panel breaks out
\texttt{{control\_allocator\_status.actuator\_saturation[]}} fields with
\texttt{{ACTUATOR\_SATURATION\_OK}},
\texttt{{ACTUATOR\_SATURATION\_UPPER\_DYN}},
\texttt{{ACTUATOR\_SATURATION\_UPPER}},
\texttt{{ACTUATOR\_SATURATION\_LOWER\_DYN}}, and
\texttt{{ACTUATOR\_SATURATION\_LOWER}} state labels. The third panel shows
normalized thrust when logged, and the fourth panel plots all logged ESC RPM
channels from \texttt{{esc\_status.esc[i].esc\_rpm}} together.}}
\end{{figure}}

\section{{Local XY Position}}
\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{{rel_figures["local_xy"]}}}
\caption{{EKF local horizontal position. The first panel is the local XY track with
start/end markers; the second and third panels show local x and local y as
functions of flight-log relative time.}}
\end{{figure}}

\section{{Control Statuses}}
\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth,height=0.88\textheight,keepaspectratio]{{{rel_figures["control_status_flags"]}}}
\caption{{All logged \texttt{{estimator\_status\_flags.cs\_*}} control-status
fields, grouped by subsystem. Each field is plotted as a stacked 0/1 trace.}}
\end{{figure}}

\clearpage
\section{{Inertial Dead Reckoning Diagnostics}}
\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth,height=0.88\textheight,keepaspectratio]{{{rel_figures["inertial_dead_reckoning"]}}}
\caption{{Inertial dead-reckoning context. The figure combines EKF local
velocity and acceleration, dead-reckoning and vehicle-state flags, horizontal
aiding and height control status, per-sample optical-flow aid-source status, and
horizontal velocity/position reset counters. Dotted reset-event markers identify
\texttt{{reset\_vel\_to\_flow}} and
\texttt{{reset\_pos\_to\_last\_known}} reports when logged. This separates an
enabled aiding control path from measurements that were actually fused.}}
\end{{figure}}

\section{{Reset Counters}}
\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{{rel_figures["reset_counters"]}}}
\caption{{State reset counters from \texttt{{estimator\_status}} and
\texttt{{vehicle\_local\_position}}. Each counter is shown in its own subplot;
red dotted lines mark logged reset-event counter increments.}}
\end{{figure}}

\section{{Estimator Status Masks}}
The core \texttt{{estimator\_status}} masks are shown separately so control-mode
state, filter faults, and solution status remain readable without a raw-value
table. Each plot annotates the observed raw mask values.

\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{{rel_figures["control_mode_flags"]}}}
\caption{{\texttt{{estimator\_status.control\_mode\_flags}} active observed bits.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{{rel_figures["filter_fault_flags"]}}}
\caption{{\texttt{{estimator\_status.filter\_fault\_flags}} active observed bits.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{{rel_figures["solution_status_flags"]}}}
\caption{{\texttt{{estimator\_status.solution\_status\_flags}} active observed bits.}}
\end{{figure}}

\section{{Innovation Check Flags}}
\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth,height=0.88\textheight,keepaspectratio]{{{rel_figures["innovation_check_flags"]}}}
\caption{{Decoded \texttt{{estimator\_status.innovation\_check\_flags}} using the
same bit grouping Flight Review uses for velocity, horizontal position, vertical
position, magnetometer, yaw, airspeed, synthetic sideslip, HAGL, and optical-flow
checks. Multi-bit groups show the grouped integer value.}}
\end{{figure}}

\section{{Optical Flow Fusion Gating}}
\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth,height=0.84\textheight,keepaspectratio]{{{rel_figures["optical_flow_gating"]}}}
\caption{{Optical-flow fusion gating diagnostics: normalized
\texttt{{flow[0]}}/\texttt{{flow[1]}} test ratios, innovations against
$\pm5\sigma$ gates, \texttt{{timestamp - time\_last\_fuse}} fusion delay,
and logged fusion/rejection/fault status bits. Sensor latency is collected in
the Sensor Latency section.}}
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
\includegraphics[width=\textwidth,height=0.88\textheight,keepaspectratio]{{{rel_figures["ratios"]}}}
\caption{{Innovation test ratios. 1.0 is the configured gate; 0.36 is approximately 3-sigma with a 5-sigma gate.
\texttt{{baro\_vpos}} and \texttt{{rng\_vpos}} are the key channels for baro compensation evaluation.}}
\end{{figure}}

\section{{Height Innovation Test Ratios}}
\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{{rel_figures["height_ratios"]}}}
\caption{{Height-related innovation test ratios for \texttt{{hagl}},
\texttt{{hagl\_rate}}, \texttt{{baro\_vpos}}, and \texttt{{rng\_vpos}}. These
channels are plotted when logged; the report does not synthesize missing HAGL
test ratios. All-zero logged channels are annotated explicitly. The horizontal
red dashed line marks the 1.0 rejection gate; vertical red dotted lines mark
upward crossings where the logged test ratio first exceeds 1.0.}}
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

\section{{Logged EKF2, MPC, and SENS Parameters}}
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
    try:
        subprocess.run(
            ["latexmk", "-pdf", "-interaction=nonstopmode", tex_path.name],
            cwd=tex_path.parent,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        output = "\n".join(part for part in [exc.stdout, exc.stderr] if part)
        raise RuntimeError(f"latexmk failed for {tex_path}:\n{output}") from exc
    pdf_path = tex_path.with_suffix(".pdf")
    if not pdf_path.exists():
        raise FileNotFoundError(f"latexmk completed but did not produce {pdf_path}")
    return pdf_path


def generate_review(config: ReviewConfig) -> ReviewArtifacts:
    output_dir = config.output_dir
    fig_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not config.log.path.exists():
        raise FileNotFoundError(f"log file not found: {config.log.path}")

    ulog = ULog(str(config.log.path))
    base_timestamp = ulog.start_timestamp

    figures = {
        "position": plot_position_overview(
            ulog, base_timestamp, fig_dir, shade_modes=config.shade_flight_modes
        ),
        "local_xy": plot_local_xy_position(
            ulog, base_timestamp, fig_dir, shade_modes=config.shade_flight_modes
        ),
        "nav_state": plot_nav_state_timeline(
            ulog, base_timestamp, fig_dir, shade_modes=config.shade_flight_modes
        ),
        "position_tracking": plot_position_tracking(
            ulog, base_timestamp, fig_dir, shade_modes=config.shade_flight_modes
        ),
        "velocity_tracking": plot_velocity_tracking(
            ulog, base_timestamp, fig_dir, shade_modes=config.shade_flight_modes
        ),
        "px4_setpoint_commands": plot_px4_setpoint_commands(
            ulog, base_timestamp, fig_dir, shade_modes=config.shade_flight_modes
        ),
        "px4_setpoint_response": plot_px4_setpoint_response(
            ulog, base_timestamp, fig_dir, shade_modes=config.shade_flight_modes
        ),
        "manual_control": plot_manual_control_inputs(
            ulog, base_timestamp, fig_dir, shade_modes=config.shade_flight_modes
        ),
        "attitude_overview": plot_attitude_overview(
            ulog, base_timestamp, fig_dir, shade_modes=config.shade_flight_modes
        ),
        "forward_lurch_diagnostics": plot_forward_lurch_diagnostics(
            ulog, base_timestamp, fig_dir, shade_modes=config.shade_flight_modes
        ),
        "actuator_outputs": plot_actuator_outputs(
            ulog, base_timestamp, fig_dir, shade_modes=config.shade_flight_modes
        ),
        "terrain": plot_terrain_estimate(
            ulog, base_timestamp, fig_dir, shade_modes=config.shade_flight_modes
        ),
        "sensor_latency": plot_sensor_latency(
            ulog, base_timestamp, fig_dir, shade_modes=config.shade_flight_modes
        ),
        "imu_health": plot_imu_health(
            ulog, base_timestamp, fig_dir, shade_modes=config.shade_flight_modes
        ),
        "accel_bias_adjustment": plot_accel_bias_adjustment(
            ulog, base_timestamp, fig_dir, shade_modes=config.shade_flight_modes
        ),
        "gyro_bias_adjustment": plot_gyro_bias_adjustment(
            ulog, base_timestamp, fig_dir, shade_modes=config.shade_flight_modes
        ),
        "barometer_health": plot_barometer_health(
            ulog, base_timestamp, fig_dir, shade_modes=config.shade_flight_modes
        ),
        "magnetometer_health": plot_magnetometer_health(
            ulog, base_timestamp, fig_dir, shade_modes=config.shade_flight_modes
        ),
        "optical_flow_health": plot_optical_flow_sensor_health(
            ulog, base_timestamp, fig_dir, shade_modes=config.shade_flight_modes
        ),
        "rangefinder_health": plot_rangefinder_health(
            ulog, base_timestamp, fig_dir, shade_modes=config.shade_flight_modes
        ),
        "external_vision_position": plot_external_vision_position_comparison(
            ulog, base_timestamp, fig_dir, shade_modes=config.shade_flight_modes
        ),
        "external_vision_health": plot_external_vision_health(
            ulog, base_timestamp, fig_dir, shade_modes=config.shade_flight_modes
        ),
        "baro_propwash": plot_baro_propwash(
            ulog,
            base_timestamp,
            fig_dir,
            divergence_threshold_m=config.divergence_threshold_m,
            throttle_threshold=config.throttle_threshold,
            hover_max_speed_ms=config.hover_max_speed_ms,
            shade_modes=config.shade_flight_modes,
        ),
        "quality": plot_gps_quality(
            ulog, base_timestamp, fig_dir, shade_modes=config.shade_flight_modes
        ),
        "gps_checks": plot_gps_checks(
            ulog, base_timestamp, fig_dir, shade_modes=config.shade_flight_modes
        ),
        "control_status_flags": plot_control_status_flags(
            ulog, base_timestamp, fig_dir, shade_modes=config.shade_flight_modes
        ),
        "inertial_dead_reckoning": plot_inertial_dead_reckoning(
            ulog, base_timestamp, fig_dir, shade_modes=config.shade_flight_modes
        ),
        "reset_counters": plot_reset_counters(
            ulog, base_timestamp, fig_dir, shade_modes=config.shade_flight_modes
        ),
        "control_mode_flags": plot_status_mask_field(
            ulog,
            base_timestamp,
            fig_dir,
            field="control_mode_flags",
            title="Control mode flags",
            output_name="control_mode_flags",
            shade_modes=config.shade_flight_modes,
        ),
        "filter_fault_flags": plot_status_mask_field(
            ulog,
            base_timestamp,
            fig_dir,
            field="filter_fault_flags",
            title="Filter fault flags",
            output_name="filter_fault_flags",
            shade_modes=config.shade_flight_modes,
        ),
        "solution_status_flags": plot_status_mask_field(
            ulog,
            base_timestamp,
            fig_dir,
            field="solution_status_flags",
            title="Solution status flags",
            output_name="solution_status_flags",
            shade_modes=config.shade_flight_modes,
        ),
        "innovation_check_flags": plot_innovation_check_flags(
            ulog, base_timestamp, fig_dir, shade_modes=config.shade_flight_modes
        ),
        "optical_flow_gating": plot_optical_flow_fusion_gating(
            ulog, base_timestamp, fig_dir, shade_modes=config.shade_flight_modes
        ),
        "event_flags": plot_estimator_event_flags(
            ulog, base_timestamp, fig_dir, shade_modes=config.shade_flight_modes
        ),
        "ratios": plot_innovation_ratios(
            ulog, base_timestamp, fig_dir, shade_modes=config.shade_flight_modes
        ),
        "height_ratios": plot_height_innovation_test_ratios(
            ulog, base_timestamp, fig_dir, shade_modes=config.shade_flight_modes
        ),
        "vertical_residuals": plot_residuals(
            ulog,
            base_timestamp,
            fig_dir,
            keys=["baro_vpos", "rng_vpos", "gps_vpos", "gps_vvel"],
            name="vertical_residual_thresholds",
            title_prefix="Vertical",
            shade_modes=config.shade_flight_modes,
        ),
        "horizontal_residuals": plot_residuals(
            ulog,
            base_timestamp,
            fig_dir,
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
            ulog,
            base_timestamp,
            divergence_threshold_m=config.divergence_threshold_m,
            throttle_threshold=config.throttle_threshold,
            hover_max_speed_ms=config.hover_max_speed_ms,
        ),
    }
    sensor_rows = sensor_status_rows(ulog)
    bitmask_rows = parameter_bitmask_rows(ulog)
    enum_rows = parameter_enum_rows(ulog)
    fusion_rows = active_fusion_state_rows(ulog, base_timestamp)
    reset_rows = reset_event_rows(ulog, base_timestamp)
    exception_rows = estimator_exception_rows(ulog, base_timestamp)
    nav_state_rows = logged_nav_state_rows(ulog, base_timestamp)
    parameter_rows = logged_parameter_rows(ulog)

    summary_path = write_summary(output_dir, summary)

    tex = build_latex(
        config,
        figures,
        summary,
        sensor_rows,
        bitmask_rows,
        enum_rows,
        fusion_rows,
        reset_rows,
        exception_rows,
        nav_state_rows,
        parameter_rows,
    )
    tex_path = output_dir / "ekf2_review.tex"
    tex_path.write_text(tex, encoding="utf-8")

    pdf_path = compile_latex(tex_path)

    return ReviewArtifacts(
        output_dir=output_dir,
        tex_path=tex_path,
        summary_path=summary_path,
        pdf_path=pdf_path,
        figures=figures,
    )
