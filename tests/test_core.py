import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from ekf2_playback_review.analysis import (
    active_fusion_state_rows,
    bool_spans,
    estimator_exception_rows,
    exceedance_start_times,
    format_spans,
    compute_baro_gps_divergence,
    compute_tilt_deg,
    describe_mpc_alt_mode,
    describe_rangefinder_config,
    detect_propwash_events,
    gps_disabled_label,
    interp_at,
    interp_step_previous,
    innovation_check_flag_group_values,
    log_start_utc_us,
    logged_nav_state_rows,
    logged_parameter_rows,
    nav_state_spans,
    reset_event_rows,
    system_time_note,
    terrain_estimate_summary,
)


def _make_ds(name: str, timestamp: np.ndarray, **fields) -> MagicMock:
    ds = MagicMock()
    ds.name = name
    ds.multi_id = 0
    ds.data = {"timestamp": timestamp, **fields}
    return ds


def _make_ulog(datasets: list) -> MagicMock:
    ulog = MagicMock()
    ulog.data_list = datasets
    ulog.initial_parameters = {}
    ulog.msg_info_dict = {}
    return ulog


BASE_TS = int(1_000_000)


class SpanTests(unittest.TestCase):
    def test_bool_spans_empty(self):
        self.assertEqual(bool_spans(np.array([]), np.array([])), [])

    def test_bool_spans_segments(self):
        t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        values = np.array([False, True, True, False, True])
        self.assertEqual(bool_spans(t, values), [(1.0, 2.0), (4.0, 4.0)])

    def test_format_spans(self):
        self.assertEqual(format_spans([]), "none")
        self.assertEqual(format_spans([(1.0, 2.0), (4.0, 4.0)]), "1.000--2.000s, 4.000s")


class ExceedanceTests(unittest.TestCase):
    def test_exceedance_start_times_only_reports_upward_crossings(self):
        t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        values = np.array([0.2, 1.2, 1.4, 0.9, 1.1])

        np.testing.assert_allclose(exceedance_start_times(t, values, threshold=1.0), [1.0, 4.0])


class InnovationCheckFlagTests(unittest.TestCase):
    def test_decodes_single_and_grouped_innovation_check_bits(self):
        raw = np.array([0, 512, 1024, 2048, 3072], dtype=int)

        np.testing.assert_array_equal(innovation_check_flag_group_values(raw, 9, 0x1), [0, 1, 0, 0, 0])
        np.testing.assert_array_equal(innovation_check_flag_group_values(raw, 10, 0x3), [0, 0, 1, 2, 3])


class InterpolationTests(unittest.TestCase):
    def test_interp_step_previous_holds_setpoint_until_next_sample(self):
        src_t = np.array([1.0, 3.0, 5.0])
        src_y = np.array([10.0, 20.0, 30.0])
        dst_t = np.array([0.0, 1.0, 2.5, 3.0, 4.9, 6.0])

        result = interp_step_previous(src_t, src_y, dst_t)

        np.testing.assert_allclose(result[1:], [10.0, 10.0, 20.0, 20.0, 30.0])
        self.assertTrue(np.isnan(result[0]))


class BaroDivergenceTests(unittest.TestCase):
    def _build_ulog(self, baro_alt, gps_alt, t_s=None):
        n = len(baro_alt)
        t_us = (BASE_TS + np.arange(n) * int(1e6)).astype(np.int64) if t_s is None else (BASE_TS + (t_s * 1e6).astype(np.int64))
        air = _make_ds("vehicle_air_data", t_us, baro_alt_meter=np.asarray(baro_alt, float))
        gps_t = (BASE_TS + np.arange(n) * int(1e6)).astype(np.int64)
        gps = _make_ds("vehicle_gps_position", gps_t, altitude_msl_m=np.asarray(gps_alt, float))
        return _make_ulog([air, gps])

    def test_returns_none_when_no_gps(self):
        air = _make_ds("vehicle_air_data", np.array([BASE_TS]), baro_alt_meter=np.array([100.0]))
        ulog = _make_ulog([air])
        self.assertIsNone(compute_baro_gps_divergence(ulog, BASE_TS))

    def test_zero_divergence_when_aligned(self):
        alt = np.array([100.0, 101.0, 102.0, 103.0])
        ulog = self._build_ulog(alt, alt)
        result = compute_baro_gps_divergence(ulog, BASE_TS)
        self.assertIsNotNone(result)
        t, baro, div, offset = result
        np.testing.assert_allclose(div, 0.0, atol=1e-9)

    def test_constant_offset_removed(self):
        gps_alt = np.array([100.0, 101.0, 102.0, 103.0])
        baro_alt = gps_alt + 5.0
        ulog = self._build_ulog(baro_alt, gps_alt)
        result = compute_baro_gps_divergence(ulog, BASE_TS)
        self.assertIsNotNone(result)
        t, baro, div, offset = result
        self.assertAlmostEqual(offset, 5.0, places=6)
        np.testing.assert_allclose(div, 0.0, atol=1e-9)

    def test_propwash_spike_detected(self):
        n = 20
        gps_alt = np.ones(n) * 50.0
        baro_alt = np.ones(n) * 50.0
        baro_alt[10:13] += 5.0  # 5 m spike in baro
        throttle = np.ones(n) * 0.6
        t_us = (BASE_TS + np.arange(n) * int(1e6)).astype(np.int64)
        air = _make_ds("vehicle_air_data", t_us, baro_alt_meter=baro_alt)
        gps = _make_ds("vehicle_gps_position", t_us, altitude_msl_m=gps_alt)
        dist = _make_ds("distance_sensor", t_us, current_distance=gps_alt, signal_quality=np.ones(n, dtype=int) * 100)
        act = _make_ds("actuator_controls_0", t_us, **{"control[3]": throttle})
        ulog = _make_ulog([air, gps, dist, act])
        result = detect_propwash_events(ulog, BASE_TS, divergence_threshold_m=2.0, throttle_threshold=0.3)
        self.assertIsNotNone(result)
        t, pw_mask, div = result
        self.assertTrue(pw_mask[10], "expected propwash at spike start")
        self.assertTrue(pw_mask[12], "expected propwash at spike end")
        self.assertFalse(pw_mask[0], "no propwash before spike")
        self.assertFalse(pw_mask[15], "no propwash after spike")

    def test_no_propwash_below_throttle(self):
        n = 10
        gps_alt = np.ones(n) * 50.0
        baro_alt = gps_alt + 5.0
        throttle = np.ones(n) * 0.1  # below threshold
        t_us = (BASE_TS + np.arange(n) * int(1e6)).astype(np.int64)
        air = _make_ds("vehicle_air_data", t_us, baro_alt_meter=baro_alt)
        gps = _make_ds("vehicle_gps_position", t_us, altitude_msl_m=gps_alt)
        dist = _make_ds("distance_sensor", t_us, current_distance=gps_alt, signal_quality=np.ones(n, dtype=int) * 100)
        act = _make_ds("actuator_controls_0", t_us, **{"control[3]": throttle})
        ulog = _make_ulog([air, gps, dist, act])
        result = detect_propwash_events(ulog, BASE_TS, divergence_threshold_m=2.0, throttle_threshold=0.3)
        self.assertIsNotNone(result)
        t, pw_mask, div = result
        self.assertFalse(pw_mask.any(), "low throttle should suppress propwash flag")


class RangefinderConfigTests(unittest.TestCase):
    def test_no_distance_sensor_reports_absent_topic_and_params(self):
        ulog = _make_ulog([])
        ulog.initial_parameters = {
            "SYS_HAS_NUM_DIST": 0,
            "EKF2_RNG_CTRL": 0,
        }

        config = describe_rangefinder_config(ulog)

        self.assertFalse(config.has_topic)
        self.assertIn("no distance_sensor topic", config.report_summary)
        self.assertIn("no enabled rangefinder driver params found", config.report_summary)
        self.assertIn("SYS_HAS_NUM_DIST=0", config.report_summary)
        self.assertIn("EKF2_RNG_CTRL=0", config.report_summary)

    def test_logged_sf1xx_rangefinder_reports_topic_and_enabled_param(self):
        t_us = (BASE_TS + np.arange(3) * int(1e6)).astype(np.int64)
        dist = _make_ds(
            "distance_sensor",
            t_us,
            type=np.zeros(3, dtype=int),
            orientation=np.ones(3, dtype=int) * 25,
            min_distance=np.ones(3) * 0.2,
            max_distance=np.ones(3) * 120.0,
            current_distance=np.ones(3) * 5.0,
            device_id=np.ones(3, dtype=int) * 123,
        )
        ulog = _make_ulog([dist])
        ulog.initial_parameters = {
            "SENS_EN_SF1XX": 4,
            "SF1XX_MODE": 1,
            "EKF2_RNG_CTRL": 1,
        }

        config = describe_rangefinder_config(ulog)

        self.assertTrue(config.has_topic)
        self.assertIn("distance_sensor logged", config.report_summary)
        self.assertIn("type laser", config.report_summary)
        self.assertIn("downward-facing", config.report_summary)
        self.assertIn("Lightware SF1XX family", config.report_summary)
        self.assertIn("SF1XX_MODE=1", config.report_summary)


class MpcAltModeTests(unittest.TestCase):
    def test_mode_2_reports_terrain_hold_and_related_params(self):
        ulog = _make_ulog([])
        ulog.initial_parameters = {
            "MPC_ALT_MODE": 2,
            "MPC_HOLD_MAX_XY": 0.8,
            "EKF2_HGT_REF": 0,
            "EKF2_RNG_CTRL": 1,
        }

        summary = describe_mpc_alt_mode(ulog)

        self.assertEqual(summary["mpc_alt_mode"], "2")
        self.assertEqual(summary["mpc_alt_mode_label"], "Terrain hold")
        self.assertIn("ground-relative height", summary["mpc_alt_mode_detail"])
        self.assertIn("MPC_HOLD_MAX_XY=0.8", summary["mpc_alt_mode_related_params"])
        self.assertIn("EKF2_HGT_REF=0", summary["mpc_alt_mode_related_params"])


class LoggedParameterTests(unittest.TestCase):
    def test_logged_parameter_rows_only_include_ekf2_and_mpc_prefixes(self):
        ulog = _make_ulog([])
        ulog.initial_parameters = {
            "COM_ARM_WO_GPS": 1,
            "EKF2_HGT_REF": 0,
            "EKF2_RNG_CTRL": 1.0,
            "MPC_ALT_MODE": 2,
            "MPC_HOLD_MAX_XY": 0.8,
        }

        rows = logged_parameter_rows(ulog)

        self.assertEqual([row.name for row in rows], [
            "EKF2_HGT_REF",
            "EKF2_RNG_CTRL",
            "MPC_ALT_MODE",
            "MPC_HOLD_MAX_XY",
        ])
        self.assertEqual([row.prefix for row in rows], ["EKF2", "EKF2", "MPC", "MPC"])
        self.assertEqual([row.value for row in rows], ["0", "1", "2", "0.8"])


class NavStateTests(unittest.TestCase):
    def test_log_start_utc_uses_boot_time_metadata(self):
        ulog = _make_ulog([])
        ulog.msg_info_dict = {"boot_time_utc_us": 1_700_000_000_000_000}

        self.assertEqual(log_start_utc_us(ulog, 42_000_000), 1_700_000_042_000_000)

    def test_log_start_utc_returns_none_without_boot_time_metadata(self):
        self.assertIsNone(log_start_utc_us(_make_ulog([]), BASE_TS))

    def test_system_time_note_reports_missing_boot_time_metadata(self):
        self.assertIn("unavailable", system_time_note(_make_ulog([]), BASE_TS))

    def test_logged_nav_state_rows_use_vehicle_status_enum_values(self):
        t_us = (BASE_TS + np.arange(4) * int(1e6)).astype(np.int64)
        status = _make_ds("vehicle_status", t_us, nav_state=np.array([2, 2, 14, 14]))
        ulog = _make_ulog([status])
        ulog.initial_parameters = {"NAV_STATE": 99}

        rows = logged_nav_state_rows(ulog, BASE_TS)

        self.assertEqual([row.value for row in rows], [2, 14])
        self.assertEqual([row.name for row in rows], ["NAVIGATION_STATE_POSCTL", "NAVIGATION_STATE_OFFBOARD"])
        self.assertEqual([row.spans for row in rows], ["0.000--1.000s", "2.000--3.000s"])
        self.assertAlmostEqual(rows[0].duration_s, 2.0)
        self.assertAlmostEqual(rows[1].duration_s, 2.0)

    def test_nav_state_spans_follow_mode_changes(self):
        t_us = (BASE_TS + np.arange(5) * int(1e6)).astype(np.int64)
        status = _make_ds("vehicle_status", t_us, nav_state=np.array([2, 2, 14, 14, 18]))
        spans = nav_state_spans(_make_ulog([status]), BASE_TS)

        self.assertEqual([(span.start_s, span.end_s, span.value, span.name) for span in spans], [
            (0.0, 2.0, 2, "NAVIGATION_STATE_POSCTL"),
            (2.0, 4.0, 14, "NAVIGATION_STATE_OFFBOARD"),
            (4.0, 4.0, 18, "NAVIGATION_STATE_AUTO_LAND"),
        ])

    def test_gps_disabled_label_reports_sys_has_gps_zero(self):
        ulog = _make_ulog([])
        ulog.initial_parameters = {"SYS_HAS_GPS": 0}

        self.assertEqual(gps_disabled_label(ulog), "SYS_HAS_GPS=0")


class TerrainEstimateTests(unittest.TestCase):
    def test_terrain_summary_uses_local_position_dist_bottom_and_flags(self):
        t_us = (BASE_TS + np.arange(4) * int(1e6)).astype(np.int64)
        lpos = _make_ds(
            "vehicle_local_position",
            t_us,
            dist_bottom=np.array([0.5, 0.6, 0.7, 0.8]),
            dist_bottom_valid=np.array([0, 1, 1, 0]),
            dist_bottom_reset_counter=np.array([2, 2, 3, 3]),
            dist_bottom_sensor_bitfield=np.array([0, 0, 2, 2]),
        )
        flags = _make_ds(
            "estimator_status_flags",
            t_us,
            cs_rng_terrain=np.array([0, 1, 1, 0]),
            cs_opt_flow_terrain=np.zeros(4, dtype=int),
            cs_rng_hgt=np.array([0, 0, 1, 1]),
            cs_rng_kin_consistent=np.array([1, 1, 1, 0]),
        )
        ulog = _make_ulog([lpos, flags])
        ulog.initial_parameters = {"TERRAIN": 99}

        summary = terrain_estimate_summary(ulog, BASE_TS)

        self.assertEqual(summary["terrain_dist_bottom"], "0.50 to 0.80 m")
        self.assertEqual(summary["terrain_dist_bottom_valid"], "1.000--2.000s")
        self.assertEqual(summary["terrain_dist_bottom_reset_counter"], "2--3")
        self.assertEqual(summary["terrain_dist_bottom_sensor_bitfield"], "0, 2")
        self.assertEqual(summary["cs_rng_terrain"], "1.000--2.000s")
        self.assertEqual(summary["cs_opt_flow_terrain"], "none")
        self.assertEqual(summary["cs_rng_hgt"], "2.000--3.000s")
        self.assertEqual(summary["cs_rng_kin_consistent"], "0.000--2.000s")


class TiltTests(unittest.TestCase):
    def test_level_attitude_zero_tilt(self):
        t_us = (BASE_TS + np.arange(5) * int(1e6)).astype(np.int64)
        # Identity quaternion: w=1, x=0, y=0, z=0 → tilt=0
        qw = np.ones(5)
        qx = np.zeros(5)
        qy = np.zeros(5)
        qz = np.zeros(5)
        att = _make_ds("vehicle_attitude", t_us, **{"q[0]": qw, "q[1]": qx, "q[2]": qy, "q[3]": qz})
        ulog = _make_ulog([att])
        result = compute_tilt_deg(ulog, BASE_TS)
        self.assertIsNotNone(result)
        t, tilt = result
        np.testing.assert_allclose(tilt, 0.0, atol=1e-6)

    def test_45deg_tilt(self):
        import math
        t_us = (BASE_TS + np.arange(1) * int(1e6)).astype(np.int64)
        # 45° rotation about Y axis: q = (cos22.5°, 0, sin22.5°, 0)
        angle = math.radians(45)
        qw = np.array([math.cos(angle / 2)])
        qx = np.zeros(1)
        qy = np.array([math.sin(angle / 2)])
        qz = np.zeros(1)
        att = _make_ds("vehicle_attitude", t_us, **{"q[0]": qw, "q[1]": qx, "q[2]": qy, "q[3]": qz})
        ulog = _make_ulog([att])
        result = compute_tilt_deg(ulog, BASE_TS)
        self.assertIsNotNone(result)
        t, tilt = result
        self.assertAlmostEqual(float(tilt[0]), 45.0, places=5)


class EstimatorTableTests(unittest.TestCase):
    def test_active_fusion_rows_only_include_active_fields(self):
        t_us = (BASE_TS + np.arange(4) * int(1e6)).astype(np.int64)
        status = _make_ds("vehicle_status", t_us, arming_state=np.array([1, 2, 2, 1]))
        flags = _make_ds(
            "estimator_status_flags",
            t_us,
            cs_gps=np.array([0, 1, 1, 0]),
            cs_baro_hgt=np.array([0, 1, 0, 0]),
            cs_rng_hgt=np.zeros(4, dtype=int),
        )
        rows = active_fusion_state_rows(_make_ulog([status, flags]), BASE_TS)

        self.assertEqual([row.field for row in rows], ["cs_gps", "cs_baro_hgt"])
        self.assertAlmostEqual(rows[0].duration_s, 2.0)
        self.assertAlmostEqual(rows[0].percent, 100.0)
        self.assertAlmostEqual(rows[1].duration_s, 1.0)
        self.assertAlmostEqual(rows[1].percent, 50.0)

    def test_mag_aiding_row_uses_or_not_sum(self):
        t_us = (BASE_TS + np.arange(5) * int(1e6)).astype(np.int64)
        status = _make_ds("vehicle_status", t_us, arming_state=np.array([2, 2, 2, 2, 2]))
        flags = _make_ds(
            "estimator_status_flags",
            t_us,
            cs_mag_hdg=np.array([1, 1, 0, 0, 0]),
            cs_mag_3d=np.array([0, 1, 1, 0, 0]),
        )

        rows = active_fusion_state_rows(_make_ulog([status, flags]), BASE_TS)

        mag_row = rows[0]
        self.assertEqual(mag_row.field, "cs_mag_hdg_or_3d")
        self.assertAlmostEqual(mag_row.duration_s, 3.0)
        self.assertAlmostEqual(mag_row.percent, 60.0)
        self.assertEqual(mag_row.spans, "0.000--2.000s")

    def test_reset_event_rows_include_attitude_and_heading_deltas(self):
        t_us = (BASE_TS + np.arange(4) * int(1e6)).astype(np.int64)
        attitude = _make_ds(
            "vehicle_attitude",
            t_us,
            quat_reset_counter=np.array([0, 0, 1, 1]),
            **{
                "delta_q_reset[0]": np.array([1.0, 1.0, 0.999048, 1.0]),
                "delta_q_reset[1]": np.zeros(4),
                "delta_q_reset[2]": np.zeros(4),
                "delta_q_reset[3]": np.array([0.0, 0.0, 0.043619, 0.0]),
            },
        )
        local_position = _make_ds(
            "vehicle_local_position",
            t_us,
            heading_reset_counter=np.array([0, 0, 1, 1]),
            delta_heading=np.array([0.0, 0.0, np.deg2rad(5.0), 0.0]),
        )

        rows = reset_event_rows(_make_ulog([attitude, local_position]), BASE_TS)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].source, "vehicle_attitude.quat_reset_counter")
        self.assertAlmostEqual(rows[0].time_s, 2.0)
        self.assertIn("5.00 deg yaw", rows[0].delta)
        self.assertEqual(rows[1].source, "vehicle_local_position.heading_reset_counter")
        self.assertIn("5.00 deg heading", rows[1].delta)

    def test_exception_rows_only_include_occurred_unhealthy_statuses(self):
        t_us = (BASE_TS + np.arange(5) * int(1e6)).astype(np.int64)
        status = _make_ds("vehicle_status", t_us, arming_state=np.array([1, 2, 2, 2, 1]))
        estimator_status = _make_ds(
            "estimator_status",
            t_us,
            filter_fault_flags=np.zeros(5, dtype=int),
            innovation_check_flags=np.zeros(5, dtype=int),
            timeout_flags=np.zeros(5, dtype=int),
            health_flags=np.zeros(5, dtype=int),
            gps_check_fail_flags=np.array([0, 0, 1, 0, 0]),
        )
        gps_status = _make_ds(
            "estimator_gps_status",
            t_us,
            check_fail_max_vert_spd_err=np.array([0, 0, 1, 1, 0]),
            check_fail_gps_fix=np.zeros(5, dtype=int),
        )

        rows = estimator_exception_rows(_make_ulog([status, estimator_status, gps_status]), BASE_TS)

        fields = [row.field for row in rows]
        self.assertIn("gps_check_fail_flags", fields)
        self.assertIn("check_fail_max_vert_spd_err", fields)
        self.assertNotIn("filter_fault_flags", fields)
        row = next(row for row in rows if row.field == "check_fail_max_vert_spd_err")
        self.assertAlmostEqual(row.duration_s, 2.0)
        self.assertEqual(row.meaning, "maximum allowed vertical velocity discrepancy fail")


if __name__ == "__main__":
    unittest.main()
