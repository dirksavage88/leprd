import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import numpy as np

from ekf2_playback_review.analysis import (
    active_bit_indices,
    active_fusion_state_rows,
    actuator_saturation_label,
    bool_spans,
    boolean_time_percentage_label,
    compute_baro_gps_divergence,
    compute_tilt_deg,
    control_status_flag_groups,
    describe_mpc_alt_mode,
    describe_rangefinder_config,
    detect_propwash_events,
    estimator_exception_rows,
    ev_test_ratio_panel_specs,
    exceedance_start_times,
    format_spans,
    front_rear_motor_indices,
    get_actuator_saturation_series,
    get_esc_rpm_series,
    get_motor_output_set,
    gps_disabled_label,
    innovation_check_flag_group_values,
    interp_step_previous,
    local_minus_ev_position_delta,
    log_start_utc_us,
    logged_nav_state_rows,
    logged_parameter_rows,
    measurement_count_s,
    measurement_interval_s,
    measurement_latency_s,
    nav_state_spans,
    parameter_bitmask_rows,
    parameter_enum_rows,
    plot_accel_bias_adjustment,
    plot_forward_lurch_diagnostics,
    plot_gyro_bias_adjustment,
    plot_inertial_dead_reckoning,
    rangefinder_test_ratio_series,
    reset_event_rows,
    sensor_latency_series,
    sensor_status_rows,
    source_log_latex_line,
    status_mask_rows,
    system_time_note,
    terrain_estimate_summary,
    terrain_hagl_state_series,
    time_since_last_fuse_s,
)
from ekf2_playback_review.config import LogSpec, ReviewConfig


def _make_ds(name: str, timestamp: np.ndarray, multi_id: int = 0, **fields) -> MagicMock:
    ds = MagicMock()
    ds.name = name
    ds.multi_id = multi_id
    ds.data = {"timestamp": timestamp, **fields}
    return ds


def _make_ulog(datasets: list) -> MagicMock:
    ulog = MagicMock()
    ulog.data_list = datasets
    ulog.initial_parameters = {}
    ulog.msg_info_dict = {}
    return ulog


BASE_TS = 1_000_000


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

    def test_boolean_time_percentage_label_uses_sample_durations(self):
        t = np.array([0.0, 1.0, 3.0, 6.0])
        values = np.array([False, True, False, True])

        self.assertEqual(boolean_time_percentage_label(t, values), "50.0% enabled time")


class ExceedanceTests(unittest.TestCase):
    def test_exceedance_start_times_only_reports_upward_crossings(self):
        t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        values = np.array([0.2, 1.2, 1.4, 0.9, 1.1])

        np.testing.assert_allclose(exceedance_start_times(t, values, threshold=1.0), [1.0, 4.0])


class InnovationCheckFlagTests(unittest.TestCase):
    def test_decodes_single_and_grouped_innovation_check_bits(self):
        raw = np.array([0, 512, 1024, 2048, 3072], dtype=int)

        np.testing.assert_array_equal(
            innovation_check_flag_group_values(raw, 9, 0x1), [0, 1, 0, 0, 0]
        )
        np.testing.assert_array_equal(
            innovation_check_flag_group_values(raw, 10, 0x3), [0, 0, 1, 2, 3]
        )


class FusionTimingTests(unittest.TestCase):
    def test_time_since_last_fuse_uses_topic_timestamp_and_masks_invalid_values(self):
        t_us = (BASE_TS + np.arange(4) * int(1e6)).astype(np.uint64)
        ds = _make_ds(
            "estimator_aid_src_optical_flow",
            t_us,
            time_last_fuse=np.array(
                [
                    0,
                    BASE_TS + 500_000,
                    BASE_TS + 1_000_000,
                    BASE_TS + 4_000_000,
                ],
                dtype=np.uint64,
            ),
        )

        result = time_since_last_fuse_s(ds, BASE_TS)

        self.assertIsNotNone(result)
        rel_t, delta_s = result
        np.testing.assert_allclose(rel_t, [0.0, 1.0, 2.0, 3.0])
        self.assertTrue(np.isnan(delta_s[0]))
        self.assertAlmostEqual(delta_s[1], 0.5)
        self.assertAlmostEqual(delta_s[2], 1.0)
        self.assertTrue(np.isnan(delta_s[3]))

    def test_measurement_interval_uses_sample_timestamp_and_masks_repeats(self):
        t_us = (BASE_TS + np.arange(5) * int(1e6)).astype(np.uint64)
        ds = _make_ds(
            "vehicle_optical_flow",
            t_us,
            timestamp_sample=np.array(
                [
                    BASE_TS,
                    BASE_TS + 20_000,
                    BASE_TS + 20_000,
                    BASE_TS + 50_000,
                    BASE_TS + 90_000,
                ],
                dtype=np.uint64,
            ),
        )

        result = measurement_interval_s(ds, BASE_TS)

        self.assertIsNotNone(result)
        rel_t, interval_s, source_field = result
        self.assertEqual(source_field, "timestamp_sample")
        np.testing.assert_allclose(rel_t, [0.0, 1.0, 2.0, 3.0, 4.0])
        self.assertTrue(np.isnan(interval_s[0]))
        self.assertAlmostEqual(interval_s[1], 0.02)
        self.assertTrue(np.isnan(interval_s[2]))
        self.assertAlmostEqual(interval_s[3], 0.03)
        self.assertAlmostEqual(interval_s[4], 0.04)

    def test_measurement_interval_falls_back_to_topic_timestamp(self):
        t_us = np.array(
            [
                BASE_TS,
                BASE_TS + 100_000,
                BASE_TS + 250_000,
            ],
            dtype=np.uint64,
        )
        ds = _make_ds("vehicle_optical_flow", t_us)

        result = measurement_interval_s(ds, BASE_TS)

        self.assertIsNotNone(result)
        _rel_t, interval_s, source_field = result
        self.assertEqual(source_field, "timestamp")
        self.assertTrue(np.isnan(interval_s[0]))
        self.assertAlmostEqual(interval_s[1], 0.1)
        self.assertAlmostEqual(interval_s[2], 0.15)

    def test_measurement_latency_uses_signed_timestamp_sample_delta(self):
        t_us = np.array(
            [
                BASE_TS,
                BASE_TS + 100_000,
                BASE_TS + 200_000,
                BASE_TS + 300_000,
            ],
            dtype=np.uint64,
        )
        ds = _make_ds(
            "vehicle_imu",
            t_us,
            timestamp_sample=np.array(
                [
                    0,
                    BASE_TS + 80_000,
                    BASE_TS + 250_000,
                    BASE_TS + 300_000,
                ],
                dtype=np.uint64,
            ),
        )

        result = measurement_latency_s(ds, BASE_TS)

        self.assertIsNotNone(result)
        rel_t, latency_s, source_field = result
        self.assertEqual(source_field, "timestamp_sample")
        np.testing.assert_allclose(rel_t, [0.0, 0.1, 0.2, 0.3])
        self.assertTrue(np.isnan(latency_s[0]))
        self.assertAlmostEqual(latency_s[1], 0.02)
        self.assertAlmostEqual(latency_s[2], -0.05)
        self.assertAlmostEqual(latency_s[3], 0.0)

    def test_measurement_latency_returns_none_without_sample_timestamp(self):
        ds = _make_ds("vehicle_imu", np.array([BASE_TS], dtype=np.uint64))

        self.assertIsNone(measurement_latency_s(ds, BASE_TS))

    def test_sensor_latency_series_separates_logged_sensor_topics(self):
        imu0 = _make_ds(
            "vehicle_imu",
            np.array([BASE_TS + 10_000], dtype=np.uint64),
            multi_id=0,
            timestamp_sample=np.array([BASE_TS], dtype=np.uint64),
        )
        imu1 = _make_ds(
            "vehicle_imu",
            np.array([BASE_TS + 30_000], dtype=np.uint64),
            multi_id=1,
            timestamp_sample=np.array([BASE_TS + 10_000], dtype=np.uint64),
        )
        flow = _make_ds(
            "vehicle_optical_flow",
            np.array([BASE_TS + 25_000], dtype=np.uint64),
            timestamp_sample=np.array([BASE_TS + 20_000], dtype=np.uint64),
        )
        no_sample_time = _make_ds(
            "vehicle_air_data",
            np.array([BASE_TS + 25_000], dtype=np.uint64),
        )
        ulog = _make_ulog([imu1, no_sample_time, flow, imu0])

        series = sensor_latency_series(ulog, BASE_TS)

        self.assertEqual(
            [item.label for item in series],
            [
                "IMU[0] (vehicle_imu.timestamp - timestamp_sample)",
                "IMU[1] (vehicle_imu.timestamp - timestamp_sample)",
                "Optical flow (vehicle_optical_flow.timestamp - timestamp_sample)",
            ],
        )
        np.testing.assert_allclose([item.values[0] for item in series], [10.0, 20.0, 5.0])

    def test_measurement_count_returns_cumulative_count(self):
        t_us = np.array(
            [
                BASE_TS,
                BASE_TS + 50_000,
                BASE_TS + 200_000,
            ],
            dtype=np.uint64,
        )
        ds = _make_ds("vehicle_optical_flow", t_us)

        result = measurement_count_s(ds, BASE_TS)

        self.assertIsNotNone(result)
        rel_t, counts = result
        np.testing.assert_allclose(rel_t, [0.0, 0.05, 0.2])
        np.testing.assert_array_equal(counts, [1, 2, 3])

    def test_measurement_count_returns_none_without_timestamp(self):
        ds = MagicMock()
        ds.data = {}

        self.assertIsNone(measurement_count_s(ds, BASE_TS))


class InterpolationTests(unittest.TestCase):
    def test_interp_step_previous_holds_setpoint_until_next_sample(self):
        src_t = np.array([1.0, 3.0, 5.0])
        src_y = np.array([10.0, 20.0, 30.0])
        dst_t = np.array([0.0, 1.0, 2.5, 3.0, 4.9, 6.0])

        result = interp_step_previous(src_t, src_y, dst_t)

        np.testing.assert_allclose(result[1:], [10.0, 10.0, 20.0, 20.0, 30.0])
        self.assertTrue(np.isnan(result[0]))

    def test_local_minus_ev_position_delta_interpolates_local_to_ev_samples(self):
        odom = _make_ds(
            "vehicle_visual_odometry",
            (BASE_TS + np.array([0, 1_000_000, 2_000_000], dtype=np.uint64)),
            **{"position[0]": np.array([0.0, 2.0, 4.0])},
        )
        local_position = _make_ds(
            "vehicle_local_position",
            (BASE_TS + np.array([0, 2_000_000], dtype=np.uint64)),
            x=np.array([1.0, 5.0]),
        )

        result = local_minus_ev_position_delta(odom, local_position, BASE_TS, "position[0]", "x")

        self.assertIsNotNone(result)
        rel_t, delta = result
        np.testing.assert_allclose(rel_t, [0.0, 1.0, 2.0])
        np.testing.assert_allclose(delta, [1.0, 1.0, 1.0])

    def test_ev_test_ratio_panel_specs_split_position_height_and_yaw(self):
        t_us = (BASE_TS + np.arange(2) * int(1e6)).astype(np.int64)
        ev_pos = _make_ds(
            "estimator_aid_src_ev_pos",
            t_us,
            **{
                "test_ratio[0]": np.array([0.1, 0.2]),
                "test_ratio[1]": np.array([0.3, 0.4]),
            },
        )
        ev_hgt = _make_ds("estimator_aid_src_ev_hgt", t_us, test_ratio=np.array([0.5, 0.6]))
        ev_yaw = _make_ds("estimator_aid_src_ev_yaw", t_us, test_ratio=np.array([0.7, 0.8]))
        ulog = _make_ulog([ev_pos, ev_hgt, ev_yaw])
        ulog.initial_parameters = {
            "EKF2_EVP_GATE": 5.0,
            "EKF2_HDG_GATE": 2.6,
        }

        specs = ev_test_ratio_panel_specs(ulog)

        self.assertEqual(
            [(topic, fields) for topic, _title, fields, _prefix in specs],
            [
                ("estimator_aid_src_ev_pos", ["test_ratio[0]", "test_ratio[1]"]),
                ("estimator_aid_src_ev_hgt", ["test_ratio"]),
                ("estimator_aid_src_ev_yaw", ["test_ratio"]),
            ],
        )
        self.assertIn("EKF2_EVP_GATE=5", specs[0][1])
        self.assertIn("EKF2_EVP_GATE=5", specs[1][1])
        self.assertIn("EKF2_HDG_GATE=2.6", specs[2][1])


class RangefinderRatioTests(unittest.TestCase):
    def test_rangefinder_test_ratio_prefers_direct_aid_source_and_separates_hagl_rate(self):
        t_us = (BASE_TS + np.arange(2) * int(1e6)).astype(np.int64)
        aid = _make_ds(
            "estimator_aid_src_rng_hgt",
            t_us,
            test_ratio=np.array([0.1, 0.2]),
        )
        aggregate = _make_ds(
            "estimator_innovation_test_ratios",
            t_us,
            rng_vpos=np.array([0.3, 0.4]),
            hagl=np.array([0.5, 0.6]),
            hagl_rate=np.array([0.7, 0.8]),
        )

        range_height, hagl_rate = rangefinder_test_ratio_series(
            _make_ulog([aid, aggregate]), BASE_TS
        )

        self.assertIsNotNone(range_height)
        self.assertIsNotNone(hagl_rate)
        self.assertEqual(range_height.source, "estimator_aid_src_rng_hgt.test_ratio")
        np.testing.assert_allclose(range_height.values, [0.1, 0.2])
        self.assertEqual(hagl_rate.source, "estimator_innovation_test_ratios.hagl_rate")
        np.testing.assert_allclose(hagl_rate.values, [0.7, 0.8])

    def test_rangefinder_test_ratio_falls_back_to_aggregate_hagl(self):
        t_us = (BASE_TS + np.arange(2) * int(1e6)).astype(np.int64)
        aggregate = _make_ds(
            "estimator_innovation_test_ratios",
            t_us,
            rng_vpos=np.array([0.3, 0.4]),
            hagl=np.array([0.5, 0.6]),
        )

        range_height, hagl_rate = rangefinder_test_ratio_series(_make_ulog([aggregate]), BASE_TS)

        self.assertIsNotNone(range_height)
        self.assertIsNone(hagl_rate)
        self.assertEqual(range_height.source, "estimator_innovation_test_ratios.hagl")
        np.testing.assert_allclose(range_height.values, [0.5, 0.6])


class BaroDivergenceTests(unittest.TestCase):
    def _build_ulog(self, baro_alt, gps_alt, t_s=None):
        n = len(baro_alt)
        t_us = (
            (BASE_TS + np.arange(n) * int(1e6)).astype(np.int64)
            if t_s is None
            else (BASE_TS + (t_s * 1e6).astype(np.int64))
        )
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
        _t, _baro, div, _offset = result
        np.testing.assert_allclose(div, 0.0, atol=1e-9)

    def test_constant_offset_removed(self):
        gps_alt = np.array([100.0, 101.0, 102.0, 103.0])
        baro_alt = gps_alt + 5.0
        ulog = self._build_ulog(baro_alt, gps_alt)
        result = compute_baro_gps_divergence(ulog, BASE_TS)
        self.assertIsNotNone(result)
        _t, _baro, div, offset = result
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
        dist = _make_ds(
            "distance_sensor",
            t_us,
            current_distance=gps_alt,
            signal_quality=np.ones(n, dtype=int) * 100,
        )
        act = _make_ds("actuator_controls_0", t_us, **{"control[3]": throttle})
        ulog = _make_ulog([air, gps, dist, act])
        result = detect_propwash_events(
            ulog, BASE_TS, divergence_threshold_m=2.0, throttle_threshold=0.3
        )
        self.assertIsNotNone(result)
        _t, pw_mask, _div = result
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
        dist = _make_ds(
            "distance_sensor",
            t_us,
            current_distance=gps_alt,
            signal_quality=np.ones(n, dtype=int) * 100,
        )
        act = _make_ds("actuator_controls_0", t_us, **{"control[3]": throttle})
        ulog = _make_ulog([air, gps, dist, act])
        result = detect_propwash_events(
            ulog, BASE_TS, divergence_threshold_m=2.0, throttle_threshold=0.3
        )
        self.assertIsNotNone(result)
        _t, pw_mask, _div = result
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
    def test_logged_parameter_rows_only_include_reported_parameter_prefixes(self):
        ulog = _make_ulog([])
        ulog.initial_parameters = {
            "COM_ARM_WO_GPS": 1,
            "EKF2_HGT_REF": 0,
            "EKF2_RNG_CTRL": 1.0,
            "MPC_ALT_MODE": 2,
            "MPC_HOLD_MAX_XY": 0.8,
            "SDLOG_MODE": 1,
            "SDLOG_PROFILE": 131,
            "SENS_FLOW_ROT": 6,
            "SENS_FLOW_SCALE": 1.0,
        }

        rows = logged_parameter_rows(ulog)

        self.assertEqual(
            [row.name for row in rows],
            [
                "EKF2_HGT_REF",
                "EKF2_RNG_CTRL",
                "MPC_ALT_MODE",
                "MPC_HOLD_MAX_XY",
                "SDLOG_MODE",
                "SDLOG_PROFILE",
                "SENS_FLOW_ROT",
                "SENS_FLOW_SCALE",
            ],
        )
        self.assertEqual(
            [row.prefix for row in rows],
            ["EKF2", "EKF2", "MPC", "MPC", "SDLOG", "SDLOG", "SENS", "SENS"],
        )
        self.assertEqual(
            [row.value for row in rows],
            ["0", "1", "2", "0.8", "1", "131", "6", "1"],
        )


class SensorStatusTests(unittest.TestCase):
    def test_sensor_status_rows_include_expected_sys_has_and_logged_ctrl_params(self):
        ulog = _make_ulog([])
        ulog.initial_parameters = {
            "SYS_HAS_MAG": 1,
            "SYS_HAS_BARO": 0,
            "HAS_ACC": 1,
            "HAS_GYRO": 1,
            "EKF2_RNG_CTRL": 2,
            "EKF2_OF_CTRL": 0,
            "EKF2_FOO_CTRL": 7,
        }

        rows = sensor_status_rows(ulog)
        by_name = {row.name: row for row in rows}

        self.assertEqual(by_name["SYS_HAS_MAG"].sensor, "Magnetometer")
        self.assertEqual(by_name["SYS_HAS_MAG"].status, "ENABLED")
        self.assertEqual(by_name["SYS_HAS_BARO"].status, "DISABLED")
        self.assertEqual(by_name["EKF2_RNG_CTRL"].sensor, "Range finder")
        self.assertEqual(by_name["EKF2_RNG_CTRL"].status, "ENABLED")
        self.assertEqual(by_name["EKF2_OF_CTRL"].status, "DISABLED")
        self.assertEqual(by_name["EKF2_FOO_CTRL"].sensor, "Foo")
        self.assertEqual(by_name["SYS_HAS_ACC"].value, "not logged")
        self.assertNotIn("HAS_ACC", by_name)
        self.assertNotIn("HAS_GYRO", by_name)


class ParameterBitmaskTests(unittest.TestCase):
    def test_decodes_mag_and_gps_check_bits_from_initial_parameters(self):
        ulog = _make_ulog([])
        ulog.initial_parameters = {
            "EKF2_IMU_CTRL": 3,
            "EKF2_GPS_CTRL": 7,
            "EKF2_MAG_CHECK": 1,
            "EKF2_GPS_CHECK": 245,
        }

        rows = parameter_bitmask_rows(ulog)
        by_key = {(row.parameter, row.bit): row for row in rows}

        self.assertEqual(by_key[("EKF2_IMU_CTRL", 0)].status, "ENABLED")
        self.assertEqual(by_key[("EKF2_IMU_CTRL", 1)].status, "ENABLED")
        self.assertEqual(by_key[("EKF2_IMU_CTRL", 2)].status, "DISABLED")
        self.assertEqual(by_key[("EKF2_GPS_CTRL", 0)].status, "ENABLED")
        self.assertEqual(by_key[("EKF2_GPS_CTRL", 1)].status, "ENABLED")
        self.assertEqual(by_key[("EKF2_GPS_CTRL", 2)].status, "ENABLED")
        self.assertEqual(by_key[("EKF2_GPS_CTRL", 3)].status, "DISABLED")
        self.assertEqual(by_key[("EKF2_MAG_CHECK", 0)].status, "ENABLED")
        self.assertEqual(by_key[("EKF2_MAG_CHECK", 1)].status, "DISABLED")
        self.assertEqual(by_key[("EKF2_MAG_CHECK", 2)].status, "DISABLED")
        self.assertEqual(by_key[("EKF2_GPS_CHECK", 0)].status, "ENABLED")
        self.assertEqual(by_key[("EKF2_GPS_CHECK", 1)].status, "DISABLED")
        self.assertEqual(by_key[("EKF2_GPS_CHECK", 2)].status, "ENABLED")
        self.assertEqual(by_key[("EKF2_GPS_CHECK", 4)].status, "ENABLED")
        self.assertEqual(by_key[("EKF2_GPS_CHECK", 5)].status, "ENABLED")
        self.assertEqual(by_key[("EKF2_GPS_CHECK", 6)].status, "ENABLED")
        self.assertEqual(by_key[("EKF2_GPS_CHECK", 7)].status, "ENABLED")
        self.assertEqual(by_key[("EKF2_GPS_CHECK", 8)].status, "DISABLED")
        self.assertIn("Horizontal position error", by_key[("EKF2_GPS_CHECK", 2)].meaning)

    def test_parameter_bitmask_rows_mark_missing_values(self):
        rows = parameter_bitmask_rows(_make_ulog([]))

        self.assertTrue(rows)
        self.assertTrue(all(row.raw_value == "not logged" for row in rows))
        self.assertTrue(all(row.status == "NOT LOGGED" for row in rows))

    def test_parameter_enum_rows_decode_rangefinder_control(self):
        ulog = _make_ulog([])
        ulog.initial_parameters = {"EKF2_RNG_CTRL": 2}

        rows = parameter_enum_rows(ulog)
        by_value = {row.value: row for row in rows if row.parameter == "EKF2_RNG_CTRL"}

        self.assertFalse(by_value[0].selected)
        self.assertEqual(by_value[0].status, "not selected")
        self.assertTrue(by_value[2].selected)
        self.assertEqual(by_value[2].status, "ENABLED")


class StatusMaskTests(unittest.TestCase):
    def test_active_bit_indices_reports_observed_mask_bits(self):
        values = np.array([0, 1, 4, 5, 8], dtype=int)

        self.assertEqual(active_bit_indices(values), [0, 2, 3])

    def test_status_mask_rows_list_raw_values_and_active_bits(self):
        t_us = (BASE_TS + np.arange(3) * int(1e6)).astype(np.int64)
        status = _make_ds(
            "estimator_status",
            t_us,
            control_mode_flags=np.array([0, 1, 5], dtype=int),
            solution_status_flags=np.array([2, 2, 6], dtype=int),
        )
        rows = status_mask_rows(_make_ulog([status]))
        by_field = {row.field: row for row in rows}

        self.assertEqual(by_field["control_mode_flags"].observed_values, "0, 1, 5")
        self.assertEqual(by_field["control_mode_flags"].active_bits, "bit 0, bit 2")
        self.assertEqual(by_field["solution_status_flags"].observed_values, "2, 6")
        self.assertEqual(by_field["solution_status_flags"].active_bits, "bit 1, bit 2")

    def test_control_status_flag_groups_include_all_logged_cs_fields_once(self):
        t_us = (BASE_TS + np.arange(2) * int(1e6)).astype(np.int64)
        flags = _make_ds(
            "estimator_status_flags",
            t_us,
            cs_yaw_align=np.ones(2, dtype=int),
            cs_rng_hgt=np.array([0, 1], dtype=int),
            cs_ev_pos=np.array([1, 1], dtype=int),
            cs_unexpected_new_flag=np.array([0, 1], dtype=int),
            reject_hagl=np.array([0, 1], dtype=int),
        )

        groups = control_status_flag_groups(flags)
        grouped_fields = [field for _title, fields in groups for field in fields]

        self.assertEqual(
            sorted(grouped_fields),
            ["cs_ev_pos", "cs_rng_hgt", "cs_unexpected_new_flag", "cs_yaw_align"],
        )
        self.assertEqual(len(grouped_fields), len(set(grouped_fields)))
        self.assertEqual(groups[-1], ("Other control status fields", ["cs_unexpected_new_flag"]))


class NavStateTests(unittest.TestCase):
    def test_log_start_utc_uses_boot_time_metadata(self):
        ulog = _make_ulog([])
        ulog.msg_info_dict = {"boot_time_utc_us": 1_700_000_000_000_000}

        self.assertEqual(log_start_utc_us(ulog, 42_000_000), 1_700_000_042_000_000)

    def test_log_start_utc_returns_none_without_boot_time_metadata(self):
        self.assertIsNone(log_start_utc_us(_make_ulog([]), BASE_TS))

    def test_system_time_note_is_empty_without_boot_time_metadata(self):
        self.assertEqual(system_time_note(_make_ulog([]), BASE_TS), "")

    def test_logged_nav_state_rows_use_vehicle_status_enum_values(self):
        t_us = (BASE_TS + np.arange(4) * int(1e6)).astype(np.int64)
        status = _make_ds("vehicle_status", t_us, nav_state=np.array([2, 2, 14, 14]))
        ulog = _make_ulog([status])
        ulog.initial_parameters = {"NAV_STATE": 99}

        rows = logged_nav_state_rows(ulog, BASE_TS)

        self.assertEqual([row.value for row in rows], [2, 14])
        self.assertEqual(
            [row.name for row in rows], ["NAVIGATION_STATE_POSCTL", "NAVIGATION_STATE_OFFBOARD"]
        )
        self.assertEqual([row.spans for row in rows], ["0.000--1.000s", "2.000--3.000s"])
        self.assertAlmostEqual(rows[0].duration_s, 2.0)
        self.assertAlmostEqual(rows[1].duration_s, 2.0)

    def test_nav_state_spans_follow_mode_changes(self):
        t_us = (BASE_TS + np.arange(5) * int(1e6)).astype(np.int64)
        status = _make_ds("vehicle_status", t_us, nav_state=np.array([2, 2, 14, 14, 18]))
        spans = nav_state_spans(_make_ulog([status]), BASE_TS)

        self.assertEqual(
            [(span.start_s, span.end_s, span.value, span.name) for span in spans],
            [
                (0.0, 2.0, 2, "NAVIGATION_STATE_POSCTL"),
                (2.0, 4.0, 14, "NAVIGATION_STATE_OFFBOARD"),
                (4.0, 4.0, 18, "NAVIGATION_STATE_AUTO_LAND"),
            ],
        )

    def test_gps_disabled_label_reports_sys_has_gps_zero(self):
        ulog = _make_ulog([])
        ulog.initial_parameters = {"SYS_HAS_GPS": 0}

        self.assertEqual(gps_disabled_label(ulog), "SYS_HAS_GPS=0")


class LatexReportTests(unittest.TestCase):
    def test_source_log_latex_line_includes_full_path(self):
        log_path = (
            "/Users/gonk/analysis/Brecourt/August 11 2026 (VIO In Loop) "
            "/Last Logs (No Vel)/sess129_log115.ulg"
        )
        config = ReviewConfig(
            title="PX4 EKF2 Review",
            output_dir=Path("/tmp/reports"),
            log=LogSpec(key="flight", title="sess129_log115.ulg", path=Path(log_path)),
        )

        line = source_log_latex_line(config)

        self.assertIn("Source ULog", line)
        self.assertIn("Users", line)
        self.assertIn("Brecourt", line)
        self.assertIn("sess129\\_\\allowbreak{}log115.ulg", line)


class TerrainEstimateTests(unittest.TestCase):
    def test_terrain_hagl_state_series_includes_local_and_global_derivations(self):
        t_us = (BASE_TS + np.arange(3) * int(1e6)).astype(np.int64)
        lpos = _make_ds(
            "vehicle_local_position",
            t_us,
            z=np.array([-4.0, -3.5, -3.0]),
            dist_bottom=np.array([4.0, 4.2, 4.4]),
        )
        gpos = _make_ds(
            "vehicle_global_position",
            t_us,
            alt=np.array([100.0, 101.0, 102.0]),
            alt_valid=np.array([1, 1, 1]),
            terrain_alt=np.array([96.0, 96.5, 97.0]),
            terrain_alt_valid=np.array([1, 1, 1]),
        )
        estimator_states = _make_ds(
            "estimator_states",
            t_us,
            **{"states[24]": np.array([-96.0, -96.5, -97.0])},
        )

        series = terrain_hagl_state_series(_make_ulog([lpos, gpos, estimator_states]), BASE_TS)
        by_label = {item.label: item.values for item in series}

        np.testing.assert_allclose(
            by_label["terrain state: estimator_states.states[24] (_state.terrain)"],
            [-96.0, -96.5, -97.0],
        )
        np.testing.assert_allclose(
            by_label["HAGL: vehicle_local_position.dist_bottom"], [4.0, 4.2, 4.4]
        )
        np.testing.assert_allclose(by_label["terrain local NED: z + dist_bottom"], [0.0, 0.7, 1.4])
        np.testing.assert_allclose(
            by_label["terrain global NED rel start: -delta terrain_alt"],
            [0.0, -0.5, -1.0],
        )
        np.testing.assert_allclose(
            by_label["HAGL: vehicle_global_position.alt - terrain_alt"], [4.0, 4.5, 5.0]
        )

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
        _t, tilt = result
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
        _t, tilt = result
        self.assertAlmostEqual(float(tilt[0]), 45.0, places=5)


class ActuatorExtractionTests(unittest.TestCase):
    def test_motor_output_set_prefers_actuator_motors(self):
        t_us = (BASE_TS + np.arange(3) * int(1e6)).astype(np.int64)
        motors = _make_ds(
            "actuator_motors",
            t_us,
            **{
                "control[0]": np.array([0.1, 0.2, 0.3]),
                "control[1]": np.array([0.2, 0.3, 0.4]),
                "control[2]": np.array([np.nan, np.nan, np.nan]),
            },
        )
        outputs = _make_ds(
            "actuator_outputs",
            t_us,
            noutputs=np.array([4, 4, 4]),
            **{"output[0]": np.array([1000.0, 1100.0, 1200.0])},
        )

        result = get_motor_output_set(_make_ulog([outputs, motors]), BASE_TS)

        self.assertIsNotNone(result)
        self.assertEqual(result.source, "actuator_motors.control[]")
        self.assertEqual([series.label for series in result.series], ["Motor 1", "Motor 2"])

    def test_motor_output_set_falls_back_to_varying_actuator_outputs(self):
        t_us = (BASE_TS + np.arange(3) * int(1e6)).astype(np.int64)
        outputs = _make_ds(
            "actuator_outputs",
            t_us,
            noutputs=np.array([3, 3, 3]),
            **{
                "output[0]": np.array([1000.0, 1000.0, 1000.0]),
                "output[1]": np.array([1000.0, 1100.0, 1200.0]),
                "output[2]": np.array([1200.0, 1100.0, 1000.0]),
            },
        )

        result = get_motor_output_set(_make_ulog([outputs]), BASE_TS)

        self.assertIsNotNone(result)
        self.assertEqual(result.source, "actuator_outputs.output[]")
        self.assertEqual([series.label for series in result.series], ["Output 1", "Output 2"])

    def test_esc_rpm_series_uses_esc_count_and_includes_zero_channels(self):
        t_us = (BASE_TS + np.arange(3) * int(1e6)).astype(np.int64)
        esc = _make_ds(
            "esc_status",
            t_us,
            esc_count=np.array([3, 3, 3]),
            **{
                "esc[0].esc_rpm": np.array([0.0, 0.0, 0.0]),
                "esc[1].esc_rpm": np.array([100.0, 120.0, 110.0]),
                "esc[2].esc_rpm": np.array([200.0, 210.0, 220.0]),
            },
        )

        result = get_esc_rpm_series(_make_ulog([esc]), BASE_TS)

        self.assertEqual(
            [series.label for series in result],
            ["ESC 1 RPM", "ESC 2 RPM", "ESC 3 RPM"],
        )

    def test_actuator_saturation_series_breaks_out_fields(self):
        t_us = (BASE_TS + np.arange(3) * int(1e6)).astype(np.int64)
        allocator = _make_ds(
            "control_allocator_status",
            t_us,
            **{
                "actuator_saturation[0]": np.array([0, 1, 2], dtype=np.int8),
                "actuator_saturation[1]": np.array([0, -1, -2], dtype=np.int8),
            },
        )

        result = get_actuator_saturation_series(_make_ulog([allocator]), BASE_TS)

        self.assertEqual(
            [series.label for series in result],
            ["actuator_saturation[0]", "actuator_saturation[1]"],
        )
        np.testing.assert_allclose(result[0].t, [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(result[0].values, [0, 1, 2])
        self.assertEqual(actuator_saturation_label(0), "ACTUATOR_SATURATION_OK")
        self.assertEqual(actuator_saturation_label(2), "ACTUATOR_SATURATION_UPPER")
        self.assertEqual(actuator_saturation_label(-2), "ACTUATOR_SATURATION_LOWER")

    def test_actuator_saturation_series_labels_multi_instance_status_topics(self):
        t_us = (BASE_TS + np.arange(2) * int(1e6)).astype(np.int64)
        allocator_0 = _make_ds(
            "control_allocator_status",
            t_us,
            multi_id=0,
            **{"actuator_saturation[0]": np.array([0, 1], dtype=np.int8)},
        )
        allocator_1 = _make_ds(
            "control_allocator_status",
            t_us,
            multi_id=1,
            **{"actuator_saturation[0]": np.array([0, -1], dtype=np.int8)},
        )

        result = get_actuator_saturation_series(_make_ulog([allocator_1, allocator_0]), BASE_TS)

        self.assertEqual(
            [series.label for series in result],
            [
                "control_allocator_status[0].actuator_saturation[0]",
                "control_allocator_status[1].actuator_saturation[0]",
            ],
        )


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

    def test_reset_event_rows_include_local_position_z_and_hagl_deltas(self):
        t_us = (BASE_TS + np.arange(4) * int(1e6)).astype(np.int64)
        local_position = _make_ds(
            "vehicle_local_position",
            t_us,
            z_reset_counter=np.array([2, 2, 3, 3]),
            delta_z=np.array([0.0, 0.0, -0.25, 0.0]),
            dist_bottom_reset_counter=np.array([4, 4, 4, 5]),
            delta_dist_bottom=np.array([0.0, 0.0, 0.0, 0.12]),
        )

        rows = reset_event_rows(_make_ulog([local_position]), BASE_TS)

        self.assertEqual(
            [row.source for row in rows],
            [
                "vehicle_local_position.z_reset_counter",
                "vehicle_local_position.dist_bottom_reset_counter",
            ],
        )
        self.assertAlmostEqual(rows[0].time_s, 2.0)
        self.assertIn("-0.250 m z", rows[0].delta)
        self.assertAlmostEqual(rows[1].time_s, 3.0)
        self.assertIn("0.120 m HAGL", rows[1].delta)

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


class InertialDeadReckoningPlotTests(unittest.TestCase):
    def test_plot_combines_state_aiding_flow_status_and_resets(self):
        t_us = (BASE_TS + np.arange(4) * int(1e6)).astype(np.int64)
        local_position = _make_ds(
            "vehicle_local_position",
            t_us,
            vx=np.array([0.0, -0.1, 0.0, -0.1]),
            vy=np.zeros(4),
            vz=np.zeros(4),
            ax=np.array([0.0, -0.1, 0.0, -0.1]),
            ay=np.zeros(4),
            az=np.zeros(4),
            vxy_reset_counter=np.array([1, 1, 2, 2]),
            xy_reset_counter=np.array([3, 3, 4, 4]),
        )
        flags = _make_ds(
            "estimator_status_flags",
            t_us,
            cs_inertial_dead_reckoning=np.array([0, 1, 0, 1]),
            cs_opt_flow=np.ones(4, dtype=int),
            cs_ev_pos=np.zeros(4, dtype=int),
            cs_rng_hgt=np.ones(4, dtype=int),
        )
        aid_flow = _make_ds(
            "estimator_aid_src_optical_flow",
            t_us,
            fusion_enabled=np.zeros(4, dtype=int),
            fused=np.zeros(4, dtype=int),
            innovation_rejected=np.zeros(4, dtype=int),
        )
        events = _make_ds(
            "estimator_event_flags",
            t_us,
            reset_vel_to_flow=np.array([0, 1, 0, 1]),
            reset_pos_to_last_known=np.array([0, 1, 0, 1]),
        )
        ulog = _make_ulog([local_position, flags, aid_flow, events])

        with TemporaryDirectory() as tmp_dir:
            output = plot_inertial_dead_reckoning(ulog, BASE_TS, Path(tmp_dir), shade_modes=False)
            self.assertTrue(output.exists())
            self.assertEqual(output.name, "inertial_dead_reckoning.png")
            self.assertTrue(output.with_suffix(".pdf").exists())


class ImuBiasAdjustmentPlotTests(unittest.TestCase):
    def test_plots_sensor_combined_adjusted_and_bias_status(self):
        t_us = (BASE_TS + np.arange(4) * int(1e6)).astype(np.int64)
        sensor_accel = _make_ds(
            "sensor_accel",
            t_us,
            x=np.array([0.1, 0.2, 0.3, 0.4]),
            y=np.array([0.0, 0.1, 0.0, 0.1]),
            z=np.array([-9.8, -9.7, -9.8, -9.7]),
        )
        sensor_gyro = _make_ds(
            "sensor_gyro",
            t_us,
            x=np.array([0.01, 0.02, 0.01, 0.02]),
            y=np.zeros(4),
            z=np.zeros(4),
        )
        sensor_combined = _make_ds(
            "sensor_combined",
            t_us,
            **{
                "accelerometer_m_s2[0]": np.array([0.1, 0.2, 0.3, 0.4]),
                "accelerometer_m_s2[1]": np.zeros(4),
                "accelerometer_m_s2[2]": np.full(4, -9.8),
                "gyro_rad[0]": np.array([0.01, 0.02, 0.01, 0.02]),
                "gyro_rad[1]": np.zeros(4),
                "gyro_rad[2]": np.zeros(4),
            },
        )
        vehicle_acceleration = _make_ds(
            "vehicle_acceleration",
            t_us,
            **{f"xyz[{index}]": np.zeros(4) for index in range(3)},
        )
        vehicle_angular_velocity = _make_ds(
            "vehicle_angular_velocity",
            t_us,
            **{f"xyz[{index}]": np.zeros(4) for index in range(3)},
        )
        bias_fields = {
            "accel_bias_limit": np.full(4, 0.4),
            "accel_bias_valid": np.ones(4, dtype=int),
            "accel_bias_stable": np.zeros(4, dtype=int),
            "gyro_bias_limit": np.full(4, 0.1),
            "gyro_bias_valid": np.ones(4, dtype=int),
            "gyro_bias_stable": np.ones(4, dtype=int),
        }
        for prefix in ("accel", "gyro"):
            for index in range(3):
                bias_fields[f"{prefix}_bias[{index}]"] = np.full(4, 0.01 * (index + 1))
                bias_fields[f"{prefix}_bias_variance[{index}]"] = np.full(4, 0.0001)
        estimator_sensor_bias = _make_ds("estimator_sensor_bias", t_us, **bias_fields)
        ulog = _make_ulog(
            [
                sensor_accel,
                sensor_gyro,
                sensor_combined,
                vehicle_acceleration,
                vehicle_angular_velocity,
                estimator_sensor_bias,
            ]
        )

        with TemporaryDirectory() as tmp_dir:
            fig_dir = Path(tmp_dir)
            accel_output = plot_accel_bias_adjustment(ulog, BASE_TS, fig_dir, shade_modes=False)
            gyro_output = plot_gyro_bias_adjustment(ulog, BASE_TS, fig_dir, shade_modes=False)
            self.assertEqual(accel_output.name, "accel_bias_adjustment.png")
            self.assertTrue(accel_output.with_suffix(".pdf").exists())
            self.assertEqual(gyro_output.name, "gyro_bias_adjustment.png")
            self.assertTrue(gyro_output.with_suffix(".pdf").exists())


class ForwardLurchDiagnosticPlotTests(unittest.TestCase):
    def test_derives_motor_groups_from_geometry_and_plots_response_chain(self):
        t_us = (BASE_TS + np.arange(4) * int(1e6)).astype(np.int64)
        attitude = _make_ds(
            "vehicle_attitude",
            t_us,
            **{
                "q[0]": np.ones(4),
                "q[1]": np.zeros(4),
                "q[2]": np.zeros(4),
                "q[3]": np.zeros(4),
            },
        )
        attitude_sp = _make_ds(
            "vehicle_attitude_setpoint",
            t_us,
            roll_body=np.zeros(4),
            pitch_body=np.radians(np.array([0.0, 2.0, 4.0, 2.0])),
            yaw_body=np.zeros(4),
        )
        angular_velocity = _make_ds(
            "vehicle_angular_velocity",
            t_us,
            **{"xyz[1]": np.radians(np.array([0.0, 1.0, 2.0, 1.0]))},
        )
        rates_sp = _make_ds(
            "vehicle_rates_setpoint",
            t_us,
            pitch=np.radians(np.array([0.0, 2.0, 3.0, 1.0])),
        )
        torque_sp = _make_ds(
            "vehicle_torque_setpoint", t_us, **{"xyz[1]": np.array([0.0, 0.1, 0.2, 0.1])}
        )
        allocator = _make_ds(
            "control_allocator_status", t_us, **{"unallocated_torque[1]": np.zeros(4)}
        )
        motors = _make_ds(
            "actuator_motors",
            t_us,
            **{
                "control[0]": np.array([0.2, 0.3, 0.4, 0.3]),
                "control[1]": np.array([0.2, 0.2, 0.3, 0.2]),
                "control[2]": np.array([0.2, 0.3, 0.4, 0.3]),
                "control[3]": np.array([0.2, 0.2, 0.3, 0.2]),
            },
        )
        bias = _make_ds(
            "estimator_sensor_bias",
            t_us,
            **{f"accel_bias[{index}]": np.full(4, 0.01 * index) for index in range(3)},
        )
        ulog = _make_ulog(
            [attitude, attitude_sp, angular_velocity, rates_sp, torque_sp, allocator, motors, bias]
        )
        ulog.initial_parameters = {
            "CA_ROTOR_COUNT": 4,
            "CA_ROTOR0_PX": 0.1,
            "CA_ROTOR1_PX": -0.1,
            "CA_ROTOR2_PX": 0.1,
            "CA_ROTOR3_PX": -0.1,
        }

        self.assertEqual(front_rear_motor_indices(ulog, 4), ([0, 2], [1, 3]))
        with TemporaryDirectory() as tmp_dir:
            output = plot_forward_lurch_diagnostics(ulog, BASE_TS, Path(tmp_dir), shade_modes=False)
            self.assertEqual(output.name, "forward_lurch_diagnostics.png")
            self.assertTrue(output.with_suffix(".pdf").exists())


if __name__ == "__main__":
    unittest.main()
