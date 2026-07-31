from pathlib import Path
import tempfile
import unittest

import numpy as np

from ekf2_playback_review.ardupilot_source import build_dataset
from ekf2_playback_review.config import config_from_log, load_config
from ekf2_playback_review.logsource import ARDUPILOT, ULOG, detect_log_type, resolve_log_type
from ekf2_playback_review.rover import assess_nav_health, steering_gains, turn_rate_overshoot


class BuildDatasetTests(unittest.TestCase):
    def test_returns_none_for_empty(self):
        self.assertIsNone(build_dataset("STER", []))

    def test_builds_typed_columns(self):
        records = [
            {"timestamp": 1000, "DesTurnRate": 10.0, "TurnRate": 9.0},
            {"timestamp": 2000, "DesTurnRate": 12.0, "TurnRate": 13.0},
        ]
        ds = build_dataset("STER", records)
        self.assertEqual(ds.name, "STER")
        self.assertEqual(ds.data["timestamp"].dtype, np.int64)
        self.assertEqual(ds.data["DesTurnRate"].dtype, float)
        np.testing.assert_array_equal(ds.data["TurnRate"], [9.0, 13.0])

    def test_multi_instance_filtered_by_core(self):
        records = [
            {"timestamp": 1000, "C": 0, "SV": 0.2},
            {"timestamp": 1000, "C": 1, "SV": 0.9},
            {"timestamp": 2000, "C": 0, "SV": 0.3},
        ]
        core0 = build_dataset("XKF4", records, multi_id=0)
        core1 = build_dataset("XKF4", records, multi_id=1)
        np.testing.assert_array_equal(core0.data["SV"], [0.2, 0.3])
        np.testing.assert_array_equal(core1.data["SV"], [0.9])
        self.assertIsNone(build_dataset("XKF4", records, multi_id=2))


class LogTypeTests(unittest.TestCase):
    def test_detects_from_extension(self):
        self.assertEqual(detect_log_type(Path("flight.ulg")), ULOG)
        self.assertEqual(detect_log_type(Path("rover.BIN")), ARDUPILOT)
        self.assertEqual(detect_log_type(Path("rover.log")), ARDUPILOT)

    def test_unknown_extension_raises(self):
        with self.assertRaises(ValueError):
            detect_log_type(Path("rover.dat"))

    def test_explicit_type_overrides_extension(self):
        # A dataflash log saved under an unrecognised name still resolves.
        self.assertEqual(resolve_log_type(Path("rover.dat"), ARDUPILOT), ARDUPILOT)


class ConfigLogTypeTests(unittest.TestCase):
    def _load(self, text: str):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "review.toml"
            config_path.write_text(text, encoding="utf-8")
            return load_config(config_path)

    def test_defaults_to_auto(self):
        config = self._load('[log]\npath = "rover.BIN"\n')
        self.assertEqual(config.log.log_type, "auto")
        self.assertEqual(config.log.path.name, "rover.BIN")

    def test_top_level_log_type_is_the_default(self):
        config = self._load('log_type = "ardupilot"\n[log]\npath = "rover.dat"\n')
        self.assertEqual(config.log.log_type, ARDUPILOT)

    def test_per_log_log_type_wins(self):
        config = self._load(
            'log_type = "ulog"\n[log]\npath = "rover.dat"\nlog_type = "ardupilot"\n'
        )
        self.assertEqual(config.log.log_type, ARDUPILOT)

    def test_default_title_names_the_analysis(self):
        self.assertIn("ArduPilot Rover", self._load('[log]\npath = "rover.BIN"\n').title)
        self.assertIn("PX4 EKF2", self._load('[log]\npath = "flight.ulg"\n').title)
        self.assertIn("ArduPilot Rover", config_from_log("rover.BIN").title)
        self.assertIn("PX4 EKF2", config_from_log("flight.ulg").title)

    def test_unknown_extension_does_not_break_title(self):
        # open_log reports the unresolvable type later, with a better message.
        self.assertIn("PX4 EKF2", self._load('[log]\npath = "rover.dat"\n').title)


class SteeringGainsTests(unittest.TestCase):
    def test_resolves_rate_params(self):
        params = {"ATC_STR_RAT_FF": 0.2, "ATC_STR_RAT_P": 0.18, "ATC_STR_RAT_I": 0.15}
        gains = steering_gains(params)
        self.assertEqual(gains["ff"], 0.2)
        self.assertEqual(gains["p"], 0.18)
        self.assertEqual(gains["i"], 0.15)
        self.assertIsNone(gains["d"])

    def test_falls_back_to_bare_names(self):
        # A build that logs the abbreviated names still resolves.
        gains = steering_gains({"ATC_STR_P": 0.3, "ATC_STR_I": 0.25})
        self.assertEqual(gains["p"], 0.3)
        self.assertEqual(gains["i"], 0.25)


class TurnRateOvershootTests(unittest.TestCase):
    def test_empty(self):
        result = turn_rate_overshoot(np.array([]), np.array([]), np.array([]))
        self.assertEqual(result["n_events"], 0)
        self.assertEqual(result["peak_overshoot_pct"], 0.0)

    def test_detects_overshoot_event(self):
        t = np.arange(6, dtype=float)
        des = np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0])
        # Achieved rate overshoots the 20 deg/s command by 50% mid-segment.
        act = np.array([20.0, 20.0, 30.0, 30.0, 20.0, 20.0])
        result = turn_rate_overshoot(t, des, act)
        self.assertEqual(result["n_events"], 1)
        self.assertAlmostEqual(result["peak_overshoot_pct"], 50.0, places=3)

    def test_ignores_low_command_noise(self):
        # Command stays below threshold, so jitter is not scored as overshoot.
        t = np.arange(4, dtype=float)
        des = np.array([0.0, 1.0, 0.5, 0.0])
        act = np.array([2.0, 3.0, 2.5, 1.0])
        result = turn_rate_overshoot(t, des, act)
        self.assertEqual(result["n_events"], 0)

    def test_opposite_direction_not_overshoot(self):
        # Achieved rate larger in magnitude but opposite sign is tracking error,
        # not overshoot.
        t = np.arange(3, dtype=float)
        des = np.array([20.0, 20.0, 20.0])
        act = np.array([-25.0, -25.0, -25.0])
        result = turn_rate_overshoot(t, des, act)
        self.assertEqual(result["n_events"], 0)


class NavHealthTests(unittest.TestCase):
    def test_healthy_when_within_gate(self):
        verdict, _ = assess_nav_health(
            {"vel": 0.4, "pos": 0.5, "hgt": 0.3, "mag": 0.6}, {}, 0, 0, [0]
        )
        self.assertEqual(verdict, "HEALTHY")

    def test_marginal_on_large_test_ratio(self):
        # A big transient GPS/position rejection without a filter fault is marginal.
        verdict, note = assess_nav_health(
            {"pos": 305.0}, {"pos": 0.007}, 0, 0, [0]
        )
        self.assertEqual(verdict, "MARGINAL")
        self.assertIn("pos", note)

    def test_marginal_on_core_switch(self):
        verdict, _ = assess_nav_health({}, {}, 0, 0, [0, 1])
        self.assertEqual(verdict, "MARGINAL")

    def test_degraded_on_filter_fault(self):
        verdict, _ = assess_nav_health({}, {}, 4, 0, [0])
        self.assertEqual(verdict, "DEGRADED")

    def test_brief_minor_exceedance_stays_healthy(self):
        verdict, note = assess_nav_health({"mag": 1.3}, {"mag": 0.001}, 0, 0, [0])
        self.assertEqual(verdict, "HEALTHY")
        self.assertIn("minor", note)


if __name__ == "__main__":
    unittest.main()
