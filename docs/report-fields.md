# Report Field Notes

## Fusion Flags

The report relies on `estimator_status_flags` for fusion state. In replay logs, `estimator_aid_src_*` topics are not always present even when the corresponding fusion path is active.

For scalar height innovations, the report prefers direct aid-source topics when logged:

- `estimator_aid_src_baro_hgt.innovation`
- `estimator_aid_src_rng_hgt.innovation`
- `estimator_aid_src_gnss_hgt.innovation`

When those topics are missing, it falls back to the aggregate `estimator_innovations.*_vpos`, `estimator_innovation_variances.*_vpos`, and `estimator_innovation_test_ratios.*_vpos` fields. PX4 publishes the aggregate baro height fields from the same internal `aid_src_baro_hgt` state.

Important flags:

- `cs_gps`: GNSS horizontal aiding active
- `cs_gnss_vel`: GNSS velocity aiding active
- `cs_gps_hgt`: GNSS height aiding active
- `cs_inertial_dead_reckoning`: estimator is dead reckoning without enough aiding
- `fs_bad_acc_vertical`: vertical acceleration consistency fault

## Innovation Test Ratios

PX4 logs normalized innovation test ratios. A value greater than `1.0` means the measurement is outside the configured gate. With a 5-sigma gate, a value around `0.36` is roughly equivalent to a 3-sigma residual.

## GPS Checks

`estimator_gps_status.check_fail_max_vert_spd_err` is useful for identifying disagreement between GPS vertical speed and the EKF propagated state. It only affects GPS acceptance if the corresponding bit is enabled in `EKF2_GPS_CHECK`.

## ArduPilot Rover (dataflash `.BIN`)

The Rover review (selected when the reviewed log is an ArduPilot BIN) draws on these messages:

- `STER` — `SteerIn`, `SteerOut`, `DesLatAcc`, `LatAcc`, `DesTurnRate`, `TurnRate`. `DesTurnRate`/`TurnRate` are in deg/s and share timestamps, so the desired-vs-achieved comparison and overshoot scoring need no interpolation.
- `PIDS` — the steering-rate PID controller: `Tar`, `Act`, `Err`, `P`, `I`, `D`, `FF`. The term contributions reflect the configured `ATC_STR_RAT_*` gains.
- `XKF3` — EKF3 innovations (`IVN/IVE/IVD`, `IPN/IPE/IPD`, `IMX/IMY/IMZ`, `IYAW`).
- `XKF4` — EKF3 normalized innovation test ratios (`SV`, `SP`, `SH`, `SM`) plus fault/timeout/solution-status masks (`FS`, `TS`, `SS`, `GPS`, `PI`). A ratio `> 1.0` means the measurement is outside the gate.
- `PARM` — onboard parameters, used to report the steering-rate gains.

### Steering Gains

The Rover steering-rate controller is `ATC_STR_RAT_FF` (feed forward), `ATC_STR_RAT_P`, `ATC_STR_RAT_I`, `ATC_STR_RAT_D`, with `ATC_STR_RAT_IMAX` clamping integrator authority. There is no bare `ATC_STR_P`/`ATC_STR_I` in stock ArduPilot — those abbreviations map to the `_RAT_` params. High `FF` relative to `P`/`I`, or large `IMAX`, tends to show up as turn-rate overshoot in the `STER` comparison.

### Overshoot

Overshoot is the fraction by which the achieved turn rate exceeds the commanded turn rate in the same direction, scored only while a meaningful turn is commanded (`|DesTurnRate| > 3 deg/s`). The report gives peak overshoot %, the fraction of commanded time spent overshooting, and the active intervals.
