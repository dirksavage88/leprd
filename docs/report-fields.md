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
