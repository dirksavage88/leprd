# Report Field Notes

## Fusion Flags

The report relies on `estimator_status_flags` for fusion state. In replay logs, `estimator_aid_src_*` topics are not always present even when the corresponding fusion path is active.

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
