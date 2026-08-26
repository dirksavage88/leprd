# Report Field Notes

## Source Log

The first page prints the full source ULog path so generated PDFs can be traced
back to the exact log even when several logs share similar basenames.

## Fusion Flags

The report relies on `estimator_status_flags` for fusion state. In replay logs, `estimator_aid_src_*` topics are not always present even when the corresponding fusion path is active.

## Sensor Health

The Sensor Health section starts with a table built from ULog `initial_parameters`.
It lists logged `SYS_HAS_*` and `EKF2_*_CTRL` values, plus expected common
sensor rows as `not logged` when absent from the log metadata. Legacy `HAS_*`
parameters are omitted from the sensor health table.
`ENABLED` and `DISABLED` rows are highlighted green and red in the PDF.

The larger sensor subsections use sensor-specific topics when available:

- IMU: calibrated body-frame accelerometer and gyroscope measurements from `sensor_combined`
  (`accelerometer_m_s2[]`, `gyro_rad[]`) when logged, falling back to
  `vehicle_imu.delta_velocity[] / delta_velocity_dt` and
  `vehicle_imu.delta_angle[] / delta_angle_dt`, plus `vehicle_imu_status`,
  `estimator_status`, and non-control-status IMU flags from
  `estimator_status_flags`
- Barometer: `vehicle_air_data`, baro height aid-source or aggregate test ratios, and height reset/timeout flags
- Magnetometer: `vehicle_magnetometer` or `sensor_mag`, heading test ratios, and yaw rejection flags
- GPS/GNSS: `vehicle_gps_position` quality/jamming/spoofing fields, plus
  separate `estimator_gps_status.check_fail_*` plots
- Optical flow: `vehicle_optical_flow.quality`, `vehicle_optical_flow.pixel_flow[0/1]`, `estimator_aid_src_optical_flow`, `estimator_optical_flow_vel`, and OF aid-source/rejection/fault flags
- Range finder: `distance_sensor`, `vehicle_local_position.dist_bottom`, `estimator_aid_src_rng_hgt`, aggregate `hagl_rate` when logged, and range/HAGL aid-source/event/rejection flags
- External vision: `vehicle_visual_odometry.position[0/1/2]`, `vehicle_local_position.x/y/z`, EV aid-source topics, and EV event flags. EV aid-source test ratios are split by source: horizontal position `test_ratio[0/1]`, velocity `test_ratio[0/1/2]` when logged, height `test_ratio`, and yaw `test_ratio`.

Sensor-specific figures do not plot `estimator_status_flags.cs_*` traces. Those
control-status bits are centralized in the Control Statuses section, where every
logged `cs_*` field is plotted in grouped stacked 0/1 panels.

After the individual sensor subsections, the Sensor Latency section plots
PX4-side sample-to-publication latency for each logged sensor topic with both
`timestamp` and `timestamp_sample`. Each topic or topic instance gets its own
panel and uses signed `timestamp - timestamp_sample` in milliseconds. For
MAVLink-fed topics, `timestamp_sample` is assigned or preserved in the PX4
receiver/bridge path; driver-backed topics use their driver/sample time, so the
plot is PX4-side time-tag latency rather than guaranteed end-to-end transport
latency.

UTC system-time axes are only drawn when the ULog metadata contains
`boot_time_utc_us`; logs without that metadata do not get an empty or
unavailable system-time plot annotation.

The IMU vibration panel follows Flight Review-style accel vibration bands:
values below 5 are green, below 10 are yellow, and values above 10 are red.
When multiple `vehicle_imu_status` instances are logged, each accel vibration
instance is plotted separately.
The separate IMU Raw and Bias-Adjusted Measurements subsection preserves the
processing-stage distinction: `sensor_accel`/`sensor_gyro` are sensor/board FRD
measurements, `sensor_combined` is calibrated body FRD, and
`vehicle_acceleration`/`vehicle_angular_velocity` are body-frame outputs with
the matching `estimator_sensor_bias` in-run bias removed and filtering applied.
It plots X/Y/Z comparisons, bias estimates with one-sigma uncertainty and
limits, and the corresponding bias `valid` and `stable` fields. Sensor-level
streams may appear sparse because PX4 logging rates are independent of hardware
sampling rates.

The Pitch and Forward-Lurch Diagnostics subsection plots the pitch attitude and
pitch-rate response chains, commanded and unallocated pitch torque, front/rear
motor command means and their differential, and
`estimator_sensor_bias.accel_bias[0..2]`. Front and rear motor groups are derived
from the sign of each logged `CA_ROTOR*_PX` geometry parameter; they are not
hard-coded to one motor numbering scheme.
The attitude overview uses `vehicle_attitude`, `vehicle_attitude_setpoint`,
`vehicle_angular_velocity`, and `vehicle_local_position` when available.

The PX4 Setpoint Diagnostics section uses two figures so the y-axis height of
each subplot stays readable. The command figure plots `trajectory_setpoint`,
`offboard_control_mode`, and `vehicle_local_position_setpoint`, with extra
height on the velocity panels. The response figure plots
`vehicle_local_position`, `distance_sensor`, `estimator_states`, and
`vehicle_attitude_setpoint`. The vertical-velocity response panel overlays
`trajectory_setpoint.velocity[2]`, `vehicle_local_position_setpoint.vz`,
`vehicle_local_position.vz`, and `vehicle_local_position.z_deriv`. The
range/terrain panel plots HAGL/range signals and `estimator_states.states[24]`
when logged so vertical setpoint behavior can be compared with the EKF terrain
state.

The motor / actuator / ESC section prefers `actuator_motors.control[]`, falls
back to varying `actuator_outputs.output[]` channels, plots
`control_allocator_status.actuator_saturation[]` fields against PX4's
`ACTUATOR_SATURATION_OK`, `ACTUATOR_SATURATION_UPPER_DYN`,
`ACTUATOR_SATURATION_UPPER`, `ACTUATOR_SATURATION_LOWER_DYN`, and
`ACTUATOR_SATURATION_LOWER` state labels, plots thrust from
`vehicle_thrust_setpoint` or `actuator_controls_0`, and plots all logged ESC RPM
channels from `esc_status.esc[i].esc_rpm` together.

The Estimator Status Masks section shows `control_mode_flags`,
`filter_fault_flags`, and `solution_status_flags` separately. Observed raw mask
values are annotated inside each figure instead of repeated in a table.

The Sensor Health subsections also include separate decoded parameter tables:

- IMU: `EKF2_IMU_CTRL` bitmask
- Magnetometer: `EKF2_MAG_CHECK` bitmask
- GPS/GNSS: `EKF2_GPS_CTRL` and `EKF2_GPS_CHECK` bitmasks
- Range finder: `EKF2_RNG_CTRL` enum mode

Each table omits the repeated parameter-name column and lists only the decoded
bit or value rows for that subsection.

The External Vision Integrity subsection includes a dedicated XYZ position
comparison figure. It overlays logged `vehicle_visual_odometry.position[0/1/2]`
measurements with `vehicle_local_position.x/y/z`, then plots local estimate
minus EV measurement at the EV sample times.

The terrain estimate figure plots actual HAGL from
`vehicle_local_position.dist_bottom` against the HAGL implied by local-z
setpoints, computed as `vehicle_local_position.z + dist_bottom - z_setpoint`.
This makes terrain-hold and range-height effects visible in HAGL units instead
of requiring comparison of NED-down local `z` traces.

The Altitude Overview figure also includes a terrain/HAGL state panel. It plots
`estimator_states.states[24]` as the raw EKF `_state.terrain` value when that
25-state topic is logged, plots `vehicle_local_position.dist_bottom` as the EKF
HAGL state, and derives a local terrain vertical state as
`vehicle_local_position.z + dist_bottom`. When `vehicle_global_position` is logged, it also plots
`vehicle_global_position.terrain_alt` as a NED-down delta relative to its first
valid sample and the global-position HAGL implied by
`vehicle_global_position.alt - terrain_alt`.
PX4 C++ code often refers to this global altitude state through the
`gpos.altitude()` accessor, but `_gpos` is EKF2's internal global-position
bookkeeping object. It can carry the propagated altitude used by terrain/HAGL
logic even when no publishable `vehicle_global_position` topic exists because
the EKF has no valid global origin. When the topic is logged, the corresponding
ULog field name is `vehicle_global_position.alt`.
Older 24-state EKF logs do not include `_state.terrain` in `estimator_states`;
in those logs `states[23]` is wind velocity, not terrain.

The Range Finder Integrity figure plots one range-height innovation test ratio:
direct `estimator_aid_src_rng_hgt.test_ratio` when logged, otherwise aggregate
`estimator_innovation_test_ratios.hagl`, falling back to aggregate `rng_vpos`.
Current PX4 publishes aggregate `hagl` and `rng_vpos` from the same internal
`aid_src_rng_hgt.test_ratio`, so the report does not overlay them as separate
signals. Aggregate `estimator_innovation_test_ratios.hagl_rate` is plotted in a
separate panel when logged because it is a different kinematic consistency
ratio.

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

All logged `cs_*` fields are plotted in the Control Statuses section rather than
inside individual sensor figures.

The Inertial Dead Reckoning Diagnostics section gathers the signals needed to
distinguish inertial propagation from active aiding. It plots local NED
velocity and acceleration, dead-reckoning and vehicle-state flags, horizontal
aiding and height control bits, per-sample optical-flow `fusion_enabled`,
`fused`, and `innovation_rejected` status, and horizontal velocity/position reset counters.
When logged, dotted event markers identify `reset_vel_to_flow` and
`reset_pos_to_last_known` events. An active `cs_opt_flow` bit is therefore shown
separately from proof that an optical-flow sample was actually fused.

## Innovation Test Ratios

PX4 logs normalized innovation test ratios. A value greater than `1.0` means the measurement is outside the configured gate. With a 5-sigma gate, a value around `0.36` is roughly equivalent to a 3-sigma residual.

## Optical Flow Fusion Gating

The report has a dedicated Optical Flow Fusion Gating section. It plots
`flow[0]` and `flow[1]` normalized test ratios against the 1.0 rejection gate,
then plots the corresponding optical-flow innovations against +/-5-sigma gates
from logged innovation variances. When `estimator_aid_src_optical_flow` is
logged, the section also plots `timestamp - time_last_fuse` and the aid-source
`fused` boolean. The optical-flow health and gating figures include cumulative
measurement counts for raw `sensor_optical_flow` and EKF-input
`vehicle_optical_flow`. Optical-flow `vehicle_optical_flow.timestamp -
vehicle_optical_flow.timestamp_sample` latency is collected in the Sensor
Latency section. When `estimator_aid_src_optical_flow.fusion_enabled` is logged, its
boolean panel title includes the time-weighted enabled percentage. Direct
`estimator_aid_src_optical_flow` fields are preferred when present; aggregate
`estimator_innovation_*` flow fields are used as fallback for older or replay
logs.

## GPS Checks

`estimator_gps_status.check_fail_max_vert_spd_err` is useful for identifying disagreement between GPS vertical speed and the EKF propagated state. It only affects GPS acceptance if the corresponding bit is enabled in `EKF2_GPS_CHECK`.

The report plots each logged `check_fail_*` field in its own 0/1 subplot so
`check_fail_max_pdop`, speed, drift, and position checks remain legible.

## Estimator Status Masks

`estimator_status.control_mode_flags`, `filter_fault_flags`, and
`solution_status_flags` are bitmasks. The report plots each field separately and
annotates observed raw values inside the figure.

The logged EKF2/MPC/SENS/SDLOG parameter appendix uses paired columns:
`Parameter`, `Logged value`, `Parameter`, `Logged value`.

## Split GPS Innovation Ratios

GPS velocity and position test ratios are plotted by channel:

- `gps_hvel[0]`
- `gps_hvel[1]`
- `gps_vvel`
- `gps_hpos[0]`
- `gps_hpos[1]`
- `gps_vpos`
