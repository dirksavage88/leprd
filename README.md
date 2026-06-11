# EKF2 Playback Review

A small report generator for PX4 EKF2 replay logs. It reads one or more `.ulg` files, generates comparison plots, writes a text summary, and optionally compiles a LaTeX PDF.

## What It Produces

- Raw GPS altitude and EKF local/global position plots
- GPS quality, DOP, accuracy, jamming/spoofing, and EKF GPS check plots
- Fusion flags, dead reckoning, bad vertical accel, and reset counters
- Innovation test ratios
- Vertical and horizontal residuals against approximate gate thresholds
- Replay comparison plots and a summary table
- `summary.txt` with key metrics for each replay
- `ekf2_replay_review.pdf` when `latexmk` is available

## Quick Start

From this repo directory:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
ekf2-review generate --config path/to/your-review.toml
```

For a PX4 workspace without a venv:

```bash
PYTHONPATH=src python -m ekf2_playback_review generate --config path/to/your-review.toml
```

## Config Format

The tool uses TOML so it does not need a YAML dependency.

```toml
title = "PX4 EKF Replay GNSS Analysis"
root = "/path/to/px4-firmware"
output_dir = "reports/my-review"
original_key = "baseline"
latest_key = "replay1"
compile_pdf = true

[[logs]]
key = "baseline"
title = "Original flight log"
path = "path/to/original.ulg"

[[logs]]
key = "replay1"
title = "Replay 1: GPS_CTRL=7"
path = "build/px4_sitl_default_replay/rootfs/log/YYYY-MM-DD/HH_MM_SS_replayed.ulg"
```

`root` is resolved relative to the config file. Log paths are resolved relative to `root`. `output_dir` is resolved relative to the config file unless absolute.

## Notes

- `estimator_aid_src_gnss_vel` is often not logged in replay output. The tool uses `estimator_status_flags.cs_gnss_vel` as the fusion-active signal.
- Innovation test ratio `1.0` means outside the configured EKF gate, not one sigma.
- The tool intentionally keeps generated reports out of git via `.gitignore`.
