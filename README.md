# EKF2 Playback Review

A small report generator for flight-controller logs. It reads one log, generates plots, writes a text summary, and compiles a LaTeX PDF. Two formats are supported:

- **PX4 EKF2** — `.ulg` (ULog) files.
- **ArduPilot Rover** — `.BIN`/`.log` dataflash logs.

The log's format selects the analysis; the same `generate` command handles both.

## What It Produces

For **PX4** logs:

- Raw GPS altitude and EKF local/global position plots
- GPS quality, DOP, accuracy, jamming/spoofing, and EKF GPS check plots
- Fusion flags, dead reckoning, bad vertical accel, and reset counters
- Innovation test ratios
- Height setpoint-versus-estimate deltas and flight-mode background shading
- Vertical and horizontal residuals against approximate gate thresholds
- `ekf2_review.pdf`

For **ArduPilot Rover** logs:

- Steering-rate controller gains (`ATC_STR_RAT_FF`/`_P`/`_I`/`_D`/`_IMAX`)
- Desired-vs-achieved turn rate with overshoot intervals shaded, plus metrics (RMS error, peak overshoot %, time overshooting)
- Steering-rate PID target/actual and P/I/D/FF term contributions
- EKF3 innovations (`XKF3`) and normalized test ratios (`XKF4`) with a navigation-health verdict
- `rover_review.pdf`

Both write a `summary.txt` of key metrics.

## Quick Start

From this repo directory:

```bash
uv sync
uv run ekf2-review generate /path/to/flight.ulg
uv run ekf2-review generate /path/to/rover.BIN
```

Or with a plain venv:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
ekf2-review generate /path/to/flight.ulg
```

## System Dependencies

Python dependencies are declared in `pyproject.toml` and locked by `uv.lock`.

PDF generation requires `latexmk` and a LaTeX distribution.

macOS:

```bash
brew install --cask mactex-no-gui
```

If you prefer a smaller install, BasicTeX also works once the required LaTeX packages are installed:

```bash
brew install --cask basictex
sudo tlmgr update --self
sudo tlmgr install latexmk collection-latexrecommended collection-fontsrecommended
```

Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install latexmk texlive-latex-recommended texlive-fonts-recommended texlive-latex-extra
```

Fedora:

```bash
sudo dnf install latexmk texlive-scheme-medium
```

## Output Defaults

By default, reports are written to `reports/<log-stem>-review` under the current directory.

```bash
uv run ekf2-review generate /path/to/flight.ulg
uv run ekf2-review generate /path/to/flight.ulg --output-dir reports/height-reference
uv run ekf2-review generate /path/to/flight.ulg --no-mode-shading
```

## Optional Review Config

Most runs do not need TOML. Use `--config` only when you want to save non-default settings.

```toml
title = "PX4 EKF2 Height Reference Review"
output_dir = "reports/height-reference-review"
shade_flight_modes = true

[log]
key = "flight"
title = "Flight log"
path = "/absolute/path/to/flight.ulg"
```

Relative log paths and `output_dir` are resolved relative to the config file. Use an absolute `path` when the log lives outside this repo.

### Log Type

`log_type` selects the analysis: `auto` (the default), `ulog`, or `ardupilot`. With `auto`, `.ulg` is treated as PX4 and `.BIN`/`.log` as ArduPilot. Set it top-level, inside `[log]`, or with `--log-type` on the command line — useful for a dataflash log saved under an unrecognised extension.

```toml
title = "Boat5 Rover Steering Review"
output_dir = "reports/boat5-review"
log_type = "ardupilot"

[log]
key = "boat5"
title = "boat5_ark_06_18_2026"
path = "boat5_ark_06_18_2026.BIN"
```

## Notes

- `estimator_aid_src_*` topics are preferred for scalar height innovations when present. The aggregate `estimator_innovations`, `estimator_innovation_variances`, and `estimator_innovation_test_ratios` topics are used as fallbacks for older/replay logs.
- `estimator_aid_src_gnss_vel` is often not logged in replay output. The tool uses `estimator_status_flags.cs_gnss_vel` as the fusion-active signal.
- Innovation test ratio `1.0` means outside the configured EKF gate, not one sigma (PX4 and ArduPilot EKF3 alike).
- ArduPilot Rover steering-rate gains are `ATC_STR_RAT_*`; there is no bare `ATC_STR_P`/`ATC_STR_I` in stock firmware.
- `pyulog` and `pymavlink` are imported lazily, so only the backend for the format you use has to work.
- The tool intentionally keeps generated reports out of git via `.gitignore`.
