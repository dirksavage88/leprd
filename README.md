# EKF2 Playback Review

A small report generator for PX4 EKF2 replay logs. It reads a `.ulg` file, generates comparison plots, writes a text summary, and compiles a LaTeX PDF.

## What It Produces

- Raw GPS altitude and EKF local/global position plots
- GPS quality, DOP, accuracy, jamming/spoofing, and EKF GPS check plots
- Fusion flags, dead reckoning, bad vertical accel, and reset counters
- Innovation test ratios
- Height setpoint-versus-estimate deltas and flight-mode background shading
- Vertical and horizontal residuals against approximate gate thresholds
- Replay comparison plots and a summary table
- `summary.txt` with key metrics for each replay
- `ekf2_review.pdf`

## Quick Start

From this repo directory:

```bash
uv sync
uv run ekf2-review generate /path/to/flight.ulg
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

Relative log paths and `output_dir` are resolved relative to the config file. Use an absolute `path` when the `.ulg` lives outside this repo.

## Notes

- `estimator_aid_src_*` topics are preferred for scalar height innovations when present. The aggregate `estimator_innovations`, `estimator_innovation_variances`, and `estimator_innovation_test_ratios` topics are used as fallbacks for older/replay logs.
- `estimator_aid_src_gnss_vel` is often not logged in replay output. The tool uses `estimator_status_flags.cs_gnss_vel` as the fusion-active signal.
- Innovation test ratio `1.0` means outside the configured EKF gate, not one sigma.
- The tool intentionally keeps generated reports out of git via `.gitignore`.
