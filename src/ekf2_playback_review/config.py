from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib

from .logsource import ARDUPILOT, AUTO, resolve_log_type


@dataclass(frozen=True)
class LogSpec:
    key: str
    title: str
    path: Path
    #: ``auto`` (infer from the file extension), ``ulog``, or ``ardupilot``.
    log_type: str = AUTO


@dataclass(frozen=True)
class ReviewConfig:
    title: str
    output_dir: Path
    log: LogSpec
    divergence_threshold_m: float = 0.5
    throttle_threshold: float = 0.3
    hover_max_speed_ms: float = 1.5
    shade_flight_modes: bool = True


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def _review_label(log_path: Path, log_type: str) -> str:
    """Name the analysis the log will get, for default report titles.

    An unresolvable ``log_type`` is not an error here; `open_log` reports it with a
    better message when the review actually runs.
    """
    try:
        resolved = resolve_log_type(log_path, log_type)
    except ValueError:
        resolved = ""
    return "ArduPilot Rover Review" if resolved == ARDUPILOT else "PX4 EKF2 Review"


def config_from_log(
    log_path: str | Path,
    output_dir: str | Path | None = None,
    title: str | None = None,
    log_type: str = AUTO,
) -> ReviewConfig:
    resolved_log = Path(log_path).expanduser().resolve()
    resolved_output = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else (Path.cwd() / "reports" / f"{resolved_log.stem}-review").resolve()
    )

    return ReviewConfig(
        title=title or f"{_review_label(resolved_log, log_type)}: {resolved_log.stem}",
        output_dir=resolved_output,
        log=LogSpec(
            key="flight",
            title=resolved_log.name,
            path=resolved_log,
            log_type=log_type,
        ),
        shade_flight_modes=True,
    )


def load_config(path: str | Path) -> ReviewConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as fh:
        data = tomllib.load(fh)

    config_dir = config_path.parent

    raw_log = data.get("log")
    if not raw_log:
        raise ValueError("config must contain a [log] table")

    try:
        log_path = _resolve(config_dir, raw_log["path"])
    except KeyError as exc:
        raise ValueError(f"missing required log field: {exc.args[0]}") from exc

    log = LogSpec(
        key=str(raw_log.get("key", "log")),
        title=str(raw_log.get("title", log_path.name)),
        path=log_path,
        # Per-log log_type wins; the top-level key sets the default.
        log_type=str(raw_log.get("log_type", data.get("log_type", AUTO))),
    )

    output_dir = _resolve(config_dir, data.get("output_dir", "reports/review"))

    return ReviewConfig(
        title=str(data.get("title", _review_label(log.path, log.log_type))),
        output_dir=output_dir,
        log=log,
        divergence_threshold_m=float(data.get("divergence_threshold_m", 0.5)),
        throttle_threshold=float(data.get("throttle_threshold", 0.3)),
        hover_max_speed_ms=float(data.get("hover_max_speed_ms", 1.5)),
        shade_flight_modes=bool(data.get("shade_flight_modes", True)),
    )
