from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class LogSpec:
    key: str
    title: str
    path: Path


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


def config_from_log(
    log_path: str | Path,
    output_dir: str | Path | None = None,
    title: str | None = None,
) -> ReviewConfig:
    resolved_log = Path(log_path).expanduser().resolve()
    resolved_output = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else (Path.cwd() / "reports" / f"{resolved_log.stem}-review").resolve()
    )

    return ReviewConfig(
        title=title or f"PX4 EKF2 Review: {resolved_log.stem}",
        output_dir=resolved_output,
        log=LogSpec(
            key="flight",
            title=resolved_log.name,
            path=resolved_log,
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
    )

    output_dir = _resolve(config_dir, data.get("output_dir", "reports/review"))

    return ReviewConfig(
        title=str(data.get("title", "PX4 EKF2 Review")),
        output_dir=output_dir,
        log=log,
        divergence_threshold_m=float(data.get("divergence_threshold_m", 0.5)),
        throttle_threshold=float(data.get("throttle_threshold", 0.3)),
        hover_max_speed_ms=float(data.get("hover_max_speed_ms", 1.5)),
        shade_flight_modes=bool(data.get("shade_flight_modes", True)),
    )
