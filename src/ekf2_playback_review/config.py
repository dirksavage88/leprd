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
    root: Path
    output_dir: Path
    original_key: str
    latest_key: str
    compile_pdf: bool
    logs: tuple[LogSpec, ...]


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def load_config(path: str | Path) -> ReviewConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as fh:
        data = tomllib.load(fh)

    config_dir = config_path.parent
    root = _resolve(config_dir, data.get("root", "."))
    output_dir = _resolve(config_dir, data.get("output_dir", "reports/review"))

    logs = []
    for raw in data.get("logs", []):
        try:
            key = str(raw["key"])
            title = str(raw.get("title", key))
            rel_path = raw["path"]
        except KeyError as exc:
            raise ValueError(f"missing required log field: {exc.args[0]}") from exc
        logs.append(LogSpec(key=key, title=title, path=_resolve(root, rel_path)))

    if not logs:
        raise ValueError("config must contain at least one [[logs]] entry")

    original_key = str(data.get("original_key", logs[0].key))
    latest_key = str(data.get("latest_key", logs[-1].key))
    keys = {log.key for log in logs}
    if original_key not in keys:
        raise ValueError(f"original_key {original_key!r} is not in logs")
    if latest_key not in keys:
        raise ValueError(f"latest_key {latest_key!r} is not in logs")

    return ReviewConfig(
        title=str(data.get("title", "PX4 EKF2 Playback Review")),
        root=root,
        output_dir=output_dir,
        original_key=original_key,
        latest_key=latest_key,
        compile_pdf=bool(data.get("compile_pdf", False)),
        logs=tuple(logs),
    )
