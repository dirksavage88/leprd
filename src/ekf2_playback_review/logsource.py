"""Common log-source abstraction.

Both PX4 ULog (`pyulog`) and ArduPilot dataflash BIN (`pymavlink.DFReader`) logs
are exposed through a single interface so the analysis pipeline can stay agnostic
about the underlying format:

- `Dataset` mirrors the small slice of the `pyulog` dataset interface the analysis
  code actually uses: a ``name``, a ``multi_id`` (instance/core), and a ``data``
  dict of equal-length arrays keyed by field name. ``data["timestamp"]`` is always
  microseconds (``int64``) so `analysis.t_rel` works unchanged.
- `LogSource` is the per-file handle: it resolves datasets by topic/message name,
  exposes the onboard parameter values, and reports a ``start_timestamp`` (µs) used
  as the shared relative-time base for plots.

`open_log` is the factory: it picks an implementation from an explicit ``log_type``
or by file extension, importing the concrete backend lazily so a missing optional
dependency (``pyulog`` or ``pymavlink``) only matters for the format that needs it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ULOG = "ulog"
ARDUPILOT = "ardupilot"
AUTO = "auto"


@dataclass(frozen=True)
class Dataset:
    """A single topic/message instance as equal-length field arrays."""

    name: str
    multi_id: int
    data: dict[str, np.ndarray]


class LogSource(ABC):
    """Uniform handle over one log file."""

    #: One of ``ULOG`` / ``ARDUPILOT``; set by the concrete implementation.
    log_type: str

    @property
    @abstractmethod
    def start_timestamp(self) -> int:
        """Earliest sample timestamp in microseconds."""

    @property
    @abstractmethod
    def params(self) -> dict[str, float]:
        """Onboard parameter values (PX4 initial params / ArduPilot ``PARM``)."""

    @abstractmethod
    def get_dataset(self, name: str, multi_id: int = 0):
        """Return the named dataset instance, or ``None`` if absent."""


def detect_log_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".ulg":
        return ULOG
    if suffix in (".bin", ".log"):
        return ARDUPILOT
    raise ValueError(
        f"cannot infer log type from {path.name!r}; set log_type explicitly "
        f"(one of {ULOG!r}, {ARDUPILOT!r})"
    )


def resolve_log_type(path: Path, log_type: str = AUTO) -> str:
    """Resolve an explicit or ``auto`` ``log_type`` to a concrete backend name."""
    return detect_log_type(path) if log_type in (AUTO, "", None) else log_type


def open_log(path: str | Path, log_type: str = AUTO, keep_types=None) -> LogSource:
    """Open ``path`` as a `LogSource`, choosing a backend by ``log_type``.

    ``log_type`` of ``"auto"`` (or empty) infers from the file extension. Backend
    modules are imported lazily so the unused format's dependency is optional.
    ``keep_types`` (ArduPilot only) restricts which message types are retained,
    keeping memory bounded on large dataflash logs; it is ignored for ULog.
    """
    path = Path(path)
    resolved = resolve_log_type(path, log_type)

    if resolved == ULOG:
        from .ulog_source import UlogSource

        return UlogSource(path)
    if resolved == ARDUPILOT:
        from .ardupilot_source import ArduPilotSource

        return ArduPilotSource(path, keep_types=keep_types)
    raise ValueError(f"unknown log_type {resolved!r}")
