"""ArduPilot dataflash (`.BIN`) backend for the `LogSource` abstraction.

ArduPilot logs are a stream of self-describing messages (``GPS``, ``STER``,
``PIDS``, ``XKF1``..``XKF5``, ``PARM``, ...). We parse the whole file once with
`pymavlink.DFReader`, bucket records by message type, and convert each bucket to a
`Dataset` of column arrays on demand.

Conventions that keep this interchangeable with the PX4 path:

- ``TimeUS`` (microseconds since boot) becomes ``data["timestamp"]`` so the shared
  `analysis.t_rel` relative-time helper works unchanged.
- Multi-instance messages (EKF cores ``C``, sensor instances ``I``/``Instance``)
  map onto ``multi_id``; ``multi_id=0`` selects the primary instance, mirroring
  ULog's ``multi_id`` semantics.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .logsource import ARDUPILOT, Dataset, LogSource

# Field that carries the per-message instance number, in priority order.
_INSTANCE_FIELDS = ("C", "I", "Instance", "IMU")

# Always retained regardless of any keep_types filter: parameter records back the
# onboard-parameter API.
_ALWAYS_KEEP = frozenset({"PARM"})


def _is_numeric(value) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def build_dataset(name: str, records: list[dict], multi_id: int = 0) -> Dataset | None:
    """Convert buffered message dicts into a `Dataset` for one instance.

    ``records`` are field dicts (as from `pymavlink` ``to_dict``) that already
    carry a ``timestamp`` key in microseconds. If the message type is
    multi-instance, only rows matching ``multi_id`` are kept. Returns ``None`` when
    no rows remain.
    """
    if not records:
        return None

    instance_field = next((f for f in _INSTANCE_FIELDS if f in records[0]), None)
    if instance_field is not None:
        rows = [r for r in records if int(r.get(instance_field, 0)) == multi_id]
    else:
        rows = records if multi_id == 0 else []
    if not rows:
        return None

    data: dict[str, np.ndarray] = {}
    for key in rows[0]:
        column = [r.get(key) for r in rows]
        if key == "timestamp":
            data[key] = np.asarray(column, dtype=np.int64)
        elif all(_is_numeric(v) for v in column):
            data[key] = np.asarray(column, dtype=float)
        else:
            data[key] = np.asarray(column, dtype=object)
    return Dataset(name=name, multi_id=multi_id, data=data)


class ArduPilotSource(LogSource):
    """Parse an ArduPilot dataflash BIN into instance-addressable datasets."""

    log_type = ARDUPILOT

    def __init__(self, path: str | Path, keep_types: Iterable[str] | None = None):
        keep = None if keep_types is None else (frozenset(keep_types) | _ALWAYS_KEEP)
        self._records, self._params, self._start = _parse_bin(Path(path), keep)
        self._cache: dict[tuple[str, int], Dataset | None] = {}

    @property
    def start_timestamp(self) -> int:
        return self._start

    @property
    def params(self) -> dict[str, float]:
        return dict(self._params)

    def message_types(self) -> list[str]:
        """Message types present in the log (useful for diagnostics/tests)."""
        return sorted(self._records)

    def get_dataset(self, name: str, multi_id: int = 0):
        cache_key = (name, multi_id)
        if cache_key not in self._cache:
            self._cache[cache_key] = build_dataset(name, self._records.get(name, []), multi_id)
        return self._cache[cache_key]


def _parse_bin(
    path: Path, keep_types: frozenset[str] | None = None
) -> tuple[dict[str, list[dict]], dict[str, float], int]:
    from pymavlink import mavutil

    mlog = mavutil.mavlink_connection(str(path))
    records: dict[str, list[dict]] = {}
    params: dict[str, float] = {}
    start = None

    while True:
        msg = mlog.recv_match(blocking=False)
        if msg is None:
            break
        mtype = msg.get_type()
        if mtype == "BAD_DATA":
            continue
        # Skip unwanted types before the costly to_dict() so large logs stay cheap.
        if keep_types is not None and mtype not in keep_types:
            continue

        fields = msg.to_dict()
        fields.pop("mavpackettype", None)

        time_us = getattr(msg, "TimeUS", None)
        if time_us is None:
            time_us = int(float(getattr(msg, "_timestamp", 0.0)) * 1e6)
        time_us = int(time_us)
        fields["timestamp"] = time_us
        if time_us > 0 and (start is None or time_us < start):
            start = time_us

        records.setdefault(mtype, []).append(fields)

        if mtype == "PARM":
            try:
                params[str(msg.Name)] = float(msg.Value)
            except (AttributeError, TypeError, ValueError):
                pass

    return records, params, int(start or 0)
