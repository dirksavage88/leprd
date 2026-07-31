"""PX4 ULog backend for the `LogSource` abstraction."""

from __future__ import annotations

from pathlib import Path

from pyulog import ULog

from .logsource import ULOG, LogSource


class UlogSource(LogSource):
    """Wrap a `pyulog.ULog` so it satisfies `LogSource`.

    ULog dataset objects already expose ``name`` / ``multi_id`` / ``data``, so
    `get_dataset` returns them directly rather than copying into a `Dataset`.
    """

    log_type = ULOG

    def __init__(self, path: str | Path):
        self._ulog = ULog(str(path))

    @property
    def raw(self) -> ULog:
        """The wrapped `pyulog.ULog`.

        The PX4 analysis is inherently PX4-specific and reads ULog attributes that
        are outside the `LogSource` interface (``data_list``, ``msg_info_dict``,
        ``initial_parameters``), so it unwraps the source rather than going through
        `get_dataset`.
        """
        return self._ulog

    @property
    def start_timestamp(self) -> int:
        return int(self._ulog.start_timestamp)

    @property
    def params(self) -> dict[str, float]:
        # initial_parameters covers the values present at log start.
        return {str(k): float(v) for k, v in self._ulog.initial_parameters.items()}

    def get_dataset(self, name: str, multi_id: int = 0):
        for d in self._ulog.data_list:
            if d.name == name and d.multi_id == multi_id:
                return d
        return None
