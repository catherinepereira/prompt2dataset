"""A UI-agnostic progress callback for long-running operations.

The library reports progress by calling an optional on_progress(Progress) callback. It
never imports Rich or any UI, so a caller renders progress however it likes (a CLI
spinner, a web progress bar, or nothing).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class Progress:
    total: int = 0
    done: int = 0
    message: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


OnProgress = Optional[Callable[[Progress], None]]


class Reporter:
    """Tracks progress and forwards each update to an optional callback."""

    def __init__(self, on_progress: OnProgress = None) -> None:
        self._cb = on_progress
        self.state = Progress()

    def start(self, total: int, message: str = "") -> None:
        self.state = Progress(total=total, done=0, message=message)
        self._emit()

    def advance(self, message: str | None = None, step: int = 1) -> None:
        self.state.done += step
        if message is not None:
            self.state.message = message
        self._emit()

    def set_message(self, message: str) -> None:
        self.state.message = message
        self._emit()

    def _emit(self) -> None:
        if self._cb is not None:
            self._cb(self.state)
