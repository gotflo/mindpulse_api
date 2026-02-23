"""
Sliding window manager for real-time PPI stream processing.

Maintains a time-based sliding window of PPI samples and emits
complete windows at configurable step intervals.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from app.config.settings import SignalConfig

logger = logging.getLogger(__name__)


@dataclass
class WindowData:
    ppi_ms: np.ndarray
    timestamps: np.ndarray
    window_start: float
    window_end: float
    sample_count: int


@dataclass
class _Sample:
    timestamp: float
    ppi_ms: int


class SlidingWindow:
    def __init__(self, config: SignalConfig):
        self._config = config
        self._buffer: deque[_Sample] = deque()
        self._last_emit_time: float = 0.0
        self._on_window: Optional[Callable[[WindowData], None]] = None

    def on_window(self, callback: Callable[[WindowData], None]):
        self._on_window = callback

    def add_samples(self, ppi_ms: list[int], timestamp: float):
        """Add new PPI samples. Timestamps are reconstructed from PPI durations."""
        # Reconstruct timestamps going backward from `timestamp`
        # then add to buffer in chronological order (oldest first).
        t = timestamp
        batch: list[_Sample] = []
        for ppi in reversed(ppi_ms):
            batch.append(_Sample(timestamp=t, ppi_ms=ppi))
            t -= ppi / 1000.0
        # batch is newest-first → reverse to get oldest-first
        for s in reversed(batch):
            self._buffer.append(s)

        self._evict_old()

        # Debug: show buffer state (debug level to reduce I/O)
        span = self.buffer_duration_sec
        needed = self._config.window_size_sec * 0.8
        logger.debug("Buffer: %d samples, span=%.1fs / needed=%.1fs",
                      len(self._buffer), span, needed)

        self._try_emit()

    def _evict_old(self):
        if not self._buffer:
            return
        cutoff = self._buffer[-1].timestamp - self._config.window_size_sec
        while self._buffer and self._buffer[0].timestamp < cutoff:
            self._buffer.popleft()

    def _try_emit(self):
        now = time.time()
        if now - self._last_emit_time < self._config.window_step_sec:
            return

        if not self._buffer:
            return

        span = self._buffer[-1].timestamp - self._buffer[0].timestamp
        # First emission: accept with only 5s of data (~30%) for fast start.
        # After that: require 60% fill for quality.
        # With 15s window: first at ~5s, then need ~9s of data.
        min_fill = 0.33 if self._last_emit_time == 0.0 else 0.6
        if span < self._config.window_size_sec * min_fill:
            return

        window = WindowData(
            ppi_ms=np.array([s.ppi_ms for s in self._buffer], dtype=np.float64),
            timestamps=np.array([s.timestamp for s in self._buffer]),
            window_start=self._buffer[0].timestamp,
            window_end=self._buffer[-1].timestamp,
            sample_count=len(self._buffer),
        )

        self._last_emit_time = now
        if self._on_window:
            self._on_window(window)

    def reset(self):
        self._buffer.clear()
        self._last_emit_time = 0.0

    @property
    def buffer_duration_sec(self) -> float:
        if len(self._buffer) < 2:
            return 0.0
        return self._buffer[-1].timestamp - self._buffer[0].timestamp

    @property
    def sample_count(self) -> int:
        return len(self._buffer)
