"""Time-based exponential moving average."""

from __future__ import annotations

import math


class EMA:
    """EMA whose smoothing depends on elapsed frame time: alpha = 1 − exp(−dt / tau).

    This behaves the same at 10 or 30 FPS. `tau_s <= 0` disables smoothing.
    """

    def __init__(self, tau_s: float) -> None:
        self.tau_ms = tau_s * 1000.0
        self.value: float | None = None
        self._last_ts: float | None = None

    def update(self, x: float, timestamp_ms: float) -> float:
        """Add a sample and return the smoothed value. A non-increasing timestamp leaves the value unchanged."""
        if self.value is None or self._last_ts is None or self.tau_ms <= 0:
            self.value = float(x)
            self._last_ts = timestamp_ms
            return self.value
        dt = timestamp_ms - self._last_ts
        if dt <= 0:
            return self.value
        alpha = 1.0 - math.exp(-dt / self.tau_ms)
        self.value += alpha * (float(x) - self.value)
        self._last_ts = timestamp_ms
        return self.value

    def reset(self) -> None:
        self.value = None
        self._last_ts = None
