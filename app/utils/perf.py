"""Rolling FPS and latency percentiles over the last N frames."""

from __future__ import annotations

import time
from collections import deque

import numpy as np


class PerfTracker:
    """Keeps (completion time, latency) for the last `window` frames.

    Completion times use time.perf_counter() (monotonic, high resolution; time.monotonic()
    ticks only every ~15.6 ms on Windows): this measures wall-clock throughput,
    unlike the pipeline's temporal logic, which uses client frame timestamps.
    """

    def __init__(self, window: int) -> None:
        if window < 2:
            raise ValueError("window must be >= 2")
        self._samples: deque[tuple[float, float]] = deque(maxlen=window)

    def add(self, latency_ms: float, now: float | None = None) -> None:
        """Record one completed frame. `now` (seconds, perf_counter) is injectable for tests."""
        t = time.perf_counter() if now is None else now
        self._samples.append((t, latency_ms))

    def __len__(self) -> int:
        return len(self._samples)

    def fps(self) -> float:
        """Frames per second over the window: (n - 1) intervals / time span. 0 if undefined."""
        if len(self._samples) < 2:
            return 0.0
        span = self._samples[-1][0] - self._samples[0][0]
        return (len(self._samples) - 1) / span if span > 0 else 0.0

    def percentile(self, p: float) -> float:
        """Latency percentile `p` (0–100) in ms over the window. 0 if empty."""
        if not self._samples:
            return 0.0
        return float(np.percentile([lat for _, lat in self._samples], p))

    def snapshot(self) -> dict[str, float]:
        """Current fps, p50 and p95 latency."""
        return {"fps": self.fps(), "p50_ms": self.percentile(50), "p95_ms": self.percentile(95)}
