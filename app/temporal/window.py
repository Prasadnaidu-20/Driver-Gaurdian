"""Time-based sliding window over client frame timestamps (ms)."""

from __future__ import annotations

from collections import deque


class TimeWindow:
    """Keeps `(timestamp_ms, value)` samples from the last `duration_s` seconds.

    Time-weighted statistics give each sample the duration until the next sample, capped at
    `max_dt_s`, so a gap in the stream (no face, paused video) does not count as time. The
    newest sample has no successor yet and gets zero weight.
    """

    def __init__(self, duration_s: float, max_dt_s: float) -> None:
        self.duration_ms = duration_s * 1000.0
        self.max_dt_ms = max_dt_s * 1000.0
        self._samples: deque[tuple[float, float]] = deque()

    def __len__(self) -> int:
        return len(self._samples)

    def add(self, timestamp_ms: float, value: float) -> None:
        """Append a sample and drop samples older than the window. Non-increasing timestamps are ignored."""
        if self._samples and timestamp_ms <= self._samples[-1][0]:
            return
        self._samples.append((timestamp_ms, float(value)))
        cutoff = timestamp_ms - self.duration_ms
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def clear(self) -> None:
        self._samples.clear()

    def _weighted(self) -> tuple[float, float]:
        """(sum of value × duration, total duration) in ms."""
        total = 0.0
        weighted = 0.0
        prev: tuple[float, float] | None = None
        for ts, value in self._samples:
            if prev is not None:
                dt = min(ts - prev[0], self.max_dt_ms)
                total += dt
                weighted += prev[1] * dt
            prev = (ts, value)
        return weighted, total

    def span_s(self) -> float:
        """Time (s) from the oldest to the newest sample, gaps included."""
        if len(self._samples) < 2:
            return 0.0
        return (self._samples[-1][0] - self._samples[0][0]) / 1000.0

    def covered_s(self) -> float:
        """Time (s) covered by samples, excluding gaps longer than `max_dt_s`."""
        return self._weighted()[1] / 1000.0

    def time_mean(self, min_span_s: float = 0.0) -> float:
        """Time-weighted mean of the values. The denominator is at least `min_span_s`.

        For 0/1 values this is the fraction of time the condition held (e.g. PERCLOS).
        """
        weighted, total = self._weighted()
        denom = max(total, min_span_s * 1000.0)
        return weighted / denom if denom > 0 else 0.0

    def values(self) -> list[float]:
        return [v for _, v in self._samples]
