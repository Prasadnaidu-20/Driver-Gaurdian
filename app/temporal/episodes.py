"""Tracking of continuous episodes of a boolean condition (eyes closed, mouth open, off road)."""

from __future__ import annotations


class EpisodeTracker:
    """Tracks how long a condition has held continuously, by frame timestamps.

    Dropouts (condition false, or missing frames) shorter than `max_gap_s` do not end an
    episode: one misdetected frame inside a closure does not split it. When an episode
    ends, `update` returns its duration once. The duration runs from the first true
    sample to the last true sample.
    """

    def __init__(self, max_gap_s: float) -> None:
        self.max_gap_ms = max_gap_s * 1000.0
        self.reset()

    def reset(self) -> None:
        self._start: float | None = None
        self._last_true: float | None = None

    @property
    def active(self) -> bool:
        return self._start is not None

    @property
    def start_ms(self) -> float | None:
        """Timestamp of the first sample of the current episode (identifies the episode)."""
        return self._start

    def duration_s(self) -> float:
        """Duration of the current episode (0 if none)."""
        if self._start is None or self._last_true is None:
            return 0.0
        return (self._last_true - self._start) / 1000.0

    def update(self, condition: bool | None, timestamp_ms: float) -> float | None:
        """Feed one frame. `condition=None` means "no observation" (e.g. no face).

        Returns the duration (s) of an episode that just ended, else None.
        """
        if self._last_true is not None and timestamp_ms < self._last_true:
            self.reset()  # time went backwards: drop the episode
        if condition:
            if self._start is None:
                self._start = timestamp_ms
            elif timestamp_ms - self._last_true > self.max_gap_ms:
                ended = self.duration_s()
                self._start = timestamp_ms
                self._last_true = timestamp_ms
                return ended
            self._last_true = timestamp_ms
            return None
        if self._start is not None and timestamp_ms - self._last_true > self.max_gap_ms:
            ended = self.duration_s()
            self.reset()
            return ended
        return None

    def finish(self) -> float | None:
        """Force-end the current episode (if any) and return its duration."""
        if self._start is None:
            return None
        ended = self.duration_s()
        self.reset()
        return ended
