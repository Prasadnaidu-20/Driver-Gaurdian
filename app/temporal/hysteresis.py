"""Multi-level hysteresis state machine with minimum persistence before switching."""

from __future__ import annotations


class HysteresisStateMachine:
    """Maps a score stream to ordered levels without flicker.

    `levels[0]` is the resting level. The machine wants level *i* when the score is at
    least `enter[i]`. While it is in level *i*, it stays there until the score drops
    below `exit[i]` (with exit[i] < enter[i]). A change only happens once the wanted
    level has persisted for `min_enter_s` (going up) or `min_exit_s` (going down),
    measured in frame time. If the wanted level changes before then, the timer restarts.
    """

    def __init__(
        self,
        levels: list[str],
        enter: list[float],
        exit: list[float],
        min_enter_s: float,
        min_exit_s: float,
    ) -> None:
        if not (len(levels) == len(enter) == len(exit)):
            raise ValueError("levels, enter and exit must have the same length")
        self.levels = list(levels)
        self.enter = list(enter)
        self.exit = list(exit)
        self.min_enter_ms = min_enter_s * 1000.0
        self.min_exit_ms = min_exit_s * 1000.0
        self.reset()

    def reset(self) -> None:
        self.index = 0
        self._candidate: int | None = None
        self._candidate_since: float | None = None

    @property
    def state(self) -> str:
        return self.levels[self.index]

    def target(self, score: float) -> int:
        """Level the score asks for, given the current level (pure, no timing)."""
        target = self.index
        # Escalate to the highest level whose enter threshold is reached.
        for i in range(len(self.levels) - 1, self.index, -1):
            if score >= self.enter[i]:
                return i
        # De-escalate while the score is below the current level's exit threshold.
        while target > 0 and score < self.exit[target]:
            target -= 1
        return target

    def update(self, score: float, timestamp_ms: float) -> str:
        """Feed one score and return the (possibly switched) level."""
        target = self.target(score)
        if target == self.index:
            self._candidate = None
            self._candidate_since = None
            return self.state
        if target != self._candidate or self._candidate_since is None or timestamp_ms < self._candidate_since:
            self._candidate = target
            self._candidate_since = timestamp_ms
        hold_ms = self.min_enter_ms if target > self.index else self.min_exit_ms
        if timestamp_ms - self._candidate_since >= hold_ms:
            self.index = target
            self._candidate = None
            self._candidate_since = None
        return self.state
