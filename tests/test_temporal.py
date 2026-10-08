import math

import pytest

from app.temporal.ema import EMA
from app.temporal.episodes import EpisodeTracker
from app.temporal.hysteresis import HysteresisStateMachine
from app.temporal.window import TimeWindow


# ---- TimeWindow ----

def test_window_time_fraction_and_pruning() -> None:
    w = TimeWindow(duration_s=10, max_dt_s=0.5)
    for i in range(300):  # 10 s at 30 fps, closed in the first 3 s
        w.add(i * 1000 / 30, 1.0 if i < 90 else 0.0)
    assert w.time_mean() == pytest.approx(0.3, abs=0.01)
    for i in range(300, 600):  # 10 more seconds open: the closed samples age out
        w.add(i * 1000 / 30, 0.0)
    assert w.time_mean() == 0.0
    assert w.span_s() == pytest.approx(10, abs=0.05)


def test_window_gaps_do_not_count_and_min_span() -> None:
    w = TimeWindow(duration_s=60, max_dt_s=0.5)
    w.add(0, 1.0)
    w.add(100, 1.0)
    w.add(5000, 0.0)  # 4.9 s gap counts only as max_dt (0.5 s)
    assert w.covered_s() == pytest.approx(0.6)
    assert w.time_mean() == pytest.approx(0.6 / 0.6)
    assert w.time_mean(min_span_s=30) == pytest.approx(0.6 / 30)


def test_window_ignores_non_increasing_timestamps() -> None:
    w = TimeWindow(10, 0.5)
    w.add(100, 1.0)
    w.add(100, 0.0)
    w.add(50, 0.0)
    assert len(w) == 1


# ---- EMA ----

def test_ema_time_constant() -> None:
    ema = EMA(tau_s=1.0)
    ema.update(0.0, 0)
    assert ema.update(1.0, 1000) == pytest.approx(1 - math.exp(-1))
    assert ema.update(5.0, 1000) == pytest.approx(1 - math.exp(-1))  # dt = 0: unchanged


def test_ema_is_frame_rate_independent() -> None:
    a, b = EMA(0.5), EMA(0.5)
    a.update(0, 0)
    b.update(0, 0)
    for i in range(1, 31):
        va = a.update(1.0, i * 1000 / 30)
    for i in range(1, 11):
        vb = b.update(1.0, i * 100)
    assert va == pytest.approx(vb, abs=1e-9)


# ---- EpisodeTracker ----

def test_episode_duration_and_end() -> None:
    ep = EpisodeTracker(max_gap_s=0.25)
    ended = []
    for i in range(61):  # true for 2 s at 30 fps
        r = ep.update(True, i * 1000 / 30)
        assert r is None
    assert ep.duration_s() == pytest.approx(2.0)
    t = 2000
    while True:
        t += 1000 / 30
        r = ep.update(False, t)
        if r is not None:
            ended.append(r)
            break
    assert ended == [pytest.approx(2.0)]
    assert not ep.active


def test_episode_survives_short_dropout() -> None:
    ep = EpisodeTracker(max_gap_s=0.25)
    ep.update(True, 0)
    ep.update(True, 100)
    ep.update(None, 133)  # one missing frame
    ep.update(False, 166)  # one misdetected frame
    ep.update(True, 200)
    assert ep.active and ep.duration_s() == pytest.approx(0.2)


# ---- Hysteresis ----

def _machine(min_enter=0.3, min_exit=2.0) -> HysteresisStateMachine:
    return HysteresisStateMachine(["low", "mid", "high"], [0.0, 0.3, 0.6], [0.0, 0.2, 0.45], min_enter, min_exit)


def test_hysteresis_enter_and_exit_delays() -> None:
    m = _machine()
    assert m.update(0.9, 0) == "low"
    assert m.update(0.9, 200) == "low"
    assert m.update(0.9, 300) == "high"      # held for min_enter_s
    assert m.update(0.0, 400) == "high"
    assert m.update(0.0, 2300) == "high"
    assert m.update(0.0, 2400) == "low"      # held for min_exit_s


def test_hysteresis_band_keeps_level() -> None:
    m = _machine(min_enter=0, min_exit=0)
    m.update(0.7, 0)
    assert m.state == "high"
    assert m.update(0.5, 10) == "high"  # between exit (0.45) and enter (0.6)
    assert m.update(0.3, 20) == "mid"   # below high exit, above mid exit
    assert m.update(0.25, 30) == "mid"
    assert m.update(0.1, 40) == "low"


def test_hysteresis_prevents_flicker() -> None:
    m = _machine()
    states = []
    # Score oscillates around the "mid" enter threshold every frame for 10 s.
    for i in range(300):
        score = 0.32 if i % 2 == 0 else 0.25
        states.append(m.update(score, i * 1000 / 30))
    changes = sum(1 for a, b in zip(states, states[1:]) if a != b)
    assert changes == 0  # never persists 0.3 s above the enter threshold → stays low
    # Noise inside the hysteresis band once in "mid" never drops it.
    m = _machine(min_enter=0.0)
    m.update(0.4, 0)
    assert m.state == "mid"
    states = [m.update(0.32 if i % 2 else 0.22, i * 33) for i in range(1, 300)]
    assert set(states) == {"mid"}


def test_hysteresis_short_spike_ignored_and_reset() -> None:
    m = _machine()
    m.update(1.0, 0)
    m.update(1.0, 200)
    m.update(0.0, 233)  # spike shorter than min_enter_s
    m.update(1.0, 266)
    assert m.update(1.0, 500) == "low"  # timer restarted at 266
    m.reset()
    assert m.state == "low"
