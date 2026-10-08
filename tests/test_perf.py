import pytest

from app.utils.perf import PerfTracker


def test_empty_tracker_defaults() -> None:
    perf = PerfTracker(10)
    assert perf.fps() == 0.0
    assert perf.percentile(50) == 0.0
    assert perf.snapshot() == {"fps": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}


def test_single_sample_has_no_fps() -> None:
    perf = PerfTracker(10)
    perf.add(5.0, now=1.0)
    assert perf.fps() == 0.0
    assert perf.percentile(50) == 5.0


def test_fps_from_synthetic_times() -> None:
    perf = PerfTracker(100)
    for i in range(31):  # 31 frames over exactly 1 s → 30 FPS
        perf.add(1.0, now=i / 30)
    assert perf.fps() == pytest.approx(30.0)


def test_percentiles() -> None:
    perf = PerfTracker(100)
    for i in range(1, 101):  # latencies 1..100 ms
        perf.add(float(i), now=float(i))
    assert perf.percentile(50) == pytest.approx(50.5)
    assert perf.percentile(95) == pytest.approx(95.05)


def test_window_evicts_old_samples() -> None:
    perf = PerfTracker(5)
    for i in range(5):
        perf.add(1000.0, now=float(i))  # slow, 1 FPS
    for i in range(5):
        perf.add(10.0, now=10 + i * 0.1)  # fast, 10 FPS — fully replaces the window
    assert len(perf) == 5
    assert perf.percentile(95) == pytest.approx(10.0)
    assert perf.fps() == pytest.approx(10.0)


def test_window_must_hold_two_samples() -> None:
    with pytest.raises(ValueError):
        PerfTracker(1)
