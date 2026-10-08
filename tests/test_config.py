from fastapi.testclient import TestClient

from app.config import load_settings

SECTIONS = [
    "server", "stream", "face", "calibration", "drowsiness", "distraction",
    "objects", "emotion", "risk", "alerts", "logging", "mode",
]


def test_default_config_loads_with_all_sections() -> None:
    settings = load_settings()
    for section in SECTIONS:
        assert hasattr(settings, section)
    assert settings.mode in ("rules", "ml", "hybrid")
    assert settings.stream.frame_width == 640


def test_health_endpoint() -> None:
    from app.main import app

    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}
    page = client.get("/")
    assert page.status_code == 200
    assert "<title>DriveGuardian</title>" in page.text
