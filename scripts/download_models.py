"""Download model weights into models_store/.

Usage:
    python scripts/download_models.py           # download missing models
    python scripts/download_models.py --force   # re-download everything

URLs and destination paths come from config/default.yaml. Later phases add their
weights to `model_specs()`.
"""

from __future__ import annotations

import argparse
import logging
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings, load_settings  # noqa: E402

logger = logging.getLogger("download_models")


def model_specs(settings: Settings) -> list[tuple[str, str, Path]]:
    """(name, url, destination) for every downloadable model."""
    return [
        ("face_landmarker", settings.face.model_url, settings.face.resolved_model_path()),
    ]


def download(url: str, dest: Path, force: bool = False) -> bool:
    """Download `url` to `dest` via a .part file. Returns False if skipped because it exists."""
    if dest.exists() and not force:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
        while chunk := resp.read(1 << 16):
            f.write(chunk)
    tmp.replace(dest)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="re-download even if the file exists")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    failed = 0
    for name, url, dest in model_specs(load_settings()):
        try:
            if download(url, dest, args.force):
                logger.info("%s: downloaded %s (%.1f MB)", name, dest, dest.stat().st_size / 1e6)
            else:
                logger.info("%s: already present at %s (use --force to re-download)", name, dest)
        except Exception as exc:  # network errors, HTTP errors, disk errors
            logger.error("%s: download from %s failed: %s", name, url, exc)
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
