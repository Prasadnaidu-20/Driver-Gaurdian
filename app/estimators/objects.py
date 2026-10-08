"""Object detection (phone, cup, bottle) with YOLO and the activity rules built on it.

* `ObjectDetector` wraps an ultralytics YOLO model (COCO weights; ultralytics is AGPL-3.0).
  It has no per-stream state, so the loaded model is shared by all streams.
* `ActivityRules` turns detections into `phone_use` / `drinking` (pure logic, unit-tested).

There is no hand detector, so "near the face/hand region" is a region relative to the face:
the face box grown sideways (phone at the ear), slightly up and well down (texting at chest
height). Drinking needs a cup/bottle box touching a square around the mouth centre.
All geometry is done in pixels so the regions are not distorted by the frame aspect ratio.
"""

from __future__ import annotations

import logging
import os
import threading
from functools import lru_cache
from typing import Any

import numpy as np
from pydantic import BaseModel

from app.config import ObjectsSettings
from app.estimators.base import Activities
from app.temporal.episodes import EpisodeTracker

logger = logging.getLogger(__name__)

Box = tuple[float, float, float, float]  # x0, y0, x1, y1 in pixels


class Detection(BaseModel):
    """One detected object. `bbox` is [x, y, w, h], normalized to the frame (0-1)."""

    label: str       # COCO class name, e.g. "cell phone"
    kind: str        # "phone" | "drink"
    conf: float
    bbox: list[float]


# ---------- YOLO wrapper ----------


@lru_cache(maxsize=2)
def _load_yolo(path: str) -> Any:
    """Load YOLO once per process. Raises on failure (not cached, so a later stream retries)."""
    # Keep ultralytics from reaching the network (update checks, auto-installs) at runtime.
    os.environ.setdefault("YOLO_OFFLINE", "1")
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")
    os.environ.setdefault("YOLO_VERBOSE", "False")
    from ultralytics import YOLO  # heavy import (torch): only when objects are enabled

    return YOLO(path)


class ObjectDetector:
    """YOLO detector filtered to the configured phone / drink classes. Never raises from `detect`."""

    _lock = threading.Lock()  # the shared model is not guaranteed thread-safe across streams

    def __init__(self, cfg: ObjectsSettings) -> None:
        self.cfg = cfg
        self.available = False
        self._model: Any = None
        self._kinds: dict[int, tuple[str, str]] = {}  # class id -> (label, kind)
        path = cfg.resolved_model_path()
        if not path.is_file():
            # Checked here: ultralytics would otherwise try to download into the working directory.
            logger.error("YOLO weights not found at %s (run scripts/download_models.py): objects unavailable", path)
            return
        try:
            self._model = _load_yolo(str(path))
        except Exception as exc:
            logger.error("Could not load YOLO from %s: %s — objects unavailable", path, exc)
            return
        names: dict[int, str] = self._model.names
        wanted = {**{c: "phone" for c in cfg.phone_classes}, **{c: "drink" for c in cfg.drink_classes}}
        self._kinds = {i: (n, wanted[n]) for i, n in names.items() if n in wanted}
        missing = set(wanted) - {n for n, _ in self._kinds.values()}
        if missing:
            logger.warning("YOLO model has no classes %s", sorted(missing))
        self.available = bool(self._kinds)

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        """Phone / drink detections in one BGR frame (empty on error)."""
        if not self.available:
            return []
        h, w = frame_bgr.shape[:2]
        try:
            with self._lock:
                results = self._model.predict(
                    frame_bgr, imgsz=self.cfg.imgsz, conf=self.cfg.conf,
                    classes=list(self._kinds), verbose=False,
                )
        except Exception:
            logger.exception("YOLO inference failed")
            return []
        out: list[Detection] = []
        boxes = results[0].boxes
        for (x0, y0, x1, y1), cls, conf in zip(boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist()):
            label, kind = self._kinds[int(cls)]
            out.append(Detection(label=label, kind=kind, conf=float(conf),
                                 bbox=[x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h]))
        return out


# ---------- activity rules ----------


def _to_px(bbox: list[float], w: int, h: int) -> Box:
    x, y, bw, bh = bbox
    return x * w, y * h, (x + bw) * w, (y + bh) * h


def _overlaps(a: Box, b: Box) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


class ActivityRules:
    """Phone use and drinking from detections, relative to the face. One instance per stream.

    Feed it only frames where YOLO actually ran, so durations are based on real observations.
    """

    def __init__(self, cfg: ObjectsSettings) -> None:
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self._phone = EpisodeTracker(self.cfg.max_gap_s)
        self._drink = EpisodeTracker(self.cfg.max_gap_s)
        self._face: tuple[list[float], tuple[float, float] | None, float] | None = None  # bbox, mouth, ts
        self._last_ts: float | None = None
        self.detections: list[Detection] = []  # last detections after resolve_phone_at_mouth (for display)

    def phone_region(self, face_px: Box) -> Box:
        """Face box grown by the configured multiples of the face size."""
        x0, y0, x1, y1 = face_px
        fw, fh = x1 - x0, y1 - y0
        c = self.cfg
        return (x0 - c.phone_region_side * fw, y0 - c.phone_region_up * fh,
                x1 + c.phone_region_side * fw, y1 + c.phone_region_down * fh)

    def mouth_region(self, mouth_px: tuple[float, float], face_width_px: float) -> Box:
        r = self.cfg.mouth_region_scale * face_width_px
        mx, my = mouth_px
        return mx - r, my - r, mx + r, my + r

    def resolve_phone_at_mouth(self, detections: list[Detection], face_bbox: list[float] | None,
                               mouth: tuple[float, float] | None, frame_size: tuple[int, int]) -> list[Detection]:
        """Relabel "phone" boxes that cover the mouth centre as drinks.

        A hand-held glass at the mouth is often misread by YOLO as "cell phone". A real phone
        at the ear sits beside the face and one being texted on is below it, so neither covers the mouth.
        """
        if not self.cfg.phone_at_mouth_is_drink or face_bbox is None or mouth is None:
            return detections
        w, h = frame_size
        face_px = _to_px(face_bbox, w, h)
        r = self.cfg.mouth_point_margin * (face_px[2] - face_px[0])
        mx, my = mouth[0] * w, mouth[1] * h
        mouth_box = (mx - r, my - r, mx + r, my + r)
        return [
            d.model_copy(update={"kind": "drink", "label": f"drink ({d.label} at mouth)"})
            if d.kind == "phone" and _overlaps(_to_px(d.bbox, w, h), mouth_box) else d
            for d in detections
        ]

    def update(self, detections: list[Detection], face_bbox: list[float] | None,
               mouth: tuple[float, float] | None, frame_size: tuple[int, int], timestamp_ms: float) -> Activities:
        """One detection frame. `face_bbox` [x, y, w, h] and `mouth` (x, y) are normalized; None = no face."""
        if self._last_ts is not None and timestamp_ms < self._last_ts:
            self.reset()  # seek / replay
        self._last_ts = timestamp_ms
        cfg = self.cfg
        w, h = frame_size

        # A cup in front of the mouth can hide the face: keep using the last face for a while.
        if face_bbox is not None:
            self._face = (face_bbox, mouth, timestamp_ms)
        elif self._face is not None and timestamp_ms - self._face[2] <= cfg.face_hold_s * 1000.0:
            face_bbox, mouth = self._face[0], self._face[1]

        detections = self.resolve_phone_at_mouth(detections, face_bbox, mouth, frame_size)
        self.detections = detections
        phones = [_to_px(d.bbox, w, h) for d in detections if d.kind == "phone"]
        drinks = [_to_px(d.bbox, w, h) for d in detections if d.kind == "drink"]

        if face_bbox is None:
            # Looking down at a phone often loses the face: any phone in view counts.
            phone_near = bool(phones)
            drink_near = False
        else:
            face_px = _to_px(face_bbox, w, h)
            region = self.phone_region(face_px)
            phone_near = any(_overlaps(p, region) for p in phones)
            drink_near = False
            if mouth is not None:
                mouth_box = self.mouth_region((mouth[0] * w, mouth[1] * h), face_px[2] - face_px[0])
                drink_near = any(_overlaps(d, mouth_box) for d in drinks)

        self._phone.update(phone_near, timestamp_ms)
        self._drink.update(drink_near, timestamp_ms)
        phone_s = self._phone.duration_s() if self._phone.active else 0.0
        drink_s = self._drink.duration_s() if self._drink.active else 0.0
        phone_use = self._phone.active and phone_s >= cfg.phone_min_s
        drinking = self._drink.active and drink_s >= cfg.drink_min_s

        reasons: list[str] = []
        if phone_use:
            reasons.append(f"phone use {phone_s:.1f} s")
        if drinking:
            reasons.append(f"drinking {drink_s:.1f} s")
        return Activities(phone_use=phone_use, drinking=drinking, phone_s=phone_s,
                          drinking_s=drink_s, reasons=reasons)
