"""Estimator factory: picks implementations by `settings.mode`."""

from __future__ import annotations

import logging

from app.config import Settings
from app.estimators.base import Estimator
from app.pipeline.calibration import Baseline

logger = logging.getLogger(__name__)


def build_estimators(settings: Settings, baseline: Baseline) -> dict[str, Estimator]:
    """Enabled estimators keyed by task name ("drowsiness", "distraction", "emotion").

    Only `rules` exists so far; `ml` / `hybrid` (Phase 8) fall back to rules with a warning.
    Emotion is always the pretrained model, whatever the mode. If its model cannot be loaded
    it is left out, so `FrameResult.emotion` stays None ("unavailable").
    """
    from app.estimators.distraction_rules import DistractionRules
    from app.estimators.drowsiness_rules import DrowsinessRules
    from app.estimators.emotion import EmotionEstimator, EmotionModel

    if settings.mode != "rules":
        logger.warning("mode=%s is not implemented yet; using rule-based estimators", settings.mode)
    estimators: dict[str, Estimator] = {}
    if settings.drowsiness.enabled:
        estimators["drowsiness"] = DrowsinessRules(settings.drowsiness, baseline)
    if settings.distraction.enabled:
        estimators["distraction"] = DistractionRules(settings.distraction, baseline)
    if settings.emotion.enabled:
        model = EmotionModel(settings.emotion)
        if model.available:
            estimators["emotion"] = EmotionEstimator(settings.emotion, model)
    return estimators
