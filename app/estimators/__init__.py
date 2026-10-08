"""Estimator factory: picks implementations by `settings.mode`."""

from __future__ import annotations

import logging

from app.config import Settings
from app.estimators.base import Estimator
from app.pipeline.calibration import Baseline

logger = logging.getLogger(__name__)


def build_estimators(settings: Settings, baseline: Baseline) -> dict[str, Estimator]:
    """Enabled estimators keyed by task name ("drowsiness", "distraction").

    Only `rules` exists so far; `ml` / `hybrid` (Phase 8) fall back to rules with a warning.
    """
    from app.estimators.distraction_rules import DistractionRules
    from app.estimators.drowsiness_rules import DrowsinessRules

    if settings.mode != "rules":
        logger.warning("mode=%s is not implemented yet; using rule-based estimators", settings.mode)
    estimators: dict[str, Estimator] = {}
    if settings.drowsiness.enabled:
        estimators["drowsiness"] = DrowsinessRules(settings.drowsiness, baseline)
    if settings.distraction.enabled:
        estimators["distraction"] = DistractionRules(settings.distraction, baseline)
    return estimators
