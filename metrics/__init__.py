"""Metrics for DACON 236749 — deep voice detection.

The official metric lives in `dacon`; nothing else may compute EER.
See docs/validation/02-metric-harness.md for the metric register.
"""

from metrics.dacon import MetricSet, auc, dacon_score, eer, roll_up

__all__ = ["MetricSet", "auc", "dacon_score", "eer", "roll_up"]
