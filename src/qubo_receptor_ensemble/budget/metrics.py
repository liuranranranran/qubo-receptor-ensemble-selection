"""Budget-aware screening metrics: undocked ligands are ranked last.

Docked ligands are scored by the ragged fusion; ligands that never got a cell
keep ``-inf`` so they can never enter the top of the ranking (you cannot find
what you did not dock).  This is the screening convention used by E2.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ..headroom.metrics_fast import METRIC_NAMES, metric_values, positions, ranking_order

BUDGET_METRICS: tuple[str, ...] = (
    "pr_auc",
    "bedroc20",
    "recall1",
    "recall5",
    "recall10",
    "ef1",
    "ef5",
    "ef10",
)

_RECALL_FRACTIONS: dict[str, float] = {"recall1": 0.01, "recall5": 0.05, "recall10": 0.10}


def recall_at_fraction(scores: np.ndarray, labels: np.ndarray, fraction: float) -> float:
    """Fraction of actives inside the top ``fraction`` of the full ranking."""
    values = np.asarray(scores, dtype=np.float64)
    active = np.asarray(labels) > 0.5
    active_total = int(active.sum())
    if values.size == 0 or active_total == 0:
        return float("nan")
    top_n = max(1, int(np.ceil(values.size * fraction)))
    order = ranking_order(values)
    return float(active[order[:top_n]].sum()) / active_total


def budget_metric_values(
    scores: np.ndarray,
    labels: np.ndarray,
    metrics: Sequence[str] = BUDGET_METRICS,
    alpha: float = 20.0,
) -> dict[str, float]:
    """All requested budget metrics for one ranking-score vector."""
    requested = tuple(metrics)
    screening = [name for name in requested if name in METRIC_NAMES]
    values: dict[str, float] = {}
    if screening:
        values.update(metric_values(scores, labels, screening, alpha))
    for name in requested:
        if name in _RECALL_FRACTIONS:
            values[name] = recall_at_fraction(scores, labels, _RECALL_FRACTIONS[name])
    unknown = [name for name in requested if name not in values]
    if unknown:
        raise ValueError(f"unknown budget metrics: {unknown}")
    return values
