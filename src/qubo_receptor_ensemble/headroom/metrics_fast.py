"""O(n) screening metrics numerically identical to ``screening.py``.

Every implementation is a numpy re-derivation of the Python loops in
:mod:`qubo_receptor_ensemble.screening`, with the same ranking convention:

- rows are ranked by ``(ranking_score descending, ligand_id ascending)`` where
  ``ranking_score = -docking_score``;
- BEDROC/PR-AUC/EF use the 1-based rank positions of that total order;
- ROC-AUC uses the pairwise (tie-corrected) definition.

The parity contract is checked by ``tests/test_headroom_metrics_parity.py``
against ``screening.ranked_metrics_with_ids`` (tolerance 1e-12).
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

METRIC_NAMES: tuple[str, ...] = (
    "pr_auc",
    "bedroc20",
    "roc_auc",
    "ef1",
    "ef5",
    "ef10",
)

_EF_FRACTIONS: dict[str, float] = {"ef1": 0.01, "ef5": 0.05, "ef10": 0.10}

_SCREENING_NAMES: dict[str, str] = {
    "pr_auc": "pr_auc_average_precision",
    "bedroc20": "bedroc_alpha_20",
    "roc_auc": "roc_auc",
    "ef1": "EF1%",
    "ef5": "EF5%",
    "ef10": "EF10%",
}


class MetricError(ValueError):
    """Raised when a metric request violates its contract."""


def positions(n: int) -> np.ndarray:
    """1-based rank positions ``1..n`` as float64."""
    return np.arange(1, n + 1, dtype=np.float64)


def ranking_order(
    scores: np.ndarray,
    ligand_rank: np.ndarray | None = None,
) -> np.ndarray:
    """Indices of ``scores`` sorted by ``(-score, ligand_id ascending)``.

    ``scores`` must already be ranking scores (higher-is-better).  When
    ``ligand_rank`` is given it is the ascending order key of the ligand IDs;
    otherwise a stable argsort is used, which matches the screening tie-break
    when rows are pre-sorted by ligand ID.
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1:
        raise MetricError("scores must be a 1-D array")
    if values.size == 0:
        raise MetricError("scores must not be empty")
    if ligand_rank is None:
        return np.argsort(-values, kind="stable")
    keys = np.asarray(ligand_rank)
    if keys.shape != values.shape:
        raise MetricError("ligand_rank must match the score vector shape")
    return np.lexsort((keys, -values))


def _labels_array(labels: np.ndarray) -> np.ndarray:
    values = np.asarray(labels)
    if values.ndim != 1:
        raise MetricError("labels must be a 1-D array")
    return values


def _active_mask(labels: np.ndarray) -> np.ndarray:
    return _labels_array(labels) > 0.5


def bedroc_from_order(order: np.ndarray, labels: np.ndarray, alpha: float) -> float:
    """BEDROC (finite-rank normalized RIE) of one ranking order."""
    n = int(order.size)
    active = _active_mask(labels)[order]
    active_total = int(active.sum())
    if n == 0 or active_total == 0 or active_total == n:
        return float("nan")
    weights = np.exp(-alpha * positions(n) / n)
    random_expected = active_total * float(weights.mean())
    observed = float(weights[active].sum()) / random_expected
    max_rie = float(weights[:active_total].sum()) / random_expected
    min_rie = float(weights[n - active_total:].sum()) / random_expected
    if max_rie == min_rie:
        return float("nan")
    return (observed - min_rie) / (max_rie - min_rie)


def pr_auc_from_order(order: np.ndarray, labels: np.ndarray) -> float:
    """Average precision (PR-AUC) of one ranking order."""
    n = int(order.size)
    active = _active_mask(labels)[order]
    active_total = int(active.sum())
    if n == 0 or active_total == 0:
        return float("nan")
    cumulative = np.cumsum(active)
    precision_sum = float((cumulative[active] / positions(n)[active]).sum())
    return precision_sum / active_total


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Tie-averaged 1-based ranks of an already sorted 1-D array."""
    n = values.size
    if n == 0:
        return np.empty(0, dtype=np.float64)
    boundaries = np.flatnonzero(np.r_[True, values[1:] != values[:-1]])
    ends = np.r_[boundaries[1:], n]
    averages = (boundaries + 1 + ends) / 2.0
    ranks = np.empty(n, dtype=np.float64)
    for start, end, average in zip(boundaries, ends, averages):
        ranks[start:end] = average
    return ranks


def roc_auc_pairwise(scores: np.ndarray, labels: np.ndarray) -> float:
    """Pairwise ROC-AUC with 0.5 credit for ties (screening definition)."""
    values = np.asarray(scores, dtype=np.float64)
    active = _active_mask(labels)
    n_active = int(active.sum())
    n_decoy = int(active.size - n_active)
    if n_active == 0 or n_decoy == 0:
        return float("nan")
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = average_ranks(values[order])
    positive_rank_sum = float(ranks[active].sum())
    return (positive_rank_sum - n_active * (n_active + 1) / 2.0) / (n_active * n_decoy)


def enrichment_factor_from_order(
    order: np.ndarray, labels: np.ndarray, fraction: float
) -> float:
    """Enrichment factor at ``fraction`` of the ranked list."""
    n = int(order.size)
    active = _active_mask(labels)
    active_total = int(active.sum())
    if n == 0 or active_total == 0:
        return float("nan")
    top_n = max(1, int(np.ceil(n * fraction)))
    top_active = int(active[order[:top_n]].sum())
    return (top_active / top_n) / (active_total / n)


def _resolve_metric(metric: str) -> str:
    if metric in METRIC_NAMES:
        return metric
    for short, screening_name in _SCREENING_NAMES.items():
        if metric == screening_name:
            return short
    raise MetricError(f"unknown metric: {metric}")


def metric_value(
    scores: np.ndarray,
    labels: np.ndarray,
    metric: str = "pr_auc",
    alpha: float = 20.0,
    ligand_rank: np.ndarray | None = None,
) -> float:
    """Scalar metric of a ranking-score vector (hot path for subset enumeration)."""
    resolved = _resolve_metric(metric)
    if resolved == "roc_auc":
        return roc_auc_pairwise(scores, labels)
    order = ranking_order(scores, ligand_rank)
    if resolved == "pr_auc":
        return pr_auc_from_order(order, labels)
    if resolved == "bedroc20":
        return bedroc_from_order(order, labels, alpha)
    return enrichment_factor_from_order(order, labels, _EF_FRACTIONS[resolved])


def metric_values(
    scores: np.ndarray,
    labels: np.ndarray,
    metrics: Sequence[str] = METRIC_NAMES,
    alpha: float = 20.0,
    ligand_rank: np.ndarray | None = None,
) -> dict[str, float]:
    """All requested metrics of one ranking-score vector."""
    requested = tuple(_resolve_metric(metric) for metric in metrics)
    order = ranking_order(scores, ligand_rank)
    values: dict[str, float] = {}
    if "roc_auc" in requested:
        values["roc_auc"] = roc_auc_pairwise(scores, labels)
    if "pr_auc" in requested:
        values["pr_auc"] = pr_auc_from_order(order, labels)
    if "bedroc20" in requested:
        values["bedroc20"] = bedroc_from_order(order, labels, alpha)
    for name in ("ef1", "ef5", "ef10"):
        if name in requested:
            values[name] = enrichment_factor_from_order(order, labels, _EF_FRACTIONS[name])
    return values


def screening_key_map() -> Mapping[str, str]:
    """Short product names mapped to the canonical ``screening.py`` keys."""
    return dict(_SCREENING_NAMES)