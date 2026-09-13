"""Train-only features for the E2 allocation law (exploratory, no gate impact).

The parent plan's E2 deliverable is a *law*: how the critical budget ``B*``
relates to (single-receptor quality, pool complementarity, library size,
activity rate).  Everything here is computed from the training fold only
(labels allowed, held-out scores never), so the features can be reported next
to the observed ``B*`` without leaking the evaluation fold.
"""

from __future__ import annotations

import itertools
from typing import Mapping, Sequence

import numpy as np

from ..headroom.fusion import FrozenFusion, FusionScorer
from ..headroom.metrics_fast import average_ranks, metric_value
from .fusion_ragged import build_ragged_fusion, fuse_ragged

FEATURE_NAMES: tuple[str, ...] = (
    "best_single_train_pr_auc",
    "best_pair_train_pr_auc",
    "pair_gain_train_pr_auc",
    "receptor_diversity_spearman",
    "receptor_count",
    "ligand_count",
    "scaffold_count",
    "mean_scaffold_size",
    "train_activity_rate",
)

SCORE_FEATURES: tuple[str, ...] = (
    "best_single_train_pr_auc",
    "best_pair_train_pr_auc",
    "pair_gain_train_pr_auc",
    "receptor_diversity_spearman",
)

GAIN_FEATURES: tuple[str, ...] = (
    "best_gain_first_budget",
    "best_gain_max_budget",
)


def _rank_vector(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    order = np.argsort(vector, kind="stable")
    ranks = np.empty(vector.size, dtype=np.float64)
    ranks[order] = average_ranks(vector[order])
    return ranks


def receptor_diversity(train_scores: np.ndarray) -> float:
    """Mean pairwise Spearman correlation between receptor columns (train fold)."""
    matrix = np.asarray(train_scores, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        return float("nan")
    ranks = np.column_stack([_rank_vector(matrix[:, column]) for column in range(matrix.shape[1])])
    values: list[float] = []
    for left, right in itertools.combinations(range(matrix.shape[1]), 2):
        first = ranks[:, left]
        second = ranks[:, right]
        if np.std(first) <= 0 or np.std(second) <= 0:
            continue
        values.append(float(np.corrcoef(first, second)[0, 1]))
    return float(np.mean(values)) if values else float("nan")


def fold_train_features(
    frozen: FrozenFusion,
    train_scores: np.ndarray,
    train_labels: np.ndarray,
    metric: str = "pr_auc",
    alpha: float = 20.0,
) -> dict[str, float]:
    """Best-single / best-pair train utility and pool diversity of one fold.

    ``best_pair`` is the exact best fused pair under the frozen phi, so
    ``pair_gain = best_pair - best_single`` measures how much in-sample utility
    the pool's *complementarity* adds over the best single receptor.
    """
    scores = np.asarray(train_scores, dtype=np.float64)
    labels = np.asarray(train_labels, dtype=np.float64)
    scorer = FusionScorer(frozen, scores)
    single = [
        metric_value(scorer.score_columns((column,)), labels, metric, alpha)
        for column in range(scorer.n_receptors)
    ]
    best_single = float(np.nanmax(single))
    best_pair = float("nan")
    if scorer.n_receptors >= 2:
        ragged = build_ragged_fusion(frozen, scores)
        pair_values: list[float] = []
        for left, right in itertools.combinations(range(scorer.n_receptors), 2):
            mask = np.zeros(scores.shape, dtype=bool)
            mask[:, (left, right)] = True
            pair_values.append(float(metric_value(fuse_ragged(ragged, mask), labels, metric, alpha)))
        best_pair = float(np.nanmax(pair_values))
    return {
        "best_single_train_pr_auc": best_single,
        "best_pair_train_pr_auc": best_pair,
        "pair_gain_train_pr_auc": best_pair - best_single,
        "receptor_diversity_spearman": receptor_diversity(scores),
    }


def aggregate_fold_features(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Mean of each feature over folds (finite values only)."""
    output: dict[str, float] = {}
    for name in FEATURE_NAMES:
        values = np.asarray([float(row[name]) for row in rows if name in row], dtype=np.float64)
        finite = values[np.isfinite(values)]
        output[name] = float(finite.mean()) if finite.size else float("nan")
    return output


def spearman_rank_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Spearman correlation of two finite pairs (nan when fewer than three)."""
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if np.isfinite(x) and np.isfinite(y)]
    if len(pairs) < 3:
        return float("nan")
    first = _rank_vector(np.asarray([item[0] for item in pairs]))
    second = _rank_vector(np.asarray([item[1] for item in pairs]))
    if np.std(first) <= 0 or np.std(second) <= 0:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def build_law_summary(
    rows: Sequence[Mapping[str, object]],
    budgets: Sequence[int],
) -> dict[str, object]:
    """Exploratory B* vs feature summary (descriptive, not a gate)."""
    max_budget = int(max(budgets)) if budgets else 0
    switching = [row for row in rows if row.get("b_star") is not None]
    locked = [row for row in rows if row.get("b_star") is None]
    b_star_values = [
        float(row["b_star"]) if row.get("b_star") is not None else float(max_budget + 600)
        for row in rows
    ]
    correlations: dict[str, float] = {}
    for name in (*FEATURE_NAMES, *GAIN_FEATURES):
        if not any(name in row for row in rows):
            continue
        correlations[name] = spearman_rank_correlation(
            [float(row[name]) for row in rows if name in row],
            [value for row, value in zip(rows, b_star_values) if name in row],
        )
    gain_correlations: dict[str, float] = {}
    for gain in GAIN_FEATURES:
        if not any(gain in row for row in rows):
            continue
        for name in FEATURE_NAMES:
            gain_correlations[f"{gain}~{name}"] = spearman_rank_correlation(
                [float(row[name]) for row in rows if name in row and gain in row],
                [float(row[gain]) for row in rows if name in row and gain in row],
            )

    def group_means(group: Sequence[Mapping[str, object]]) -> dict[str, float]:
        output: dict[str, float] = {}
        for name in (*FEATURE_NAMES, *GAIN_FEATURES):
            values = np.asarray(
                [float(row[name]) for row in group if name in row and np.isfinite(float(row[name]))],
                dtype=np.float64,
            )
            output[name] = float(values.mean()) if values.size else float("nan")
        return output

    return {
        "schema": "e2_law_summary_v1",
        "note": (
            "exploratory descriptive summary of parent-plan section 3 E2 (law); "
            "features are train-fold only and do not enter gate G2"
        ),
        "b_star_convention": "None (width-locked) counted as max_budget + 600 for rank correlations",
        "n_rows": len(rows),
        "n_switching": len(switching),
        "n_width_locked": len(locked),
        "n_distinct_b_star": len({row.get("b_star") for row in rows}),
        "spearman_feature_vs_b_star_rank": correlations,
        "spearman_gain_vs_feature": gain_correlations,
        "group_means": {
            "width_locked": group_means(locked),
            "switching": group_means(switching),
        },
    }


__all__ = [
    "FEATURE_NAMES",
    "GAIN_FEATURES",
    "SCORE_FEATURES",
    "aggregate_fold_features",
    "build_law_summary",
    "fold_train_features",
    "receptor_diversity",
    "spearman_rank_correlation",
]
