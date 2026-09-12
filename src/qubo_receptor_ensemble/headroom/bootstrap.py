"""Scaffold-cluster bootstrap, evaluation noise floor and MDE.

The pre-registered E1 rule is: paired unit = ``greedy`` vs ``single`` on the
same ``(target, fold, phi, k)``; resampling = whole scaffold clusters with
replacement (keeps within-cluster correlation); ``noise_floor = SE``;
``MDE = t(0.975, df) * SE * sqrt(2)`` for the paired design.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from .metrics_fast import metric_value

NORMAL_975 = 1.959963984540054


class BootstrapError(ValueError):
    """Raised when a bootstrap request violates its contract."""


def cluster_index(clusters: Sequence[str]) -> tuple[list[str], tuple[np.ndarray, ...]]:
    """Stable cluster labels and the row indices of each cluster."""
    if not clusters:
        raise BootstrapError("clusters must not be empty")
    order: list[str] = []
    groups: dict[str, list[int]] = {}
    for index, cluster in enumerate(clusters):
        key = str(cluster)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(index)
    return order, tuple(np.asarray(groups[key], dtype=np.int64) for key in order)


def bootstrap_resample_indices(
    cluster_rows: Sequence[np.ndarray],
    iterations: int,
    seed: int,
) -> list[np.ndarray]:
    """Whole-cluster resamples of the ligand row indices."""
    if iterations <= 0:
        raise BootstrapError("iterations must be positive")
    rng = np.random.default_rng(seed)
    n_clusters = len(cluster_rows)
    resamples: list[np.ndarray] = []
    for _ in range(iterations):
        draw = rng.integers(0, n_clusters, size=n_clusters)
        resamples.append(np.concatenate([cluster_rows[index] for index in draw]))
    return resamples


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile over the finite values."""
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(finite, q * 100.0))


def t_quantile_975(degrees_of_freedom: int | float) -> float:
    """Student-t 0.975 quantile; falls back to the normal quantile."""
    try:  # scipy ships with the project's sklearn dependency but is optional here.
        from scipy.stats import t as student_t

        if degrees_of_freedom and np.isfinite(degrees_of_freedom) and degrees_of_freedom > 0:
            return float(student_t.ppf(0.975, degrees_of_freedom))
    except Exception:  # pragma: no cover - optional dependency path
        pass
    return NORMAL_975


def minimum_detectable_effect(standard_error: float, degrees_of_freedom: int | float) -> float:
    """``t(0.975, df) * SE * sqrt(2)`` (paired design)."""
    if not np.isfinite(standard_error) or standard_error < 0:
        return float("nan")
    return t_quantile_975(degrees_of_freedom) * standard_error * np.sqrt(2.0)


def cluster_bootstrap(
    statistic: Callable[[np.ndarray], float],
    clusters: Sequence[str],
    iterations: int = 2000,
    seed: int = 0,
) -> dict[str, object]:
    """Bootstrap one ligand-level statistic by resampling whole clusters."""
    _, cluster_rows = cluster_index(clusters)
    resamples = bootstrap_resample_indices(cluster_rows, iterations, seed)
    values: list[float] = []
    skipped = 0
    for rows in resamples:
        value = statistic(rows)
        if value is None or not np.isfinite(value):
            skipped += 1
            continue
        values.append(float(value))
    return summarize_bootstrap(
        values,
        skipped=skipped,
        iterations=iterations,
        unit="scaffold_cluster",
        n_clusters=len(cluster_rows),
    )


def summarize_bootstrap(
    values: Sequence[float],
    *,
    skipped: int,
    iterations: int,
    unit: str,
    n_clusters: int,
) -> dict[str, object]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {
            "mean": float("nan"),
            "se": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "mde": float("nan"),
            "unit": unit,
            "iterations": int(iterations),
            "n_used": 0,
            "n_skipped": int(skipped),
            "n_clusters": int(n_clusters),
            "degrees_of_freedom": int(n_clusters - 1),
        }
    standard_error = float(array.std(ddof=1)) if array.size > 1 else float("nan")
    degrees_of_freedom = max(int(n_clusters - 1), 1)
    return {
        "mean": float(array.mean()),
        "se": standard_error,
        "ci95_low": percentile(array, 0.025),
        "ci95_high": percentile(array, 0.975),
        "mde": minimum_detectable_effect(standard_error, degrees_of_freedom),
        "unit": unit,
        "iterations": int(iterations),
        "n_used": int(array.size),
        "n_skipped": int(skipped),
        "n_clusters": int(n_clusters),
        "degrees_of_freedom": degrees_of_freedom,
    }


def cluster_bootstrap_delta(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    labels: np.ndarray,
    clusters: Sequence[str],
    metric: str = "pr_auc",
    alpha: float = 20.0,
    ligand_rank: np.ndarray | None = None,
    iterations: int = 2000,
    seed: int = 0,
) -> dict[str, object]:
    """Cluster bootstrap of ``U(a) - U(b)`` on one ligand panel."""

    def statistic(rows: np.ndarray) -> float:
        rank = None if ligand_rank is None else np.asarray(ligand_rank)[rows]
        value_a = metric_value(scores_a[rows], labels[rows], metric, alpha, rank)
        value_b = metric_value(scores_b[rows], labels[rows], metric, alpha, rank)
        return value_a - value_b

    report = cluster_bootstrap(statistic, clusters, iterations=iterations, seed=seed)
    report["statistic"] = "metric(a) - metric(b)"
    report["metric"] = metric
    return report