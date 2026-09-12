"""Cluster bootstrap, noise floor and MDE contracts."""

from __future__ import annotations

import numpy as np
import pytest

from qubo_receptor_ensemble.headroom.bootstrap import (
    cluster_bootstrap,
    cluster_bootstrap_delta,
    cluster_index,
    minimum_detectable_effect,
    percentile,
    t_quantile_975,
)


def toy_panel(seed: int = 0, ligands: int = 90):
    rng = np.random.default_rng(seed)
    scores = rng.normal(-8.0, 1.0, size=ligands)
    labels = np.zeros(ligands)
    labels[rng.choice(ligands, ligands // 4, replace=False)] = 1.0
    clusters = [f"scaffold_{index % 12:02d}" for index in range(ligands)]
    return scores, labels, clusters


def test_cluster_index_groups_rows_stably() -> None:
    names, rows = cluster_index(["a", "b", "a", "c"])
    assert names == ["a", "b", "c"]
    assert [row.tolist() for row in rows] == [[0, 2], [1], [3]]


def test_bootstrap_is_reproducible_for_a_fixed_seed() -> None:
    scores, labels, clusters = toy_panel()
    first = cluster_bootstrap_delta(scores, -scores, labels, clusters, iterations=40, seed=3)
    second = cluster_bootstrap_delta(scores, -scores, labels, clusters, iterations=40, seed=3)
    assert first == second
    other = cluster_bootstrap_delta(scores, -scores, labels, clusters, iterations=40, seed=4)
    assert other["mean"] != first["mean"]


def test_mde_rule_is_t975_standard_error_sqrt_two() -> None:
    scores, labels, clusters = toy_panel(seed=2)
    report = cluster_bootstrap_delta(scores, -scores, labels, clusters, iterations=200, seed=0)
    expected = t_quantile_975(report["degrees_of_freedom"]) * report["se"] * np.sqrt(2.0)
    assert report["mde"] == pytest.approx(expected)
    assert report["ci95_low"] <= report["mean"] <= report["ci95_high"]


def test_zero_delta_gives_zero_noise_floor() -> None:
    scores, labels, clusters = toy_panel(seed=5)
    report = cluster_bootstrap_delta(scores, scores, labels, clusters, iterations=50, seed=0)
    assert report["mean"] == pytest.approx(0.0)
    assert report["se"] == pytest.approx(0.0)
    assert minimum_detectable_effect(0.0, 10) == pytest.approx(0.0)


def test_percentile_is_monotone() -> None:
    values = list(np.linspace(-1.0, 1.0, 21))
    assert percentile(values, 0.025) < percentile(values, 0.5) < percentile(values, 0.975)


def test_cluster_bootstrap_skips_degenerate_resamples() -> None:
    scores = np.asarray([0.0, 1.0, 0.0, 1.0])
    labels = np.asarray([1.0, 1.0, 0.0, 0.0])
    clusters = ["a", "a", "b", "b"]

    def statistic(rows: np.ndarray) -> float:
        subset = labels[rows]
        return 0.0 if subset.sum() in (0, subset.size) else 1.0

    report = cluster_bootstrap(statistic, clusters, iterations=20, seed=0)
    assert report["n_used"] + report["n_skipped"] == 20