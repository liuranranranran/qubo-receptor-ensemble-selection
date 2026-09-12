"""Subset enumeration contracts (exhaustive, mask-stable, no approximation)."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from qubo_receptor_ensemble.headroom.fusion import FusionSpec, fit_fusion, FusionScorer
from qubo_receptor_ensemble.headroom.subsets import (
    SubsetScan,
    columns_mask,
    count_subsets,
    iter_combinations,
    iter_masks,
    iter_masks_up_to,
    mask_columns,
    oracle_scan,
    popcount,
    utility,
)


def test_mask_and_column_round_trip() -> None:
    for columns in ((0,), (0, 2), (1, 2, 3)):
        mask = columns_mask(columns, 5)
        assert popcount(mask) == len(columns)
        assert mask_columns(mask, 5) == columns
    with pytest.raises(ValueError):
        columns_mask((5,), 5)
    with pytest.raises(ValueError):
        mask_columns(1 << 6, 5)


def test_enumeration_covers_every_subset_exactly_once() -> None:
    receptors, k_max = 7, 4
    masks = list(iter_masks_up_to(receptors, k_max))
    assert len(masks) == count_subsets(receptors, k_max)
    assert len(set(masks)) == len(masks)
    assert count_subsets(receptors, k_max) == sum(
        math.comb(receptors, k) for k in range(1, k_max + 1)
    )
    for k in range(1, k_max + 1):
        expected = set(itertools.combinations(range(receptors), k))
        assert set(iter_combinations(receptors, k)) == expected
        assert len(list(iter_masks(receptors, k))) == len(expected)


def test_oracle_scan_finds_the_known_best_subset() -> None:
    # receptor 0 gives the actives the lowest (best) docking scores
    scores = np.asarray(
        [
            [-9.0, -6.0, -8.0],
            [-8.0, -6.5, -7.5],
            [-6.0, -9.0, -6.5],
            [-6.5, -8.5, -6.0],
        ]
    )
    labels = np.asarray([1.0, 1.0, 0.0, 0.0])
    spec = FusionSpec(name="mean")
    frozen = fit_fusion(spec, scores)
    scorer = FusionScorer(frozen, scores)
    scan = oracle_scan(scorer, 3, 1, labels, "pr_auc", 20.0)
    assert scan.best_columns == (0,)
    assert scan.n_subsets == 3
    assert scan.best_mask == columns_mask((0,), 3)
    assert scan.best_value == pytest.approx(1.0)


def test_utility_is_higher_is_better() -> None:
    # utility scores are already in ranking direction (higher = better)
    good = np.asarray([-5.0, -4.5, -9.0, -8.5])
    bad = np.asarray([-9.0, -8.5, -5.0, -4.5])
    labels = np.asarray([1.0, 1.0, 0.0, 0.0])
    assert utility(good, labels, "pr_auc") > utility(bad, labels, "pr_auc")


def test_scan_subset_columns_requires_subsets() -> None:
    from qubo_receptor_ensemble.headroom.subsets import scan_subset_columns

    spec = FusionSpec(name="mean")
    frozen = fit_fusion(spec, np.asarray([[-8.0, -7.0], [-6.0, -9.0]]))
    scorer = FusionScorer(frozen, np.asarray([[-8.0, -7.0], [-6.0, -9.0]]))
    with pytest.raises(ValueError):
        scan_subset_columns(scorer, [], np.asarray([1.0, 0.0]), "pr_auc")


def test_subset_scan_dataclass_serializes() -> None:
    scan = SubsetScan(n_subsets=3, best_value=0.5, best_columns=(1, 2), best_mask=6)
    payload = scan.as_dict()
    assert payload["best_columns"] == [1, 2]
    assert payload["best_mask"] == 6