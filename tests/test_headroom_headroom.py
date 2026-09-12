"""H_raw / H_nested / H_perm contracts and the A/B scaffold split."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qubo_receptor_ensemble.headroom.fusion import FusionSpec
from qubo_receptor_ensemble.headroom.headroom import (
    HeadroomConfig,
    headroom_nested,
    headroom_perm,
    headroom_raw,
    permutation_null,
    scan_shard,
    scaffold_ab_split,
    scaffold_allocation,
)
from qubo_receptor_ensemble.headroom.fusion import FusionScorer, fit_fusion


def synthetic_panel(seed: int = 0, ligands: int = 120, receptors: int = 6, folds: int = 4):
    rng = np.random.default_rng(seed)
    scores = rng.normal(-8.0, 1.2, size=(ligands, receptors))
    labels = np.zeros(ligands)
    actives = rng.choice(ligands, ligands // 5, replace=False)
    labels[actives] = 1.0
    # make the first receptor informative so the oracle is meaningful
    scores[actives, 0] -= 1.5
    fold_ids = np.asarray([index % folds for index in range(ligands)], dtype=np.int64)
    scaffolds = [f"scaffold_{index % 15:02d}" for index in range(ligands)]
    return scores, labels, fold_ids, scaffolds


def test_scaffold_split_is_deterministic_and_disjoint() -> None:
    _, labels, _, scaffolds = synthetic_panel()
    a_one, b_one = scaffold_ab_split(scaffolds, labels)
    a_two, b_two = scaffold_ab_split(scaffolds, labels)
    assert np.array_equal(a_one, a_two) and np.array_equal(b_one, b_two)
    assert set(a_one).isdisjoint(set(b_one))
    assert sorted(np.r_[a_one, b_one].tolist()) == list(range(len(scaffolds)))
    assert labels[a_one].sum() > 0 and labels[b_one].sum() > 0
    assert abs(len(a_one) - len(b_one)) <= 8  # at most one scaffold of imbalance


def test_scaffold_allocation_covers_every_row() -> None:
    _, _, _, scaffolds = synthetic_panel()
    assignment = scaffold_allocation(scaffolds, 3)
    assert assignment.shape == (len(scaffolds),)
    assert set(assignment.tolist()) == {0, 1, 2}
    # every scaffold lands in exactly one bucket
    for scaffold in set(scaffolds):
        buckets = {int(assignment[index]) for index, key in enumerate(scaffolds) if key == scaffold}
        assert len(buckets) == 1


def scan_one(config: HeadroomConfig | None = None, spec: FusionSpec | None = None) -> dict:
    scores, labels, fold_ids, scaffolds = synthetic_panel(seed=4)
    fold = 1
    train = np.flatnonzero(fold_ids != fold)
    test = np.flatnonzero(fold_ids == fold)
    return scan_shard(
        target_id="SYN",
        fold=fold,
        spec=spec or FusionSpec(name="mean"),
        train_scores=scores[train],
        train_labels=labels[train],
        train_scaffolds=[scaffolds[index] for index in train],
        test_scores=scores[test],
        test_labels=labels[test],
        test_scaffolds=[scaffolds[index] for index in test],
        config=config or HeadroomConfig(k_list=(1, 2, 3), top_m=50, permutations=20, perm_ks=(2,)),
    )


def test_scan_shard_exposes_all_three_definitions() -> None:
    payload = scan_one()
    assert payload["schema"] == "e1_shard_v1"
    assert payload["receptor_count"] == 6
    assert payload["a_size"] + payload["b_size"] == payload["test_ligand_count"]
    cells = {int(cell["k"]): cell for cell in payload["cells"]}
    assert set(cells) == {1, 2, 3}
    for k, cell in cells.items():
        assert cell["n_subsets"] == math.comb(6, k)
        assert cell["h_raw"] == pytest.approx(cell["u_test_oracle"] - cell["u_test_ref"])
        if cell["ref_source"] == "greedy":
            assert cell["h_raw"] >= -1e-12
        assert cell["h_nested"] == pytest.approx(0.5 * (cell["h_ab"] + cell["h_ba"]))
        assert cell["noise_floor"]["se"] >= 0.0
        assert set(cell["metric_oracle"]) == {"pr_auc", "bedroc20", "roc_auc", "ef1", "ef5", "ef10"}
    assert cells[2]["top_masks"] is not None
    assert cells[3]["top_masks"] is None
    assert len(cells[2]["top_masks"]) <= 50


def test_oracle_dominates_every_size_k_baseline() -> None:
    payload = scan_one()
    for cell in payload["cells"]:
        # the oracle enumerates all size-k subsets, so greedy (size k) can never win
        assert cell["u_test_oracle"] >= cell["u_test_greedy"] - 1e-12
        assert cell["u_test_oracle"] >= min(cell["u_test_greedy"], cell["u_test_single"]) - 1e-12
        if cell["ref_source"] == "greedy":
            assert cell["u_test_oracle"] >= cell["u_test_ref"] - 1e-12


def test_permutation_null_is_finite_and_correction_subtracts_q95() -> None:
    scores, labels, fold_ids, scaffolds = synthetic_panel(seed=6)
    fold = 1
    train = np.flatnonzero(fold_ids != fold)
    test = np.flatnonzero(fold_ids == fold)
    spec = FusionSpec(name="mean")
    config = HeadroomConfig(k_list=(2,), top_m=200, permutations=30)
    payload = scan_shard(
        target_id="SYN",
        fold=fold,
        spec=spec,
        train_scores=scores[train],
        train_labels=labels[train],
        train_scaffolds=[scaffolds[index] for index in train],
        test_scores=scores[test],
        test_labels=labels[test],
        test_scaffolds=[scaffolds[index] for index in test],
        config=config,
    )
    cell = payload["cells"][0]
    frozen = fit_fusion(spec, scores[train])
    scorer = FusionScorer(frozen, scores[test])
    null = permutation_null(scorer, cell["top_masks"], labels[test], config, seed=1)
    assert np.isfinite(null["q95"])
    assert null["n_subsets"] == len(cell["top_masks"])
    assert null["q50"] <= null["q95"]


def test_plan_level_wrappers_return_per_fold_per_k_records() -> None:
    scores, labels, fold_ids, scaffolds = synthetic_panel(seed=8)
    config = HeadroomConfig(k_list=(1, 2), top_m=20, permutations=5, perm_ks=(2,), bootstrap_iterations=20)
    raw = headroom_raw(
        scores, labels, fold_ids, FusionSpec(name="mean"), scaffolds=scaffolds, config=config
    )
    nested = headroom_nested(
        scores,
        labels,
        fold_ids,
        FusionSpec(name="mean"),
        split_key="scaffold",
        scaffolds=scaffolds,
        config=config,
    )
    perm = headroom_perm(
        scores,
        labels,
        fold_ids,
        FusionSpec(name="mean"),
        scaffolds=scaffolds,
        config=config,
        top_m=20,
        n_perm=5,
    )
    assert set(raw) == {0, 1, 2, 3}
    assert set(raw[0]) == {1, 2}
    assert raw[0][2]["h_raw"] == pytest.approx(nested[0][2]["h_raw"])
    assert perm[0][2]["h_perm"] is not None