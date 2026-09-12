"""T1: ``metrics_fast`` must match ``screening.py`` numerically (tolerance 1e-12)."""

from __future__ import annotations

import numpy as np

from qubo_receptor_ensemble.headroom.metrics_fast import metric_values, screening_key_map
from qubo_receptor_ensemble.screening import ranked_metrics_with_ids

TOLERANCE = 1e-12


def screening_reference(scores: np.ndarray, labels: np.ndarray, ligand_ids) -> dict[str, float]:
    data = {
        str(ligand_id): {
            "label": "active" if float(label) > 0.5 else "decoy",
            "score": float(-score),  # screening ranks by ascending docking score
        }
        for ligand_id, label, score in zip(ligand_ids, labels, scores)
    }
    return dict(ranked_metrics_with_ids(data, score_key="score", bedroc_alpha=20.0))


def test_metrics_match_screening_on_two_hundred_random_vectors() -> None:
    rng = np.random.default_rng(20260911)
    mapping = screening_key_map()
    worst = 0.0
    for trial in range(200):
        size = int(rng.integers(40, 400))
        scores = rng.normal(-8.0, 1.5, size=size)
        labels = (rng.random(size) < float(rng.uniform(0.05, 0.45))).astype(np.float64)
        if labels.sum() == 0 or labels.sum() == size:
            np.put(labels, rng.integers(0, size), 1.0)
        ligand_ids = [f"L{index:05d}" for index in range(size)]
        reference = screening_reference(scores, labels, ligand_ids)
        mine = metric_values(scores, labels, alpha=20.0)
        for short, long_name in mapping.items():
            worst = max(worst, abs(float(reference[long_name]) - float(mine[short])))
    assert worst < TOLERANCE


def test_tie_break_matches_screening_with_many_exact_ties() -> None:
    rng = np.random.default_rng(7)
    size = 240
    scores = np.round(rng.normal(-8.0, 1.0, size=size), 1)  # deliberate ties
    labels = np.zeros(size, dtype=np.float64)
    labels[rng.choice(size, 48, replace=False)] = 1.0
    ligand_ids = [f"LIG{index:04d}" for index in range(size)]
    reference = screening_reference(scores, labels, ligand_ids)
    mine = metric_values(scores, labels, alpha=20.0)
    mapping = screening_key_map()
    for short, long_name in mapping.items():
        assert abs(float(reference[long_name]) - float(mine[short])) < TOLERANCE


def test_ligand_id_tie_break_is_honoured() -> None:
    """Equal scores must rank by ascending ligand id, as screening.py does."""

    scores = np.asarray([-8.0, -8.0, -7.0, -9.0], dtype=np.float64)
    labels = np.asarray([1.0, 0.0, 1.0, 0.0])
    unordered_ids = ["d", "a", "c", "b"]
    order = np.argsort(unordered_ids, kind="stable")  # ligand-id ascending
    mine = metric_values(scores[order], labels[order], alpha=20.0)
    reference = screening_reference(
        scores[order], labels[order], [unordered_ids[index] for index in order]
    )
    assert abs(reference["pr_auc_average_precision"] - mine["pr_auc"]) < TOLERANCE