"""T2: protocol parity against the golden V5 nested outer evaluation.

The fuse of E1.  Two anchors are checked on the real MK14 panel:

1. the current golden code path (``scripts/nested_outer_k_evaluation.py``:
   ``solve_subset`` + ``subset_metrics``) must agree with the fast path to
   better than 1e-9;
2. the selected fixed-k subsets must match the archived V5 decision log
   exactly (``tests/fixtures/e1/mk14_v5_fixed_k.csv``).

The archived ``folds_long.csv`` values are rounded to six decimals and are
kept only as a sanity bound (tolerance 1e-5); they show a legacy offset of up
to 1.74e-6 on MK14 that a fresh golden re-run does not reproduce, so the hard
criterion is the re-run, not the rounding.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import pytest

from qubo_receptor_ensemble.headroom.fusion import FusionScorer, FusionSpec, fit_fusion
from qubo_receptor_ensemble.headroom.metrics_fast import metric_value

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "e1" / "mk14_v5_fixed_k.csv"


def first_file(env_name: str, *candidates: str) -> Path | None:
    """First existing path, honouring an explicit environment override."""
    override = os.environ.get(env_name, "").strip()
    if override:
        path = Path(override)
        return path if path.is_file() else None
    for value in candidates:
        path = Path(value)
        if path.is_file():
            return path
    return None


MK14_MATRIX = first_file(
    "E1_MK14_MATRIX",
    r"E:\Quant\remote_runs\mk14_adaptive_remote\matrices\primary_median_matrix.csv",
    "/root/autodl-tmp/qubo_data_root/results/runs/mk14_adaptive_remote/matrices/primary_median_matrix.csv",
)
MK14_MANIFEST = first_file(
    "E1_MK14_MANIFEST",
    r"E:\Quant\remote_runs\mk14_adaptive_remote\prepared_ligands.csv",
    "/root/autodl-tmp/qubo_data_root/results/runs/mk14_adaptive_remote/prepared_ligands.csv",
)

pytestmark = pytest.mark.skipif(
    MK14_MATRIX is None or MK14_MANIFEST is None,
    reason="MK14 canonical carrier not found (set E1_MK14_MATRIX / E1_MK14_MANIFEST)",
)


def load_fixture() -> dict[tuple[int, int], dict[str, object]]:
    rows: dict[tuple[int, int], dict[str, object]] = {}
    with FIXTURE.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[(int(row["outer_fold"]), int(row["k"]))] = {
                "subset": row["subset"].split("+"),
                "archived": float(row["archived_bedroc20"]),
            }
    return rows


def test_fixed_k_values_match_the_current_golden_protocol() -> None:
    from scripts.nested_outer_k_evaluation import load_rows, solve_subset, subset_metrics

    assert MK14_MATRIX is not None and MK14_MANIFEST is not None
    rows, receptors = load_rows(str(MK14_MATRIX), str(MK14_MANIFEST))
    fixture = load_fixture()
    worst_golden = 0.0
    worst_archived = 0.0
    for fold in sorted({int(row["outer_fold"]) for row in rows}):
        train = [row for row in rows if int(row["outer_fold"]) != fold]
        test = [row for row in rows if int(row["outer_fold"]) == fold]
        train_matrix = np.asarray(
            [[float(row[receptor]) for receptor in receptors] for row in train], dtype=np.float64
        )
        test_matrix = np.asarray(
            [[float(row[receptor]) for receptor in receptors] for row in test], dtype=np.float64
        )
        test_labels = np.asarray(
            [1.0 if str(row["label"]) == "active" else 0.0 for row in test], dtype=np.float64
        )
        frozen = fit_fusion(FusionSpec(name="mean"), train_matrix)
        for k in range(1, 7):
            subset = solve_subset(train, receptors, k, 0.25)
            assert list(subset) == fixture[(fold, k)]["subset"], (fold, k)
            golden_value, _ = subset_metrics(test, subset)
            columns = tuple(receptors.index(receptor) for receptor in subset)
            fused = FusionScorer(frozen, test_matrix).score_columns(columns)
            mine = metric_value(fused, test_labels, metric="bedroc20", alpha=20.0)
            worst_golden = max(worst_golden, abs(mine - float(golden_value)))
            worst_archived = max(worst_archived, abs(mine - float(fixture[(fold, k)]["archived"])))
    assert worst_golden < 1e-9
    assert worst_archived < 1e-5