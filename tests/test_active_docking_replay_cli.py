from __future__ import annotations

import csv
from pathlib import Path

from scripts.run_active_docking import _read_matrix


def test_matrix_reader_can_exclude_hidden_labels_before_prediction(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix.csv"
    with matrix.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["ligand_id", "label", "r0", "r1"],
        )
        writer.writeheader()
        writer.writerow({"ligand_id": "L0000", "label": "active", "r0": "-8", "r1": "-9"})

    scores, labels = _read_matrix(matrix, include_labels=False)

    assert scores[("L0000", "r0")] == -8.0
    assert scores[("L0000", "r1")] == -9.0
    assert labels == {}
