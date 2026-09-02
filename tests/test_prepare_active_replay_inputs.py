from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from qubo_receptor_ensemble.active_docking.replay_inputs import (
    prepare_anonymous_replay_inputs,
)


def _write_input_files(tmp_path: Path) -> tuple[Path, Path]:
    manifest = tmp_path / "active_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "ligands": [
                    {
                        "ligand_id": "MK14_active_L000001",
                        "smiles": "CCO",
                        "scaffold": "CCO",
                        "features": [1.0, 2.0],
                        "pdbqt_path": "/secret/MK14_active_L000001.pdbqt",
                    },
                    {
                        "ligand_id": "MK14_decoy_L000002",
                        "smiles": "c1ccccc1",
                        "scaffold": "c1ccccc1",
                        "features": [3.0, 4.0],
                        "pdbqt_path": "/secret/MK14_decoy_L000002.pdbqt",
                    },
                ],
                "receptors": [
                    {"receptor_id": "11OY", "cluster": "cluster_000", "features": [1.0]},
                    {"receptor_id": "1A9U", "cluster": "cluster_001", "features": [2.0]},
                ],
            }
        ),
        encoding="utf-8",
    )
    matrix = tmp_path / "primary_median_matrix.csv"
    with matrix.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["target_id", "ligand_id", "label", "selection_role", "11OY", "1A9U"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "target_id": "MK14",
                    "ligand_id": "MK14_active_L000001",
                    "label": "active",
                    "selection_role": "train",
                    "11OY": "-8.0",
                    "1A9U": "-8.5",
                },
                {
                    "target_id": "MK14",
                    "ligand_id": "MK14_decoy_L000002",
                    "label": "decoy",
                    "selection_role": "train",
                    "11OY": "-6.0",
                    "1A9U": "-6.2",
                },
            ]
        )
    return manifest, matrix


def test_prepare_anonymous_replay_inputs_separates_labels_and_source_ids(
    tmp_path: Path,
) -> None:
    manifest, matrix = _write_input_files(tmp_path)
    output = tmp_path / "replay_inputs_anon"

    result = prepare_anonymous_replay_inputs(manifest, matrix, output)

    assert result["ligand_count"] == 2
    ligands = json.loads((output / "ligands.json").read_text(encoding="utf-8"))
    receptors = json.loads((output / "receptors.json").read_text(encoding="utf-8"))
    scores = json.loads((output / "matrix.json").read_text(encoding="utf-8"))
    labels = json.loads((output / "labels.json").read_text(encoding="utf-8"))
    id_map = json.loads((output / "id_map.json").read_text(encoding="utf-8"))

    assert [row["ligand_id"] for row in ligands] == ["L0000", "L0001"]
    assert [row["receptor_id"] for row in receptors] == ["11OY", "1A9U"]
    assert set(labels) == {"L0000", "L0001"}
    assert labels == {"L0000": "active", "L0001": "decoy"}
    assert id_map["ligand_id_map"] == [
        {"opaque_ligand_id": "L0000", "source_ligand_id": "MK14_active_L000001"},
        {"opaque_ligand_id": "L0001", "source_ligand_id": "MK14_decoy_L000002"},
    ]
    assert all("label" not in row for row in ligands)
    assert all("pdbqt_path" not in row for row in ligands)
    assert all(row["ligand_id"].startswith("L") for row in scores["scores"])
    assert "MK14_active_L000001" not in (output / "ligands.json").read_text(encoding="utf-8")
    assert "MK14_decoy_L000002" not in (output / "matrix.json").read_text(encoding="utf-8")


def test_prepare_anonymous_replay_inputs_rejects_incomplete_matrix(tmp_path: Path) -> None:
    manifest, matrix = _write_input_files(tmp_path)
    matrix.write_text(
        matrix.read_text(encoding="utf-8").replace(",-6.0,-6.2", ",-6.0,"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="finite score"):
        prepare_anonymous_replay_inputs(manifest, matrix, tmp_path / "output")
