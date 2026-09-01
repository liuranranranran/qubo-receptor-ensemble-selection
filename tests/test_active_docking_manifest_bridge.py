from __future__ import annotations

import csv
from pathlib import Path

import pytest

from qubo_receptor_ensemble.active_docking.manifest_bridge import (
    build_active_manifests,
    read_evaluation_labels,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _pdb(path: Path, offset: float) -> None:
    path.write_text(
        "".join(
            f"ATOM  {index:5d}  CA  ALA A{index:4d}    {offset + index:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 20.00           C\n"
            for index in range(1, 4)
        )
        + "END\n",
        encoding="ascii",
    )


def test_prepared_manifests_are_sanitized_and_structural_clusters_are_deterministic(tmp_path: Path) -> None:
    ligand_manifest = tmp_path / "prepared_ligands.csv"
    receptor_manifest = tmp_path / "selected_receptors.csv"
    _write_csv(
        ligand_manifest,
        [
            {
                "ligand_id": "L1",
                "smiles": "CCO",
                "label": "active",
                "selection_role": "development_train",
                "scaffold_smiles": "CC",
                "pdbqt_path": "ligands_pdbqt/L1.pdbqt",
                "sdf_path": "ligands_sdf/L1.sdf",
            },
            {
                "ligand_id": "L2",
                "smiles": "c1ccccc1",
                "label": "decoy",
                "selection_role": "development_train",
                "scaffold_smiles": "c1ccccc1",
                "pdbqt_path": "ligands_pdbqt/L2.pdbqt",
                "sdf_path": "ligands_sdf/L2.sdf",
            },
        ],
    )
    for name, offset in (("11OY", 0.0), ("1A9U", 0.2), ("1BL6", 10.0)):
        _pdb(tmp_path / f"{name}.pdb", offset)
    _write_csv(
        receptor_manifest,
        [
            {
                "conformer_id": "11OY",
                "receptor_pdb": "11OY.pdb",
                "receptor_pdbqt": "receptors/11OY.pdbqt",
                "status": "ok",
            },
            {
                "conformer_id": "1A9U",
                "receptor_pdb": "1A9U.pdb",
                "receptor_pdbqt": "receptors/1A9U.pdbqt",
                "status": "ok",
            },
            {
                "conformer_id": "1BL6",
                "receptor_pdb": "1BL6.pdb",
                "receptor_pdbqt": "receptors/1BL6.pdbqt",
                "status": "ok",
            },
        ],
    )

    first = build_active_manifests(
        ligand_manifest,
        receptor_manifest,
        data_root=tmp_path,
        baseline_receptor="11OY",
        cluster_threshold_angstrom=2.0,
    )
    second = build_active_manifests(
        ligand_manifest,
        receptor_manifest,
        data_root=tmp_path,
        baseline_receptor="11OY",
        cluster_threshold_angstrom=2.0,
    )

    assert first.ligands == second.ligands
    assert first.receptors == second.receptors
    assert [row["receptor_id"] for row in first.receptors] == ["11OY", "1A9U", "1BL6"]
    assert first.receptors[0]["cluster"] == first.receptors[1]["cluster"]
    assert first.receptors[0]["cluster"] != first.receptors[2]["cluster"]
    assert all("label" not in row and "selection_role" not in row for row in first.ligands)
    assert all(
        value not in {"active", "decoy"}
        for row in first.ligands
        for value in row.values()
        if isinstance(value, str)
    )
    assert all(row["features"] for row in first.ligands)
    assert all(row["features"] for row in first.receptors)


def test_manifest_bridge_rejects_missing_real_baseline_and_reads_labels_only_separately(tmp_path: Path) -> None:
    ligand_manifest = tmp_path / "prepared_ligands.csv"
    receptor_manifest = tmp_path / "selected_receptors.csv"
    _write_csv(
        ligand_manifest,
        [{
            "ligand_id": "L1",
            "smiles": "CCO",
            "label": "active",
            "selection_role": "development_train",
            "scaffold_smiles": "CC",
            "pdbqt_path": "L1.pdbqt",
        }],
    )
    _pdb(tmp_path / "R1.pdb", 0.0)
    _write_csv(
        receptor_manifest,
        [{"conformer_id": "R1", "receptor_pdb": "R1.pdb", "receptor_pdbqt": "R1.pdbqt", "status": "ok"}],
    )

    with pytest.raises(ValueError, match="baseline"):
        build_active_manifests(
            ligand_manifest,
            receptor_manifest,
            data_root=tmp_path,
            baseline_receptor="11OY",
        )

    labels = read_evaluation_labels(ligand_manifest)
    assert labels == {"L1": "active"}
