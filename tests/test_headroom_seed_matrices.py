"""Per-seed matrix extraction and min aggregation."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from qubo_receptor_ensemble.headroom.assets import AssetSpec, load_matrix_panel, verify_panel
from qubo_receptor_ensemble.headroom.seed_matrices import (
    SeedMatrixError,
    aggregate_seed_matrices,
    discover_seeds,
    extract_seed_matrix,
    reference_receptor_order,
    write_seed_matrix,
)


def write_score_table(path: Path, receptor: str, seed: int, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "target_id",
                "receptor_id",
                "ligand_id",
                "label",
                "pose_rank",
                "docking_score",
                "status",
                "seed",
                "engine",
                "pose_path",
                "log_path",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    "TST",
                    receptor,
                    row["ligand_id"],
                    row["label"],
                    row["pose_rank"],
                    row["docking_score"],
                    row["status"],
                    seed,
                    "unidock",
                    f"/poses/{receptor}_{row['ligand_id']}.pdbqt",
                    f"/logs/{receptor}.log",
                ]
            )


def build_score_tables(root: Path) -> None:
    ligands = [("L0001", "active"), ("L0002", "decoy"), ("L0003", "decoy")]
    for seed, shift in ((11, 0.0), (12, 0.5)):
        for receptor, base in (("R1", -8.0), ("R2", -7.0)):
            rows = []
            for index, (ligand_id, label) in enumerate(ligands):
                rows.append(
                    {
                        "ligand_id": ligand_id,
                        "label": label,
                        "pose_rank": 1,
                        "docking_score": base - 0.1 * index + shift,
                        "status": "ok",
                    }
                )
                # decoy rows that must be ignored
                rows.append(
                    {
                        "ligand_id": ligand_id,
                        "label": label,
                        "pose_rank": 2,
                        "docking_score": base - 1.0,
                        "status": "ok",
                    }
                )
                rows.append(
                    {
                        "ligand_id": ligand_id,
                        "label": label,
                        "pose_rank": 3,
                        "docking_score": "",
                        "status": "failed",
                    }
                )
            write_score_table(root / f"seed_{seed}__{receptor}.csv", receptor, seed, rows)


def write_manifest(path: Path, ligands: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ligand_id", "scaffold_smiles", "outer_fold"])
        for index, (ligand_id, _) in enumerate(ligands):
            writer.writerow([ligand_id, f"scaffold_{index}", index % 2])


def test_discover_seeds_and_extract_matrix(headroom_workspace: Path) -> None:
    build_score_tables(headroom_workspace)
    assert discover_seeds(headroom_workspace) == (11, 12)
    matrix = extract_seed_matrix(headroom_workspace, 11)
    assert matrix.target_id == "TST"
    assert matrix.receptor_ids == ("R1", "R2")
    assert matrix.ligand_ids == ("L0001", "L0002", "L0003")
    assert list(matrix.labels) == ["active", "decoy", "decoy"]
    assert matrix.scores[0, 0] == pytest.approx(-8.0)
    assert matrix.scores[2, 1] == pytest.approx(-7.2)
    assert matrix.audit["duplicate_cells_min_folded"] == 0
    assert len(matrix.audit["score_table_sha256"]) == 2


def test_duplicate_pose_rank_one_cells_fold_to_min(headroom_workspace: Path) -> None:
    build_score_tables(headroom_workspace)
    table = headroom_workspace / "seed_11__R1.csv"
    with table.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["TST", "R1", "L0001", "active", 1, -9.5, "ok", 11, "unidock", "p", "l"])
    matrix = extract_seed_matrix(headroom_workspace, 11)
    assert matrix.scores[0, 0] == pytest.approx(-9.5)
    assert matrix.audit["duplicate_cells_min_folded"] == 1


def test_reference_receptor_order_is_enforced(headroom_workspace: Path) -> None:
    build_score_tables(headroom_workspace)
    reference = headroom_workspace / "reference.csv"
    with reference.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["target_id", "ligand_id", "label", "selection_role", "R2", "R1"])
        writer.writerow(["TST", "L0001", "active", "development_train", -7.0, -8.0])
    order = reference_receptor_order(reference)
    assert order == ("R2", "R1")
    matrix = extract_seed_matrix(headroom_workspace, 11, receptor_order=order)
    assert matrix.receptor_ids == ("R2", "R1")
    with pytest.raises(SeedMatrixError):
        extract_seed_matrix(headroom_workspace, 11, receptor_order=("R3",))


def test_aggregate_min_is_elementwise(headroom_workspace: Path) -> None:
    build_score_tables(headroom_workspace)
    first = extract_seed_matrix(headroom_workspace, 11)
    second = extract_seed_matrix(headroom_workspace, 12)
    combined = aggregate_seed_matrices({11: first, 12: second}, aggregation="min")
    assert np.allclose(combined.scores, np.minimum(first.scores, second.scores))
    median = aggregate_seed_matrices({11: first, 12: second}, aggregation="median")
    assert np.allclose(median.scores, (first.scores + second.scores) / 2.0)


def test_incomplete_seed_matrix_is_rejected(headroom_workspace: Path) -> None:
    build_score_tables(headroom_workspace)
    table = headroom_workspace / "seed_11__R2.csv"
    rows = [line for line in table.read_text(encoding="utf-8").splitlines() if "L0003" not in line]
    table.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(SeedMatrixError):
        extract_seed_matrix(headroom_workspace, 11)


def test_written_matrix_round_trips_through_the_asset_loader(headroom_workspace: Path) -> None:
    build_score_tables(headroom_workspace)
    matrix = extract_seed_matrix(headroom_workspace, 11)
    matrix_path = headroom_workspace / "seed_11_matrix.csv"
    write_seed_matrix(matrix, matrix_path)
    manifest = headroom_workspace / "prepared.csv"
    write_manifest(manifest, [(ligand_id, label) for ligand_id, label in zip(matrix.ligand_ids, matrix.labels)])
    panel = load_matrix_panel(
        AssetSpec(target_id="TST", matrix=matrix_path, manifest=manifest), matrix_path, manifest
    )
    report = verify_panel(panel)
    assert report["status"] == "ok"
    assert panel.receptor_ids == ("R1", "R2")
    assert report["ligand_count"] == 3


MK14_RUN = Path(r"E:\Quant\remote_runs\mk14_adaptive_remote")
MK14_SCORE_TABLES = MK14_RUN / "score_tables"
MK14_MIN_MATRIX = MK14_RUN / "matrices" / "sensitivity_minimum_matrix.csv"
MK14_PRIMARY = MK14_RUN / "matrices" / "primary_median_matrix.csv"


@pytest.mark.skipif(
    not (MK14_SCORE_TABLES.is_dir() and MK14_MIN_MATRIX.is_file() and MK14_PRIMARY.is_file()),
    reason="MK14 canonical score tables are not available on this machine",
)
def test_extracted_min_matrix_matches_the_canonical_file() -> None:
    seeds = discover_seeds(MK14_SCORE_TABLES)
    order = reference_receptor_order(MK14_PRIMARY)
    matrices = {
        seed: extract_seed_matrix(MK14_SCORE_TABLES, seed, receptor_order=order) for seed in seeds
    }
    aggregated = aggregate_seed_matrices(matrices, aggregation="min")
    with MK14_MIN_MATRIX.open("r", encoding="utf-8-sig", newline="") as handle:
        canonical = {row["ligand_id"]: row for row in csv.DictReader(handle)}
    assert set(canonical) == set(aggregated.ligand_ids)
    worst = 0.0
    for row_index, ligand_id in enumerate(aggregated.ligand_ids):
        for column_index, receptor in enumerate(aggregated.receptor_ids):
            worst = max(
                worst,
                abs(float(canonical[ligand_id][receptor]) - aggregated.scores[row_index, column_index]),
            )
    assert worst == 0.0