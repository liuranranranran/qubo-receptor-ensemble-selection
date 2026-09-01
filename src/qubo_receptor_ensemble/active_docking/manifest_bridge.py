"""Bridge canonical ``prepare`` manifests into visible active-run manifests."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

from ..io import read_csv
from ..ligand_selection import scaffold_smiles
from ..pdb import parse_pdb


@dataclass(frozen=True)
class ActiveManifestBridgeResult:
    """Visible manifests and cluster audit derived from old prepare artifacts."""

    ligands: list[dict[str, object]]
    receptors: list[dict[str, object]]
    receptor_cluster_distances: dict[str, dict[str, float]]


def _resolve_path(value: object, data_root: Path, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (data_root / path).resolve()


def _required(value: Mapping[str, object], key: str, name: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"{name} requires non-empty {key}")
    return item.strip()


def _ligand_features(smiles: str) -> list[float]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"invalid ligand SMILES: {smiles}")
    return [
        float(Descriptors.MolWt(molecule)),
        float(Descriptors.MolLogP(molecule)),
        float(Lipinski.NumHAcceptors(molecule)),
        float(Lipinski.NumHDonors(molecule)),
        float(Lipinski.NumRotatableBonds(molecule)),
        float(Descriptors.RingCount(molecule)),
        float(Descriptors.FractionCSP3(molecule)),
        float(molecule.GetNumHeavyAtoms()),
    ]


def _ca_coordinates(path: Path) -> np.ndarray:
    _, atoms = parse_pdb(path)
    coordinates = [
        atom.coord
        for atom in atoms
        if atom.record == "ATOM" and atom.atom_name == "CA"
    ]
    if not coordinates:
        raise ValueError(f"receptor PDB has no C-alpha atoms: {path}")
    return np.asarray(coordinates, dtype=float)


def _receptor_features(coordinates: np.ndarray) -> list[float]:
    centroid = coordinates.mean(axis=0)
    spread = coordinates.std(axis=0)
    radius = float(np.sqrt(np.mean(np.sum((coordinates - centroid) ** 2, axis=1))))
    return [
        float(len(coordinates)),
        *[float(value) for value in centroid],
        *[float(value) for value in spread],
        radius,
    ]


def _structural_distance(first: np.ndarray, second: np.ndarray) -> float:
    width = min(len(first), len(second))
    if width == 0:
        raise ValueError("receptor structures must contain C-alpha coordinates")
    if len(first) == len(second):
        delta = first - second
    else:
        # Aligned structures with different residue counts use the common prefix;
        # the deterministic fallback keeps clustering independent of row order.
        delta = first[:width] - second[:width]
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def _cluster_coordinates(
    receptor_ids: list[str],
    coordinates: dict[str, np.ndarray],
    threshold_angstrom: float,
    baseline_receptor: str,
) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
    ordered = [baseline_receptor] + sorted(
        receptor_id for receptor_id in receptor_ids if receptor_id != baseline_receptor
    )
    representatives: list[str] = []
    clusters: dict[str, str] = {}
    distances: dict[str, dict[str, float]] = {receptor_id: {} for receptor_id in ordered}
    for receptor_id in ordered:
        assigned = None
        for index, representative in enumerate(representatives):
            distance = _structural_distance(
                coordinates[receptor_id], coordinates[representative]
            )
            distances[receptor_id][representative] = distance
            distances[representative][receptor_id] = distance
            if distance <= threshold_angstrom:
                assigned = f"cluster_{index:03d}"
                break
        if assigned is None:
            representatives.append(receptor_id)
            assigned = f"cluster_{len(representatives) - 1:03d}"
        clusters[receptor_id] = assigned
    return clusters, distances


def _read_manifest(path: Path, name: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"{name} must not be empty: {path}")
    return rows


def build_active_manifests(
    ligand_manifest: str | Path,
    receptor_manifest: str | Path,
    *,
    data_root: str | Path,
    baseline_receptor: str,
    cluster_threshold_angstrom: float = 2.0,
) -> ActiveManifestBridgeResult:
    """Create active manifests from the old prepare CSVs without labels."""
    if not baseline_receptor:
        raise ValueError("baseline receptor must be non-empty")
    if not math.isfinite(float(cluster_threshold_angstrom)) or cluster_threshold_angstrom <= 0:
        raise ValueError("cluster threshold must be positive and finite")
    root = Path(data_root).resolve()
    ligand_rows = _read_manifest(Path(ligand_manifest), "ligand manifest")
    receptor_rows = _read_manifest(Path(receptor_manifest), "receptor manifest")

    ligands: list[dict[str, object]] = []
    ligand_ids: set[str] = set()
    for row in ligand_rows:
        ligand_id = _required(row, "ligand_id", "ligand manifest row")
        if ligand_id in ligand_ids:
            raise ValueError(f"duplicate ligand_id: {ligand_id}")
        ligand_ids.add(ligand_id)
        smiles = _required(row, "smiles", f"ligand {ligand_id}")
        scaffold = str(row.get("scaffold_smiles", "")).strip() or scaffold_smiles(smiles)
        pdbqt_path = _resolve_path(row.get("pdbqt_path"), root, f"ligand {ligand_id}.pdbqt_path")
        ligands.append(
            {
                "ligand_id": ligand_id,
                "smiles": smiles,
                "scaffold": scaffold,
                "features": _ligand_features(smiles),
                "pdbqt_path": pdbqt_path.as_posix(),
            }
        )

    receptor_ids: list[str] = []
    coordinates: dict[str, np.ndarray] = {}
    receptor_records: dict[str, dict[str, object]] = {}
    for row in receptor_rows:
        receptor_id = str(row.get("conformer_id", row.get("receptor_id", ""))).strip()
        if not receptor_id:
            raise ValueError("receptor manifest row requires conformer_id")
        if receptor_id in receptor_records:
            raise ValueError(f"duplicate receptor ID: {receptor_id}")
        pdb_path = _resolve_path(
            row.get("receptor_pdb", row.get("aligned_pdb")),
            root,
            f"receptor {receptor_id}.receptor_pdb",
        )
        pdbqt_path = _resolve_path(
            row.get("receptor_pdbqt"), root, f"receptor {receptor_id}.receptor_pdbqt"
        )
        coords = _ca_coordinates(pdb_path)
        receptor_ids.append(receptor_id)
        coordinates[receptor_id] = coords
        receptor_records[receptor_id] = {
            "receptor_id": receptor_id,
            "features": _receptor_features(coords),
            "receptor_pdb": pdb_path.as_posix(),
            "receptor_pdbqt": pdbqt_path.as_posix(),
        }
    if baseline_receptor not in receptor_records:
        raise ValueError(f"baseline receptor is not present in receptor manifest: {baseline_receptor}")

    clusters, distances = _cluster_coordinates(
        receptor_ids,
        coordinates,
        float(cluster_threshold_angstrom),
        baseline_receptor,
    )
    receptors = []
    for receptor_id in [baseline_receptor] + sorted(
        item for item in receptor_ids if item != baseline_receptor
    ):
        record = dict(receptor_records[receptor_id])
        record["cluster"] = clusters[receptor_id]
        receptors.append(record)
    return ActiveManifestBridgeResult(
        ligands=ligands,
        receptors=receptors,
        receptor_cluster_distances=distances,
    )


def read_evaluation_labels(path: str | Path) -> dict[str, str]:
    """Read labels through an explicit evaluation-only boundary."""
    rows = _read_manifest(Path(path), "evaluation label manifest")
    labels: dict[str, str] = {}
    for row in rows:
        ligand_id = _required(row, "ligand_id", "evaluation label row")
        label = _required(row, "label", f"evaluation label {ligand_id}")
        if ligand_id in labels:
            raise ValueError(f"duplicate evaluation label: {ligand_id}")
        labels[ligand_id] = label
    return labels
