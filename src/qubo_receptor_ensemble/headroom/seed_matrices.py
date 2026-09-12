"""Per-seed primary matrices extracted from a canonical run's ``score_tables``.

The E1 plan requires the three docking seeds to be scanned independently
(section 7.2) and the min-aggregation matrix to be scanned as a sensitivity
(section 7.1).  The canonical runs keep every seed's ``pose_rank=1`` scores in
``<run>/score_tables/seed_<seed>__<receptor>.csv``, so both sensitivities can
be produced without any new docking:

- ``extract_seed_matrix`` rebuilds a single-seed ligand x receptor matrix with
  the frozen receptor column order of the canonical ``primary_median_matrix``;
- ``aggregate_seed_matrices`` combines the per-seed matrices into the
  minimum-over-seeds sensitivity matrix (the remote ``sensitivity_minimum``
  file is only downloaded for some runs).

Every extraction writes a JSON audit (files, rows, missing cells, SHA-256) so
the D1 manifest can prove where the matrix came from.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..io import file_sha256

METADATA_COLUMNS: frozenset[str] = frozenset(
    {"target_id", "ligand_id", "label", "selection_role"}
)
REQUIRED_COLUMNS = {"target_id", "receptor_id", "ligand_id", "label", "pose_rank", "docking_score", "status"}


class SeedMatrixError(ValueError):
    """Raised when score tables cannot be turned into a seed matrix."""


@dataclass(frozen=True)
class SeedMatrix:
    """One single-seed ligand x receptor docking score matrix."""

    seed: int
    target_id: str
    receptor_ids: tuple[str, ...]
    ligand_ids: tuple[str, ...]
    labels: tuple[str, ...]
    scores: np.ndarray
    audit: dict[str, object]

    @property
    def n_ligands(self) -> int:
        return int(self.scores.shape[0])

    @property
    def n_receptors(self) -> int:
        return int(self.scores.shape[1])


def discover_seeds(score_tables_dir: Path) -> tuple[int, ...]:
    """Seeds discovered from ``seed_<seed>__<receptor>.csv`` file names."""
    seeds: set[int] = set()
    for path in sorted(Path(score_tables_dir).glob("seed_*__*.csv")):
        prefix = path.name.split("__", 1)[0]
        if prefix.startswith("seed_"):
            try:
                seeds.add(int(prefix.replace("seed_", "")))
            except ValueError:
                continue
    if not seeds:
        raise SeedMatrixError(f"no seed score tables under {score_tables_dir}")
    return tuple(sorted(seeds))


def reference_receptor_order(matrix_path: Path | None) -> tuple[str, ...] | None:
    """Frozen receptor column order of a canonical primary matrix."""
    if matrix_path is None:
        return None
    with Path(matrix_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
    if not header:
        raise SeedMatrixError(f"empty reference matrix: {matrix_path}")
    receptors = tuple(name for name in header if name not in METADATA_COLUMNS)
    if not receptors:
        raise SeedMatrixError(f"reference matrix has no receptor columns: {matrix_path}")
    return receptors


def _read_score_table(path: Path) -> tuple[str, list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        missing = REQUIRED_COLUMNS.difference(fieldnames)
        if missing:
            raise SeedMatrixError(f"{path} is missing columns: {sorted(missing)}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise SeedMatrixError(f"empty score table: {path}")
    return str(rows[0]["receptor_id"]), rows


def extract_seed_matrix(
    score_tables_dir: Path,
    seed: int,
    *,
    target_id: str | None = None,
    receptor_order: Sequence[str] | None = None,
    representative_pose_rank: int = 1,
) -> SeedMatrix:
    """Rebuild one ligand x receptor matrix from a canonical run's score tables."""
    score_tables_dir = Path(score_tables_dir)
    paths = sorted(score_tables_dir.glob(f"seed_{int(seed)}__*.csv"))
    if not paths:
        raise SeedMatrixError(f"no score tables for seed {seed} under {score_tables_dir}")
    labels: dict[str, str] = {}
    targets: set[str] = set()
    values: dict[str, dict[str, float]] = {}
    duplicates: list[str] = []
    rejected: list[str] = []
    for path in paths:
        receptor_id, rows = _read_score_table(path)
        for row in rows:
            ligand_id = str(row["ligand_id"])
            ligand_target = str(row.get("target_id", ""))
            if ligand_target:
                targets.add(ligand_target)
            rendered_pose_rank = str(row.get("pose_rank", ""))
            if rendered_pose_rank in ("", "0"):
                continue
            if int(float(rendered_pose_rank)) != int(representative_pose_rank):
                continue
            if str(row.get("status", "")) != "ok":
                rejected.append(f"{receptor_id}/{ligand_id}")
                continue
            try:
                score = float(row["docking_score"])
            except (TypeError, ValueError):
                rejected.append(f"{receptor_id}/{ligand_id}")
                continue
            if not np.isfinite(score):
                rejected.append(f"{receptor_id}/{ligand_id}")
                continue
            label = str(row["label"])
            if label not in {"active", "decoy"}:
                raise SeedMatrixError(f"unsupported label {label!r} for {ligand_id}")
            previous = labels.setdefault(ligand_id, label)
            if previous != label:
                raise SeedMatrixError(f"ligand {ligand_id} has conflicting labels")
            slot = values.setdefault(ligand_id, {})
            if receptor_id in slot:
                duplicates.append(f"{receptor_id}/{ligand_id}")
                slot[receptor_id] = min(slot[receptor_id], score)
            else:
                slot[receptor_id] = score
    discovered = sorted({str(row["receptor_id"]) for path in paths for row in _read_score_table(path)[1]})
    if receptor_order is not None:
        receptors = tuple(str(value) for value in receptor_order)
        missing_receptors = [value for value in receptors if value not in discovered]
        if missing_receptors:
            raise SeedMatrixError(
                f"reference receptor order contains columns absent from seed {seed}: {missing_receptors}"
            )
    else:
        receptors = tuple(discovered)
    if not receptors:
        raise SeedMatrixError(f"no receptor columns found for seed {seed}")
    ligand_ids = tuple(sorted(values))
    if not ligand_ids:
        raise SeedMatrixError(f"no ligands found for seed {seed}")
    scores = np.full((len(ligand_ids), len(receptors)), np.nan, dtype=np.float64)
    missing_cells: list[str] = []
    for row_index, ligand_id in enumerate(ligand_ids):
        slot = values[ligand_id]
        for column_index, receptor_id in enumerate(receptors):
            score = slot.get(receptor_id)
            if score is None:
                missing_cells.append(f"{receptor_id}/{ligand_id}")
            else:
                scores[row_index, column_index] = score
    if missing_cells:
        raise SeedMatrixError(
            f"seed {seed} is incomplete: {len(missing_cells)} missing cells "
            f"(first: {missing_cells[:3]})"
        )
    resolved_target = target_id or (sorted(targets)[0] if targets else "UNKNOWN")
    return SeedMatrix(
        seed=int(seed),
        target_id=resolved_target,
        receptor_ids=receptors,
        ligand_ids=ligand_ids,
        labels=tuple(labels[ligand_id] for ligand_id in ligand_ids),
        scores=scores,
        audit={
            "seed": int(seed),
            "score_tables": [path.as_posix() for path in paths],
            "score_table_sha256": {path.as_posix(): file_sha256(path) for path in paths},
            "ligand_count": len(ligand_ids),
            "receptor_count": len(receptors),
            "duplicate_cells_min_folded": len(duplicates),
            "rejected_rows": len(rejected),
            "representative_pose_rank": int(representative_pose_rank),
        },
    )


def write_seed_matrix(
    seed_matrix: SeedMatrix,
    matrix_path: Path,
    *,
    selection_role: str = "development_train",
) -> dict[str, object]:
    """Write the seed matrix in the canonical matrix CSV layout."""
    path = Path(matrix_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["target_id", "ligand_id", "label", "selection_role", *seed_matrix.receptor_ids])
        for row_index, ligand_id in enumerate(seed_matrix.ligand_ids):
            writer.writerow(
                [
                    seed_matrix.target_id,
                    ligand_id,
                    seed_matrix.labels[row_index],
                    selection_role,
                    *[f"{value:.6f}" for value in seed_matrix.scores[row_index]],
                ]
            )
    return {
        "path": path.as_posix(),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
        "metadata": seed_matrix.audit,
    }


def aggregate_seed_matrices(
    matrices: Mapping[int, SeedMatrix],
    *,
    aggregation: str = "min",
) -> SeedMatrix:
    """Combine per-seed matrices into a min (or median) aggregated matrix."""
    if not matrices:
        raise SeedMatrixError("no seed matrices to aggregate")
    seeds = sorted(matrices)
    first = matrices[seeds[0]]
    if aggregation not in {"min", "median"}:
        raise SeedMatrixError("aggregation must be 'min' or 'median'")
    for seed in seeds:
        candidate = matrices[seed]
        if candidate.receptor_ids != first.receptor_ids:
            raise SeedMatrixError(f"seed {seed} has a different receptor order")
        if candidate.ligand_ids != first.ligand_ids:
            raise SeedMatrixError(f"seed {seed} has a different ligand order")
        if candidate.labels != first.labels:
            raise SeedMatrixError(f"seed {seed} has different labels")
    stack = np.stack([matrices[seed].scores for seed in seeds], axis=0)
    aggregated = stack.min(axis=0) if aggregation == "min" else np.median(stack, axis=0)
    return SeedMatrix(
        seed=-1,
        target_id=first.target_id,
        receptor_ids=first.receptor_ids,
        ligand_ids=first.ligand_ids,
        labels=first.labels,
        scores=aggregated,
        audit={
            "aggregation": aggregation,
            "seeds": seeds,
            "ligand_count": first.n_ligands,
            "receptor_count": first.n_receptors,
        },
    )