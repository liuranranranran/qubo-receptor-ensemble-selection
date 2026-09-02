"""Prepare anonymized inputs for offline masked active-docking replay."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Mapping


class ReplayInputError(ValueError):
    """Raised when canonical active outputs cannot form a replay input set."""


_ALLOWED_LABELS = {"active", "decoy", "inactive"}


def _read_json(path: Path, name: str) -> object:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReplayInputError(f"{name} is not valid JSON: {path}") from exc


def _read_active_manifest(path: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    payload = _read_json(path, "active manifest")
    if not isinstance(payload, Mapping):
        raise ReplayInputError("active manifest must be a JSON object")
    ligands_raw = payload.get("ligands")
    receptors_raw = payload.get("receptors")
    if not isinstance(ligands_raw, list) or not isinstance(receptors_raw, list):
        raise ReplayInputError("active manifest must contain ligands and receptors lists")
    ligands = [dict(row) for row in ligands_raw if isinstance(row, Mapping)]
    receptors = [dict(row) for row in receptors_raw if isinstance(row, Mapping)]
    if len(ligands) != len(ligands_raw) or len(receptors) != len(receptors_raw):
        raise ReplayInputError("active manifest rows must be objects")
    if not ligands or not receptors:
        raise ReplayInputError("active manifest ligands and receptors must not be empty")
    return ligands, receptors


def _required_id(row: Mapping[str, object], key: str, name: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ReplayInputError(f"{name} requires a non-empty {key}")
    return value.strip()


def _features(row: Mapping[str, object], name: str) -> list[float]:
    raw = row.get("features")
    if not isinstance(raw, list) or not raw:
        raise ReplayInputError(f"{name}.features must be a non-empty list")
    result: list[float] = []
    for value in raw:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ReplayInputError(f"{name}.features must contain finite numbers") from exc
        if not math.isfinite(number):
            raise ReplayInputError(f"{name}.features must contain finite numbers")
        result.append(number)
    return result


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def prepare_anonymous_replay_inputs(
    active_manifest: str | Path,
    score_matrix: str | Path,
    output_directory: str | Path,
) -> dict[str, object]:
    """Create replay JSON files without exposing source IDs or labels to selection code.

    The wide matrix is the canonical source for labels, but labels are written to a
    separate evaluation-only file. Source ligand IDs and docking paths remain only
    in ``id_map.json`` for post-run analysis.
    """
    manifest_path = Path(active_manifest).resolve()
    matrix_path = Path(score_matrix).resolve()
    output = Path(output_directory).resolve()
    source_ligands, source_receptors = _read_active_manifest(manifest_path)

    source_ligand_ids = [
        _required_id(row, "ligand_id", "ligand manifest row") for row in source_ligands
    ]
    if len(set(source_ligand_ids)) != len(source_ligand_ids):
        raise ReplayInputError("active manifest ligand IDs must be unique")
    receptor_ids = [
        _required_id(row, "receptor_id", "receptor manifest row") for row in source_receptors
    ]
    if len(set(receptor_ids)) != len(receptor_ids):
        raise ReplayInputError("active manifest receptor IDs must be unique")

    opaque_ids = {
        source_id: f"L{index:04d}" for index, source_id in enumerate(source_ligand_ids)
    }
    ligands: list[dict[str, object]] = []
    for source_id, row in zip(source_ligand_ids, source_ligands):
        ligand: dict[str, object] = {
            "ligand_id": opaque_ids[source_id],
            "scaffold": str(row.get("scaffold", row.get("scaffold_smiles", "__unknown__"))),
            "features": _features(row, f"ligand {source_id}"),
        }
        if isinstance(row.get("smiles"), str) and row["smiles"].strip():
            ligand["smiles"] = row["smiles"].strip()
        ligands.append(ligand)

    receptors: list[dict[str, object]] = []
    for receptor_id, row in zip(receptor_ids, source_receptors):
        receptors.append(
            {
                "receptor_id": receptor_id,
                "cluster": str(row.get("cluster", row.get("receptor_cluster", "__unknown__"))),
                "features": _features(row, f"receptor {receptor_id}"),
            }
        )

    if not matrix_path.is_file():
        raise FileNotFoundError(matrix_path)
    with matrix_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        missing_columns = [receptor_id for receptor_id in receptor_ids if receptor_id not in fieldnames]
        if missing_columns:
            raise ReplayInputError(f"score matrix is missing receptor columns: {missing_columns}")
        rows = list(reader)

    if not rows:
        raise ReplayInputError("score matrix must not be empty")
    rows_by_ligand: dict[str, dict[str, str]] = {}
    labels_by_source_id: dict[str, str] = {}
    for row in rows:
        source_id = str(row.get("ligand_id", "")).strip()
        if not source_id:
            raise ReplayInputError("score matrix requires ligand_id")
        if source_id in rows_by_ligand:
            raise ReplayInputError(f"score matrix contains duplicate ligand_id: {source_id}")
        if source_id not in opaque_ids:
            raise ReplayInputError(f"score matrix contains unknown ligand_id: {source_id}")
        label = str(row.get("label", "")).strip().lower()
        if label not in _ALLOWED_LABELS:
            raise ReplayInputError(
                f"score matrix label for {source_id} must be active, decoy or inactive"
            )
        rows_by_ligand[source_id] = row
        labels_by_source_id[source_id] = label

    if set(rows_by_ligand) != set(source_ligand_ids):
        missing = sorted(set(source_ligand_ids) - set(rows_by_ligand))
        extra = sorted(set(rows_by_ligand) - set(source_ligand_ids))
        raise ReplayInputError(f"score matrix ligand IDs differ; missing={missing}, extra={extra}")

    scores: list[dict[str, object]] = []
    for source_id in source_ligand_ids:
        row = rows_by_ligand[source_id]
        for receptor_id in receptor_ids:
            raw_score = str(row.get(receptor_id, "")).strip()
            try:
                score = float(raw_score)
            except (TypeError, ValueError) as exc:
                raise ReplayInputError(
                    f"score matrix must contain a finite score for {source_id}/{receptor_id}"
                ) from exc
            if not math.isfinite(score):
                raise ReplayInputError(
                    f"score matrix must contain a finite score for {source_id}/{receptor_id}"
                )
            scores.append(
                {
                    "ligand_id": opaque_ids[source_id],
                    "receptor_id": receptor_id,
                    "score": score,
                }
            )

    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "ligands.json", ligands)
    _write_json(output / "receptors.json", receptors)
    _write_json(output / "matrix.json", {"scores": scores})
    _write_json(
        output / "labels.json",
        {opaque_ids[source_id]: labels_by_source_id[source_id] for source_id in source_ligand_ids},
    )
    _write_json(
        output / "id_map.json",
        {
            "ligand_id_map": [
                {"opaque_ligand_id": opaque_ids[source_id], "source_ligand_id": source_id}
                for source_id in source_ligand_ids
            ],
            "receptor_ids": receptor_ids,
        },
    )
    return {
        "status": "written",
        "output_directory": str(output),
        "ligand_count": len(ligands),
        "receptor_count": len(receptors),
        "score_count": len(scores),
        "label_count": len(labels_by_source_id),
    }
