"""E1 input assets: matrix/manifest loading, D1 verification, SHA-256 manifests.

Inputs may come from two equivalent carriers:

1. ``matrix_csv`` + ``prepared manifest`` (the frozen primary matrix and the
   ligand manifest that owns ``scaffold_smiles``/``outer_fold``);
2. ``problem.json`` -- the canonical V5 run artifact, which embeds the full
   primary score rows plus ligand metadata.  The V5 report explicitly used
   these carriers for the four remote adaptive runs whose ``matrices/``
   directories were not downloaded, and reproduced the frozen BEDROC20 values;
   D1 re-verifies the carrier before use and records the substitution.

D1 checks (pre-registered): frozen receptor column order, no missing scores,
``label in {active, decoy}``, exactly one ``outer_fold`` per ligand, fold
coverage, ``scaffold_smiles`` non-empty rate (backfilled from the manifest when
needed and the backfill count recorded), and SHA-256 for every input file.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..io import file_sha256

METADATA_COLUMNS: frozenset[str] = frozenset(
    {"target_id", "ligand_id", "label", "selection_role"}
)
REQUIRED_LABELS: frozenset[str] = frozenset({"active", "decoy"})


class AssetError(ValueError):
    """Raised when an E1 input asset is missing or violates the D1 contract."""


@dataclass(frozen=True)
class AssetSpec:
    """One target entry of ``configs/e1_assets.json``."""

    target_id: str
    role: str = "primary"
    matrix: Path | None = None
    manifest: Path | None = None
    problem_json: Path | None = None
    backfill_problem_json: Path | None = None
    manifest_scaffold_column: str = "scaffold_smiles"
    manifest_fold_column: str = "outer_fold"
    note: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "target_id": self.target_id,
            "role": self.role,
            "matrix": None if self.matrix is None else self.matrix.as_posix(),
            "manifest": None if self.manifest is None else self.manifest.as_posix(),
            "problem_json": None if self.problem_json is None else self.problem_json.as_posix(),
            "backfill_problem_json": (
                None if self.backfill_problem_json is None else self.backfill_problem_json.as_posix()
            ),
            "manifest_scaffold_column": self.manifest_scaffold_column,
            "manifest_fold_column": self.manifest_fold_column,
            "note": self.note,
        }


@dataclass(frozen=True)
class LigandPanel:
    """Aligned ligand-level score panel with frozen receptor order."""

    target_id: str
    receptor_ids: tuple[str, ...]
    ligand_ids: tuple[str, ...]
    labels: np.ndarray
    scaffolds: tuple[str, ...]
    folds: np.ndarray
    scores: np.ndarray
    source: str
    source_paths: dict[str, str] = field(default_factory=dict)
    verification: dict[str, object] = field(default_factory=dict)

    @property
    def n_ligands(self) -> int:
        return int(self.scores.shape[0])

    @property
    def n_receptors(self) -> int:
        return int(self.scores.shape[1])

    def fold_ids(self) -> tuple[int, ...]:
        return tuple(sorted({int(value) for value in self.folds}))

    def subset(self, mask: np.ndarray) -> "LigandPanel":
        index = np.flatnonzero(mask)
        return LigandPanel(
            target_id=self.target_id,
            receptor_ids=self.receptor_ids,
            ligand_ids=tuple(self.ligand_ids[i] for i in index),
            labels=self.labels[index],
            scaffolds=tuple(self.scaffolds[i] for i in index),
            folds=self.folds[index],
            scores=self.scores[index],
            source=self.source,
            source_paths=dict(self.source_paths),
            verification=dict(self.verification),
        )


def read_json_file(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssetError(f"JSON root must be an object: {path}")
    return value


def resolve_placeholders(value: str, roots: Mapping[str, str]) -> str:
    """Expand ``{root_name}`` placeholders in an asset path."""
    resolved = value
    for key, root in roots.items():
        resolved = resolved.replace("{" + key + "}", str(root))
    return resolved


def load_asset_specs(config_path: Path) -> tuple[dict[str, str], list[AssetSpec]]:
    payload = read_json_file(config_path)
    roots = {str(key): str(value) for key, value in dict(payload.get("roots", {})).items()}
    entries = payload.get("targets")
    if not isinstance(entries, list) or not entries:
        raise AssetError("e1_assets.json needs a non-empty targets list")
    specs: list[AssetSpec] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise AssetError("each target entry must be an object")
        target_id = str(entry.get("target_id", "")).strip()
        if not target_id:
            raise AssetError("target entry is missing target_id")

        def optional_path(key: str) -> Path | None:
            raw = entry.get(key)
            if raw in (None, ""):
                return None
            return Path(resolve_placeholders(str(raw), roots)).expanduser()

        specs.append(
            AssetSpec(
                target_id=target_id,
                role=str(entry.get("role", "primary")),
                matrix=optional_path("matrix"),
                manifest=optional_path("manifest"),
                problem_json=optional_path("problem_json"),
                backfill_problem_json=optional_path("backfill_problem_json"),
                manifest_scaffold_column=str(entry.get("manifest_scaffold_column", "scaffold_smiles")),
                manifest_fold_column=str(entry.get("manifest_fold_column", "outer_fold")),
                note=str(entry.get("note", "")),
            )
        )
    return roots, specs


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = [str(name) for name in (reader.fieldnames or [])]
        rows = [dict(row) for row in reader]
    if not fieldnames or not rows:
        raise AssetError(f"empty CSV asset: {path}")
    return fieldnames, rows


def _manifest_lookup(
    manifest_path: Path,
    label_manifest_paths: Sequence[Path] | None = None,
    scaffold_column: str = "scaffold_smiles",
    fold_column: str = "outer_fold",
) -> dict[str, dict[str, str]]:
    paths = [manifest_path, *(label_manifest_paths or [])]
    lookup: dict[str, dict[str, str]] = {}
    for path in paths:
        if path is None or not Path(path).is_file():
            continue
        _, rows = _read_csv_rows(Path(path))
        for row in rows:
            ligand_id = str(row.get("ligand_id", "")).strip()
            if not ligand_id or ligand_id in lookup:
                continue
            lookup[ligand_id] = row
    if not lookup:
        raise AssetError(f"no ligand manifest rows available: {manifest_path}")
    return lookup


def load_matrix_panel(
    spec: AssetSpec,
    matrix_path: Path,
    manifest_path: Path,
    target_id: str | None = None,
) -> LigandPanel:
    """Load a matrix CSV + prepared ligand manifest into a verified panel."""
    fieldnames, matrix_rows = _read_csv_rows(matrix_path)
    if "ligand_id" not in fieldnames or "label" not in fieldnames:
        raise AssetError(f"matrix is missing ligand_id/label columns: {matrix_path}")
    receptor_ids = tuple(name for name in fieldnames if name not in METADATA_COLUMNS)
    if not receptor_ids:
        raise AssetError(f"matrix has no receptor columns: {matrix_path}")
    manifest = _manifest_lookup(manifest_path)
    panel_target = target_id or str(matrix_rows[0].get("target_id", "")) or spec.target_id

    ligand_ids: list[str] = []
    labels: list[float] = []
    scaffolds: list[str] = []
    folds: list[int] = []
    scores: list[list[float]] = []
    missing_scaffold = 0
    backfilled = 0
    for row in matrix_rows:
        ligand_id = str(row["ligand_id"])
        ligand_ids.append(ligand_id)
        raw_label = str(row["label"])
        if raw_label not in REQUIRED_LABELS:
            raise AssetError(f"unsupported label for {ligand_id}: {raw_label}")
        labels.append(1.0 if raw_label == "active" else 0.0)
        meta = manifest.get(ligand_id)
        if meta is None:
            raise AssetError(f"ligand {ligand_id} is missing from the prepared manifest")
        scaffold = str(meta.get(spec.manifest_scaffold_column, "")).strip()
        if not scaffold:
            missing_scaffold += 1
            scaffold = f"UNKNOWN_SCAFFOLD::{ligand_id}"
            backfilled += 1
        scaffolds.append(scaffold)
        fold_value = meta.get(spec.manifest_fold_column)
        if fold_value in (None, ""):
            raise AssetError(f"ligand {ligand_id} has no outer_fold in the manifest")
        folds.append(int(float(str(fold_value))))
        try:
            scores.append([float(row[receptor]) for receptor in receptor_ids])
        except (TypeError, ValueError) as exc:
            raise AssetError(f"non-numeric score row for {ligand_id}") from exc

    order = sorted(range(len(ligand_ids)), key=lambda index: (ligand_ids[index], index))
    panel = LigandPanel(
        target_id=panel_target,
        receptor_ids=receptor_ids,
        ligand_ids=tuple(ligand_ids[index] for index in order),
        labels=np.asarray([labels[index] for index in order], dtype=np.float64),
        scaffolds=tuple(scaffolds[index] for index in order),
        folds=np.asarray([folds[index] for index in order], dtype=np.int64),
        scores=np.asarray([scores[index] for index in order], dtype=np.float64),
        source="matrix_csv",
        source_paths={"matrix": matrix_path.as_posix(), "manifest": manifest_path.as_posix()},
        verification={
            "missing_scaffold_rows": missing_scaffold,
            "backfilled_scaffold_rows": backfilled,
        },
    )
    return panel


def load_problem_panel(
    spec: AssetSpec,
    problem_path: Path,
    target_id: str | None = None,
    backfill_path: Path | None = None,
) -> LigandPanel:
    """Load the embedded primary matrix from a V5 ``problem.json`` carrier."""
    payload = read_json_file(problem_path)
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise AssetError(f"problem.json has no rows: {problem_path}")
    problem_config = payload.get("problem_config")
    problem_meta = payload.get("problem")
    if isinstance(problem_config, dict) and problem_config.get("receptor_ids"):
        receptor_ids = tuple(str(value) for value in problem_config["receptor_ids"])
    elif isinstance(problem_meta, dict) and problem_meta.get("receptor_ids"):
        receptor_ids = tuple(str(value) for value in problem_meta["receptor_ids"])
    else:
        raise AssetError(f"problem.json has no receptor_ids: {problem_path}")
    available = set(str(key) for key in rows[0].keys())
    missing = [receptor for receptor in receptor_ids if receptor not in available]
    if missing:
        raise AssetError(f"problem.json rows miss receptor columns: {missing}")

    backfill: dict[str, dict[str, object]] = {}
    if backfill_path is not None and Path(backfill_path).is_file():
        donor = read_json_file(Path(backfill_path))
        for row in donor.get("rows", []):
            ligand_id = str(row.get("ligand_id", ""))
            if ligand_id:
                backfill[ligand_id] = row

    ligand_ids: list[str] = []
    labels: list[float] = []
    scaffolds: list[str] = []
    folds: list[int] = []
    scores: list[list[float]] = []
    backfilled_scaffold = 0
    backfilled_fold = 0
    for row in rows:
        ligand_id = str(row.get("ligand_id", ""))
        if not ligand_id:
            raise AssetError(f"problem.json row without ligand_id: {problem_path}")
        ligand_ids.append(ligand_id)
        raw_label = str(row.get("label", ""))
        if raw_label not in REQUIRED_LABELS:
            raise AssetError(f"unsupported label for {ligand_id}: {raw_label}")
        labels.append(1.0 if raw_label == "active" else 0.0)
        scaffold = str(row.get("scaffold_smiles", "") or "").strip()
        if not scaffold:
            donor_row = backfill.get(ligand_id, {})
            scaffold = str(donor_row.get("scaffold_smiles", "") or "").strip()
            backfilled_scaffold += 1
        if not scaffold:
            scaffold = f"UNKNOWN_SCAFFOLD::{ligand_id}"
        scaffolds.append(scaffold)
        fold_raw = row.get("outer_fold")
        if fold_raw in (None, ""):
            donor_row = backfill.get(ligand_id, {})
            fold_raw = donor_row.get("outer_fold")
            backfilled_fold += 1
        if fold_raw in (None, ""):
            raise AssetError(f"ligand {ligand_id} has no outer_fold in the problem carrier")
        folds.append(int(float(str(fold_raw))))
        scores.append([float(row[receptor]) for receptor in receptor_ids])

    order = sorted(range(len(ligand_ids)), key=lambda index: (ligand_ids[index], index))
    panel_target = target_id or str(rows[0].get("target_id", "")) or spec.target_id
    return LigandPanel(
        target_id=panel_target,
        receptor_ids=receptor_ids,
        ligand_ids=tuple(ligand_ids[index] for index in order),
        labels=np.asarray([labels[index] for index in order], dtype=np.float64),
        scaffolds=tuple(scaffolds[index] for index in order),
        folds=np.asarray([folds[index] for index in order], dtype=np.int64),
        scores=np.asarray([scores[index] for index in order], dtype=np.float64),
        source="problem_json",
        source_paths={"problem_json": problem_path.as_posix()},
        verification={
            "missing_scaffold_rows": sum(1 for value in scaffolds if not value),
            "backfilled_scaffold_rows": backfilled_scaffold,
            "backfilled_fold_rows": backfilled_fold,
            "backfill_source": None if backfill_path is None else Path(backfill_path).as_posix(),
        },
    )


def load_target_panel(spec: AssetSpec) -> LigandPanel:
    """Load one target through the best available carrier, in frozen priority."""
    if spec.matrix is not None and spec.manifest is not None:
        if spec.matrix.is_file() and spec.manifest.is_file():
            return load_matrix_panel(spec, spec.matrix, spec.manifest)
    if spec.problem_json is not None and spec.problem_json.is_file():
        return load_problem_panel(
            spec, spec.problem_json, backfill_path=spec.backfill_problem_json
        )
    missing = {
        "matrix": None if spec.matrix is None else spec.matrix.as_posix(),
        "manifest": None if spec.manifest is None else spec.manifest.as_posix(),
        "problem_json": None if spec.problem_json is None else spec.problem_json.as_posix(),
    }
    raise AssetError(f"no readable carrier for target {spec.target_id}: {missing}")


def verify_panel(panel: LigandPanel, *, strict: bool = True) -> dict[str, object]:
    """D1 contract checks; returns a machine-readable verification record."""
    problems: list[str] = []
    if panel.scores.shape != (len(panel.ligand_ids), len(panel.receptor_ids)):
        problems.append("score matrix shape does not match ligand/receptor counts")
    if not np.isfinite(panel.scores).all():
        problems.append("score matrix contains missing or non-finite values")
    elif np.isnan(panel.scores).any():
        problems.append("score matrix contains NaN values")
    if len(set(panel.ligand_ids)) != len(panel.ligand_ids):
        problems.append("duplicate ligand_id values")
    if len(set(panel.receptor_ids)) != len(panel.receptor_ids):
        problems.append("duplicate receptor columns")
    labels = set(float(value) for value in np.unique(panel.labels))
    if not labels.issubset({0.0, 1.0}):
        problems.append(f"labels outside {{active, decoy}}: {sorted(labels)}")
    folds = panel.folds
    if len(set(int(value) for value in folds)) < 2:
        problems.append("fewer than two outer folds")
    if any(int(value) < 0 for value in folds):
        problems.append("negative outer_fold values")
    empty_scaffold = sum(1 for scaffold in panel.scaffolds if not str(scaffold).strip())
    if empty_scaffold:
        problems.append(f"{empty_scaffold} ligands have empty scaffold_smiles")
    per_fold: dict[str, dict[str, int]] = {}
    for fold in panel.fold_ids():
        mask = panel.folds == fold
        per_fold[str(fold)] = {
            "ligands": int(mask.sum()),
            "active": int(panel.labels[mask].sum()),
            "scaffolds": int(len({panel.scaffolds[index] for index in np.flatnonzero(mask)})),
        }
    report = {
        "target_id": panel.target_id,
        "source": panel.source,
        "source_paths": panel.source_paths,
        "ligand_count": panel.n_ligands,
        "active_count": int(panel.labels.sum()),
        "decoy_count": int((1.0 - panel.labels).sum()),
        "receptor_count": panel.n_receptors,
        "receptor_order": list(panel.receptor_ids),
        "fold_count": len(panel.fold_ids()),
        "per_fold": per_fold,
        "scaffold_count": len(set(panel.scaffolds)),
        "scaffold_fill_rate": 1.0 - empty_scaffold / max(panel.n_ligands, 1),
        "backfilled_scaffold_rows": int(panel.verification.get("backfilled_scaffold_rows", 0)),
        "backfilled_fold_rows": int(panel.verification.get("backfilled_fold_rows", 0)),
        "status": "ok" if not problems else "failed",
        "problems": problems,
    }
    if problems and strict:
        raise AssetError(f"D1 verification failed for {panel.target_id}: {problems}")
    return report


def sha256_records(paths: Sequence[Path]) -> dict[str, dict[str, object]]:
    """SHA-256 + size records for every existing input path."""
    records: dict[str, dict[str, object]] = {}
    for path in paths:
        candidate = Path(path)
        if not candidate.is_file():
            continue
        records[candidate.as_posix()] = {
            "sha256": file_sha256(candidate),
            "size_bytes": candidate.stat().st_size,
        }
    return records