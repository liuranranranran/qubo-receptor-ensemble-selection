"""Deterministic, label-free warm-start planning for masked replay."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping, Sequence

from .state import Task


@dataclass(frozen=True)
class WarmStartConfig:
    baseline_receptor: str
    cluster_fraction: float = 0.1
    min_ligands_per_cluster: int = 1
    random_seed: int = 0

    def __post_init__(self) -> None:
        if not self.baseline_receptor:
            raise ValueError("baseline_receptor must be non-empty")
        if not 0.0 <= self.cluster_fraction <= 1.0:
            raise ValueError("cluster_fraction must be between 0 and 1")
        if self.min_ligands_per_cluster < 0:
            raise ValueError("min_ligands_per_cluster must be non-negative")


def _stable_key(seed: int, *parts: object) -> str:
    payload = "|".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _ligand_id(row: Mapping[str, object]) -> str:
    return str(row.get("ligand_id", ""))


def _scaffold(row: Mapping[str, object]) -> str:
    return str(row.get("scaffold", row.get("scaffold_smiles", "__unknown__")))


def _cluster(row: Mapping[str, object]) -> str:
    return str(row.get("cluster", row.get("receptor_cluster", "__unknown__")))


def plan_warm_start(
    ligand_manifest: Sequence[Mapping[str, object]],
    receptor_manifest: Sequence[Mapping[str, object]],
    config: WarmStartConfig,
) -> tuple[Task, ...]:
    """Return baseline plus deterministic scaffold-stratified cluster coverage."""
    ligand_rows = sorted(ligand_manifest, key=lambda row: _ligand_id(row))
    receptor_rows = sorted(receptor_manifest, key=lambda row: str(row.get("receptor_id", "")))
    receptor_ids = {str(row.get("receptor_id", "")) for row in receptor_rows}
    if config.baseline_receptor not in receptor_ids:
        raise ValueError(f"baseline receptor is not in receptor manifest: {config.baseline_receptor}")
    if not ligand_rows:
        raise ValueError("ligand manifest must not be empty")
    if not receptor_rows:
        raise ValueError("receptor manifest must not be empty")

    planned: set[Task] = {(_ligand_id(row), config.baseline_receptor) for row in ligand_rows}
    total = len(ligand_rows)
    requested = max(config.min_ligands_per_cluster, int(round(total * config.cluster_fraction)))
    by_scaffold: dict[str, list[Mapping[str, object]]] = {}
    for row in ligand_rows:
        by_scaffold.setdefault(_scaffold(row), []).append(row)
    ordered_scaffolds = sorted(by_scaffold, key=lambda value: _stable_key(config.random_seed, "scaffold", value))

    for receptor_row in receptor_rows:
        receptor_id = str(receptor_row.get("receptor_id", ""))
        if receptor_id == config.baseline_receptor:
            continue
        selected: list[Mapping[str, object]] = []
        for scaffold in ordered_scaffolds:
            rows = sorted(
                by_scaffold[scaffold],
                key=lambda row: _stable_key(config.random_seed, receptor_id, _ligand_id(row)),
            )
            if rows:
                selected.append(rows[0])
        remaining = [
            row for row in ligand_rows if row not in selected
        ]
        selected.extend(
            sorted(
                remaining,
                key=lambda row: _stable_key(config.random_seed, "fill", receptor_id, _ligand_id(row)),
            )[: max(0, requested - len(selected))]
        )
        for row in selected[: min(total, requested)]:
            planned.add((_ligand_id(row), receptor_id))
    return tuple(sorted(planned))
