"""Exhaustive receptor-subset enumeration and scalar utility helpers.

Subsets are represented both as Python tuples of column indices (used for
score slicing) and as integer bit masks (used for stable bookkeeping,
checkpoints and permutation work).  E1 enumerates every subset exactly:
there is no beam search or other approximation in the pre-registered
protocol.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

import numpy as np

from .metrics_fast import metric_value


@dataclass(frozen=True)
class SubsetScan:
    """Exhaustive scan result of one (fold, phi, k, label-set) combination."""

    n_subsets: int
    best_value: float
    best_columns: tuple[int, ...]
    best_mask: int

    def as_dict(self) -> dict[str, object]:
        return {
            "n_subsets": self.n_subsets,
            "best_value": self.best_value,
            "best_columns": list(self.best_columns),
            "best_mask": self.best_mask,
        }


def popcount(mask: int) -> int:
    """Number of set bits of a non-negative integer mask."""
    return int(mask).bit_count()


def mask_columns(mask: int, receptor_count: int) -> tuple[int, ...]:
    """Ascending column indices encoded by ``mask``."""
    if mask < 0 or mask >> receptor_count:
        raise ValueError(f"mask {mask} does not fit {receptor_count} receptors")
    return tuple(index for index in range(receptor_count) if mask >> index & 1)


def columns_mask(columns: Sequence[int], receptor_count: int) -> int:
    mask = 0
    for column in columns:
        if not 0 <= int(column) < receptor_count:
            raise ValueError(f"column {column} outside {receptor_count} receptors")
        mask |= 1 << int(column)
    return mask


def iter_masks(receptor_count: int, k: int) -> Iterator[int]:
    """All bit masks with exactly ``k`` bits over ``receptor_count`` receptors."""
    for combination in itertools.combinations(range(receptor_count), k):
        mask = 0
        for column in combination:
            mask |= 1 << column
        yield mask


def iter_masks_up_to(receptor_count: int, k_max: int) -> Iterator[int]:
    """All bit masks with 1..``k_max`` bits, ordered by increasing cardinality."""
    for k in range(1, k_max + 1):
        yield from iter_masks(receptor_count, k)


def iter_combinations(receptor_count: int, k: int) -> Iterable[tuple[int, ...]]:
    """All ascending ``k``-tuples of receptor column indices."""
    return itertools.combinations(range(receptor_count), k)


def count_subsets(receptor_count: int, k_max: int) -> int:
    """``sum_{k=1..k_max} C(R, k)`` without materializing the subsets."""
    total = 0
    term = 1.0
    for k in range(1, k_max + 1):
        term = term * (receptor_count - k + 1) / k
        total += int(round(term))
    return total


def utility(
    fused: np.ndarray,
    labels: np.ndarray,
    metric: str,
    alpha: float = 20.0,
    ligand_rank: np.ndarray | None = None,
) -> float:
    """Scalar utility of a fused *utility score* vector (higher-is-better).

    ``fused`` is the output of :func:`fusion.apply_fusion`; ``labels`` is the
    ligand-level ``active`` mask of the same rows.
    """
    return metric_value(fused, labels, metric=metric, alpha=alpha, ligand_rank=ligand_rank)


def scan_subset_columns(
    scorer,
    columns: Sequence[Sequence[int]],
    labels: np.ndarray,
    metric: str,
    alpha: float = 20.0,
    ligand_rank: np.ndarray | None = None,
) -> SubsetScan:
    """Exhaustively score an explicit list of column subsets."""
    best_value = float("-inf")
    best_columns: tuple[int, ...] = ()
    best_mask = 0
    count = 0
    for subset in columns:
        count += 1
        value = utility(scorer.score_columns(subset), labels, metric, alpha, ligand_rank)
        if value > best_value:
            best_value = value
            best_columns = tuple(subset)
            best_mask = columns_mask(subset, scorer.n_receptors)
    if count == 0:
        raise ValueError("no subsets were provided")
    return SubsetScan(
        n_subsets=count,
        best_value=best_value,
        best_columns=best_columns,
        best_mask=best_mask,
    )


def oracle_scan(
    scorer,
    receptor_count: int,
    k: int,
    labels: np.ndarray,
    metric: str,
    alpha: float = 20.0,
    ligand_rank: np.ndarray | None = None,
) -> SubsetScan:
    """Exact ``max_{|S|=k} U(S)`` over every receptor subset of size ``k``."""
    return scan_subset_columns(
        scorer,
        iter_combinations(receptor_count, k),
        labels,
        metric,
        alpha,
        ligand_rank,
    )