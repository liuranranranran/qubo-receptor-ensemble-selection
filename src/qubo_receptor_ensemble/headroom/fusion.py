"""Eight frozen fusion rules ``phi`` with train-only parameters.

Score convention
----------------
Docking scores follow the repository convention: Vina scores are
lower-is-better.  :func:`apply_fusion` returns a *utility* score in the
opposite direction (higher-is-better), matching ``ranking_score =
-docking_score`` from :mod:`qubo_receptor_ensemble.screening`, so every fusion
output can be consumed directly by :mod:`metrics_fast`.

``mean``/``min``/``max``/``zmean``/``gmean``/``hmean``/``ranksum`` are negated
docking-style aggregates; ``rrf`` already is a similarity (higher-is-better).

Leakage discipline
------------------
Every fitted parameter (z-score mean/std, geometric and harmonic shift, rank
reference frame) is computed from the training fold only.  Test-fold scores
never enter :func:`fit_fusion`.  The rank reference of ``ranksum``/``rrf`` is
the sorted training-fold score vector of each receptor; a score is mapped
through that frozen frame, so changing test-fold values cannot change any
fitted parameter (see T4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

FUSION_NAMES: tuple[str, ...] = (
    "mean",
    "min",
    "max",
    "zmean",
    "gmean",
    "hmean",
    "ranksum",
    "rrf",
)

FUSION_DEFINITIONS: dict[str, str] = {
    "mean": "per-ligand mean docking score (current V5 baseline)",
    "min": "per-ligand minimum docking score (BEmin)",
    "max": "per-ligand maximum docking score (control)",
    "zmean": "mean of per-receptor z-scores, train-fold statistics",
    "gmean": "geometric mean of shifted scores, positive shift from train fold",
    "hmean": "harmonic mean of shifted scores, positive shift from train fold",
    "ranksum": "sum of per-receptor normalized train-reference ranks (0 = best)",
    "rrf": "sum of 1 / (K + rank_r) with fixed K",
}

MIN_SHIFTED_SCORE = 1.0e-9


class FusionError(ValueError):
    """Raised when a fusion specification or input matrix violates its contract."""


@dataclass(frozen=True)
class FusionSpec:
    """A frozen fusion rule definition (no fitted parameters)."""

    name: str
    rrf_k: int = 60
    shift_rule: str = "1.0 - min(train_scores)"

    def __post_init__(self) -> None:
        if self.name not in FUSION_NAMES:
            raise FusionError(f"unknown fusion rule: {self.name}")
        if self.name == "rrf" and self.rrf_k < 1:
            raise FusionError("rrf_k must be a positive integer")
        if self.name in {"gmean", "hmean"} and self.shift_rule != "1.0 - min(train_scores)":
            raise FusionError(f"unsupported shift rule: {self.shift_rule}")

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "rrf_k": self.rrf_k,
            "shift_rule": self.shift_rule,
        }


@dataclass(frozen=True, eq=False)
class FrozenFusion:
    """A :class:`FusionSpec` plus parameters fitted on one training fold."""

    spec: FusionSpec
    receptor_means: np.ndarray | None = None
    receptor_stds: np.ndarray | None = None
    shift: float | None = None
    rank_references: tuple[np.ndarray, ...] | None = None
    degenerate_receptors: tuple[str, ...] = field(default=())

    @property
    def name(self) -> str:
        return self.spec.name

    def as_dict(self) -> dict[str, object]:
        return {
            **self.spec.as_dict(),
            "shift": self.shift,
            "receptor_count": (
                int(self.receptor_means.size) if self.receptor_means is not None else None
            ),
            "degenerate_receptor_count": len(self.degenerate_receptors),
            "rank_reference_sizes": (
                [int(reference.size) for reference in self.rank_references]
                if self.rank_references is not None
                else None
            ),
        }


def _as_matrix(scores: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(scores, dtype=np.float64)
    if matrix.ndim != 2:
        raise FusionError(f"{name} must be a 2-D (n_ligands, n_receptors) array")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise FusionError(f"{name} must not be empty")
    if not np.isfinite(matrix).all():
        raise FusionError(f"{name} contains non-finite scores")
    return matrix


def fit_fusion(spec: FusionSpec, train_scores: np.ndarray) -> FrozenFusion:
    """Fit the train-only parameters of ``spec`` on ``train_scores``."""
    train = _as_matrix(train_scores, "train_scores")
    if spec.name == "zmean":
        means = train.mean(axis=0)
        stds = train.std(axis=0, ddof=0)
        degenerate = tuple(
            f"receptor_{index}" for index, std in enumerate(stds) if std <= MIN_SHIFTED_SCORE
        )
        safe_stds = np.where(stds <= MIN_SHIFTED_SCORE, 1.0, stds)
        return FrozenFusion(
            spec=spec,
            receptor_means=means,
            receptor_stds=safe_stds,
            degenerate_receptors=degenerate,
        )
    if spec.name in {"gmean", "hmean"}:
        return FrozenFusion(spec=spec, shift=1.0 - float(train.min()))
    if spec.name in {"ranksum", "rrf"}:
        references = tuple(np.sort(train[:, column]) for column in range(train.shape[1]))
        return FrozenFusion(spec=spec, rank_references=references)
    return FrozenFusion(spec=spec)


def _rank_against_reference(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """1-based average rank of ``values`` inside the sorted ``reference`` frame.

    Ties are averaged; values better than every reference score are clamped to
    rank 1.0 so that normalized ranks stay non-negative.
    """
    left = np.searchsorted(reference, values, side="left")
    right = np.searchsorted(reference, values, side="right")
    ranks = 0.5 * (left + right) + 0.5
    return np.maximum(ranks, 1.0)


class FusionScorer:
    """Materialized per-column terms for one (fold, phi) so subset scoring is cheap.

    ``scores`` is the full (n_ligands, n_receptors) matrix in the frozen
    receptor order.  ``score_columns`` returns the utility score of one
    receptor subset without re-fitting anything.
    """

    def __init__(self, frozen: FrozenFusion, scores: np.ndarray) -> None:
        self.frozen = frozen
        self.name = frozen.name
        self.scores = _as_matrix(scores, "scores")
        self.n_ligands, self.n_receptors = self.scores.shape
        shape = (self.n_ligands, self.n_receptors)
        if frozen.name == "zmean":
            if frozen.receptor_means is None or frozen.receptor_stds is None:
                raise FusionError("zmean fusion is missing fitted statistics")
            if frozen.receptor_means.size != self.n_receptors:
                raise FusionError("zmean statistics do not match the receptor count")
            self.terms = (self.scores - frozen.receptor_means) / frozen.receptor_stds
            self._mode = "mean"
        elif frozen.name == "gmean":
            if frozen.shift is None:
                raise FusionError("gmean fusion is missing its shift")
            shifted = np.clip(self.scores + frozen.shift, MIN_SHIFTED_SCORE, None)
            self.terms = np.log(shifted)
            self._mode = "exp_mean"
        elif frozen.name == "hmean":
            if frozen.shift is None:
                raise FusionError("hmean fusion is missing its shift")
            shifted = np.clip(self.scores + frozen.shift, MIN_SHIFTED_SCORE, None)
            self.terms = 1.0 / shifted
            self._mode = "reciprocal_sum"
        elif frozen.name in {"ranksum", "rrf"}:
            if frozen.rank_references is None:
                raise FusionError("rank fusion is missing its train reference frame")
            if len(frozen.rank_references) != self.n_receptors:
                raise FusionError("rank references do not match the receptor count")
            ranks = np.empty(shape, dtype=np.float64)
            for column, reference in enumerate(frozen.rank_references):
                ranks[:, column] = _rank_against_reference(reference, self.scores[:, column])
            if frozen.name == "ranksum":
                self.terms = (ranks - 1.0) / reference_size(frozen)
                self._mode = "sum"
            else:
                self.terms = 1.0 / (frozen.spec.rrf_k + ranks)
                self._mode = "sum"
        else:  # mean / min / max operate directly on the docking scores
            self.terms = self.scores
            self._mode = frozen.name

    def score_columns(self, columns: Sequence[int]) -> np.ndarray:
        """Utility score (higher-is-better) of one receptor subset."""
        selected = self.terms[:, list(columns)]
        if self._mode == "mean":
            return -selected.mean(axis=1)
        if self._mode == "exp_mean":
            return -np.exp(selected.mean(axis=1))
        if self._mode == "reciprocal_sum":
            return -selected.shape[1] / selected.sum(axis=1)
        if self._mode == "sum":
            aggregate = selected.sum(axis=1)
            return -aggregate if self.name == "ranksum" else aggregate
        if self._mode == "min":
            return -selected.min(axis=1)
        if self._mode == "max":
            return -selected.max(axis=1)
        raise FusionError(f"unsupported materialized mode: {self._mode}")


def reference_size(frozen: FrozenFusion) -> int:
    if frozen.rank_references is None or not frozen.rank_references:
        raise FusionError("rank fusion has no reference frame")
    return int(frozen.rank_references[0].size)


def apply_fusion(frozen: FrozenFusion, scores: np.ndarray) -> np.ndarray:
    """Apply a frozen fusion to all columns of ``scores`` (full receptor order)."""
    matrix = _as_matrix(scores, "scores")
    scorer = FusionScorer(frozen, matrix)
    return scorer.score_columns(tuple(range(matrix.shape[1])))


def apply_fusion_subset(
    frozen: FrozenFusion, scores: np.ndarray, columns: Sequence[int]
) -> np.ndarray:
    """Apply a frozen fusion to ``columns`` of the full-width ``scores`` matrix."""
    return FusionScorer(frozen, scores).score_columns(columns)


def build_specs(names: Sequence[str] | None = None, rrf_k: int = 60) -> tuple[FusionSpec, ...]:
    """Return the frozen fusion family in the pre-registered order."""
    selected = tuple(FUSION_NAMES if names is None else names)
    unknown = [name for name in selected if name not in FUSION_NAMES]
    if unknown:
        raise FusionError(f"unknown fusion rules: {unknown}")
    return tuple(FusionSpec(name=name, rrf_k=rrf_k) for name in selected)


def fusion_direction_table() -> dict[str, str]:
    """Human-readable direction of each raw (un-negated) aggregate."""
    return {
        name: ("higher-is-better" if name == "rrf" else "lower-is-better")
        for name in FUSION_NAMES
    }