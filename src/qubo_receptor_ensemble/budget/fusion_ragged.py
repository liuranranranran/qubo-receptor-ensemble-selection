"""Ragged (heterogeneous-depth) fusion for budget-constrained docking.

E2 docks a *different* number of conformers for different ligands, so a ligand
score is the fusion of its own observed receptor subset instead of the same
subset for every ligand (which is what :mod:`headroom.fusion` materializes).
The mathematics is identical to :class:`headroom.fusion.FusionScorer`; for a
uniform allocation the two agree bit-for-bit (see the parity test).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..headroom.fusion import FrozenFusion, FusionScorer, MIN_SHIFTED_SCORE


class RaggedFusionError(ValueError):
    """Raised when a ragged fusion request violates its contract."""


@dataclass(frozen=True, eq=False)
class RaggedFusion:
    """Per-column terms + the aggregation mode of a frozen fusion rule."""

    frozen: FrozenFusion
    terms: np.ndarray
    mode: str

    @property
    def name(self) -> str:
        return self.frozen.name


def build_ragged_fusion(frozen: FrozenFusion, full_scores: np.ndarray) -> RaggedFusion:
    """Materialize per-column terms for a full-width score matrix."""
    scorer = FusionScorer(frozen, full_scores)
    return RaggedFusion(frozen=frozen, terms=scorer.terms, mode=scorer._mode)  # noqa: SLF001


def fuse_ragged(ragged: RaggedFusion, mask: np.ndarray) -> np.ndarray:
    """Fuse every ligand over its own observed columns.

    ``mask`` is ``(n_ligands, n_receptors)`` boolean.  Ligands with no observed
    cell get ``-inf`` (they are ranked last by the screening metrics).  Returns
    utility scores (higher-is-better), exactly like ``FusionScorer``.
    """
    observed = np.asarray(mask, dtype=bool)
    if observed.ndim != 2 or observed.shape != ragged.terms.shape:
        raise RaggedFusionError(
            f"mask shape {observed.shape} does not match terms shape {ragged.terms.shape}"
        )
    counts = observed.sum(axis=1)
    utilities = np.full(observed.shape[0], -np.inf, dtype=np.float64)
    docked = counts > 0
    if not docked.any():
        return utilities
    terms = np.where(observed, ragged.terms, 0.0)
    if ragged.mode == "mean":
        utilities[docked] = -terms[docked].sum(axis=1) / counts[docked]
    elif ragged.mode == "exp_mean":
        utilities[docked] = -np.exp(terms[docked].sum(axis=1) / counts[docked])
    elif ragged.mode == "reciprocal_sum":
        utilities[docked] = -counts[docked] / terms[docked].sum(axis=1)
    elif ragged.mode == "sum":
        total = terms[docked].sum(axis=1)
        utilities[docked] = -total if ragged.name == "ranksum" else total
    elif ragged.mode == "min":
        filled = np.where(observed, ragged.terms, np.inf)
        utilities[docked] = -filled[docked].min(axis=1)
    elif ragged.mode == "max":
        filled = np.where(observed, ragged.terms, -np.inf)
        utilities[docked] = -filled[docked].max(axis=1)
    else:  # pragma: no cover - guarded by FusionScorer
        raise RaggedFusionError(f"unsupported ragged mode: {ragged.mode}")
    return utilities


def fit_ragged_fusion(spec, train_scores: np.ndarray) -> RaggedFusion:
    """Fit a frozen fusion on the training fold and materialize its terms."""
    from ..headroom.fusion import fit_fusion

    frozen = fit_fusion(spec, train_scores)
    return build_ragged_fusion(frozen, train_scores)


__all__ = ["RaggedFusion", "RaggedFusionError", "build_ragged_fusion", "fit_ragged_fusion", "fuse_ragged", "MIN_SHIFTED_SCORE"]