"""Visible-data score predictors for active ligand-receptor docking."""

from __future__ import annotations

import math
import hashlib
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

import numpy as np

from .state import PartialObservationState, StateError, Task


@dataclass(frozen=True)
class ScorePrediction:
    mean: float
    variance: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.mean) or not math.isfinite(self.variance) or self.variance <= 0:
            raise ValueError("score prediction requires finite mean and positive variance")


@dataclass(frozen=True)
class PredictorConfig:
    baseline_receptor: str
    prior_precision: float = 1.0
    noise_variance: float = 1.0
    posterior_samples: int = 64
    random_seed: int = 0

    def __post_init__(self) -> None:
        if not self.baseline_receptor:
            raise ValueError("baseline_receptor must be non-empty")
        if self.prior_precision <= 0 or self.noise_variance <= 0:
            raise ValueError("prior_precision and noise_variance must be positive")
        if self.posterior_samples <= 0:
            raise ValueError("posterior_samples must be positive")


class ScorePredictor(Protocol):
    def fit(self, observed_state: PartialObservationState, training_data: Sequence[Mapping[str, object]] | None = None) -> "ScorePredictor":
        ...

    def predict(self, candidate_tasks: Sequence[Task]) -> dict[Task, ScorePrediction]:
        ...

    def sample(self, candidate_tasks: Sequence[Task], random_state: int | np.random.Generator | None = None) -> list[dict[Task, float]]:
        ...

    def calibration_report(self, hidden_observations: Mapping[Task, float]) -> dict[str, float | int]:
        ...


def _numeric_features(row: Mapping[str, object], excluded: set[str]) -> list[float]:
    values: list[float] = []
    raw_features = row.get("features", [])
    if isinstance(raw_features, Sequence) and not isinstance(raw_features, (str, bytes)):
        for value in raw_features:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.append(float(value))
    descriptors = row.get("descriptors", {})
    if isinstance(descriptors, Mapping):
        for key in sorted(descriptors):
            if key in excluded:
                continue
            value = descriptors[key]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.append(float(value))
    return values


def _hash_bucket(value: object, width: int = 4) -> list[float]:
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:4], "little") % width
    return [1.0 if bucket == index else 0.0 for index in range(width)]


class BayesianResidualPredictor:
    """Bayesian linear residual model with an explicit predictive covariance.

    The posterior is the conjugate Gaussian posterior of a ridge Bayesian linear
    model.  Its features include visible ligand/receptor side information and
    their numeric cross-products; no active labels are consumed.
    """

    def __init__(self, config: PredictorConfig) -> None:
        self.config = config
        self._fitted = False

    def _vector(self, ligand_id: str, receptor_id: str) -> np.ndarray:
        ligand = self._ligands[ligand_id]
        receptor = self._receptors[receptor_id]
        ligand_values = _numeric_features(ligand, {"label"})
        receptor_values = _numeric_features(receptor, {"label"})
        ligand_values = ligand_values[: self._ligand_width] + [0.0] * max(0, self._ligand_width - len(ligand_values))
        receptor_values = receptor_values[: self._receptor_width] + [0.0] * max(0, self._receptor_width - len(receptor_values))
        interaction = [left * right for left in ligand_values for right in receptor_values]
        return np.asarray(
            [1.0, *ligand_values, *receptor_values, *interaction,
             *_hash_bucket(ligand.get("scaffold", ligand.get("scaffold_smiles", "__unknown__"))),
             *_hash_bucket(receptor.get("cluster", receptor.get("receptor_cluster", "__unknown__")))],
            dtype=float,
        )

    def fit(
        self,
        observed_state: PartialObservationState,
        training_data: Sequence[Mapping[str, object]] | None = None,
    ) -> "BayesianResidualPredictor":
        observed_state.assert_no_hidden_information()
        self._state = observed_state.copy()
        self._ligands = {str(row["ligand_id"]): row for row in observed_state.ligand_manifest}
        self._receptors = {str(row["receptor_id"]): row for row in observed_state.receptor_manifest}
        if self.config.baseline_receptor not in self._receptors:
            raise StateError("predictor baseline receptor is absent from state")
        self._ligand_width = max(
            [_numeric_features(row, {"label"}) for row in observed_state.ligand_manifest],
            key=len,
            default=[],
        ).__len__()
        self._receptor_width = max(
            [_numeric_features(row, {"label"}) for row in observed_state.receptor_manifest],
            key=len,
            default=[],
        ).__len__()
        baseline_scores: dict[str, float] = {}
        for task, score in observed_state.observed_scores.items():
            if task[1] == self.config.baseline_receptor:
                baseline_scores[task[0]] = score
        rows: list[tuple[Task, float]] = []
        for task, score in sorted(observed_state.observed_scores.items()):
            if task[0] not in baseline_scores:
                continue
            rows.append((task, score - baseline_scores[task[0]]))
        for row in training_data or []:
            if any(str(key).lower() in {"label", "active", "decoy", "hidden_score", "hidden_label"} for key in row):
                raise StateError("training_data for score prediction cannot contain hidden labels or scores")
            ligand_id = str(row.get("ligand_id", ""))
            receptor_id = str(row.get("receptor_id", ""))
            score_value = row.get("score", row.get("docking_score"))
            if not ligand_id or receptor_id not in self._receptors or score_value is None:
                raise StateError("training_data rows require ligand_id, receptor_id and score")
            baseline = row.get("baseline_score", baseline_scores.get(ligand_id))
            if baseline is None:
                continue
            rows.append(((ligand_id, receptor_id), float(score_value) - float(baseline)))
        if not rows:
            raise StateError("at least one visible residual observation is required")
        design = np.vstack([self._vector(*task) for task, _ in rows])
        response = np.asarray([residual for _, residual in rows], dtype=float)
        dimension = design.shape[1]
        precision = np.eye(dimension) * self.config.prior_precision
        precision += design.T @ design / self.config.noise_variance
        self._covariance = np.linalg.pinv(precision)
        self._coefficient_mean = self._covariance @ design.T @ response / self.config.noise_variance
        self._fitted = True
        self._baseline_scores = baseline_scores
        return self

    def _prediction(self, task: Task) -> ScorePrediction:
        if not self._fitted:
            raise StateError("predictor must be fitted before prediction")
        ligand_id, receptor_id = task
        if ligand_id not in self._ligands or receptor_id not in self._receptors:
            raise StateError(f"unknown prediction task: {task}")
        if ligand_id not in self._baseline_scores:
            raise StateError(f"baseline score is not visible for ligand: {ligand_id}")
        vector = self._vector(ligand_id, receptor_id)
        mean = self._baseline_scores[ligand_id] + float(vector @ self._coefficient_mean)
        variance = self.config.noise_variance + float(vector @ self._covariance @ vector)
        return ScorePrediction(mean=mean, variance=max(variance, 1e-12))

    def predict(self, candidate_tasks: Sequence[Task]) -> dict[Task, ScorePrediction]:
        return {tuple(task): self._prediction(tuple(task)) for task in candidate_tasks}

    def sample(
        self,
        candidate_tasks: Sequence[Task],
        random_state: int | np.random.Generator | None = None,
    ) -> list[dict[Task, float]]:
        predictions = self.predict(candidate_tasks)
        rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(
            self.config.random_seed if random_state is None else random_state
        )
        return [
            {
                task: float(rng.normal(item.mean, math.sqrt(item.variance)))
                for task, item in predictions.items()
            }
            for _ in range(self.config.posterior_samples)
        ]

    def calibration_report(self, hidden_observations: Mapping[Task, float]) -> dict[str, float | int]:
        predictions = self.predict(tuple(hidden_observations))
        errors = np.asarray([predictions[task].mean - float(score) for task, score in hidden_observations.items()])
        variances = np.asarray([predictions[task].variance for task in hidden_observations])
        standardized = np.abs(errors) / np.sqrt(variances)
        return {
            "count": int(len(errors)),
            "mae": float(np.mean(np.abs(errors))) if len(errors) else 0.0,
            "rmse": float(np.sqrt(np.mean(errors**2))) if len(errors) else 0.0,
            "coverage_95": float(np.mean(standardized <= 1.96)) if len(errors) else 0.0,
            "mean_nll": float(np.mean(0.5 * (np.log(2 * np.pi * variances) + errors**2 / variances))) if len(errors) else 0.0,
        }


class ObservedScoreMeanPredictor:
    """Visible-score mean baseline with a constant empirical variance."""

    def fit(self, observed_state: PartialObservationState, training_data: Sequence[Mapping[str, object]] | None = None) -> "ObservedScoreMeanPredictor":
        observed_state.assert_no_hidden_information()
        values = np.asarray(list(observed_state.observed_scores.values()), dtype=float)
        if not len(values):
            raise StateError("mean baseline requires visible scores")
        self._mean = float(np.mean(values))
        self._variance = max(float(np.var(values)), 1e-6)
        self._state = observed_state.copy()
        return self

    def predict(self, candidate_tasks: Sequence[Task]) -> dict[Task, ScorePrediction]:
        return {tuple(task): ScorePrediction(self._mean, self._variance) for task in candidate_tasks}

    def sample(self, candidate_tasks: Sequence[Task], random_state: int | np.random.Generator | None = None) -> list[dict[Task, float]]:
        rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
        return [{tuple(task): float(rng.normal(self._mean, math.sqrt(self._variance))) for task in candidate_tasks} for _ in range(64)]

    def calibration_report(self, hidden_observations: Mapping[Task, float]) -> dict[str, float | int]:
        predictions = self.predict(tuple(hidden_observations))
        errors = np.asarray([predictions[task].mean - float(score) for task, score in hidden_observations.items()])
        return {"count": int(len(errors)), "mae": float(np.mean(np.abs(errors))) if len(errors) else 0.0, "rmse": float(np.sqrt(np.mean(errors**2))) if len(errors) else 0.0}


class NearestReceptorPredictor(ObservedScoreMeanPredictor):
    """Nearest visible receptor baseline using receptor side information."""

    def fit(self, observed_state: PartialObservationState, training_data: Sequence[Mapping[str, object]] | None = None) -> "NearestReceptorPredictor":
        super().fit(observed_state, training_data)
        self._receptors = {str(row["receptor_id"]): row for row in observed_state.receptor_manifest}
        return self

    @staticmethod
    def _distance(left: Mapping[str, object], right: Mapping[str, object]) -> float:
        first = np.asarray(_numeric_features(left, {"label"}), dtype=float)
        second = np.asarray(_numeric_features(right, {"label"}), dtype=float)
        width = max(len(first), len(second))
        first = np.pad(first, (0, width - len(first)))
        second = np.pad(second, (0, width - len(second)))
        return float(np.linalg.norm(first - second))

    def predict(self, candidate_tasks: Sequence[Task]) -> dict[Task, ScorePrediction]:
        output: dict[Task, ScorePrediction] = {}
        for task in candidate_tasks:
            ligand_id, receptor_id = tuple(task)
            visible = [(other_receptor, score) for (other_ligand, other_receptor), score in self._state.observed_scores.items() if other_ligand == ligand_id]
            if visible:
                nearest = min(visible, key=lambda item: (self._distance(self._receptors[item[0]], self._receptors[receptor_id]), item[0]))
                mean = nearest[1]
            else:
                mean = self._mean
            output[(ligand_id, receptor_id)] = ScorePrediction(mean, self._variance)
        return output
