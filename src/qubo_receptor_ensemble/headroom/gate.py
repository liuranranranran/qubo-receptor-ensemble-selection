"""Pre-registered G1 decision logic for the E1 headroom scan.

The gate is evaluated on the frozen primary grid (``k in {2, 3}``, all eight
fusion rules, the five pre-registered targets, all five outer folds).  Three
fractions are reported for transparency:

``fraction_cells_go_per_phi_cell``
    literal "400 cells" reading: every (target, fold, phi, k) cell on its own;
``fraction_cells_go_fold_oracle_phi``
    per (target, fold, k) unit the best-performing phi of that fold (upper
    bound of what phi can buy);
``fraction_cells_go_train_selected_phi``
    per unit the phi selected on the training fold only -- the deployable
    level and the headline decision input.

A cell counts as GO when ``h_nested_lower95 > noise_floor`` **and**
``ratio > 1`` (both pre-registered).
"""

from __future__ import annotations

from typing import Mapping, Sequence

DECISIONS = ("NO_GO", "GO_PHI", "GREY_ZONE", "ORACLE_ONLY", "NOT_EVALUATED")


class GateError(ValueError):
    """Raised when the gate input is incomplete or malformed."""


def _cell_passes(cell: Mapping[str, object]) -> bool:
    """Pre-registered pass rule: lower bound beats the noise floor and ratio > 1.

    ``noise_floor`` is the product column name (``noise_se`` is accepted as a
    legacy alias for tests and hand-built cells).
    """
    raw_noise = cell.get("noise_floor")
    if raw_noise is None:
        raw_noise = cell.get("noise_se")
    lower = float(cell.get("h_nested_lower95", float("nan")))
    noise = float(raw_noise) if raw_noise is not None else float("nan")
    ratio = float(cell.get("ratio", float("nan")))
    return bool(lower > noise and ratio > 1.0 and noise > 0.0)


def _fraction(values: Sequence[bool]) -> float:
    if not values:
        return float("nan")
    return sum(1 for value in values if value) / len(values)


def evaluate_gate(
    cells: Sequence[Mapping[str, object]],
    prereg: Mapping[str, object],
    phi_selection: Mapping[str, Mapping[str, object]] | None = None,
    k_displacement: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Apply the frozen G1 rule and return the full decision record."""
    gate_config = prereg.get("gate_g1")
    if not isinstance(gate_config, Mapping):
        raise GateError("preregistration is missing gate_g1")
    no_go_below = float(gate_config.get("no_go_below", 0.10))
    go_above = float(gate_config.get("go_above", 0.30))
    require_lower_bound = bool(gate_config.get("require_lower_bound", True))
    primary = prereg.get("primary_cells")
    if not isinstance(primary, Mapping):
        raise GateError("preregistration is missing primary_cells")
    primary_ks = tuple(int(value) for value in primary.get("k", (2, 3)))
    targets = tuple(str(value) for value in prereg.get("targets", ()))
    fusion_family = tuple(str(value) for value in prereg.get("fusion_family", ()))

    index: dict[tuple[str, int, str, int], Mapping[str, object]] = {}
    for cell in cells:
        key = (str(cell["target_id"]), int(cell["fold"]), str(cell["phi"]), int(cell["k"]))
        index[key] = cell

    units = [
        (target, fold, k)
        for target in targets
        for fold in sorted({int(cell["fold"]) for cell in cells if str(cell["target_id"]) == target})
        for k in primary_ks
    ]
    per_phi_cells: list[bool] = []
    for target, fold, k in units:
        for phi in fusion_family:
            cell = index.get((target, fold, phi, k))
            if cell is None:
                continue
            per_phi_cells.append(_cell_passes(cell))

    fold_oracle_units: list[bool] = []
    fold_oracle_choices: list[dict[str, object]] = []
    for target, fold, k in units:
        best: tuple[float, str, Mapping[str, object]] | None = None
        for phi in fusion_family:
            cell = index.get((target, fold, phi, k))
            if cell is None:
                continue
            value = float(cell.get("h_nested", float("nan")))
            if best is None or value > best[0] or (value == best[0] and phi < best[1]):
                best = (value, phi, cell)
        if best is None:
            continue
        fold_oracle_units.append(_cell_passes(best[2]))
        fold_oracle_choices.append(
            {
                "target_id": target,
                "fold": fold,
                "k": k,
                "phi": best[1],
                "h_nested": best[0],
                "passed": _cell_passes(best[2]),
            }
        )

    train_units: list[bool] = []
    train_choices: list[dict[str, object]] = []
    for target, fold, k in units:
        selection = (phi_selection or {}).get(target, {}).get(str(fold), {})
        phi = None
        if isinstance(selection, Mapping):
            chosen = selection.get("selected_phi_by_k", {})
            if isinstance(chosen, Mapping):
                phi = chosen.get(str(k))
        if not phi:
            phi = "mean"
        cell = index.get((target, fold, str(phi), k))
        if cell is None:
            continue
        train_units.append(_cell_passes(cell))
        train_choices.append(
            {
                "target_id": target,
                "fold": fold,
                "k": k,
                "phi": str(phi),
                "h_nested": float(cell.get("h_nested", float("nan"))),
                "passed": _cell_passes(cell),
            }
        )

    per_phi_fraction = _fraction(per_phi_cells)
    oracle_fraction = _fraction(fold_oracle_units)
    train_fraction = _fraction(train_units)
    headline = train_fraction if train_units else oracle_fraction

    if not cells or not train_units or headline != headline:
        decision = "NOT_EVALUATED"
        branch = (
            "no primary-grid cells were available for this scan (partial run); "
            "run the frozen primary targets before reading G1"
        )
    elif headline >= go_above:
        decision = "GO_PHI"
        branch = (
            "headline GO fraction >= go_above: rerun V5 conclusions with the "
            "(k, phi) joint decision and register phi selection in E4"
        )
    elif headline < no_go_below:
        if oracle_fraction >= go_above:
            decision = "ORACLE_ONLY"
            branch = (
                "fold-oracle phi GO but train-selected phi NO-GO: phi has space, "
                "yet choosing phi is a new unlabelled decision problem -> hand to E4"
            )
        else:
            decision = "NO_GO"
            branch = (
                "headline GO fraction < no_go_below: close the subset-selection line "
                "with an upper-bound conclusion; move to E2 + E5 arm A"
            )
    else:
        decision = "GREY_ZONE"
        branch = "10%-30% grey zone: apply the pre-registered tie-break trace"

    tie_break: dict[str, object] = {
        "rule": (
            "1) is the fold-oracle phi level significantly better than the "
            "train-selected phi level; 2) is k* shifted right by >= 2 consistently "
            "across all five targets"
        ),
        "oracle_minus_train_fraction": (
            None if not train_units else float(oracle_fraction - train_fraction)
        ),
    }
    if k_displacement is not None:
        displacements = []
        consistent = 0
        for target, block in k_displacement.items():
            value = block.get("displacement_best_vs_mean")
            if value is None:
                continue
            displacements.append(float(value))
            if float(value) >= 2:
                consistent += 1
        tie_break["k_star_displacement"] = {
            "targets": len(displacements),
            "targets_shift_right_ge_2": consistent,
            "all_targets_shift_right_ge_2": bool(displacements) and consistent == len(displacements),
            "values": displacements,
        }

    return {
        "schema": "e1_gate_g1_v1",
        "decision": decision,
        "branch": branch,
        "thresholds": {
            "no_go_below": no_go_below,
            "go_above": go_above,
            "require_lower_bound": require_lower_bound,
        },
        "primary_ks": list(primary_ks),
        "targets": list(targets),
        "fusion_family": list(fusion_family),
        "unit_count": len(units),
        "fraction_cells_go_per_phi_cell": per_phi_fraction,
        "fraction_cells_go_fold_oracle_phi": oracle_fraction,
        "fraction_cells_go_train_selected_phi": train_fraction,
        "fraction_cells_go": train_fraction,
        "n_cells_go_per_phi_cell": sum(per_phi_cells),
        "n_cells_considered_per_phi_cell": len(per_phi_cells),
        "n_units": len(units),
        "n_units_fold_oracle": len(fold_oracle_units),
        "n_units_train_selected": len(train_units),
        "tie_break_trace": tie_break,
        "fold_oracle_choices": fold_oracle_choices,
        "train_selected_choices": train_choices,
        "pass_rule": "h_nested_lower95 > noise_floor and ratio > 1",
    }