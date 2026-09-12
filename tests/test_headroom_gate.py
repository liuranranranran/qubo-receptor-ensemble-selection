"""Frozen G1 decision logic."""

from __future__ import annotations

from qubo_receptor_ensemble.headroom.gate import evaluate_gate

PREREG = {
    "schema": "e1_headroom_v1",
    "targets": ["T1", "T2"],
    "fusion_family": ["mean", "min", "max", "zmean", "gmean", "hmean", "ranksum", "rrf"],
    "primary_cells": {"k": [2, 3], "phi": "all"},
    "gate_g1": {"no_go_below": 0.10, "go_above": 0.30, "require_lower_bound": True},
}


def build_cells(passed: set[tuple[str, int, str, int]]) -> list[dict[str, object]]:
    cells = []
    for target in ("T1", "T2"):
        for fold in range(1, 6):
            for phi in PREREG["fusion_family"]:
                for k in (2, 3):
                    ok = (target, fold, phi, k) in passed
                    cells.append(
                        {
                            "target_id": target,
                            "fold": fold,
                            "phi": phi,
                            "k": k,
                            "h_nested": 0.05 if ok else -0.05,
                            "h_nested_lower95": 0.04 if ok else -0.04,
                            "noise_floor": 0.02,
                            "ratio": 2.5 if ok else -2.5,
                        }
                    )
    return cells


def all_keys() -> set[tuple[str, int, str, int]]:
    return {
        (target, fold, phi, k)
        for target in ("T1", "T2")
        for fold in range(1, 6)
        for phi in PREREG["fusion_family"]
        for k in (2, 3)
    }


def test_no_cells_yields_not_evaluated() -> None:
    gate = evaluate_gate([], PREREG)
    assert gate["decision"] == "NOT_EVALUATED"


def test_all_cells_passing_is_go_phi() -> None:
    gate = evaluate_gate(build_cells(all_keys()), PREREG)
    assert gate["decision"] == "GO_PHI"
    assert gate["fraction_cells_go_per_phi_cell"] == 1.0
    assert gate["fraction_cells_go"] == 1.0


def test_all_cells_failing_is_no_go() -> None:
    gate = evaluate_gate(build_cells(set()), PREREG)
    assert gate["decision"] == "NO_GO"
    assert gate["fraction_cells_go_per_phi_cell"] == 0.0


def test_grey_zone_between_thresholds() -> None:
    keys = all_keys()
    passed = set(list(keys)[: int(len(keys) * 0.2)])
    gate = evaluate_gate(build_cells(passed), PREREG)
    assert gate["decision"] == "GREY_ZONE"
    assert "k*" in gate["tie_break_trace"]["rule"]


def test_oracle_only_when_train_selected_phi_fails() -> None:
    # every fold-oracle phi passes, but the train-selected phi is always "mean" and fails
    keys = all_keys()
    passed = {key for key in keys if key[2] == "min"}
    phi_selection = {
        target: {
            str(fold): {"selected_phi_by_k": {"2": "mean", "3": "mean"}}
            for fold in range(1, 6)
        }
        for target in ("T1", "T2")
    }
    gate = evaluate_gate(build_cells(passed), PREREG, phi_selection=phi_selection)
    assert gate["decision"] == "ORACLE_ONLY"
    assert gate["fraction_cells_go_fold_oracle_phi"] > gate["fraction_cells_go_train_selected_phi"]


def test_strict_lower_bound_is_required() -> None:
    cells = build_cells(all_keys())
    for cell in cells:
        cell["h_nested_lower95"] = 0.01  # below the 0.02 noise floor
    gate = evaluate_gate(cells, PREREG)
    assert gate["decision"] == "NO_GO"


def test_k_star_displacement_is_reported_in_tie_break() -> None:
    keys = all_keys()
    passed = set(list(keys)[: int(len(keys) * 0.2)])
    displacement = {
        "T1": {"displacement_best_vs_mean": 2},
        "T2": {"displacement_best_vs_mean": 3},
    }
    gate = evaluate_gate(build_cells(passed), PREREG, k_displacement=displacement)
    trace = gate["tie_break_trace"]["k_star_displacement"]
    assert trace["targets_shift_right_ge_2"] == 2
    assert trace["all_targets_shift_right_ge_2"] is True

def test_legacy_noise_se_alias_still_works() -> None:
    cells = build_cells(all_keys())
    for cell in cells:
        cell["noise_se"] = cell.pop("noise_floor")
    gate = evaluate_gate(cells, PREREG)
    assert gate["decision"] == "GO_PHI"
