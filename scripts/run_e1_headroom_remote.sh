#!/usr/bin/env bash
set -euo pipefail

# E1 headroom battery on the Linux host (no new docking).
#
#   REPO_ROOT=/root/qubo-receptor-ensemble-selection
#   DATA_ROOT=/root/autodl-tmp/qubo_data_root
#
# Steps: T1-T6 tests -> D1 verify -> main scan (9 assets) -> assemble/report
#        -> extract per-seed matrices from canonical score_tables
#        -> sensitivity scan (5 min matrices + 15 single-seed matrices)
#        -> product audit.
#
# Env overrides: RUN_ID, JOBS, PYTHON_BIN, HEADROOM_ROOT, DATA_ROOT, SKIP_TESTS=1.

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

data_root="${DATA_ROOT:-/root/autodl-tmp/qubo_data_root}"
headroom_root="${HEADROOM_ROOT:-$data_root/results/headroom}"
run_id="${RUN_ID:-e1_$(date +%Y%m%d)}"
out="$headroom_root/$run_id"
sensitivity_out="${SENSITIVITY_OUT:-$headroom_root/${run_id}_sensitivity}"
jobs="${JOBS:-32}"
python_bin="${PYTHON_BIN:-python}"
prereg="${PREREG:-configs/experiments/e1_headroom_preregistration.json}"
assets="${ASSETS:-configs/e1_assets_remote.json}"
assets_sensitivity="${ASSETS_SENSITIVITY:-configs/e1_assets_remote_sensitivity.json}"
seeds="${SEEDS:-20260821,20260822,20260823}"

# Small numpy ops: keep BLAS/OpenMP from oversubscribing the shard workers.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

if [[ ! -d "$data_root" ]]; then
  echo "DATA_ROOT does not exist: $data_root" >&2
  exit 1
fi
for target in mk14 ppara pparg bace1 esr1; do
  canonical="$data_root/results/runs/${target}_adaptive_remote"
  if [[ ! -d "$canonical" ]]; then
    echo "canonical run directory is missing: $canonical" >&2
    exit 1
  fi
done
if [[ ! -d "$data_root/results/runs/ppara_pool30_fixed_k4_remote" ]]; then
  echo "pool30 run directory is missing" >&2
  exit 1
fi
if [[ ! -f "$prereg" || ! -f "$assets" || ! -f "$assets_sensitivity" ]]; then
  echo "pre-registration or asset tables are missing" >&2
  exit 1
fi

mkdir -p "$out" "$sensitivity_out"
echo "[e1] repo=$repo_root data_root=$data_root run_id=$run_id jobs=$jobs"

if [[ "${SKIP_TESTS:-0}" != "1" ]]; then
  echo "[e1] step 0/6: T1-T6 + implementation tests"
  "$python_bin" -m pytest -q \
    tests/test_headroom_fusion.py \
    tests/test_headroom_metrics_parity.py \
    tests/test_headroom_protocol_parity.py \
    tests/test_headroom_subsets.py \
    tests/test_headroom_headroom.py \
    tests/test_headroom_bootstrap.py \
    tests/test_headroom_gate.py \
    tests/test_headroom_assets.py \
    tests/test_headroom_seed_matrices.py \
    tests/test_headroom_runner.py
fi

echo "[e1] step 1/6: D1 input verification"
"$python_bin" scripts/headroom_scan.py verify \
  --prereg "$prereg" \
  --assets "$assets" \
  --output-dir "$out"

echo "[e1] step 2/6: main exact scan (${jobs} shard workers)"
"$python_bin" scripts/headroom_scan.py run \
  --prereg "$prereg" \
  --assets "$assets" \
  --output-dir "$out" \
  --jobs "$jobs" \
  --resume \
  --skip-perm \
  --skip-phi-selection \
  --skip-figures

echo "[e1] step 3/6: permutation corrections + train-only phi selection + products"
"$python_bin" scripts/headroom_scan.py report \
  --prereg "$prereg" \
  --assets "$assets" \
  --output-dir "$out" \
  --jobs "$jobs"

echo "[e1] step 4/6: rebuild per-seed matrices from canonical score_tables"
for target in mk14 ppara pparg bace1 esr1; do
  "$python_bin" scripts/headroom_scan.py extract-seeds \
    --run-dir "$data_root/results/runs/${target}_adaptive_remote" \
    --output-dir "$out/seed_matrices/$target" \
    --seeds "$seeds"
done

echo "[e1] step 5/6: sensitivity scan (min aggregation + single seeds)"
"$python_bin" scripts/headroom_scan.py run \
  --prereg "$prereg" \
  --assets "$assets_sensitivity" \
  --root "run_root=$out" \
  --output-dir "$sensitivity_out" \
  --jobs "$jobs" \
  --resume \
  --skip-perm \
  --skip-phi-selection \
  --skip-figures
"$python_bin" scripts/headroom_scan.py report \
  --prereg "$prereg" \
  --assets "$assets_sensitivity" \
  --root "run_root=$out" \
  --output-dir "$sensitivity_out" \
  --jobs "$jobs" \
  --skip-figures

echo "[e1] step 6/6: product audit"
"$python_bin" - "$out" "$sensitivity_out" <<'PY'
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

main = Path(sys.argv[1])
sensitivity = Path(sys.argv[2])
problems: list[str] = []


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


gate = read_json(main / "gate_g1.json")
if gate.get("decision") == "NOT_EVALUATED":
    problems.append("main gate is NOT_EVALUATED (primary grid incomplete)")
if gate.get("n_cells_considered_per_phi_cell") != 400:
    problems.append(f"primary grid is not 400 cells: {gate.get('n_cells_considered_per_phi_cell')}")

manifest = read_json(main / "input_manifest.json")
if manifest.get("primary_targets_missing"):
    problems.append(f"primary targets missing: {manifest['primary_targets_missing']}")
for entry in manifest.get("targets", []):
    if entry.get("status") != "ok":
        problems.append(f"asset not ok: {entry.get('asset_key')} -> {entry.get('status')}")

shards = sorted((main / "cells").glob("*.json"))
if len(shards) < 350:
    problems.append(f"main shard count is too low: {len(shards)}")

rows = list(csv.DictReader((main / "cell_metrics.csv").open(encoding="utf-8")))
if len(rows) < 4000:
    problems.append(f"cell_metrics rows are too low: {len(rows)}")
methods = {row["method"] for row in rows}
if not {"oracle", "ref", "greedy", "single", "train_selected"} <= methods:
    problems.append(f"cell_metrics methods incomplete: {sorted(methods)}")

boot = read_json(main / "bootstrap_report.json")
if len(boot.get("per_cell", {})) < 300:
    problems.append(f"bootstrap report cells are too low: {len(boot.get('per_cell', {}))}")

perm = read_json(main / "permutations.json").get("per_cell", {})
if len(perm) < 300:
    problems.append(f"permutation cells are too low: {len(perm)}")

phi = read_json(main / "phi_selection.json").get("targets", {})
if len(phi) < 7:
    problems.append(f"phi selection records are too low: {len(phi)}")

sens_shards = sorted((sensitivity / "cells").glob("*.json"))
if len(sens_shards) < 20 * 4 * 8:
    problems.append(f"sensitivity shard count is too low: {len(sens_shards)}")

print(f"main: shards={len(shards)} cell_rows={len(rows)} perm={len(perm)} phi={len(phi)}")
print(f"gate: {gate.get('decision')} | per-cell={gate.get('fraction_cells_go_per_phi_cell')} "
      f"oracle-phi={gate.get('fraction_cells_go_fold_oracle_phi')} "
      f"train-phi={gate.get('fraction_cells_go_train_selected_phi')}")
print(f"sensitivity: shards={len(sens_shards)}")
if problems:
    for problem in problems:
        print(f"FAIL: {problem}")
    raise SystemExit(1)
print("AUDIT PASS")
PY

echo "[e1] done: $out"
echo "[e1] copy back:  rsync -av $out $sensitivity_out <local>:/path/to/remote_runs/"