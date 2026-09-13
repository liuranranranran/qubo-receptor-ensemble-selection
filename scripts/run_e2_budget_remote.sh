#!/usr/bin/env bash
set -uo pipefail

# E2 budget-allocation battery on the Linux host (zero docking).
#
#   nohup env RUN_ID=e2_20260915 JOBS=32 AUTO_SHUTDOWN=1 \
#     bash scripts/run_e2_budget_remote.sh \
#     > "$DATA_ROOT/results/budget/e2_20260915.log" 2>&1 &
#
# Steps: unit tests -> budget scan (sharded, resumable) -> assemble -> archive -> optional shutdown.

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

data_root="${DATA_ROOT:-/root/autodl-tmp/qubo_data_root}"
run_id="${RUN_ID:-e2_$(date +%Y%m%d)}"
out="${OUT:-$data_root/results/budget/$run_id}"
jobs="${JOBS:-32}"
python_bin="${PYTHON_BIN:-python}"
prereg="${PREREG:-configs/experiments/e2_budget_preregistration.json}"
assets="${ASSETS:-configs/e1_assets_remote.json}"
auto_shutdown="${AUTO_SHUTDOWN:-0}"
keep_cells="${KEEP_CELLS:-0}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

log() { echo "[e2 $(date '+%F %T')] $*"; }

[[ -d "$data_root" ]] || { echo "DATA_ROOT missing: $data_root" >&2; exit 1; }
[[ -f "$prereg" && -f "$assets" ]] || { echo "prereg/assets missing" >&2; exit 1; }
mkdir -p "$out"

log "step 1/4: unit tests"
"$python_bin" -m pytest -q tests/test_budget_core.py

log "step 2/4: budget scan ($jobs shard workers)"
"$python_bin" scripts/budget_scan.py run \
  --prereg "$prereg" \
  --assets "$assets" \
  --output-dir "$out" \
  --jobs "$jobs" \
  --resume

log "step 3/4: assemble products (curves + paired bootstrap + figures)"
"$python_bin" scripts/budget_scan.py report \
  --prereg "$prereg" \
  --assets "$assets" \
  --output-dir "$out"

log "step 4/4: summary + archive"
"$python_bin" - "$out" <<'PY'
import csv, json, sys
from pathlib import Path
run_dir = Path(sys.argv[1])
gate = json.loads((run_dir / "gate_g2.json").read_text(encoding="utf-8"))
print("G2 decision:", gate.get("decision"),
      "| width-locked fraction:", gate.get("fraction_width_locked"),
      "| stable policy@budget@phi:", gate.get("stable_policy_budget_phi"))
law_path = run_dir / "law_summary.json"
if law_path.is_file():
    law = json.loads(law_path.read_text(encoding="utf-8"))
    print("law: rows", law.get("n_rows"), "switching", law.get("n_switching"),
          "| spearman(pair_gain, B*)", law.get("spearman_feature_vs_b_star_rank", {}).get("pair_gain_train_pr_auc"))
curves = list(csv.DictReader((run_dir / "budget_curves.csv").open(encoding="utf-8")))
primary = ("MK14", "PPARG", "BACE1", "ESR1", "PPARA")
for target in primary:
    rows = [row for row in curves if row["target_id"] == target and row["phi"] == "mean"]
    if not rows:
        continue
    best = {}
    for row in rows:
        budget = int(row["budget"])
        value = float(row["pr_auc_mean"])
        if budget not in best or value > best[budget][1]:
            best[budget] = (row["policy"], value)
    print(f"  {target:6s} best policy by budget:",
          ", ".join(f"{budget}:{policy}" for budget, (policy, _) in sorted(best.items())))
PY

stamp="$(date +%Y%m%d_%H%M%S)"
archive="$data_root/results/budget/${run_id}_products_${stamp}.tar.gz"
excludes=()
[[ "$keep_cells" != "1" ]] && excludes+=(--exclude='*/cells')
if tar -czf "$archive" -C "$data_root/results/budget" "${excludes[@]}" "$(basename "$out")"; then
  log "archive -> $archive ($(du -h "$archive" | cut -f1))"
else
  log "tar failed; products stay under $out"
fi
{
  echo "run_id: $run_id"
  echo "finished_at: $(date '+%F %T')"
  echo "gate_g2: $(python3 -c "import json,sys; print(json.load(open(sys.argv[1],encoding='utf-8')).get('decision'))" "$out/gate_g2.json" 2>/dev/null || echo unknown)"
  echo "archive: $archive"
} > "$data_root/results/budget/${run_id}_STATUS.txt"

if [[ "$auto_shutdown" == "1" && -f "$out/gate_g2.json" ]]; then
  log "shutting down"
  sync
  shutdown -h now || poweroff || halt -p || log "shutdown failed; stop the instance from the console"
else
  log "done (AUTO_SHUTDOWN=$auto_shutdown)"
fi