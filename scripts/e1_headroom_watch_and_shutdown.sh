#!/usr/bin/env bash
set -uo pipefail

# Watch a running E1 battery; when it exits: summarize -> archive -> (optionally) shut down.
#
#   nohup env RUN_ID=e1_20260912 AUTO_SHUTDOWN=1 \
#     bash scripts/e1_headroom_watch_and_shutdown.sh \
#     > /root/autodl-tmp/qubo_data_root/results/headroom/e1_20260912.watch.log 2>&1 &
#
# Env knobs:
#   RUN_ID          (default: e1_<today>)
#   DATA_ROOT       (default: /root/autodl-tmp/qubo_data_root)
#   TARGET_PID      (default: the running scripts/run_e1_headroom_remote.sh)
#   CHECK_INTERVAL  seconds between process checks (default 60)
#   AUTO_SHUTDOWN   1 = run `shutdown -h now` after a successful archive (default 1)
#   REQUIRE_SUCCESS 1 = only shut down when the main gate exists and 800 sensitivity shards are present (default 1)
#   KEEP_CELLS      1 = include the per-shard checkpoints in the archive (default 0, they are large)
#   PYTHON_BIN      python interpreter (default python)

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

data_root="${DATA_ROOT:-/root/autodl-tmp/qubo_data_root}"
run_id="${RUN_ID:-e1_$(date +%Y%m%d)}"
out="$data_root/results/headroom/$run_id"
sensitivity_out="${SENSITIVITY_OUT:-${out}_sensitivity}"
check_interval="${CHECK_INTERVAL:-60}"
auto_shutdown="${AUTO_SHUTDOWN:-1}"
require_success="${REQUIRE_SUCCESS:-1}"
keep_cells="${KEEP_CELLS:-0}"
python_bin="${PYTHON_BIN:-python}"
status_file="$data_root/results/headroom/${run_id}_STATUS.txt"

log() { echo "[watch $(date '+%F %T')] $*"; }

target_pid="${TARGET_PID:-$(pgrep -f 'run_e1_headroom_remote.sh' | head -1)}"
if [[ -n "$target_pid" ]]; then
  log "tracking battery PID $target_pid (run_id=$run_id)"
  while kill -0 "$target_pid" 2>/dev/null; do
    sleep "$check_interval"
  done
  log "battery process $target_pid finished"
else
  log "no running battery found; summarizing existing products only"
fi
sleep 15  # let the last file writes flush

# ---- summary ----
summary_json="$data_root/results/headroom/${run_id}_summary.json"
summary_text="$data_root/results/headroom/${run_id}_summary.txt"
"$python_bin" scripts/headroom_scan.py summarize \
  --run-dir "$out" \
  --sensitivity-dir "$sensitivity_out" \
  --output-json "$summary_json" \
  --output-text "$summary_text" || log "summarize failed (continuing)"

# ---- archive ----
stamp="$(date +%Y%m%d_%H%M%S)"
archive="$data_root/results/headroom/${run_id}_products_${stamp}.tar.gz"
members=()
[[ -d "$out" ]] && members+=("$(basename "$out")")
[[ -d "$sensitivity_out" ]] && members+=("$(basename "$sensitivity_out")")
if [[ ${#members[@]} -eq 0 ]]; then
  log "nothing to archive"
else
  excludes=()
  [[ "$keep_cells" != "1" ]] && excludes+=(--exclude='*/cells')
  if tar -czf "$archive" -C "$data_root/results/headroom" "${excludes[@]}" "${members[@]}"; then
    log "archive -> $archive ($(du -h "$archive" | cut -f1))"
  else
    log "tar failed; products stay under $data_root/results/headroom/"
    archive="(archive failed)"
  fi
fi

# ---- status ----
gate_ok=0
[[ -f "$out/gate_g1.json" ]] && gate_ok=1
sens_shards=0
[[ -d "$sensitivity_out/cells" ]] && sens_shards="$(ls "$sensitivity_out/cells" 2>/dev/null | wc -l)"
sens_ok=0
[[ "$sens_shards" -ge 640 ]] && sens_ok=1
decision="unknown"
if [[ -f "$out/gate_g1.json" ]]; then
  decision="$("$python_bin" -c "import json,sys; print(json.load(open(sys.argv[1],encoding='utf-8')).get('decision'))" "$out/gate_g1.json" 2>/dev/null || echo unknown)"
fi
{
  echo "run_id: $run_id"
  echo "finished_at: $(date '+%F %T')"
  echo "gate: $decision (gate_file=$gate_ok)"
  echo "sensitivity_shards: $sens_shards (ok=$sens_ok, expected 800)"
  echo "summary: $summary_json"
  echo "archive: $archive"
} | tee "$status_file"

# ---- shutdown ----
if [[ "$auto_shutdown" == "1" ]]; then
  if [[ "$require_success" == "1" && ( "$gate_ok" != "1" || "$sens_ok" != "1" ) ]]; then
    log "NOT shutting down: success requirements not met (gate_ok=$gate_ok sens_ok=$sens_ok); inspect first"
  else
    log "shutting down (sync first)"
    sync
    shutdown -h now || poweroff || halt -p || log "shutdown command failed; stop the instance from the AutoDL console"
  fi
else
  log "AUTO_SHUTDOWN=0; skipping shutdown"
fi
