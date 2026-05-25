#!/usr/bin/env bash
set -uo pipefail

ROOT=""
RUN_NAME="usable_ocr_path"
BATCH="192"
MAX_HOURS="8"
SKIP_TROCR="0"
SKIP_ROW_CTC="0"
SKIP_SYLLABLE="0"
SMOKE_ONLY="0"
ROW_DATA=""
SYLL_DATA=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --batch) BATCH="$2"; shift 2 ;;
    --max-hours) MAX_HOURS="$2"; shift 2 ;;
    --skip-trocr) SKIP_TROCR="1"; shift ;;
    --skip-row-ctc) SKIP_ROW_CTC="1"; shift ;;
    --skip-syllable) SKIP_SYLLABLE="1"; shift ;;
    --smoke-only) SMOKE_ONLY="1"; shift ;;
    --row-data) ROW_DATA="$2"; shift 2 ;;
    --syllable-data) SYLL_DATA="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "$ROOT" ]]; then
  ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
if [[ ! -d "$ROOT" ]]; then
  echo "root not found: $ROOT" >&2
  exit 2
fi

cd "$ROOT"

resolve_python() {
  local candidates=(
    "/root/miniconda3/bin/python"
    "/root/miniconda3/bin/python3"
    "$ROOT/.venv/bin/python"
    "$ROOT/.venv-ocr/bin/python"
    "python3.12"
    "python3"
    "python"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ "$candidate" == */* && ! -x "$candidate" ]]; then
      continue
    fi
    if "$candidate" -c 'import sys; print(sys.version)' >/dev/null 2>&1; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

PYTHON="$(resolve_python)" || { echo "no working python found" >&2; exit 2; }

START_TS="$(date +%s)"
DEADLINE_TS="$("$PYTHON" - <<PY
import time
print(int(time.time() + float("$MAX_HOURS") * 3600))
PY
)"
RUN_ROOT="$ROOT/ipa_ocr_work/runs/$RUN_NAME"
LOG_ROOT="$RUN_ROOT/logs"
mkdir -p "$LOG_ROOT"
STATUS_PATH="$RUN_ROOT/status.json"
EXPERIMENTS_JSON="$RUN_ROOT/experiments.jsonl"
: > "$EXPERIMENTS_JSON"

export KMP_AFFINITY=disabled
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

write_status() {
  local stage="$1"
  local gpu="nvidia-smi unavailable"
  if command -v nvidia-smi >/dev/null 2>&1; then
    gpu="$(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader 2>/dev/null || true)"
  fi
  "$PYTHON" - "$STATUS_PATH" "$stage" "$ROOT" "$RUN_ROOT" "$PYTHON" "$START_TS" "$DEADLINE_TS" "$gpu" "$EXPERIMENTS_JSON" <<'PY'
import json
import sys
from pathlib import Path

status_path, stage, root, run_root, python, start_ts, deadline_ts, gpu, exp_path = sys.argv[1:]
experiments = []
path = Path(exp_path)
if path.exists():
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            experiments.append(json.loads(line))
payload = {
    "stage": stage,
    "root": root,
    "run_root": run_root,
    "python": python,
    "started_at_unix": int(start_ts),
    "deadline_unix": int(deadline_ts),
    "gpu": gpu,
    "experiments": experiments,
}
Path(status_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
PY
}

record_experiment() {
  local name="$1"
  local state="$2"
  local reason="$3"
  local out_dir="$4"
  "$PYTHON" - "$EXPERIMENTS_JSON" "$name" "$state" "$reason" "$out_dir" <<'PY'
import json
import sys
import time
from pathlib import Path

path, name, state, reason, out_dir = sys.argv[1:]
items = []
p = Path(path)
if p.exists():
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            items.append(json.loads(line))
found = False
for item in items:
    if item["name"] == name:
        item.update({"state": state, "reason": reason, "out_dir": out_dir, "updated_at_unix": int(time.time())})
        found = True
if not found:
    items.append({"name": name, "state": state, "reason": reason, "out_dir": out_dir, "updated_at_unix": int(time.time())})
p.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in items) + "\n", encoding="utf-8")
PY
  write_status "$name"
}

run_step() {
  local name="$1"
  local out_dir="$2"
  shift 2
  if [[ "$(date +%s)" -ge "$DEADLINE_TS" ]]; then
    record_experiment "$name" "skipped" "deadline reached before start" "$out_dir"
    return 124
  fi
  mkdir -p "$out_dir"
  local log="$LOG_ROOT/$name.log"
  local err="$LOG_ROOT/$name.err.log"
  record_experiment "$name" "running" "" "$out_dir"
  echo "RUN $name: $*" > "$log"
  "$@" >> "$log" 2>> "$err" &
  local pid="$!"
  while kill -0 "$pid" >/dev/null 2>&1; do
    sleep 30
    write_status "$name"
    if [[ "$(date +%s)" -ge "$DEADLINE_TS" ]]; then
      kill "$pid" >/dev/null 2>&1 || true
      record_experiment "$name" "timeout" "deadline reached" "$out_dir"
      return 124
    fi
  done
  wait "$pid"
  local code="$?"
  if [[ "$code" -eq 0 ]]; then
    record_experiment "$name" "done" "" "$out_dir"
  else
    record_experiment "$name" "failed" "exit code $code; see $log and $err" "$out_dir"
  fi
  return "$code"
}

score_row_predictions() {
  local name="$1"
  local predictions="$2"
  local out_dir="$3"
  local mode="$4"
  run_step "${name}_score" "$out_dir" "$PYTHON" \
    ipa_ocr_work/scripts/score_ocr_experiment.py \
    --eval-manifest "$ROW_DATA/eval_manifest.tsv" \
    --predictions "$predictions" \
    --out-prefix "$out_dir/score" \
    --prediction-mode "$mode" \
    --ipa-label-source from-wupin \
    --include-missing
}

decode_and_score() {
  local name="$1"
  local predictions="$2"
  local out_dir="$3"
  local mode="$4"
  local decoded="$out_dir/predictions_lexicon.tsv"
  run_step "${name}_lexicon_decode" "$out_dir" "$PYTHON" \
    ipa_ocr_work/scripts/apply_wupin_lexicon_decoder.py \
    --eval-manifest "$ROW_DATA/eval_manifest.tsv" \
    --predictions "$predictions" \
    --out "$decoded" \
    --prediction-mode "$mode" \
    --lexicon-split train \
    --max-syllable-distance 2 \
    --row-nearest \
    --row-nearest-max-cer 0.18
  local code="$?"
  [[ "$code" -eq 0 ]] || return "$code"
  score_row_predictions "${name}_lexicon" "$decoded" "$out_dir" wupin
}

write_status "initializing"

if [[ -z "$ROW_DATA" ]]; then
  ROW_DATA="$ROOT/ipa_ocr_work/dataset/shaoxing_pdf136_clean/ocr_selected_all"
fi
if [[ -z "$SYLL_DATA" ]]; then
  SYLL_DATA="$ROOT/ipa_ocr_work/dataset/shaoxing_pdf136_clean/syllable_ocr_all"
fi
[[ -f "$ROW_DATA/eval_manifest.tsv" ]] || { echo "missing row manifest: $ROW_DATA" >&2; exit 2; }
[[ -f "$SYLL_DATA/eval_manifest.tsv" ]] || { echo "missing syllable manifest: $SYLL_DATA" >&2; exit 2; }

if [[ "$SMOKE_ONLY" == "1" ]]; then
  BATCH="16"
fi
CTC_BATCH="${CTC_BATCH:-192}"
if [[ "$SMOKE_ONLY" == "1" ]]; then
  CTC_BATCH="16"
fi

if [[ "$SKIP_SYLLABLE" != "1" ]]; then
  OUT="$RUN_ROOT/E1_syllable_closed_set"
  EPOCHS="180"
  [[ "$SMOKE_ONLY" == "1" ]] && EPOCHS="2"
  if run_step "E1_syllable_closed_set" "$OUT" "$PYTHON" \
    ipa_ocr_work/scripts/train_syllable_classifier.py \
    --eval-dir "$SYLL_DATA" \
    --out-dir "$OUT" \
    --variant syllable_crop \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH" \
    --height 64 \
    --width 192 \
    --lr 0.001 \
    --save-every 20; then
    run_step "E1_syllable_closed_set_row_score" "$OUT" "$PYTHON" \
      ipa_ocr_work/scripts/score_syllable_ocr_rows.py \
      --manifest "$SYLL_DATA/eval_manifest.tsv" \
      --predictions "$OUT/predictions_syllable_crop.tsv" \
      --out "$OUT/row_score.tsv" || true
  fi
fi

if [[ "$SKIP_ROW_CTC" != "1" ]]; then
  OUT="$RUN_ROOT/E2_variable_width_ctc_svtr_tiny"
  EPOCHS="220"
  [[ "$SMOKE_ONLY" == "1" ]] && EPOCHS="2"
  if run_step "E2_variable_width_ctc_svtr_tiny" "$OUT" "$PYTHON" \
    ipa_ocr_work/scripts/train_crnn_ipa_digits.py \
    --eval-dir "$ROW_DATA" \
    --out-dir "$OUT" \
    --variant original_export \
    --train-variants original_export \
    --epochs "$EPOCHS" \
    --batch-size "$CTC_BATCH" \
    --height 64 \
    --max-width 960 \
    --backbone svtr_tiny \
    --lr 0.0005 \
    --save-every 20; then
    PRED="$OUT/predictions_original_export.tsv"
    score_row_predictions "E2_variable_width_ctc_svtr_tiny" "$PRED" "$OUT" ipa || true
    decode_and_score "E2_variable_width_ctc_svtr_tiny" "$PRED" "$OUT" ipa || true
  fi
fi

if [[ "$SKIP_TROCR" != "1" ]]; then
  OUT="$RUN_ROOT/E3_trocr_pad_square_control"
  EPOCHS="4"
  [[ "$SMOKE_ONLY" == "1" ]] && EPOCHS="1"
  if run_step "E3_trocr_pad_square_control" "$OUT" "$PYTHON" \
    ipa_ocr_work/scripts/train_trocr_wupin.py \
    --eval-dir "$ROW_DATA" \
    --out-dir "$OUT" \
    --variant original_export \
    --train-variants original_export \
    --model microsoft/trocr-base-printed \
    --epochs "$EPOCHS" \
    --batch-size 4 \
    --lr 0.00001 \
    --max-label-length 64 \
    --label-source ipa-from-wupin \
    --image-mode pad-square; then
    PRED="$OUT/predictions_original_export.tsv"
    score_row_predictions "E3_trocr_pad_square_control" "$PRED" "$OUT" ipa || true
    decode_and_score "E3_trocr_pad_square_control" "$PRED" "$OUT" ipa || true
  fi
fi

write_status "finished"
echo "STATUS=$STATUS_PATH"
