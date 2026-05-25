#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/shaoxing_tts}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
DATASET="${DATASET:-$ROOT/ipa_ocr_work/dataset/shaoxing_pdf136_clean/trocr_direct_span_v1_conf70}"
BASE_MODEL="${BASE_MODEL:-/root/shaoxing_runs/trocr_runs/trocr_E3_segmenter_span_p90_b36_sysdisk_from_b48best_lr5e7_e8_20260524_2032/best}"
RUN_ROOT="${RUN_ROOT:-/root/shaoxing_runs/trocr_runs}"
LOG_ROOT="${LOG_ROOT:-/root/shaoxing_runs/remote_launch_logs}"
RUN_NAME="${1:-trocr_E3_direct_span_conf70_b36_from_segspanbest_lr5e7_e16_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="$RUN_ROOT/$RUN_NAME"

mkdir -p "$RUN_ROOT" "$LOG_ROOT"

if [[ ! -f "$DATASET/eval_manifest.tsv" ]]; then
  echo "missing dataset manifest: $DATASET/eval_manifest.tsv" >&2
  exit 2
fi
if [[ ! -d "$BASE_MODEL" ]]; then
  echo "missing base model directory: $BASE_MODEL" >&2
  exit 2
fi

"$PYTHON" - "$DATASET" <<'PY'
import pandas as pd
import sys
from pathlib import Path

root = Path(sys.argv[1])
df = pd.read_csv(root / "eval_manifest.tsv", sep="\t", keep_default_na=False)
images = sum((root / p).exists() for p in df["image"])
print({"rows": len(df), "split": df["source_split"].value_counts().to_dict(), "images": images})
if images != len(df):
    raise SystemExit(f"missing images: {len(df) - images}")
PY

cd "$ROOT"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TRAIN_LOG="$LOG_ROOT/$RUN_NAME.train.log"
ERR_LOG="$LOG_ROOT/$RUN_NAME.train.err"

setsid "$PYTHON" ipa_ocr_work/scripts/train_trocr_wupin.py \
  --eval-dir "$DATASET" \
  --variant direct_phonetic_span_v1 \
  --model "$BASE_MODEL" \
  --out-dir "$OUT_DIR" \
  --epochs 16 \
  --batch-size 36 \
  --lr 5e-7 \
  --max-label-length 48 \
  --label-source ipa-from-wupin \
  --image-mode pad-square \
  --save-epoch-step 2 \
  --predict-epoch-step 2 \
  >"$TRAIN_LOG" 2>"$ERR_LOG" < /dev/null &

PID=$!
echo "pid=$PID"
echo "out_dir=$OUT_DIR"
echo "train_log=$TRAIN_LOG"
echo "err_log=$ERR_LOG"
sleep 8
ps -fp "$PID" || true
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
