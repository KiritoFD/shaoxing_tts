#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/shaoxing_tts}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
DATASET="${DATASET:-$ROOT/ipa_ocr_work/dataset/shaoxing_pdf136_clean/trocr_direct_span_v1_conf70}"
MODEL_DIR="${MODEL_DIR:-/root/shaoxing_runs/hf_models/trocr-base-printed}"
RUN_ROOT="${RUN_ROOT:-/root/shaoxing_runs/trocr_runs}"
LOG_ROOT="${LOG_ROOT:-/root/shaoxing_runs/remote_launch_logs}"
RUN_NAME="${1:-trocr_official_base_direct_conf70_b36_lr1e5_e24_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="$RUN_ROOT/$RUN_NAME"

mkdir -p "$RUN_ROOT" "$LOG_ROOT" "$(dirname "$MODEL_DIR")"
cd "$ROOT"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/shaoxing_runs/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DOWNLOAD_LOG="$LOG_ROOT/$RUN_NAME.download.log"
TRAIN_LOG="$LOG_ROOT/$RUN_NAME.train.log"
ERR_LOG="$LOG_ROOT/$RUN_NAME.train.err"

(
  set -euo pipefail
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

  if [[ ! -f "$MODEL_DIR/config.json" || ! -f "$MODEL_DIR/preprocessor_config.json" ]]; then
    "$PYTHON" - "$MODEL_DIR" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

target = Path(sys.argv[1])
target.mkdir(parents=True, exist_ok=True)
path = snapshot_download(
    repo_id="microsoft/trocr-base-printed",
    local_dir=str(target),
    local_dir_use_symlinks=False,
    resume_download=True,
)
print(path)
PY
  else
    echo "model already present: $MODEL_DIR"
  fi

  "$PYTHON" ipa_ocr_work/scripts/train_trocr_wupin.py \
    --eval-dir "$DATASET" \
    --variant direct_phonetic_span_v1 \
    --model "$MODEL_DIR" \
    --out-dir "$OUT_DIR" \
    --epochs 24 \
    --batch-size 36 \
    --lr 1e-5 \
    --max-label-length 48 \
    --label-source ipa-from-wupin \
    --image-mode pad-square \
    --save-epoch-step 0 \
    --predict-epoch-step 2
) >"$TRAIN_LOG" 2>"$ERR_LOG" < /dev/null &

PID=$!
echo "pid=$PID"
echo "out_dir=$OUT_DIR"
echo "model_dir=$MODEL_DIR"
echo "train_log=$TRAIN_LOG"
echo "err_log=$ERR_LOG"
sleep 8
ps -fp "$PID" || true
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
