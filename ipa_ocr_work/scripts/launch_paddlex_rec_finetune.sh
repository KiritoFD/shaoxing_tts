#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/shaoxing_tts}"
WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp/shaoxing_tts/paddlex_runs}"
VENV="${VENV:-$ROOT/.venv-paddleocr}"
RUN_NAME="${RUN_NAME:-paddlex_ppocrv5_mobile_rec_ipa_$(date +%Y%m%d_%H%M%S)}"
MODEL="${MODEL:-mobile}"
BATCH="${BATCH:-192}"
EPOCHS="${EPOCHS:-80}"
LR="${LR:-0.0005}"
DEVICE="${DEVICE:-gpu:0}"
LABEL_COLUMN="${LABEL_COLUMN:-label}"
SAVE_EPOCH_STEP="${SAVE_EPOCH_STEP:-5}"

source "$VENV/bin/activate"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/root/autodl-tmp/cache/pip}"
export PADDLE_HOME="${PADDLE_HOME:-/root/autodl-tmp/models/paddle}"
export PDX_CACHE_HOME="${PDX_CACHE_HOME:-/root/autodl-tmp/models/paddlex}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/models/huggingface}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

MANIFEST="${MANIFEST:-$ROOT/ipa_ocr_work/dataset/shaoxing_pdf136_clean/ocr_selected_all/eval_manifest.tsv}"
RUN_DIR="$WORK_ROOT/$RUN_NAME"
DATASET_DIR="$RUN_DIR/dataset"
OUTPUT_DIR="$RUN_DIR/output"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$RUN_DIR" "$LOG_DIR" "$OUTPUT_DIR"

python "$ROOT/ipa_ocr_work/scripts/prepare_paddlex_rec_dataset.py" \
  --manifest "$MANIFEST" \
  --out-dir "$DATASET_DIR" \
  --label-column "$LABEL_COLUMN"

PADDLEOCR_REPO="$VENV/lib/python3.12/site-packages/paddlex/repo_manager/repos/PaddleOCR"
if [[ "$MODEL" == "server" ]]; then
  BASE_CONFIG="$PADDLEOCR_REPO/configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml"
  PRETRAIN_URL="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_rec_pretrained.pdparams"
else
  BASE_CONFIG="$PADDLEOCR_REPO/configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml"
  PRETRAIN_URL="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_rec_pretrained.pdparams"
fi

CONFIG="$RUN_DIR/config.yaml"
python - "$BASE_CONFIG" "$CONFIG" "$DATASET_DIR" "$OUTPUT_DIR" "$DEVICE" "$BATCH" "$EPOCHS" "$LR" "$PRETRAIN_URL" "$SAVE_EPOCH_STEP" <<'PY'
from __future__ import annotations

import sys
import yaml
from pathlib import Path

base, out, dataset, output, device, batch, epochs, lr, pretrain_url, save_epoch_step = sys.argv[1:]
cfg = yaml.safe_load(Path(base).read_text(encoding="utf-8"))
cfg["Global"]["use_gpu"] = device.startswith("gpu")
cfg["Global"]["epoch_num"] = int(epochs)
cfg["Global"]["print_batch_step"] = 10
cfg["Global"]["save_model_dir"] = str(Path(output) / "checkpoints")
cfg["Global"]["save_epoch_step"] = int(save_epoch_step)
cfg["Global"]["eval_batch_step"] = [0, 10]
cfg["Global"]["pretrained_model"] = pretrain_url
cfg["Global"]["checkpoints"] = None
cfg["Global"]["character_dict_path"] = str(Path(dataset) / "dict.txt")
cfg["Global"]["max_text_length"] = 40
cfg["Global"]["use_space_char"] = False
cfg["Global"]["distributed"] = False
cfg["Global"]["save_res_path"] = str(Path(output) / "predicts.txt")
cfg["Optimizer"]["lr"]["learning_rate"] = float(lr)
cfg["Optimizer"]["lr"]["warmup_epoch"] = min(5, max(1, int(epochs) // 10))
for head in cfg["Architecture"]["Head"].get("head_list", []):
    if "NRTRHead" in head:
        head["NRTRHead"]["max_text_length"] = 40
for transform in cfg["Train"]["dataset"]["transforms"]:
    if "RecConAug" in transform:
        transform["RecConAug"]["max_text_length"] = 40
cfg["Train"]["dataset"]["data_dir"] = "/"
cfg["Train"]["dataset"]["label_file_list"] = [str(Path(dataset) / "train.txt")]
cfg["Train"]["sampler"]["first_bs"] = int(batch)
cfg["Train"]["loader"]["batch_size_per_card"] = int(batch)
cfg["Train"]["loader"]["num_workers"] = 8
cfg["Eval"]["dataset"]["data_dir"] = "/"
cfg["Eval"]["dataset"]["label_file_list"] = [str(Path(dataset) / "val.txt")]
cfg["Eval"]["loader"]["batch_size_per_card"] = min(int(batch), 256)
cfg["Eval"]["loader"]["num_workers"] = 4
Path(out).write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
PY

cd "$PADDLEOCR_REPO"
python tools/train.py -c "$CONFIG" > "$LOG_DIR/train.out" 2> "$LOG_DIR/train.err"
