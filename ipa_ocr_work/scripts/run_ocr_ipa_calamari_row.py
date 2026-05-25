"""Run OCR-IPA Calamari SavedModel on Shaoxing row OCR manifest."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_DIR = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_dual_model" / "ocr_selected"
DEFAULT_MODEL = PROJECT_ROOT / "ocr-ipa-main" / "model" / "calamari" / "best.ckpt"
DEFAULT_CHARSET = PROJECT_ROOT / "ocr-ipa-main" / "model" / "calamari" / "best.ckpt.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OCR-IPA Calamari on row manifest.")
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--charset-json", type=Path, default=DEFAULT_CHARSET)
    parser.add_argument("--variant", default="original_export")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--target-height", type=int, default=48)
    parser.add_argument("--output-key", default="root_3")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def scale_to_h(img: np.ndarray, target_height: int) -> np.ndarray:
    h, w = img.shape
    ratio = target_height / max(1, h)
    target_w = max(1, int(round(w * ratio)))
    return cv.resize(img, (target_w, target_height), interpolation=cv.INTER_AREA)


def preprocess_image(path: Path, target_height: int) -> np.ndarray:
    img = np.array(Image.open(path).convert("L"), dtype=np.uint8)
    _, binary = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    scaled = scale_to_h(binary, target_height)
    data = scaled.astype(np.float32) / 255.0
    data = np.expand_dims(data, axis=-1)
    data = np.amax(data) - data
    data = np.swapaxes(data, 1, 0)
    pad = 16
    data = np.vstack([np.zeros((pad, data.shape[1], 1), dtype=np.float32), data, np.zeros((pad, data.shape[1], 1), dtype=np.float32)])
    return (data * 255).astype(np.uint8)[np.newaxis, :, :, :]


def ctc_decode(logits: np.ndarray, charset: list[str]) -> str:
    ids = tf.argmax(logits, axis=-1).numpy()
    chars = []
    prev = -1
    for idx in ids:
        idx = int(idx)
        if idx != prev and 0 < idx < len(charset):
            chars.append(charset[idx])
        prev = idx
    return "".join(chars)


def main() -> None:
    args = parse_args()
    with args.charset_json.open("r", encoding="utf-8") as f:
        charset = json.load(f)["scenario"]["data"]["codec"]["charset"]
    model = tf.saved_model.load(str(args.model))
    infer = model.signatures["serving_default"]

    df = pd.read_csv(args.eval_dir / "eval_manifest.tsv", sep="\t", keep_default_na=False)
    df = df[df["variant"] == args.variant].sort_values(["source_split", "page", "row_index"]).reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    rows = []
    for idx, row in df.iterrows():
        img = preprocess_image(args.eval_dir / row["image"], args.target_height)
        result = infer(img=img, img_len=np.array([[img.shape[1]]], dtype=np.int32))
        output_key = args.output_key if args.output_key in result else sorted(result.keys())[-1]
        logits = result[output_key][0, :, :].numpy()
        pred = re.sub(r"\s+", "", ctc_decode(logits, charset).strip())
        rows.append({"sample_id": row["sample_id"], "variant": row["variant"], "prediction": pred})
        if (idx + 1) % 200 == 0:
            print(f"processed {idx + 1}/{len(df)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "variant", "prediction"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
