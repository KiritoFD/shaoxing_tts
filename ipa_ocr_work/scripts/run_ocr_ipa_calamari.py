"""Run the OCR-IPA Calamari SavedModel on the enhancement eval set."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_DIR = PROJECT_ROOT / "ipa_ocr_work" / "eval" / "enhancement_ab"
DEFAULT_MODEL = PROJECT_ROOT / "ocr-ipa-main" / "model" / "calamari" / "best.ckpt"
DEFAULT_CHARSET = PROJECT_ROOT / "ocr-ipa-main" / "model" / "calamari" / "best.ckpt.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OCR-IPA Calamari baseline.")
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--charset-json", type=Path, default=DEFAULT_CHARSET)
    parser.add_argument("--variants", nargs="+", default=["original", "superres_gray", "superres_otsu"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--target-height", type=int, default=48)
    parser.add_argument("--output-key", default="root_3")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def scale_to_h(img: np.ndarray, target_height: int) -> np.ndarray:
    h, w = img.shape
    if h == target_height:
        return img
    ratio = target_height / h
    target_w = max(1, int(round(w * ratio)))
    return cv.resize(img, (target_w, target_height), interpolation=cv.INTER_AREA)


def simple_prep(img: np.ndarray, invert: bool = True, transpose: bool = True, pad: int = 16) -> np.ndarray:
    data = img.astype(np.float32) / 255.0
    data = np.expand_dims(data, axis=-1)
    if invert:
        data = np.amax(data) - data
    if transpose:
        data = np.swapaxes(data, 1, 0)
    if pad > 0:
        if transpose:
            width = data.shape[1]
            data = np.vstack(
                [
                    np.zeros((pad, width, 1), dtype=np.float32),
                    data,
                    np.zeros((pad, width, 1), dtype=np.float32),
                ]
            )
        else:
            height = data.shape[0]
            data = np.hstack(
                [
                    np.zeros((height, pad, 1), dtype=np.float32),
                    data,
                    np.zeros((height, pad, 1), dtype=np.float32),
                ]
            )
    return (data * 255).astype(np.uint8).squeeze(-1)


def preprocess_image(path: Path, target_height: int) -> np.ndarray:
    img = np.array(Image.open(path).convert("L"), dtype=np.uint8)
    # Otsu is applied per crop because the model was trained on clean line images.
    _, binary = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    scaled = scale_to_h(binary, target_height)
    prepared = simple_prep(scaled)
    return prepared[np.newaxis, :, :, np.newaxis]


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


def normalize_prediction(text: str) -> str:
    # This model emits IPA-style text; keep a compact OCR string for scoring.
    return re.sub(r"\s+", "", text.strip())


def main() -> None:
    args = parse_args()
    with args.charset_json.open("r", encoding="utf-8") as f:
        charset = json.load(f)["scenario"]["data"]["codec"]["charset"]

    model = tf.saved_model.load(str(args.model))
    infer = model.signatures["serving_default"]

    df = pd.read_csv(args.eval_dir / "eval_manifest.tsv", sep="\t")
    df = df[df["variant"].isin(args.variants)].copy()
    if args.limit:
        df = df.groupby("variant", group_keys=False).head(args.limit)
    df = df.sort_values(["variant", "sample_id"]).reset_index(drop=True)

    rows = []
    for i, row in df.iterrows():
        img = preprocess_image(args.eval_dir / row["image"], args.target_height)
        result = infer(img=img, img_len=np.array([[img.shape[1]]], dtype=np.int32))
        output_key = args.output_key if args.output_key in result else sorted(result.keys())[-1]
        logits = result[output_key][0, :, :].numpy()
        pred = normalize_prediction(ctc_decode(logits, charset))
        rows.append(
            {
                "sample_id": row["sample_id"],
                "variant": row["variant"],
                "prediction": pred,
            }
        )
        if (i + 1) % 50 == 0 or i + 1 == len(df):
            print(f"processed {i + 1}/{len(df)}")

    out = args.out or (args.eval_dir / "predictions_ocr_ipa_calamari.tsv")
    pd.DataFrame(rows).to_csv(out, sep="\t", index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
