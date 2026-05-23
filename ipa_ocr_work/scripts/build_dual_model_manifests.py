"""Build OCR and tone-position detector manifests for the dual-model pipeline."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_ipa_matched_skip3"
DEFAULT_FLAGGED = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_ipa_no_both_tone" / "manifest_with_tone_flags.tsv"
DEFAULT_STRUCTURED = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_structured_tone_labels" / "structured_tone_syllables.tsv"
DEFAULT_LOW_EVAL = PROJECT_ROOT / "ipa_ocr_work" / "eval" / "ipa_digits_low_non_both_review" / "eval_manifest.tsv"
DEFAULT_LOW_PRED = PROJECT_ROOT / "ipa_ocr_work" / "eval" / "ipa_digits_low_non_both_review" / "predictions_strong_model.tsv"
DEFAULT_OUT = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_dual_model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dual-model training manifests.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--flagged-manifest", type=Path, default=DEFAULT_FLAGGED)
    parser.add_argument("--structured-syllables", type=Path, default=DEFAULT_STRUCTURED)
    parser.add_argument("--low-eval-manifest", type=Path, default=DEFAULT_LOW_EVAL)
    parser.add_argument("--low-predictions", type=Path, default=DEFAULT_LOW_PRED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--low-cer-threshold", type=float, default=0.2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def split_for_page(page: int) -> str:
    if page % 10 == 9:
        return "test"
    if page % 10 == 8:
        return "val"
    return "train"


def rel(out_dir: Path, target: Path) -> str:
    return target.resolve().relative_to(out_dir.resolve(), walk_up=True).as_posix()


def promoted_low_keys(low_eval_path: Path, low_pred_path: Path, threshold: float) -> set[tuple[int, int]]:
    if not low_eval_path.exists() or not low_pred_path.exists():
        return set()
    low_eval = pd.read_csv(low_eval_path, sep="\t", keep_default_na=False)
    low_pred = pd.read_csv(low_pred_path, sep="\t", keep_default_na=False)
    scored = low_eval.merge(low_pred[["sample_id", "cer"]], on="sample_id", how="inner")
    promoted = scored[scored["cer"] <= threshold]
    return set(zip(promoted["page"].astype(int), promoted["row_index"].astype(int)))


def trainable_rows(flagged: pd.DataFrame, promoted: set[tuple[int, int]]) -> pd.DataFrame:
    rows = []
    for _, row in flagged.iterrows():
        if not row.get("image", "") or not row.get("ipa_digits", ""):
            continue
        quality = row.get("quality", "")
        page = int(row["page"])
        row_index = int(row["row_index"])
        if quality in {"matched", "weak_match"}:
            rows.append(row.to_dict())
        elif quality == "low_match" and (page, row_index) in promoted:
            item = row.to_dict()
            item["quality"] = "promoted_low_cer0.2"
            rows.append(item)
    return pd.DataFrame(rows).sort_values(["page", "row_index"]).reset_index(drop=True)


def write_ocr_manifest(rows: pd.DataFrame, dataset_dir: Path, out_dir: Path) -> None:
    ocr_dir = out_dir / "ocr_selected"
    ocr_dir.mkdir(parents=True, exist_ok=True)
    out_rows = []
    for idx, row in rows.iterrows():
        page = int(row["page"])
        row_index = int(row["row_index"])
        out_rows.append(
            {
                "sample_id": f"p{page:03d}_{row_index:04d}_{idx:05d}",
                "variant": "original_export",
                "image": rel(ocr_dir, dataset_dir / row["image"]),
                "label": row["ipa_digits"],
                "page": page,
                "row_index": row_index,
                "hanzi": row.get("hanzi", ""),
                "source_split": split_for_page(page),
                "quality": row.get("quality", ""),
                "tone_position": row.get("tone_position", ""),
                "wupin": row.get("wupin", ""),
                "ipa": row.get("ipa", ""),
            }
        )
    pd.DataFrame(out_rows).to_csv(ocr_dir / "eval_manifest.tsv", sep="\t", index=False)


def dark_bounds_and_spans(image: Image.Image, syllables: pd.DataFrame) -> list[tuple[int, int]]:
    arr = np.asarray(image.convert("L"))
    _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark_x = np.where(binary > 0)[1]
    if len(dark_x):
        left = float(np.percentile(dark_x, 1))
        right = float(np.percentile(dark_x, 99))
    else:
        left, right = 0.0, float(image.width)
    if right <= left:
        right = float(image.width)
    weights = [
        max(1.0, len(str(row["ipa_base"])) + len(str(row["selected_tone"])) * 0.7)
        for _, row in syllables.iterrows()
    ]
    total = sum(weights)
    spans = []
    cur = left
    for weight in weights:
        nxt = cur + (right - left) * weight / total
        pad = max(10.0, (nxt - cur) * 0.2)
        spans.append((max(0, int(cur - pad)), min(image.width, int(nxt + pad))))
        cur = nxt
    return spans


def write_detector_manifest(rows: pd.DataFrame, structured: pd.DataFrame, dataset_dir: Path, out_dir: Path) -> None:
    det_dir = out_dir / "tone_position_detector"
    image_dir = det_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    trainable_keys = {(int(row["page"]), int(row["row_index"])) for _, row in rows.iterrows()}
    out_rows = []
    for (page, row_index), group in structured.groupby(["page", "row_index"], sort=True):
        key = (int(page), int(row_index))
        if key not in trainable_keys:
            continue
        image_rel = str(group.iloc[0]["image"])
        if not image_rel:
            continue
        image_path = dataset_dir / image_rel
        if not image_path.exists():
            continue
        image = Image.open(image_path).convert("L")
        spans = dark_bounds_and_spans(image, group)
        for span, (_, syl) in zip(spans, group.iterrows()):
            label = 1 if str(syl.get("lower_tone", "")) else 0
            sample_id = f"p{int(page):03d}_{int(row_index):04d}_s{int(syl['syllable_index']):02d}"
            crop = image.crop((span[0], 0, span[1], image.height))
            crop_rel = Path("images") / f"{sample_id}.png"
            crop.save(det_dir / crop_rel)
            out_rows.append(
                {
                    "sample_id": sample_id,
                    "image": crop_rel.as_posix(),
                    "label": label,
                    "page": int(page),
                    "row_index": int(row_index),
                    "syllable_index": int(syl["syllable_index"]),
                    "source_split": split_for_page(int(page)),
                    "quality": syl.get("quality", ""),
                    "tone_position": syl.get("tone_position", ""),
                    "tone_policy": syl.get("tone_policy", ""),
                    "wupin_base": syl.get("wupin_base", ""),
                    "ipa_base": syl.get("ipa_base", ""),
                    "selected_tone": syl.get("selected_tone", ""),
                    "upper_tone": syl.get("upper_tone", ""),
                    "lower_tone": syl.get("lower_tone", ""),
                }
            )
    pd.DataFrame(out_rows).to_csv(det_dir / "detector_manifest.tsv", sep="\t", index=False)


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and args.overwrite:
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    flagged = pd.read_csv(args.flagged_manifest, sep="\t", keep_default_na=False)
    structured = pd.read_csv(args.structured_syllables, sep="\t", keep_default_na=False)
    promoted = promoted_low_keys(args.low_eval_manifest, args.low_predictions, args.low_cer_threshold)
    rows = trainable_rows(flagged, promoted)
    rows.to_csv(args.out_dir / "trainable_rows.tsv", sep="\t", index=False)
    write_ocr_manifest(rows, args.dataset_dir, args.out_dir)
    write_detector_manifest(rows, structured, args.dataset_dir, args.out_dir)

    det = pd.read_csv(args.out_dir / "tone_position_detector" / "detector_manifest.tsv", sep="\t", keep_default_na=False)
    lines = [
        f"trainable OCR rows: {len(rows)}",
        f"promoted low rows: {len(promoted)}",
        "",
        "OCR row qualities:",
        rows["quality"].value_counts().to_string(),
        "",
        "OCR tone positions:",
        rows["tone_position"].value_counts().to_string(),
        "",
        f"detector syllables: {len(det)}",
        "detector labels:",
        det["label"].value_counts().to_string(),
        "",
        "detector split x label:",
        det.groupby(["source_split", "label"]).size().to_string(),
    ]
    (args.out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"out: {args.out_dir}")


if __name__ == "__main__":
    main()
