"""Evaluate a deterministic geometry detector for lower tone positions.

The detector uses the same visual source as the label builder: connected
components in the original row crop, split into syllable spans left-to-right.
It is intended as the high-precision front-end gate before any learned model.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROW_DATASET = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_ipa_matched_skip3"
DEFAULT_TRAINABLE = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_dual_model" / "trainable_rows.tsv"
DEFAULT_DETECTOR = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_dual_model" / "tone_position_detector" / "detector_manifest.tsv"
DEFAULT_OUT = PROJECT_ROOT / "ipa_ocr_work" / "reports" / "tone_geometry_detector_eval.tsv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate deterministic tone-position geometry detector.")
    parser.add_argument("--row-dataset", type=Path, default=DEFAULT_ROW_DATASET)
    parser.add_argument("--trainable-rows", type=Path, default=DEFAULT_TRAINABLE)
    parser.add_argument("--detector-manifest", type=Path, default=DEFAULT_DETECTOR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def visual_tone_hints(image_path: Path, syllables: list[dict[str, str]]) -> list[int]:
    image = Image.open(image_path).convert("L")
    arr = np.asarray(image)
    _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num, _, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)

    large_centers = []
    small_components = []
    for idx in range(1, num):
        x, y, w, h, area = [float(v) for v in stats[idx]]
        cx, cy = [float(v) for v in centroids[idx]]
        if area < 20:
            continue
        if h > 50 or area > 900:
            large_centers.append(cy)
        if 4 <= w <= 35 and 8 <= h <= 48 and 40 <= area <= 700:
            small_components.append({"x": x, "y": y, "w": w, "h": h, "cx": cx, "cy": cy})

    if large_centers:
        baseline = float(np.median(large_centers))
    else:
        dark_y = np.where(binary > 0)[0]
        baseline = float(np.median(dark_y)) if len(dark_y) else image.height / 2.0

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
        for row in syllables
    ]
    total = sum(weights)
    spans = []
    cur = left
    for weight in weights:
        nxt = cur + (right - left) * weight / total
        spans.append((cur, nxt))
        cur = nxt

    lower_components = [component for component in small_components if component["y"] > baseline - 5]
    predictions = []
    for start, end in spans:
        pad = max(10.0, (end - start) * 0.15)
        has_lower = any(start - pad <= component["cx"] <= end + pad for component in lower_components)
        predictions.append(1 if has_lower else 0)
    return predictions


def metrics(rows: list[dict[str, object]]) -> dict[str, float]:
    tp = sum(1 for row in rows if row["prediction"] == 1 and row["label"] == 1)
    tn = sum(1 for row in rows if row["prediction"] == 0 and row["label"] == 0)
    fp = sum(1 for row in rows if row["prediction"] == 1 and row["label"] == 0)
    fn = sum(1 for row in rows if row["prediction"] == 0 and row["label"] == 1)
    n = max(1, len(rows))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "rows": n,
        "accuracy": (tp + tn) / n,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main() -> None:
    args = parse_args()
    row_images = {
        (row["page"], row["row_index"]): row["image"]
        for row in read_tsv(args.trainable_rows)
        if row.get("image")
    }
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in read_tsv(args.detector_manifest):
        grouped[(row["page"], row["row_index"])].append(row)
    for group in grouped.values():
        group.sort(key=lambda row: int(row["syllable_index"]))

    out_rows = []
    for key, group in sorted(grouped.items(), key=lambda item: (int(item[0][0]), int(item[0][1]))):
        row_image = row_images.get(key)
        if not row_image:
            continue
        image_path = args.row_dataset / row_image
        if not image_path.exists():
            continue
        predictions = visual_tone_hints(image_path, group)
        for row, prediction in zip(group, predictions):
            out_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "source_split": row["source_split"],
                    "page": row["page"],
                    "row_index": row["row_index"],
                    "syllable_index": row["syllable_index"],
                    "label": int(row["label"]),
                    "prediction": prediction,
                    "correct": int(prediction == int(row["label"])),
                    "tone_policy": row["tone_policy"],
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(out_rows)

    for split in ["train", "val", "test", "all"]:
        split_rows = out_rows if split == "all" else [row for row in out_rows if row["source_split"] == split]
        result = metrics(split_rows)
        print(
            f"{split}: rows={result['rows']} acc={result['accuracy']:.4f} "
            f"precision={result['precision']:.4f} recall={result['recall']:.4f} f1={result['f1']:.4f} "
            f"tp={result['tp']} tn={result['tn']} fp={result['fp']} fn={result['fn']}"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
