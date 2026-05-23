"""Build tighter tone-position detector crops by clustering glyph components."""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROW_DATASET = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_ipa_matched_skip3"
DEFAULT_TRAINABLE = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_dual_model" / "trainable_rows.tsv"
DEFAULT_DETECTOR = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_dual_model" / "tone_position_detector" / "detector_manifest.tsv"
DEFAULT_OUT = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_tone_detector_clustered"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build clustered syllable crops for tone-position detection.")
    parser.add_argument("--row-dataset", type=Path, default=DEFAULT_ROW_DATASET)
    parser.add_argument("--trainable-rows", type=Path, default=DEFAULT_TRAINABLE)
    parser.add_argument("--detector-manifest", type=Path, default=DEFAULT_DETECTOR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def component_boxes(image: Image.Image, syllable_count: int) -> list[tuple[int, int, int, int]]:
    arr = np.asarray(image.convert("L"))
    height, width = arr.shape
    _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num, _, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)

    main = []
    for idx in range(1, num):
        x, y, w, h, area = [float(v) for v in stats[idx]]
        cx, cy = [float(v) for v in centroids[idx]]
        if area < 12:
            continue
        is_main = (
            h >= 0.38 * height
            or area >= 450
            or (h >= 0.25 * height and w >= 10 and 0.25 * height <= cy <= 0.88 * height)
        )
        if is_main:
            main.append({"x": x, "y": y, "w": w, "h": h, "area": area, "cx": cx, "cy": cy})

    if syllable_count <= 0:
        return []

    if len(main) < syllable_count:
        dark_x = np.where(binary > 0)[1]
        if len(dark_x):
            left = float(np.percentile(dark_x, 1))
            right = float(np.percentile(dark_x, 99))
        else:
            left, right = 0.0, float(width)
        if right <= left:
            right = float(width)
        return [
            (
                max(0, int(left + (right - left) * idx / syllable_count - 8)),
                0,
                min(width, int(left + (right - left) * (idx + 1) / syllable_count + 14)),
                height,
            )
            for idx in range(syllable_count)
        ]

    main.sort(key=lambda item: item["cx"])
    gaps = [
        (main[idx + 1]["x"] - (main[idx]["x"] + main[idx]["w"]), idx)
        for idx in range(len(main) - 1)
    ]
    cuts = sorted(idx for _, idx in sorted(gaps, reverse=True)[: syllable_count - 1])
    groups = []
    start = 0
    for cut in cuts:
        groups.append(main[start : cut + 1])
        start = cut + 1
    groups.append(main[start:])

    boxes = []
    for group_idx, group in enumerate(groups[:syllable_count]):
        x0 = min(item["x"] for item in group)
        x1 = max(item["x"] + item["w"] for item in group)
        prev_right = max((item["x"] + item["w"] for item in groups[group_idx - 1]), default=0) if group_idx > 0 else 0
        next_left = min((item["x"] for item in groups[group_idx + 1]), default=width) if group_idx + 1 < len(groups) else width
        left = max(0, int(max(x0 - 14, (prev_right + x0) / 2 if group_idx > 0 else x0 - 22)))
        right = min(width, int(min(next_left - 1, x1 + 55)))
        if group_idx + 1 == len(groups):
            right = min(width, int(x1 + 70))
        if right <= left + 4:
            left = max(0, int(x0 - 10))
            right = min(width, int(x1 + 30))
        boxes.append((left, 0, right, height))

    while len(boxes) < syllable_count:
        boxes.append((0, 0, width, height))
    return boxes


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and args.overwrite:
        shutil.rmtree(args.out_dir)
    image_dir = args.out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

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
        image = Image.open(image_path).convert("L")
        boxes = component_boxes(image, len(group))
        for row, box in zip(group, boxes):
            crop = image.crop(box)
            crop_name = f"{row['sample_id']}.png"
            crop.save(image_dir / crop_name)
            item = dict(row)
            item["image"] = f"images/{crop_name}"
            item["crop_bbox"] = ",".join(str(v) for v in box)
            out_rows.append(item)

    with (args.out_dir / "detector_manifest.tsv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(out_rows[0].keys()) if out_rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(out_rows)

    counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in out_rows:
        counts[(row["source_split"], row["label"])] += 1
    lines = [f"rows: {len(out_rows)}", "split x label:"]
    for key in sorted(counts):
        lines.append(f"{key[0]}\t{key[1]}\t{counts[key]}")
    (args.out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"out: {args.out_dir}")


if __name__ == "__main__":
    main()
