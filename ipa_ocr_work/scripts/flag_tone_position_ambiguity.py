"""Flag OCR crops that show both upper and lower tone digits.

The trusted labels store ordinary digits, but the source book sometimes prints
two visual tone numbers around a syllable. Those samples should be reviewed
separately because the annotator's rule is to prefer the lower-right number
when it exists. This script adds an image-derived tone-position flag to an
exported manifest and builds a trainable subset that excludes ambiguous crops.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_ipa_matched_skip3"
DEFAULT_OUT = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_ipa_no_both_tone"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flag crops with both upper and lower tone digits.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifest-name", default="manifest.tsv")
    parser.add_argument("--x-pair-tolerance", type=float, default=38.0)
    parser.add_argument("--review-limit", type=int, default=120)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def component_flags(image_path: Path, x_pair_tolerance: float) -> dict:
    image = Image.open(image_path).convert("L")
    arr = np.asarray(image)
    _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num, _, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)

    components = []
    large_centers = []
    for idx in range(1, num):
        x, y, w, h, area = [float(v) for v in stats[idx]]
        cx, cy = [float(v) for v in centroids[idx]]
        if area < 20:
            continue
        # Main IPA letters are usually tall/wide. Their vertical center gives a
        # stable local baseline for this crop even when the crop is loose.
        if h > 50 or area > 900:
            large_centers.append(cy)
        # Tone digits are small connected components. Keep this permissive:
        # the review sheet is used to audit false positives.
        if 4 <= w <= 35 and 8 <= h <= 48 and 40 <= area <= 700:
            components.append({"x": x, "y": y, "w": w, "h": h, "area": area, "cx": cx, "cy": cy})

    if large_centers:
        baseline = float(np.median(large_centers))
    else:
        dark_y = np.where(binary > 0)[0]
        baseline = float(np.median(dark_y)) if len(dark_y) else image.height / 2.0

    upper = [c for c in components if c["cy"] < baseline - 30]
    lower = [c for c in components if c["y"] > baseline - 5]
    paired = [
        (u, l)
        for u in upper
        for l in lower
        if abs(u["cx"] - l["cx"]) <= x_pair_tolerance
    ]

    if paired:
        tone_position = "both_upper_lower"
    elif lower:
        tone_position = "lower_only"
    elif upper:
        tone_position = "upper_only"
    else:
        tone_position = "no_detected_tone_digits"

    return {
        "tone_position": tone_position,
        "tone_upper_components": len(upper),
        "tone_lower_components": len(lower),
        "tone_paired_components": len(paired),
        "tone_baseline_y": f"{baseline:.2f}",
    }


def relative_image_path(out_dir: Path, dataset_dir: Path, image: str) -> str:
    return (dataset_dir / image).resolve().relative_to(out_dir.resolve(), walk_up=True).as_posix()


def split_for_page(page: int) -> str:
    if page % 10 == 9:
        return "test"
    if page % 10 == 8:
        return "val"
    return "train"


def write_eval_manifest(df: pd.DataFrame, dataset_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, row in df.reset_index(drop=True).iterrows():
        page = int(row["page"])
        row_index = int(row["row_index"])
        rows.append(
            {
                "sample_id": f"p{page:03d}_{row_index:04d}_{idx:05d}",
                "variant": "original_export",
                "image": relative_image_path(out_dir, dataset_dir, row["image"]),
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
    pd.DataFrame(rows).to_csv(out_dir / "eval_manifest.tsv", sep="\t", index=False)


def write_review_sheet(df: pd.DataFrame, dataset_dir: Path, out_path: Path, limit: int) -> None:
    samples = df[df["tone_position"].eq("both_upper_lower") & df["image"].ne("")].head(limit)
    thumbs = []
    for _, row in samples.iterrows():
        image = Image.open(dataset_dir / row["image"]).convert("RGB")
        image.thumbnail((300, 90))
        tile = Image.new("RGB", (320, 128), "white")
        tile.paste(image, (0, 26))
        draw = ImageDraw.Draw(tile)
        draw.text((0, 0), f"p{int(row['page'])} r{int(row['row_index'])} {row['quality']}", fill="black")
        draw.text((0, 14), str(row["wupin"])[:48], fill="black")
        thumbs.append(tile)
    if not thumbs:
        return
    cols = 2
    sheet = Image.new("RGB", (cols * 320, ((len(thumbs) + cols - 1) // cols) * 128), "white")
    for idx, tile in enumerate(thumbs):
        sheet.paste(tile, ((idx % cols) * 320, (idx // cols) * 128))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> None:
    args = parse_args()
    manifest_path = args.dataset_dir / args.manifest_name
    df = pd.read_csv(manifest_path, sep="\t", keep_default_na=False)

    flags = []
    for _, row in df.iterrows():
        image_rel = row.get("image", "")
        image_path = args.dataset_dir / image_rel if image_rel else None
        if not image_path or not image_path.exists():
            flags.append(
                {
                    "tone_position": "no_image",
                    "tone_upper_components": 0,
                    "tone_lower_components": 0,
                    "tone_paired_components": 0,
                    "tone_baseline_y": "",
                }
            )
            continue
        flags.append(component_flags(image_path, args.x_pair_tolerance))

    flagged = pd.concat([df, pd.DataFrame(flags)], axis=1)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    flagged.to_csv(args.out_dir / "manifest_with_tone_flags.tsv", sep="\t", index=False)

    strict_trainable = flagged[
        flagged["image"].ne("")
        & flagged["ipa_digits"].ne("")
        & flagged["quality"].eq("matched")
        & ~flagged["tone_position"].eq("both_upper_lower")
    ].copy()
    trainable = flagged[
        flagged["image"].ne("")
        & flagged["ipa_digits"].ne("")
        & flagged["quality"].isin(["matched", "weak_match"])
        & ~flagged["tone_position"].eq("both_upper_lower")
    ].copy()
    strict_trainable.to_csv(args.out_dir / "manifest_trainable_strict_no_both_tone.tsv", sep="\t", index=False)
    trainable.to_csv(args.out_dir / "manifest_trainable_no_both_tone.tsv", sep="\t", index=False)
    write_eval_manifest(trainable, args.dataset_dir, args.out_dir)

    strict_eval_dir = args.out_dir / "strict_matched_only"
    write_eval_manifest(strict_trainable, args.dataset_dir, strict_eval_dir)

    ambiguous = flagged[flagged["tone_position"].eq("both_upper_lower")].copy()
    ambiguous.to_csv(args.out_dir / "ambiguous_both_tone.tsv", sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)
    write_review_sheet(ambiguous, args.dataset_dir, args.out_dir / "ambiguous_both_tone_review.png", args.review_limit)

    print("tone_position counts:")
    print(flagged["tone_position"].value_counts().to_string())
    print("\nquality x tone_position:")
    print(flagged.groupby(["quality", "tone_position"]).size().to_string())
    print(f"\nstrict_matched_no_both_tone: {len(strict_trainable)}")
    print(f"trainable_matched_plus_weak_no_both_tone: {len(trainable)}")
    print(f"out: {args.out_dir}")


if __name__ == "__main__":
    main()
