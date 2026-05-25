"""Build cleaner syllable OCR crops from trusted row crops.

The first syllable-crop builder reused tone-detector component boxes. That is
too loose for OCR: it often includes neighboring syllables or isolates only a
tone digit. This builder uses the known syllable sequence as a weak alignment
guide, snaps inter-syllable cuts to low-ink vertical projection valleys, then
tightens each segment around its own ink.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from wupin_ipa_convert import DEFAULT_MAP, canonicalize_wupin_base, load_mapping, wupin_syllable_to_ipa


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROW_SOURCE = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_ipa_matched_skip3"
DEFAULT_CLEAN_FLAGS = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_pdf136_clean" / "row_cleaning_flags.tsv"
DEFAULT_STRUCTURED = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_structured_tone_labels" / "structured_tone_syllables.tsv"
DEFAULT_OUT = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_pdf136_clean" / "syllable_ocr_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cleaner syllable OCR crops.")
    parser.add_argument("--row-source", type=Path, default=DEFAULT_ROW_SOURCE)
    parser.add_argument("--row-cleaning-flags", type=Path, default=DEFAULT_CLEAN_FLAGS)
    parser.add_argument("--structured-syllables", type=Path, default=DEFAULT_STRUCTURED)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--subset", choices=["all", "strict"], default="all")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--margin-x", type=int, default=5)
    parser.add_argument("--min-width", type=int, default=10)
    parser.add_argument("--debug-contact-sheet", action="store_true")
    return parser.parse_args()


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", keep_default_na=False)


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)


def split_for_row(row: pd.Series) -> str:
    split = str(row.get("split_row", row.get("split", "")))
    return "train" if split == "review" else split


def ink_binary(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image.convert("L"))
    if arr.size == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    blur = cv2.GaussianBlur(arr, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Remove isolated specks but keep small tone digits.
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    clean = np.zeros_like(binary)
    for idx in range(1, num):
        x, y, w, h, area = stats[idx]
        if area >= 6 and h >= 2 and w >= 1:
            clean[labels == idx] = 255
    return clean


def smooth_projection(binary: np.ndarray) -> np.ndarray:
    projection = (binary > 0).sum(axis=0).astype(np.float32)
    if projection.size < 7:
        return projection
    kernel = np.array([1, 2, 3, 2, 1], dtype=np.float32)
    kernel /= kernel.sum()
    return np.convolve(projection, kernel, mode="same")


def dark_bounds(binary: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(binary > 0)
    height, width = binary.shape
    if len(xs) == 0:
        return 0, width, 0, height
    left = max(0, int(np.percentile(xs, 0.5)) - 1)
    right = min(width, int(np.percentile(xs, 99.5)) + 2)
    top = max(0, int(np.percentile(ys, 0.5)) - 2)
    bottom = min(height, int(np.percentile(ys, 99.5)) + 3)
    if right <= left:
        right = min(width, left + 1)
    if bottom <= top:
        bottom = min(height, top + 1)
    return left, right, top, bottom


def visual_weight(row: pd.Series) -> float:
    base = str(row.get("ipa_base", ""))
    tone = str(row.get("selected_tone", ""))
    # IPA symbols are not equal-width, but base character count is a useful weak
    # prior. Tone digits are narrow and often stacked, so count them lightly.
    return max(1.0, len(base) * 1.0 + len(tone) * 0.45)


def find_valley(projection: np.ndarray, expected: float, left_limit: int, right_limit: int) -> int:
    width = len(projection)
    left = max(1, min(width - 2, int(left_limit)))
    right = max(left + 1, min(width - 1, int(right_limit)))
    center = int(round(expected))
    radius = max(10, int((right - left) * 0.35))
    lo = max(left, center - radius)
    hi = min(right, center + radius)
    if hi <= lo:
        lo, hi = left, right
    segment = projection[lo:hi]
    if len(segment) == 0:
        return center
    min_val = float(segment.min())
    candidates = np.where(segment <= min_val + 0.5)[0] + lo
    # Prefer the valley closest to the expected boundary to avoid jumping into a
    # neighboring wide whitespace region.
    return int(candidates[np.argmin(np.abs(candidates - expected))])


def segment_boxes(image: Image.Image, syllables: pd.DataFrame, margin_x: int, min_width: int) -> list[tuple[int, int, int, int]]:
    binary = ink_binary(image)
    height, width = binary.shape
    if len(syllables) <= 0:
        return []
    left, right, top, bottom = dark_bounds(binary)
    if len(syllables) == 1:
        return [(max(0, left - margin_x), 0, min(width, right + margin_x), height)]

    projection = smooth_projection(binary)
    weights = [visual_weight(row) for _, row in syllables.iterrows()]
    total = sum(weights)
    usable_left, usable_right = left, right
    usable_width = max(1, usable_right - usable_left)
    expected_cuts = []
    acc = 0.0
    for weight in weights[:-1]:
        acc += weight
        expected_cuts.append(usable_left + usable_width * acc / total)

    raw_cuts = []
    prev = usable_left
    for idx, expected in enumerate(expected_cuts):
        next_expected = expected_cuts[idx + 1] if idx + 1 < len(expected_cuts) else usable_right
        left_limit = (prev + expected) / 2
        right_limit = (expected + next_expected) / 2
        cut = find_valley(projection, expected, int(left_limit), int(right_limit))
        if raw_cuts and cut <= raw_cuts[-1] + min_width:
            cut = min(width - 1, raw_cuts[-1] + min_width)
        raw_cuts.append(cut)
        prev = cut

    edges = [0] + raw_cuts + [width]
    boxes = []
    for idx in range(len(syllables)):
        seg_left = max(0, edges[idx])
        seg_right = min(width, edges[idx + 1])
        region = binary[:, seg_left:seg_right]
        ys, xs = np.where(region > 0)
        if len(xs):
            x0 = seg_left + int(xs.min())
            x1 = seg_left + int(xs.max()) + 1
            y0 = int(ys.min())
            y1 = int(ys.max()) + 1
        else:
            x0, x1, y0, y1 = seg_left, seg_right, 0, height
        x0 = max(0, max(seg_left, x0 - margin_x))
        x1 = min(width, min(seg_right, x1 + margin_x))
        if x1 < x0 + min_width:
            mid = (x0 + x1) // 2
            x0 = max(seg_left, mid - min_width // 2)
            x1 = min(seg_right, x0 + min_width)
        boxes.append((int(x0), 0, int(max(x1, x0 + 1)), height))
    return boxes


def make_manifest(args: argparse.Namespace) -> pd.DataFrame:
    flags = read_tsv(args.row_cleaning_flags)
    structured = read_tsv(args.structured_syllables)
    mapping = load_mapping(args.mapping)

    keep_col = "clean_all" if args.subset == "all" else "clean_strict"
    cols = [
        "page",
        "source_page",
        "pdf_page",
        "row_index",
        "image",
        "quality",
        "split",
        "clean_all",
        "clean_strict",
        "cleaning_flags",
    ]
    merged = structured.merge(flags[cols], on=["page", "row_index"], how="left", suffixes=("", "_row"))
    selected = merged[merged[keep_col].fillna(False)].copy()

    image_dir = args.out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    out_rows = []
    for (page, row_index), group in selected.groupby(["page", "row_index"], sort=True):
        group = group.sort_values("syllable_index").reset_index(drop=True)
        image_rel = str(group.iloc[0].get("image_row", ""))
        image_path = args.row_source / image_rel
        if not image_path.exists():
            continue
        image = Image.open(image_path).convert("L")
        boxes = segment_boxes(image, group, args.margin_x, args.min_width)
        if len(boxes) != len(group):
            continue
        for box, (_, row) in zip(boxes, group.iterrows()):
            wupin_base = canonicalize_wupin_base(row.get("wupin_base", ""))
            tone = str(row.get("selected_tone", ""))
            ipa_label, ipa_error = wupin_syllable_to_ipa(wupin_base, tone, mapping, "digits")
            label = ipa_label if not ipa_error else f"{row.get('ipa_base', '')}{tone}"
            sample_id = f"p{int(page):03d}_{int(row_index):04d}_s{int(row['syllable_index']):02d}"
            crop_name = f"{sample_id}.png"
            image.crop(box).save(image_dir / crop_name)
            out_rows.append(
                {
                    "sample_id": sample_id,
                    "variant": "syllable_crop",
                    "image": f"images/{crop_name}",
                    "label": label,
                    "source_split": split_for_row(row),
                    "original_source_split": row.get("split_row", row.get("split", "")),
                    "page": int(page),
                    "source_page": row.get("source_page", page),
                    "pdf_page": row.get("pdf_page", ""),
                    "row_index": int(row_index),
                    "syllable_index": int(row["syllable_index"]),
                    "tone_policy": row.get("tone_policy", ""),
                    "wupin_base": wupin_base,
                    "ipa_base": label[: -len(tone)] if tone else label,
                    "legacy_wupin_base": row.get("wupin_base", ""),
                    "legacy_ipa_base": row.get("ipa_base", ""),
                    "selected_tone": tone,
                    "ipa_conversion_status": "ok" if not ipa_error else ipa_error,
                    "cleaning_flags": row.get("cleaning_flags", ""),
                    "quality": row.get("quality_row", row.get("quality", "")),
                    "crop_bbox": ",".join(str(v) for v in box),
                    "crop_width": box[2] - box[0],
                }
            )
    return pd.DataFrame(out_rows)


def write_contact_sheet(df: pd.DataFrame, out_dir: Path, seed: int = 11, n: int = 48) -> None:
    if df.empty:
        return
    sample = df[df["source_split"].eq("val")].sample(min(n, int(df["source_split"].eq("val").sum())), random_state=seed)
    thumb_w, thumb_h, label_h, cols = 180, 88, 34, 6
    rows = math.ceil(len(sample) / cols)
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), "white")
    from PIL import ImageDraw

    draw = ImageDraw.Draw(sheet)
    for idx, row in enumerate(sample.itertuples()):
        image = Image.open(out_dir / row.image).convert("RGB")
        image.thumbnail((thumb_w - 8, thumb_h - 8), Image.Resampling.LANCZOS)
        x = (idx % cols) * thumb_w
        y = (idx // cols) * (thumb_h + label_h)
        sheet.paste(image, (x + 4, y + 4))
        draw.text((x + 4, y + thumb_h), f"{row.sample_id} {row.label}", fill=(0, 0, 0))
    sheet.save(out_dir / "contact_sheet_val.png")


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and args.overwrite:
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = make_manifest(args)
    write_tsv(df, args.out_dir / "eval_manifest.tsv")

    lines = [f"rows: {len(df)}", f"unique_labels: {df['label'].nunique() if len(df) else 0}", "split counts:"]
    if len(df):
        for (split, quality), count in df.groupby(["source_split", "quality"]).size().items():
            lines.append(f"{split}\t{quality}\t{count}")
        lines.extend(
            [
                "",
                "crop_width_summary:",
                df["crop_width"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_string(),
            ]
        )
    (args.out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if args.debug_contact_sheet:
        write_contact_sheet(df, args.out_dir)
    print("\n".join(lines))
    print(f"wrote {args.out_dir / 'eval_manifest.tsv'}")


if __name__ == "__main__":
    main()
