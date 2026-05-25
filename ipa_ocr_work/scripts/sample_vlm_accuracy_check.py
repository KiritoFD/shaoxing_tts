"""Create a random VLM OCR sample sheet for visual accuracy checking."""

from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VLM = PROJECT_ROOT / "ipa_ocr_work" / "eval" / "vlm_pages" / "qwen36plus_post136_current_best.rows.csv"
DEFAULT_MANIFEST = PROJECT_ROOT / "ipa_ocr_work" / "eval" / "vlm_pages" / "post136_cropped_180dpi" / "page_manifest.tsv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "ipa_ocr_work" / "eval" / "vlm_pages" / "sample_accuracy_20"


TONE_TRANSLATION = str.maketrans(
    {
        "¹": "1",
        "²": "2",
        "³": "3",
        "⁴": "4",
        "⁵": "5",
        "⁶": "6",
        "⁷": "7",
        "⁸": "8",
        "⁹": "9",
        "⁰": "0",
        "₁": "1",
        "₂": "2",
        "₃": "3",
        "₄": "4",
        "₅": "5",
        "₆": "6",
        "₇": "7",
        "₈": "8",
        "₉": "9",
        "₀": "0",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a random VLM accuracy check sheet.")
    parser.add_argument("--vlm-csv", type=Path, default=DEFAULT_VLM)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--context-bands-before", type=int, default=1)
    parser.add_argument("--context-bands-after", type=int, default=3)
    return parser.parse_args()


def normalize_ipa(text: object) -> str:
    text = unicodedata.normalize("NFC", str(text or "")).strip().translate(TONE_TRANSLATION)
    return re.sub(r"\s+", "", text).replace("?", "ʔ").replace("ɔ", "ɒ")


def vlm_ipa_from_syllables(row: pd.Series) -> str:
    raw = str(row.get("syllables_json", "") or "").strip()
    if raw:
        try:
            syllables = json.loads(raw)
            if isinstance(syllables, list):
                finals = [str(item.get("final", "")) for item in syllables if isinstance(item, dict) and item.get("final")]
                if finals:
                    return normalize_ipa("".join(finals))
        except json.JSONDecodeError:
            pass
    return normalize_ipa(row.get("ipa", ""))


def detect_line_bands(image: Image.Image) -> list[tuple[int, int]]:
    gray = np.array(image.convert("L"))
    # Remove extreme margins for projection, but keep Chinese and IPA columns.
    x0 = int(gray.shape[1] * 0.12)
    x1 = int(gray.shape[1] * 0.92)
    crop = gray[:, x0:x1]
    dark = crop < 210
    projection = dark.sum(axis=1)
    active = projection > max(8, crop.shape[1] * 0.005)

    bands: list[tuple[int, int]] = []
    start = None
    for y, is_active in enumerate(active):
        if is_active and start is None:
            start = y
        elif not is_active and start is not None:
            if y - start >= 3:
                bands.append((start, y))
            start = None
    if start is not None:
        bands.append((start, len(active) - 1))

    merged: list[tuple[int, int]] = []
    for a, b in bands:
        if not merged or a - merged[-1][1] > 18:
            merged.append((a, b))
        else:
            merged[-1] = (merged[-1][0], b)

    height = gray.shape[0]
    filtered = []
    for a, b in merged:
        if a < 160 or b > height - 120:
            continue
        if b - a < 8:
            continue
        filtered.append((a, b))
    return filtered


def crop_for_row(
    image: Image.Image,
    row_ordinal: int,
    rows_on_page: int,
    context_before: int,
    context_after: int,
) -> tuple[Image.Image, str]:
    bands = detect_line_bands(image)
    if bands:
        idx = min(max(row_ordinal, 0), len(bands) - 1)
        band_start = max(0, idx - context_before)
        band_end = min(len(bands) - 1, idx + context_after)
        y0 = bands[band_start][0]
        y1 = bands[band_end][1]
        pad = 32
        top = max(0, y0 - pad)
        bottom = min(image.height, y1 + pad)
        return image.crop((0, top, image.width, bottom)), (
            f"band:{idx}/{len(bands)} context={band_start}-{band_end} y={top}-{bottom}"
        )

    # Fallback: evenly divide the useful page area.
    top_margin = int(image.height * 0.10)
    bottom_margin = int(image.height * 0.92)
    step = max(1, (bottom_margin - top_margin) / max(rows_on_page, 1))
    center = int(top_margin + (row_ordinal + 0.5) * step)
    top = max(0, center - 75)
    bottom = min(image.height, center + 75)
    return image.crop((0, top, image.width, bottom)), f"even y={top}-{bottom}"


def load_font(size: int) -> ImageFont.ImageFont:
    for path in [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    vlm = pd.read_csv(args.vlm_csv, keep_default_na=False)
    manifest = pd.read_csv(args.manifest, sep="\t", keep_default_na=False)
    image_by_pdf = {int(row.pdf_page): args.manifest.parent / str(row.image) for row in manifest.itertuples()}

    rng = random.Random(args.seed)
    sampled_indices = sorted(rng.sample(list(vlm.index), args.n))
    sample = vlm.loc[sampled_indices].copy().reset_index(names="vlm_row_index")
    rows_per_page = vlm.groupby("pdf_page").size().to_dict()

    records = []
    crops = []
    for i, row in sample.iterrows():
        pdf_page = int(row["pdf_page"])
        image_path = image_by_pdf[pdf_page]
        image = Image.open(image_path).convert("RGB")
        row_ordinal = int(row.get("row_ordinal", row.get("row_index", 0)) or 0)
        crop, crop_note = crop_for_row(
            image,
            row_ordinal,
            int(rows_per_page.get(pdf_page, 1)),
            args.context_bands_before,
            args.context_bands_after,
        )
        max_width = 1350
        if crop.width > max_width:
            scale = max_width / crop.width
            crop = crop.resize((max_width, max(1, int(crop.height * scale))), Image.Resampling.LANCZOS)
        crop_path = args.out_dir / f"sample_{i+1:02d}_pdf{pdf_page}_p{int(row['source_page'])}_r{row_ordinal}.png"
        crop.save(crop_path)
        rec = {
            "sample_id": i + 1,
            "vlm_row_index": int(row["vlm_row_index"]),
            "pdf_page": pdf_page,
            "source_page": int(row["source_page"]),
            "row_ordinal": row_ordinal,
            "headword": row.get("headword", ""),
            "vlm_ipa": vlm_ipa_from_syllables(row),
            "vlm_ipa_field": normalize_ipa(row.get("ipa", "")),
            "confidence": row.get("confidence", ""),
            "crop_path": str(crop_path),
            "crop_note": crop_note,
            "manual_headword_ok": "",
            "manual_ipa_segment_ok": "",
            "manual_tone_digits_ok": "",
            "manual_all_ok": "",
            "manual_note": "",
        }
        records.append(rec)
        crops.append((rec, crop))

    out_csv = args.out_dir / "vlm_random20_accuracy_check.csv"
    pd.DataFrame(records).to_csv(out_csv, index=False, encoding="utf-8-sig")

    font = load_font(22)
    small_font = load_font(18)
    sheet_width = 1560
    blocks = []
    for rec, crop in crops:
        header_h = 92
        block = Image.new("RGB", (sheet_width, header_h + crop.height + 18), "white")
        draw = ImageDraw.Draw(block)
        header = (
            f"#{rec['sample_id']} PDF {rec['pdf_page']} / source {rec['source_page']} / row {rec['row_ordinal']}    "
            f"{rec['headword']}    VLM: {rec['vlm_ipa']}"
        )
        draw.text((16, 10), header, fill=(0, 0, 0), font=font)
        draw.text((16, 48), rec["crop_note"], fill=(90, 90, 90), font=small_font)
        block.paste(crop, (16, header_h))
        draw.rectangle((12, header_h - 2, 20 + crop.width, header_h + crop.height + 2), outline=(220, 80, 50), width=3)
        blocks.append(block)

    sheet_h = sum(block.height for block in blocks)
    sheet = Image.new("RGB", (sheet_width, sheet_h), "white")
    y = 0
    for block in blocks:
        sheet.paste(block, (0, y))
        y += block.height
    sheet_path = args.out_dir / "vlm_random20_accuracy_check_sheet.png"
    sheet.save(sheet_path)
    print(json.dumps({
        "sample_csv": str(out_csv),
        "sheet": str(sheet_path),
        "n": len(records),
        "seed": args.seed,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
