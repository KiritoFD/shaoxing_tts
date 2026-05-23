"""Export Wu-pinyin OCR training data from the Shaoxing PDF.

The manually entered labels are in result_all_converted.xlsx column "拼音".
The source image still contains the printed phonetic notation. The model is
trained to map the notation crop directly to Wu-pinyin text.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import fitz
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PDF = PROJECT_ROOT / "ipa_ocr_work" / "data" / "shaoxing_123-351.pdf"
DEFAULT_EXCEL = PROJECT_ROOT / "result_all_converted.xlsx"
DEFAULT_OUT = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_wupin"


@dataclass
class VisualRow:
    y: float
    x0: float
    x1: float
    y0: float
    y1: float
    text: str
    spans: list[dict]
    crop_start_x: float
    crop_y0: float
    crop_y1: float
    quality: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Wu-pinyin OCR crops.")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--excel", type=Path, default=DEFAULT_EXCEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--start-page", type=int, default=123)
    parser.add_argument("--end-page", type=int, default=351)
    parser.add_argument("--page-offset", type=int, default=123)
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--min-phonetic-width",
        type=float,
        default=28.0,
        help="Minimum crop width in PDF points for auto samples.",
    )
    return parser.parse_args()


def norm_label(value: object) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFC", str(value).strip())
    return re.sub(r"\s+", "", text)


def has_cjk(text: str) -> bool:
    return any("\u3400" <= ch <= "\u9fff" for ch in text)


def phoneticish_score(text: str) -> int:
    score = 0
    for ch in text:
        if ch.isspace():
            continue
        if ch.isascii() and (ch.isalpha() or ch.isdigit() or ch in "?~'`,.;:-)(/"):
            score += 1
            continue
        name = unicodedata.name(ch, "")
        if "LATIN" in name or "MODIFIER" in name or "COMBINING" in name:
            score += 1
    return score


def first_phonetic_span_index(spans: list[dict]) -> int | None:
    for i, span in enumerate(spans):
        text = span["text"]
        if phoneticish_score(text) >= max(1, len(text.strip()) // 2):
            return i
    return None


def crop_start_for_row(spans: list[dict]) -> float | None:
    if not spans:
        return None
    first = spans[0]
    first_text = first["text"]
    if has_cjk(first_text) and first["bbox"][0] < 280:
        return first["bbox"][2] + 4
    idx = first_phonetic_span_index(spans)
    if idx is None:
        return None
    return spans[idx]["bbox"][0] - 6


def load_labels(excel_path: Path, start_page: int, end_page: int) -> dict[int, list[dict]]:
    df = pd.read_excel(excel_path)
    required = {"页码", "汉字", "拼音"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Excel missing required columns: {sorted(missing)}")

    df = df[df["页码"].notna()].copy()
    df["页码"] = df["页码"].astype(int)
    df["label"] = df["拼音"].map(norm_label)
    df = df[(df["页码"] >= start_page) & (df["页码"] <= end_page)]
    df = df[df["label"] != ""]

    labels: dict[int, list[dict]] = {}
    for page, group in df.groupby("页码", sort=True):
        labels[int(page)] = group.to_dict("records")
    return labels


def page_visual_rows(page_obj: fitz.Page) -> list[VisualRow]:
    raw_lines = []
    for block in page_obj.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            x0, y0, x1, y1 = [float(v) for v in line["bbox"]]
            if y0 < 90 or y0 > 1315:
                continue
            spans = []
            for span in line.get("spans", []):
                text = span.get("text", "")
                if not text:
                    continue
                spans.append({"text": text, "bbox": tuple(float(v) for v in span["bbox"])})
            if spans:
                raw_lines.append((y0, x0, x1, y1, spans))

    raw_lines.sort(key=lambda item: (round(item[0] / 7) * 7, item[1]))
    groups: list[list[tuple]] = []
    for item in raw_lines:
        y0 = item[0]
        if not groups or abs(groups[-1][0][0] - y0) > 7:
            groups.append([item])
        else:
            groups[-1].append(item)

    rows: list[VisualRow] = []
    for group in groups:
        spans = [span for item in group for span in item[4]]
        spans.sort(key=lambda s: (s["bbox"][0], s["bbox"][1]))
        text = "".join(span["text"] for span in spans)
        row_x0 = min(item[1] for item in group)
        row_x1 = max(item[2] for item in group)
        row_y0 = min(item[0] for item in group)
        row_y1 = max(item[3] for item in group)
        score = phoneticish_score(text)
        if score < 3 and (row_x1 - row_x0) < 120:
            continue

        crop_start_x = crop_start_for_row(spans)
        if crop_start_x is None:
            continue

        quality = "auto"
        if row_x1 - crop_start_x < 28 or score < 5:
            quality = "review"
        if text.lstrip().startswith(("(", "（", '"', "“")):
            quality = "review"

        rows.append(
            VisualRow(
                y=row_y0,
                x0=row_x0,
                x1=row_x1,
                y0=row_y0,
                y1=row_y1,
                text=text,
                spans=spans,
                crop_start_x=crop_start_x,
                crop_y0=row_y0 - 6,
                crop_y1=row_y1 + 7,
                quality=quality,
            )
        )

    return rows


def choose_candidate_rows(rows: list[VisualRow], expected_count: int) -> list[VisualRow]:
    # Preserve reading order. Low-confidence rows are routed to review later;
    # dropping them here would shift all following labels.
    return rows[:expected_count]


def split_pages(pages: list[int], train_ratio: float, val_ratio: float, seed: int) -> dict[int, str]:
    shuffled = list(pages)
    random.Random(seed).shuffle(shuffled)
    n_train = int(len(shuffled) * train_ratio)
    n_val = int(len(shuffled) * val_ratio)
    train = set(shuffled[:n_train])
    val = set(shuffled[n_train : n_train + n_val])
    return {p: "train" if p in train else "val" if p in val else "test" for p in pages}


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def hanzi_len(value: object) -> int:
    text = "" if value is None else str(value)
    count = sum(1 for ch in text if "\u3400" <= ch <= "\u9fff")
    return count or len(text.strip())


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and args.overwrite:
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    labels_by_page = load_labels(args.excel, args.start_page, args.end_page)
    page_split = split_pages(sorted(labels_by_page), args.train_ratio, args.val_ratio, args.seed)
    doc = fitz.open(args.pdf)
    matrix = fitz.Matrix(args.dpi / 72.0, args.dpi / 72.0)

    paddle_lines = {"train": [], "val": [], "test": [], "review": []}
    charset = set()
    manifest_rows = []
    stats = {"written": 0, "review": 0, "missing": 0, "extra": 0}

    for page, page_labels in labels_by_page.items():
        page_obj = doc[page - args.page_offset]
        rows = page_visual_rows(page_obj)
        page_mismatch = len(rows) != len(page_labels)
        chosen = choose_candidate_rows(rows, len(page_labels))
        if len(rows) > len(page_labels):
            stats["extra"] += len(rows) - len(page_labels)
        if len(chosen) < len(page_labels):
            stats["missing"] += len(page_labels) - len(chosen)

        for idx, label_row in enumerate(page_labels):
            label = label_row["label"]
            split = page_split[page]
            quality = "missing"
            image_rel = ""
            pdf_text = ""
            crop_bbox = ""

            if idx < len(chosen):
                visual = chosen[idx]
                hanzi_count = hanzi_len(label_row.get("汉字", ""))
                # Hidden OCR often groups the leading Chinese word and phonetic
                # text into one span. Use the Excel word length to push the crop
                # start past the headword.
                estimated_after_hanzi = visual.x0 + hanzi_count * 28.0 + 6.0
                crop_start = max(visual.crop_start_x - 3, estimated_after_hanzi)
                crop_start = min(crop_start, max(visual.x0, visual.x1 - args.min_phonetic_width))
                estimated_width = max(args.min_phonetic_width, min(visual.x1 - visual.crop_start_x, len(label) * 9.0 + 24.0))
                crop_end = max(crop_start + args.min_phonetic_width, min(visual.x1 + 4, crop_start + estimated_width))
                crop_bbox_tuple = (
                    crop_start,
                    visual.crop_y0,
                    crop_end,
                    visual.crop_y1,
                )
                if page_mismatch or visual.quality != "auto" or (crop_bbox_tuple[2] - crop_bbox_tuple[0]) < args.min_phonetic_width:
                    split = "review"
                    stats["review"] += 1
                quality = "page_mismatch" if page_mismatch else visual.quality
                stem = f"page_{page:03d}_{idx:04d}"
                image_rel = f"{split}/images/{stem}.png"
                image_path = args.out_dir / image_rel
                gt_path = args.out_dir / split / "gt" / f"{stem}.gt.txt"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                page_obj.get_pixmap(matrix=matrix, clip=fitz.Rect(crop_bbox_tuple), alpha=False).save(str(image_path))
                write_text(gt_path, label + "\n")
                paddle_lines[split].append(f"{image_rel}\t{label}")
                charset.update(label)
                pdf_text = visual.text
                crop_bbox = repr(crop_bbox_tuple)
                stats["written"] += 1
            else:
                split = "review"
                quality = "missing"

            manifest_rows.append(
                {
                    "split": split,
                    "page": page,
                    "row_index": idx,
                    "hanzi": label_row.get("汉字", ""),
                    "wupin": label,
                    "image": image_rel,
                    "quality": quality,
                    "pdf_text": pdf_text,
                    "crop_bbox": crop_bbox,
                }
            )

    for split, lines in paddle_lines.items():
        write_text(args.out_dir / f"{split}.txt", "\n".join(lines) + ("\n" if lines else ""))
    write_text(args.out_dir / "charset.txt", "\n".join(sorted(charset)) + "\n")

    with (args.out_dir / "manifest.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "page",
                "row_index",
                "hanzi",
                "wupin",
                "image",
                "quality",
                "pdf_text",
                "crop_bbox",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"written: {stats['written']}")
    print(f"review: {stats['review']}")
    print(f"missing: {stats['missing']}")
    print(f"extra candidate rows: {stats['extra']}")
    print(f"charset size: {len(charset)}")
    print(f"out: {args.out_dir}")


if __name__ == "__main__":
    main()
