"""Export IPA OCR training crops using IPA labels derived from trusted Wu-pinyin.

Input labels come from result_all_converted.with_ipa.csv column ipa_from_wupin.
The crop logic is shared with export_wupin_training_data.py so the two datasets
can be compared directly.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

import fitz
import pandas as pd

from export_wupin_training_data import (
    PROJECT_ROOT,
    choose_candidate_rows,
    hanzi_len,
    norm_label,
    page_visual_rows,
    split_pages,
    write_text,
)


DEFAULT_PDF = PROJECT_ROOT / "ipa_ocr_work" / "data" / "shaoxing_123-351.pdf"
DEFAULT_LABEL_CSV = PROJECT_ROOT / "result_all_converted.with_ipa.csv"
DEFAULT_OUT = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_ipa_from_wupin"
SUPERSCRIPT_TO_DIGIT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export IPA OCR crops from derived IPA labels.")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--label-csv", type=Path, default=DEFAULT_LABEL_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--start-page", type=int, default=123)
    parser.add_argument("--end-page", type=int, default=351)
    parser.add_argument("--page-offset", type=int, default=123)
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--min-phonetic-width", type=float, default=28.0)
    return parser.parse_args()


def load_labels(path: Path, start_page: int, end_page: int) -> dict[int, list[dict]]:
    df = pd.read_csv(path, keep_default_na=False)
    required = {"page", "hanzi", "wupin", "ipa_from_wupin", "ipa_conversion_status"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")
    df = df[(df["page"] >= start_page) & (df["page"] <= end_page)].copy()
    df = df[df["ipa_conversion_status"].eq("ok")]
    df["label"] = df["ipa_from_wupin"].map(norm_label)
    df["label_digits"] = df["label"].map(lambda text: text.translate(SUPERSCRIPT_TO_DIGIT))
    df = df[df["label"] != ""]
    labels: dict[int, list[dict]] = {}
    for page, group in df.groupby("page", sort=True):
        labels[int(page)] = group.to_dict("records")
    return labels


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and args.overwrite:
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    labels_by_page = load_labels(args.label_csv, args.start_page, args.end_page)
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
                hanzi_count = hanzi_len(label_row.get("hanzi", ""))
                estimated_after_hanzi = visual.x0 + hanzi_count * 28.0 + 6.0
                crop_start = max(visual.crop_start_x - 3, estimated_after_hanzi)
                crop_start = min(crop_start, max(visual.x0, visual.x1 - args.min_phonetic_width))
                estimated_width = max(
                    args.min_phonetic_width,
                    min(visual.x1 - visual.crop_start_x, len(label_row.get("wupin", label)) * 9.0 + 24.0),
                )
                crop_end = max(crop_start + args.min_phonetic_width, min(visual.x1 + 4, crop_start + estimated_width))
                crop_bbox_tuple = (crop_start, visual.crop_y0, crop_end, visual.crop_y1)
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
                    "hanzi": label_row.get("hanzi", ""),
                    "wupin": label_row.get("wupin", ""),
                    "ipa": label,
                    "ipa_digits": label_row.get("label_digits", ""),
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
                "ipa",
                "ipa_digits",
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
