"""Prepare a PaddleX text-recognition dataset from a Shaoxing OCR manifest."""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import unicodedata
from pathlib import Path


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", value.strip()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--label-column",
        default="label",
        choices=["label", "ipa", "wupin"],
        help="Column to use as recognition ground truth.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Optional base directory for manifest image paths. Defaults to manifest parent.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images into the PaddleX dataset instead of using absolute paths.",
    )
    return parser.parse_args()


def split_name(row: dict[str, str]) -> str:
    split = (row.get("source_split") or row.get("split") or "").strip().lower()
    if split in {"train", "val", "test"}:
        return split
    # Weak/review rows in the current clean manifest are training samples unless
    # the split has already been fixed upstream.
    return "train"


def main() -> None:
    args = parse_args()
    manifest = args.manifest.resolve()
    image_root = (args.image_root or manifest.parent).resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_by_split: dict[str, list[tuple[str, str, str]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    charset: set[str] = set()
    copied = 0
    skipped = 0

    with manifest.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("image_exists", "True").strip().lower() == "false":
                skipped += 1
                continue
            label = normalize_text(row.get(args.label_column, ""))
            if not label:
                skipped += 1
                continue
            image_value = row.get("image", "").strip()
            if not image_value:
                skipped += 1
                continue
            src = Path(image_value)
            if not src.is_absolute():
                src = (image_root / src).resolve()
            if not src.exists():
                skipped += 1
                continue

            if args.copy_images:
                rel = Path("images") / split_name(row) / src.name
                dst = out_dir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    shutil.copy2(src, dst)
                    copied += 1
                image_for_list = rel.as_posix()
            else:
                image_for_list = src.as_posix()

            for ch in label:
                charset.add(ch)
            rows_by_split[split_name(row)].append(
                (image_for_list, label, row.get("sample_id", ""))
            )

    # PaddleX requires train.txt and val.txt. Keep test.txt for our own scorer.
    for split, rows in rows_by_split.items():
        with (out_dir / f"{split}.txt").open("w", encoding="utf-8", newline="\n") as f:
            for image_path, label, _sample_id in rows:
                f.write(f"{image_path}\t{label}\n")
        with (out_dir / f"{split}.samples.tsv").open("w", encoding="utf-8", newline="\n") as f:
            f.write("image\tlabel\tsample_id\n")
            for image_path, label, sample_id in rows:
                f.write(f"{image_path}\t{label}\t{sample_id}\n")

    with (out_dir / "dict.txt").open("w", encoding="utf-8", newline="\n") as f:
        for ch in sorted(charset):
            f.write(ch + "\n")

    with (out_dir / "summary.tsv").open("w", encoding="utf-8", newline="\n") as f:
        f.write("key\tvalue\n")
        f.write(f"manifest\t{manifest.as_posix()}\n")
        f.write(f"label_column\t{args.label_column}\n")
        f.write(f"train\t{len(rows_by_split['train'])}\n")
        f.write(f"val\t{len(rows_by_split['val'])}\n")
        f.write(f"test\t{len(rows_by_split['test'])}\n")
        f.write(f"charset\t{len(charset)}\n")
        f.write(f"copied_images\t{copied}\n")
        f.write(f"skipped\t{skipped}\n")


if __name__ == "__main__":
    main()
