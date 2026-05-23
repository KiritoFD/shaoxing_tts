"""Build an eval-style manifest from every exported crop with a trusted label.

The normal enhancement eval builder only keeps `quality == auto`; this script is
for larger training runs where we intentionally include review/page-mismatch
crops too. Rows without an image remain excluded because there is nothing to
train on.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_ipa_from_wupin"
DEFAULT_OUT = PROJECT_ROOT / "ipa_ocr_work" / "eval" / "ipa_digits_all_available"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build all-available IPA+digit training manifest.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--label-column", default="ipa_digits")
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    return parser.parse_args()


def relative_image_path(out_dir: Path, dataset_dir: Path, image: str) -> str:
    target = dataset_dir / image
    return target.resolve().relative_to(out_dir.resolve(), walk_up=True).as_posix()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.dataset_dir / "manifest.tsv"
    df = pd.read_csv(manifest_path, sep="\t", keep_default_na=False)
    if args.label_column not in df.columns:
        raise ValueError(f"manifest has no label column {args.label_column!r}")

    df = df[(df["image"] != "") & (df[args.label_column] != "")].copy()
    df = df.sort_values(["page", "row_index"]).reset_index(drop=True)

    ids = list(range(len(df)))
    rng = random.Random(args.seed)
    rng.shuffle(ids)
    train_cut = int(len(ids) * args.train_ratio)
    val_cut = train_cut + int(len(ids) * args.val_ratio)
    split_by_idx = {}
    for idx in ids[:train_cut]:
        split_by_idx[idx] = "train"
    for idx in ids[train_cut:val_cut]:
        split_by_idx[idx] = "val"
    for idx in ids[val_cut:]:
        split_by_idx[idx] = "test"

    rows = []
    for idx, row in df.iterrows():
        page = int(row["page"])
        row_index = int(row["row_index"])
        sample_id = f"p{page:03d}_{row_index:04d}_{idx:05d}"
        rows.append(
            {
                "sample_id": sample_id,
                "variant": "original_export",
                "image": relative_image_path(args.out_dir, args.dataset_dir, row["image"]),
                "label": row[args.label_column],
                "page": page,
                "row_index": row_index,
                "hanzi": row.get("hanzi", ""),
                "source_split": split_by_idx[idx],
                "quality": row.get("quality", ""),
                "original_split": row.get("split", ""),
                "wupin": row.get("wupin", ""),
                "ipa": row.get("ipa", ""),
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out_dir / "eval_manifest.tsv", sep="\t", index=False)
    print(f"input rows with images: {len(out_df)}")
    print(out_df.groupby(["source_split", "quality"]).size().to_string())
    print(f"out: {args.out_dir}")


if __name__ == "__main__":
    main()
