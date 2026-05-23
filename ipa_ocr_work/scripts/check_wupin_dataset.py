"""Quick checks for the exported Wu-pinyin OCR dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_wupin"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check exported Wu-pinyin dataset.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.dataset / "manifest.tsv"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    df = pd.read_csv(manifest_path, sep="\t")
    print("rows:", len(df))
    print("\nsplit counts:")
    print(df["split"].value_counts(dropna=False).to_string())
    print("\nquality counts:")
    print(df["quality"].value_counts(dropna=False).to_string())

    trainable = df[df["split"].isin(["train", "val", "test"])].copy()
    missing_images = []
    for image in trainable["image"].dropna():
        if not (args.dataset / image).exists():
            missing_images.append(image)
    print("\ntrainable rows:", len(trainable))
    print("missing trainable images:", len(missing_images))
    if missing_images:
        print("\n".join(missing_images[:20]))

    chars = sorted(set("".join(trainable["wupin"].fillna("").astype(str))))
    print("\ntrainable charset size:", len(chars))
    print("".join(chars))

    for split in ["train", "val", "test", "review"]:
        split_path = args.dataset / f"{split}.txt"
        if split_path.exists():
            lines = split_path.read_text(encoding="utf-8").splitlines()
            print(f"{split}.txt:", len(lines))


if __name__ == "__main__":
    main()
