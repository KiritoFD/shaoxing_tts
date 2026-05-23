"""Build an eval manifest from selected rows of a dataset manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build review eval manifest.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--qualities", nargs="+", default=["low_match"])
    parser.add_argument("--exclude-tone-position", nargs="*", default=["both_upper_lower", "no_image"])
    return parser.parse_args()


def split_for_page(page: int) -> str:
    if page % 10 == 9:
        return "test"
    if page % 10 == 8:
        return "val"
    return "train"


def relative_image_path(out_dir: Path, dataset_dir: Path, image: str) -> str:
    return (dataset_dir / image).resolve().relative_to(out_dir.resolve(), walk_up=True).as_posix()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.manifest, sep="\t", keep_default_na=False)
    df = df[
        df["image"].ne("")
        & df["ipa_digits"].ne("")
        & df["quality"].isin(args.qualities)
        & ~df["tone_position"].isin(args.exclude_tone_position)
    ].copy()
    df = df.sort_values(["page", "row_index"]).reset_index(drop=True)

    rows = []
    for idx, row in df.iterrows():
        page = int(row["page"])
        row_index = int(row["row_index"])
        rows.append(
            {
                "sample_id": f"p{page:03d}_{row_index:04d}_{idx:05d}",
                "variant": "original_export",
                "image": relative_image_path(args.out_dir, args.dataset_dir, row["image"]),
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
    pd.DataFrame(rows).to_csv(args.out_dir / "eval_manifest.tsv", sep="\t", index=False)
    print(f"rows: {len(rows)}")
    print(f"out: {args.out_dir}")


if __name__ == "__main__":
    main()
