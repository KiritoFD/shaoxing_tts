"""Combine matched crops with index-based fallback crops.

Priority:
1. matched/weak_match crops from shaoxing_ipa_matched
2. fallback crops from shaoxing_ipa_from_wupin for labels still lacking images

This keeps as many labeled samples as possible while preserving a quality/source
flag so noisy fallback rows can be downweighted or filtered later.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATCHED = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_ipa_matched"
DEFAULT_FALLBACK = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_ipa_from_wupin"
DEFAULT_OUT = PROJECT_ROOT / "ipa_ocr_work" / "eval" / "ipa_digits_combined_all"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build combined IPA+digit manifest.")
    parser.add_argument("--matched-dir", type=Path, default=DEFAULT_MATCHED)
    parser.add_argument("--fallback-dir", type=Path, default=DEFAULT_FALLBACK)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    return parser.parse_args()


def rel(out_dir: Path, root: Path, image: str) -> str:
    return (root / image).resolve().relative_to(out_dir.resolve(), walk_up=True).as_posix()


def key(row: pd.Series) -> tuple[int, int, str]:
    return int(row["page"]), int(row["row_index"]), str(row.get("wupin", ""))


def assign_splits(keys: list[tuple[int, int, str]], train_ratio: float, val_ratio: float, seed: int) -> dict:
    shuffled = list(keys)
    random.Random(seed).shuffle(shuffled)
    train_cut = int(len(shuffled) * train_ratio)
    val_cut = train_cut + int(len(shuffled) * val_ratio)
    out = {}
    for k in shuffled[:train_cut]:
        out[k] = "train"
    for k in shuffled[train_cut:val_cut]:
        out[k] = "val"
    for k in shuffled[val_cut:]:
        out[k] = "test"
    return out


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    matched = pd.read_csv(args.matched_dir / "manifest.tsv", sep="\t", keep_default_na=False)
    fallback = pd.read_csv(args.fallback_dir / "manifest.tsv", sep="\t", keep_default_na=False)
    matched = matched[matched["image"] != ""].copy()
    fallback = fallback[fallback["image"] != ""].copy()

    chosen = {}
    for _, row in fallback.iterrows():
        k = key(row)
        chosen[k] = {
            "root": args.fallback_dir,
            "row": row,
            "source": f"fallback_{row.get('quality', '')}",
            "quality_rank": 2,
        }
    for _, row in matched.iterrows():
        k = key(row)
        rank = 0 if row["quality"] == "matched" else 1
        current = chosen.get(k)
        if current is None or rank <= current["quality_rank"]:
            chosen[k] = {
                "root": args.matched_dir,
                "row": row,
                "source": row["quality"],
                "quality_rank": rank,
            }

    split_by_key = assign_splits(list(chosen), args.train_ratio, args.val_ratio, args.seed)
    rows = []
    for idx, (k, item) in enumerate(sorted(chosen.items())):
        row = item["row"]
        label = row.get("ipa_digits", "")
        if not label:
            continue
        rows.append(
            {
                "sample_id": f"p{int(row['page']):03d}_{int(row['row_index']):04d}_{idx:05d}",
                "variant": "combined",
                "image": rel(args.out_dir, item["root"], row["image"]),
                "label": label,
                "page": int(row["page"]),
                "row_index": int(row["row_index"]),
                "hanzi": row.get("hanzi", ""),
                "source_split": split_by_key[k],
                "quality": item["source"],
                "wupin": row.get("wupin", ""),
                "ipa": row.get("ipa", ""),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(args.out_dir / "eval_manifest.tsv", sep="\t", index=False)
    print(f"combined rows: {len(out)}")
    print(out.groupby(["source_split", "quality"]).size().to_string())
    print(f"out: {args.out_dir}")


if __name__ == "__main__":
    main()
