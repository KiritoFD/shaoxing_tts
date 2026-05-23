"""Create oracle or blank predictions for eval framework sanity checks."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL = PROJECT_ROOT / "ipa_ocr_work" / "eval" / "enhancement_ab" / "eval_manifest.tsv"
DEFAULT_OUT = PROJECT_ROOT / "ipa_ocr_work" / "eval" / "enhancement_ab"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create sanity-check predictions.")
    parser.add_argument("--eval-manifest", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--mode", choices=("oracle", "blank"), default="oracle")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.eval_manifest, sep="\t", keep_default_na=False)
    out = args.out or (DEFAULT_OUT / f"predictions_{args.mode}.tsv")
    pred = df[["sample_id", "variant"]].copy()
    pred["prediction"] = df["label"] if args.mode == "oracle" else ""
    out.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out, sep="\t", index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
