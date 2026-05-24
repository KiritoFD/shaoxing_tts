"""Export high-agreement subsets from the merged VLM-vs-OCR CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IN = PROJECT_ROOT / "ipa_ocr_work" / "eval" / "vlm_pages" / "qwen36plus_post136_vlm_vs_trocr_merged.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "ipa_ocr_work" / "eval" / "vlm_pages"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export full-match and digit-match subsets.")
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prefix", default="qwen36plus_post136_vlm_vs_trocr")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input, keep_default_na=False)
    both = df[df["合并状态"].eq("两边都有")].copy()

    ipa_full = both[both["IPA完全相同"].eq("是")].copy()
    ipa_digits = both[both["IPA数字相同"].eq("是")].copy()
    wupin_full = both[both["吴拼完全相同"].eq("是")].copy()
    wupin_digits = both[both["吴拼数字相同"].eq("是")].copy()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "ipa_full_match_csv": args.out_dir / f"{args.prefix}.ipa_full_match.csv",
        "ipa_digits_match_csv": args.out_dir / f"{args.prefix}.ipa_digits_match.csv",
        "wupin_full_match_csv": args.out_dir / f"{args.prefix}.wupin_full_match.csv",
        "wupin_digits_match_csv": args.out_dir / f"{args.prefix}.wupin_digits_match.csv",
    }
    ipa_full.to_csv(paths["ipa_full_match_csv"], index=False, encoding="utf-8-sig")
    ipa_digits.to_csv(paths["ipa_digits_match_csv"], index=False, encoding="utf-8-sig")
    wupin_full.to_csv(paths["wupin_full_match_csv"], index=False, encoding="utf-8-sig")
    wupin_digits.to_csv(paths["wupin_digits_match_csv"], index=False, encoding="utf-8-sig")

    summary = {
        "input": str(args.input),
        "matched_rows": int(len(both)),
        "ipa_full_match_rows": int(len(ipa_full)),
        "ipa_digits_match_rows": int(len(ipa_digits)),
        "wupin_full_match_rows": int(len(wupin_full)),
        "wupin_digits_match_rows": int(len(wupin_digits)),
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    summary_path = args.out_dir / f"{args.prefix}.match_subsets.summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
