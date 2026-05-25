"""Build OCR manifests whose reliability is judged by phonetic labels.

The first-stage exporter assigns `quality` from Chinese headword matching.
For IPA OCR, a CJK candidate mismatch is not fatal: the crop can still contain
the correct phonetic string. This variant keeps rows that are parseable,
image-backed, and at least weakly aligned, while excluding the visibly risky
low-match and bracket-context rows.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLEAN_ROOT = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_pdf136_clean"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build phonetic-reliable OCR manifests.")
    parser.add_argument("--clean-root", type=Path, default=DEFAULT_CLEAN_ROOT)
    parser.add_argument("--include-low", action="store_true", help="Also include low_match rows.")
    parser.add_argument("--include-bracket-context", action="store_true")
    return parser.parse_args()


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", keep_default_na=False)


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)


def row_keep_mask(df: pd.DataFrame, include_low: bool, include_bracket_context: bool) -> pd.Series:
    allowed_quality = {"matched", "weak_match"}
    if include_low:
        allowed_quality.add("low_match")
    mask = (
        df["clean_all"].astype(str).eq("True")
        & df["ipa_conversion_status"].astype(str).eq("ok")
        & df["has_image"].astype(str).eq("True")
        & df["image_exists"].astype(str).eq("True")
        & df["quality"].astype(str).isin(allowed_quality)
    )
    if not include_bracket_context:
        mask &= ~df["has_bracket_context"].astype(str).eq("True")
    return mask


def summarize(df: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": int(len(df)),
        "splits": df["source_split"].value_counts().to_dict(),
        "qualities": df["quality"].value_counts().to_dict(),
        "unique_labels": int(df["label"].nunique()),
        "bracket_context": int(df["has_bracket_context"].astype(str).eq("True").sum()),
    }


def main() -> None:
    args = parse_args()
    src = args.clean_root / "ocr_selected_all" / "eval_manifest.tsv"
    df = read_tsv(src)
    out = df[row_keep_mask(df, args.include_low, args.include_bracket_context)].copy()
    out_dir = args.clean_root / "ocr_selected_phonetic_reliable"
    write_tsv(out, out_dir / "eval_manifest.tsv")
    summary = summarize(out)
    lines = [
        "# phonetic reliable OCR manifest",
        "",
        "Criterion: clean_all + IPA conversion ok + image exists + quality in matched/weak_match.",
        "Chinese candidate_headword mismatch is ignored because OCR target is the phonetic crop.",
        "low_match and bracket-context rows are excluded by default.",
        "",
        f"rows\t{summary['rows']}",
        f"splits\t{summary['splits']}",
        f"qualities\t{summary['qualities']}",
        f"unique_labels\t{summary['unique_labels']}",
        f"bracket_context\t{summary['bracket_context']}",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((out_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
