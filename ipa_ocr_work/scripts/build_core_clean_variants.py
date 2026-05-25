"""Build core clean OCR manifests by excluding inventory-review rows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLEAN_ROOT = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_pdf136_clean"
DEFAULT_REVIEW = PROJECT_ROOT / "ipa_ocr_work" / "reports" / "wupin_rule_audit_pdf136.inventory_review.tsv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build core clean dataset variants.")
    parser.add_argument("--clean-root", type=Path, default=DEFAULT_CLEAN_ROOT)
    parser.add_argument("--inventory-review", type=Path, default=DEFAULT_REVIEW)
    return parser.parse_args()


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", keep_default_na=False)


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)


def key_df(df: pd.DataFrame) -> pd.Series:
    return df["page"].astype(str) + "\t" + df["row_index"].astype(str)


def summarize(df: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": int(len(df)),
        "split_counts": df["source_split"].value_counts().to_dict() if "source_split" in df.columns else {},
        "unique_labels": int(df["label"].nunique()) if "label" in df.columns else 0,
    }


def main() -> None:
    args = parse_args()
    review = read_tsv(args.inventory_review)
    review_keys = set(key_df(review))

    outputs = {}
    for name in ["ocr_selected_all", "ocr_selected_strict"]:
        src = args.clean_root / name / "eval_manifest.tsv"
        if not src.exists():
            continue
        df = read_tsv(src)
        keep = ~key_df(df).isin(review_keys)
        out_dir = args.clean_root / name.replace("selected", "selected_core")
        out = df[keep].copy()
        write_tsv(out, out_dir / "eval_manifest.tsv")
        outputs[out_dir.name] = summarize(out)

    for name in ["syllable_ocr_all", "syllable_ocr_strict"]:
        src = args.clean_root / name / "eval_manifest.tsv"
        if not src.exists():
            continue
        df = read_tsv(src)
        keep = ~key_df(df).isin(review_keys)
        out_dir = args.clean_root / name.replace("ocr", "ocr_core")
        out = df[keep].copy()
        write_tsv(out, out_dir / "eval_manifest.tsv")
        outputs[out_dir.name] = summarize(out)

    lines = ["# core clean variants", "", "variant\trows\tunique_labels\tsplits"]
    for variant, summary in outputs.items():
        lines.append(f"{variant}\t{summary['rows']}\t{summary['unique_labels']}\t{summary['split_counts']}")
    (args.clean_root / "core_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((args.clean_root / "core_summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
