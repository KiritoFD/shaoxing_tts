"""Score OCR predictions against a Wu-pinyin enhancement eval manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL = PROJECT_ROOT / "ipa_ocr_work" / "eval" / "enhancement_ab" / "eval_manifest.tsv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score OCR predictions.")
    parser.add_argument("--eval-manifest", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--source-splits", nargs="+", default=None)
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="Score missing predictions as blank strings. Default scores only provided predictions.",
    )
    return parser.parse_args()


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(
                min(
                    prev[j] + 1,
                    cur[j - 1] + 1,
                    prev[j - 1] + (ca != cb),
                )
            )
        prev = cur
    return prev[-1]


def digit_only(text: str) -> str:
    superscript = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
    text = text.translate(superscript)
    return "".join(ch for ch in text if ch.isdigit())


def letter_only(text: str) -> str:
    superscript_digits = set("⁰¹²³⁴⁵⁶⁷⁸⁹")
    return "".join(ch for ch in text if not ch.isdigit() and ch not in superscript_digits)


def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", keep_default_na=False)
    required = {"sample_id", "variant", "prediction"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"predictions missing columns: {sorted(missing)}")
    return df


def metric_rows(merged: pd.DataFrame) -> list[dict]:
    rows = []
    for _, row in merged.iterrows():
        label = str(row["label"])
        pred = str(row["prediction"])
        label_digits = digit_only(label)
        pred_digits = digit_only(pred)
        label_letters = letter_only(label)
        pred_letters = letter_only(pred)
        rows.append(
            {
                "sample_id": row["sample_id"],
                "variant": row["variant"],
                "label": label,
                "prediction": pred,
                "exact": int(label == pred),
                "edit_distance": edit_distance(label, pred),
                "label_len": len(label),
                "digit_edit_distance": edit_distance(label_digits, pred_digits),
                "digit_label_len": len(label_digits),
                "letter_edit_distance": edit_distance(label_letters, pred_letters),
                "letter_label_len": len(label_letters),
            }
        )
    return rows


def summarize(scored: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for variant, group in scored.groupby("variant", sort=True):
        total_edits = group["edit_distance"].sum()
        total_len = group["label_len"].sum()
        digit_edits = group["digit_edit_distance"].sum()
        digit_len = group["digit_label_len"].sum()
        letter_edits = group["letter_edit_distance"].sum()
        letter_len = group["letter_label_len"].sum()
        summaries.append(
            {
                "variant": variant,
                "n": len(group),
                "exact_rate": group["exact"].mean(),
                "cer": total_edits / total_len if total_len else 0.0,
                "digit_cer": digit_edits / digit_len if digit_len else 0.0,
                "letter_cer": letter_edits / letter_len if letter_len else 0.0,
                "avg_edit_distance": group["edit_distance"].mean(),
            }
        )
    return pd.DataFrame(summaries)


def main() -> None:
    args = parse_args()
    eval_df = pd.read_csv(args.eval_manifest, sep="\t", keep_default_na=False)
    if args.source_splits:
        eval_df = eval_df[eval_df["source_split"].isin(args.source_splits)].copy()
    pred_df = load_predictions(args.predictions)
    merged = eval_df.merge(
        pred_df,
        on=["sample_id", "variant"],
        how="left" if args.include_missing else "inner",
    )
    merged["prediction"] = merged["prediction"].fillna("")

    scored = pd.DataFrame(metric_rows(merged))
    summary = summarize(scored)

    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        scored_path = args.out.with_suffix(".scored.tsv")
        summary_path = args.out.with_suffix(".summary.tsv")
        scored.to_csv(scored_path, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)
        summary.to_csv(summary_path, sep="\t", index=False)
        print(f"wrote {scored_path}")
        print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
