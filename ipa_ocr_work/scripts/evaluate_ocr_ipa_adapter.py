"""Evaluate OCR-IPA predictions with IPA labels and a Wu-pinyin adapter.

The OCR-IPA model emits IPA-like text. The project target label is Wu-pinyin.
This script reports:
- direct OCR-IPA vs the legacy/manual IPA识别 column
- adapted Wu-pinyin by nearest-neighbor matching in IPA space
"""

from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL = PROJECT_ROOT / "ipa_ocr_work" / "eval" / "enhancement_ab" / "eval_manifest.tsv"
DEFAULT_EXCEL = PROJECT_ROOT / "result_all_converted.xlsx"
DEFAULT_PREDS = PROJECT_ROOT / "ipa_ocr_work" / "eval" / "enhancement_ab" / "predictions_ocr_ipa_calamari.tsv"
DEFAULT_OUT = PROJECT_ROOT / "ipa_ocr_work" / "eval" / "enhancement_ab" / "ocr_ipa_adapter"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OCR-IPA with adapter.")
    parser.add_argument("--eval-manifest", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--excel", type=Path, default=DEFAULT_EXCEL)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDS)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--adapter-source",
        choices=("train", "all"),
        default="train",
        help="Which eval rows may be used as IPA->Wu-pinyin adapter exemplars.",
    )
    return parser.parse_args()


def normalize(text: object) -> str:
    if pd.isna(text):
        return ""
    text = unicodedata.normalize("NFC", str(text).strip())
    return re.sub(r"\s+", "", text)


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def digit_only(text: str) -> str:
    return "".join(ch for ch in text if ch.isdigit())


def letter_only(text: str) -> str:
    return "".join(ch for ch in text if not ch.isdigit())


def add_excel_labels(eval_df: pd.DataFrame, excel_path: Path) -> pd.DataFrame:
    df = pd.read_excel(excel_path)
    df = df[df["页码"].notna()].copy()
    df["页码"] = df["页码"].astype(int)
    df["wupin_excel"] = df["拼音"].map(normalize)
    df["ipa_label"] = df["IPA识别"].map(normalize)
    df = df[df["wupin_excel"] != ""]
    df["row_index"] = df.groupby("页码").cumcount()
    labels = df[["页码", "row_index", "wupin_excel", "ipa_label"]].rename(columns={"页码": "page"})
    merged = eval_df.merge(labels, on=["page", "row_index"], how="left")
    merged["ipa_label"] = merged["ipa_label"].fillna("")
    merged["wupin_excel"] = merged["wupin_excel"].fillna(merged["label"])
    return merged


def score_rows(df: pd.DataFrame, label_col: str, pred_col: str) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        label = normalize(row[label_col])
        pred = normalize(row[pred_col])
        rows.append(
            {
                "sample_id": row["sample_id"],
                "variant": row["variant"],
                "label": label,
                "prediction": pred,
                "exact": int(label == pred),
                "edit_distance": edit_distance(label, pred),
                "label_len": len(label),
                "digit_edit_distance": edit_distance(digit_only(label), digit_only(pred)),
                "digit_label_len": len(digit_only(label)),
                "letter_edit_distance": edit_distance(letter_only(label), letter_only(pred)),
                "letter_label_len": len(letter_only(label)),
            }
        )
    return pd.DataFrame(rows)


def summarize(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in scored.groupby("variant", sort=True):
        rows.append(
            {
                "variant": variant,
                "n": len(group),
                "exact_rate": group["exact"].mean(),
                "cer": group["edit_distance"].sum() / max(1, group["label_len"].sum()),
                "digit_cer": group["digit_edit_distance"].sum() / max(1, group["digit_label_len"].sum()),
                "letter_cer": group["letter_edit_distance"].sum() / max(1, group["letter_label_len"].sum()),
                "avg_edit_distance": group["edit_distance"].mean(),
            }
        )
    return pd.DataFrame(rows)


def nearest_adapter(pred: str, exemplars: list[tuple[str, str]]) -> tuple[str, str, int]:
    if not exemplars:
        return "", "", 0
    best_ipa, best_wupin = exemplars[0]
    best_dist = edit_distance(pred, best_ipa)
    for ipa, wupin in exemplars[1:]:
        dist = edit_distance(pred, ipa)
        if dist < best_dist:
            best_ipa, best_wupin, best_dist = ipa, wupin, dist
    return best_wupin, best_ipa, best_dist


def main() -> None:
    args = parse_args()
    eval_df = pd.read_csv(args.eval_manifest, sep="\t", keep_default_na=False)
    pred_df = pd.read_csv(args.predictions, sep="\t", keep_default_na=False)
    merged = eval_df.merge(pred_df, on=["sample_id", "variant"], how="inner")
    merged = add_excel_labels(merged, args.excel)
    merged["prediction"] = merged["prediction"].map(normalize)

    with_ipa = merged[merged["ipa_label"] != ""].copy()
    ipa_scored = score_rows(with_ipa, "ipa_label", "prediction")
    ipa_summary = summarize(ipa_scored)

    if args.adapter_source == "train":
        adapter_rows = with_ipa[with_ipa["source_split"] == "train"]
    else:
        adapter_rows = with_ipa
    # One exemplar per (IPA, Wu-pinyin) pair.
    exemplars = sorted(
        set(zip(adapter_rows["ipa_label"].map(normalize), adapter_rows["wupin_excel"].map(normalize)))
    )

    adapted = []
    for _, row in with_ipa.iterrows():
        wupin_pred, matched_ipa, dist = nearest_adapter(row["prediction"], exemplars)
        out = row.to_dict()
        out["adapted_wupin"] = wupin_pred
        out["matched_ipa"] = matched_ipa
        out["adapter_ipa_distance"] = dist
        adapted.append(out)
    adapted_df = pd.DataFrame(adapted)
    wupin_scored = score_rows(adapted_df, "wupin_excel", "adapted_wupin")
    wupin_summary = summarize(wupin_scored)

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    ipa_scored.to_csv(args.out_prefix.with_suffix(".ipa_scored.tsv"), sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)
    ipa_summary.to_csv(args.out_prefix.with_suffix(".ipa_summary.tsv"), sep="\t", index=False)
    adapted_df[
        [
            "sample_id",
            "variant",
            "prediction",
            "ipa_label",
            "wupin_excel",
            "adapted_wupin",
            "matched_ipa",
            "adapter_ipa_distance",
        ]
    ].to_csv(args.out_prefix.with_suffix(".adapter_predictions.tsv"), sep="\t", index=False)
    wupin_scored.to_csv(args.out_prefix.with_suffix(".wupin_scored.tsv"), sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)
    wupin_summary.to_csv(args.out_prefix.with_suffix(".wupin_summary.tsv"), sep="\t", index=False)

    print("IPA-space OCR score:")
    print(ipa_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print(f"Adapted Wu-pinyin score ({args.adapter_source} adapter):")
    print(wupin_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nwrote {args.out_prefix}.*.tsv")


if __name__ == "__main__":
    main()
