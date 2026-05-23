"""Adapt OCR predictions to the Excel Wu-pinyin format.

Supported modes:
- direct: prediction is already text-like Wu-pinyin; normalize only.
- ipa-nearest: prediction is IPA-like; map to Wu-pinyin by nearest IPA label
  from Excel rows in the chosen adapter split.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL = PROJECT_ROOT / "ipa_ocr_work" / "eval" / "enhancement_ab" / "eval_manifest.tsv"
DEFAULT_EXCEL = PROJECT_ROOT / "result_all_converted.xlsx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adapt OCR predictions to Wu-pinyin.")
    parser.add_argument("--eval-manifest", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--excel", type=Path, default=DEFAULT_EXCEL)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--mode", choices=("direct", "ipa-nearest"), required=True)
    parser.add_argument("--adapter-split", default="train")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def normalize(text: object) -> str:
    if pd.isna(text):
        return ""
    text = unicodedata.normalize("NFC", str(text).strip())
    return re.sub(r"\s+", "", text)


def wupin_normalize(text: object) -> str:
    text = normalize(text).lower()
    return re.sub(r"[^a-z0-9]", "", text)


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def add_excel_labels(eval_df: pd.DataFrame, excel_path: Path) -> pd.DataFrame:
    df = pd.read_excel(excel_path)
    df = df[df["页码"].notna()].copy()
    df["页码"] = df["页码"].astype(int)
    df["wupin_excel"] = df["拼音"].map(wupin_normalize)
    df["ipa_label"] = df["IPA识别"].map(normalize)
    df = df[df["wupin_excel"] != ""]
    df["row_index"] = df.groupby("页码").cumcount()
    labels = df[["页码", "row_index", "wupin_excel", "ipa_label"]].rename(columns={"页码": "page"})
    merged = eval_df.merge(labels, on=["page", "row_index"], how="left")
    merged["wupin_excel"] = merged["wupin_excel"].fillna(merged["label"].map(wupin_normalize))
    merged["ipa_label"] = merged["ipa_label"].fillna("")
    return merged


def nearest_ipa_to_wupin(pred: str, exemplars: list[tuple[str, str]]) -> str:
    if not exemplars:
        return ""
    best_ipa, best_wupin = exemplars[0]
    best_dist = edit_distance(pred, best_ipa)
    for ipa, wupin in exemplars[1:]:
        dist = edit_distance(pred, ipa)
        if dist < best_dist:
            best_ipa, best_wupin, best_dist = ipa, wupin, dist
    return best_wupin


def main() -> None:
    args = parse_args()
    eval_df = pd.read_csv(args.eval_manifest, sep="\t", keep_default_na=False)
    eval_df = add_excel_labels(eval_df, args.excel)
    pred_df = pd.read_csv(args.predictions, sep="\t", keep_default_na=False)
    merged = eval_df.merge(pred_df, on=["sample_id", "variant"], how="inner")
    merged["raw_prediction"] = merged["prediction"].map(normalize)

    if args.mode == "direct":
        merged["prediction"] = merged["raw_prediction"].map(wupin_normalize)
    else:
        adapter_df = eval_df[
            (eval_df["source_split"] == args.adapter_split)
            & (eval_df["ipa_label"] != "")
            & (eval_df["wupin_excel"] != "")
        ].copy()
        exemplars = sorted(set(zip(adapter_df["ipa_label"].map(normalize), adapter_df["wupin_excel"].map(wupin_normalize))))
        merged["prediction"] = merged["raw_prediction"].map(lambda pred: nearest_ipa_to_wupin(pred, exemplars))

    out_df = merged[["sample_id", "variant", "prediction", "raw_prediction", "source_split"]].copy()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, sep="\t", index=False)
    print(f"wrote {args.out}")
    print(f"rows: {len(out_df)}")


if __name__ == "__main__":
    main()
