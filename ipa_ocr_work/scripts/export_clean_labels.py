"""Export clean CSV labels from result_all_converted.xlsx.

The workbook is treated as messy input. The trusted source column is 拼音.
The legacy IPA识别 column is copied for reference only and should not be used
as OCR ground truth.
"""

from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_XLSX = PROJECT_ROOT / "result_all_converted.xlsx"
DEFAULT_OUT = PROJECT_ROOT / "result_all_converted.clean.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export normalized CSV labels.")
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFC", str(value).strip())
    return re.sub(r"\s+", "", text)


def clean_note(value: object) -> str:
    if pd.isna(value):
        return ""
    return unicodedata.normalize("NFC", str(value).strip())


def main() -> None:
    args = parse_args()
    df = pd.read_excel(args.xlsx, keep_default_na=False)
    required = ["页码", "汉字", "拼音"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    out = pd.DataFrame()
    out["row_id"] = [f"r{i:05d}" for i in range(len(df))]
    out["page"] = pd.to_numeric(df["页码"], errors="coerce").astype("Int64")
    out["hanzi"] = df["汉字"].map(clean_text)
    out["wupin"] = df["拼音"].map(clean_text).str.lower()
    out["alt_wupin"] = df.get("其他读法", pd.Series([""] * len(df))).map(clean_note).str.lower()
    out["note"] = df.get("后续汉字", pd.Series([""] * len(df))).map(clean_note)
    out["legacy_ipa_ocr"] = df.get("IPA识别", pd.Series([""] * len(df))).map(clean_text)
    out["has_wupin"] = out["wupin"].ne("")
    out["source"] = "result_all_converted.xlsx"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    print(f"rows: {len(out)}")
    print(f"usable wupin rows: {int(out['has_wupin'].sum())}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
