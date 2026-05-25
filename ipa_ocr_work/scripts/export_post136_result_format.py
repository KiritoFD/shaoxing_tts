"""Convert post-136 OCR predictions to result_all_converted.csv format."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IN = PROJECT_ROOT / "ipa_ocr_work" / "eval" / "post136_trocr_best" / "post136_trocr_best_predictions.csv"
DEFAULT_OUT = PROJECT_ROOT / "result_all_converted.post136_trocr_best.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export post-136 OCR output in result_all_converted.csv columns.")
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--sample-out",
        type=Path,
        default=PROJECT_ROOT
        / "ipa_ocr_work"
        / "eval"
        / "post136_trocr_best"
        / "post136_trocr_best_result_format_sample.csv",
    )
    return parser.parse_args()


def clean_wupin(text: object) -> str:
    # ɿ is the apical vowel; this Wu-pinyin inventory uses y for that final.
    return str(text).replace("<ɿ>", "y")


def review_marker(status: object) -> str:
    status_text = str(status)
    if status_text == "ok":
        return ""
    parts = set(part for part in status_text.split(";") if part)
    if parts and parts <= {"unknown_ipa:ɿ"}:
        return ""
    return "OCR_REVIEW:" + status_text


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input, keep_default_na=False)
    out = pd.DataFrame(
        {
            "页码": df["source_page"].astype(int),
            "汉字": df["candidate_headword"],
            "拼音": df["pred_wupin"].map(clean_wupin),
            "Unnamed: 3": "",
            "其他读法": "",
            "标记": df["pred_wupin_status"].map(review_marker),
            "后续汉字": df["pdf_text"],
            "IPA识别": df["pred_ipa"],
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    sample = out.groupby("页码", group_keys=False).head(2).head(80)
    args.sample_out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.sample_out, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    print(f"rows: {len(out)}")
    print(f"pages: {int(out['页码'].min())}-{int(out['页码'].max())}")
    print(f"review rows: {int((out['标记'] != '').sum())}")
    print(f"wrote {args.out}")
    print(f"wrote {args.sample_out}")


if __name__ == "__main__":
    main()
