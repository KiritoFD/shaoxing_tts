"""Compare VLM page OCR rows with a model-produced post-136 CSV.

The comparison key is source page + headword:

* VLM: source_page + headword
* Model CSV: 页码 + 汉字

This measures agreement between two OCR outputs. It is not ground truth
accuracy unless one side is manually verified.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "ipa_ocr_work" / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "ipa_ocr_work" / "scripts"))

from wupin_ipa_convert import DEFAULT_MAP, ipa_to_wupin, load_mapping, normalize  # noqa: E402


DEFAULT_VLM = PROJECT_ROOT / "ipa_ocr_work" / "eval" / "vlm_pages" / "qwen36plus_post136_current_best.rows.csv"
DEFAULT_MODEL = PROJECT_ROOT / "result_all_converted.post136_trocr_v2_epoch001_full_20260525.csv"
DEFAULT_OUT_PREFIX = PROJECT_ROOT / "ipa_ocr_work" / "eval" / "vlm_pages" / "qwen36plus_vs_trocr_v2_epoch001"


TONE_TRANSLATION = str.maketrans(
    {
        "¹": "1",
        "²": "2",
        "³": "3",
        "⁴": "4",
        "⁵": "5",
        "⁶": "6",
        "⁷": "7",
        "⁸": "8",
        "⁹": "9",
        "⁰": "0",
        "₁": "1",
        "₂": "2",
        "₃": "3",
        "₄": "4",
        "₅": "5",
        "₆": "6",
        "₇": "7",
        "₈": "8",
        "₉": "9",
        "₀": "0",
    }
)

IPA_VISUAL_EQUIV = str.maketrans(
    {
        "ɔ": "ɒ",
        "ꞏ": "",
        "·": "",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare VLM OCR rows with model CSV rows.")
    parser.add_argument("--vlm-csv", type=Path, default=DEFAULT_VLM)
    parser.add_argument("--model-csv", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    return parser.parse_args()


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFC", str(value)).strip()
    return re.sub(r"\s+", "", text)


def clean_hanzi(value: object) -> str:
    text = clean_text(value)
    return text.replace("【", "").replace("】", "")


def tone_digits(text: object) -> str:
    return clean_text(text).translate(TONE_TRANSLATION)


def normalize_ipa(text: object, visual_equiv: bool = False) -> str:
    text = tone_digits(text)
    text = text.replace("?", "ʔ")
    text = text.replace("ɡ", "g")
    if visual_equiv:
        text = text.translate(IPA_VISUAL_EQUIV)
    return unicodedata.normalize("NFC", text)


def normalize_wupin(text: object) -> str:
    text = normalize(text)
    text = text.replace("?", "q")
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def vlm_ipa_from_syllables(row: pd.Series) -> tuple[str, str]:
    raw = str(row.get("syllables_json", "") or "").strip()
    if raw:
        try:
            syllables = json.loads(raw)
            if isinstance(syllables, list):
                finals: list[str] = []
                for syllable in syllables:
                    if not isinstance(syllable, dict):
                        continue
                    final = syllable.get("final")
                    if final:
                        finals.append(str(final))
                if finals:
                    return "".join(finals), "syllables_final"
        except json.JSONDecodeError:
            pass
    ipa = str(row.get("ipa", "") or "")
    return ipa, "ipa_field"


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(
                min(
                    prev[j] + 1,
                    cur[j - 1] + 1,
                    prev[j - 1] + (0 if ca == cb else 1),
                )
            )
        prev = cur
    return prev[-1]


def similarity(a: str, b: str) -> float:
    denom = max(len(a), len(b))
    if denom == 0:
        return 1.0
    return 1.0 - edit_distance(a, b) / denom


def safe_int(value: object) -> int | None:
    try:
        if pd.isna(value):
            return None
        text = str(value).strip()
        if not text:
            return None
        return int(float(text))
    except Exception:
        return None


def build_model_index(model_df: pd.DataFrame) -> dict[tuple[int, str], list[int]]:
    index: dict[tuple[int, str], list[int]] = defaultdict(list)
    for idx, row in model_df.iterrows():
        page = safe_int(row.get("页码"))
        hanzi = clean_hanzi(row.get("汉字"))
        if page is None or not hanzi:
            continue
        index[(page, hanzi)].append(int(idx))
    return index


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matched = [row for row in rows if row["matched"]]
    conv_ok = [row for row in matched if row["vlm_wupin_from_ipa_status"] == "ok"]
    conv_ok_visual = [row for row in matched if row["vlm_wupin_from_ipa_visual_status"] == "ok"]
    return {
        "vlm_rows": len(rows),
        "matched_rows": len(matched),
        "matched_ratio_of_vlm": len(matched) / len(rows) if rows else 0.0,
        "unmatched_vlm_rows": len(rows) - len(matched),
        "ipa_exact_strict_on_matched": sum(1 for row in matched if row["ipa_exact_strict"]) / len(matched) if matched else 0.0,
        "ipa_similarity_strict_on_matched": mean([row["ipa_similarity_strict"] for row in matched]),
        "ipa_exact_visual_on_matched": sum(1 for row in matched if row["ipa_exact_visual"]) / len(matched) if matched else 0.0,
        "ipa_similarity_visual_on_matched": mean([row["ipa_similarity_visual"] for row in matched]),
        "vlm_ipa_to_wupin_success_strict_on_matched": len(conv_ok) / len(matched) if matched else 0.0,
        "vlm_ipa_to_wupin_success_visual_on_matched": len(conv_ok_visual) / len(matched) if matched else 0.0,
        "wupin_exact_strict_on_convertible": sum(1 for row in conv_ok if row["wupin_exact_strict"]) / len(conv_ok) if conv_ok else 0.0,
        "wupin_similarity_strict_on_convertible": mean([row["wupin_similarity_strict"] for row in conv_ok]),
        "wupin_exact_visual_on_convertible": sum(1 for row in conv_ok_visual if row["wupin_exact_visual"]) / len(conv_ok_visual) if conv_ok_visual else 0.0,
        "wupin_similarity_visual_on_convertible": mean([row["wupin_similarity_visual"] for row in conv_ok_visual]),
    }


def main() -> None:
    args = parse_args()
    mapping = load_mapping(args.mapping)
    vlm_df = pd.read_csv(args.vlm_csv, keep_default_na=False)
    model_df = pd.read_csv(args.model_csv, keep_default_na=False)
    model_index = build_model_index(model_df)
    used_model_rows: set[int] = set()

    out_rows: list[dict[str, Any]] = []
    conversion_issues = Counter()
    conversion_visual_issues = Counter()

    for _, vlm_row in vlm_df.iterrows():
        source_page = safe_int(vlm_row.get("source_page"))
        headword = clean_hanzi(vlm_row.get("headword"))
        candidates = list(model_index.get((source_page, headword), [])) if source_page is not None else []
        model_idx = next((idx for idx in candidates if idx not in used_model_rows), None)
        if model_idx is None and candidates:
            model_idx = candidates[0]
        matched = model_idx is not None
        if matched:
            used_model_rows.add(int(model_idx))
            model_row = model_df.loc[model_idx]
        else:
            model_row = pd.Series(dtype=object)

        vlm_ipa_raw, vlm_ipa_source = vlm_ipa_from_syllables(vlm_row)
        vlm_ipa = normalize_ipa(vlm_ipa_raw, visual_equiv=False)
        vlm_ipa_visual = normalize_ipa(vlm_ipa_raw, visual_equiv=True)
        model_ipa = normalize_ipa(model_row.get("IPA识别", ""), visual_equiv=False)
        model_ipa_visual = normalize_ipa(model_row.get("IPA识别", ""), visual_equiv=True)
        model_wupin = normalize_wupin(model_row.get("拼音", ""))

        vlm_wupin, wupin_errors = ipa_to_wupin(vlm_ipa, mapping)
        vlm_wupin_visual, wupin_visual_errors = ipa_to_wupin(vlm_ipa_visual, mapping)
        vlm_wupin = normalize_wupin(vlm_wupin)
        vlm_wupin_visual = normalize_wupin(vlm_wupin_visual)
        for error in wupin_errors:
            conversion_issues[error] += 1
        for error in wupin_visual_errors:
            conversion_visual_issues[error] += 1

        ipa_sim_strict = similarity(vlm_ipa, model_ipa) if matched else 0.0
        ipa_sim_visual = similarity(vlm_ipa_visual, model_ipa_visual) if matched else 0.0
        wupin_sim_strict = similarity(vlm_wupin, model_wupin) if matched and not wupin_errors else 0.0
        wupin_sim_visual = similarity(vlm_wupin_visual, model_wupin) if matched and not wupin_visual_errors else 0.0

        out_rows.append(
            {
                "source_page": source_page if source_page is not None else "",
                "pdf_page": safe_int(vlm_row.get("pdf_page")) or "",
                "headword": headword,
                "matched": bool(matched),
                "model_row_index": int(model_idx) if matched else "",
                "vlm_ipa_source": vlm_ipa_source,
                "vlm_ipa": vlm_ipa,
                "vlm_ipa_visual": vlm_ipa_visual,
                "model_ipa": model_ipa,
                "ipa_exact_strict": bool(matched and vlm_ipa == model_ipa),
                "ipa_similarity_strict": ipa_sim_strict,
                "ipa_exact_visual": bool(matched and vlm_ipa_visual == model_ipa_visual),
                "ipa_similarity_visual": ipa_sim_visual,
                "vlm_wupin_from_ipa": vlm_wupin,
                "vlm_wupin_from_ipa_status": "ok" if not wupin_errors else ";".join(wupin_errors),
                "vlm_wupin_from_ipa_visual": vlm_wupin_visual,
                "vlm_wupin_from_ipa_visual_status": "ok" if not wupin_visual_errors else ";".join(wupin_visual_errors),
                "model_wupin": model_wupin,
                "wupin_exact_strict": bool(matched and not wupin_errors and vlm_wupin == model_wupin),
                "wupin_similarity_strict": wupin_sim_strict,
                "wupin_exact_visual": bool(matched and not wupin_visual_errors and vlm_wupin_visual == model_wupin),
                "wupin_similarity_visual": wupin_sim_visual,
                "model_hanzi": clean_hanzi(model_row.get("汉字", "")),
                "model_page": safe_int(model_row.get("页码")) if matched else "",
            }
        )

    matched_model_keys = {(row["model_page"], row["model_hanzi"]) for row in out_rows if row["matched"]}
    model_in_vlm_pages = model_df[model_df["页码"].map(safe_int).isin({safe_int(v) for v in vlm_df["source_page"]})].copy()

    summary = summarize(out_rows)
    summary.update(
        {
            "vlm_csv": str(args.vlm_csv),
            "model_csv": str(args.model_csv),
            "model_rows_total": int(len(model_df)),
            "model_rows_on_vlm_source_pages": int(len(model_in_vlm_pages)),
            "unique_vlm_source_pages": int(vlm_df["source_page"].map(safe_int).nunique()),
            "unique_vlm_headword_page_keys": int(len({(row["source_page"], row["headword"]) for row in out_rows})),
            "strict_conversion_issue_top": conversion_issues.most_common(20),
            "visual_conversion_issue_top": conversion_visual_issues.most_common(20),
            "model_unique_keys_matched": int(len(matched_model_keys)),
        }
    )

    out_df = pd.DataFrame(out_rows)
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_prefix.with_suffix(".matched_rows.csv"), index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    out_df.to_csv(args.out_prefix.with_suffix(".matched_rows.tsv"), index=False, sep="\t", encoding="utf-8", quoting=csv.QUOTE_MINIMAL)
    args.out_prefix.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    mismatch_df = out_df[out_df["matched"] & (~out_df["ipa_exact_visual"] | ~out_df["wupin_exact_visual"])].copy()
    if not mismatch_df.empty:
        mismatch_df.sort_values(["ipa_similarity_visual", "wupin_similarity_visual", "source_page", "headword"], inplace=True)
        mismatch_df.head(200).to_csv(
            args.out_prefix.with_suffix(".top_mismatches.csv"),
            index=False,
            encoding="utf-8-sig",
            quoting=csv.QUOTE_MINIMAL,
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {args.out_prefix.with_suffix('.matched_rows.csv')}")
    print(f"wrote {args.out_prefix.with_suffix('.summary.json')}")


if __name__ == "__main__":
    main()
