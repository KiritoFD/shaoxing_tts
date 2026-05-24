"""Merge post-136 VLM OCR rows with model OCR CSV for side-by-side review."""

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
SCRIPT_DIR = PROJECT_ROOT / "ipa_ocr_work" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from wupin_ipa_convert import DEFAULT_MAP, ipa_to_wupin, load_mapping, normalize  # noqa: E402


DEFAULT_VLM = PROJECT_ROOT / "ipa_ocr_work" / "eval" / "vlm_pages" / "qwen36plus_post136_current_best.rows.csv"
DEFAULT_OCR = PROJECT_ROOT / "result_all_converted.post136_trocr_v2_epoch001_full_20260525.csv"
DEFAULT_OUT = PROJECT_ROOT / "ipa_ocr_work" / "eval" / "vlm_pages" / "qwen36plus_post136_vlm_vs_trocr_merged.csv"


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

IPA_VISUAL_EQUIV = str.maketrans({"ɔ": "ɒ", "ꞏ": "", "·": ""})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge VLM and OCR CSV results.")
    parser.add_argument("--vlm-csv", type=Path, default=DEFAULT_VLM)
    parser.add_argument("--ocr-csv", type=Path, default=DEFAULT_OCR)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFC", str(value)).strip()
    return re.sub(r"\s+", "", text)


def clean_hanzi(value: object) -> str:
    return clean_text(value).replace("【", "").replace("】", "")


def tone_digits(text: object) -> str:
    return clean_text(text).translate(TONE_TRANSLATION)


def normalize_ipa(text: object, visual_equiv: bool = True) -> str:
    text = tone_digits(text)
    text = text.replace("?", "ʔ").replace("ɡ", "g")
    if visual_equiv:
        text = text.translate(IPA_VISUAL_EQUIV)
    return unicodedata.normalize("NFC", text)


def normalize_wupin(text: object) -> str:
    text = normalize(text)
    text = text.replace("?", "q")
    return re.sub(r"[^a-z0-9]+", "", text)


def letters_only(text: str) -> str:
    return re.sub(r"\d+", "", text)


def digits_only(text: str) -> str:
    return "".join(re.findall(r"\d+", text))


def compare_parts(left: str, right: str) -> dict[str, bool]:
    exact = bool(left or right) and left == right
    letters_equal = bool(left or right) and letters_only(left) == letters_only(right)
    digits_equal = bool(left or right) and digits_only(left) == digits_only(right)
    return {
        "exact": exact,
        "letters_equal": letters_equal,
        "digits_equal": digits_equal,
        "only_digits_differ": letters_equal and not exact,
    }


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
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if ca == cb else 1)))
        prev = cur
    return prev[-1]


def similarity(a: str, b: str) -> float:
    denom = max(len(a), len(b))
    return 1.0 if denom == 0 else 1.0 - edit_distance(a, b) / denom


def safe_int(value: object) -> int | None:
    try:
        if pd.isna(value):
            return None
        text = str(value).strip()
        return int(float(text)) if text else None
    except Exception:
        return None


def vlm_ipa_from_syllables(row: pd.Series) -> tuple[str, str]:
    raw = str(row.get("syllables_json", "") or "").strip()
    if raw:
        try:
            syllables = json.loads(raw)
            if isinstance(syllables, list):
                finals = [str(item.get("final", "")) for item in syllables if isinstance(item, dict) and item.get("final")]
                if finals:
                    return "".join(finals), "syllables_json.final"
        except json.JSONDecodeError:
            pass
    return str(row.get("ipa", "") or ""), "ipa"


def build_ocr_index(ocr_df: pd.DataFrame) -> dict[tuple[int, str], list[int]]:
    index: dict[tuple[int, str], list[int]] = defaultdict(list)
    for idx, row in ocr_df.iterrows():
        page = safe_int(row.get("页码"))
        hanzi = clean_hanzi(row.get("汉字"))
        if page is not None and hanzi:
            index[(page, hanzi)].append(int(idx))
    return index


def bool_zh(value: bool) -> str:
    return "是" if value else "否"


def main() -> None:
    args = parse_args()
    mapping = load_mapping(args.mapping)
    vlm_df = pd.read_csv(args.vlm_csv, keep_default_na=False)
    ocr_df = pd.read_csv(args.ocr_csv, keep_default_na=False)
    ocr_index = build_ocr_index(ocr_df)
    used_ocr: set[int] = set()

    records: list[dict[str, Any]] = []

    def build_record(vlm_row: pd.Series | None, ocr_row: pd.Series | None, ocr_idx: int | None, status: str) -> dict[str, Any]:
        if vlm_row is not None:
            source_page = safe_int(vlm_row.get("source_page"))
            pdf_page = safe_int(vlm_row.get("pdf_page"))
            word = clean_hanzi(vlm_row.get("headword"))
            vlm_ipa_raw, vlm_ipa_source = vlm_ipa_from_syllables(vlm_row)
            vlm_ipa = normalize_ipa(vlm_ipa_raw)
            vlm_confidence = vlm_row.get("confidence", "")
            vlm_notes = vlm_row.get("notes", "")
        else:
            source_page = safe_int(ocr_row.get("页码")) if ocr_row is not None else None
            pdf_page = ""
            word = clean_hanzi(ocr_row.get("汉字")) if ocr_row is not None else ""
            vlm_ipa_raw = ""
            vlm_ipa_source = ""
            vlm_ipa = ""
            vlm_confidence = ""
            vlm_notes = ""

        if ocr_row is not None:
            ocr_page = safe_int(ocr_row.get("页码"))
            ocr_word = clean_hanzi(ocr_row.get("汉字"))
            ocr_wupin = normalize_wupin(ocr_row.get("拼音", ""))
            ocr_ipa = normalize_ipa(ocr_row.get("IPA识别", ""))
            ocr_following = clean_text(ocr_row.get("后续汉字", ""))
        else:
            ocr_page = ""
            ocr_word = ""
            ocr_wupin = ""
            ocr_ipa = ""
            ocr_following = ""

        vlm_wupin, errors = ipa_to_wupin(vlm_ipa, mapping) if vlm_ipa else ("", [])
        vlm_wupin = normalize_wupin(vlm_wupin)
        ipa_cmp = compare_parts(vlm_ipa, ocr_ipa) if ocr_row is not None and vlm_row is not None else {
            "exact": False,
            "letters_equal": False,
            "digits_equal": False,
            "only_digits_differ": False,
        }
        wupin_cmp = compare_parts(vlm_wupin, ocr_wupin) if ocr_row is not None and vlm_row is not None and not errors else {
            "exact": False,
            "letters_equal": False,
            "digits_equal": False,
            "only_digits_differ": False,
        }
        return {
            "合并状态": status,
            "source_page": source_page if source_page is not None else "",
            "pdf_page": pdf_page if pdf_page is not None else "",
            "词条": word,
            "OCR行号": ocr_idx if ocr_idx is not None else "",
            "OCR页码": ocr_page,
            "OCR汉字": ocr_word,
            "VLM_IPA": vlm_ipa,
            "OCR_IPA": ocr_ipa,
            "IPA完全相同": bool_zh(ipa_cmp["exact"]),
            "IPA数字相同": bool_zh(ipa_cmp["digits_equal"]),
            "IPA音段相同": bool_zh(ipa_cmp["letters_equal"]),
            "IPA只有数字不同": bool_zh(ipa_cmp["only_digits_differ"]),
            "IPA相似度": round(similarity(vlm_ipa, ocr_ipa), 6) if vlm_row is not None and ocr_row is not None else "",
            "VLM吴拼_由IPA反推": vlm_wupin,
            "OCR吴拼": ocr_wupin,
            "吴拼完全相同": bool_zh(wupin_cmp["exact"]),
            "吴拼数字相同": bool_zh(wupin_cmp["digits_equal"]),
            "吴拼字母相同": bool_zh(wupin_cmp["letters_equal"]),
            "吴拼只有数字不同": bool_zh(wupin_cmp["only_digits_differ"]),
            "吴拼相似度": round(similarity(vlm_wupin, ocr_wupin), 6) if vlm_row is not None and ocr_row is not None and not errors else "",
            "VLM吴拼反推状态": "ok" if not errors else ";".join(errors),
            "VLM_IPA来源": vlm_ipa_source,
            "VLM_原始IPA字段": normalize_ipa(vlm_row.get("ipa", "")) if vlm_row is not None else "",
            "VLM_ipa_raw": tone_digits(vlm_ipa_raw),
            "VLM置信度": vlm_confidence,
            "VLM备注": vlm_notes,
            "OCR后续汉字": ocr_following,
        }

    for _, vlm_row in vlm_df.iterrows():
        page = safe_int(vlm_row.get("source_page"))
        word = clean_hanzi(vlm_row.get("headword"))
        candidates = list(ocr_index.get((page, word), [])) if page is not None else []
        ocr_idx = next((idx for idx in candidates if idx not in used_ocr), None)
        if ocr_idx is None and candidates:
            ocr_idx = candidates[0]
        if ocr_idx is not None:
            used_ocr.add(int(ocr_idx))
            records.append(build_record(vlm_row, ocr_df.loc[ocr_idx], int(ocr_idx), "两边都有"))
        else:
            records.append(build_record(vlm_row, None, None, "仅VLM"))

    vlm_pages = {safe_int(page) for page in vlm_df["source_page"]}
    for idx, ocr_row in ocr_df.iterrows():
        page = safe_int(ocr_row.get("页码"))
        if int(idx) in used_ocr or page not in vlm_pages:
            continue
        records.append(build_record(None, ocr_row, int(idx), "仅OCR"))

    out_df = pd.DataFrame(records)
    out_df.sort_values(
        by=["source_page", "pdf_page", "合并状态", "词条", "OCR行号"],
        inplace=True,
        kind="stable",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    out_df.to_csv(args.out.with_suffix(".tsv"), index=False, sep="\t", encoding="utf-8", quoting=csv.QUOTE_MINIMAL)

    both = out_df[out_df["合并状态"].eq("两边都有")]
    summary = {
        "vlm_csv": str(args.vlm_csv),
        "ocr_csv": str(args.ocr_csv),
        "out_csv": str(args.out),
        "rows_total": int(len(out_df)),
        "status_counts": dict(Counter(out_df["合并状态"])),
        "matched_rows": int(len(both)),
        "ipa_exact": int(both["IPA完全相同"].eq("是").sum()),
        "ipa_digits_same": int(both["IPA数字相同"].eq("是").sum()),
        "ipa_only_digits_differ": int(both["IPA只有数字不同"].eq("是").sum()),
        "wupin_exact": int(both["吴拼完全相同"].eq("是").sum()),
        "wupin_digits_same": int(both["吴拼数字相同"].eq("是").sum()),
        "wupin_only_digits_differ": int(both["吴拼只有数字不同"].eq("是").sum()),
        "avg_ipa_similarity": float(pd.to_numeric(both["IPA相似度"], errors="coerce").mean()) if len(both) else 0.0,
        "avg_wupin_similarity": float(pd.to_numeric(both["吴拼相似度"], errors="coerce").mean()) if len(both) else 0.0,
    }
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
