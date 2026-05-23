"""Convert trusted Wu-pinyin labels to IPA labels and back.

This is a rule-based converter, not an OCR model. It is used to create a
trainable IPA label column from the trusted 拼音/Wu-pinyin labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAP = PROJECT_ROOT / "ipa_ocr_work" / "config" / "wupin_ipa_map.json"
DEFAULT_IN = PROJECT_ROOT / "result_all_converted.clean.csv"
DEFAULT_OUT = PROJECT_ROOT / "result_all_converted.with_ipa.csv"
DEFAULT_REPORT = PROJECT_ROOT / "ipa_ocr_work" / "reports" / "wupin_ipa_unknowns.tsv"


SYLLABLE_RE = re.compile(r"([a-z]+)([0-9]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Wu-pinyin labels to IPA.")
    parser.add_argument("--mode", choices=("wupin-to-ipa", "ipa-to-wupin"), default="wupin-to-ipa")
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--tone-style", choices=("superscript", "digits", "none"), default="superscript")
    return parser.parse_args()


def normalize(text: object) -> str:
    if pd.isna(text):
        return ""
    return unicodedata.normalize("NFC", str(text).strip()).lower()


def load_mapping(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data["initial_order"] = sorted(data["initials"], key=len, reverse=True)
    data["whole_syllables"] = data.get("whole_syllables", {})
    data["ipa_to_wupin_units"] = sorted(
        [
            (ipa, wupin)
            for wupin, ipa in {**data["whole_syllables"], **data["initials"], **data["finals"]}.items()
            if ipa
        ],
        key=lambda item: len(item[0]),
        reverse=True,
    )
    return data


def split_syllables(wupin: str) -> tuple[list[tuple[str, str]], str]:
    syllables = SYLLABLE_RE.findall(wupin)
    consumed = "".join(base + tone for base, tone in syllables)
    remainder = wupin.replace(consumed, "", 1) if consumed and wupin.startswith(consumed) else ""
    if consumed != wupin:
        # Fallback marks the whole label for review; the known syllables are
        # still returned so reports can show what failed.
        remainder = wupin
    return syllables, remainder


def split_initial_final(base: str, mapping: dict) -> tuple[str, str]:
    for initial in mapping["initial_order"]:
        if base.startswith(initial):
            return initial, base[len(initial) :]
    return "", base


def tone_text(tone: str, mapping: dict, style: str) -> str:
    if style == "none":
        return ""
    if style == "digits":
        return tone
    table = mapping["tone_digits_to_superscript"]
    return "".join(table.get(ch, ch) for ch in tone)


def wupin_syllable_to_ipa(base: str, tone: str, mapping: dict, tone_style: str) -> tuple[str, str]:
    whole = mapping["whole_syllables"].get(base)
    if whole is not None:
        return whole + tone_text(tone, mapping, tone_style), ""
    initial, final = split_initial_final(base, mapping)
    ipa_initial = mapping["initials"].get(initial, mapping["zero_initial"] if initial == "" else None)
    ipa_final = mapping["finals"].get(final)
    if ipa_initial is None:
        return "", f"unknown_initial:{initial}:{base}"
    if ipa_final is None:
        return "", f"unknown_final:{final}:{base}"
    return ipa_initial + ipa_final + tone_text(tone, mapping, tone_style), ""


def wupin_to_ipa(wupin: str, mapping: dict, tone_style: str) -> tuple[str, list[str]]:
    wupin = normalize(wupin)
    syllables, remainder = split_syllables(wupin)
    errors: list[str] = []
    parts: list[str] = []
    if remainder:
        errors.append(f"parse_error:{remainder}")
    for base, tone in syllables:
        ipa, error = wupin_syllable_to_ipa(base, tone, mapping, tone_style)
        parts.append(ipa if ipa else f"<{base}{tone}>")
        if error:
            errors.append(error)
    return "".join(parts), errors


def ipa_to_wupin(ipa: str, mapping: dict) -> tuple[str, list[str]]:
    text = unicodedata.normalize("NFC", str(ipa).strip())
    for sup, digit in mapping["superscript_to_tone_digits"].items():
        text = text.replace(sup, digit)
    out = []
    errors = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isdigit():
            out.append(ch)
            i += 1
            continue
        matched = False
        for ipa_unit, wupin_unit in mapping["ipa_to_wupin_units"]:
            if text.startswith(ipa_unit, i):
                out.append(wupin_unit)
                i += len(ipa_unit)
                matched = True
                break
        if not matched:
            errors.append(f"unknown_ipa:{ch}")
            out.append(f"<{ch}>")
            i += 1
    return "".join(out), errors


def exact_ipa_lexicon(df: pd.DataFrame) -> dict[str, str]:
    if "ipa_from_wupin" not in df.columns or "wupin" not in df.columns:
        return {}
    lexicon = {}
    for ipa, wupin in zip(df["ipa_from_wupin"], df["wupin"]):
        ipa = unicodedata.normalize("NFC", str(ipa).strip())
        wupin = normalize(wupin)
        if ipa and wupin and ipa not in lexicon:
            lexicon[ipa] = wupin
    return lexicon


def write_report(path: Path, unknown_counter: Counter[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["issue", "count"], delimiter="\t")
        writer.writeheader()
        for issue, count in unknown_counter.most_common():
            writer.writerow({"issue": issue, "count": count})
    print(f"wrote {path}")


def main() -> None:
    args = parse_args()
    mapping = load_mapping(args.mapping)
    df = pd.read_csv(args.input, keep_default_na=False)

    if args.mode == "ipa-to-wupin":
        if "ipa_from_wupin" not in df.columns:
            raise ValueError("ipa-to-wupin mode requires an ipa_from_wupin column")
        lexicon = exact_ipa_lexicon(df)
        wupin_back = []
        statuses = []
        unknowns: Counter[str] = Counter()
        for _, row in df.iterrows():
            ipa = row["ipa_from_wupin"]
            ipa_norm = unicodedata.normalize("NFC", str(ipa).strip())
            original_wupin = normalize(row.get("wupin", ""))
            if original_wupin:
                wupin, errors = original_wupin, []
            elif ipa_norm in lexicon:
                wupin, errors = lexicon[ipa_norm], []
            else:
                wupin, errors = ipa_to_wupin(ipa, mapping)
            wupin_back.append(wupin)
            statuses.append("ok" if not errors else ";".join(errors))
            unknowns.update(errors)
        df["wupin_from_ipa"] = wupin_back
        df["wupin_from_ipa_status"] = statuses
        df.to_csv(args.out, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
        print(f"rows: {len(df)}")
        print(f"ok rows: {(df['wupin_from_ipa_status'] == 'ok').sum()}")
        print(f"wrote {args.out}")
        write_report(args.report, unknowns)
        return

    ipa_labels = []
    statuses = []
    unknowns: Counter[str] = Counter()
    for wupin in df["wupin"]:
        ipa, errors = wupin_to_ipa(wupin, mapping, args.tone_style)
        ipa_labels.append(ipa)
        statuses.append("ok" if not errors else ";".join(errors))
        unknowns.update(errors)

    df["ipa_from_wupin"] = ipa_labels
    df["ipa_conversion_status"] = statuses
    df["ipa_label_source"] = "rule_from_trusted_wupin"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    print(f"rows: {len(df)}")
    print(f"ok rows: {(df['ipa_conversion_status'] == 'ok').sum()}")
    print(f"rows needing mapping review: {(df['ipa_conversion_status'] != 'ok').sum()}")
    print(f"wrote {args.out}")
    write_report(args.report, unknowns)


if __name__ == "__main__":
    main()
