"""Audit Shaoxing Wu-pinyin rule compliance for OCR training data."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from wupin_ipa_convert import DEFAULT_MAP, canonicalize_wupin_base, canonicalize_wupin_label, load_mapping, wupin_to_ipa


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = PROJECT_ROOT / "result_all_converted.clean.csv"
DEFAULT_MANIFEST = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_pdf136_clean" / "ocr_selected_all" / "eval_manifest.tsv"
DEFAULT_SYLLABLES = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_pdf136_clean" / "syllable_ocr_all" / "eval_manifest.tsv"
DEFAULT_OUT = PROJECT_ROOT / "ipa_ocr_work" / "reports" / "wupin_rule_audit_pdf136.json"
SYLLABLE_RE = re.compile(r"([a-z]+)([0-9]+)")


CONFIRMED_INITIALS = [
    "tsh",
    "ts",
    "dz",
    "ph",
    "th",
    "ch",
    "sh",
    "zh",
    "gn",
    "ng",
    "kh",
    "gh",
    "p",
    "b",
    "m",
    "f",
    "v",
    "t",
    "d",
    "n",
    "l",
    "s",
    "z",
    "c",
    "j",
    "k",
    "g",
    "h",
    "y",
    "w",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Wu-pinyin rules and OCR manifests.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--syllables", type=Path, default=DEFAULT_SYLLABLES)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix == ".tsv" else ","
    return pd.read_csv(path, sep=sep, keep_default_na=False)


def split_syllables(text: object) -> list[tuple[str, str]]:
    return SYLLABLE_RE.findall(str(text).strip().lower())


def split_initial_final(base: str, mapping: dict) -> tuple[str, str]:
    for initial in sorted(mapping["initials"], key=len, reverse=True):
        if base.startswith(initial):
            return initial, base[len(initial) :]
    return "", base


def sample_rows(df: pd.DataFrame, mask: pd.Series, columns: list[str], limit: int = 8) -> list[dict[str, str]]:
    rows = df[mask].head(limit)
    return rows[[col for col in columns if col in rows.columns]].to_dict("records")


def main() -> None:
    args = parse_args()
    mapping = load_mapping(args.mapping)
    full = read_table(args.csv)
    manifest = read_table(args.manifest)
    syllables = read_table(args.syllables)

    full_wupin = full["wupin"].astype(str).str.lower()
    manifest_wupin = manifest["wupin"].astype(str).str.lower()

    raw_variant_counts = {
        "ghi": int(full_wupin.str.contains("ghi", regex=False).sum()),
        "ghu": int(full_wupin.str.contains("ghu", regex=False).sum()),
        "ieq": int(full_wupin.str.contains("ieq", regex=False).sum()),
        "ieh": int(full_wupin.str.contains("ieh", regex=False).sum()),
    }

    conversion_bad = []
    canonical_changes = 0
    base_counter: Counter[str] = Counter()
    final_counter: Counter[str] = Counter()
    initial_counter: Counter[str] = Counter()
    unknown_initial_final = []
    for _, row in full.iterrows():
        original = str(row["wupin"]).strip().lower()
        canonical = canonicalize_wupin_label(original)
        if canonical != original:
            canonical_changes += 1
        ipa, errors = wupin_to_ipa(canonical, mapping, "digits")
        if errors:
            conversion_bad.append({"row_id": row.get("row_id", ""), "wupin": original, "errors": ";".join(errors)})
        for base, _tone in split_syllables(canonical):
            base_counter[base] += 1
            canonical_base = canonicalize_wupin_base(base)
            if canonical_base in mapping["whole_syllables"]:
                initial_counter["<whole>"] += 1
                final_counter["<whole>"] += 1
                continue
            initial, final = split_initial_final(canonical_base, mapping)
            initial_counter[initial] += 1
            final_counter[final] += 1
            if initial and initial not in CONFIRMED_INITIALS:
                unknown_initial_final.append({"wupin": original, "base": base, "initial": initial, "final": final})
            if final not in mapping["finals"]:
                unknown_initial_final.append({"wupin": original, "base": base, "initial": initial, "final": final})

    labels = manifest["label"].astype(str)
    source_root = args.manifest.parent
    image_exists = manifest["image"].map(lambda value: (source_root / str(value)).resolve().exists())

    special_examples = {}
    for pattern in ["sy", "zy", "yu", "yan", "wo", "wu", "iq", "yoeq"]:
        special_examples[pattern] = sample_rows(
            manifest,
            manifest_wupin.str.contains(pattern, regex=False),
            ["hanzi", "wupin", "label", "legacy_ipa_digits", "source_split", "pdf_page", "row_index"],
            limit=5,
        )

    payload = {
        "inputs": {
            "csv": str(args.csv),
            "manifest": str(args.manifest),
            "syllables": str(args.syllables),
            "mapping": str(args.mapping),
        },
        "counts": {
            "csv_rows": int(len(full)),
            "csv_has_wupin": int(full["wupin"].astype(str).str.strip().ne("").sum()),
            "manifest_rows": int(len(manifest)),
            "syllable_rows": int(len(syllables)),
            "manifest_split_counts": manifest["source_split"].value_counts().to_dict(),
            "manifest_original_split_counts": manifest.get("original_source_split", manifest["source_split"]).value_counts().to_dict(),
            "canonicalized_full_csv_rows": int(canonical_changes),
            "new_label_differs_from_legacy": int((manifest["label"].astype(str) != manifest["legacy_ipa_digits"].astype(str)).sum())
            if "legacy_ipa_digits" in manifest.columns
            else None,
        },
        "rule_checks": {
            "raw_variant_counts_full_csv": raw_variant_counts,
            "conversion_bad_full_csv": len(conversion_bad),
            "conversion_bad_examples": conversion_bad[:20],
            "unknown_initial_or_final": len(unknown_initial_final),
            "unknown_initial_or_final_examples": unknown_initial_final[:20],
            "label_contains_apical_i_ɿ": int(labels.str.contains("ɿ", regex=False).sum()),
            "label_contains_question_mark": int(labels.str.contains("?", regex=False).sum()),
            "manifest_bad_conversion_status": manifest["ipa_conversion_status"].value_counts().drop(labels=["ok"], errors="ignore").to_dict()
            if "ipa_conversion_status" in manifest.columns
            else {},
            "missing_manifest_images": int((~image_exists).sum()),
        },
        "top_bases": dict(base_counter.most_common(40)),
        "top_initials": dict(initial_counter.most_common(40)),
        "top_finals": dict(final_counter.most_common(60)),
        "special_examples": special_examples,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    inventory_review = []
    for _, row in manifest.iterrows():
        for base, tone in split_syllables(row["wupin"]):
            canonical_base = canonicalize_wupin_base(base)
            if canonical_base in mapping["whole_syllables"]:
                continue
            initial, final = split_initial_final(canonical_base, mapping)
            if (initial and initial not in CONFIRMED_INITIALS) or final not in mapping["finals"]:
                inventory_review.append(
                    {
                        "sample_id": row.get("sample_id", ""),
                        "source_split": row.get("source_split", ""),
                        "pdf_page": row.get("pdf_page", ""),
                        "page": row.get("page", ""),
                        "row_index": row.get("row_index", ""),
                        "hanzi": row.get("hanzi", ""),
                        "wupin": row.get("wupin", ""),
                        "base": base,
                        "tone": tone,
                        "initial": initial,
                        "final": final,
                        "label": row.get("label", ""),
                    }
                )
    pd.DataFrame(inventory_review).to_csv(args.out.with_name(args.out.stem + ".inventory_review.tsv"), sep="\t", index=False)
    with args.out.with_suffix(".summary.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["metric", "value"])
        for group in ["counts", "rule_checks"]:
            for key, value in payload[group].items():
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, ensure_ascii=False)
                writer.writerow([f"{group}.{key}", value])
    print(json.dumps(payload["counts"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["rule_checks"], ensure_ascii=False, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
