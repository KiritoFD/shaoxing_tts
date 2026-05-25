"""Refresh IPA labels in OCR manifests from trusted Wu-pinyin.

Use this after changing `wupin_ipa_map.json`.  It treats Wu-pinyin as the
source of truth and rewrites derived IPA target columns in every manifest.
Images, splits, crop boxes, qualities, and legacy columns are left untouched.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from wupin_ipa_convert import DEFAULT_MAP, load_mapping, wupin_to_ipa


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_pdf136_clean"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh manifest IPA labels from Wu-pinyin.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--pattern", default="**/eval_manifest.tsv")
    parser.add_argument("--tone-style", choices=["digits", "superscript", "none"], default="digits")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def should_refresh(path: Path, df: pd.DataFrame) -> bool:
    if "wupin" not in df.columns:
        return False
    if len(df) == 0:
        return False
    # Smoke synthetic manifests are also useful to refresh if they carry wupin,
    # but rows with synthetic=True and no real wupin should be skipped row-wise.
    return True


def refresh_one(path: Path, mapping: dict, tone_style: str, dry_run: bool) -> dict[str, object]:
    df = pd.read_csv(path, sep="\t", keep_default_na=False)
    if not should_refresh(path, df):
        return {
            "path": str(path),
            "rows": int(len(df)),
            "refreshed": False,
            "reason": "missing_or_empty_wupin",
        }

    old_label = df["label"].astype(str).copy() if "label" in df.columns else pd.Series([""] * len(df))
    old_ipa = df["ipa"].astype(str).copy() if "ipa" in df.columns else pd.Series([""] * len(df))
    labels: list[str] = []
    statuses: list[str] = []
    for wupin in df["wupin"]:
        if str(wupin).strip() == "":
            labels.append("")
            statuses.append("empty_wupin")
            continue
        ipa, errors = wupin_to_ipa(str(wupin), mapping, tone_style)
        labels.append(ipa)
        statuses.append("ok" if not errors else ";".join(errors))

    label_series = pd.Series(labels, index=df.index)
    status_series = pd.Series(statuses, index=df.index)
    ok_mask = status_series.eq("ok")

    if "label" in df.columns:
        df.loc[ok_mask, "label"] = label_series.loc[ok_mask]
    if "ipa" in df.columns:
        df.loc[ok_mask, "ipa"] = label_series.loc[ok_mask]
    else:
        df["ipa"] = ""
        df.loc[ok_mask, "ipa"] = label_series.loc[ok_mask]
    if "ipa_conversion_status" in df.columns:
        df["ipa_conversion_status"] = status_series
    else:
        df["ipa_conversion_status"] = status_series
    if "ipa_label_source" in df.columns:
        df["ipa_label_source"] = "rule_from_trusted_wupin_refreshed"

    changed_label = int(((old_label != df.get("label", old_label).astype(str)) & ok_mask).sum()) if "label" in df.columns else 0
    changed_ipa = int(((old_ipa != df["ipa"].astype(str)) & ok_mask).sum())
    bad = int((~ok_mask).sum())
    if not dry_run:
        df.to_csv(path, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)

    examples = []
    changed_rows = df.loc[(old_label != df.get("label", old_label).astype(str)) & ok_mask].head(5) if "label" in df.columns else df.head(0)
    for row_index, row in changed_rows.iterrows():
        examples.append(
            {
                "row": int(row_index),
                "sample_id": str(row.get("sample_id", "")),
                "wupin": str(row.get("wupin", "")),
                "old_label": str(old_label.iloc[row_index]),
                "new_label": str(row.get("label", "")),
            }
        )

    return {
        "path": str(path),
        "rows": int(len(df)),
        "refreshed": True,
        "changed_label": changed_label,
        "changed_ipa": changed_ipa,
        "bad_conversion": bad,
        "status_counts": status_series.value_counts().to_dict(),
        "examples": examples,
    }


def main() -> None:
    args = parse_args()
    mapping = load_mapping(args.mapping)
    paths = sorted(args.root.glob(args.pattern))
    reports = [refresh_one(path, mapping, args.tone_style, args.dry_run) for path in paths]
    report = {
        "root": str(args.root),
        "mapping": str(args.mapping),
        "tone_style": args.tone_style,
        "dry_run": args.dry_run,
        "manifest_count": len(paths),
        "refreshed_count": sum(1 for row in reports if row.get("refreshed")),
        "total_rows": sum(int(row.get("rows", 0)) for row in reports if row.get("refreshed")),
        "total_changed_label": sum(int(row.get("changed_label", 0)) for row in reports),
        "total_bad_conversion": sum(int(row.get("bad_conversion", 0)) for row in reports),
        "manifests": reports,
    }
    out = args.root / "ipa_refresh_report.json"
    if not args.dry_run:
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not args.dry_run:
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
