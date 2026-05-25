"""Filter a TrOCR dataset manifest and copy its crop images.

This is intended for making explicit clean tiers from a generated OCR dataset,
for example keeping only matched rows with high direct-span confidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import pandas as pd

from wupin_ipa_convert import DEFAULT_MAP, ipa_to_wupin, load_mapping, wupin_to_ipa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter TrOCR eval_manifest.tsv and copy images.")
    parser.add_argument("--src-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-confidence", type=float, default=None)
    parser.add_argument("--quality", action="append", default=None, help="Allowed quality value; repeatable.")
    parser.add_argument("--cleaning-flags", action="append", default=None, help="Allowed cleaning_flags value; repeatable.")
    parser.add_argument("--ipa-conversion-status", action="append", default=None, help="Allowed ipa_conversion_status value; repeatable.")
    parser.add_argument("--require-roundtrip-exact", action="store_true", help="Keep only rows where wupin -> IPA -> wupin is exactly unchanged.")
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and args.overwrite:
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.src_dir / "eval_manifest.tsv", sep="\t", keep_default_na=False)
    mask = pd.Series(True, index=df.index)
    if args.min_confidence is not None:
        mask &= pd.to_numeric(df["span_confidence"], errors="coerce").fillna(-999) >= args.min_confidence
    if args.quality:
        mask &= df["quality"].isin(args.quality)
    if args.cleaning_flags:
        mask &= df["cleaning_flags"].isin(args.cleaning_flags)
    if args.ipa_conversion_status:
        mask &= df["ipa_conversion_status"].isin(args.ipa_conversion_status)
    if args.require_roundtrip_exact:
        mapping = load_mapping(args.mapping)
        keep = []
        for wupin in df["wupin"].astype(str):
            ipa, forward_errors = wupin_to_ipa(wupin, mapping, "digits")
            if forward_errors:
                keep.append(False)
                continue
            back, backward_errors = ipa_to_wupin(ipa, mapping)
            keep.append(not backward_errors and back == wupin)
        mask &= pd.Series(keep, index=df.index)
    out = df[mask].copy()
    if args.variant:
        out["variant"] = args.variant

    for image in out["image"]:
        src = args.src_dir / str(image)
        dst = args.out_dir / str(image)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    out = out.sort_values(["source_split", "page", "row_index"]).reset_index(drop=True)
    out.to_csv(args.out_dir / "eval_manifest.tsv", sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)
    summary = {
        "src_dir": str(args.src_dir),
        "rows": int(len(out)),
        "split_counts": out["source_split"].value_counts().to_dict() if len(out) else {},
        "quality_counts": out["quality"].value_counts().to_dict() if "quality" in out else {},
        "cleaning_flags_counts": out["cleaning_flags"].value_counts().to_dict() if "cleaning_flags" in out else {},
        "min_confidence": args.min_confidence,
        "quality": args.quality,
        "cleaning_flags": args.cleaning_flags,
        "ipa_conversion_status": args.ipa_conversion_status,
        "require_roundtrip_exact": args.require_roundtrip_exact,
        "variant": args.variant,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
