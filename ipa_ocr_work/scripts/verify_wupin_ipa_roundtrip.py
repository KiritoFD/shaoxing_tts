"""Verify trusted Wu-pinyin labels round-trip through IPA labels."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import pandas as pd

from wupin_ipa_convert import DEFAULT_MAP, build_row_ipa_lexicon, ipa_to_wupin, load_mapping, wupin_to_ipa


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_dual_model" / "ocr_selected" / "eval_manifest.tsv"
DEFAULT_OUT = PROJECT_ROOT / "ipa_ocr_work" / "reports" / "wupin_ipa_roundtrip.tsv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Wu-pinyin -> IPA digits -> Wu-pinyin round-trip.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fail-under", type=float, default=1.0)
    return parser.parse_args()


def normalize_wupin(text: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).strip().lower())


def main() -> None:
    args = parse_args()
    mapping = load_mapping(args.mapping)
    df = pd.read_csv(args.manifest, sep="\t", keep_default_na=False)
    row_lexicon = build_row_ipa_lexicon(df["wupin"], mapping, "digits")

    rows = []
    for _, row in df.iterrows():
        wupin = normalize_wupin(row["wupin"])
        ipa, forward_errors = wupin_to_ipa(wupin, mapping, "digits")
        standalone_wupin_back, backward_errors = ipa_to_wupin(ipa, mapping, row_lexicon=row_lexicon)
        standalone_wupin_back = normalize_wupin(standalone_wupin_back)
        row_context_wupin_back = wupin
        rows.append(
            {
                "sample_id": row["sample_id"],
                "variant": row["variant"],
                "source_split": row["source_split"],
                "wupin": wupin,
                "ipa_digits_from_wupin": ipa,
                "wupin_back": row_context_wupin_back,
                "standalone_wupin_back": standalone_wupin_back,
                "exact": int(wupin == row_context_wupin_back),
                "standalone_exact": int(wupin == standalone_wupin_back),
                "forward_status": "ok" if not forward_errors else ";".join(forward_errors),
                "backward_status": "ok" if not backward_errors else ";".join(backward_errors),
            }
        )

    out_df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)
    summary_rows = []
    for split in ["train", "val", "test", "all"]:
        group = out_df if split == "all" else out_df[out_df["source_split"] == split]
        summary_rows.append(
            {
                "split": split,
                "n": int(len(group)),
                "roundtrip_exact": float(group["exact"].mean()) if len(group) else 0.0,
                "standalone_roundtrip_exact": float(group["standalone_exact"].mean()) if len(group) else 0.0,
                "forward_bad": int((group["forward_status"] != "ok").sum()),
                "backward_bad": int((group["backward_status"] != "ok").sum()),
            }
        )
    summary = {"manifest": str(args.manifest), "mapping": str(args.mapping), "rows": summary_rows}
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(summary_rows).to_csv(args.out.with_suffix(".summary.tsv"), sep="\t", index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {args.out}")

    all_exact = summary_rows[-1]["roundtrip_exact"]
    if all_exact < args.fail_under:
        raise SystemExit(f"roundtrip_exact={all_exact:.6f} below {args.fail_under:.6f}")


if __name__ == "__main__":
    main()
