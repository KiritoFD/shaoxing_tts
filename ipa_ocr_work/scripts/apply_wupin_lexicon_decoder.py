"""Apply constrained Wu-pinyin decoding to OCR predictions.

This postprocesses model output into the project target format. It is intended
for two fair use cases:

1. Train-split syllable lexicon correction for validation/test experiments.
2. Full-book production decoding when a larger trusted lexicon is available.

The default is conservative: build the lexicon from train rows only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import unicodedata
from pathlib import Path

import pandas as pd

from score_ocr_experiment import edit_distance, normalize, wupin_normalize
from wupin_ipa_convert import DEFAULT_MAP, ipa_to_wupin, load_mapping, split_syllables


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_dual_model" / "ocr_selected" / "eval_manifest.tsv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode OCR predictions with a Wu-pinyin lexicon.")
    parser.add_argument("--eval-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--prediction-mode", choices=["ipa", "wupin"], default="ipa")
    parser.add_argument("--lexicon-split", choices=["train", "train-val", "all"], default="train")
    parser.add_argument("--max-syllable-distance", type=int, default=2)
    parser.add_argument("--row-nearest", action="store_true", help="Also try nearest complete train-row label.")
    parser.add_argument("--row-nearest-max-cer", type=float, default=0.20)
    return parser.parse_args()


def syllable_text(base: str, tone: str) -> str:
    return f"{base}{tone}"


def read_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", keep_default_na=False)
    required = {"sample_id", "prediction"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"prediction file missing columns: {sorted(missing)}")
    if "variant" not in df.columns:
        df["variant"] = ""
    return df


def lexicon_rows(manifest: pd.DataFrame, split: str) -> pd.DataFrame:
    if split == "all":
        return manifest
    allowed = {"train"} if split == "train" else {"train", "val"}
    return manifest[manifest["source_split"].isin(allowed)]


def build_syllable_lexicon(wupins: pd.Series) -> list[str]:
    values: set[str] = set()
    for wupin in wupins:
        syllables, remainder = split_syllables(wupin_normalize(wupin))
        if remainder:
            continue
        for base, tone in syllables:
            values.add(syllable_text(base, tone))
    return sorted(values, key=lambda item: (len(item), item))


def nearest_syllable(token: str, lexicon: list[str], max_distance: int) -> tuple[str, int]:
    if token in lexicon:
        return token, 0
    best = token
    best_dist = math.inf
    for item in lexicon:
        if abs(len(item) - len(token)) > max_distance:
            continue
        dist = edit_distance(token, item)
        if dist < best_dist:
            best = item
            best_dist = dist
            if dist == 1:
                break
    if best_dist <= max_distance:
        return best, int(best_dist)
    return token, int(best_dist if best_dist < math.inf else 999)


def split_or_fallback(text: str) -> list[str]:
    syllables, remainder = split_syllables(text)
    if not remainder and syllables:
        return [syllable_text(base, tone) for base, tone in syllables]
    return re.findall(r"[a-z]+[0-9]+", text)


def decode_syllables(wupin: str, lexicon: list[str], max_distance: int) -> tuple[str, str, int]:
    tokens = split_or_fallback(wupin)
    if not tokens:
        return wupin, "no_parse", 0
    corrected: list[str] = []
    changed = 0
    statuses = []
    for token in tokens:
        fixed, dist = nearest_syllable(token, lexicon, max_distance)
        corrected.append(fixed)
        if fixed != token:
            changed += 1
            statuses.append(f"{token}->{fixed}:{dist}")
    return "".join(corrected), "ok" if not statuses else ";".join(statuses), changed


def nearest_row(pred: str, row_lexicon: list[str], max_cer: float) -> tuple[str, str]:
    if not pred or not row_lexicon:
        return pred, "row_skip"
    best = pred
    best_dist = math.inf
    for label in row_lexicon:
        if abs(len(label) - len(pred)) > max(3, int(max(len(label), len(pred)) * 0.35)):
            continue
        dist = edit_distance(pred, label)
        if dist < best_dist:
            best = label
            best_dist = dist
    cer = best_dist / max(1, len(best))
    if best != pred and cer <= max_cer:
        return best, f"row_nearest:{best_dist}:{cer:.4f}"
    return pred, "row_keep"


def main() -> None:
    args = parse_args()
    mapping = load_mapping(args.mapping)
    manifest = pd.read_csv(args.eval_manifest, sep="\t", keep_default_na=False)
    predictions = read_predictions(args.predictions)
    merge_keys = ["sample_id"] if (predictions["variant"] == "").all() else ["sample_id", "variant"]
    merged = manifest.merge(predictions[merge_keys + ["prediction"]], on=merge_keys, how="left")
    merged["prediction"] = merged["prediction"].fillna("").map(normalize)

    lex_rows = lexicon_rows(manifest, args.lexicon_split)
    syllable_lexicon = build_syllable_lexicon(lex_rows["wupin"])
    row_lexicon = sorted({wupin_normalize(value) for value in lex_rows["wupin"] if wupin_normalize(value)})

    out_rows = []
    for _, row in merged.iterrows():
        raw = normalize(row["prediction"])
        if args.prediction_mode == "ipa":
            wupin, errors = ipa_to_wupin(raw, mapping)
            initial_status = "ipa_ok" if not errors else ";".join(errors)
            wupin = wupin_normalize(wupin)
        else:
            wupin = wupin_normalize(raw)
            initial_status = "direct"
        corrected, syl_status, changed = decode_syllables(wupin, syllable_lexicon, args.max_syllable_distance)
        row_status = "row_disabled"
        if args.row_nearest:
            row_corrected, row_status = nearest_row(corrected, row_lexicon, args.row_nearest_max_cer)
            corrected = row_corrected
        out_rows.append(
            {
                "sample_id": row["sample_id"],
                "variant": row.get("variant", ""),
                "prediction": corrected,
                "raw_prediction": raw,
                "raw_wupin": wupin,
                "decode_status": initial_status,
                "syllable_status": syl_status,
                "row_status": row_status,
                "changed_syllables": changed,
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["sample_id", "variant", "prediction", "raw_prediction", "raw_wupin", "decode_status", "syllable_status", "row_status", "changed_syllables"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(out_rows)

    summary = {
        "eval_manifest": str(args.eval_manifest),
        "predictions": str(args.predictions),
        "out": str(args.out),
        "prediction_mode": args.prediction_mode,
        "lexicon_split": args.lexicon_split,
        "syllable_lexicon_size": len(syllable_lexicon),
        "row_lexicon_size": len(row_lexicon),
        "rows": len(out_rows),
        "changed_rows": sum(1 for row in out_rows if row["raw_wupin"] != row["prediction"]),
        "changed_syllables": sum(int(row["changed_syllables"]) for row in out_rows),
    }
    summary_path = args.summary or args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
