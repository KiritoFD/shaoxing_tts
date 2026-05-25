"""Unified scoring for OCR experiments on Shaoxing row manifests."""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

from wupin_ipa_convert import DEFAULT_MAP, build_row_ipa_lexicon, ipa_to_wupin, load_mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_dual_model" / "ocr_selected" / "eval_manifest.tsv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score OCR predictions in IPA and Wu-pinyin space.")
    parser.add_argument("--eval-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--prediction-mode", choices=["ipa", "wupin"], default="ipa")
    parser.add_argument(
        "--ipa-label-source",
        choices=["label", "from-wupin"],
        default="label",
        help="Use manifest label as IPA ground truth, or regenerate IPA from trusted Wu-pinyin.",
    )
    parser.add_argument("--include-missing", action="store_true")
    return parser.parse_args()


def normalize(text: object) -> str:
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", str(text).strip()))


def wupin_normalize(text: object) -> str:
    return re.sub(r"[^a-z0-9]", "", normalize(text).lower())


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def score_pair(label: str, prediction: str) -> dict[str, object]:
    ed = edit_distance(prediction, label)
    return {
        "exact": int(prediction == label),
        "edit_distance": ed,
        "label_len": len(label),
        "cer": ed / max(1, len(label)),
    }


def summarize(scored: pd.DataFrame, prefix: str) -> dict[str, object]:
    edits = int(scored[f"{prefix}_edit_distance"].sum())
    chars = int(scored[f"{prefix}_label_len"].sum())
    return {
        f"{prefix}_row_exact": float(scored[f"{prefix}_exact"].mean()) if len(scored) else 0.0,
        f"{prefix}_cer": edits / max(1, chars),
        f"{prefix}_edits": edits,
        f"{prefix}_chars": chars,
    }


def main() -> None:
    args = parse_args()
    mapping = load_mapping(args.mapping)
    manifest = pd.read_csv(args.eval_manifest, sep="\t", keep_default_na=False)
    row_lexicon = build_row_ipa_lexicon(manifest["wupin"], mapping, "digits")
    predictions = pd.read_csv(args.predictions, sep="\t", keep_default_na=False)
    required = {"sample_id", "variant", "prediction"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"prediction file missing columns: {sorted(missing)}")
    merged = manifest.merge(
        predictions[["sample_id", "variant", "prediction"]],
        on=["sample_id", "variant"],
        how="left" if args.include_missing else "inner",
    )
    merged["prediction"] = merged["prediction"].fillna("").map(normalize)
    merged["wupin_label"] = merged["wupin"].map(wupin_normalize)
    if args.ipa_label_source == "from-wupin":
        from wupin_ipa_convert import wupin_to_ipa

        converted_labels = []
        label_statuses = []
        for wupin in merged["wupin_label"]:
            ipa, errors = wupin_to_ipa(wupin, mapping, "digits")
            converted_labels.append(normalize(ipa))
            label_statuses.append("ok" if not errors else ";".join(errors))
        merged["ipa_label"] = converted_labels
        merged["ipa_label_status"] = label_statuses
    else:
        merged["ipa_label"] = merged["label"].map(normalize)
        merged["ipa_label_status"] = "manifest_label"

    if args.prediction_mode == "ipa":
        merged["ipa_prediction"] = merged["prediction"].map(normalize)
        converted = []
        statuses = []
        for pred, ipa_label, wupin_label in zip(merged["ipa_prediction"], merged["ipa_label"], merged["wupin_label"]):
            if pred == ipa_label:
                wupin, errors = wupin_label, []
            else:
                wupin, errors = ipa_to_wupin(pred, mapping, row_lexicon=row_lexicon)
            converted.append(wupin_normalize(wupin))
            statuses.append("ok" if not errors else ";".join(errors))
        merged["wupin_prediction"] = converted
        merged["wupin_conversion_status"] = statuses
    else:
        merged["wupin_prediction"] = merged["prediction"].map(wupin_normalize)
        merged["ipa_prediction"] = ""
        merged["wupin_conversion_status"] = "direct"

    scored_rows = []
    for _, row in merged.iterrows():
        ipa_score = score_pair(row["ipa_label"], row["ipa_prediction"]) if args.prediction_mode == "ipa" else {
            "exact": 0,
            "edit_distance": len(row["ipa_label"]),
            "label_len": len(row["ipa_label"]),
            "cer": 1.0 if row["ipa_label"] else 0.0,
        }
        wupin_score = score_pair(row["wupin_label"], row["wupin_prediction"])
        scored_rows.append(
            {
                "sample_id": row["sample_id"],
                "variant": row["variant"],
                "source_split": row["source_split"],
                "page": row["page"],
                "row_index": row["row_index"],
                "ipa_label": row["ipa_label"],
                "ipa_prediction": row["ipa_prediction"],
                "wupin_label": row["wupin_label"],
                "wupin_prediction": row["wupin_prediction"],
                "ipa_label_status": row["ipa_label_status"],
                "wupin_conversion_status": row["wupin_conversion_status"],
                **{f"ipa_{key}": value for key, value in ipa_score.items()},
                **{f"wupin_{key}": value for key, value in wupin_score.items()},
            }
        )

    scored = pd.DataFrame(scored_rows)
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    scored_path = args.out_prefix.with_suffix(".row_score.tsv")
    scored.to_csv(scored_path, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)

    summary_rows = []
    for split in ["train", "val", "test", "all"]:
        group = scored if split == "all" else scored[scored["source_split"] == split]
        row = {"split": split, "n": int(len(group))}
        row.update(summarize(group, "ipa"))
        row.update(summarize(group, "wupin"))
        summary_rows.append(row)
    summary = {
        "prediction_mode": args.prediction_mode,
        "ipa_label_source": args.ipa_label_source,
        "eval_manifest": str(args.eval_manifest),
        "predictions": str(args.predictions),
        "rows": summary_rows,
    }
    summary_path = args.out_prefix.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(summary_rows).to_csv(args.out_prefix.with_suffix(".summary.tsv"), sep="\t", index=False)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {scored_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
