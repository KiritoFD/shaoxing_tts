"""Score syllable OCR predictions after concatenating them back to rows."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from train_crnn_ipa_digits import edit_distance, normalize_label


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_syllable_ocr" / "eval_manifest.tsv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score syllable predictions at syllable and row level.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument(
        "--exclude-flag-substrings",
        nargs="*",
        default=[],
        help="Skip manifest rows whose cleaning_flags contains any of these substrings.",
    )
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main() -> None:
    args = parse_args()
    manifest = {
        row["sample_id"]: row
        for row in read_tsv(args.manifest)
        if not any(token and token in row.get("cleaning_flags", "") for token in args.exclude_flag_substrings)
    }
    preds = {row["sample_id"]: row for row in read_tsv(args.predictions)}

    syllable_rows = []
    for sample_id, row in manifest.items():
        pred = normalize_label(preds.get(sample_id, {}).get("prediction", ""))
        label = normalize_label(row["label"])
        ed = edit_distance(pred, label)
        syllable_rows.append({**row, "prediction": pred, "edit_distance": ed, "correct": int(pred == label)})

    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in syllable_rows:
        grouped[(row["source_split"], row["page"], row["row_index"])].append(row)

    row_scores = []
    for key, group in grouped.items():
        group.sort(key=lambda row: int(row["syllable_index"]))
        label = "".join(row["label"] for row in group)
        pred = "".join(row["prediction"] for row in group)
        ed = edit_distance(pred, label)
        row_scores.append(
            {
                "source_split": key[0],
                "page": key[1],
                "row_index": key[2],
                "label": label,
                "prediction": pred,
                "edit_distance": ed,
                "cer": ed / max(1, len(label)),
                "exact": int(pred == label),
                "syllables": len(group),
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "source_split",
            "page",
            "row_index",
            "label",
            "prediction",
            "edit_distance",
            "cer",
            "exact",
            "syllables",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(row_scores)

    summary_rows = []
    for split in ["train", "val", "test", "all"]:
        rows = row_scores if split == "all" else [row for row in row_scores if row["source_split"] == split]
        exact = sum(int(row["exact"]) for row in rows) / max(1, len(rows))
        edits = sum(int(row["edit_distance"]) for row in rows)
        chars = sum(len(row["label"]) for row in rows)
        row = {"split": split, "n_rows": len(rows), "row_exact": exact, "row_cer": edits / max(1, chars)}
        summary_rows.append(row)
        print(f"{split}: rows={len(rows)} row_exact={row['row_exact']:.4f} row_CER={row['row_cer']:.4f}")

    for split in ["train", "val", "test", "all"]:
        rows = syllable_rows if split == "all" else [row for row in syllable_rows if row["source_split"] == split]
        exact = sum(int(row["correct"]) for row in rows) / max(1, len(rows))
        edits = sum(int(row["edit_distance"]) for row in rows)
        chars = sum(len(row["label"]) for row in rows)
        for row in summary_rows:
            if row["split"] == split:
                row.update({"n_syllables": len(rows), "syllable_exact": exact, "syllable_cer": edits / max(1, chars)})
                break
        print(f"{split}: syllables={len(rows)} syllable_exact={exact:.4f} syllable_CER={edits / max(1, chars):.4f}")

    summary_path = args.summary or args.out.with_suffix(".summary.json")
    summary = {
        "manifest": str(args.manifest),
        "predictions": str(args.predictions),
        "row_score": str(args.out),
        "exclude_flag_substrings": args.exclude_flag_substrings,
        "rows": summary_rows,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with args.out.with_suffix(".summary.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"wrote {args.out}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
