"""Build a syllable-level OCR manifest from tone detector crops.

Each sample is one visually segmented syllable. The target is
`ipa_base + selected_tone`, which can later be concatenated back to the row
text in syllable order.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DETECTOR_DIR = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_tone_detector_clustered_v2"
DEFAULT_OUT = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_syllable_ocr"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build syllable OCR eval_manifest.tsv.")
    parser.add_argument("--detector-dir", type=Path, default=DEFAULT_DETECTOR_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    in_path = args.detector_dir / "detector_manifest.tsv"
    out_path = args.out_dir / "eval_manifest.tsv"
    rows = []
    with in_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            label = f"{row.get('ipa_base', '')}{row.get('selected_tone', '')}".strip()
            if not label:
                continue
            item = {
                "sample_id": row["sample_id"],
                "variant": "syllable_crop",
                "image": (args.detector_dir / row["image"]).resolve().relative_to(args.out_dir.resolve(), walk_up=True).as_posix(),
                "label": label,
                "source_split": row["source_split"],
                "page": row["page"],
                "row_index": row["row_index"],
                "syllable_index": row["syllable_index"],
                "tone_label": row["label"],
                "tone_policy": row["tone_policy"],
                "wupin_base": row.get("wupin_base", ""),
                "ipa_base": row.get("ipa_base", ""),
                "selected_tone": row.get("selected_tone", ""),
            }
            rows.append(item)

    rows.sort(key=lambda row: (row["source_split"], int(row["page"]), int(row["row_index"]), int(row["syllable_index"])))
    with out_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["source_split"], row["variant"])
        counts[key] = counts.get(key, 0) + 1
    lines = [f"rows: {len(rows)}", f"unique_labels: {len({row['label'] for row in rows})}", "split counts:"]
    for key, value in sorted(counts.items()):
        lines.append(f"{key[0]}\t{value}")
    (args.out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
