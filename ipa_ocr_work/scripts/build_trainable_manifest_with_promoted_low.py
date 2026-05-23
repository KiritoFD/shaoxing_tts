"""Build a trainable manifest with model-vetted low-match rows promoted."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build promoted trainable IPA manifest.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--tone-flag-manifest", type=Path, required=True)
    parser.add_argument("--low-eval-manifest", type=Path, required=True)
    parser.add_argument("--low-predictions", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--low-cer-threshold", type=float, default=0.2)
    return parser.parse_args()


def split_for_page(page: int) -> str:
    if page % 10 == 9:
        return "test"
    if page % 10 == 8:
        return "val"
    return "train"


def relative_image_path(out_dir: Path, dataset_dir: Path, image: str) -> str:
    return (dataset_dir / image).resolve().relative_to(out_dir.resolve(), walk_up=True).as_posix()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    flagged = pd.read_csv(args.tone_flag_manifest, sep="\t", keep_default_na=False)
    low_eval = pd.read_csv(args.low_eval_manifest, sep="\t", keep_default_na=False)
    low_pred = pd.read_csv(args.low_predictions, sep="\t", keep_default_na=False)
    low_scored = low_eval.merge(low_pred[["sample_id", "prediction", "cer"]], on="sample_id", how="inner")
    promoted_keys = set(
        zip(
            low_scored[low_scored["cer"] <= args.low_cer_threshold]["page"].astype(int),
            low_scored[low_scored["cer"] <= args.low_cer_threshold]["row_index"].astype(int),
        )
    )

    rows = []
    for _, row in flagged.iterrows():
        if not row.get("image", "") or not row.get("ipa_digits", ""):
            continue
        if row.get("tone_position") == "both_upper_lower":
            continue
        quality = row.get("quality", "")
        page = int(row["page"])
        row_index = int(row["row_index"])
        promoted = False
        if quality in {"matched", "weak_match"}:
            promoted = True
        elif quality == "low_match" and (page, row_index) in promoted_keys:
            quality = f"promoted_low_cer{args.low_cer_threshold:g}"
            promoted = True
        if not promoted:
            continue
        rows.append({**row.to_dict(), "quality": quality})

    trainable = pd.DataFrame(rows).sort_values(["page", "row_index"]).reset_index(drop=True)
    trainable.to_csv(args.out_dir / "manifest_trainable.tsv", sep="\t", index=False)

    eval_rows = []
    for idx, row in trainable.iterrows():
        page = int(row["page"])
        row_index = int(row["row_index"])
        eval_rows.append(
            {
                "sample_id": f"p{page:03d}_{row_index:04d}_{idx:05d}",
                "variant": "original_export",
                "image": relative_image_path(args.out_dir, args.dataset_dir, row["image"]),
                "label": row["ipa_digits"],
                "page": page,
                "row_index": row_index,
                "hanzi": row.get("hanzi", ""),
                "source_split": split_for_page(page),
                "quality": row.get("quality", ""),
                "tone_position": row.get("tone_position", ""),
                "wupin": row.get("wupin", ""),
                "ipa": row.get("ipa", ""),
            }
        )
    pd.DataFrame(eval_rows).to_csv(args.out_dir / "eval_manifest.tsv", sep="\t", index=False)

    print(f"base matched+weak non-both: {len(flagged[flagged['image'].ne('') & flagged['quality'].isin(['matched','weak_match']) & ~flagged['tone_position'].eq('both_upper_lower')])}")
    print(f"promoted low: {len(promoted_keys)}")
    print(f"trainable total: {len(trainable)}")
    print(trainable["quality"].value_counts().to_string())
    print(f"out: {args.out_dir}")


if __name__ == "__main__":
    main()
