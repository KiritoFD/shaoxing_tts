"""Build candidate phonetic spans for training a segmentation quality model.

The model trained on this dataset does not read IPA labels. It learns to choose
which horizontal span in a row crop looks like the phonetic transcription field.
Positive labels come from high-confidence geometry on clean rows; negatives
come from competing spans, bracket/context rows, and low-quality rows.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image

from build_syllable_ocr_crops_v3 import (
    DEFAULT_CLEAN_FLAGS,
    DEFAULT_ROW_SOURCE,
    DEFAULT_STRUCTURED,
    components,
    dark_bounds,
    expected_width,
    group_by_large_gaps,
    ink_binary,
    score_span,
    tone_like_count,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_pdf136_clean" / "phonetic_segmenter_candidates"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build phonetic span candidate dataset.")
    parser.add_argument("--row-source", type=Path, default=DEFAULT_ROW_SOURCE)
    parser.add_argument("--row-cleaning-flags", type=Path, default=DEFAULT_CLEAN_FLAGS)
    parser.add_argument("--structured-syllables", type=Path, default=DEFAULT_STRUCTURED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--subset", choices=["all", "strict"], default="all")
    parser.add_argument("--positive-qualities", nargs="*", default=["matched", "weak_match"])
    parser.add_argument("--max-negatives-per-row", type=int, default=5)
    parser.add_argument("--limit-rows", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", keep_default_na=False)


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)


def candidate_spans(image: Image.Image, group: pd.DataFrame) -> list[dict[str, object]]:
    binary = ink_binary(image)
    height, width = binary.shape
    comps = components(binary)
    if not comps:
        return [
            {
                "span_x0": 0,
                "span_x1": width,
                "candidate_kind": "empty_fallback",
                "teacher_score": -10.0,
                "width_ratio": 1.0,
                "tone_ratio": 0.0,
                "component_ratio": 0.0,
                "cjk_like_ratio": 0.0,
                "span_fraction": 1.0,
            }
        ]

    spans: list[tuple[int, int, str]] = []
    gap_groups = group_by_large_gaps(comps, height)
    for i in range(len(gap_groups)):
        for j in range(i, len(gap_groups)):
            inside = [c for block in gap_groups[i : j + 1] for c in block]
            x0 = max(0, min(c.x for c in inside) - 4)
            x1 = min(width, max(c.x1 for c in inside) + 4)
            spans.append((x0, x1, f"gap_groups:{i}-{j}"))
    left, right, _, _ = dark_bounds(binary)
    spans.append((left, right, "all_dark_bounds"))

    expected = max(1.0, expected_width(group, height))
    tone_digits = sum(len(str(row.get("selected_tone", ""))) for _, row in group.iterrows())
    out = []
    seen: set[tuple[int, int]] = set()
    for x0, x1, kind in spans:
        x0 = max(0, min(width - 1, int(x0)))
        x1 = max(x0 + 1, min(width, int(x1)))
        if (x0, x1) in seen:
            continue
        seen.add((x0, x1))
        inside = [c for c in comps if c.x >= x0 and c.x1 <= x1]
        cjk_like = sum(1 for c in inside if c.h > 0.55 * height and c.w > 0.20 * height)
        score = score_span((x0, x1), comps, group, height)
        out.append(
            {
                "span_x0": x0,
                "span_x1": x1,
                "candidate_kind": kind,
                "teacher_score": score,
                "width_ratio": (x1 - x0) / expected,
                "tone_ratio": tone_like_count(inside, height) / max(1, tone_digits),
                "component_ratio": len(inside) / max(2, len(group) * 3),
                "cjk_like_ratio": cjk_like / max(1, len(inside)),
                "span_fraction": (x1 - x0) / max(1, width),
            }
        )
    out.sort(key=lambda row: float(row["teacher_score"]), reverse=True)
    return out


def row_is_positive_source(row: pd.Series, positive_qualities: set[str]) -> bool:
    flags = str(row.get("cleaning_flags", ""))
    quality = str(row.get("quality_row", row.get("quality", "")))
    return (
        bool(row.get("clean_all", False))
        and quality in positive_qualities
        and "bracket" not in flags
        and "low_match" not in quality
    )


def positive_candidate_ok(candidate: dict[str, object]) -> bool:
    return (
        float(candidate["teacher_score"]) >= 0.65
        and 0.35 <= float(candidate["width_ratio"]) <= 2.20
        and float(candidate["tone_ratio"]) >= 0.20
        and float(candidate["span_fraction"]) >= 0.08
    )


def split_for_row(row: pd.Series) -> str:
    split = str(row.get("split_row", row.get("split", "")))
    return "train" if split == "review" else split


def make_dataset(args: argparse.Namespace) -> pd.DataFrame:
    flags = read_tsv(args.row_cleaning_flags)
    structured = read_tsv(args.structured_syllables)
    keep_col = "clean_all" if args.subset == "all" else "clean_strict"
    cols = [
        "page",
        "source_page",
        "pdf_page",
        "row_index",
        "image",
        "quality",
        "split",
        "clean_all",
        "clean_strict",
        "cleaning_flags",
    ]
    merged = structured.merge(flags[cols], on=["page", "row_index"], how="left", suffixes=("", "_row"))
    selected = merged[merged[keep_col].fillna(False)].copy()
    image_dir = args.out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    positive_qualities = set(args.positive_qualities)
    grouped = list(selected.groupby(["page", "row_index"], sort=True))
    if args.limit_rows:
        grouped = grouped[: args.limit_rows]

    for (page, row_index), group in grouped:
        group = group.sort_values("syllable_index").reset_index(drop=True)
        image_rel = str(group.iloc[0].get("image_row", ""))
        image_path = args.row_source / image_rel
        if not image_path.exists():
            continue
        image = Image.open(image_path).convert("L")
        candidates = candidate_spans(image, group)
        if not candidates:
            continue
        can_have_positive = row_is_positive_source(group.iloc[0], positive_qualities)
        best = candidates[0]
        positive_index = 0 if can_have_positive and positive_candidate_ok(best) else -1
        negatives = [idx for idx, cand in enumerate(candidates) if idx != positive_index]
        if positive_index >= 0:
            keep_indices = [positive_index] + negatives[: args.max_negatives_per_row]
        else:
            keep_indices = negatives[: max(1, args.max_negatives_per_row)]

        for rank, cand_idx in enumerate(keep_indices):
            cand = candidates[cand_idx]
            label = int(cand_idx == positive_index)
            sample_id = f"p{int(page):03d}_{int(row_index):04d}_c{rank:02d}"
            crop_name = f"{sample_id}.png"
            x0, x1 = int(cand["span_x0"]), int(cand["span_x1"])
            image.crop((x0, 0, x1, image.height)).save(image_dir / crop_name)
            row0 = group.iloc[0]
            rows.append(
                {
                    "sample_id": sample_id,
                    "row_id": f"p{int(page):03d}_{int(row_index):04d}",
                    "image": f"images/{crop_name}",
                    "source_split": split_for_row(row0),
                    "page": int(page),
                    "source_page": row0.get("source_page", page),
                    "pdf_page": row0.get("pdf_page", ""),
                    "row_index": int(row_index),
                    "syllable_count": len(group),
                    "label": label,
                    "candidate_rank": cand_idx,
                    "candidate_kind": cand["candidate_kind"],
                    "span_x0": x0,
                    "span_x1": x1,
                    "row_width": image.width,
                    "teacher_score": cand["teacher_score"],
                    "width_ratio": cand["width_ratio"],
                    "tone_ratio": cand["tone_ratio"],
                    "component_ratio": cand["component_ratio"],
                    "cjk_like_ratio": cand["cjk_like_ratio"],
                    "span_fraction": cand["span_fraction"],
                    "quality": row0.get("quality_row", row0.get("quality", "")),
                    "cleaning_flags": row0.get("cleaning_flags", ""),
                    "positive_source": int(can_have_positive),
                    "wupin": "".join(str(v) for v in group["wupin_base"]),
                }
            )
    return pd.DataFrame(rows)


def write_contact_sheet(df: pd.DataFrame, out_dir: Path, seed: int = 17, n: int = 60) -> None:
    if df.empty:
        return
    sample = df.sample(min(n, len(df)), random_state=seed)
    thumb_w, thumb_h, label_h, cols = 180, 80, 30, 6
    sheet = Image.new("RGB", (cols * thumb_w, math.ceil(len(sample) / cols) * (thumb_h + label_h)), "white")
    from PIL import ImageDraw

    draw = ImageDraw.Draw(sheet)
    for idx, row in enumerate(sample.itertuples()):
        image = Image.open(out_dir / row.image).convert("RGB")
        image.thumbnail((thumb_w - 8, thumb_h - 8), Image.Resampling.LANCZOS)
        x = (idx % cols) * thumb_w
        y = (idx // cols) * (thumb_h + label_h)
        sheet.paste(image, (x + 4, y + 4))
        draw.text((x + 4, y + thumb_h), f"{row.sample_id} y={row.label}", fill=(0, 0, 0))
    sheet.save(out_dir / "contact_sheet_candidates.png")


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and args.overwrite:
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = make_dataset(args)
    write_tsv(df, args.out_dir / "candidate_manifest.tsv")
    write_contact_sheet(df, args.out_dir)
    lines = [f"rows: {len(df)}"]
    if len(df):
        lines.append(f"positives: {int(df['label'].sum())}")
        lines.append(f"positive_rows: {df[df['label'].eq(1)]['row_id'].nunique()}")
        lines.append("split x label:")
        for key, value in df.groupby(["source_split", "label"]).size().items():
            lines.append(f"{key[0]}\t{key[1]}\t{value}")
    (args.out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"wrote {args.out_dir / 'candidate_manifest.tsv'}")


if __name__ == "__main__":
    main()
