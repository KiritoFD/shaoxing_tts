"""Build syllable OCR crops with explicit phonetic-region isolation.

The tone detector crops are good enough to locate tone position, but they are
not clean OCR crops: a row may include Chinese gloss/context before or after
the phonetic text. This builder first selects the horizontal span that best
matches the expected phonetic syllable sequence, then splits that span into
syllable crops.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from wupin_ipa_convert import DEFAULT_MAP, canonicalize_wupin_base, load_mapping, wupin_syllable_to_ipa


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROW_SOURCE = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_ipa_matched_skip3"
DEFAULT_CLEAN_FLAGS = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_pdf136_clean" / "row_cleaning_flags.tsv"
DEFAULT_STRUCTURED = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_structured_tone_labels" / "structured_tone_syllables.tsv"
DEFAULT_OUT = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_pdf136_clean" / "syllable_ocr_v3"


@dataclass(frozen=True)
class Component:
    x: int
    y: int
    w: int
    h: int
    area: int

    @property
    def x1(self) -> int:
        return self.x + self.w


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build syllable OCR crops with phonetic-region isolation.")
    parser.add_argument("--row-source", type=Path, default=DEFAULT_ROW_SOURCE)
    parser.add_argument("--row-cleaning-flags", type=Path, default=DEFAULT_CLEAN_FLAGS)
    parser.add_argument("--structured-syllables", type=Path, default=DEFAULT_STRUCTURED)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--subset", choices=["all", "strict"], default="all")
    parser.add_argument("--exclude-flag-substrings", nargs="*", default=["bracket"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--margin-x", type=int, default=5)
    parser.add_argument("--min-width", type=int, default=12)
    parser.add_argument("--debug-contact-sheet", action="store_true")
    return parser.parse_args()


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", keep_default_na=False)


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)


def split_for_row(row: pd.Series) -> str:
    split = str(row.get("split_row", row.get("split", "")))
    return "train" if split == "review" else split


def ink_binary(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image.convert("L"))
    if arr.size == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    padded = np.pad(arr.astype(np.float32), 1, mode="edge")
    blur = (
        padded[:-2, :-2]
        + padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + padded[1:-1, :-2]
        + padded[1:-1, 1:-1]
        + padded[1:-1, 2:]
        + padded[2:, :-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    ) / 9.0
    threshold = otsu_threshold(blur.astype(np.uint8))
    return np.where(blur < threshold, 255, 0).astype(np.uint8)


def otsu_threshold(arr: np.ndarray) -> int:
    hist = np.bincount(arr.ravel(), minlength=256).astype(np.float64)
    total = arr.size
    sum_total = float(np.dot(np.arange(256), hist))
    sum_bg = 0.0
    weight_bg = 0.0
    max_between = -1.0
    threshold = 128
    for value in range(256):
        weight_bg += hist[value]
        if weight_bg <= 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg <= 0:
            break
        sum_bg += value * hist[value]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if between > max_between:
            max_between = between
            threshold = value
    return threshold


def components(binary: np.ndarray) -> list[Component]:
    labels, stats = connected_components_with_stats(binary)
    comps: list[Component] = []
    for x, y, w, h, area in stats:
        if area < 10 or h < 3 or w < 2:
            continue
        comps.append(Component(x, y, w, h, area))
    return sorted(comps, key=lambda c: (c.x, c.y))


def connected_components_with_stats(binary: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int, int, int]]]:
    mask = binary > 0
    try:
        from scipy import ndimage  # type: ignore

        labels, num = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
        stats = []
        objects = ndimage.find_objects(labels)
        for idx, slc in enumerate(objects, start=1):
            if slc is None:
                continue
            ys, xs = slc
            area = int((labels[slc] == idx).sum())
            stats.append((int(xs.start), int(ys.start), int(xs.stop - xs.start), int(ys.stop - ys.start), area))
        return labels, stats
    except Exception:
        pass

    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    stats: list[tuple[int, int, int, int, int]] = []
    label = 0
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or labels[y, x]:
                continue
            label += 1
            stack = [(x, y)]
            labels[y, x] = label
            min_x = max_x = x
            min_y = max_y = y
            area = 0
            while stack:
                cx, cy = stack.pop()
                area += 1
                min_x = min(min_x, cx)
                max_x = max(max_x, cx)
                min_y = min(min_y, cy)
                max_y = max(max_y, cy)
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if mask[ny, nx] and not labels[ny, nx]:
                            labels[ny, nx] = label
                            stack.append((nx, ny))
            stats.append((min_x, min_y, max_x - min_x + 1, max_y - min_y + 1, area))
    return labels, stats


def dark_bounds(binary: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(binary > 0)
    height, width = binary.shape
    if len(xs) == 0:
        return 0, width, 0, height
    left = max(0, int(np.percentile(xs, 0.5)) - 1)
    right = min(width, int(np.percentile(xs, 99.5)) + 2)
    top = max(0, int(np.percentile(ys, 0.5)) - 2)
    bottom = min(height, int(np.percentile(ys, 99.5)) + 3)
    return left, max(left + 1, right), top, max(top + 1, bottom)


def visual_weight(row: pd.Series) -> float:
    base = str(row.get("ipa_base", ""))
    tone = str(row.get("selected_tone", ""))
    return max(1.0, len(base) + len(tone) * 0.45)


def expected_width(group: pd.DataFrame, height: int) -> float:
    # The old scans are roughly 200 px tall. In clean crops one visual-weight
    # unit is usually 32-45 px. Keep the prior intentionally broad.
    return sum(visual_weight(row) for _, row in group.iterrows()) * max(28.0, height * 0.18)


def tone_like_count(comps: list[Component], height: int) -> int:
    count = 0
    for c in comps:
        small = c.h <= 0.28 * height and c.w <= 0.28 * height
        upper = c.y <= 0.45 * height
        lower = c.y + c.h >= 0.58 * height
        if small and (upper or lower):
            count += 1
    return count


def group_by_large_gaps(comps: list[Component], height: int) -> list[list[Component]]:
    if not comps:
        return []
    gap_threshold = max(24, int(height * 0.12))
    groups: list[list[Component]] = [[comps[0]]]
    current_right = comps[0].x1
    for comp in comps[1:]:
        gap = comp.x - current_right
        if gap > gap_threshold:
            groups.append([comp])
        else:
            groups[-1].append(comp)
        current_right = max(current_right, comp.x1)
    return groups


def score_span(span: tuple[int, int], comps: list[Component], group: pd.DataFrame, height: int) -> float:
    x0, x1 = span
    width = max(1, x1 - x0)
    inside = [c for c in comps if c.x >= x0 and c.x1 <= x1]
    if not inside:
        return -1e9
    expected = expected_width(group, height)
    width_score = -abs(math.log(width / max(1.0, expected)))
    tone_digits = sum(len(str(row.get("selected_tone", ""))) for _, row in group.iterrows())
    tone_score = min(1.0, tone_like_count(inside, height) / max(1, tone_digits))
    component_score = min(1.0, len(inside) / max(2, len(group) * 3))
    # Very broad, tall components are often Chinese radicals. Penalize them but
    # do not hard-delete; some IPA bases can be wide.
    cjk_like = sum(1 for c in inside if c.h > 0.55 * height and c.w > 0.20 * height)
    return 1.7 * tone_score + 0.8 * component_score + width_score - 0.45 * cjk_like


def select_phonetic_span(binary: np.ndarray, group: pd.DataFrame) -> tuple[int, int, str]:
    height, width = binary.shape
    comps = components(binary)
    if not comps:
        return (0, width, "empty_fallback")
    gap_groups = group_by_large_gaps(comps, height)
    if not gap_groups:
        left, right, _, _ = dark_bounds(binary)
        return (left, right, "dark_bounds")

    candidates: list[tuple[float, int, int, str]] = []
    for i in range(len(gap_groups)):
        for j in range(i, len(gap_groups)):
            inside = [c for block in gap_groups[i : j + 1] for c in block]
            x0 = max(0, min(c.x for c in inside) - 4)
            x1 = min(width, max(c.x1 for c in inside) + 4)
            label = f"gap_groups:{i}-{j}"
            candidates.append((score_span((x0, x1), comps, group, height), x0, x1, label))

    left, right, _, _ = dark_bounds(binary)
    candidates.append((score_span((left, right), comps, group, height) - 0.15, left, right, "all_dark_bounds"))
    best = max(candidates, key=lambda item: item[0])
    return best[1], best[2], f"{best[3]};score={best[0]:.3f}"


def smooth_projection(binary: np.ndarray) -> np.ndarray:
    projection = (binary > 0).sum(axis=0).astype(np.float32)
    if projection.size < 7:
        return projection
    kernel = np.array([1, 2, 3, 2, 1], dtype=np.float32)
    kernel /= kernel.sum()
    return np.convolve(projection, kernel, mode="same")


def find_valley(projection: np.ndarray, expected: float, left_limit: int, right_limit: int) -> int:
    width = len(projection)
    left = max(1, min(width - 2, int(left_limit)))
    right = max(left + 1, min(width - 1, int(right_limit)))
    center = int(round(expected))
    radius = max(10, int((right - left) * 0.38))
    lo = max(left, center - radius)
    hi = min(right, center + radius)
    if hi <= lo:
        lo, hi = left, right
    segment = projection[lo:hi]
    if len(segment) == 0:
        return center
    min_val = float(segment.min())
    candidates = np.where(segment <= min_val + 0.5)[0] + lo
    return int(candidates[np.argmin(np.abs(candidates - expected))])


def segment_boxes(image: Image.Image, syllables: pd.DataFrame, margin_x: int, min_width: int) -> tuple[list[tuple[int, int, int, int]], str]:
    binary_full = ink_binary(image)
    height, width = binary_full.shape
    if len(syllables) <= 0:
        return [], "no_syllables"
    span_left, span_right, span_reason = select_phonetic_span(binary_full, syllables)
    span_left = max(0, min(width - 1, span_left))
    span_right = max(span_left + 1, min(width, span_right))
    binary = binary_full[:, span_left:span_right]
    span_width = binary.shape[1]

    if len(syllables) == 1:
        return [(span_left, 0, span_right, height)], span_reason

    projection = smooth_projection(binary)
    weights = [visual_weight(row) for _, row in syllables.iterrows()]
    total = sum(weights)
    expected_cuts = []
    acc = 0.0
    for weight in weights[:-1]:
        acc += weight
        expected_cuts.append(span_width * acc / total)

    raw_cuts = []
    prev = 0
    for idx, expected in enumerate(expected_cuts):
        next_expected = expected_cuts[idx + 1] if idx + 1 < len(expected_cuts) else span_width
        left_limit = (prev + expected) / 2
        right_limit = (expected + next_expected) / 2
        cut = find_valley(projection, expected, int(left_limit), int(right_limit))
        if raw_cuts and cut <= raw_cuts[-1] + min_width:
            cut = min(span_width - 1, raw_cuts[-1] + min_width)
        raw_cuts.append(cut)
        prev = cut

    edges = [0] + raw_cuts + [span_width]
    boxes = []
    for idx in range(len(syllables)):
        seg_left = max(0, edges[idx])
        seg_right = min(span_width, edges[idx + 1])
        region = binary[:, seg_left:seg_right]
        ys, xs = np.where(region > 0)
        if len(xs):
            x0 = seg_left + int(xs.min())
            x1 = seg_left + int(xs.max()) + 1
        else:
            x0, x1 = seg_left, seg_right
        x0 = max(0, max(seg_left, x0 - margin_x))
        x1 = min(span_width, min(seg_right, x1 + margin_x))
        if x1 < x0 + min_width:
            mid = (x0 + x1) // 2
            x0 = max(seg_left, mid - min_width // 2)
            x1 = min(seg_right, x0 + min_width)
        boxes.append((int(span_left + x0), 0, int(span_left + max(x1, x0 + 1)), height))
    return boxes, span_reason


def make_manifest(args: argparse.Namespace) -> pd.DataFrame:
    flags = read_tsv(args.row_cleaning_flags)
    structured = read_tsv(args.structured_syllables)
    mapping = load_mapping(args.mapping)

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
    if args.exclude_flag_substrings:
        mask = selected["cleaning_flags"].astype(str).map(
            lambda value: not any(token and token in value for token in args.exclude_flag_substrings)
        )
        selected = selected[mask].copy()

    image_dir = args.out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    out_rows = []
    for (page, row_index), group in selected.groupby(["page", "row_index"], sort=True):
        group = group.sort_values("syllable_index").reset_index(drop=True)
        image_rel = str(group.iloc[0].get("image_row", ""))
        image_path = args.row_source / image_rel
        if not image_path.exists():
            continue
        image = Image.open(image_path).convert("L")
        boxes, span_reason = segment_boxes(image, group, args.margin_x, args.min_width)
        if len(boxes) != len(group):
            continue
        for box, (_, row) in zip(boxes, group.iterrows()):
            wupin_base = canonicalize_wupin_base(row.get("wupin_base", ""))
            tone = str(row.get("selected_tone", ""))
            ipa_label, ipa_error = wupin_syllable_to_ipa(wupin_base, tone, mapping, "digits")
            label = ipa_label if not ipa_error else f"{row.get('ipa_base', '')}{tone}"
            sample_id = f"p{int(page):03d}_{int(row_index):04d}_s{int(row['syllable_index']):02d}"
            crop_name = f"{sample_id}.png"
            image.crop(box).save(image_dir / crop_name)
            out_rows.append(
                {
                    "sample_id": sample_id,
                    "variant": "syllable_crop",
                    "image": f"images/{crop_name}",
                    "label": label,
                    "source_split": split_for_row(row),
                    "original_source_split": row.get("split_row", row.get("split", "")),
                    "page": int(page),
                    "source_page": row.get("source_page", page),
                    "pdf_page": row.get("pdf_page", ""),
                    "row_index": int(row_index),
                    "syllable_index": int(row["syllable_index"]),
                    "tone_policy": row.get("tone_policy", ""),
                    "wupin_base": wupin_base,
                    "ipa_base": label[: -len(tone)] if tone else label,
                    "legacy_wupin_base": row.get("wupin_base", ""),
                    "legacy_ipa_base": row.get("ipa_base", ""),
                    "selected_tone": tone,
                    "ipa_conversion_status": "ok" if not ipa_error else ipa_error,
                    "cleaning_flags": row.get("cleaning_flags", ""),
                    "quality": row.get("quality_row", row.get("quality", "")),
                    "crop_bbox": ",".join(str(v) for v in box),
                    "crop_width": box[2] - box[0],
                    "span_reason": span_reason,
                }
            )
    return pd.DataFrame(out_rows)


def write_contact_sheet(df: pd.DataFrame, out_dir: Path, seed: int = 13, n: int = 72) -> None:
    if df.empty:
        return
    sample = df[df["source_split"].eq("val")].sample(min(n, int(df["source_split"].eq("val").sum())), random_state=seed)
    thumb_w, thumb_h, label_h, cols = 190, 88, 34, 6
    rows = math.ceil(len(sample) / cols)
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), "white")
    from PIL import ImageDraw

    draw = ImageDraw.Draw(sheet)
    for idx, row in enumerate(sample.itertuples()):
        image = Image.open(out_dir / row.image).convert("RGB")
        image.thumbnail((thumb_w - 8, thumb_h - 8), Image.Resampling.LANCZOS)
        x = (idx % cols) * thumb_w
        y = (idx // cols) * (thumb_h + label_h)
        sheet.paste(image, (x + 4, y + 4))
        draw.text((x + 4, y + thumb_h), f"{row.sample_id} {row.label}", fill=(0, 0, 0))
    sheet.save(out_dir / "contact_sheet_val.png")


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and args.overwrite:
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = make_manifest(args)
    write_tsv(df, args.out_dir / "eval_manifest.tsv")

    lines = [f"rows: {len(df)}", f"unique_labels: {df['label'].nunique() if len(df) else 0}", "split counts:"]
    if len(df):
        for (split, quality), count in df.groupby(["source_split", "quality"]).size().items():
            lines.append(f"{split}\t{quality}\t{count}")
        lines.extend(
            [
                "",
                "crop_width_summary:",
                df["crop_width"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_string(),
                "",
                "span_reason_top:",
                df["span_reason"].value_counts().head(20).to_string(),
            ]
        )
    (args.out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if args.debug_contact_sheet:
        write_contact_sheet(df, args.out_dir)
    print("\n".join(lines))
    print(f"wrote {args.out_dir / 'eval_manifest.tsv'}")


if __name__ == "__main__":
    main()
