"""Build structured per-syllable tone labels from trusted Wu-pinyin.

The hand label gives the tone value that should be selected for downstream
Shaoxing output. To make rows with one visual tone and rows with upper+lower
visual tones share one schema, each syllable gets two nullable slots:

  upper_tone, lower_tone

Policy used here:
- upper_only/no_detected_tone_digits: selected tone is stored as upper_tone,
  lower_tone is null.
- lower_only: selected tone is stored as lower_tone.
- both_upper_lower: selected tone is stored as lower_tone, because the
  annotation rule prefers the lower-right number when it exists. The unselected
  upper value is unknown until we run a physical tone OCR pass.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from wupin_ipa_convert import (
    DEFAULT_MAP,
    SYLLABLE_RE,
    load_mapping,
    normalize,
    split_initial_final,
    wupin_syllable_to_ipa,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_ipa_no_both_tone" / "manifest_with_tone_flags.tsv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_structured_tone_labels"
DEFAULT_DATASET_DIR = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_ipa_matched_skip3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build structured tone labels.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    return parser.parse_args()


def ipa_base_for_wupin_base(base: str, mapping: dict) -> tuple[str, str, str]:
    whole = mapping["whole_syllables"].get(base)
    if whole is not None:
        return whole, "", base
    initial, final = split_initial_final(base, mapping)
    ipa_initial = mapping["initials"].get(initial, mapping["zero_initial"] if initial == "" else None)
    ipa_final = mapping["finals"].get(final)
    if ipa_initial is None or ipa_final is None:
        return "", initial, final
    return ipa_initial + ipa_final, initial, final


def tone_slots(selected_tone: str, tone_position: str) -> tuple[str | None, str | None, str]:
    if tone_position == "both_upper_lower":
        return None, selected_tone, "selected_as_lower__upper_unlabeled"
    if tone_position == "lower_only":
        return None, selected_tone, "selected_as_lower"
    if tone_position in {"upper_only", "no_detected_tone_digits"}:
        return selected_tone, None, "single_tone__lower_null"
    return selected_tone, None, "tone_position_unknown__lower_null"


def visual_tone_slots(image_path: Path, syllables: list[dict]) -> list[str | None]:
    """Return per-syllable slot hints: lower, upper, or None.

    The crop is split left-to-right in proportion to each syllable's expected
    visual width. If a lower-position tone component falls in a syllable span,
    that syllable is treated as lower; otherwise it defaults to upper. This is
    deliberately conservative and recorded as a heuristic in tone_policy.
    """
    if not image_path.exists() or not syllables:
        return [None for _ in syllables]
    image = Image.open(image_path).convert("L")
    arr = np.asarray(image)
    _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num, _, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)

    large_centers = []
    small_components = []
    for idx in range(1, num):
        x, y, w, h, area = [float(v) for v in stats[idx]]
        cx, cy = [float(v) for v in centroids[idx]]
        if area < 20:
            continue
        if h > 50 or area > 900:
            large_centers.append(cy)
        if 4 <= w <= 35 and 8 <= h <= 48 and 40 <= area <= 700:
            small_components.append({"x": x, "y": y, "w": w, "h": h, "cx": cx, "cy": cy})

    if large_centers:
        baseline = float(np.median(large_centers))
    else:
        dark_y = np.where(binary > 0)[0]
        baseline = float(np.median(dark_y)) if len(dark_y) else image.height / 2.0

    dark_x = np.where(binary > 0)[1]
    if len(dark_x):
        left = float(np.percentile(dark_x, 1))
        right = float(np.percentile(dark_x, 99))
    else:
        left, right = 0.0, float(image.width)
    if right <= left:
        right = float(image.width)

    weights = [max(1.0, len(str(s["ipa_base"])) + len(str(s["selected_tone"])) * 0.7) for s in syllables]
    total = sum(weights)
    spans = []
    cur = left
    for weight in weights:
        nxt = cur + (right - left) * weight / total
        spans.append((cur, nxt))
        cur = nxt

    lower_components = [c for c in small_components if c["y"] > baseline - 5]
    hints = []
    for start, end in spans:
        pad = max(10.0, (end - start) * 0.15)
        has_lower = any(start - pad <= c["cx"] <= end + pad for c in lower_components)
        hints.append("lower" if has_lower else "upper")
    return hints


def syllables_for_wupin(
    wupin: str,
    tone_position: str,
    mapping: dict,
    image_path: Path | None = None,
) -> tuple[list[dict], str]:
    text = normalize(wupin)
    parts = SYLLABLE_RE.findall(text)
    consumed = "".join(base + tone for base, tone in parts)
    status = "ok" if consumed == text else "parse_error"
    syllables = []
    for idx, (base, tone) in enumerate(parts):
        ipa_base, initial, final = ipa_base_for_wupin_base(base, mapping)
        ipa_selected, error = wupin_syllable_to_ipa(base, tone, mapping, "digits")
        syllables.append(
            {
                "syllable_index": idx,
                "wupin_base": base,
                "wupin_initial": initial,
                "wupin_final": final,
                "ipa_base": ipa_base,
                "selected_tone": tone,
                "upper_tone": None,
                "lower_tone": None,
                "tone_policy": "",
                "ipa_selected": ipa_selected,
                "conversion_error": error,
            }
        )
    visual_hints = visual_tone_slots(image_path, syllables) if image_path else [None for _ in syllables]
    for syllable, hint in zip(syllables, visual_hints):
        tone = syllable["selected_tone"]
        if tone_position == "both_upper_lower" and hint in {"upper", "lower"}:
            if hint == "lower":
                syllable["upper_tone"] = None
                syllable["lower_tone"] = tone
                syllable["tone_policy"] = "visual_segment_selected_as_lower"
            else:
                syllable["upper_tone"] = tone
                syllable["lower_tone"] = None
                syllable["tone_policy"] = "visual_segment_single_tone__lower_null"
        else:
            upper_tone, lower_tone, tone_policy = tone_slots(tone, tone_position)
            syllable["upper_tone"] = upper_tone
            syllable["lower_tone"] = lower_tone
            syllable["tone_policy"] = tone_policy
    return syllables, status


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    mapping = load_mapping(args.mapping)
    df = pd.read_csv(args.manifest, sep="\t", keep_default_na=False)

    row_records = []
    syllable_records = []
    for _, row in df.iterrows():
        wupin = row.get("wupin", "")
        tone_position = row.get("tone_position", "")
        image = row.get("image", "")
        image_path = args.dataset_dir / image if image else None
        syllables, parse_status = syllables_for_wupin(wupin, tone_position, mapping, image_path)
        row_id = f"p{int(row['page']):03d}_{int(row['row_index']):04d}"
        structured = {
            "row_id": row_id,
            "page": int(row["page"]),
            "row_index": int(row["row_index"]),
            "hanzi": row.get("hanzi", ""),
            "wupin": wupin,
            "ipa_digits": row.get("ipa_digits", ""),
            "image": row.get("image", ""),
            "quality": row.get("quality", ""),
            "tone_position": tone_position,
            "parse_status": parse_status,
            "syllables": syllables,
        }
        row_records.append(structured)
        for syllable in syllables:
            syllable_records.append(
                {
                    "row_id": row_id,
                    "page": structured["page"],
                    "row_index": structured["row_index"],
                    "hanzi": structured["hanzi"],
                    "wupin": wupin,
                    "image": structured["image"],
                    "quality": structured["quality"],
                    "tone_position": tone_position,
                    **syllable,
                }
            )

    jsonl_path = args.out_dir / "structured_tone_labels.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in row_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    tsv_path = args.out_dir / "structured_tone_syllables.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "row_id",
            "page",
            "row_index",
            "hanzi",
            "wupin",
            "image",
            "quality",
            "tone_position",
            "syllable_index",
            "wupin_base",
            "wupin_initial",
            "wupin_final",
            "ipa_base",
            "selected_tone",
            "upper_tone",
            "lower_tone",
            "tone_policy",
            "ipa_selected",
            "conversion_error",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(syllable_records)

    summary = pd.DataFrame(syllable_records)
    summary_path = args.out_dir / "summary.txt"
    lines = [
        f"rows: {len(row_records)}",
        f"syllables: {len(syllable_records)}",
        "",
        "tone_policy counts:",
        summary["tone_policy"].value_counts(dropna=False).to_string() if not summary.empty else "",
        "",
        "tone_position x tone_policy:",
        summary.groupby(["tone_position", "tone_policy"]).size().to_string() if not summary.empty else "",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"wrote {jsonl_path}")
    print(f"wrote {tsv_path}")


if __name__ == "__main__":
    main()
