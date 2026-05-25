"""Build clean OCR manifests for the trusted first 136 PDF pages.

The annotation spreadsheet uses the printed/source page number, while the
annotator's "136 pages" refers to PDF page order. In the current export,
printed page 123 is PDF page 1, so trusted PDF page = source page - 122.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image

from build_clustered_tone_detector_manifest import component_boxes
from wupin_ipa_convert import (
    DEFAULT_MAP,
    canonicalize_wupin_base,
    canonicalize_wupin_label,
    load_mapping,
    wupin_syllable_to_ipa,
    wupin_to_ipa,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROW_SOURCE = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_ipa_matched_skip3"
DEFAULT_STRUCTURED = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_structured_tone_labels" / "structured_tone_syllables.tsv"
DEFAULT_OUT = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_pdf136_clean"
BRACKET_RE = re.compile(r"[\[\]［］【】]")
SYLLABLE_RE = re.compile(r"([a-z]+)([0-9]+)")
BRACKET_RE = re.compile(r"[\[\]【】［］]")
BRACKET_RE = re.compile(r"[\[\]【】［］]")
BRACKET_RE = re.compile(r"[\[\]\u3010\u3011\uff3b\uff3d]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build clean OCR datasets for trusted PDF pages.")
    parser.add_argument("--row-source", type=Path, default=DEFAULT_ROW_SOURCE)
    parser.add_argument("--structured-syllables", type=Path, default=DEFAULT_STRUCTURED)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-pdf-page", type=int, default=136)
    parser.add_argument(
        "--page-offset",
        type=int,
        default=122,
        help="source page minus PDF page. Current export: source page 123 == PDF page 1.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", keep_default_na=False)


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)


def normalize_wupin(text: object) -> str:
    return re.sub(r"[^a-z0-9]", "", canonicalize_wupin_label(text))


def parse_wupin_ok(text: str) -> bool:
    parts = SYLLABLE_RE.findall(text)
    return bool(parts) and "".join(base + tone for base, tone in parts) == text


def flag_rows(row_manifest: pd.DataFrame, max_pdf_page: int, page_offset: int, row_source: Path, mapping: dict) -> pd.DataFrame:
    rows = row_manifest.copy()
    rows["source_page"] = rows["page"].astype(int)
    rows["pdf_page"] = rows["source_page"] - int(page_offset)
    rows["wupin_norm"] = rows["wupin"].map(normalize_wupin)
    rows["is_trusted_page"] = rows["pdf_page"].between(1, max_pdf_page)
    rows["has_wupin"] = rows["wupin_norm"].ne("")
    rows["has_image"] = rows.get("image", "").astype(str).str.strip().ne("")
    rows["image_exists"] = rows.get("image", "").astype(str).map(lambda value: bool(value) and (row_source / value).exists())
    rows["parse_ok"] = rows["wupin_norm"].map(parse_wupin_ok)
    conversions = rows["wupin_norm"].map(lambda value: wupin_to_ipa(value, mapping, "digits"))
    rows["ipa_digits_rule"] = conversions.map(lambda item: item[0])
    rows["ipa_conversion_status"] = conversions.map(lambda item: "ok" if not item[1] else ";".join(item[1]))
    rows["has_bracket_context"] = rows.get("pdf_text", "").astype(str).map(lambda text: bool(BRACKET_RE.search(text)))
    rows["candidate_exact"] = rows.get("candidate_headword", "").astype(str).eq(rows["hanzi"].astype(str))
    rows["is_matched"] = rows.get("quality", "").astype(str).eq("matched")
    rows["clean_all"] = rows["is_trusted_page"] & rows["has_wupin"] & rows["parse_ok"] & rows["ipa_conversion_status"].eq("ok") & rows["has_image"] & rows["image_exists"]
    rows["clean_strict"] = rows["clean_all"] & rows["is_matched"] & ~rows["has_bracket_context"]

    reasons = []
    for _, row in rows.iterrows():
        issue = []
        if not row["is_trusted_page"]:
            issue.append("outside_trusted_pdf_pages")
        if not row["has_wupin"]:
            issue.append("empty_wupin")
        if not row["has_image"]:
            issue.append("empty_image")
        elif not row["image_exists"]:
            issue.append("missing_image")
        if row["has_wupin"] and not row["parse_ok"]:
            issue.append("wupin_parse_error")
        if row["ipa_conversion_status"] != "ok":
            issue.append(f"ipa_conversion:{row['ipa_conversion_status']}")
        if row["has_bracket_context"]:
            issue.append("bracket_context_review")
        if not row["is_matched"]:
            issue.append(f"non_matched_quality:{row.get('quality', '')}")
        if not row["candidate_exact"]:
            issue.append("candidate_headword_differs")
        reasons.append(";".join(issue) if issue else "ok")
    rows["cleaning_flags"] = reasons
    return rows


def relative_image_to_clean(row: pd.Series, source_root: Path, out_root: Path) -> str:
    src = source_root / row["image"]
    try:
        return src.resolve().relative_to(out_root.resolve(), walk_up=True).as_posix()
    except TypeError:
        return src.resolve().as_posix()


def make_row_manifest(flags: pd.DataFrame, source_root: Path, out_dir: Path, subset: str) -> pd.DataFrame:
    merged = flags.copy()
    if subset == "all":
        out = merged[merged["clean_all"].fillna(False)].copy()
    elif subset == "strict":
        out = merged[merged["clean_strict"].fillna(False)].copy()
    else:
        raise ValueError(subset)
    out["image"] = out.apply(lambda row: relative_image_to_clean(row, source_root, out_dir), axis=1)
    out["original_source_split"] = out.get("split", "")
    out = out.rename(columns={"split": "source_split", "ipa_digits": "legacy_ipa_digits", "ipa": "legacy_ipa"})
    out["label"] = out["ipa_digits_rule"]
    out["ipa"] = out["ipa_digits_rule"]
    out["source_split"] = out["source_split"].replace({"review": "train"})
    keep = [
        "sample_id",
        "variant",
        "image",
        "label",
        "page",
        "source_page",
        "pdf_page",
        "row_index",
        "hanzi",
        "source_split",
        "original_source_split",
        "quality",
        "wupin",
        "ipa",
        "legacy_ipa",
        "legacy_ipa_digits",
        "ipa_conversion_status",
        "clean_all",
        "clean_strict",
        "cleaning_flags",
        "has_image",
        "image_exists",
        "has_bracket_context",
        "candidate_headword",
        "pdf_text",
    ]
    if "sample_id" not in out.columns:
        out["sample_id"] = [f"pdf{int(row.pdf_page):03d}_p{int(row.page):03d}_{int(row.row_index):04d}" for row in out.itertuples()]
    out["variant"] = "original_export"
    out = out[[col for col in keep if col in out.columns]]
    return out.reset_index(drop=True)


def make_syllable_manifest(structured: pd.DataFrame, flags: pd.DataFrame, row_source: Path, out_dir: Path, subset: str, mapping: dict) -> pd.DataFrame:
    cols = [
        "page",
        "source_page",
        "pdf_page",
        "row_index",
        "image",
        "clean_all",
        "clean_strict",
        "cleaning_flags",
        "has_bracket_context",
        "candidate_headword",
        "pdf_text",
        "quality",
        "split",
    ]
    meta = flags[cols].copy()
    merged = structured.merge(meta, on=["page", "row_index"], how="left", suffixes=("", "_row"))
    keep_col = "clean_all" if subset == "all" else "clean_strict"
    selected = merged[merged[keep_col].fillna(False)].copy()

    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    out_rows = []
    for (page, row_index), group in selected.groupby(["page", "row_index"], sort=True):
        group = group.sort_values("syllable_index")
        row_image = str(group.iloc[0]["image_row"])
        image_path = row_source / row_image
        if not image_path.exists():
            continue
        image = Image.open(image_path).convert("L")
        boxes = component_boxes(image, len(group))
        for (_, row), box in zip(group.iterrows(), boxes):
            wupin_base = canonicalize_wupin_base(row.get("wupin_base", ""))
            tone = str(row.get("selected_tone", ""))
            ipa_label, ipa_error = wupin_syllable_to_ipa(wupin_base, tone, mapping, "digits")
            sample_id = f"p{int(page):03d}_{int(row_index):04d}_s{int(row['syllable_index']):02d}"
            crop_name = f"{sample_id}.png"
            image.crop(box).save(image_dir / crop_name)
            out_rows.append(
                {
                    "sample_id": sample_id,
                    "variant": "syllable_crop",
                    "image": f"images/{crop_name}",
                    "label": ipa_label if not ipa_error else f"{row.get('ipa_base', '')}{row.get('selected_tone', '')}",
                    "source_split": "train" if row.get("split_row", row.get("split", "")) == "review" else row.get("split_row", row.get("split", "")),
                    "original_source_split": row.get("split_row", row.get("split", "")),
                    "page": page,
                    "source_page": row.get("source_page", page),
                    "pdf_page": row.get("pdf_page", ""),
                    "row_index": row_index,
                    "syllable_index": row["syllable_index"],
                    "tone_policy": row.get("tone_policy", ""),
                    "wupin_base": wupin_base,
                    "ipa_base": ipa_label[: -len(tone)] if not ipa_error and tone else ipa_label,
                    "legacy_wupin_base": row.get("wupin_base", ""),
                    "legacy_ipa_base": row.get("ipa_base", ""),
                    "selected_tone": tone,
                    "ipa_conversion_status": "ok" if not ipa_error else ipa_error,
                    "cleaning_flags": row.get("cleaning_flags", ""),
                    "quality": row.get("quality_row", row.get("quality", "")),
                    "crop_bbox": ",".join(str(v) for v in box),
                }
            )
    return pd.DataFrame(out_rows)


def write_summary(flags: pd.DataFrame, manifests: dict[str, pd.DataFrame], out_dir: Path) -> None:
    lines = [
        "# trusted PDF-page clean dataset summary",
        "",
        f"row_source_rows_total\t{len(flags)}",
        f"trusted_pdf_page_min\t{int(flags.loc[flags['is_trusted_page'], 'pdf_page'].min()) if flags['is_trusted_page'].any() else ''}",
        f"trusted_pdf_page_max\t{int(flags.loc[flags['is_trusted_page'], 'pdf_page'].max()) if flags['is_trusted_page'].any() else ''}",
        f"trusted_source_page_min\t{int(flags.loc[flags['is_trusted_page'], 'source_page'].min()) if flags['is_trusted_page'].any() else ''}",
        f"trusted_source_page_max\t{int(flags.loc[flags['is_trusted_page'], 'source_page'].max()) if flags['is_trusted_page'].any() else ''}",
        f"trusted_page_rows\t{int(flags['is_trusted_page'].sum())}",
        f"clean_all_rows\t{int(flags['clean_all'].sum())}",
        f"clean_strict_rows\t{int(flags['clean_strict'].sum())}",
        f"bracket_context_trusted_rows\t{int((flags['is_trusted_page'] & flags['has_bracket_context']).sum())}",
        "",
        "manifest\trows\tunique_pages\tunique_labels",
    ]
    for name, df in manifests.items():
        label_col = "label" if "label" in df.columns else "ipa_digits"
        lines.append(f"{name}\t{len(df)}\t{df['page'].nunique() if len(df) else 0}\t{df[label_col].nunique() if len(df) else 0}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and args.overwrite:
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_tsv(args.row_source / "manifest.tsv")
    structured = read_tsv(args.structured_syllables)
    mapping = load_mapping(args.mapping)
    flags = flag_rows(rows, args.max_pdf_page, args.page_offset, args.row_source, mapping)

    write_tsv(flags, args.out_dir / "row_cleaning_flags.tsv")
    review_cols = [
        "split",
        "page",
        "source_page",
        "pdf_page",
        "row_index",
        "hanzi",
        "candidate_headword",
        "wupin",
        "quality",
        "match_score",
        "cleaning_flags",
        "ipa_conversion_status",
        "has_image",
        "image_exists",
        "pdf_text",
        "image",
    ]
    review = flags[flags["cleaning_flags"].ne("ok") & flags["is_trusted_page"]].copy()
    write_tsv(review[[col for col in review_cols if col in review.columns]], args.out_dir / "needs_review_pdf136.tsv")
    excluded = flags[~flags["clean_all"]].copy()
    write_tsv(excluded[[col for col in review_cols if col in excluded.columns]], args.out_dir / "excluded_rows.tsv")

    manifests = {}
    for subset in ["all", "strict"]:
        row_out = args.out_dir / f"ocr_selected_{subset}"
        syll_out = args.out_dir / f"syllable_ocr_{subset}"
        row_manifest = make_row_manifest(flags, args.row_source, row_out, subset)
        syll_manifest = make_syllable_manifest(structured, flags, args.row_source, syll_out, subset, mapping)
        write_tsv(row_manifest, row_out / "eval_manifest.tsv")
        write_tsv(syll_manifest, syll_out / "eval_manifest.tsv")
        manifests[f"ocr_selected_{subset}"] = row_manifest
        manifests[f"syllable_ocr_{subset}"] = syll_manifest
        (row_out / "summary.txt").write_text(f"rows: {len(row_manifest)}\n", encoding="utf-8")
        (syll_out / "summary.txt").write_text(f"rows: {len(syll_manifest)}\n", encoding="utf-8")

    write_summary(flags, manifests, args.out_dir)
    print((args.out_dir / "summary.md").read_text(encoding="utf-8"))
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
