"""Build a paired OCR evaluation set for image enhancement experiments.

Each selected labeled crop is exported in multiple variants:
- original: direct PDF render crop
- original_otsu: direct PDF render crop, then Otsu binarization
- superres_gray: denoise + upscale + sharpen, grayscale
- superres_otsu: same enhancement, then Otsu binarization
- superres_adaptive: same enhancement, then adaptive binarization
- superres_sauvola: same enhancement, then Sauvola local binarization
- preserve_sauvola: Preserve Details style resampling, then Sauvola
- threshold_T: simple fixed threshold, e.g. threshold_160
- denoise_threshold_T: light denoise, then fixed threshold

OCR systems should run on every image and write predictions for scoring.
"""

from __future__ import annotations

import argparse
import ast
import csv
import random
from pathlib import Path

import cv2 as cv
import fitz
import numpy as np
import pandas as pd

from enhance_pdf_superres import enhance


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PDF = PROJECT_ROOT / "ipa_ocr_work" / "data" / "shaoxing_123-351.pdf"
DEFAULT_MANIFEST = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_wupin" / "manifest.tsv"
DEFAULT_OUT = PROJECT_ROOT / "ipa_ocr_work" / "eval" / "enhancement_ab"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paired enhancement OCR eval crops.")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--page-offset", type=int, default=123)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--label-column", default="wupin", help="Manifest column to use as OCR label.")
    parser.add_argument("--pages", nargs="+", type=int, default=None, help="Only export these book page numbers.")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Only generate selected variants. Defaults to all variants.",
    )
    parser.add_argument(
        "--threshold-values",
        nargs="+",
        type=int,
        default=[120, 140, 160, 180, 200],
        help="Fixed binary thresholds used for threshold_T and denoise_threshold_T variants.",
    )
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--original-dpi", type=int, default=400)
    parser.add_argument("--superres-render-dpi", type=int, default=300)
    parser.add_argument("--superres-upscale", type=float, default=1.5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def render_clip(page: fitz.Page, bbox: tuple[float, float, float, float], dpi: int) -> np.ndarray:
    scale = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=fitz.Rect(bbox), alpha=False)
    rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv.imwrite(str(path), image):
        raise OSError(f"failed to write {path}")


def fixed_threshold(gray: np.ndarray, threshold: int) -> np.ndarray:
    _, binary = cv.threshold(gray, threshold, 255, cv.THRESH_BINARY)
    return binary


def light_denoise(gray: np.ndarray) -> np.ndarray:
    return cv.fastNlMeansDenoising(gray, h=3.0, templateWindowSize=7, searchWindowSize=21)


def load_rows(
    manifest_path: Path,
    splits: list[str],
    max_samples: int,
    seed: int,
    pages: list[int] | None,
    label_column: str,
) -> pd.DataFrame:
    df = pd.read_csv(manifest_path, sep="\t")
    if label_column not in df.columns:
        raise ValueError(f"manifest has no label column {label_column!r}; columns={list(df.columns)}")
    df = df[df["split"].isin(splits)].copy()
    if pages:
        df = df[df["page"].isin(pages)].copy()
    df = df[df["image"].notna() & df["crop_bbox"].notna() & df[label_column].notna()]
    df = df[df["quality"].eq("auto")]
    df = df.sort_values(["page", "row_index"]).reset_index(drop=True)
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=seed).sort_values(["page", "row_index"])
    return df.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.manifest, args.splits, args.max_samples, args.seed, args.pages, args.label_column)
    doc = fitz.open(args.pdf)
    eval_rows = []
    selected_variants = set(args.variants) if args.variants else None

    for sample_idx, row in rows.iterrows():
        page_num = int(row["page"])
        pdf_page = doc[page_num - args.page_offset]
        bbox = ast.literal_eval(row["crop_bbox"])
        sample_id = f"p{page_num:03d}_{int(row['row_index']):04d}"

        threshold_variants = {f"threshold_{value}" for value in args.threshold_values}
        denoise_threshold_variants = {f"denoise_threshold_{value}" for value in args.threshold_values}
        available_variants = {
            "original",
            "original_otsu",
            "superres_gray",
            "superres_otsu",
            "superres_adaptive",
            "superres_sauvola",
            "preserve_sauvola",
        } | threshold_variants | denoise_threshold_variants
        if selected_variants:
            unknown = selected_variants.difference(available_variants)
            if unknown:
                raise ValueError(f"unknown variants: {sorted(unknown)}")
            variant_names = [name for name in available_variants if name in selected_variants]
        else:
            variant_names = [
                "original",
                "original_otsu",
                "superres_gray",
                "superres_otsu",
                "superres_adaptive",
                "superres_sauvola",
                "preserve_sauvola",
            ]

        variants = {}
        needs_original = (
            "original" in variant_names
            or "original_otsu" in variant_names
            or any(name.startswith("threshold_") for name in variant_names)
            or any(name.startswith("denoise_threshold_") for name in variant_names)
        )
        if needs_original:
            original = render_clip(pdf_page, bbox, args.original_dpi)
            if "original" in variant_names:
                variants["original"] = original
            if "original_otsu" in variant_names:
                _, variants["original_otsu"] = cv.threshold(original, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            for threshold in args.threshold_values:
                variant = f"threshold_{threshold}"
                if variant in variant_names:
                    variants[variant] = fixed_threshold(original, threshold)
                variant = f"denoise_threshold_{threshold}"
                if variant in variant_names:
                    variants[variant] = fixed_threshold(light_denoise(original), threshold)

        superres_names = [name for name in variant_names if name.startswith("superres_") or name == "preserve_sauvola"]
        if superres_names:
            base_rgb = cv.cvtColor(render_clip(pdf_page, bbox, args.superres_render_dpi), cv.COLOR_GRAY2RGB)
            if "superres_gray" in superres_names:
                variants["superres_gray"] = enhance(
                    base_rgb,
                    upscale=args.superres_upscale,
                    denoise=5.0,
                    sharpen=0.55,
                    threshold="none",
                )
            if "superres_otsu" in superres_names:
                variants["superres_otsu"] = enhance(
                    base_rgb,
                    upscale=args.superres_upscale,
                    denoise=5.0,
                    sharpen=0.55,
                    threshold="otsu",
                )
            if "superres_adaptive" in superres_names:
                variants["superres_adaptive"] = enhance(
                    base_rgb,
                    upscale=args.superres_upscale,
                    denoise=5.0,
                    sharpen=0.55,
                    threshold="adaptive",
                )
            if "superres_sauvola" in superres_names:
                variants["superres_sauvola"] = enhance(
                    base_rgb,
                    upscale=args.superres_upscale,
                    denoise=5.0,
                    sharpen=0.55,
                    threshold="sauvola",
                )
            if "preserve_sauvola" in superres_names:
                variants["preserve_sauvola"] = enhance(
                    base_rgb,
                    upscale=args.superres_upscale,
                    denoise=5.0,
                    sharpen=0.0,
                    threshold="sauvola",
                    preserve_details=True,
                )
        for variant, image in variants.items():
            rel = Path("images") / variant / f"{sample_id}.png"
            write_png(args.out_dir / rel, image)
            eval_rows.append(
                {
                    "sample_id": sample_id,
                    "variant": variant,
                    "image": rel.as_posix(),
                    "label": row[args.label_column],
                    "page": page_num,
                    "row_index": int(row["row_index"]),
                    "hanzi": row.get("hanzi", ""),
                    "source_split": row["split"],
                }
            )

    manifest_out = args.out_dir / "eval_manifest.tsv"
    with manifest_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "variant",
                "image",
                "label",
                "page",
                "row_index",
                "hanzi",
                "source_split",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(eval_rows)

    labels_out = args.out_dir / "labels.tsv"
    with labels_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "variant", "label"], delimiter="\t")
        writer.writeheader()
        for row in eval_rows:
            writer.writerow({k: row[k] for k in ["sample_id", "variant", "label"]})

    print(f"samples: {len(rows)}")
    print(f"eval images: {len(eval_rows)}")
    print(f"wrote {manifest_out}")


if __name__ == "__main__":
    main()
