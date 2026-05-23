"""Render Shaoxing IPA PDF pages into OCR-friendly images.

The source PDF is scanned/embedded as faint grayscale pages. This script keeps
the original page render and writes enhanced variants for OCR experiments.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2 as cv
import fitz
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PDF = PROJECT_ROOT / "ipa_ocr_work" / "data" / "shaoxing_123-351.pdf"
DEFAULT_OUT = PROJECT_ROOT / "ipa_ocr_work" / "output" / "enhanced_pages"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render and enhance PDF pages for IPA OCR."
    )
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--start-page", type=int, default=123)
    parser.add_argument("--end-page", type=int, default=123)
    parser.add_argument(
        "--page-offset",
        type=int,
        default=123,
        help="Book page number corresponding to PDF page index 0.",
    )
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument(
        "--threshold",
        choices=("none", "otsu", "adaptive"),
        default="adaptive",
    )
    parser.add_argument("--save-original", action="store_true")
    parser.add_argument(
        "--write-pdf",
        action="store_true",
        help="Also write a multi-page black/white PDF for the processed range.",
    )
    return parser.parse_args()


def render_page(page: fitz.Page, dpi: int) -> np.ndarray:
    scale = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    )


def enhance_for_ocr(rgb: np.ndarray, threshold: str) -> np.ndarray:
    gray = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)

    # Lift faint strokes without making tone marks bleed into neighboring text.
    denoised = cv.fastNlMeansDenoising(gray, h=7, templateWindowSize=7, searchWindowSize=21)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16)).apply(denoised)
    sharpened = cv.addWeighted(clahe, 1.45, cv.GaussianBlur(clahe, (0, 0), 1.0), -0.45, 0)

    if threshold == "none":
        return sharpened
    if threshold == "otsu":
        _, binary = cv.threshold(sharpened, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        return binary

    return cv.adaptiveThreshold(
        sharpened,
        255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY,
        35,
        11,
    )


def save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.ndim == 3:
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    ok = cv.imwrite(str(path), image)
    if not ok:
        raise OSError(f"Failed to write {path}")


def to_pil_page(image: np.ndarray, dpi: int) -> Image.Image:
    if image.ndim == 3:
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    pil = Image.fromarray(image).convert("1")
    pil.info["dpi"] = (dpi, dpi)
    return pil


def main() -> None:
    args = parse_args()
    if args.end_page < args.start_page:
        raise ValueError("--end-page must be >= --start-page")

    doc = fitz.open(args.pdf)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pdf_pages: list[Image.Image] = []

    for book_page in range(args.start_page, args.end_page + 1):
        page_index = book_page - args.page_offset
        if page_index < 0 or page_index >= len(doc):
            raise IndexError(
                f"Book page {book_page} maps to PDF index {page_index}, "
                f"but document has {len(doc)} pages."
            )

        rgb = render_page(doc[page_index], args.dpi)
        enhanced = enhance_for_ocr(rgb, args.threshold)

        stem = f"page_{book_page:03d}_{args.dpi}dpi_{args.threshold}"
        save_png(args.out_dir / f"{stem}.png", enhanced)
        if args.save_original:
            save_png(args.out_dir / f"page_{book_page:03d}_{args.dpi}dpi_original.png", rgb)
        if args.write_pdf:
            pdf_pages.append(to_pil_page(enhanced, args.dpi))

        print(f"wrote {args.out_dir / f'{stem}.png'}")

    if args.write_pdf and pdf_pages:
        suffix = f"{args.start_page:03d}-{args.end_page:03d}_{args.dpi}dpi_{args.threshold}_bw.pdf"
        pdf_path = args.out_dir / suffix
        first, *rest = pdf_pages
        first.save(
            pdf_path,
            save_all=True,
            append_images=rest,
            resolution=args.dpi,
        )
        print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
