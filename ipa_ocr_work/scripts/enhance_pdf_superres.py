"""Denoise and upscale scanned PDF pages for OCR.

This is an OCR-oriented enhancement pipeline, not a document beautifier:
it preserves tiny tone digits and diacritics by avoiding erosion/opening.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2 as cv
import fitz
import numpy as np
from PIL import Image
import PIL.JpegImagePlugin  # noqa: F401 - registers PDF JPEG encoder on some installs


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PDF = PROJECT_ROOT / "ipa_ocr_work" / "data" / "shaoxing_123-351.pdf"
DEFAULT_OUT = PROJECT_ROOT / "ipa_ocr_work" / "output" / "superres_pages"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Denoise and upscale PDF pages.")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--start-page", type=int, default=123)
    parser.add_argument("--end-page", type=int, default=130)
    parser.add_argument("--page-offset", type=int, default=123)
    parser.add_argument("--render-dpi", type=int, default=300)
    parser.add_argument("--upscale", type=float, default=1.5)
    parser.add_argument("--denoise", type=float, default=5.0)
    parser.add_argument("--sharpen", type=float, default=0.55)
    parser.add_argument("--threshold", choices=("none", "otsu", "adaptive", "sauvola"), default="none")
    parser.add_argument("--write-pdf", action="store_true")
    parser.add_argument("--save-original", action="store_true")
    return parser.parse_args()


def render_page(page: fitz.Page, dpi: int) -> np.ndarray:
    scale = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)


def guided_filter(gray: np.ndarray, radius: int = 2, eps: float = 1000.0) -> np.ndarray:
    gray_f = gray.astype(np.float32)
    kernel = (2 * radius + 1, 2 * radius + 1)
    mean_i = cv.boxFilter(gray_f, cv.CV_32F, kernel, borderType=cv.BORDER_REFLECT)
    mean_p = cv.boxFilter(gray_f, cv.CV_32F, kernel, borderType=cv.BORDER_REFLECT)
    mean_ip = cv.boxFilter(gray_f * gray_f, cv.CV_32F, kernel, borderType=cv.BORDER_REFLECT)
    cov_ip = mean_ip - mean_i * mean_p
    mean_ii = cv.boxFilter(gray_f * gray_f, cv.CV_32F, kernel, borderType=cv.BORDER_REFLECT)
    var_i = mean_ii - mean_i * mean_i
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    mean_a = cv.boxFilter(a, cv.CV_32F, kernel, borderType=cv.BORDER_REFLECT)
    mean_b = cv.boxFilter(b, cv.CV_32F, kernel, borderType=cv.BORDER_REFLECT)
    return mean_a * gray_f + mean_b


def sauvola_threshold(gray: np.ndarray, k: float = 0.2, r: float = 128.0) -> np.ndarray:
    win = max(15, min(gray.shape[:2]) // 20)
    win = win if win % 2 == 1 else win + 1
    gray_f = gray.astype(np.float32)
    mean = cv.boxFilter(gray_f, cv.CV_32F, (win, win), borderType=cv.BORDER_REFLECT)
    mean_sq = cv.boxFilter(gray_f * gray_f, cv.CV_32F, (win, win), borderType=cv.BORDER_REFLECT)
    std = np.sqrt(np.maximum(mean_sq - mean * mean, 0))
    threshold = mean * (1 + k * ((std / r) - 1))
    return (gray_f > threshold).astype(np.uint8) * 255


def enhance(
    rgb: np.ndarray,
    upscale: float,
    denoise: float,
    sharpen: float,
    threshold: str,
    preserve_details: bool = False,
) -> np.ndarray:
    gray = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)

    if denoise > 0:
        gray = cv.fastNlMeansDenoising(
            gray,
            h=denoise,
            templateWindowSize=7,
            searchWindowSize=21,
        )

    # Gentle local contrast; high clip limits make small tone digits break up.
    gray = cv.createCLAHE(clipLimit=1.8, tileGridSize=(12, 12)).apply(gray)

    if upscale != 1:
        h, w = gray.shape
        gray = cv.resize(
            gray,
            (max(1, round(w * upscale)), max(1, round(h * upscale))),
            interpolation=cv.INTER_CUBIC if preserve_details else cv.INTER_LANCZOS4,
        )
        if preserve_details:
            guided = guided_filter(gray)
            detail = gray.astype(np.float32) - guided
            gray = np.clip(guided + 1.4 * detail, 0, 255).astype(np.uint8)

    if sharpen > 0:
        blur = cv.GaussianBlur(gray, (0, 0), 1.0)
        gray = cv.addWeighted(gray, 1.0 + sharpen, blur, -sharpen, 0)

    if threshold == "none":
        return gray
    if threshold == "otsu":
        _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        return binary
    if threshold == "sauvola":
        return sauvola_threshold(gray)
    return cv.adaptiveThreshold(
        gray,
        255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY,
        41,
        13,
    )


def save_png(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv.imwrite(str(path), img):
        raise OSError(f"failed to write {path}")


def pil_page(img: np.ndarray, output_dpi: int, threshold: str) -> Image.Image:
    pil = Image.fromarray(img)
    if threshold != "none":
        pil = pil.convert("1")
    else:
        pil = pil.convert("L")
    pil.info["dpi"] = (output_dpi, output_dpi)
    return pil


def main() -> None:
    args = parse_args()
    if args.end_page < args.start_page:
        raise ValueError("--end-page must be >= --start-page")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(args.pdf)
    effective_dpi = round(args.render_dpi * args.upscale)
    pages: list[Image.Image] = []

    suffix = f"{args.render_dpi}dpi_x{args.upscale:g}_{args.threshold}"
    for book_page in range(args.start_page, args.end_page + 1):
        pdf_index = book_page - args.page_offset
        if pdf_index < 0 or pdf_index >= len(doc):
            raise IndexError(f"page {book_page} maps to invalid PDF index {pdf_index}")

        rgb = render_page(doc[pdf_index], args.render_dpi)
        enhanced = enhance(rgb, args.upscale, args.denoise, args.sharpen, args.threshold)
        out_png = args.out_dir / f"page_{book_page:03d}_{suffix}.png"
        save_png(out_png, enhanced)
        if args.save_original:
            original_gray = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)
            save_png(args.out_dir / f"page_{book_page:03d}_{args.render_dpi}dpi_original.png", original_gray)
        if args.write_pdf:
            pages.append(pil_page(enhanced, effective_dpi, args.threshold))
        print(f"wrote {out_png}")

    if args.write_pdf and pages:
        pdf_name = f"{args.start_page:03d}-{args.end_page:03d}_{suffix}.pdf"
        first, *rest = pages
        first.save(
            args.out_dir / pdf_name,
            save_all=True,
            append_images=rest,
            resolution=effective_dpi,
        )
        print(f"wrote {args.out_dir / pdf_name}")


if __name__ == "__main__":
    main()
