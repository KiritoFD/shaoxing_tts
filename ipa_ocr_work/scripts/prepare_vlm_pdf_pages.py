"""Prepare cropped Shaoxing PDF page images for VLM OCR.

The VLM pipeline works best with one page image per request.  This helper
renders PDF pages, trims white margins, writes a page-image manifest, and can
also assemble the cropped images back into a PDF for visual review.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import fitz
import numpy as np
from PIL import Image, ImageEnhance
import PIL.JpegImagePlugin  # noqa: F401 - register PDF JPEG writer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PDF = PROJECT_ROOT / "ipa_ocr_work" / "data" / "shaoxing_123-351.pdf"
DEFAULT_OUT = PROJECT_ROOT / "ipa_ocr_work" / "eval" / "vlm_pages" / "post136_cropped"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render and crop PDF pages for VLM OCR.")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--start-pdf-page", type=int, default=137, help="1-based PDF page.")
    parser.add_argument("--end-pdf-page", type=int, default=0, help="1-based inclusive; 0 means last page.")
    parser.add_argument("--page-list", default="", help="Comma/range PDF pages, e.g. 7,17,27-29.")
    parser.add_argument("--source-page-offset", type=int, default=122)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--crop-threshold", type=int, default=246)
    parser.add_argument("--margin-px", type=int, default=24)
    parser.add_argument("--max-long-side", type=int, default=2400, help="0 disables resizing.")
    parser.add_argument("--contrast", type=float, default=1.08)
    parser.add_argument("--write-pdf", action="store_true", default=True)
    parser.add_argument("--no-pdf", dest="write_pdf", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_page_list(text: str, start_pdf_page: int, end_pdf_page: int) -> list[int]:
    if not text.strip():
        return list(range(start_pdf_page, end_pdf_page + 1))
    pages: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = [int(piece) for piece in part.split("-", 1)]
            pages.extend(range(lo, hi + 1))
        else:
            pages.append(int(part))
    return sorted(dict.fromkeys(pages))


def render_page(page: fitz.Page, dpi: int) -> Image.Image:
    scale = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    mode = "RGB" if pix.n >= 3 else "L"
    return Image.frombytes(mode, (pix.width, pix.height), pix.samples).convert("RGB")


def content_bbox(image: Image.Image, threshold: int, margin: int) -> tuple[int, int, int, int]:
    gray = np.asarray(image.convert("L"))
    mask = gray < threshold
    if not mask.any():
        return (0, 0, image.width, image.height)
    ys, xs = np.where(mask)
    left = max(0, int(xs.min()) - margin)
    right = min(image.width, int(xs.max()) + margin + 1)
    top = max(0, int(ys.min()) - margin)
    bottom = min(image.height, int(ys.max()) + margin + 1)
    return (left, top, right, bottom)


def resize_long_side(image: Image.Image, max_long_side: int) -> Image.Image:
    if max_long_side <= 0:
        return image
    long_side = max(image.width, image.height)
    if long_side <= max_long_side:
        return image
    scale = max_long_side / long_side
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def main() -> None:
    args = parse_args()
    if args.start_pdf_page < 1:
        raise ValueError("--start-pdf-page must be 1 or greater")

    doc = fitz.open(args.pdf)
    end_pdf_page = args.end_pdf_page or len(doc)
    if end_pdf_page > len(doc):
        raise ValueError(f"--end-pdf-page={end_pdf_page} exceeds PDF page count {len(doc)}")
    if end_pdf_page < args.start_pdf_page and not args.page_list.strip():
        raise ValueError("--end-pdf-page must be >= --start-pdf-page")

    images_dir = args.out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []
    pdf_pages: list[Image.Image] = []

    pages = parse_page_list(args.page_list, args.start_pdf_page, end_pdf_page)
    for pdf_page in pages:
        page = doc[pdf_page - 1]
        rendered = render_page(page, args.dpi)
        bbox = content_bbox(rendered, args.crop_threshold, args.margin_px)
        cropped = rendered.crop(bbox)
        if args.contrast and args.contrast != 1.0:
            cropped = ImageEnhance.Contrast(cropped).enhance(args.contrast)
        cropped = resize_long_side(cropped, args.max_long_side)

        source_page = pdf_page + args.source_page_offset
        name = f"pdf{pdf_page:03d}_p{source_page:03d}.png"
        image_path = images_dir / name
        if args.overwrite or not image_path.exists():
            cropped.save(image_path)

        rel_image = image_path.relative_to(args.out_dir).as_posix()
        manifest_rows.append(
            {
                "pdf_page": pdf_page,
                "source_page": source_page,
                "image": rel_image,
                "width": cropped.width,
                "height": cropped.height,
                "crop_bbox_render_px": repr(bbox),
                "dpi": args.dpi,
                "max_long_side": args.max_long_side,
            }
        )
        if args.write_pdf:
            pdf_page_img = cropped.convert("RGB")
            pdf_page_img.info["dpi"] = (args.dpi, args.dpi)
            pdf_pages.append(pdf_page_img)
        print(f"wrote {image_path}")

    manifest_path = args.out_dir / "page_manifest.tsv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "pdf": str(args.pdf),
        "out_dir": str(args.out_dir),
        "start_pdf_page": min(pages) if pages else args.start_pdf_page,
        "end_pdf_page": max(pages) if pages else end_pdf_page,
        "page_list": pages,
        "pages": len(manifest_rows),
        "dpi": args.dpi,
        "crop_threshold": args.crop_threshold,
        "margin_px": args.margin_px,
        "max_long_side": args.max_long_side,
        "manifest": str(manifest_path),
    }

    if args.write_pdf and pdf_pages:
        pdf_path = args.out_dir / f"shaoxing_pdf{min(pages):03d}-{max(pages):03d}_cropped_{args.dpi}dpi.pdf"
        first, *rest = pdf_pages
        first.save(pdf_path, save_all=True, append_images=rest, resolution=args.dpi)
        summary["cropped_pdf"] = str(pdf_path)
        print(f"wrote {pdf_path}")

    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {manifest_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
