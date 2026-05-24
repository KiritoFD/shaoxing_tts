"""Build a smaller-image retry manifest for failed VLM batch pages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ORIG_MANIFEST = PROJECT_ROOT / "ipa_ocr_work" / "eval" / "vlm_pages" / "post136_cropped_180dpi" / "page_manifest.tsv"
DEFAULT_BATCH_DIR = PROJECT_ROOT / "ipa_ocr_work" / "eval" / "vlm_pages" / "qwen36plus_post136_batch_20260525"
DEFAULT_OUT_DIR = PROJECT_ROOT / "ipa_ocr_work" / "eval" / "vlm_pages" / "post136_retry_failed_jpeg1600"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create compressed retry images for failed pages.")
    parser.add_argument("--orig-manifest", type=Path, default=DEFAULT_ORIG_MANIFEST)
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--statuses", default="http_error,error", help="Comma-separated statuses to include.")
    parser.add_argument("--max-long-side", type=int, default=1600)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resize_to_jpeg(src: Path, dst: Path, max_long_side: int, quality: int, overwrite: bool) -> tuple[int, int]:
    if dst.exists() and not overwrite:
        with Image.open(dst) as img:
            return img.size
    with Image.open(src) as img:
        image = img.convert("RGB")
        long_side = max(image.size)
        if long_side > max_long_side:
            scale = max_long_side / long_side
            size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
            image = image.resize(size, Image.Resampling.LANCZOS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        image.save(dst, format="JPEG", quality=quality, optimize=True)
        return image.size


def read_failed_pages(batch_dir: Path, statuses: set[str]) -> dict[int, dict[str, object]]:
    out: dict[int, dict[str, object]] = {}
    for done_path in sorted(batch_dir.glob("pdf*_p*/done.json")):
        try:
            done = json.loads(done_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if str(done.get("status", "")) in statuses:
            out[int(done["pdf_page"])] = done
    return out


def main() -> None:
    args = parse_args()
    statuses = {piece.strip() for piece in args.statuses.split(",") if piece.strip()}
    failed = read_failed_pages(args.batch_dir, statuses)
    if not failed:
        raise SystemExit("No failed pages found.")

    manifest = pd.read_csv(args.orig_manifest, sep="\t", keep_default_na=False)
    manifest = manifest[manifest["pdf_page"].astype(int).isin(failed)].copy()
    manifest = manifest.sort_values("pdf_page")
    image_dir = args.out_dir / "images"
    rows: list[dict[str, object]] = []
    for item in manifest.to_dict("records"):
        pdf_page = int(item["pdf_page"])
        src = args.orig_manifest.parent / str(item["image"])
        dst_name = Path(str(item["image"])).with_suffix(".jpg").name
        dst = image_dir / dst_name
        width, height = resize_to_jpeg(src, dst, args.max_long_side, args.jpeg_quality, args.overwrite)
        done = failed[pdf_page]
        row = dict(item)
        row["image"] = f"images/{dst_name}"
        row["width"] = width
        row["height"] = height
        row["retry_from_status"] = done.get("status", "")
        row["retry_from_http_status"] = done.get("http_status", "")
        row["retry_from_error"] = str(done.get("error", ""))[:300]
        row["max_long_side"] = args.max_long_side
        row["jpeg_quality"] = args.jpeg_quality
        rows.append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_manifest = args.out_dir / "page_manifest.tsv"
    pd.DataFrame(rows).to_csv(out_manifest, sep="\t", index=False)
    summary = {
        "orig_manifest": str(args.orig_manifest),
        "batch_dir": str(args.batch_dir),
        "out_manifest": str(out_manifest),
        "n_pages": len(rows),
        "statuses": sorted(statuses),
        "max_long_side": args.max_long_side,
        "jpeg_quality": args.jpeg_quality,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
