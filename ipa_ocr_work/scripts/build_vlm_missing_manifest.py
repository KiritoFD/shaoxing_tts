"""Build a retry manifest for post-136 pages that still have no collected rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ORIG_MANIFEST = PROJECT_ROOT / "ipa_ocr_work" / "eval" / "vlm_pages" / "post136_cropped_180dpi" / "page_manifest.tsv"
DEFAULT_CURRENT_SUMMARY = PROJECT_ROOT / "ipa_ocr_work" / "eval" / "vlm_pages" / "qwen36plus_post136_current_best.summary.json"
DEFAULT_CURRENT_PAGES = PROJECT_ROOT / "ipa_ocr_work" / "eval" / "vlm_pages" / "qwen36plus_post136_current_best.pages.csv"
DEFAULT_CURRENT_ATTEMPTS = PROJECT_ROOT / "ipa_ocr_work" / "eval" / "vlm_pages" / "qwen36plus_post136_current_best.attempts.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "ipa_ocr_work" / "eval" / "vlm_pages" / "post136_missing_retry_jpeg1400"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create compressed images and manifest for pages still missing VLM rows.")
    parser.add_argument("--orig-manifest", type=Path, default=DEFAULT_ORIG_MANIFEST)
    parser.add_argument("--current-summary", type=Path, default=DEFAULT_CURRENT_SUMMARY)
    parser.add_argument("--current-pages", type=Path, default=DEFAULT_CURRENT_PAGES)
    parser.add_argument("--current-attempts", type=Path, default=DEFAULT_CURRENT_ATTEMPTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-long-side", type=int, default=1400)
    parser.add_argument("--jpeg-quality", type=int, default=82)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resize_to_jpeg(src: Path, dst: Path, max_long_side: int, quality: int, overwrite: bool) -> tuple[int, int, int]:
    if dst.exists() and not overwrite:
        with Image.open(dst) as img:
            return img.width, img.height, dst.stat().st_size
    with Image.open(src) as img:
        image = img.convert("RGB")
        long_side = max(image.size)
        if long_side > max_long_side:
            scale = max_long_side / long_side
            size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
            image = image.resize(size, Image.Resampling.LANCZOS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        image.save(dst, format="JPEG", quality=quality, optimize=True)
        return image.width, image.height, dst.stat().st_size


def latest_page_status(current_pages: Path) -> dict[int, dict[str, object]]:
    if not current_pages.exists():
        return {}
    pages = pd.read_csv(current_pages, keep_default_na=False)
    out: dict[int, dict[str, object]] = {}
    for row in pages.to_dict("records"):
        try:
            out[int(row.get("pdf_page", 0))] = row
        except Exception:
            continue
    return out


def latest_attempt_status(current_attempts: Path, fallback_pages: Path) -> dict[int, dict[str, object]]:
    if not current_attempts.exists():
        return latest_page_status(fallback_pages)
    attempts = pd.read_csv(current_attempts, keep_default_na=False)
    if attempts.empty:
        return latest_page_status(fallback_pages)
    out: dict[int, dict[str, object]] = {}
    for _, row in attempts.iterrows():
        try:
            pdf_page = int(row.get("pdf_page", 0))
            priority = int(row.get("priority", 0))
        except Exception:
            continue
        previous = out.get(pdf_page)
        previous_priority = int(previous.get("priority", -1)) if previous else -1
        if previous is None or priority >= previous_priority:
            out[pdf_page] = dict(row)
    return out


def main() -> None:
    args = parse_args()
    summary = json.loads(args.current_summary.read_text(encoding="utf-8"))
    missing_pages = [int(page) for page in summary.get("missing_pages", [])]
    status_by_page = latest_attempt_status(args.current_attempts, args.current_pages)
    manifest = pd.read_csv(args.orig_manifest, sep="\t", keep_default_na=False)
    manifest = manifest[manifest["pdf_page"].astype(int).isin(missing_pages)].copy()
    manifest = manifest.sort_values("pdf_page")

    image_dir = args.out_dir / "images"
    rows: list[dict[str, object]] = []
    reason_counts: dict[str, int] = {}
    for item in manifest.to_dict("records"):
        pdf_page = int(item["pdf_page"])
        src = args.orig_manifest.parent / str(item["image"])
        dst_name = Path(str(item["image"])).with_suffix(".jpg").name
        dst = image_dir / dst_name
        width, height, bytes_out = resize_to_jpeg(src, dst, args.max_long_side, args.jpeg_quality, args.overwrite)
        latest = status_by_page.get(pdf_page, {})
        reason = f"{latest.get('status', 'missing')}:{latest.get('http_status', '')}:{latest.get('json_source', '')}"
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        row = dict(item)
        row["image"] = f"images/{dst_name}"
        row["width"] = width
        row["height"] = height
        row["jpeg_bytes"] = bytes_out
        row["missing_reason"] = reason
        row["latest_status"] = latest.get("status", "")
        row["latest_http_status"] = latest.get("http_status", "")
        row["latest_json_source"] = latest.get("json_source", "")
        row["latest_error"] = str(latest.get("error", ""))[:300]
        row["max_long_side"] = args.max_long_side
        row["jpeg_quality"] = args.jpeg_quality
        rows.append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_manifest = args.out_dir / "page_manifest.tsv"
    pd.DataFrame(rows).to_csv(out_manifest, sep="\t", index=False)
    out_summary = {
        "orig_manifest": str(args.orig_manifest),
        "current_summary": str(args.current_summary),
        "current_pages": str(args.current_pages),
        "current_attempts": str(args.current_attempts),
        "out_manifest": str(out_manifest),
        "n_pages": len(rows),
        "pdf_pages": [int(row["pdf_page"]) for row in rows],
        "source_pages": [int(row["source_page"]) for row in rows],
        "reason_counts": reason_counts,
        "max_long_side": args.max_long_side,
        "jpeg_quality": args.jpeg_quality,
        "total_jpeg_bytes": sum(int(row["jpeg_bytes"]) for row in rows),
        "max_jpeg_bytes": max([int(row["jpeg_bytes"]) for row in rows], default=0),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(out_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
