"""Run a TrOCR checkpoint on Shaoxing PDF pages after the trusted 136 pages.

This is an unlabeled inference/export helper. It uses the same PDF row/candidate
geometry that the training exporters use, crops the printed phonetic field, and
writes a CSV with IPA predictions plus a rule-based Wu-pinyin conversion.
"""

from __future__ import annotations

import argparse
import csv
import json
import unicodedata
from pathlib import Path

import fitz
import pandas as pd
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from export_ipa_training_data_matched import candidates_for_page
from export_wupin_training_data import has_cjk, page_visual_rows, phoneticish_score
from train_trocr_wupin import prepare_image
from wupin_ipa_convert import DEFAULT_MAP, ipa_to_wupin, load_mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PDF = PROJECT_ROOT / "ipa_ocr_work" / "data" / "shaoxing_123-351.pdf"
DEFAULT_MODEL = PROJECT_ROOT / "ipa_ocr_work" / "models" / "E3_best_local"
DEFAULT_OUT_DIR = PROJECT_ROOT / "ipa_ocr_work" / "eval" / "post136_trocr_best"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer post-136 Shaoxing PDF rows with TrOCR.")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--start-pdf-page", type=int, default=137)
    parser.add_argument("--end-pdf-page", type=int, default=0, help="1-based PDF page, inclusive. 0 means last page.")
    parser.add_argument("--source-page-offset", type=int, default=122, help="source_page = pdf_page + offset")
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--image-mode", choices=["raw", "pad-square"], default="pad-square")
    parser.add_argument("--min-phonetic-width", type=float, default=28.0)
    parser.add_argument("--max-phonetic-width", type=float, default=280.0)
    parser.add_argument("--require-following-phonetic", action="store_true")
    parser.add_argument("--skip-secondary-candidates", action="store_true")
    parser.add_argument("--limit-pages", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_prediction(text: object) -> str:
    return unicodedata.normalize("NFC", str(text)).strip().replace(" ", "")


def ensure_processor_files(model_dir: Path) -> None:
    """Make old saved TrOCRProcessor directories readable by newer transformers."""
    processor_config = model_dir / "processor_config.json"
    preprocessor_config = model_dir / "preprocessor_config.json"
    legacy_config = model_dir / "processor_config.legacy_nested.json"
    if not processor_config.exists() or preprocessor_config.exists():
        return
    payload = json.loads(processor_config.read_text(encoding="utf-8"))
    image_processor = payload.get("image_processor")
    if not isinstance(image_processor, dict):
        return
    if not legacy_config.exists():
        legacy_config.write_text(processor_config.read_text(encoding="utf-8"), encoding="utf-8")
    preprocessor_config.write_text(json.dumps(image_processor, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    processor_config.write_text(json.dumps({"processor_class": "TrOCRProcessor"}, indent=2) + "\n", encoding="utf-8")


def definition_like_span(text: str, width: float, gap: float, started: bool) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    phonetic_score = phoneticish_score(stripped)
    cjk = has_cjk(stripped)
    if any(ch in stripped for ch in "：:。；;，,、"):
        return phonetic_score == 0 or cjk
    if not cjk:
        return False
    # Hidden PDF extraction mangles some IPA glyphs into single CJK-looking
    # glyphs. Those are usually short and adjacent to Latin/digit spans. Real
    # explanations are wider or separated after the phonetic cluster.
    if phonetic_score == 0 and (width >= 38 or (started and gap >= 12)):
        return True
    return False


def phonetic_stop_offset(text: str) -> int | None:
    """Return a character offset where explanations start inside a PDF span."""
    for i, ch in enumerate(text):
        prefix = text[:i].strip()
        if ch.isspace():
            continue
        if not prefix or phoneticish_score(prefix) < 3:
            continue
        suffix = text[i:]
        prev = text[i - 1] if i else ""
        if ch in "([{（［【\"“‘《〈" and has_cjk(suffix):
            return i
        if phoneticish_score(ch) > 0:
            continue
        if has_cjk(ch):
            # Single corrupted IPA glyphs can be embedded between Latin pieces.
            # Treat CJK as explanation only after a clear separator.
            if prev.isspace() or prev in "，,。；;：:":
                return i
    return None


def span_x_at_offset(text: str, bbox: tuple[float, float, float, float], offset: int) -> float:
    x0, _, x1, _ = bbox
    if offset <= 0 or not text:
        return x0
    return x0 + (x1 - x0) * min(offset, len(text)) / len(text)


def starts_parenthetical_explanation(spans: list[dict], span_index: int) -> bool:
    text = spans[span_index]["text"].strip()
    if text not in {"(", "（", "[", "［", "【"}:
        return False
    saw_close = False
    for next_span in spans[span_index + 1 :]:
        next_text = next_span["text"].strip()
        if not next_text:
            continue
        if has_cjk(next_text):
            return saw_close or text in {"[", "［", "【"}
        if next_text in {")", "）", "]", "］", "】"}:
            saw_close = True
            continue
        if all(ch.isdigit() or ch.isspace() for ch in next_text):
            continue
        return False
    return False


def phonetic_bbox_for_candidate(page_obj: fitz.Page, cand, next_x: float, args: argparse.Namespace) -> tuple[float, float, float, float] | None:
    visual_rows = page_visual_rows(page_obj)
    if cand.row_no < 0 or cand.row_no >= len(visual_rows):
        crop_start = cand.head_x1 + 3
        crop_end = max(crop_start + args.min_phonetic_width, min(next_x, crop_start + args.max_phonetic_width))
        return (crop_start, cand.y0, crop_end, cand.y1)

    row = visual_rows[cand.row_no]
    crop_start = cand.head_x1 + 3
    hard_end = min(next_x, crop_start + args.max_phonetic_width, row.x1 + 4)
    last_end = crop_start
    stop_x: float | None = None
    started = False
    saw_usable = False
    for span_index, span in enumerate(row.spans):
        text = span["text"]
        x0, _, x1, _ = span["bbox"]
        if x1 <= crop_start:
            continue
        if x0 >= hard_end:
            break
        width = x1 - x0
        gap = max(0.0, x0 - last_end)
        if started and starts_parenthetical_explanation(row.spans, span_index):
            stop_x = x0 - 2
            break
        stop_offset = phonetic_stop_offset(text) if started or phoneticish_score(text) >= 3 else None
        if stop_offset is not None:
            stop_x = span_x_at_offset(text, span["bbox"], stop_offset) - 2
            break
        if definition_like_span(text, width, gap, started):
            stop_x = x0 - 2
            break
        score = phoneticish_score(text)
        if score > 0 or any(ch.isdigit() for ch in text) or not has_cjk(text):
            saw_usable = True
        elif not started and width >= 24:
            break
        started = True
        last_end = max(last_end, min(x1, hard_end))

    if not saw_usable:
        return None
    preferred_end = stop_x if stop_x is not None else last_end + 3
    crop_end = min(hard_end, max(preferred_end, crop_start + args.min_phonetic_width))
    if crop_end - crop_start < args.min_phonetic_width:
        return None
    return (crop_start, row.crop_y0, crop_end, row.crop_y1)


def crop_rows(args: argparse.Namespace) -> pd.DataFrame:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    image_root = args.out_dir / "images"
    image_root.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(args.pdf)
    end_pdf_page = args.end_pdf_page or len(doc)
    if args.limit_pages:
        end_pdf_page = min(end_pdf_page, args.start_pdf_page + args.limit_pages - 1)
    matrix = fitz.Matrix(args.dpi / 72.0, args.dpi / 72.0)

    rows: list[dict[str, object]] = []
    for pdf_page in range(args.start_pdf_page, end_pdf_page + 1):
        page_index = pdf_page - 1
        if page_index < 0 or page_index >= len(doc):
            continue
        source_page = pdf_page + args.source_page_offset
        page_obj = doc[page_index]
        candidates = candidates_for_page(source_page, page_obj, args.require_following_phonetic)
        for idx, cand in enumerate(candidates):
            if args.skip_secondary_candidates and cand.entry_no > 0:
                continue
            next_x = cand.row_x1
            if idx + 1 < len(candidates) and candidates[idx + 1].row_no == cand.row_no:
                next_x = candidates[idx + 1].head_x0 - 6
            bbox = phonetic_bbox_for_candidate(page_obj, cand, next_x, args)
            if bbox is None:
                continue
            sample_id = f"pdf{pdf_page:03d}_p{source_page:03d}_{idx:04d}"
            image_rel = f"images/{sample_id}.png"
            page_obj.get_pixmap(matrix=matrix, clip=fitz.Rect(bbox), alpha=False).save(str(args.out_dir / image_rel))
            rows.append(
                {
                    "sample_id": sample_id,
                    "pdf_page": pdf_page,
                    "source_page": source_page,
                    "row_no": cand.row_no,
                    "entry_no": cand.entry_no,
                    "candidate_headword": cand.headword,
                    "pdf_text": cand.text,
                    "crop_bbox": repr(bbox),
                    "image": image_rel,
                }
            )
        print(f"cropped pdf_page={pdf_page} source_page={source_page} candidates={len(candidates)}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "post136_manifest.tsv", sep="\t", index=False)
    return df


def predict(args: argparse.Namespace, df: pd.DataFrame) -> pd.DataFrame:
    ensure_processor_files(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = TrOCRProcessor.from_pretrained(args.model)
    model = VisionEncoderDecoderModel.from_pretrained(args.model).to(device)
    model.eval()
    mapping = load_mapping(args.mapping)

    predictions: list[str] = []
    wupin_predictions: list[str] = []
    wupin_statuses: list[str] = []
    for start in range(0, len(df), args.batch_size):
        batch = df.iloc[start : start + args.batch_size]
        images = [
            prepare_image(Image.open(args.out_dir / row.image).convert("RGB"), args.image_mode)
            for row in batch.itertuples()
        ]
        pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            generated_ids = model.generate(pixel_values, max_new_tokens=args.max_new_tokens, num_beams=1)
        texts = [normalize_prediction(text) for text in processor.batch_decode(generated_ids, skip_special_tokens=True)]
        for text in texts:
            wupin, errors = ipa_to_wupin(text, mapping)
            predictions.append(text)
            wupin_predictions.append(wupin)
            wupin_statuses.append("ok" if not errors else ";".join(errors))
        print(f"predicted {min(start + args.batch_size, len(df))}/{len(df)}")

    out = df.copy()
    out["pred_ipa"] = predictions
    out["pred_wupin"] = wupin_predictions
    out["pred_wupin_status"] = wupin_statuses
    return out


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and args.overwrite:
        for child in args.out_dir.iterdir():
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                import shutil

                shutil.rmtree(child)
    df = crop_rows(args)
    if df.empty:
        raise SystemExit("no crops found")
    out = predict(args, df)
    csv_path = args.out_dir / "post136_trocr_best_predictions.csv"
    tsv_path = args.out_dir / "post136_trocr_best_predictions.tsv"
    out.to_csv(csv_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    out.to_csv(tsv_path, sep="\t", index=False)
    sample = out.groupby("pdf_page", group_keys=False).head(2).head(40)
    sample.to_csv(args.out_dir / "post136_sample_for_review.csv", index=False, encoding="utf-8-sig")
    print(f"rows: {len(out)}")
    print(f"wrote {csv_path}")
    print(f"wrote {tsv_path}")
    print(f"wrote {args.out_dir / 'post136_sample_for_review.csv'}")


if __name__ == "__main__":
    main()
