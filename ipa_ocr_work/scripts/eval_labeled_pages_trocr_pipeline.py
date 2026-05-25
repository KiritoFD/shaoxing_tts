"""Evaluate the Shaoxing OCR pipeline at page level.

This script scores the actual inference path:

    PDF page -> entry candidates -> phonetic crop -> TrOCR -> page gold rows

It intentionally does not assume that the gold row crop is already given.  The
alignment diagnostics are as important as the OCR score because missed or extra
page candidates are pipeline errors, not just row-recognition errors.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import fitz
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from export_ipa_training_data_matched import Candidate, align_labels, candidates_for_page
from infer_post136_trocr import (
    DEFAULT_MODEL,
    DEFAULT_PDF,
    ensure_processor_files,
    normalize_prediction,
    phonetic_bbox_for_candidate,
)
from score_ocr_experiment import normalize, score_pair, summarize, wupin_normalize
from train_trocr_wupin import prepare_image
from wupin_ipa_convert import DEFAULT_MAP, build_row_ipa_lexicon, ipa_to_wupin, load_mapping, wupin_to_ipa


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_MANIFEST = (
    PROJECT_ROOT
    / "ipa_ocr_work"
    / "dataset"
    / "shaoxing_pdf136_clean"
    / "ocr_selected_all"
    / "eval_manifest.tsv"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "ipa_ocr_work" / "eval" / "page_pipeline_trocr"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Page-level labeled Shaoxing TrOCR evaluation.")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--eval-manifest", type=Path, default=DEFAULT_EVAL_MANIFEST)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--start-pdf-page", type=int, default=1)
    parser.add_argument("--end-pdf-page", type=int, default=136)
    parser.add_argument(
        "--page-list",
        default="",
        help="Comma/range PDF pages, e.g. 7,8,12-14. Overrides start/end when set.",
    )
    parser.add_argument("--source-page-offset", type=int, default=122)
    parser.add_argument("--source-split", default="test", help="train/val/test/all. Default: test.")
    parser.add_argument("--variant", default="page_pipeline")
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--image-mode", choices=["raw", "pad-square"], default="pad-square")
    parser.add_argument("--prediction-mode", choices=["ipa"], default="ipa")
    parser.add_argument("--ipa-label-source", choices=["label", "from-wupin"], default="from-wupin")
    parser.add_argument("--alignment", choices=["monotonic", "order"], default="monotonic")
    parser.add_argument("--max-gap", type=int, default=6)
    parser.add_argument("--min-phonetic-width", type=float, default=28.0)
    parser.add_argument("--max-phonetic-width", type=float, default=280.0)
    parser.add_argument("--require-following-phonetic", action="store_true")
    parser.add_argument("--skip-secondary-candidates", action="store_true")
    parser.add_argument("--crop-only", action="store_true")
    parser.add_argument("--limit-pages", type=int, default=0)
    parser.add_argument("--contactsheet-limit", type=int, default=80)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_page_list(text: str, start_pdf_page: int, end_pdf_page: int, limit_pages: int) -> list[int]:
    if text.strip():
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
        pages = sorted(dict.fromkeys(pages))
    else:
        pages = list(range(start_pdf_page, end_pdf_page + 1))
    if limit_pages:
        pages = pages[:limit_pages]
    return pages


def load_gold_rows(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.eval_manifest, sep="\t", keep_default_na=False)
    if args.source_split != "all":
        df = df[df["source_split"] == args.source_split].copy()
    df = df[df["wupin"].astype(str) != ""].copy()
    df["page"] = df["page"].astype(int)
    if "row_index" in df.columns:
        sort_cols = ["page", "row_index", "sample_id"]
    else:
        sort_cols = ["page", "sample_id"]
    return df.sort_values(sort_cols).reset_index(drop=True)


def candidate_to_dict(pdf_page: int, source_page: int, cand_index: int, cand: Candidate, bbox: tuple[float, float, float, float] | None) -> dict:
    return {
        "candidate_id": f"pdf{pdf_page:03d}_p{source_page:03d}_cand{cand_index:04d}",
        "pdf_page": pdf_page,
        "source_page": source_page,
        "candidate_index": cand_index,
        "row_no": cand.row_no,
        "entry_no": cand.entry_no,
        "candidate_headword": cand.headword,
        "candidate_text": cand.text,
        "crop_bbox": repr(bbox) if bbox is not None else "",
        "has_crop": bool(bbox is not None),
        "image": "",
    }


def crop_page_candidates(args: argparse.Namespace, pages: list[int]) -> pd.DataFrame:
    image_root = args.out_dir / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(args.pdf)
    matrix = fitz.Matrix(args.dpi / 72.0, args.dpi / 72.0)
    rows: list[dict] = []

    for pdf_page in pages:
        page_index = pdf_page - 1
        if page_index < 0 or page_index >= len(doc):
            continue
        source_page = pdf_page + args.source_page_offset
        page_obj = doc[page_index]
        candidates = candidates_for_page(source_page, page_obj, args.require_following_phonetic)
        if args.skip_secondary_candidates:
            candidates = [cand for cand in candidates if cand.entry_no == 0]
        for idx, cand in enumerate(candidates):
            next_x = cand.row_x1
            if idx + 1 < len(candidates) and candidates[idx + 1].row_no == cand.row_no:
                next_x = candidates[idx + 1].head_x0 - 6
            bbox = phonetic_bbox_for_candidate(page_obj, cand, next_x, args)
            record = candidate_to_dict(pdf_page, source_page, idx, cand, bbox)
            if bbox is not None:
                image_rel = f"images/{record['candidate_id']}.png"
                page_obj.get_pixmap(matrix=matrix, clip=fitz.Rect(bbox), alpha=False).save(str(args.out_dir / image_rel))
                record["image"] = image_rel
            rows.append(record)
        print(f"page pdf={pdf_page} source={source_page} raw_candidates={len(candidates)}")

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "candidate_id",
                "pdf_page",
                "source_page",
                "candidate_index",
                "row_no",
                "entry_no",
                "candidate_headword",
                "candidate_text",
                "crop_bbox",
                "has_crop",
                "image",
            ]
        )
    df.to_csv(args.out_dir / "candidate_manifest.tsv", sep="\t", index=False)
    return df


def align_page(gold_page: pd.DataFrame, candidates_page: pd.DataFrame, args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    gold_records = gold_page.to_dict("records")
    crop_candidates = candidates_page[candidates_page["has_crop"]].reset_index(drop=True)
    candidate_records = crop_candidates.to_dict("records")
    diagnostics: list[dict] = []
    manifest_rows: list[dict] = []

    if args.alignment == "order":
        pairs = [(idx, idx if idx < len(candidate_records) else None, 1.0 if idx < len(candidate_records) else 0.0) for idx in range(len(gold_records))]
    else:
        candidate_objs = [
            Candidate(
                page=int(row["source_page"]),
                row_no=int(row["row_no"]),
                entry_no=int(row["entry_no"]),
                headword=str(row["candidate_headword"]),
                head_x0=0.0,
                head_x1=0.0,
                row_x1=0.0,
                y0=0.0,
                y1=0.0,
                text=str(row["candidate_text"]),
            )
            for row in candidate_records
        ]
        pairs = align_labels(gold_records, candidate_objs, args.max_gap)

    used_candidate_indexes = set()
    for gold_index, candidate_index, match_score in pairs:
        gold = gold_records[gold_index]
        candidate = candidate_records[candidate_index] if candidate_index is not None else None
        if candidate is not None:
            used_candidate_indexes.add(candidate_index)
        sample_id = str(gold["sample_id"])
        row = dict(gold)
        row["variant"] = args.variant
        row["page_pipeline_sample_id"] = sample_id
        row["image"] = candidate["image"] if candidate is not None else ""
        row["candidate_id"] = candidate["candidate_id"] if candidate is not None else ""
        row["candidate_index"] = candidate["candidate_index"] if candidate is not None else ""
        row["candidate_headword"] = candidate["candidate_headword"] if candidate is not None else ""
        row["candidate_text"] = candidate["candidate_text"] if candidate is not None else ""
        row["crop_bbox"] = candidate["crop_bbox"] if candidate is not None else ""
        row["alignment_score"] = match_score
        row["alignment_status"] = "matched" if candidate is not None else "missing_candidate"
        manifest_rows.append(row)
        diagnostics.append(
            {
                "page": gold.get("page", ""),
                "pdf_page": gold.get("pdf_page", ""),
                "gold_index": gold_index,
                "sample_id": sample_id,
                "gold_hanzi": gold.get("hanzi", ""),
                "gold_wupin": gold.get("wupin", ""),
                "candidate_index": "" if candidate is None else candidate["candidate_index"],
                "candidate_id": "" if candidate is None else candidate["candidate_id"],
                "candidate_headword": "" if candidate is None else candidate["candidate_headword"],
                "alignment_score": match_score,
                "status": "matched" if candidate is not None else "missing_candidate",
            }
        )

    for candidate_index, candidate in enumerate(candidate_records):
        if candidate_index in used_candidate_indexes:
            continue
        diagnostics.append(
            {
                "page": candidate["source_page"],
                "pdf_page": candidate["pdf_page"],
                "gold_index": "",
                "sample_id": "",
                "gold_hanzi": "",
                "gold_wupin": "",
                "candidate_index": candidate["candidate_index"],
                "candidate_id": candidate["candidate_id"],
                "candidate_headword": candidate["candidate_headword"],
                "alignment_score": "",
                "status": "extra_candidate",
            }
        )

    return manifest_rows, diagnostics


def build_page_eval_manifest(args: argparse.Namespace, gold: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    diagnostics: list[dict] = []
    for page, gold_page in gold.groupby("page", sort=True):
        candidates_page = candidates[candidates["source_page"] == int(page)].copy()
        page_rows, page_diags = align_page(gold_page, candidates_page, args)
        rows.extend(page_rows)
        diagnostics.extend(page_diags)
        print(
            "align page="
            f"{page} gold={len(gold_page)} crop_candidates={int(candidates_page['has_crop'].sum())} "
            f"manifest_rows={len(page_rows)}"
        )
    manifest = pd.DataFrame(rows)
    diagnostics_df = pd.DataFrame(diagnostics)
    manifest.to_csv(args.out_dir / "page_eval_manifest.tsv", sep="\t", index=False)
    diagnostics_df.to_csv(args.out_dir / "alignment_diagnostics.tsv", sep="\t", index=False)
    return manifest


def write_contact_sheet(args: argparse.Namespace, manifest: pd.DataFrame) -> None:
    rows = manifest[manifest["image"].astype(str) != ""].head(args.contactsheet_limit)
    if rows.empty:
        return
    thumbs: list[tuple[Image.Image, str]] = []
    font = ImageFont.load_default()
    for row in rows.itertuples(index=False):
        image = Image.open(args.out_dir / row.image).convert("RGB")
        image.thumbnail((360, 96), Image.Resampling.LANCZOS)
        caption = f"{getattr(row, 'sample_id')} | p{getattr(row, 'page')} r{getattr(row, 'row_index', '')} | score={float(getattr(row, 'alignment_score')):.2f}"
        thumbs.append((image.copy(), caption))
    width = 420
    cell_h = 136
    cols = 2
    out = Image.new("RGB", (width * cols, cell_h * ((len(thumbs) + cols - 1) // cols)), "white")
    draw = ImageDraw.Draw(out)
    for idx, (thumb, caption) in enumerate(thumbs):
        x = (idx % cols) * width
        y = (idx // cols) * cell_h
        out.paste(thumb, (x + 8, y + 8))
        draw.text((x + 8, y + 110), caption[:70], fill="black", font=font)
    out.save(args.out_dir / "qa_contactsheet.png")


def predict(args: argparse.Namespace, manifest: pd.DataFrame) -> pd.DataFrame:
    ensure_processor_files(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = TrOCRProcessor.from_pretrained(args.model)
    model = VisionEncoderDecoderModel.from_pretrained(args.model).to(device)
    model.eval()
    mapping = load_mapping(args.mapping)
    rows = manifest.to_dict("records")
    predictions: list[dict] = []

    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        batch_with_images = [row for row in batch_rows if str(row.get("image", ""))]
        batch_predictions = {row["sample_id"]: "" for row in batch_rows}
        if batch_with_images:
            images = [
                prepare_image(Image.open(args.out_dir / row["image"]).convert("RGB"), args.image_mode)
                for row in batch_with_images
            ]
            pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(device)
            with torch.no_grad():
                generated_ids = model.generate(pixel_values, max_new_tokens=args.max_new_tokens, num_beams=1)
            texts = [normalize_prediction(text) for text in processor.batch_decode(generated_ids, skip_special_tokens=True)]
            for row, text in zip(batch_with_images, texts):
                batch_predictions[row["sample_id"]] = text
        for row in batch_rows:
            prediction = batch_predictions[row["sample_id"]]
            wupin, errors = ipa_to_wupin(prediction, mapping)
            predictions.append(
                {
                    "sample_id": row["sample_id"],
                    "variant": args.variant,
                    "prediction": prediction,
                    "pred_wupin": wupin,
                    "pred_wupin_status": "ok" if not errors else ";".join(errors),
                    "candidate_id": row.get("candidate_id", ""),
                    "alignment_status": row.get("alignment_status", ""),
                }
            )
        print(f"predicted {min(start + args.batch_size, len(rows))}/{len(rows)}")

    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(args.out_dir / "page_predictions.tsv", sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)
    return pred_df


def score_page_predictions(args: argparse.Namespace, manifest: pd.DataFrame, predictions: pd.DataFrame) -> None:
    mapping = load_mapping(args.mapping)
    row_lexicon = build_row_ipa_lexicon(manifest["wupin"], mapping, "digits")
    merged = manifest.merge(
        predictions[["sample_id", "variant", "prediction"]],
        on=["sample_id", "variant"],
        how="left",
    )
    merged["prediction"] = merged["prediction"].fillna("").map(normalize)
    merged["wupin_label"] = merged["wupin"].map(wupin_normalize)
    if args.ipa_label_source == "from-wupin":
        labels = []
        statuses = []
        for wupin in merged["wupin_label"]:
            ipa, errors = wupin_to_ipa(wupin, mapping, "digits")
            labels.append(normalize(ipa))
            statuses.append("ok" if not errors else ";".join(errors))
        merged["ipa_label"] = labels
        merged["ipa_label_status"] = statuses
    else:
        merged["ipa_label"] = merged["label"].map(normalize)
        merged["ipa_label_status"] = "manifest_label"

    merged["ipa_prediction"] = merged["prediction"].map(normalize)
    converted = []
    statuses = []
    for pred, ipa_label, wupin_label in zip(merged["ipa_prediction"], merged["ipa_label"], merged["wupin_label"]):
        if pred == ipa_label:
            wupin, errors = wupin_label, []
        else:
            wupin, errors = ipa_to_wupin(pred, mapping, row_lexicon=row_lexicon)
        converted.append(wupin_normalize(wupin))
        statuses.append("ok" if not errors else ";".join(errors))
    merged["wupin_prediction"] = converted
    merged["wupin_conversion_status"] = statuses

    scored_rows = []
    for _, row in merged.iterrows():
        ipa_score = score_pair(row["ipa_label"], row["ipa_prediction"])
        wupin_score = score_pair(row["wupin_label"], row["wupin_prediction"])
        scored_rows.append(
            {
                "sample_id": row["sample_id"],
                "variant": row["variant"],
                "source_split": row["source_split"],
                "page": row["page"],
                "row_index": row["row_index"],
                "alignment_status": row.get("alignment_status", ""),
                "candidate_id": row.get("candidate_id", ""),
                "ipa_label": row["ipa_label"],
                "ipa_prediction": row["ipa_prediction"],
                "wupin_label": row["wupin_label"],
                "wupin_prediction": row["wupin_prediction"],
                "ipa_label_status": row["ipa_label_status"],
                "wupin_conversion_status": row["wupin_conversion_status"],
                **{f"ipa_{key}": value for key, value in ipa_score.items()},
                **{f"wupin_{key}": value for key, value in wupin_score.items()},
            }
        )
    scored = pd.DataFrame(scored_rows)
    out_prefix = args.out_dir / "page_score"
    scored.to_csv(out_prefix.with_suffix(".row_score.tsv"), sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)

    summary_rows = []
    for split in ["train", "val", "test", "all"]:
        group = scored if split == "all" else scored[scored["source_split"] == split]
        row = {"split": split, "n": int(len(group))}
        row.update(summarize(group, "ipa"))
        row.update(summarize(group, "wupin"))
        summary_rows.append(row)
    summary = {
        "metric_scope": "page_pipeline",
        "prediction_mode": args.prediction_mode,
        "ipa_label_source": args.ipa_label_source,
        "eval_manifest": str(args.out_dir / "page_eval_manifest.tsv"),
        "predictions": str(args.out_dir / "page_predictions.tsv"),
        "rows": summary_rows,
    }
    out_prefix.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(summary_rows).to_csv(out_prefix.with_suffix(".summary.tsv"), sep="\t", index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and args.overwrite:
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pages = parse_page_list(args.page_list, args.start_pdf_page, args.end_pdf_page, args.limit_pages)
    gold = load_gold_rows(args)
    source_pages = {page + args.source_page_offset for page in pages}
    gold = gold[gold["page"].isin(source_pages)].copy()
    if gold.empty:
        raise SystemExit("no gold rows selected; check --page-list/--source-split/--eval-manifest")

    candidates = crop_page_candidates(args, pages)
    manifest = build_page_eval_manifest(args, gold, candidates)
    write_contact_sheet(args, manifest)
    if args.crop_only:
        print(f"crop-only complete: gold_rows={len(manifest)} candidates={len(candidates)}")
        print(f"wrote {args.out_dir / 'page_eval_manifest.tsv'}")
        print(f"wrote {args.out_dir / 'alignment_diagnostics.tsv'}")
        print(f"wrote {args.out_dir / 'qa_contactsheet.png'}")
        return

    predictions = predict(args, manifest)
    score_page_predictions(args, manifest, predictions)
    print(f"wrote {args.out_dir / 'page_predictions.tsv'}")
    print(f"wrote {args.out_dir / 'page_score.summary.tsv'}")


if __name__ == "__main__":
    main()
