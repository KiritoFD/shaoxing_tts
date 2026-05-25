"""Evaluate a local Qwen2-VL OCR model on full Shaoxing PDF pages.

This is an end-to-end page test: one cropped page image goes into the VLM and
the model must emit ordered dictionary rows.  Scoring aligns predicted rows to
gold rows by headword, then scores IPA strings.  Missing rows count as errors.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from score_ocr_experiment import edit_distance, normalize, score_pair, summarize
from wupin_ipa_convert import DEFAULT_MAP, load_mapping, wupin_to_ipa


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = (
    PROJECT_ROOT
    / "ipa_ocr_work"
    / "models"
    / "modelscope_cache"
    / "prithivMLmods"
    / "Qwen2-VL-OCR-2B-Instruct"
)
DEFAULT_PAGE_MANIFEST = PROJECT_ROOT / "ipa_ocr_work" / "eval" / "vlm_pages" / "test13_cropped_140dpi" / "page_manifest.tsv"
DEFAULT_EVAL_MANIFEST = (
    PROJECT_ROOT
    / "ipa_ocr_work"
    / "dataset"
    / "shaoxing_pdf136_clean"
    / "ocr_selected_all"
    / "eval_manifest.tsv"
)
DEFAULT_OUT = PROJECT_ROOT / "ipa_ocr_work" / "eval" / "vlm_pages" / "qwen2vl_ocr_2b_test13"


JSON_PROMPT = """识别这张绍兴方言词汇页。只输出词条行，不要输出页眉、页码、章节标题。
每一行按页面从上到下顺序输出 JSON，不要 Markdown：
{"rows":[{"row_index":0,"headword":"汉字词头","ipa":"页面上紧跟词头后的音标，使用 IPA 符号和声调数字","gloss":"释义，可空","confidence":0.0,"notes":"看不清处说明"}]}
要求：
1. ipa 字段只写音标/声调数字，不要汉字释义。
2. 看不清的音标用 [?]，不要脑补。
3. 保留 11、33、52、335、231 等声调数字。
4. 如果一行有括号或多个读音，只记录主读音，备注写在 notes。"""

HEADWORD_IPA_PROMPT = """Read this scanned dictionary page.
Return every dictionary entry from top to bottom, one entry per line.
Format each line exactly as:
Chinese headword<TAB>phonetic transcription
Only include the phonetic transcription immediately after the headword. Do not include the Chinese explanation.
Do not describe the image. Do not output JSON. If unclear, write [?]."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local Qwen2-VL OCR on page images and score full-page output.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--page-manifest", type=Path, default=DEFAULT_PAGE_MANIFEST)
    parser.add_argument("--eval-manifest", type=Path, default=DEFAULT_EVAL_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--source-split", default="test")
    parser.add_argument("--page-list", default="", help="PDF pages, e.g. 7,17,27. Empty means all pages in manifest.")
    parser.add_argument("--limit-pages", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-long-side", type=int, default=1400)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--prompt-mode", choices=["headword-ipa", "json"], default="headword-ipa")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--load-dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--score-only", action="store_true")
    return parser.parse_args()


def parse_page_list(text: str) -> set[int]:
    if not text.strip():
        return set()
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
    return set(pages)


def resize_image(path: Path, max_long_side: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if max_long_side <= 0:
        return image
    long_side = max(image.size)
    if long_side <= max_long_side:
        return image
    scale = max_long_side / long_side
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


def cjk_prefix(text: str) -> tuple[str, str]:
    chars = []
    i = 0
    for ch in text.strip():
        code = ord(ch)
        if (
            0x3400 <= code <= 0x9FFF
            or 0xF900 <= code <= 0xFAFF
            or ch in "[]［］（）()《》"
        ):
            chars.append(ch)
            i += 1
        else:
            break
    return "".join(chars).strip(), text.strip()[i:].strip()


def parse_plain_lines(text: str) -> dict[str, Any]:
    rows: list[dict[str, object]] = []
    cleaned = text.replace("<|im_end|>", "").strip()
    for line in cleaned.splitlines():
        line = line.strip().strip("-•")
        if not line:
            continue
        if "\t" in line:
            head, rest = line.split("\t", 1)
        else:
            head, rest = cjk_prefix(line)
        head = head.strip()
        rest = rest.strip()
        if not head or not rest:
            continue
        # Keep only the leading non-CJK phonetic field before the Chinese gloss.
        phonetic = []
        for ch in rest:
            code = ord(ch)
            if 0x3400 <= code <= 0x9FFF or 0xF900 <= code <= 0xFAFF:
                break
            phonetic.append(ch)
        ipa = "".join(phonetic).strip(" \t:：,，.;。")
        if not ipa:
            continue
        rows.append(
            {
                "row_index": len(rows),
                "headword": head,
                "ipa": ipa,
                "gloss": "",
                "confidence": "",
                "notes": "parsed_from_plain_text",
            }
        )
    return {"rows": rows}


def ipa_norm(text: object) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", str(text or "").strip()))


def headword_norm(text: object) -> str:
    return re.sub(r"\s+", "", str(text or "").strip())


def load_model(args: argparse.Namespace):
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.load_dtype]
    if args.device == "cpu":
        device_map = "cpu"
    elif args.device == "cuda":
        device_map = {"": 0}
    else:
        device_map = "auto"
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    return model, processor


def generate_page(model, processor, image: Image.Image, args: argparse.Namespace) -> str:
    prompt = JSON_PROMPT if args.prompt_mode == "json" else HEADWORD_IPA_PROMPT
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature if args.temperature > 0 else None,
        )
    generated_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(generated_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def read_raw(raw_path: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    if not raw_path.exists():
        return out
    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            out[int(row["pdf_page"])] = row
    return out


def flatten_predictions(raw_rows: dict[int, dict[str, Any]], out_path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for pdf_page, raw in sorted(raw_rows.items()):
        parsed = raw.get("parsed") if raw.get("status") == "ok" else {}
        pred_rows = parsed.get("rows", []) if isinstance(parsed, dict) else []
        if not isinstance(pred_rows, list):
            pred_rows = []
        for ordinal, row in enumerate(pred_rows):
            if not isinstance(row, dict):
                continue
            rows.append(
                {
                    "pdf_page": pdf_page,
                    "source_page": raw.get("source_page", ""),
                    "pred_ordinal": ordinal,
                    "pred_row_index": row.get("row_index", ""),
                    "pred_headword": row.get("headword", ""),
                    "pred_ipa": row.get("ipa", ""),
                    "pred_gloss": row.get("gloss", ""),
                    "confidence": row.get("confidence", ""),
                    "notes": row.get("notes", ""),
                    "status": raw.get("status", ""),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(out_path, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)
    return df


def align_page(gold_rows: list[dict[str, Any]], pred_rows: list[dict[str, Any]]) -> list[tuple[int | None, int | None]]:
    n, m = len(gold_rows), len(pred_rows)
    gap = 1.0
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple[int, int] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i * gap
        back[i][0] = (i - 1, 0)
    for j in range(1, m + 1):
        dp[0][j] = j * gap
        back[0][j] = (0, j - 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            g = headword_norm(gold_rows[i - 1].get("hanzi", ""))
            p = headword_norm(pred_rows[j - 1].get("pred_headword", ""))
            denom = max(1, max(len(g), len(p)))
            match_cost = min(1.0, edit_distance(p, g) / denom)
            options = [
                (dp[i - 1][j - 1] + match_cost, (i - 1, j - 1)),
                (dp[i - 1][j] + gap, (i - 1, j)),
                (dp[i][j - 1] + gap, (i, j - 1)),
            ]
            dp[i][j], back[i][j] = min(options, key=lambda item: item[0])
    pairs: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        prev = back[i][j]
        if prev is None:
            break
        pi, pj = prev
        if pi == i - 1 and pj == j - 1:
            pairs.append((i - 1, j - 1))
        elif pi == i - 1 and pj == j:
            pairs.append((i - 1, None))
        else:
            pairs.append((None, j - 1))
        i, j = pi, pj
    return list(reversed(pairs))


def score_predictions(args: argparse.Namespace, pred_df: pd.DataFrame) -> dict[str, Any]:
    gold = pd.read_csv(args.eval_manifest, sep="\t", keep_default_na=False)
    if args.source_split != "all":
        gold = gold[gold["source_split"].eq(args.source_split)].copy()
    pages = set(pd.read_csv(args.page_manifest, sep="\t", keep_default_na=False)["pdf_page"].astype(int))
    selected_pages = parse_page_list(args.page_list)
    if selected_pages:
        pages &= selected_pages
    if args.limit_pages:
        pages = set(sorted(pages)[: args.limit_pages])
    gold = gold[gold["pdf_page"].astype(int).isin(pages)].copy()
    mapping = load_mapping(args.mapping)

    scored: list[dict[str, object]] = []
    for pdf_page, gold_page in gold.groupby(gold["pdf_page"].astype(int), sort=True):
        pred_page = pred_df[pred_df["pdf_page"].astype(int).eq(pdf_page)].copy()
        gold_records = gold_page.sort_values(["row_index", "sample_id"]).to_dict("records")
        pred_records = pred_page.sort_values(["pred_ordinal"]).to_dict("records")
        for gold_idx, pred_idx in align_page(gold_records, pred_records):
            if gold_idx is None:
                pred = pred_records[pred_idx] if pred_idx is not None else {}
                scored.append(
                    {
                        "pdf_page": pdf_page,
                        "source_page": pred.get("source_page", ""),
                        "sample_id": "",
                        "hanzi": "",
                        "gold_ipa": "",
                        "pred_headword": pred.get("pred_headword", ""),
                        "pred_ipa": ipa_norm(pred.get("pred_ipa", "")),
                        "alignment_status": "extra_prediction",
                        "ipa_exact": 0,
                        "ipa_edit_distance": len(ipa_norm(pred.get("pred_ipa", ""))),
                        "ipa_label_len": 0,
                        "ipa_cer": 0.0,
                    }
                )
                continue
            gold_row = gold_records[gold_idx]
            pred = pred_records[pred_idx] if pred_idx is not None else {}
            wupin = normalize(gold_row.get("wupin", ""))
            gold_ipa, errors = wupin_to_ipa(wupin, mapping, "digits")
            gold_ipa = ipa_norm(gold_ipa if not errors else gold_row.get("label", ""))
            pred_ipa = ipa_norm(pred.get("pred_ipa", "")) if pred else ""
            ipa_score = score_pair(gold_ipa, pred_ipa)
            scored.append(
                {
                    "pdf_page": pdf_page,
                    "source_page": gold_row.get("page", ""),
                    "sample_id": gold_row.get("sample_id", ""),
                    "hanzi": gold_row.get("hanzi", ""),
                    "gold_wupin": gold_row.get("wupin", ""),
                    "gold_ipa": gold_ipa,
                    "pred_headword": pred.get("pred_headword", "") if pred else "",
                    "pred_ipa": pred_ipa,
                    "alignment_status": "matched" if pred else "missing_prediction",
                    **{f"ipa_{key}": value for key, value in ipa_score.items()},
                }
            )
    scored_df = pd.DataFrame(scored)
    scored_df.to_csv(args.out_dir / "page_score.row_score.tsv", sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)
    gold_scored = scored_df[scored_df["sample_id"].astype(str).ne("")].copy()
    row = {"split": args.source_split, "n": int(len(gold_scored))}
    row.update(summarize(gold_scored, "ipa"))
    summary = {
        "metric_scope": "full_page_vlm",
        "model": str(args.model),
        "page_manifest": str(args.page_manifest),
        "eval_manifest": str(args.eval_manifest),
        "pages": sorted(pages),
        "rows": [row],
        "extra_predictions": int(scored_df["alignment_status"].eq("extra_prediction").sum()),
        "missing_predictions": int(scored_df["alignment_status"].eq("missing_prediction").sum()),
    }
    (args.out_dir / "page_score.summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([row]).to_csv(args.out_dir / "page_score.summary.tsv", sep="\t", index=False)
    return summary


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / "raw_pages.jsonl"
    pred_path = args.out_dir / "page_predictions.tsv"

    manifest = pd.read_csv(args.page_manifest, sep="\t", keep_default_na=False)
    selected_pages = parse_page_list(args.page_list)
    if selected_pages:
        manifest = manifest[manifest["pdf_page"].astype(int).isin(selected_pages)].copy()
    if args.limit_pages:
        manifest = manifest.head(args.limit_pages)

    raw_rows = read_raw(raw_path) if args.resume else {}
    if not args.score_only:
        model, processor = load_model(args)
        for row in manifest.to_dict("records"):
            pdf_page = int(row["pdf_page"])
            if args.resume and pdf_page in raw_rows and raw_rows[pdf_page].get("status") == "ok":
                print(f"skip existing pdf_page={pdf_page}")
                continue
            image_path = args.page_manifest.parent / str(row["image"])
            image = resize_image(image_path, args.max_long_side)
            try:
                text = generate_page(model, processor, image, args)
                if args.prompt_mode == "json":
                    parsed = extract_json(text)
                else:
                    parsed = parse_plain_lines(text)
                record = {
                    "status": "ok",
                    "pdf_page": pdf_page,
                    "source_page": int(row["source_page"]),
                    "image": str(row["image"]),
                    "raw_text": text,
                    "parsed": parsed,
                }
            except Exception as exc:
                record = {
                    "status": "error",
                    "pdf_page": pdf_page,
                    "source_page": int(row["source_page"]),
                    "image": str(row["image"]),
                    "error": repr(exc),
                }
            with raw_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            raw_rows[pdf_page] = record
            print(f"done pdf_page={pdf_page} status={record['status']}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    pred_df = flatten_predictions(read_raw(raw_path), pred_path)
    summary = score_predictions(args, pred_df)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {pred_path}")
    print(f"wrote {args.out_dir / 'page_score.summary.tsv'}")


if __name__ == "__main__":
    main()
