"""Run a Qwen/DashScope-compatible VLM page OCR pass.

Input is the page manifest produced by prepare_vlm_pdf_pages.py.  The script
sends one cropped page image per request and asks the model to return JSON rows
with headword, IPA transcription, optional Wu-pinyin, and notes.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from score_ocr_experiment import edit_distance, normalize, score_pair, summarize
from wupin_ipa_convert import DEFAULT_MAP, load_mapping, wupin_to_ipa


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "ipa_ocr_work" / "eval" / "vlm_pages" / "post136_cropped" / "page_manifest.tsv"
DEFAULT_EVAL_MANIFEST = (
    PROJECT_ROOT
    / "ipa_ocr_work"
    / "dataset"
    / "shaoxing_pdf136_clean"
    / "ocr_selected_all"
    / "eval_manifest.tsv"
)
DEFAULT_OUT = PROJECT_ROOT / "ipa_ocr_work" / "eval" / "vlm_pages" / "qwen_page_ipa_ocr"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.6-plus"


SYSTEM_PROMPT = """You are an OCR transcriber for scanned Shaoxing Wu dialect dictionary pages.
Read only the visible page image. Extract dictionary rows in top-to-bottom order.
Return valid JSON only, with this schema:
{
  "rows": [
    {
      "row_index": 0,
      "headword": "Chinese headword",
      "ipa": "IPA transcription with tone digits, no spaces unless printed",
      "wupin": "Wu-pinyin if visible or inferable, otherwise empty",
      "gloss": "short Chinese gloss if visible, otherwise empty",
      "confidence": 0.0,
      "notes": "brief uncertainty notes"
    }
  ]
}
Use IPA symbols, not Latin approximations, for the ipa field. Preserve tone digits exactly.
If a row is unclear, include it with lower confidence instead of guessing silently."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call a Qwen-compatible VLM one page at a time.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", default=os.getenv("QWEN_VL_MODEL", DEFAULT_MODEL))
    parser.add_argument("--base-url", default=os.getenv("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--start-pdf-page", type=int, default=0)
    parser.add_argument("--end-pdf-page", type=int, default=0)
    parser.add_argument("--page-list", default="", help="PDF pages, e.g. 7,17,27 or 7-9. Empty means all selected pages.")
    parser.add_argument("--limit-pages", type=int, default=0)
    parser.add_argument("--eval-manifest", type=Path, default=DEFAULT_EVAL_MANIFEST)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--source-split", default="test")
    parser.add_argument("--score", action="store_true", help="Score against --eval-manifest after writing rows.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--high-resolution", action="store_true", default=True)
    parser.add_argument("--no-high-resolution", dest="high_resolution", action="store_false")
    parser.add_argument("--max-pixels", type=int, default=0, help="Optional model-side image pixel budget.")
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
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


def encode_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def response_text(payload: dict[str, Any]) -> str:
    choice = payload.get("choices", [{}])[0]
    message = choice.get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return str(content)


def extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            return json.loads(stripped[start : end + 1])
        raise


def post_chat(args: argparse.Namespace, image_path: Path, page_meta: dict[str, object]) -> dict[str, Any]:
    endpoint = args.base_url.rstrip("/") + "/chat/completions"
    prompt = (
        f"Transcribe this page. pdf_page={page_meta.get('pdf_page')}, "
        f"source_page={page_meta.get('source_page')}. Output JSON only."
    )
    body = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": encode_data_url(image_path)}},
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    if args.high_resolution:
        body["vl_high_resolution_images"] = True
    if args.max_pixels > 0:
        body["max_pixels"] = args.max_pixels
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = Request(
        endpoint,
        data=data,
        headers={
            "Authorization": f"Bearer {args.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


def read_done(jsonl_path: Path) -> set[int]:
    done: set[int] = set()
    if not jsonl_path.exists():
        return done
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "pdf_page" in row and row.get("status") == "ok":
                done.add(int(row["pdf_page"]))
    return done


def ipa_norm(text: object) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", str(text or "").strip()))


def headword_norm(text: object) -> str:
    return re.sub(r"\s+", "", str(text or "").strip())


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
            gold_head = headword_norm(gold_rows[i - 1].get("hanzi", ""))
            pred_head = headword_norm(pred_rows[j - 1].get("headword", ""))
            denom = max(1, max(len(gold_head), len(pred_head)))
            match_cost = min(1.0, edit_distance(pred_head, gold_head) / denom)
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


def score_pages(args: argparse.Namespace, rows_tsv: Path) -> dict[str, Any]:
    pred_df = pd.read_csv(rows_tsv, sep="\t", keep_default_na=False)
    gold = pd.read_csv(args.eval_manifest, sep="\t", keep_default_na=False)
    if args.source_split != "all":
        gold = gold[gold["source_split"].eq(args.source_split)].copy()

    pages = set(pd.read_csv(args.manifest, sep="\t", keep_default_na=False)["pdf_page"].astype(int))
    selected_pages = parse_page_list(args.page_list)
    if selected_pages:
        pages &= selected_pages
    if args.start_pdf_page:
        pages = {page for page in pages if page >= args.start_pdf_page}
    if args.end_pdf_page:
        pages = {page for page in pages if page <= args.end_pdf_page}
    if args.limit_pages:
        pages = set(sorted(pages)[: args.limit_pages])
    gold = gold[gold["pdf_page"].astype(int).isin(pages)].copy()
    mapping = load_mapping(args.mapping)

    scored: list[dict[str, object]] = []
    for pdf_page, gold_page in gold.groupby(gold["pdf_page"].astype(int), sort=True):
        pred_page = pred_df[pred_df["pdf_page"].astype(int).eq(pdf_page)].copy()
        gold_records = gold_page.sort_values(["row_index", "sample_id"]).to_dict("records")
        pred_records = pred_page.sort_values(["row_index"]).to_dict("records")
        for gold_idx, pred_idx in align_page(gold_records, pred_records):
            if gold_idx is None:
                pred = pred_records[pred_idx] if pred_idx is not None else {}
                pred_ipa = ipa_norm(pred.get("ipa", ""))
                scored.append(
                    {
                        "pdf_page": pdf_page,
                        "sample_id": "",
                        "hanzi": "",
                        "gold_ipa": "",
                        "pred_headword": pred.get("headword", ""),
                        "pred_ipa": pred_ipa,
                        "alignment_status": "extra_prediction",
                        "ipa_exact": 0,
                        "ipa_edit_distance": len(pred_ipa),
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
            pred_ipa = ipa_norm(pred.get("ipa", "")) if pred else ""
            ipa_score = score_pair(gold_ipa, pred_ipa)
            scored.append(
                {
                    "pdf_page": pdf_page,
                    "sample_id": gold_row.get("sample_id", ""),
                    "hanzi": gold_row.get("hanzi", ""),
                    "gold_wupin": gold_row.get("wupin", ""),
                    "gold_ipa": gold_ipa,
                    "pred_headword": pred.get("headword", "") if pred else "",
                    "pred_ipa": pred_ipa,
                    "alignment_status": "matched" if pred else "missing_prediction",
                    **{f"ipa_{key}": value for key, value in ipa_score.items()},
                }
            )

    scored_df = pd.DataFrame(scored)
    score_path = args.out_dir / "page_score.row_score.tsv"
    scored_df.to_csv(score_path, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)
    gold_scored = scored_df[scored_df["sample_id"].astype(str).ne("")].copy()
    row = {"split": args.source_split, "n": int(len(gold_scored))}
    row.update(summarize(gold_scored, "ipa"))
    summary = {
        "metric_scope": "full_page_qwen_api",
        "model": args.model,
        "page_manifest": str(args.manifest),
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
    args.api_key = args.api_key or os.getenv(args.api_key_env, "")
    if not args.api_key and not args.dry_run:
        raise SystemExit(f"Missing API key. Set {args.api_key_env} or pass --api-key.")

    manifest = pd.read_csv(args.manifest, sep="\t", keep_default_na=False)
    if args.start_pdf_page:
        manifest = manifest[pd.to_numeric(manifest["pdf_page"], errors="coerce") >= args.start_pdf_page]
    if args.end_pdf_page:
        manifest = manifest[pd.to_numeric(manifest["pdf_page"], errors="coerce") <= args.end_pdf_page]
    selected_pages = parse_page_list(args.page_list)
    if selected_pages:
        manifest = manifest[manifest["pdf_page"].astype(int).isin(selected_pages)].copy()
    if args.limit_pages:
        manifest = manifest.head(args.limit_pages)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_jsonl = args.out_dir / "qwen_page_raw.jsonl"
    rows_tsv = args.out_dir / "qwen_page_rows.tsv"
    if not args.resume:
        for path in (raw_jsonl, rows_tsv, args.out_dir / "page_score.row_score.tsv", args.out_dir / "page_score.summary.tsv"):
            if path.exists():
                path.unlink()
    done = read_done(raw_jsonl) if args.resume else set()

    flat_rows: list[dict[str, object]] = []
    for item in manifest.to_dict("records"):
        pdf_page = int(item["pdf_page"])
        if pdf_page in done:
            print(f"skip existing pdf_page={pdf_page}")
            continue
        image_path = args.manifest.parent / str(item["image"])
        if args.dry_run:
            print(f"dry-run pdf_page={pdf_page} image={image_path}")
            continue

        last_error = ""
        result: dict[str, Any] | None = None
        for attempt in range(1, args.retries + 1):
            try:
                payload = post_chat(args, image_path, item)
                text = response_text(payload)
                parsed = extract_json(text)
                result = {
                    "status": "ok",
                    "pdf_page": pdf_page,
                    "source_page": int(item["source_page"]),
                    "image": str(item["image"]),
                    "model": args.model,
                    "raw_text": text,
                    "parsed": parsed,
                }
                break
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
                last_error = repr(exc)
                print(f"attempt {attempt}/{args.retries} failed pdf_page={pdf_page}: {last_error}")
                time.sleep(min(10.0, args.sleep_seconds * attempt * 2))

        if result is None:
            result = {
                "status": "error",
                "pdf_page": pdf_page,
                "source_page": int(item["source_page"]),
                "image": str(item["image"]),
                "model": args.model,
                "error": last_error,
            }

        with raw_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        if result["status"] == "ok":
            rows = result.get("parsed", {}).get("rows", [])
            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    flat_rows.append(
                        {
                            "pdf_page": result["pdf_page"],
                            "source_page": result["source_page"],
                            "model": args.model,
                            "row_index": row.get("row_index", ""),
                            "headword": row.get("headword", ""),
                            "ipa": row.get("ipa", ""),
                            "wupin": row.get("wupin", ""),
                            "gloss": row.get("gloss", ""),
                            "confidence": row.get("confidence", ""),
                            "notes": row.get("notes", ""),
                        }
                    )
        print(f"done pdf_page={pdf_page} status={result['status']}")
        time.sleep(args.sleep_seconds)

    if flat_rows:
        write_header = not rows_tsv.exists() or not args.resume
        with rows_tsv.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()), delimiter="\t")
            if write_header:
                writer.writeheader()
            writer.writerows(flat_rows)
    print(f"wrote {raw_jsonl}")
    if flat_rows:
        print(f"wrote {rows_tsv}")
    if args.score and rows_tsv.exists():
        summary = score_pages(args, rows_tsv)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"wrote {args.out_dir / 'page_score.summary.tsv'}")


if __name__ == "__main__":
    main()
