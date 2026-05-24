"""Batch page OCR through an OpenAI-compatible VLM endpoint.

Requests are launched at a fixed interval and are not awaited before launching
the next page.  Each page writes its own raw response and extracted content so a
long run can be inspected or resumed.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import csv
import hashlib
import json
import mimetypes
import os
import re
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "ipa_ocr_work" / "eval" / "vlm_pages" / "post136_cropped_180dpi" / "page_manifest.tsv"
DEFAULT_PROMPT = PROJECT_ROOT / "ipa_ocr_work" / "eval" / "vlm_pages" / "prompt_structured_tone_ocr_zh.txt"
DEFAULT_OUT = PROJECT_ROOT / "ipa_ocr_work" / "eval" / "vlm_pages" / "qwen36plus_post136_batch"
DEFAULT_BASE_URL = "https://new-api.abrdns.com/v1"
DEFAULT_MODEL = "qwen3.6-plus-2026-04-02"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch page OCR requests at a fixed interval.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=os.getenv("OPENAI_VLM_MODEL", DEFAULT_MODEL))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--start-pdf-page", type=int, default=137)
    parser.add_argument("--end-pdf-page", type=int, default=0)
    parser.add_argument("--limit-pages", type=int, default=0)
    parser.add_argument("--launch-interval-seconds", type=float, default=2.0)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-workers", type=int, default=128)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def encode_data_uri(path: Path) -> tuple[str, dict[str, object]]:
    image_bytes = path.read_bytes()
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    data_uri = f"data:{mime};base64," + base64.b64encode(image_bytes).decode("ascii")
    return data_uri, {
        "image_bytes": len(image_bytes),
        "mime": mime,
        "sha256": hashlib.sha256(image_bytes).hexdigest(),
        "data_uri_chars": len(data_uri),
    }


def extract_jsonish(content: str) -> dict[str, Any] | None:
    stripped = content.strip()
    if stripped.startswith("{"):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", content, flags=re.S)
    for block in reversed(blocks):
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            continue
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def run_one(args: argparse.Namespace, prompt: str, item: dict[str, Any], api_key: str) -> dict[str, object]:
    pdf_page = int(item["pdf_page"])
    page_dir = args.out_dir / f"pdf{pdf_page:03d}_p{int(item['source_page']):03d}"
    page_dir.mkdir(parents=True, exist_ok=True)
    raw_path = page_dir / "raw_response.json_or_error.txt"
    done_path = page_dir / "done.json"
    if args.resume and done_path.exists():
        return {"pdf_page": pdf_page, "source_page": int(item["source_page"]), "status": "skip_done"}

    image_path = (args.manifest.parent / str(item["image"])).resolve()
    started = time.time()
    try:
        data_uri, image_meta = encode_data_uri(image_path)
        body = {
            "model": args.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            ],
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        }
        request_debug = {
            "endpoint": args.base_url.rstrip("/") + "/chat/completions",
            "model": args.model,
            "pdf_page": pdf_page,
            "source_page": int(item["source_page"]),
            "image": str(image_path),
            "image_meta": image_meta,
            "prompt_chars": len(prompt),
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        }
        (page_dir / "request_debug.json").write_text(
            json.dumps(request_debug, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = Request(
            args.base_url.rstrip("/") + "/chat/completions",
            data=payload,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=args.timeout_seconds) as response:
            raw = response.read().decode("utf-8", errors="replace")
            http_status = response.status
        raw_path.write_text(raw, encoding="utf-8")
        parsed_response = json.loads(raw)
        choice = parsed_response.get("choices", [{}])[0]
        content = str(choice.get("message", {}).get("content", ""))
        (page_dir / "assistant_content.txt").write_text(content, encoding="utf-8")
        extracted = extract_jsonish(content)
        rows_count = None
        if extracted is not None:
            (page_dir / "extracted_json.json").write_text(
                json.dumps(extracted, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            rows = extracted.get("rows") if isinstance(extracted, dict) else None
            rows_count = len(rows) if isinstance(rows, list) else None
        status = "ok"
        error = ""
        finish_reason = choice.get("finish_reason", "")
        usage = parsed_response.get("usage", {})
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raw_path.write_text(raw, encoding="utf-8")
        http_status = exc.code
        status = "http_error"
        error = raw[:1000]
        finish_reason = ""
        usage = {}
        rows_count = None
    except (URLError, TimeoutError, OSError, json.JSONDecodeError, Exception) as exc:
        raw_path.write_text(repr(exc), encoding="utf-8")
        http_status = ""
        status = "error"
        error = repr(exc)
        finish_reason = ""
        usage = {}
        rows_count = None

    elapsed = round(time.time() - started, 3)
    done = {
        "pdf_page": pdf_page,
        "source_page": int(item["source_page"]),
        "status": status,
        "http_status": http_status,
        "finish_reason": finish_reason,
        "rows_count": rows_count,
        "elapsed_seconds": elapsed,
        "usage": usage,
        "error": error,
        "page_dir": str(page_dir),
    }
    done_path.write_text(json.dumps(done, ensure_ascii=False, indent=2), encoding="utf-8")
    return done


def append_summary(path: Path, row: dict[str, object]) -> None:
    fields = [
        "pdf_page",
        "source_page",
        "status",
        "http_status",
        "finish_reason",
        "rows_count",
        "elapsed_seconds",
        "error",
        "page_dir",
    ]
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    api_key = args.api_key or os.getenv(args.api_key_env, "")
    if not api_key and not args.dry_run:
        raise SystemExit(f"Missing API key. Set {args.api_key_env} or pass --api-key.")
    prompt = args.prompt_file.read_text(encoding="utf-8")
    manifest = pd.read_csv(args.manifest, sep="\t", keep_default_na=False)
    manifest = manifest[manifest["pdf_page"].astype(int).ge(args.start_pdf_page)].copy()
    if args.end_pdf_page:
        manifest = manifest[manifest["pdf_page"].astype(int).le(args.end_pdf_page)].copy()
    if args.limit_pages:
        manifest = manifest.head(args.limit_pages)
    rows = manifest.to_dict("records")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_meta = {
        "manifest": str(args.manifest),
        "out_dir": str(args.out_dir),
        "prompt_file": str(args.prompt_file),
        "model": args.model,
        "base_url": args.base_url,
        "start_pdf_page": args.start_pdf_page,
        "end_pdf_page": args.end_pdf_page,
        "limit_pages": args.limit_pages,
        "launch_interval_seconds": args.launch_interval_seconds,
        "timeout_seconds": args.timeout_seconds,
        "max_tokens": args.max_tokens,
        "n_pages": len(rows),
    }
    (args.out_dir / "run_meta.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(run_meta, ensure_ascii=False, indent=2))

    if args.dry_run:
        for item in rows:
            print(f"dry-run pdf_page={item['pdf_page']} image={item['image']}")
        return

    summary_path = args.out_dir / "summary.tsv"
    futures: list[concurrent.futures.Future[dict[str, object]]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        for idx, item in enumerate(rows, start=1):
            pdf_page = int(item["pdf_page"])
            page_dir = args.out_dir / f"pdf{pdf_page:03d}_p{int(item['source_page']):03d}"
            if args.resume and (page_dir / "done.json").exists():
                row = {"pdf_page": pdf_page, "source_page": int(item["source_page"]), "status": "skip_done", "page_dir": str(page_dir)}
                append_summary(summary_path, row)
                print(f"[{idx}/{len(rows)}] skip_done pdf_page={pdf_page}", flush=True)
                continue
            future = executor.submit(run_one, args, prompt, item, api_key)
            futures.append(future)
            print(f"[{idx}/{len(rows)}] launched pdf_page={pdf_page}", flush=True)
            if idx < len(rows):
                time.sleep(args.launch_interval_seconds)

        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            append_summary(summary_path, row)
            print(
                f"completed pdf_page={row.get('pdf_page')} status={row.get('status')} "
                f"rows={row.get('rows_count')} elapsed={row.get('elapsed_seconds')}",
                flush=True,
            )


if __name__ == "__main__":
    main()
