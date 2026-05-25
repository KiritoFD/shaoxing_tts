"""Run full-page Shaoxing OCR through the local qwen-code CLI.

This uses qwen-code as a black-box multimodal client.  Each prompt starts with
an @image reference, so qwen-code's own attachment handling sends the page image
to the selected model.  The script then extracts the final CLI result, parses
rows, and can score with the same full-page evaluator used by the API runner.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from run_qwen_vl_page_ipa_ocr import DEFAULT_EVAL_MANIFEST, parse_page_list, score_pages
from wupin_ipa_convert import DEFAULT_MAP


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAGE_MANIFEST = PROJECT_ROOT / "ipa_ocr_work" / "eval" / "vlm_pages" / "test13_cropped_140dpi" / "page_manifest.tsv"
DEFAULT_OUT = PROJECT_ROOT / "ipa_ocr_work" / "eval" / "vlm_pages" / "qwen_code_page_ipa_ocr"
DEFAULT_MODEL = "qwen3.6-plus"


PROMPT_TEMPLATE = """{image_ref}
Read this scanned Shaoxing dialect dictionary page.
Return every dictionary entry from top to bottom, one entry per line.
Format each line exactly as:
Chinese headword<TAB>IPA phonetic transcription

Only include the phonetic transcription immediately after the headword. Do not include the Chinese explanation.
Use regular tone digits, not superscript/subscript digits: 11, 33, 231, 52.
Do not describe the image. Do not output Markdown. If unclear, write [?]."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call qwen-code on cropped page images.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_PAGE_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", default=os.getenv("QWEN_CODE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--auth-type", default="openai")
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--page-list", default="")
    parser.add_argument("--start-pdf-page", type=int, default=0)
    parser.add_argument("--end-pdf-page", type=int, default=0)
    parser.add_argument("--limit-pages", type=int, default=0)
    parser.add_argument("--eval-manifest", type=Path, default=DEFAULT_EVAL_MANIFEST)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--source-split", default="test")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def extract_first_json_array(text: str) -> list[dict[str, Any]]:
    start = text.find("[")
    if start < 0:
        raise ValueError("qwen output did not contain a JSON event array")
    decoder = json.JSONDecoder()
    events, _ = decoder.raw_decode(text[start:])
    if not isinstance(events, list):
        raise ValueError("qwen output JSON root is not an array")
    return events


def final_result_from_events(events: list[dict[str, Any]]) -> str:
    for event in reversed(events):
        if event.get("type") == "result":
            return str(event.get("result", ""))
    for event in reversed(events):
        message = event.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, list):
                texts = [str(part.get("text", "")) for part in content if isinstance(part, dict) and part.get("type") == "text"]
                if texts:
                    return "\n".join(texts)
    return ""


def extract_json_object(text: str) -> dict[str, Any]:
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


def normalize_tone_digits(text: object) -> str:
    table = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉", "01234567890123456789")
    return str(text or "").translate(table).strip()


def cjk_prefix(text: str) -> tuple[str, str]:
    chars: list[str] = []
    i = 0
    for ch in text.strip():
        code = ord(ch)
        if 0x3400 <= code <= 0x9FFF or 0xF900 <= code <= 0xFAFF or ch in "[]()（）【】":
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
        if not line or "\t" not in line and not re.match(r"^[\u3400-\u9fff\uf900-\ufaff]", line):
            continue
        if "\t" in line:
            head, rest = line.split("\t", 1)
        else:
            head, rest = cjk_prefix(line)
        head = head.strip()
        rest = rest.strip()
        if not head or not rest:
            continue
        phonetic: list[str] = []
        for ch in rest:
            code = ord(ch)
            if 0x3400 <= code <= 0x9FFF or 0xF900 <= code <= 0xFAFF:
                break
            phonetic.append(ch)
        ipa = normalize_tone_digits("".join(phonetic).strip(" \t:：;；。"))
        if not ipa:
            continue
        rows.append(
            {
                "row_index": len(rows),
                "headword": head,
                "ipa": ipa,
                "confidence": "",
                "notes": "parsed_from_plain_text",
            }
        )
    return {"rows": rows}


def run_qwen(args: argparse.Namespace, prompt: str) -> tuple[int, str, str]:
    env = os.environ.copy()
    api_key = args.api_key or env.get(args.api_key_env, "")
    if not api_key and not args.dry_run:
        raise SystemExit(f"Missing API key. Set {args.api_key_env} or pass --api-key.")
    if api_key:
        env[args.api_key_env] = api_key
    qwen_bin = shutil.which("qwen.cmd") or shutil.which("qwen") or shutil.which("qwen.ps1") or "qwen"
    command = [
        qwen_bin,
        "-m",
        args.model,
        "--auth-type",
        args.auth_type,
        "--openai-api-key",
        api_key,
        "--bare",
        "--output-format",
        "json",
        "-p",
        prompt,
    ]
    proc = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=240,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def read_done(raw_path: Path) -> set[int]:
    done: set[int] = set()
    if not raw_path.exists():
        return done
    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("status") == "ok":
                done.add(int(item["pdf_page"]))
    return done


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest, sep="\t", keep_default_na=False)
    if args.start_pdf_page:
        manifest = manifest[manifest["pdf_page"].astype(int).ge(args.start_pdf_page)].copy()
    if args.end_pdf_page:
        manifest = manifest[manifest["pdf_page"].astype(int).le(args.end_pdf_page)].copy()
    selected_pages = parse_page_list(args.page_list)
    if selected_pages:
        manifest = manifest[manifest["pdf_page"].astype(int).isin(selected_pages)].copy()
    if args.limit_pages:
        manifest = manifest.head(args.limit_pages)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / "qwen_code_page_raw.jsonl"
    rows_path = args.out_dir / "qwen_page_rows.tsv"
    if not args.resume:
        for path in (raw_path, rows_path, args.out_dir / "page_score.row_score.tsv", args.out_dir / "page_score.summary.tsv"):
            if path.exists():
                path.unlink()
    done = read_done(raw_path) if args.resume else set()

    flat_rows: list[dict[str, object]] = []
    for item in manifest.to_dict("records"):
        pdf_page = int(item["pdf_page"])
        if pdf_page in done:
            print(f"skip existing pdf_page={pdf_page}")
            continue
        image_path = (args.manifest.parent / str(item["image"])).resolve()
        image_ref = "@" + image_path.relative_to(PROJECT_ROOT).as_posix()
        prompt = PROMPT_TEMPLATE.format(image_ref=image_ref)
        if args.dry_run:
            print(f"dry-run pdf_page={pdf_page} {image_ref}")
            continue

        status = "ok"
        error = ""
        result_text = ""
        parsed: dict[str, Any] = {}
        stdout = ""
        stderr = ""
        try:
            returncode, stdout, stderr = run_qwen(args, prompt)
            if returncode != 0:
                raise RuntimeError(f"qwen exited with code {returncode}: {stderr.strip()}")
            events = extract_first_json_array(stdout)
            result_text = final_result_from_events(events)
            try:
                parsed = extract_json_object(result_text)
            except json.JSONDecodeError:
                parsed = parse_plain_lines(result_text)
                if not parsed.get("rows"):
                    raise
        except Exception as exc:
            status = "error"
            error = repr(exc)
            print(f"failed pdf_page={pdf_page}: {error}", file=sys.stderr)

        record = {
            "status": status,
            "pdf_page": pdf_page,
            "source_page": int(item["source_page"]),
            "image": str(item["image"]),
            "model": args.model,
            "raw_text": result_text,
            "parsed": parsed,
            "error": error,
            "stderr": stderr.strip(),
        }
        with raw_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if status == "ok":
            rows = parsed.get("rows", [])
            if isinstance(rows, list):
                for ordinal, row in enumerate(rows):
                    if not isinstance(row, dict):
                        continue
                    flat_rows.append(
                        {
                            "pdf_page": pdf_page,
                            "source_page": int(item["source_page"]),
                            "model": args.model,
                            "row_index": row.get("row_index", ordinal),
                            "headword": row.get("headword", ""),
                            "ipa": normalize_tone_digits(row.get("ipa", "")),
                            "wupin": "",
                            "gloss": "",
                            "confidence": row.get("confidence", ""),
                            "notes": row.get("notes", ""),
                        }
                    )
        print(f"done pdf_page={pdf_page} status={status}")
        time.sleep(args.sleep_seconds)

    if flat_rows:
        write_header = not rows_path.exists() or not args.resume
        with rows_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()), delimiter="\t")
            if write_header:
                writer.writeheader()
            writer.writerows(flat_rows)
    print(f"wrote {raw_path}")
    if flat_rows:
        print(f"wrote {rows_path}")
    if args.score and rows_path.exists():
        summary = score_pages(args, rows_path)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"wrote {args.out_dir / 'page_score.summary.tsv'}")


if __name__ == "__main__":
    main()
