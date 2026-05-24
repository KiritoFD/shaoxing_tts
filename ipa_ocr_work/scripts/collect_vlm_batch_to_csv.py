"""Collect per-page VLM OCR outputs into flat CSV/TSV files.

Reads a batch directory produced by batch_openai_vlm_pages.py.  It prefers
extracted_json.json, but can also recover JSON from assistant_content.txt or the
raw OpenAI-compatible response.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BATCH_DIR = PROJECT_ROOT / "ipa_ocr_work" / "eval" / "vlm_pages" / "qwen36plus_post136_batch_20260525"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flatten VLM page OCR JSON into CSV/TSV.")
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH_DIR)
    parser.add_argument("--out-prefix", type=Path, default=None)
    parser.add_argument("--write-utf8-sig-csv", action="store_true", default=True)
    return parser.parse_args()


def tone_digits(text: object) -> str:
    table = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉", "01234567890123456789")
    return str(text or "").translate(table)


def parse_json_from_text(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.S)
    for block in reversed(blocks):
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            continue
    # Some qwen replies have a <think> prelude and then "the JSON.{...}".
    marker = stripped.rfind('"rows"')
    if marker >= 0:
        start = stripped.rfind("{", 0, marker)
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(stripped[start : end + 1])
            except json.JSONDecodeError:
                pass
    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    for match in re.finditer(r"\{", stripped):
        try:
            obj, _ = decoder.raw_decode(stripped[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("rows"), list):
            candidates.append(obj)
    if candidates:
        return candidates[-1]
    return None


def read_page_json(page_dir: Path) -> tuple[dict[str, Any] | None, str]:
    extracted = page_dir / "extracted_json.json"
    if extracted.exists():
        try:
            return json.loads(extracted.read_text(encoding="utf-8")), "extracted_json"
        except json.JSONDecodeError:
            pass
    assistant = page_dir / "assistant_content.txt"
    if assistant.exists():
        parsed = parse_json_from_text(assistant.read_text(encoding="utf-8", errors="replace"))
        if parsed is not None:
            return parsed, "assistant_content"
    raw = page_dir / "raw_response.json_or_error.txt"
    if raw.exists():
        try:
            response = json.loads(raw.read_text(encoding="utf-8", errors="replace"))
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            parsed = parse_json_from_text(str(content))
            if parsed is not None:
                return parsed, "raw_response_content"
        except Exception:
            return None, "raw_unparseable"
    return None, "missing"


def read_done(page_dir: Path) -> dict[str, Any]:
    done = page_dir / "done.json"
    if not done.exists():
        return {}
    try:
        return json.loads(done.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def collect(batch_dir: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    pages: list[dict[str, object]] = []
    for page_dir in sorted(batch_dir.glob("pdf*_p*")):
        if not page_dir.is_dir():
            continue
        done = read_done(page_dir)
        parsed, source = read_page_json(page_dir)
        rows_obj = parsed.get("rows", []) if isinstance(parsed, dict) else []
        if not isinstance(rows_obj, list):
            rows_obj = []
        page_record = {
            "pdf_page": done.get("pdf_page", ""),
            "source_page": done.get("source_page", ""),
            "status": done.get("status", "pending"),
            "http_status": done.get("http_status", ""),
            "finish_reason": done.get("finish_reason", ""),
            "rows_count_done": done.get("rows_count", ""),
            "rows_count_collected": len(rows_obj),
            "elapsed_seconds": done.get("elapsed_seconds", ""),
            "json_source": source,
            "error": done.get("error", ""),
            "page_dir": str(page_dir),
        }
        pages.append(page_record)
        for ordinal, row in enumerate(rows_obj):
            if not isinstance(row, dict):
                continue
            syllables = row.get("syllables", [])
            rows.append(
                {
                    "pdf_page": page_record["pdf_page"],
                    "source_page": page_record["source_page"],
                    "page_status": page_record["status"],
                    "http_status": page_record["http_status"],
                    "json_source": source,
                    "row_ordinal": ordinal,
                    "row_index": row.get("row_index", ordinal),
                    "headword": row.get("headword", ""),
                    "ipa_raw": row.get("ipa_raw", ""),
                    "ipa": tone_digits(row.get("ipa", "")),
                    "confidence": row.get("confidence", ""),
                    "notes": row.get("notes", ""),
                    "syllables_json": json.dumps(syllables, ensure_ascii=False),
                    "page_dir": str(page_dir),
                }
            )
    return rows, pages


def write_table(path: Path, rows: list[dict[str, object]], delimiter: str, encoding: str) -> None:
    if not rows:
        path.write_text("", encoding=encoding)
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding=encoding, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter=delimiter, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    out_prefix = args.out_prefix or (args.batch_dir / "collected_vlm_ocr")
    row_records, page_records = collect(args.batch_dir)

    write_table(out_prefix.with_suffix(".rows.tsv"), row_records, "\t", "utf-8")
    write_table(out_prefix.with_suffix(".pages.tsv"), page_records, "\t", "utf-8")
    write_table(out_prefix.with_suffix(".rows.csv"), row_records, ",", "utf-8-sig")
    write_table(out_prefix.with_suffix(".pages.csv"), page_records, ",", "utf-8-sig")

    summary = {
        "batch_dir": str(args.batch_dir),
        "pages_seen": len(page_records),
        "pages_with_rows": sum(1 for page in page_records if int(page.get("rows_count_collected") or 0) > 0),
        "rows_collected": len(row_records),
        "status_counts": {},
        "json_source_counts": {},
    }
    for page in page_records:
        status = str(page.get("status", ""))
        source = str(page.get("json_source", ""))
        summary["status_counts"][status] = summary["status_counts"].get(status, 0) + 1
        summary["json_source_counts"][source] = summary["json_source_counts"].get(source, 0) + 1
    out_prefix.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {out_prefix.with_suffix('.rows.csv')}")
    print(f"wrote {out_prefix.with_suffix('.pages.csv')}")


if __name__ == "__main__":
    main()
