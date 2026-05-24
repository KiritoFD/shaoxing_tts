"""Merge collected VLM OCR CSVs from multiple batch attempts.

Later batch directories take precedence for a page when they have collected
rows, which lets a retry batch replace an earlier failed page.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BATCHES = [
    PROJECT_ROOT / "ipa_ocr_work" / "eval" / "vlm_pages" / "qwen36plus_post136_batch_20260525",
    PROJECT_ROOT / "ipa_ocr_work" / "eval" / "vlm_pages" / "qwen36plus_post136_retry_failed_jpeg1600_20260525",
]
DEFAULT_OUT_PREFIX = PROJECT_ROOT / "ipa_ocr_work" / "eval" / "vlm_pages" / "qwen36plus_post136_current_best"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge collected VLM OCR rows from batch dirs.")
    parser.add_argument("--batch-dir", type=Path, action="append", default=None)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, keep_default_na=False)


def main() -> None:
    args = parse_args()
    batch_dirs = args.batch_dir or DEFAULT_BATCHES

    selected_rows: dict[int, pd.DataFrame] = {}
    selected_pages: dict[int, dict[str, object]] = {}
    attempts: list[dict[str, object]] = []
    for priority, batch_dir in enumerate(batch_dirs):
        rows = read_csv(batch_dir / "collected_vlm_ocr.rows.csv")
        pages = read_csv(batch_dir / "collected_vlm_ocr.pages.csv")
        if not pages.empty:
            for _, page in pages.iterrows():
                pdf_page = int(page["pdf_page"]) if str(page.get("pdf_page", "")).isdigit() else None
                if pdf_page is None:
                    continue
                attempts.append(
                    {
                        "pdf_page": pdf_page,
                        "batch_dir": str(batch_dir),
                        "priority": priority,
                        "status": page.get("status", ""),
                        "http_status": page.get("http_status", ""),
                        "rows_count_collected": int(page.get("rows_count_collected") or 0),
                        "json_source": page.get("json_source", ""),
                        "error": page.get("error", ""),
                    }
                )
                if pdf_page not in selected_pages:
                    selected_pages[pdf_page] = dict(page)
                    selected_pages[pdf_page]["selected_batch_dir"] = str(batch_dir)
        if rows.empty:
            continue
        rows["pdf_page"] = rows["pdf_page"].astype(int)
        for pdf_page, page_rows in rows.groupby("pdf_page", sort=True):
            if len(page_rows) > 0:
                selected_rows[int(pdf_page)] = page_rows.copy()
                page_match = pages[pages["pdf_page"].astype(str).eq(str(pdf_page))]
                if not page_match.empty:
                    selected_pages[int(pdf_page)] = dict(page_match.iloc[0])
                    selected_pages[int(pdf_page)]["selected_batch_dir"] = str(batch_dir)

    merged_rows = pd.concat([selected_rows[p] for p in sorted(selected_rows)], ignore_index=True) if selected_rows else pd.DataFrame()
    merged_pages = pd.DataFrame([selected_pages[p] for p in sorted(selected_pages)])
    attempts_df = pd.DataFrame(attempts)

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    merged_rows.to_csv(args.out_prefix.with_suffix(".rows.csv"), index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    merged_rows.to_csv(args.out_prefix.with_suffix(".rows.tsv"), index=False, sep="\t", encoding="utf-8", quoting=csv.QUOTE_MINIMAL)
    merged_pages.to_csv(args.out_prefix.with_suffix(".pages.csv"), index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    attempts_df.to_csv(args.out_prefix.with_suffix(".attempts.csv"), index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)

    summary = {
        "batch_dirs": [str(path) for path in batch_dirs],
        "pages_with_rows": int(len(selected_rows)),
        "rows": int(len(merged_rows)),
        "pages_seen": int(len(selected_pages)),
        "missing_pages": [page for page in range(137, 230) if page not in selected_rows],
    }
    args.out_prefix.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {args.out_prefix.with_suffix('.rows.csv')}")
    print(f"wrote {args.out_prefix.with_suffix('.pages.csv')}")


if __name__ == "__main__":
    main()
