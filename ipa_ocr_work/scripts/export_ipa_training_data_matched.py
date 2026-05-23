"""Export IPA OCR crops by matching label headwords to PDF entry candidates.

The older exporter assumes one visual line equals one dictionary entry. Many
pages violate that: short entries can share one line, and explanation lines can
sit between entries. This script builds entry candidates from CJK-looking spans
and aligns them to the trusted xlsx/CSV labels in page order.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import fitz
import pandas as pd

from export_ipa_training_data import DEFAULT_LABEL_CSV, DEFAULT_PDF, PROJECT_ROOT, SUPERSCRIPT_TO_DIGIT
from export_wupin_training_data import has_cjk, page_visual_rows, phoneticish_score, write_text


DEFAULT_OUT = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_ipa_matched"


@dataclass
class Candidate:
    page: int
    row_no: int
    entry_no: int
    headword: str
    head_x0: float
    head_x1: float
    row_x1: float
    y0: float
    y1: float
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export matched Shaoxing IPA crops.")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--label-csv", type=Path, default=DEFAULT_LABEL_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--start-page", type=int, default=123)
    parser.add_argument("--end-page", type=int, default=351)
    parser.add_argument("--page-offset", type=int, default=123)
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--match-threshold", type=float, default=0.45)
    parser.add_argument("--weak-threshold", type=float, default=0.62)
    parser.add_argument("--min-phonetic-width", type=float, default=28.0)
    parser.add_argument("--max-gap", type=int, default=6, help="Maximum candidate skips considered by the aligner.")
    parser.add_argument("--require-following-phonetic", action="store_true")
    return parser.parse_args()


def cjk_only(text: object) -> str:
    return "".join(ch for ch in str(text) if "\u3400" <= ch <= "\u9fff")


def match_score(label: str, candidate: str) -> float:
    label = cjk_only(label)
    candidate = cjk_only(candidate)
    if not label or not candidate:
        return 0.0
    if label == candidate:
        return 1.0
    if label in candidate or candidate in label:
        return min(len(label), len(candidate)) / max(len(label), len(candidate))
    return difflib.SequenceMatcher(None, label, candidate).ratio()


def is_headword_span(text: str) -> bool:
    clean = cjk_only(text)
    if not clean:
        return False
    if len(clean) > 6:
        return False
    return True


def has_following_phonetic(spans: list[dict], span_index: int, max_gap: float = 150.0) -> bool:
    """A dictionary entry headword should be followed by phonetic text.

    Hidden PDF text is noisy, but phonetic notation usually still contains
    Latin-ish ASCII, modifier, or combining characters. Short CJK explanation
    fragments often passed the old headword test; this filter removes most of
    those without relying on the corrupted Chinese text itself.
    """
    _, _, head_x1, _ = spans[span_index]["bbox"]
    for next_span in spans[span_index + 1 :]:
        text = next_span["text"].strip()
        if not text:
            continue
        x0, _, _, _ = next_span["bbox"]
        if x0 - head_x1 > max_gap:
            return False
        if has_cjk(text) and phoneticish_score(text) == 0:
            return False
        if phoneticish_score(text) >= max(1, len(text) // 3):
            return True
    return False


def candidates_for_page(page_num: int, page_obj: fitz.Page, require_following_phonetic: bool = False) -> list[Candidate]:
    candidates: list[Candidate] = []
    for row_no, row in enumerate(page_visual_rows(page_obj)):
        cjk_spans = []
        for span_index, span in enumerate(row.spans):
            text = span["text"]
            x0, y0, x1, y1 = span["bbox"]
            if not is_headword_span(text):
                continue
            if require_following_phonetic and not has_following_phonetic(row.spans, span_index):
                continue
            # Explanations usually begin after phonetic text. Keep later CJK
            # spans only when they are separated enough to plausibly start a
            # second compact entry on the same baseline.
            if cjk_spans and x0 - cjk_spans[-1]["x1"] < 80:
                continue
            cjk_spans.append({"text": text, "x0": x0, "x1": x1})

        for entry_no, span in enumerate(cjk_spans):
            candidates.append(
                Candidate(
                    page=page_num,
                    row_no=row_no,
                    entry_no=entry_no,
                    headword=cjk_only(span["text"]),
                    head_x0=float(span["x0"]),
                    head_x1=float(span["x1"]),
                    row_x1=float(row.x1),
                    y0=float(row.crop_y0),
                    y1=float(row.crop_y1),
                    text=row.text,
                )
            )
    return candidates


def align_labels(page_labels: list[dict], candidates: list[Candidate], max_gap: int) -> list[tuple[int, int | None, float]]:
    """Return (label_index, candidate_index, score) in label order.

    PDF hidden text contains many extra CJK fragments from explanations. A
    greedy matcher can drift after the first noisy headword, so use global
    monotonic alignment: every label gets one later candidate, while the aligner
    may skip extra candidates.
    """
    n = len(page_labels)
    m = len(candidates)
    if not n or not m:
        return [(i, None, 0.0) for i in range(n)]

    scores = [
        [match_score(page_labels[i].get("hanzi", ""), candidates[j].headword) for j in range(m)]
        for i in range(n)
    ]
    candidate_best = [max((scores[i][j] for i in range(n)), default=0.0) for j in range(m)]
    neg = -1e9
    dp = [[neg] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple[str, int, int] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    for j in range(m + 1):
        dp[0][j] = 0.0

    label_skip_penalty = 0.55
    for i in range(n + 1):
        for j in range(m):
            if dp[i][j] <= neg / 2:
                continue
            # Skip noisy/explanatory candidate. A tiny penalty discourages
            # unnecessary skipping but still allows abundant extra candidates.
            if dp[i][j] - 0.002 > dp[i][j + 1]:
                dp[i][j + 1] = dp[i][j] - 0.002
                back[i][j + 1] = ("skip", i, j)
            if i < n and dp[i][j] - label_skip_penalty > dp[i + 1][j]:
                dp[i + 1][j] = dp[i][j] - label_skip_penalty
                back[i + 1][j] = ("skip_label", i, j)
            if i < n:
                score = scores[i][j]
                if candidates[j].entry_no > 0 and candidate_best[j] < 0.45:
                    score -= 2.0
                # Small positional prior: labels and candidates should advance
                # at roughly the same page fraction, but visual evidence wins.
                expected = (i / max(1, n - 1)) * max(1, m - 1)
                prior = -abs(j - expected) / max(1, m) * 0.08
                value = dp[i][j] + score + prior
                if value > dp[i + 1][j + 1]:
                    dp[i + 1][j + 1] = value
                    back[i + 1][j + 1] = ("match", i, j)

    end_j = max(range(m + 1), key=lambda j: dp[n][j])
    pairs: list[tuple[int, int | None, float]] = []
    i, j = n, end_j
    while i > 0 and j > 0:
        step = back[i][j]
        if step is None:
            break
        kind, pi, pj = step
        if kind == "match":
            label_idx = pi
            cand_idx = pj
            pairs.append((label_idx, cand_idx, scores[label_idx][cand_idx]))
        i, j = pi, pj
    pairs.reverse()
    by_label = {label_idx: (label_idx, cand_idx, score) for label_idx, cand_idx, score in pairs}
    return [by_label.get(i, (i, None, 0.0)) for i in range(n)]


def load_labels(path: Path, start_page: int, end_page: int) -> dict[int, list[dict]]:
    df = pd.read_csv(path, keep_default_na=False)
    df = df[(df["page"] >= start_page) & (df["page"] <= end_page)].copy()
    df = df[df["wupin"] != ""]
    df["ipa"] = df["ipa_from_wupin"]
    df["ipa_digits"] = df["ipa"].map(lambda text: str(text).translate(SUPERSCRIPT_TO_DIGIT))
    labels: dict[int, list[dict]] = {}
    for page, group in df.groupby("page", sort=True):
        labels[int(page)] = group.to_dict("records")
    return labels


def split_for_page(page: int) -> str:
    # Deterministic page-level split compatible with previous experiments.
    if page % 10 == 9:
        return "test"
    if page % 10 == 8:
        return "val"
    return "train"


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and args.overwrite:
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    labels_by_page = load_labels(args.label_csv, args.start_page, args.end_page)
    doc = fitz.open(args.pdf)
    matrix = fitz.Matrix(args.dpi / 72.0, args.dpi / 72.0)
    manifest_rows = []
    split_lines = {"train": [], "val": [], "test": [], "review": []}
    stats = {"written": 0, "matched": 0, "weak": 0, "low": 0, "unmatched": 0}

    for page, page_labels in labels_by_page.items():
        page_obj = doc[page - args.page_offset]
        candidates = candidates_for_page(page, page_obj, args.require_following_phonetic)
        alignment = align_labels(page_labels, candidates, args.max_gap)
        for label_idx, cand_idx, score in alignment:
            label = page_labels[label_idx]
            split = split_for_page(page)
            image_rel = ""
            crop_bbox = ""
            cand_text = ""
            cand_headword = ""
            quality = "unmatched"

            if cand_idx is not None:
                cand = candidates[cand_idx]
                next_x = cand.row_x1
                if cand_idx + 1 < len(candidates):
                    next_cand = candidates[cand_idx + 1]
                    if next_cand.row_no == cand.row_no:
                        next_x = next_cand.head_x0 - 6
                crop_start = cand.head_x1 + 3
                crop_end = max(crop_start + args.min_phonetic_width, min(next_x, crop_start + len(label["ipa_digits"]) * 9.0 + 26.0))
                bbox = (crop_start, cand.y0, crop_end, cand.y1)
                if score >= args.weak_threshold:
                    quality = "matched"
                elif score >= args.match_threshold:
                    quality = "weak_match"
                else:
                    quality = "low_match"
                if quality != "matched":
                    split = "review"
                stem = f"page_{page:03d}_{label_idx:04d}"
                image_rel = f"{split}/images/{stem}.png"
                image_path = args.out_dir / image_rel
                gt_path = args.out_dir / split / "gt" / f"{stem}.gt.txt"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                page_obj.get_pixmap(matrix=matrix, clip=fitz.Rect(bbox), alpha=False).save(str(image_path))
                write_text(gt_path, label["ipa_digits"] + "\n")
                split_lines[split].append(f"{image_rel}\t{label['ipa_digits']}")
                crop_bbox = repr(bbox)
                cand_text = cand.text
                cand_headword = cand.headword
                stats["written"] += 1
                if quality == "matched":
                    stats["matched"] += 1
                elif quality == "weak_match":
                    stats["weak"] += 1
                else:
                    stats["low"] += 1
            else:
                split = "review"
                stats["unmatched"] += 1

            manifest_rows.append(
                {
                    "split": split,
                    "page": page,
                    "row_index": label_idx,
                    "hanzi": label.get("hanzi", ""),
                    "wupin": label.get("wupin", ""),
                    "ipa": label.get("ipa", ""),
                    "ipa_digits": label.get("ipa_digits", ""),
                    "image": image_rel,
                    "quality": quality,
                    "match_score": f"{score:.4f}",
                    "candidate_headword": cand_headword,
                    "pdf_text": cand_text,
                    "crop_bbox": crop_bbox,
                }
            )

    for split, lines in split_lines.items():
        write_text(args.out_dir / f"{split}.txt", "\n".join(lines) + ("\n" if lines else ""))
    with (args.out_dir / "manifest.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "page",
                "row_index",
                "hanzi",
                "wupin",
                "ipa",
                "ipa_digits",
                "image",
                "quality",
                "match_score",
                "candidate_headword",
                "pdf_text",
                "crop_bbox",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"labels: {len(manifest_rows)}")
    print(f"written: {stats['written']}")
    print(f"matched: {stats['matched']}")
    print(f"weak: {stats['weak']}")
    print(f"low: {stats['low']}")
    print(f"unmatched: {stats['unmatched']}")
    print(f"out: {args.out_dir}")


if __name__ == "__main__":
    main()
