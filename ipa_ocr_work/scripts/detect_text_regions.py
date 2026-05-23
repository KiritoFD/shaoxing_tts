"""Detect page-level text regions for phonetic OCR cropping.

This is a fast detector baseline. It can use PaddleOCR when available, and
falls back to OpenCV connected-component grouping for quick local iteration.
The output is meant for visual QA before training recognition models.
"""

from __future__ import annotations

import argparse
import csv
import inspect
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import fitz
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PDF = PROJECT_ROOT / "ipa_ocr_work" / "data" / "shaoxing_123-351.pdf"
DEFAULT_OUT = PROJECT_ROOT / "ipa_ocr_work" / "eval" / "detector_baseline"


@dataclass
class Box:
    x0: int
    y0: int
    x1: int
    y1: int
    score: float = 1.0
    kind: str = "text"

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def h(self) -> int:
        return self.y1 - self.y0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect text regions on rendered PDF pages.")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--page", type=int, default=129, help="Book page number.")
    parser.add_argument("--page-offset", type=int, default=123)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--engine", choices=("auto", "opencv", "paddle"), default="auto")
    parser.add_argument("--threshold", type=int, default=190)
    parser.add_argument("--min-area", type=int, default=25)
    parser.add_argument("--line-gap", type=int, default=18)
    parser.add_argument("--word-gap", type=int, default=18)
    return parser.parse_args()


def render_page(pdf: Path, page: int, page_offset: int, dpi: int) -> np.ndarray:
    doc = fitz.open(pdf)
    pdf_index = page - page_offset
    if pdf_index < 0 or pdf_index >= len(doc):
        raise IndexError(f"page {page} maps to invalid PDF index {pdf_index}")
    pix = doc[pdf_index].get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
    rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return cv.cvtColor(rgb, cv.COLOR_RGB2BGR)


def merge_boxes(boxes: list[Box], kind: str) -> Box:
    return Box(
        min(box.x0 for box in boxes),
        min(box.y0 for box in boxes),
        max(box.x1 for box in boxes),
        max(box.y1 for box in boxes),
        score=float(np.mean([box.score for box in boxes])),
        kind=kind,
    )


def detect_opencv(image: np.ndarray, threshold: int, min_area: int, line_gap: int, word_gap: int) -> list[Box]:
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, threshold, 255, cv.THRESH_BINARY_INV)
    # A tiny close joins broken strokes without swallowing whole definitions.
    binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (2, 2)))
    n, _, stats, _ = cv.connectedComponentsWithStats(binary, connectivity=8)

    chars: list[Box] = []
    h_img, w_img = gray.shape
    for idx in range(1, n):
        x, y, w, h, area = stats[idx]
        if area < min_area or w < 2 or h < 3:
            continue
        if y < 0.06 * h_img or y > 0.94 * h_img:
            continue
        if x < 0.02 * w_img or x > 0.98 * w_img:
            continue
        if h > 0.08 * h_img or w > 0.35 * w_img:
            continue
        chars.append(Box(int(x), int(y), int(x + w), int(y + h), kind="component"))

    chars.sort(key=lambda box: (box.cy, box.x0))
    lines: list[list[Box]] = []
    for box in chars:
        if not lines or abs(np.mean([b.cy for b in lines[-1]]) - box.cy) > line_gap:
            lines.append([box])
        else:
            lines[-1].append(box)

    words: list[Box] = []
    for line_index, line in enumerate(lines):
        line.sort(key=lambda box: box.x0)
        group: list[Box] = []
        for box in line:
            if not group:
                group = [box]
                continue
            gap = box.x0 - group[-1].x1
            height = max(1, int(np.median([b.h for b in group])))
            allowed_gap = max(word_gap, int(height * 0.9))
            if gap <= allowed_gap:
                group.append(box)
            else:
                if len(group) >= 2:
                    words.append(merge_boxes(group, f"line_{line_index:03d}"))
                group = [box]
        if len(group) >= 2:
            words.append(merge_boxes(group, f"line_{line_index:03d}"))
    return words


def paddle_points_to_box(points: object) -> Box:
    pts = np.asarray(points, dtype=np.float32)
    return Box(
        int(np.floor(pts[:, 0].min())),
        int(np.floor(pts[:, 1].min())),
        int(np.ceil(pts[:, 0].max())),
        int(np.ceil(pts[:, 1].max())),
        kind="paddle",
    )


def detect_paddle(image: np.ndarray) -> list[Box]:
    from paddleocr import PaddleOCR

    init_sig = inspect.signature(PaddleOCR)
    if "use_textline_orientation" in init_sig.parameters:
        ocr = PaddleOCR(lang="ch", use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)
        result = ocr.predict(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        boxes = []
        for item in result:
            if hasattr(item, "get"):
                for key in ("dt_polys", "rec_polys", "det_polys"):
                    if key in item:
                        boxes.extend(paddle_points_to_box(points) for points in item[key])
                        break
            elif hasattr(item, "json"):
                data = item.json
                for key in ("dt_polys", "rec_polys", "det_polys"):
                    if key in data:
                        boxes.extend(paddle_points_to_box(points) for points in data[key])
                        break
        return boxes

    ocr = PaddleOCR(lang="ch", use_gpu=False, use_angle_cls=False, show_log=False)
    result = ocr.ocr(cv.cvtColor(image, cv.COLOR_BGR2RGB), det=True, rec=False, cls=False)
    if not result:
        return []
    boxes = result[0] if len(result) == 1 and isinstance(result[0], list) else result
    return [paddle_points_to_box(points) for points in boxes if points is not None]


def draw_boxes(image: np.ndarray, boxes: list[Box]) -> np.ndarray:
    out = image.copy()
    for idx, box in enumerate(boxes):
        color = (0, 180, 255) if box.kind.startswith("line_") else (0, 255, 0)
        cv.rectangle(out, (box.x0, box.y0), (box.x1, box.y1), color, 2)
        cv.putText(out, str(idx), (box.x0, max(12, box.y0 - 4)), cv.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return out


def write_outputs(out_dir: Path, page: int, image: np.ndarray, boxes: list[Box]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = out_dir / f"page_{page:03d}_render.png"
    boxed_path = out_dir / f"page_{page:03d}_boxes.png"
    tsv_path = out_dir / f"page_{page:03d}_detections.tsv"
    cv.imwrite(str(image_path), image)
    cv.imwrite(str(boxed_path), draw_boxes(image, boxes))
    with tsv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["box_id", "kind", "x0", "y0", "x1", "y1", "width", "height", "score"],
            delimiter="\t",
        )
        writer.writeheader()
        for idx, box in enumerate(boxes):
            writer.writerow(
                {
                    "box_id": idx,
                    "kind": box.kind,
                    "x0": box.x0,
                    "y0": box.y0,
                    "x1": box.x1,
                    "y1": box.y1,
                    "width": box.w,
                    "height": box.h,
                    "score": f"{box.score:.4f}",
                }
            )
    print(f"boxes: {len(boxes)}")
    print(f"wrote {boxed_path}")
    print(f"wrote {tsv_path}")


def main() -> None:
    args = parse_args()
    image = render_page(args.pdf, args.page, args.page_offset, args.dpi)
    boxes: list[Box]
    if args.engine == "paddle":
        boxes = detect_paddle(image)
    elif args.engine == "auto":
        try:
            boxes = detect_paddle(image)
        except Exception as exc:
            print(f"PaddleOCR unavailable, using OpenCV fallback: {exc}")
            boxes = detect_opencv(image, args.threshold, args.min_area, args.line_gap, args.word_gap)
    else:
        boxes = detect_opencv(image, args.threshold, args.min_area, args.line_gap, args.word_gap)
    boxes.sort(key=lambda box: (box.y0, box.x0))
    write_outputs(args.out_dir, args.page, image, boxes)


if __name__ == "__main__":
    main()
