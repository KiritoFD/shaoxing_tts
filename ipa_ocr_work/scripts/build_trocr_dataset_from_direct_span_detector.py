"""Build TrOCR crops from the direct phonetic span detector.

The detector takes a full row image and predicts the x-span containing the
phonetic transcription. This script crops that span, writes a TrOCR-compatible
eval_manifest.tsv, and emits optional QA artifacts for visual inspection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_direct_phonetic_span_detector import DirectSpanNet, resize_on_canvas  # noqa: E402


DEFAULT_MANIFEST = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_pdf136_clean" / "ocr_selected_phonetic_reliable" / "eval_manifest.tsv"
DEFAULT_MODEL = PROJECT_ROOT / "ipa_ocr_work" / "models" / "direct_phonetic_span_detector_p90_w768_b12_e80_20260524_2042" / "best.pt"
DEFAULT_OUT = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_pdf136_clean" / "trocr_direct_span_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TrOCR dataset from direct span detector crops.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--variant", default="direct_phonetic_span_v1")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--margin-x", type=int, default=6)
    parser.add_argument("--min-crop-width", type=int, default=32)
    parser.add_argument("--min-confidence", type=float, default=None)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--qa-count", type=int, default=96)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_image_path(manifest_path: Path, image_value: str) -> Path:
    image_path = Path(str(image_value))
    if image_path.is_absolute():
        return image_path
    resolved = (manifest_path.parent / image_path).resolve()
    if resolved.exists():
        return resolved
    return (PROJECT_ROOT / "ipa_ocr_work" / "dataset" / image_path).resolve()


class RowImageDataset(Dataset):
    def __init__(self, manifest_path: Path, rows: pd.DataFrame, height: int, width: int):
        self.manifest_path = manifest_path
        self.rows = rows.reset_index(drop=True)
        self.height = height
        self.width = width

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, object]:
        row = self.rows.iloc[idx]
        image_path = resolve_image_path(self.manifest_path, str(row["image"]))
        image = Image.open(image_path).convert("L")
        tensor, _ = resize_on_canvas(image, self.height, self.width)
        return {
            "image": tensor,
            "idx": idx,
            "image_path": str(image_path),
            "row_width": int(image.width),
            "row_height": int(image.height),
        }


def collate(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "idx": [int(item["idx"]) for item in batch],
        "image_path": [str(item["image_path"]) for item in batch],
        "row_width": [int(item["row_width"]) for item in batch],
        "row_height": [int(item["row_height"]) for item in batch],
    }


def span_confidence(mask_logits: torch.Tensor, start: int, end: int) -> float:
    probs = torch.sigmoid(mask_logits)
    start = max(0, min(probs.numel() - 1, start))
    end = max(start + 1, min(probs.numel(), end))
    inside = probs[start:end].mean()
    if start == 0 and end == probs.numel():
        outside = torch.tensor(0.0, device=probs.device)
    else:
        outside_parts = []
        if start > 0:
            outside_parts.append(probs[:start])
        if end < probs.numel():
            outside_parts.append(probs[end:])
        outside = torch.cat(outside_parts).mean() if outside_parts else torch.tensor(0.0, device=probs.device)
    return float((inside - outside).clamp(min=-1.0, max=1.0).item())


def predict_spans(args: argparse.Namespace, rows: pd.DataFrame) -> list[dict[str, object]]:
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata", {})
    height = int(metadata.get("height", 96))
    width = int(metadata.get("width", 768))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DirectSpanNet(width).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dataset = RowImageDataset(args.manifest, rows, height, width)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    predictions: list[dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch["image"].to(device))
            spans = outputs["span"]
            spans = torch.stack([torch.minimum(spans[:, 0], spans[:, 1]), torch.maximum(spans[:, 0], spans[:, 1])], dim=1)
            mask_logits = outputs["mask_logits"].detach().cpu()
            for j, idx in enumerate(batch["idx"]):
                row_width = int(batch["row_width"][j])
                raw_x0 = float(spans[j, 0].detach().cpu().item()) * row_width
                raw_x1 = float(spans[j, 1].detach().cpu().item()) * row_width
                x0 = int(math.floor(raw_x0)) - args.margin_x
                x1 = int(math.ceil(raw_x1)) + args.margin_x
                x0 = max(0, min(row_width - 1, x0))
                x1 = max(x0 + args.min_crop_width, min(row_width, x1))
                if x1 > row_width:
                    x1 = row_width
                    x0 = max(0, x1 - args.min_crop_width)
                start = int(round(max(0.0, min(1.0, raw_x0 / max(1, row_width))) * (width - 1)))
                end = int(round(max(0.0, min(1.0, raw_x1 / max(1, row_width))) * width))
                predictions.append(
                    {
                        "idx": idx,
                        "source_image_abs": batch["image_path"][j],
                        "row_width": row_width,
                        "row_height": int(batch["row_height"][j]),
                        "pred_raw_x0": raw_x0,
                        "pred_raw_x1": raw_x1,
                        "pred_x0": x0,
                        "pred_x1": x1,
                        "span_width": x1 - x0,
                        "span_width_ratio": (x1 - x0) / max(1, row_width),
                        "span_confidence": span_confidence(mask_logits[j], start, end),
                    }
                )
    return sorted(predictions, key=lambda item: int(item["idx"]))


def crop_images(args: argparse.Namespace, rows: pd.DataFrame, predictions: list[dict[str, object]]) -> pd.DataFrame:
    image_dir = args.out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    out_rows = []
    for row, pred in zip(rows.to_dict("records"), predictions):
        sample_id = str(row["sample_id"])
        source_image = Path(str(pred["source_image_abs"]))
        image = Image.open(source_image).convert("RGB")
        x0 = int(pred["pred_x0"])
        x1 = int(pred["pred_x1"])
        crop = image.crop((x0, 0, x1, image.height))
        target_image = image_dir / f"{sample_id}.png"
        crop.save(target_image)
        out_row = dict(row)
        out_row["variant"] = args.variant
        out_row["image"] = f"images/{sample_id}.png"
        out_row["source_image"] = str(row["image"])
        out_row.update({k: v for k, v in pred.items() if k != "idx"})
        out_rows.append(out_row)
    return pd.DataFrame(out_rows)


def write_contact_sheet(args: argparse.Namespace, rows: pd.DataFrame, predictions: list[dict[str, object]]) -> Path | None:
    if args.qa_count <= 0 or len(rows) == 0:
        return None
    qa_dir = args.out_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    sample_rows = rows.head(args.qa_count).reset_index(drop=True)
    sample_preds = predictions[: len(sample_rows)]
    tile_w, tile_h = 360, 104
    cols = 4
    sheet = Image.new("RGB", (cols * tile_w, math.ceil(len(sample_rows) / cols) * tile_h), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (row, pred) in enumerate(zip(sample_rows.to_dict("records"), sample_preds)):
        source = Image.open(Path(str(pred["source_image_abs"]))).convert("RGB")
        scale = min((tile_w - 10) / max(1, source.width), 62 / max(1, source.height))
        view = source.resize((max(1, int(source.width * scale)), max(1, int(source.height * scale))), Image.Resampling.BICUBIC)
        ox = (i % cols) * tile_w + 5
        oy = (i // cols) * tile_h + 18
        sheet.paste(view, (ox, oy))
        px0 = ox + int(round(int(pred["pred_x0"]) * scale))
        px1 = ox + int(round(int(pred["pred_x1"]) * scale))
        draw.rectangle((px0, oy, px1, oy + view.height), outline=(220, 30, 30), width=2)
        label = f"{row['sample_id']} {row.get('source_split', '')} conf={float(pred['span_confidence']):.2f}"
        draw.text((ox, (i // cols) * tile_h + 3), label, fill=(0, 0, 0))
    sheet_path = qa_dir / "direct_span_contactsheet.png"
    sheet.save(sheet_path)
    return sheet_path


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and args.overwrite:
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = pd.read_csv(args.manifest, sep="\t", keep_default_na=False)
    if args.max_rows:
        rows = rows.head(args.max_rows).copy()
    predictions = predict_spans(args, rows)
    if args.min_confidence is not None:
        keep = [float(pred["span_confidence"]) >= args.min_confidence for pred in predictions]
        rows = rows.loc[keep].reset_index(drop=True)
        predictions = [pred for pred, should_keep in zip(predictions, keep) if should_keep]
    out = crop_images(args, rows, predictions)
    out = out.sort_values(["source_split", "page", "row_index"]).reset_index(drop=True)
    out.to_csv(args.out_dir / "eval_manifest.tsv", sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)
    pred_cols = [
        "idx",
        "source_image_abs",
        "row_width",
        "row_height",
        "pred_raw_x0",
        "pred_raw_x1",
        "pred_x0",
        "pred_x1",
        "span_width",
        "span_width_ratio",
        "span_confidence",
    ]
    pd.DataFrame(predictions)[pred_cols].to_csv(args.out_dir / "span_predictions.tsv", sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)
    sheet_path = write_contact_sheet(args, rows, predictions)
    summary = {
        "manifest": str(args.manifest),
        "model": str(args.model),
        "variant": args.variant,
        "min_confidence": args.min_confidence,
        "rows": int(len(out)),
        "split_counts": out["source_split"].value_counts().to_dict() if len(out) else {},
        "quality_counts": out["quality"].value_counts().to_dict() if "quality" in out else {},
        "span_confidence": out["span_confidence"].describe().to_dict() if len(out) else {},
        "span_width_ratio": out["span_width_ratio"].describe().to_dict() if len(out) else {},
        "contact_sheet": str(sheet_path) if sheet_path else "",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {args.out_dir / 'eval_manifest.tsv'}")


if __name__ == "__main__":
    main()
