"""Train a direct row-image to phonetic-span detector.

This model is distilled from high-confidence spans selected by the earlier
candidate scorer. Unlike the scorer, it sees the whole row image and directly
predicts the phonetic x-span.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABELS = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_pdf136_clean" / "trocr_segmenter_clean_p90" / "eval_manifest.tsv"
DEFAULT_ROW_SOURCE = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_ipa_matched_skip3"
DEFAULT_ROW_FLAGS = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_pdf136_clean" / "row_cleaning_flags.tsv"
DEFAULT_OUT = PROJECT_ROOT / "ipa_ocr_work" / "models" / "direct_phonetic_span_detector"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train direct phonetic span detector.")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--row-source", type=Path, default=DEFAULT_ROW_SOURCE)
    parser.add_argument("--row-flags", type=Path, default=DEFAULT_ROW_FLAGS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    return parser.parse_args()


def read_training_rows(labels_path: Path, row_flags_path: Path) -> pd.DataFrame:
    labels = pd.read_csv(labels_path, sep="\t", keep_default_na=False)
    flags = pd.read_csv(row_flags_path, sep="\t", keep_default_na=False)
    flags = flags[["page", "row_index", "image", "cleaning_flags", "quality"]].copy()
    merged = labels.merge(flags, on=["page", "row_index"], how="left", suffixes=("", "_row"))
    merged = merged[merged["image_row"].astype(str).ne("")].copy()
    return merged.sort_values(["source_split", "page", "row_index"]).reset_index(drop=True)


def resize_on_canvas(image: Image.Image, height: int, width: int) -> tuple[torch.Tensor, float]:
    image = image.convert("L")
    scale = height / max(1, image.height)
    new_width = max(8, min(width, int(round(image.width * scale))))
    image = image.resize((new_width, height), Image.Resampling.BICUBIC)
    tensor = torch.full((1, height, width), 1.0, dtype=torch.float32)
    pixels = torch.from_numpy(np.asarray(image, dtype=np.float32)).unsqueeze(0).contiguous()
    pixels = (pixels / 255.0 - 0.5) / 0.5
    tensor[:, :, :new_width] = pixels
    return tensor, new_width / max(1, image.width)


def augment_image(image: Image.Image) -> Image.Image:
    if torch.rand(()) < 0.25:
        image = image.filter(ImageFilter.GaussianBlur(radius=float(torch.empty(()).uniform_(0.0, 0.45))))
    if torch.rand(()) < 0.35:
        image = ImageEnhance.Contrast(image).enhance(float(torch.empty(()).uniform_(0.85, 1.20)))
    if torch.rand(()) < 0.35:
        image = ImageEnhance.Brightness(image).enhance(float(torch.empty(()).uniform_(0.90, 1.12)))
    return image


def span_mask(width: int, x0: int, x1: int, out_width: int) -> torch.Tensor:
    mask = torch.zeros(out_width, dtype=torch.float32)
    start = max(0, min(out_width - 1, int(round(x0 / max(1, width) * out_width))))
    end = max(start + 1, min(out_width, int(round(x1 / max(1, width) * out_width))))
    mask[start:end] = 1.0
    return mask


class DirectSpanDataset(Dataset):
    def __init__(self, row_source: Path, rows: pd.DataFrame, height: int, width: int, augment: bool):
        self.row_source = row_source
        self.rows = rows.reset_index(drop=True)
        self.height = height
        self.width = width
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, object]:
        row = self.rows.iloc[idx]
        image = Image.open(self.row_source / str(row["image_row"])).convert("L")
        original_width = int(image.width)
        if self.augment:
            image = augment_image(image)
        tensor, _ = resize_on_canvas(image, self.height, self.width)
        x0 = float(row["span_x0"])
        x1 = float(row["span_x1"])
        target = torch.tensor([x0 / max(1, original_width), x1 / max(1, original_width)], dtype=torch.float32)
        mask = span_mask(original_width, int(x0), int(x1), self.width)
        start = int(round(target[0].item() * (self.width - 1)))
        end = int(round(target[1].item() * (self.width - 1)))
        return {
            "image": tensor,
            "span": target,
            "mask": mask,
            "start": torch.tensor(max(0, min(self.width - 1, start)), dtype=torch.long),
            "end": torch.tensor(max(0, min(self.width - 1, end)), dtype=torch.long),
            "sample_id": str(row["sample_id"]),
            "row_width": original_width,
        }


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False), nn.BatchNorm2d(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = F.silu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.silu(x + residual, inplace=True)


class DirectSpanNet(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.width = width
        self.stem = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.SiLU(inplace=True))
        self.features = nn.Sequential(
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64),
            ResidualBlock(64, 96, stride=2),
            ResidualBlock(96, 96),
            ResidualBlock(96, 160, stride=2),
            ResidualBlock(160, 160),
            ResidualBlock(160, 224, stride=2),
            ResidualBlock(224, 224),
        )
        self.sequence = nn.LSTM(224, 160, num_layers=2, bidirectional=True, batch_first=True, dropout=0.10)
        self.mask_head = nn.Linear(320, 1)
        self.start_head = nn.Linear(320, 1)
        self.end_head = nn.Linear(320, 1)
        self.pool_head = nn.Sequential(nn.Linear(320, 160), nn.SiLU(inplace=True), nn.Dropout(0.15), nn.Linear(160, 2))

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.features(self.stem(image))
        x = x.mean(dim=2).transpose(1, 2)
        x, _ = self.sequence(x)
        seq_len = x.shape[1]
        mask_logits = self.mask_head(x).squeeze(-1)
        start_logits = self.start_head(x).squeeze(-1)
        end_logits = self.end_head(x).squeeze(-1)
        pooled = x.mean(dim=1)
        span = torch.sigmoid(self.pool_head(pooled))
        mask_logits = F.interpolate(mask_logits.unsqueeze(1), size=self.width, mode="linear", align_corners=False).squeeze(1)
        start_logits = F.interpolate(start_logits.unsqueeze(1), size=self.width, mode="linear", align_corners=False).squeeze(1)
        end_logits = F.interpolate(end_logits.unsqueeze(1), size=self.width, mode="linear", align_corners=False).squeeze(1)
        return {"span": span, "mask_logits": mask_logits, "start_logits": start_logits, "end_logits": end_logits}


def span_iou(pred: torch.Tensor, gold: torch.Tensor) -> torch.Tensor:
    left = torch.maximum(pred[:, 0], gold[:, 0])
    right = torch.minimum(pred[:, 1], gold[:, 1])
    inter = torch.clamp(right - left, min=0)
    union = torch.clamp(torch.maximum(pred[:, 1], gold[:, 1]) - torch.minimum(pred[:, 0], gold[:, 0]), min=1e-6)
    return inter / union


def compute_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    span = outputs["span"]
    span_sorted = torch.stack([torch.minimum(span[:, 0], span[:, 1]), torch.maximum(span[:, 0], span[:, 1])], dim=1)
    l1 = F.smooth_l1_loss(span_sorted, batch["span"])
    mask = F.binary_cross_entropy_with_logits(outputs["mask_logits"], batch["mask"])
    start = F.cross_entropy(outputs["start_logits"], batch["start"])
    end = F.cross_entropy(outputs["end_logits"], batch["end"])
    return 2.0 * l1 + 0.7 * mask + 0.25 * (start + end)


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    losses = []
    ious = []
    l1s = []
    with torch.no_grad():
        for batch in loader:
            tensor_batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            outputs = model(tensor_batch["image"])
            loss = compute_loss(outputs, tensor_batch)
            span = outputs["span"]
            pred = torch.stack([torch.minimum(span[:, 0], span[:, 1]), torch.maximum(span[:, 0], span[:, 1])], dim=1)
            ious.extend(span_iou(pred, tensor_batch["span"]).cpu().tolist())
            l1s.extend(torch.abs(pred - tensor_batch["span"]).mean(dim=1).cpu().tolist())
            losses.append(float(loss.item()))
    model.train()
    return {
        "loss": sum(losses) / max(1, len(losses)),
        "iou": sum(ious) / max(1, len(ious)),
        "l1": sum(l1s) / max(1, len(l1s)),
        "iou_090": sum(1 for v in ious if v >= 0.90) / max(1, len(ious)),
        "iou_080": sum(1 for v in ious if v >= 0.80) / max(1, len(ious)),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_training_rows(args.labels, args.row_flags)
    train_rows = rows[rows["source_split"].eq("train")].copy()
    val_rows = rows[rows["source_split"].eq("val")].copy()
    test_rows = rows[rows["source_split"].eq("test")].copy()
    if args.eval_limit:
        train_rows = train_rows.head(args.eval_limit)
        val_rows = val_rows.head(max(1, args.eval_limit // 4))
        test_rows = test_rows.head(max(1, args.eval_limit // 4))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metadata = {
        "labels": str(args.labels),
        "row_source": str(args.row_source),
        "train": len(train_rows),
        "val": len(val_rows),
        "test": len(test_rows),
        "height": args.height,
        "width": args.width,
    }
    (args.out_dir / "config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"device": device, **metadata}, ensure_ascii=False))

    train_loader = DataLoader(DirectSpanDataset(args.row_source, train_rows, args.height, args.width, args.augment), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(DirectSpanDataset(args.row_source, val_rows, args.height, args.width, False), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(DirectSpanDataset(args.row_source, test_rows, args.height, args.width, False), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = DirectSpanNet(args.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    history = []
    best_iou = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            tensor_batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(tensor_batch["image"])
            loss = compute_loss(outputs, tensor_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
        scheduler.step()
        val = evaluate(model, val_loader, device)
        best = ""
        if val["iou"] > best_iou:
            best_iou = val["iou"]
            best = "*"
            torch.save({"model": model.state_dict(), "metadata": metadata, "args": vars(args)}, args.out_dir / "best.pt")
        if args.save_every and epoch % args.save_every == 0:
            torch.save({"model": model.state_dict(), "metadata": metadata, "args": vars(args)}, args.out_dir / f"epoch_{epoch:04d}.pt")
        row = {"epoch": epoch, "train_loss": sum(losses) / max(1, len(losses)), **{f"val_{k}": v for k, v in val.items()}, "best": best}
        history.append(row)
        with (args.out_dir / "history.tsv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()), delimiter="\t")
            writer.writeheader()
            writer.writerows(history)
        print(f"epoch {epoch}: train_loss={row['train_loss']:.4f} val_iou={val['iou']:.4f} val_l1={val['l1']:.4f} val_iou90={val['iou_090']:.4f}{best}")

    checkpoint = torch.load(args.out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test = evaluate(model, test_loader, device)
    (args.out_dir / "test_metrics.json").write_text(json.dumps(test, ensure_ascii=False, indent=2), encoding="utf-8")
    print("test " + json.dumps(test, ensure_ascii=False))


if __name__ == "__main__":
    main()
