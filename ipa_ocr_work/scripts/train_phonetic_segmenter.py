"""Train a visual quality model for phonetic span candidates."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_pdf136_clean" / "phonetic_segmenter_candidates"
DEFAULT_OUT = PROJECT_ROOT / "ipa_ocr_work" / "models" / "phonetic_segmenter"
FEATURE_COLUMNS = ["width_ratio", "tone_ratio", "component_ratio", "cjk_like_ratio", "span_fraction", "syllable_count"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train phonetic span candidate quality model.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    return parser.parse_args()


def read_rows(path: Path, split: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = [row for row in csv.DictReader(f, delimiter="\t") if row.get("source_split") == split]
    rows.sort(key=lambda row: (int(row["page"]), int(row["row_index"]), int(row["candidate_rank"])))
    return rows


def resize_on_canvas(image: Image.Image, height: int, width: int) -> torch.Tensor:
    image = image.convert("L")
    scale = height / max(1, image.height)
    new_width = max(8, min(width, int(round(image.width * scale))))
    image = image.resize((new_width, height), Image.Resampling.BICUBIC)
    tensor = torch.full((1, height, width), 1.0, dtype=torch.float32)
    pixels = torch.from_numpy(np.asarray(image, dtype=np.float32)).unsqueeze(0).contiguous()
    pixels = (pixels / 255.0 - 0.5) / 0.5
    tensor[:, :, :new_width] = pixels
    return tensor


def augment_image(image: Image.Image) -> Image.Image:
    if torch.rand(()) < 0.25:
        image = image.filter(ImageFilter.GaussianBlur(radius=float(torch.empty(()).uniform_(0.0, 0.45))))
    if torch.rand(()) < 0.35:
        image = ImageEnhance.Contrast(image).enhance(float(torch.empty(()).uniform_(0.85, 1.20)))
    if torch.rand(()) < 0.35:
        image = ImageEnhance.Brightness(image).enhance(float(torch.empty(()).uniform_(0.90, 1.10)))
    return image


class CandidateDataset(Dataset):
    def __init__(self, root: Path, rows: list[dict[str, str]], height: int, width: int, augment: bool):
        self.root = root
        self.rows = rows
        self.height = height
        self.width = width
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, object]:
        row = self.rows[idx]
        image = Image.open(self.root / row["image"]).convert("L")
        if self.augment:
            image = augment_image(image)
        features = []
        for col in FEATURE_COLUMNS:
            value = float(row.get(col, 0.0) or 0.0)
            if col == "width_ratio":
                value = math.log(max(0.05, value))
            elif col == "syllable_count":
                value = value / 6.0
            features.append(value)
        return {
            "image": resize_on_canvas(image, self.height, self.width),
            "features": torch.tensor(features, dtype=torch.float32),
            "target": torch.tensor(float(row["label"]), dtype=torch.float32),
            "sample_id": row["sample_id"],
            "row_id": row["row_id"],
        }


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = F.silu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.silu(x + residual, inplace=True)


class SpanQualityNet(nn.Module):
    def __init__(self, feature_dim: int = len(FEATURE_COLUMNS)):
        super().__init__()
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
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_mlp = nn.Sequential(nn.Linear(feature_dim, 48), nn.SiLU(inplace=True), nn.LayerNorm(48))
        self.head = nn.Sequential(nn.Linear(224 + 48, 192), nn.SiLU(inplace=True), nn.Dropout(0.25), nn.Linear(192, 1))

    def forward(self, image: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        visual = self.pool(self.features(self.stem(image))).flatten(1)
        numeric = self.feature_mlp(features)
        return self.head(torch.cat([visual, numeric], dim=1)).squeeze(1)


def rowwise_metrics(rows: list[dict[str, str]], probs: list[float]) -> dict[str, float]:
    by_row: dict[str, list[tuple[dict[str, str], float]]] = {}
    for row, prob in zip(rows, probs):
        by_row.setdefault(row["row_id"], []).append((row, prob))
    top1 = 0
    eligible = 0
    for items in by_row.values():
        if not any(int(item[0]["label"]) for item in items):
            continue
        eligible += 1
        best = max(items, key=lambda item: item[1])
        top1 += int(int(best[0]["label"]) == 1)
    return {"row_top1": top1 / max(1, eligible), "eligible_rows": eligible}


def evaluate(model: nn.Module, loader: DataLoader, device: str, rows: list[dict[str, str]]) -> tuple[dict[str, float], list[float]]:
    model.eval()
    losses = []
    probs = []
    targets = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(device), batch["features"].to(device))
            target = batch["target"].to(device)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            losses.append(float(loss.item()))
            probs.extend(torch.sigmoid(logits).cpu().tolist())
            targets.extend(target.cpu().tolist())
    preds = [int(p >= 0.5) for p in probs]
    acc = sum(int(pred == int(target)) for pred, target in zip(preds, targets)) / max(1, len(targets))
    positives = sum(int(target) for target in targets)
    recall = sum(int(pred == 1 and int(target) == 1) for pred, target in zip(preds, targets)) / max(1, positives)
    precision = sum(int(pred == 1 and int(target) == 1) for pred, target in zip(preds, targets)) / max(1, sum(preds))
    metrics = {"loss": sum(losses) / max(1, len(losses)), "acc": acc, "precision": precision, "recall": recall}
    metrics.update(rowwise_metrics(rows, probs))
    return metrics, probs


def sampler_weights(rows: list[dict[str, str]]) -> list[float]:
    pos = sum(int(row["label"]) for row in rows)
    neg = max(1, len(rows) - pos)
    pos_weight = len(rows) / max(1, pos)
    neg_weight = len(rows) / neg
    return [pos_weight if int(row["label"]) else neg_weight for row in rows]


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest = args.data_dir / "candidate_manifest.tsv"
    train_rows = read_rows(manifest, "train")
    val_rows = read_rows(manifest, "val")
    test_rows = read_rows(manifest, "test")
    metadata = {
        "data_dir": str(args.data_dir),
        "train": len(train_rows),
        "val": len(val_rows),
        "test": len(test_rows),
        "feature_columns": FEATURE_COLUMNS,
        "height": args.height,
        "width": args.width,
    }
    (args.out_dir / "config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False))

    train_ds = CandidateDataset(args.data_dir, train_rows, args.height, args.width, args.augment)
    val_ds = CandidateDataset(args.data_dir, val_rows, args.height, args.width, False)
    test_ds = CandidateDataset(args.data_dir, test_rows, args.height, args.width, False)
    sampler = WeightedRandomSampler(sampler_weights(train_rows), num_samples=len(train_rows), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = SpanQualityNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    best_score = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["image"].to(device), batch["features"].to(device))
            target = batch["target"].to(device)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item())
            steps += 1
        scheduler.step()
        val_metrics, _ = evaluate(model, val_loader, device, val_rows)
        score = val_metrics["row_top1"]
        best = ""
        if score > best_score:
            best_score = score
            best = "*"
            torch.save({"model": model.state_dict(), "metadata": metadata, "args": vars(args)}, args.out_dir / "best.pt")
        if args.save_every and epoch % args.save_every == 0:
            torch.save({"model": model.state_dict(), "metadata": metadata, "args": vars(args)}, args.out_dir / f"epoch_{epoch:04d}.pt")
        row = {"epoch": epoch, "train_loss": total_loss / max(1, steps), **{f"val_{k}": v for k, v in val_metrics.items()}, "best": best}
        history.append(row)
        with (args.out_dir / "history.tsv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()), delimiter="\t")
            writer.writeheader()
            writer.writerows(history)
        print(
            f"epoch {epoch}: train_loss={row['train_loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_top1={val_metrics['row_top1']:.4f} "
            f"val_acc={val_metrics['acc']:.4f} p={val_metrics['precision']:.4f} r={val_metrics['recall']:.4f}{best}"
        )

    checkpoint = torch.load(args.out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_metrics, test_probs = evaluate(model, test_loader, device, test_rows)
    with (args.out_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)
    with (args.out_dir / "predictions_test.tsv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(test_rows[0].keys()) + ["prob"] if test_rows else ["prob"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row, prob in zip(test_rows, test_probs):
            writer.writerow({**row, "prob": prob})
    print("test " + json.dumps(test_metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
