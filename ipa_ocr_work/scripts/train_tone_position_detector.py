"""Train a lightweight lower-tone-position detector on syllable crops."""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_dual_model" / "tone_position_detector"
DEFAULT_OUT = PROJECT_ROOT / "ipa_ocr_work" / "models" / "tone_position_detector"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train lower-tone detector.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--backbone", choices=["cnn", "resnet"], default="resnet")
    parser.add_argument("--target-accuracy", type=float, default=0.9)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--skip-inaccessible", action="store_true", default=True)
    return parser.parse_args()


def image_accessible(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def load_rows(data_dir: Path, split: str, skip_inaccessible: bool = True) -> list[dict[str, str]]:
    rows = []
    with (data_dir / "detector_manifest.tsv").open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["source_split"] == split:
                if skip_inaccessible and not image_accessible(data_dir / row["image"]):
                    continue
                rows.append(row)
    return rows


class ToneDataset(Dataset):
    def __init__(self, data_dir: Path, rows: list[dict[str, str]], height: int, width: int):
        self.data_dir = data_dir
        self.rows = rows
        self.height = height
        self.width = width

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        image_path = self.data_dir / row["image"]
        last_error = None
        for _ in range(5):
            try:
                image = Image.open(image_path).convert("L")
                break
            except PermissionError as exc:
                last_error = exc
                time.sleep(0.1)
        else:
            raise last_error if last_error else RuntimeError(f"cannot open {image_path}")
        scale = self.height / image.height
        new_width = max(8, min(self.width, int(round(image.width * scale))))
        image = image.resize((new_width, self.height), Image.Resampling.BICUBIC)
        tensor = torch.full((1, self.height, self.width), 1.0, dtype=torch.float32)
        pixels = torch.from_numpy(np.asarray(image, dtype=np.float32)).unsqueeze(0).contiguous()
        pixels = (pixels / 255.0 - 0.5) / 0.5
        tensor[:, :, :new_width] = pixels
        return {
            "image": tensor,
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "sample_id": row["sample_id"],
        }


class ToneCnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 192, 3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.15), nn.Linear(192, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(images)).squeeze(1)


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

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(images)
        x = F.silu(self.bn1(self.conv1(images)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.silu(x + residual, inplace=True)


class ToneResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 48, 3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True),
        )
        self.features = nn.Sequential(
            ResidualBlock(48, 64, stride=2),
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
            ResidualBlock(128, 192, stride=2),
            ResidualBlock(192, 192),
            ResidualBlock(192, 256, stride=2),
            ResidualBlock(256, 256),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(self.stem(images))).squeeze(1)


def build_model(backbone: str) -> nn.Module:
    if backbone == "cnn":
        return ToneCnn()
    if backbone == "resnet":
        return ToneResNet()
    raise ValueError(f"unknown backbone: {backbone}")


def metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).long()
    gold = labels.long()
    tp = int(((preds == 1) & (gold == 1)).sum().item())
    tn = int(((preds == 0) & (gold == 0)).sum().item())
    fp = int(((preds == 1) & (gold == 0)).sum().item())
    fn = int(((preds == 0) & (gold == 1)).sum().item())
    n = max(1, len(gold))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "accuracy": (tp + tn) / n,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def best_threshold_for_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    best_threshold = 0.5
    best_accuracy = -1.0
    for step in range(5, 96):
        threshold = step / 100.0
        metrics = metrics_from_logits(logits, labels, threshold)
        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            best_threshold = threshold
    return best_threshold


def evaluate(model: ToneCnn, loader: DataLoader, criterion: nn.Module, device: str, threshold: float | None = 0.5) -> dict[str, float]:
    model.eval()
    losses = []
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = model(images)
            losses.append(float(criterion(logits, labels).item()))
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    logits = torch.cat(all_logits) if all_logits else torch.empty(0)
    labels = torch.cat(all_labels) if all_labels else torch.empty(0)
    chosen_threshold = best_threshold_for_accuracy(logits, labels) if threshold is None else threshold
    metrics = metrics_from_logits(logits, labels, chosen_threshold)
    metrics["loss"] = sum(losses) / max(1, len(losses))
    metrics["threshold"] = chosen_threshold
    model.train()
    return metrics


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_rows = load_rows(args.data_dir, "train", args.skip_inaccessible)
    val_rows = load_rows(args.data_dir, "val", args.skip_inaccessible)
    test_rows = load_rows(args.data_dir, "test", args.skip_inaccessible)
    print(f"rows train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")
    train_loader = DataLoader(ToneDataset(args.data_dir, train_rows, args.height, args.width), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(ToneDataset(args.data_dir, val_rows, args.height, args.width), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(ToneDataset(args.data_dir, test_rows, args.height, args.width), batch_size=args.batch_size, shuffle=False)

    positives = sum(int(row["label"]) for row in train_rows)
    negatives = max(1, len(train_rows) - positives)
    pos_weight = torch.tensor([negatives / max(1, positives)], dtype=torch.float32, device=device)
    model = build_model(args.backbone).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    best_f1 = -math.inf
    history = []
    ckpts = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        steps = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["image"].to(device))
            loss = criterion(logits, batch["label"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.item())
            steps += 1
        scheduler.step()
        val = evaluate(model, val_loader, criterion, device, threshold=None)
        best = ""
        if val["f1"] > best_f1:
            best_f1 = val["f1"]
            best = "*"
            torch.save({"model": model.state_dict(), "args": vars(args), "threshold": val["threshold"]}, args.out_dir / "best.pt")
        if args.save_every and epoch % args.save_every == 0:
            path = args.out_dir / f"epoch_{epoch:04d}.pt"
            torch.save({"model": model.state_dict(), "args": vars(args), "threshold": val["threshold"]}, path)
            ckpts.append(path)
        row = {"epoch": epoch, "train_loss": total / max(1, steps), **{f"val_{k}": v for k, v in val.items()}, "best": best}
        history.append(row)
        with (args.out_dir / "history.tsv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()), delimiter="\t")
            writer.writeheader()
            writer.writerows(history)
        print(
            f"epoch {epoch}: train_loss={row['train_loss']:.4f} "
            f"val_acc={val['accuracy']:.4f} val_f1={val['f1']:.4f} threshold={val['threshold']:.2f}{best}"
        )

    eval_rows = []
    for name, path in [("best.pt", args.out_dir / "best.pt")] + [(p.name, p) for p in ckpts]:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        metrics = evaluate(model, test_loader, criterion, device, threshold=float(checkpoint.get("threshold", 0.5)))
        eval_rows.append({"checkpoint": name, **metrics})
        target = "TARGET_MET" if metrics["accuracy"] >= args.target_accuracy else "TARGET_NOT_MET"
        print(f"{name}: test_acc={metrics['accuracy']:.4f} test_f1={metrics['f1']:.4f} {target}_{args.target_accuracy:.2f}")
    with (args.out_dir / "checkpoint_eval.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(eval_rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(eval_rows)


if __name__ == "__main__":
    main()
