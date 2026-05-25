"""Train a closed-set syllable image classifier.

This is a pragmatic OCR baseline for short Shaoxing syllable crops. Unknown
validation/test labels that never occur in train are counted as incorrect; the
row-level ceiling is still about 85% for the current split.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import unicodedata
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_syllable_ocr"
DEFAULT_OUT = PROJECT_ROOT / "ipa_ocr_work" / "models" / "syllable_classifier"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train syllable classifier.")
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--variant", default="syllable_crop")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--save-every", type=int, default=20)
    return parser.parse_args()


def normalize_label(text: object) -> str:
    return unicodedata.normalize("NFC", str(text)).strip().replace(" ", "")


def read_rows(eval_dir: Path, split: str, variant: str) -> list[dict[str, str]]:
    with (eval_dir / "eval_manifest.tsv").open("r", encoding="utf-8", newline="") as f:
        rows = [
            row
            for row in csv.DictReader(f, delimiter="\t")
            if row["source_split"] == split and row["variant"] == variant and normalize_label(row["label"])
        ]
    rows.sort(key=lambda row: (int(row["page"]), int(row["row_index"]), int(row["syllable_index"])))
    return rows


class SyllableDataset(Dataset):
    def __init__(self, root: Path, rows: list[dict[str, str]], label_to_id: dict[str, int], height: int, width: int):
        self.root = root
        self.rows = rows
        self.label_to_id = label_to_id
        self.height = height
        self.width = width

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        image = Image.open(self.root / row["image"]).convert("L")
        scale = self.height / image.height
        new_width = max(8, min(self.width, int(round(image.width * scale))))
        image = image.resize((new_width, self.height), Image.Resampling.BICUBIC)
        tensor = torch.full((1, self.height, self.width), 1.0, dtype=torch.float32)
        pixels = torch.from_numpy(np.asarray(image, dtype=np.float32)).unsqueeze(0).contiguous()
        pixels = (pixels / 255.0 - 0.5) / 0.5
        tensor[:, :, :new_width] = pixels
        label = normalize_label(row["label"])
        target = self.label_to_id.get(label, -1)
        return {
            "image": tensor,
            "target": torch.tensor(target, dtype=torch.long),
            "sample_id": row["sample_id"],
            "label": label,
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


class SyllableResNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
        )
        self.features = nn.Sequential(
            ResidualBlock(64, 96, stride=2),
            ResidualBlock(96, 96),
            ResidualBlock(96, 160, stride=2),
            ResidualBlock(160, 160),
            ResidualBlock(160, 256, stride=2),
            ResidualBlock(256, 256),
            ResidualBlock(256, 384, stride=2),
            ResidualBlock(384, 384),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Dropout(0.2), nn.Linear(384, num_classes))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(self.stem(images)))


def evaluate(model: nn.Module, loader: DataLoader, device: str, id_to_label: list[str]) -> tuple[float, list[dict[str, str]]]:
    model.eval()
    rows = []
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(device))
            pred_ids = logits.argmax(dim=1).cpu().tolist()
            targets = batch["target"].tolist()
            for sample_id, gold, target, pred_id in zip(batch["sample_id"], batch["label"], targets, pred_ids):
                pred = id_to_label[pred_id]
                ok = int(target >= 0 and pred_id == target)
                correct += ok
                total += 1
                rows.append({"sample_id": sample_id, "label": gold, "prediction": pred, "correct": ok})
    model.train()
    return correct / max(1, total), rows


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_rows = read_rows(args.eval_dir, "train", args.variant)
    val_rows = read_rows(args.eval_dir, "val", args.variant)
    test_rows = read_rows(args.eval_dir, "test", args.variant)
    labels = sorted({normalize_label(row["label"]) for row in train_rows})
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    (args.out_dir / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")

    train_loader = DataLoader(SyllableDataset(args.eval_dir, train_rows, label_to_id, args.height, args.width), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(SyllableDataset(args.eval_dir, val_rows, label_to_id, args.height, args.width), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(SyllableDataset(args.eval_dir, test_rows, label_to_id, args.height, args.width), batch_size=args.batch_size, shuffle=False)

    model = SyllableResNet(len(labels)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    best_val = -math.inf
    history = []
    ckpts = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        for batch in train_loader:
            targets = batch["target"].to(device)
            mask = targets >= 0
            if not bool(mask.any()):
                continue
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["image"].to(device))
            loss = criterion(logits[mask], targets[mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item())
            steps += 1
        scheduler.step()
        val_exact, _ = evaluate(model, val_loader, device, labels)
        best = ""
        if val_exact > best_val:
            best_val = val_exact
            best = "*"
            torch.save({"model": model.state_dict(), "labels": labels, "args": vars(args)}, args.out_dir / "best.pt")
        if args.save_every and epoch % args.save_every == 0:
            path = args.out_dir / f"epoch_{epoch:04d}.pt"
            torch.save({"model": model.state_dict(), "labels": labels, "args": vars(args)}, path)
            ckpts.append(path)
        row = {"epoch": epoch, "train_loss": total_loss / max(1, steps), "val_exact": val_exact, "best": best}
        history.append(row)
        with (args.out_dir / "history.tsv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()), delimiter="\t")
            writer.writeheader()
            writer.writerows(history)
        print(f"epoch {epoch}: train_loss={row['train_loss']:.4f} val_exact={val_exact:.4f}{best}")

    eval_rows = []
    for name, path in [("best.pt", args.out_dir / "best.pt")] + [(p.name, p) for p in ckpts]:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        test_exact, pred_rows = evaluate(model, test_loader, device, labels)
        eval_rows.append({"checkpoint": name, "test_exact": test_exact})
        if name == "best.pt":
            with (args.out_dir / f"predictions_{args.variant}.tsv").open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["sample_id", "prediction"], delimiter="\t")
                writer.writeheader()
                writer.writerows({"sample_id": row["sample_id"], "prediction": row["prediction"]} for row in pred_rows)
        print(f"{name}: test_exact={test_exact:.4f}")
    with (args.out_dir / "checkpoint_eval.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["checkpoint", "test_exact"], delimiter="\t")
        writer.writeheader()
        writer.writerows(eval_rows)


if __name__ == "__main__":
    main()
