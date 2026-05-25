"""Train a syllable OCR classifier with separate IPA-base and tone heads.

The closed-set whole-syllable classifier is easy to dominate by frequent
``base+tone`` classes. This version decomposes the target into ``ipa_base`` and
``selected_tone`` so rare tone/base combinations still share supervision.
Predictions are written as ``base+tone`` and can be scored by
``score_syllable_ocr_rows.py`` unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_pdf136_clean" / "syllable_ocr_all"
DEFAULT_OUT = PROJECT_ROOT / "ipa_ocr_work" / "models" / "syllable_multitask_classifier"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train syllable base+tone classifier.")
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--variant", default="syllable_crop")
    parser.add_argument("--qualities", nargs="*", default=["matched", "weak_match"])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--height", type=int, default=72)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--tone-loss-weight", type=float, default=0.7)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--architecture", choices=["seq", "pool"], default="seq")
    parser.add_argument("--class-weight-power", type=float, default=0.0)
    parser.add_argument("--sampler-power", type=float, default=0.5)
    parser.add_argument("--exclude-flag-substrings", nargs="*", default=["bracket"])
    parser.add_argument("--balanced-sampler", action="store_true", default=True)
    parser.add_argument("--no-balanced-sampler", dest="balanced_sampler", action="store_false")
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=10)
    return parser.parse_args()


def normalize_label(text: object) -> str:
    return unicodedata.normalize("NFC", str(text)).strip().replace(" ", "")


def read_rows(
    eval_dir: Path,
    split: str,
    variant: str,
    qualities: set[str],
    exclude_flag_substrings: list[str],
) -> list[dict[str, str]]:
    with (eval_dir / "eval_manifest.tsv").open("r", encoding="utf-8", newline="") as f:
        rows = []
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("source_split") != split or row.get("variant") != variant:
                continue
            if qualities and row.get("quality", "") not in qualities:
                continue
            flags = str(row.get("cleaning_flags", ""))
            if any(token and token in flags for token in exclude_flag_substrings):
                continue
            base = normalize_label(row.get("ipa_base", ""))
            tone = normalize_label(row.get("selected_tone", ""))
            label = normalize_label(row.get("label", ""))
            if not base or not tone or not label:
                continue
            rows.append(row)
    rows.sort(key=lambda row: (int(row["page"]), int(row["row_index"]), int(row["syllable_index"])))
    return rows


def resize_on_canvas(image: Image.Image, height: int, width: int) -> torch.Tensor:
    image = image.convert("L")
    scale = height / image.height
    new_width = max(8, min(width, int(round(image.width * scale))))
    image = image.resize((new_width, height), Image.Resampling.BICUBIC)
    tensor = torch.full((1, height, width), 1.0, dtype=torch.float32)
    pixels = torch.from_numpy(np.asarray(image, dtype=np.float32)).unsqueeze(0).contiguous()
    pixels = (pixels / 255.0 - 0.5) / 0.5
    tensor[:, :, :new_width] = pixels
    return tensor


def augment_image(image: Image.Image) -> Image.Image:
    if random.random() < 0.35:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.0, 0.55)))
    if random.random() < 0.45:
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.82, 1.22))
    if random.random() < 0.45:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.88, 1.10))
    if random.random() < 0.35:
        dx = random.randint(-3, 3)
        dy = random.randint(-2, 2)
        canvas = Image.new("L", image.size, 255)
        canvas.paste(image, (dx, dy))
        image = canvas
    return image


class SyllableMultiTaskDataset(Dataset):
    def __init__(
        self,
        root: Path,
        rows: list[dict[str, str]],
        base_to_id: dict[str, int],
        tone_to_id: dict[str, int],
        height: int,
        width: int,
        augment: bool,
    ):
        self.root = root
        self.rows = rows
        self.base_to_id = base_to_id
        self.tone_to_id = tone_to_id
        self.height = height
        self.width = width
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        image = Image.open(self.root / row["image"]).convert("L")
        if self.augment:
            image = augment_image(image)
        base = normalize_label(row["ipa_base"])
        tone = normalize_label(row["selected_tone"])
        return {
            "image": resize_on_canvas(image, self.height, self.width),
            "base_target": torch.tensor(self.base_to_id.get(base, -1), dtype=torch.long),
            "tone_target": torch.tensor(self.tone_to_id.get(tone, -1), dtype=torch.long),
            "sample_id": row["sample_id"],
            "label": normalize_label(row["label"]),
            "base": base,
            "tone": tone,
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


class AttentionPool2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Conv2d(channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        weights = self.score(x).reshape(batch, 1, height * width)
        weights = F.softmax(weights, dim=-1)
        values = x.reshape(batch, channels, height * width)
        return torch.bmm(values, weights.transpose(1, 2)).squeeze(-1)


class SyllableMultiTaskNet(nn.Module):
    def __init__(self, num_bases: int, num_tones: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 48, 3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True),
        )
        self.features = nn.Sequential(
            ResidualBlock(48, 80, stride=2),
            ResidualBlock(80, 80),
            ResidualBlock(80, 128, stride=2),
            ResidualBlock(128, 128),
            ResidualBlock(128, 224, stride=2),
            ResidualBlock(224, 224),
            ResidualBlock(224, 320, stride=2),
            ResidualBlock(320, 320),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.attn_pool = AttentionPool2d(320)
        self.neck = nn.Sequential(nn.Linear(640, 384), nn.SiLU(inplace=True), nn.Dropout(0.25))
        self.base_head = nn.Linear(384, num_bases)
        self.tone_head = nn.Linear(384, num_tones)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features(self.stem(images))
        avg = self.avg_pool(x).flatten(1)
        attn = self.attn_pool(x)
        features = self.neck(torch.cat([avg, attn], dim=1))
        return self.base_head(features), self.tone_head(features)


class SequenceAttentionPool(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.score(x).transpose(1, 2), dim=-1)
        return torch.bmm(weights, x).squeeze(1)


class SyllableSequenceNet(nn.Module):
    """OCR-oriented classifier that preserves the horizontal feature sequence."""

    def __init__(self, num_bases: int, num_tones: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 48, 3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True),
        )
        self.features = nn.Sequential(
            ResidualBlock(48, 80, stride=2),
            ResidualBlock(80, 80),
            ResidualBlock(80, 128, stride=2),
            ResidualBlock(128, 128),
            ResidualBlock(128, 192, stride=2),
            ResidualBlock(192, 192),
            ResidualBlock(192, 256, stride=2),
            ResidualBlock(256, 256),
        )
        self.rnn = nn.LSTM(256, 192, num_layers=2, bidirectional=True, batch_first=True, dropout=0.15)
        self.base_pool = SequenceAttentionPool(384)
        self.tone_pool = SequenceAttentionPool(384)
        self.base_head = nn.Sequential(nn.Dropout(0.25), nn.Linear(384, num_bases))
        self.tone_head = nn.Sequential(nn.Dropout(0.2), nn.Linear(384, num_tones))

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features(self.stem(images))
        x = x.mean(dim=2).transpose(1, 2)
        x, _ = self.rnn(x)
        return self.base_head(self.base_pool(x)), self.tone_head(self.tone_pool(x))


def build_model(architecture: str, num_bases: int, num_tones: int) -> nn.Module:
    if architecture == "seq":
        return SyllableSequenceNet(num_bases, num_tones)
    if architecture == "pool":
        return SyllableMultiTaskNet(num_bases, num_tones)
    raise ValueError(f"unknown architecture: {architecture}")


def class_weights(values: list[str], vocab: list[str], device: str, power: float) -> torch.Tensor | None:
    if power <= 0:
        return None
    counts = Counter(values)
    weights = []
    for item in vocab:
        weights.append(1.0 / (max(1, counts[item]) ** power))
    tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    return tensor / tensor.mean()


def sampler_weights(rows: list[dict[str, str]], power: float) -> list[float]:
    combo_counts = Counter((normalize_label(row["ipa_base"]), normalize_label(row["selected_tone"])) for row in rows)
    base_counts = Counter(normalize_label(row["ipa_base"]) for row in rows)
    weights = []
    for row in rows:
        combo = (normalize_label(row["ipa_base"]), normalize_label(row["selected_tone"]))
        base = normalize_label(row["ipa_base"])
        weights.append(0.7 / (combo_counts[combo] ** power) + 0.3 / (base_counts[base] ** power))
    return weights


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    bases: list[str],
    tones: list[str],
) -> tuple[dict[str, float], list[dict[str, str]]]:
    model.eval()
    rows = []
    base_ok = tone_ok = label_ok = total = 0
    with torch.no_grad():
        for batch in loader:
            base_logits, tone_logits = model(batch["image"].to(device))
            base_ids = base_logits.argmax(dim=1).cpu().tolist()
            tone_ids = tone_logits.argmax(dim=1).cpu().tolist()
            base_targets = batch["base_target"].tolist()
            tone_targets = batch["tone_target"].tolist()
            for sample_id, label, gold_base, gold_tone, base_target, tone_target, base_id, tone_id in zip(
                batch["sample_id"],
                batch["label"],
                batch["base"],
                batch["tone"],
                base_targets,
                tone_targets,
                base_ids,
                tone_ids,
            ):
                pred_base = bases[base_id]
                pred_tone = tones[tone_id]
                pred = pred_base + pred_tone
                b_ok = int(base_target >= 0 and base_id == base_target)
                t_ok = int(tone_target >= 0 and tone_id == tone_target)
                ok = int(pred == label)
                base_ok += b_ok
                tone_ok += t_ok
                label_ok += ok
                total += 1
                rows.append(
                    {
                        "sample_id": sample_id,
                        "label": label,
                        "prediction": pred,
                        "base": gold_base,
                        "tone": gold_tone,
                        "base_prediction": pred_base,
                        "tone_prediction": pred_tone,
                        "base_correct": b_ok,
                        "tone_correct": t_ok,
                        "correct": ok,
                    }
                )
    model.train()
    metrics = {
        "base_exact": base_ok / max(1, total),
        "tone_exact": tone_ok / max(1, total),
        "label_exact": label_ok / max(1, total),
    }
    return metrics, rows


def write_predictions(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "prediction"], delimiter="\t")
        writer.writeheader()
        writer.writerows({"sample_id": row["sample_id"], "prediction": row["prediction"]} for row in rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    qualities = set(args.qualities)

    train_rows = read_rows(args.eval_dir, "train", args.variant, qualities, args.exclude_flag_substrings)
    val_rows = read_rows(args.eval_dir, "val", args.variant, qualities, args.exclude_flag_substrings)
    test_rows = read_rows(args.eval_dir, "test", args.variant, qualities, args.exclude_flag_substrings)
    bases = sorted({normalize_label(row["ipa_base"]) for row in train_rows})
    tones = sorted({normalize_label(row["selected_tone"]) for row in train_rows})
    base_to_id = {base: idx for idx, base in enumerate(bases)}
    tone_to_id = {tone: idx for idx, tone in enumerate(tones)}

    metadata = {
        "eval_dir": str(args.eval_dir),
        "qualities": sorted(qualities),
        "train": len(train_rows),
        "val": len(val_rows),
        "test": len(test_rows),
        "num_bases": len(bases),
        "num_tones": len(tones),
        "architecture": args.architecture,
        "exclude_flag_substrings": args.exclude_flag_substrings,
        "bases": bases,
        "tones": tones,
    }
    (args.out_dir / "labels.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in metadata.items() if k not in {"bases", "tones"}}, ensure_ascii=False))

    train_ds = SyllableMultiTaskDataset(args.eval_dir, train_rows, base_to_id, tone_to_id, args.height, args.width, args.augment)
    val_ds = SyllableMultiTaskDataset(args.eval_dir, val_rows, base_to_id, tone_to_id, args.height, args.width, False)
    test_ds = SyllableMultiTaskDataset(args.eval_dir, test_rows, base_to_id, tone_to_id, args.height, args.width, False)
    sampler = None
    shuffle = True
    if args.balanced_sampler:
        sampler = WeightedRandomSampler(sampler_weights(train_rows, args.sampler_power), num_samples=len(train_rows), replacement=True)
        shuffle = False

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(args.architecture, len(bases), len(tones)).to(device)
    base_weight = class_weights([normalize_label(row["ipa_base"]) for row in train_rows], bases, device, args.class_weight_power)
    tone_weight = class_weights([normalize_label(row["selected_tone"]) for row in train_rows], tones, device, args.class_weight_power)
    base_criterion = nn.CrossEntropyLoss(weight=base_weight, label_smoothing=args.label_smoothing)
    tone_criterion = nn.CrossEntropyLoss(weight=tone_weight, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    best_score = -math.inf
    history = []
    ckpts = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        for batch in train_loader:
            base_targets = batch["base_target"].to(device)
            tone_targets = batch["tone_target"].to(device)
            mask = (base_targets >= 0) & (tone_targets >= 0)
            if not bool(mask.any()):
                continue
            optimizer.zero_grad(set_to_none=True)
            base_logits, tone_logits = model(batch["image"].to(device))
            loss = base_criterion(base_logits[mask], base_targets[mask]) + args.tone_loss_weight * tone_criterion(
                tone_logits[mask], tone_targets[mask]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item())
            steps += 1
        scheduler.step()

        val_metrics, _ = evaluate(model, val_loader, device, bases, tones)
        best = ""
        score = val_metrics["label_exact"]
        if score > best_score:
            best_score = score
            best = "*"
            torch.save({"model": model.state_dict(), "metadata": metadata, "args": vars(args)}, args.out_dir / "best.pt")
        if args.save_every and epoch % args.save_every == 0:
            ckpt_path = args.out_dir / f"epoch_{epoch:04d}.pt"
            torch.save({"model": model.state_dict(), "metadata": metadata, "args": vars(args)}, ckpt_path)
            ckpts.append(ckpt_path)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, steps),
            "val_base_exact": val_metrics["base_exact"],
            "val_tone_exact": val_metrics["tone_exact"],
            "val_label_exact": val_metrics["label_exact"],
            "best": best,
        }
        history.append(row)
        with (args.out_dir / "history.tsv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()), delimiter="\t")
            writer.writeheader()
            writer.writerows(history)
        print(
            f"epoch {epoch}: train_loss={row['train_loss']:.4f} "
            f"val_base={row['val_base_exact']:.4f} val_tone={row['val_tone_exact']:.4f} "
            f"val_label={row['val_label_exact']:.4f}{best}"
        )

    eval_rows = []
    for name, path in [("best.pt", args.out_dir / "best.pt")] + [(p.name, p) for p in ckpts]:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        test_metrics, pred_rows = evaluate(model, test_loader, device, bases, tones)
        eval_rows.append({"checkpoint": name, **{f"test_{k}": v for k, v in test_metrics.items()}})
        if name == "best.pt":
            write_predictions(args.out_dir / f"predictions_{args.variant}.tsv", pred_rows)
        print(
            f"{name}: test_base={test_metrics['base_exact']:.4f} "
            f"test_tone={test_metrics['tone_exact']:.4f} test_label={test_metrics['label_exact']:.4f}"
        )
    with (args.out_dir / "checkpoint_eval.tsv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["checkpoint", "test_base_exact", "test_tone_exact", "test_label_exact"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(eval_rows)


if __name__ == "__main__":
    main()
