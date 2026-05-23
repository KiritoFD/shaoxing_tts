"""Train CTC OCR models for Shaoxing IPA+digit labels.

This avoids the large generative tokenizer used by TrOCR. The model can only
emit characters that occur in the training manifest, plus the CTC blank.
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
DEFAULT_EVAL_DIR = PROJECT_ROOT / "ipa_ocr_work" / "eval" / "ipa_digits_original_threshold180"
DEFAULT_OUT = PROJECT_ROOT / "ipa_ocr_work" / "models" / "crnn_ipa_digits"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CRNN+CTC on IPA+digit crops.")
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--variant", default="original")
    parser.add_argument("--train-variants", nargs="+", default=["original", "threshold_180"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--max-width", type=int, default=640)
    parser.add_argument("--backbone", choices=["crnn", "svtr_tiny", "resnet_transformer"], default="crnn")
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=0)
    return parser.parse_args()


def normalize_label(text: object) -> str:
    return unicodedata.normalize("NFC", str(text)).strip().replace(" ", "")


def load_rows(eval_dir: Path, variants: list[str], split: str, limit: int = 0) -> list[dict[str, str]]:
    rows = []
    with (eval_dir / "eval_manifest.tsv").open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("source_split") != split or row.get("variant") not in variants:
                continue
            row["label"] = normalize_label(row.get("label", ""))
            if row["label"]:
                rows.append(row)
    rows.sort(key=lambda row: (int(row.get("page", 0)), int(row.get("row_index", 0)), row.get("variant", "")))
    if limit:
        rows = rows[:limit]
    return rows


def build_charset(*frames: list[dict[str, str]]) -> list[str]:
    chars = sorted({ch for rows in frames for row in rows for ch in row["label"]})
    return ["<blank>"] + chars


class OcrDataset(Dataset):
    def __init__(self, root: Path, rows: list[dict[str, str]], char_to_id: dict[str, int], height: int, max_width: int):
        self.root = root
        self.rows = list(rows)
        self.char_to_id = char_to_id
        self.height = height
        self.max_width = max_width

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        image = Image.open(self.root / row["image"]).convert("L")
        scale = self.height / image.height
        width = max(8, min(self.max_width, int(round(image.width * scale))))
        image = image.resize((width, self.height), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32)
        tensor = torch.from_numpy(array).unsqueeze(0).contiguous()
        tensor = (tensor / 255.0 - 0.5) / 0.5
        label = normalize_label(row["label"])
        target = torch.tensor([self.char_to_id[ch] for ch in label], dtype=torch.long)
        return {
            "image": tensor,
            "width": width,
            "target": target,
            "target_len": len(target),
            "sample_id": row["sample_id"],
            "variant": row["variant"],
            "label": label,
        }


def collate(batch: list[dict]) -> dict:
    max_width = max(item["width"] for item in batch)
    images = []
    for item in batch:
        pad_width = max_width - item["width"]
        images.append(F.pad(item["image"], (0, pad_width, 0, 0), value=1.0))
    return {
        "images": torch.stack(images),
        "widths": torch.tensor([item["width"] for item in batch], dtype=torch.long),
        "targets": torch.cat([item["target"] for item in batch]),
        "target_lens": torch.tensor([item["target_len"] for item in batch], dtype=torch.long),
        "sample_id": [item["sample_id"] for item in batch],
        "variant": [item["variant"] for item in batch],
        "label": [item["label"] for item in batch],
    }


class Crnn(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
        )
        self.rnn = nn.LSTM(256, 192, num_layers=2, bidirectional=True, batch_first=True, dropout=0.15)
        self.head = nn.Linear(384, num_classes)

    @staticmethod
    def output_lengths(widths: torch.Tensor) -> torch.Tensor:
        return torch.clamp(torch.div(torch.div(widths, 2, rounding_mode="floor"), 2, rounding_mode="floor"), min=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.cnn(images)
        x = x.mean(dim=2).transpose(1, 2)
        x, _ = self.rnn(x)
        return self.head(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: tuple[int, int] = (1, 1)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels or stride != (1, 1):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual, inplace=True)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class SvtrBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, drop_path: float = 0.0, local: bool = False):
        super().__init__()
        self.local = local
        self.norm1 = nn.LayerNorm(dim)
        if local:
            self.mixer = nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
                nn.BatchNorm2d(dim),
                nn.GELU(),
                nn.Conv2d(dim, dim, 1),
            )
        else:
            self.mixer = nn.MultiheadAttention(dim, num_heads, dropout=0.1, batch_first=True)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, dim),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        residual = x
        y = self.norm1(x)
        if self.local:
            y = y.transpose(1, 2).reshape(x.size(0), x.size(2), height, width)
            y = self.mixer(y).flatten(2).transpose(1, 2)
        else:
            y, _ = self.mixer(y, y, y, need_weights=False)
        x = residual + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SvtrTinyCtc(nn.Module):
    """SVTR-style recognizer: patch embedding, local/global mixing, CTC head."""

    def __init__(self, num_classes: int, d_model: int = 192, max_width: int = 640):
        super().__init__()
        self.patch_embed = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, d_model, 3, stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
        )
        max_tokens = ((48 + 3) // 4) * (max_width // 4 + 4)
        self.pos = nn.Parameter(torch.zeros(1, max_tokens, d_model))
        drop_rates = torch.linspace(0, 0.12, 6).tolist()
        self.blocks = nn.ModuleList(
            [
                SvtrBlock(d_model, 6, drop_path=drop_rates[0], local=True),
                SvtrBlock(d_model, 6, drop_path=drop_rates[1], local=True),
                SvtrBlock(d_model, 6, drop_path=drop_rates[2], local=True),
                SvtrBlock(d_model, 6, drop_path=drop_rates[3], local=False),
                SvtrBlock(d_model, 6, drop_path=drop_rates[4], local=False),
                SvtrBlock(d_model, 6, drop_path=drop_rates[5], local=False),
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    @staticmethod
    def output_lengths(widths: torch.Tensor) -> torch.Tensor:
        return torch.clamp(torch.div(torch.div(widths, 2, rounding_mode="floor"), 2, rounding_mode="floor"), min=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(images)
        batch, channels, height, width = x.shape
        x = x.flatten(2).transpose(1, 2)
        if x.size(1) > self.pos.size(1):
            raise ValueError(f"sequence length {x.size(1)} exceeds positional table {self.pos.size(1)}")
        x = x + self.pos[:, : x.size(1)]
        for block in self.blocks:
            x = block(x, height, width)
        x = self.norm(x)
        x = x.reshape(batch, height, width, channels).mean(dim=1)
        return self.head(x)


class ResnetTransformerCtc(nn.Module):
    """Legacy stronger baseline kept for old checkpoints."""

    def __init__(self, num_classes: int, d_model: int = 384, max_width: int = 640):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 48, 3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            ResidualBlock(48, 96, stride=(2, 2)),
            ResidualBlock(96, 96),
            ResidualBlock(96, 192, stride=(2, 2)),
            ResidualBlock(192, 192),
            ResidualBlock(192, d_model, stride=(2, 1)),
            ResidualBlock(d_model, d_model, stride=(2, 1)),
        )
        self.pos = nn.Parameter(torch.zeros(1, max_width // 4 + 8, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=6,
            dim_feedforward=d_model * 4,
            dropout=0.15,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    @staticmethod
    def output_lengths(widths: torch.Tensor) -> torch.Tensor:
        return torch.clamp(torch.div(torch.div(widths, 2, rounding_mode="floor"), 2, rounding_mode="floor"), min=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.cnn(images)
        x = x.mean(dim=2).transpose(1, 2)
        if x.size(1) > self.pos.size(1):
            raise ValueError(f"sequence length {x.size(1)} exceeds positional table {self.pos.size(1)}")
        x = x + self.pos[:, : x.size(1)]
        x = self.encoder(x)
        return self.head(self.norm(x))


def build_model(backbone: str, num_classes: int, max_width: int) -> nn.Module:
    if backbone == "crnn":
        return Crnn(num_classes)
    if backbone == "svtr_tiny":
        return SvtrTinyCtc(num_classes, max_width=max_width)
    if backbone == "resnet_transformer":
        return ResnetTransformerCtc(num_classes, max_width=max_width)
    raise ValueError(f"unknown backbone: {backbone}")


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def decode(logits: torch.Tensor, lengths: torch.Tensor, id_to_char: list[str]) -> list[str]:
    ids = logits.argmax(dim=-1).cpu()
    out = []
    for seq, length in zip(ids, lengths.cpu()):
        chars = []
        prev = 0
        for token in seq[: int(length)]:
            token = int(token)
            if token != 0 and token != prev:
                chars.append(id_to_char[token])
            prev = token
        out.append("".join(chars))
    return out


def evaluate(model: Crnn, loader: DataLoader, criterion: nn.CTCLoss, device: str, id_to_char: list[str]) -> tuple[float, float, float, list[dict]]:
    model.eval()
    total_loss = 0.0
    steps = 0
    edits = 0
    chars = 0
    exact = 0
    rows = []
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            widths = batch["widths"].to(device)
            logits = model(images)
            input_lens = model.output_lengths(widths)
            loss = criterion(logits.log_softmax(2).transpose(0, 1), batch["targets"].to(device), input_lens, batch["target_lens"].to(device))
            total_loss += float(loss.item())
            steps += 1
            preds = decode(logits, input_lens, id_to_char)
            for sample_id, variant, label, pred in zip(batch["sample_id"], batch["variant"], batch["label"], preds):
                ed = edit_distance(pred, label)
                edits += ed
                chars += len(label)
                exact += int(pred == label)
                rows.append({"sample_id": sample_id, "variant": variant, "label": label, "prediction": pred})
    model.train()
    return total_loss / max(1, steps), exact / max(1, len(rows)), edits / max(1, chars), rows


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_rows = load_rows(args.eval_dir, args.train_variants, "train", args.eval_limit)
    val_rows = load_rows(args.eval_dir, [args.variant], "val", max(1, args.eval_limit // 4) if args.eval_limit else 0)
    test_rows = load_rows(args.eval_dir, [args.variant], "test", max(1, args.eval_limit // 4) if args.eval_limit else 0)
    charset = build_charset(train_rows, val_rows, test_rows)
    char_to_id = {ch: idx for idx, ch in enumerate(charset)}
    (args.out_dir / "charset.json").write_text(json.dumps(charset, ensure_ascii=False, indent=2), encoding="utf-8")

    train_loader = DataLoader(
        OcrDataset(args.eval_dir, train_rows, char_to_id, args.height, args.max_width),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        OcrDataset(args.eval_dir, val_rows, char_to_id, args.height, args.max_width),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        OcrDataset(args.eval_dir, test_rows, char_to_id, args.height, args.max_width),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    model = build_model(args.backbone, len(charset), args.max_width).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    best_val_cer = math.inf
    history = []
    checkpoint_paths = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        steps = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            images = batch["images"].to(device)
            widths = batch["widths"].to(device)
            logits = model(images)
            input_lens = model.output_lengths(widths)
            loss = criterion(logits.log_softmax(2).transpose(0, 1), batch["targets"].to(device), input_lens, batch["target_lens"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.item())
            steps += 1
        scheduler.step()
        train_loss = total / max(1, steps)
        val_loss, val_exact, val_cer, _ = evaluate(model, val_loader, criterion, device, charset)
        best = ""
        if val_cer < best_val_cer:
            best_val_cer = val_cer
            best = "*"
            torch.save({"model": model.state_dict(), "charset": charset, "args": vars(args)}, args.out_dir / "best.pt")
        if args.save_every and epoch % args.save_every == 0:
            ckpt_path = args.out_dir / f"epoch_{epoch:04d}.pt"
            torch.save({"model": model.state_dict(), "charset": charset, "args": vars(args)}, ckpt_path)
            checkpoint_paths.append(ckpt_path)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_exact": val_exact, "val_cer": val_cer, "best": best}
        )
        with (args.out_dir / "history.tsv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["epoch", "train_loss", "val_loss", "val_exact", "val_cer", "best"],
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerows(history)
        print(f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_exact={val_exact:.4f} val_cer={val_cer:.4f}{best}")

    checkpoint = torch.load(args.out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    _, test_exact, test_cer, pred_rows = evaluate(model, test_loader, criterion, device, charset)
    with (args.out_dir / f"predictions_{args.variant}.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "variant", "prediction"], delimiter="\t")
        writer.writeheader()
        writer.writerows({k: row[k] for k in ["sample_id", "variant", "prediction"]} for row in pred_rows)
    torch.save({"model": model.state_dict(), "charset": charset, "args": vars(args)}, args.out_dir / "last_loaded_best.pt")
    print(f"test_exact={test_exact:.4f} test_cer={test_cer:.4f}")
    print(f"wrote {args.out_dir / f'predictions_{args.variant}.tsv'}")

    eval_rows = [{"checkpoint": "best.pt", "test_exact": test_exact, "test_cer": test_cer}]
    for ckpt_path in checkpoint_paths:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        _, ckpt_exact, ckpt_cer, _ = evaluate(model, test_loader, criterion, device, charset)
        eval_rows.append({"checkpoint": ckpt_path.name, "test_exact": ckpt_exact, "test_cer": ckpt_cer})
        print(f"{ckpt_path.name}: test_exact={ckpt_exact:.4f} test_cer={ckpt_cer:.4f}")
    with (args.out_dir / "checkpoint_eval.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["checkpoint", "test_exact", "test_cer"], delimiter="\t")
        writer.writeheader()
        writer.writerows(eval_rows)


if __name__ == "__main__":
    main()
