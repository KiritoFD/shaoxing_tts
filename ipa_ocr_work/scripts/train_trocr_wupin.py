"""Fine-tune TrOCR to read enhanced crops as Wu-pinyin."""

from __future__ import annotations

import argparse
import csv
import copy
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_DIR = PROJECT_ROOT / "ipa_ocr_work" / "eval" / "enhancement_ab"
DEFAULT_OUT = PROJECT_ROOT / "ipa_ocr_work" / "models" / "trocr_wupin"
DEFAULT_MODEL = "microsoft/trocr-small-printed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TrOCR on Wu-pinyin crops.")
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--variant", default="superres_gray")
    parser.add_argument("--train-variants", nargs="+", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-label-length", type=int, default=48)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--save-each-best", action="store_true", default=True)
    return parser.parse_args()


class WupinDataset(Dataset):
    def __init__(self, root: Path, rows: pd.DataFrame, processor: TrOCRProcessor, max_label_length: int):
        self.root = root
        self.rows = rows.reset_index(drop=True)
        self.processor = processor
        self.max_label_length = max_label_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows.iloc[idx]
        image = Image.open(self.root / row["image"]).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.processor.tokenizer(
            str(row["label"]),
            padding="max_length",
            max_length=self.max_label_length,
            truncation=True,
        ).input_ids
        labels = [token if token != self.processor.tokenizer.pad_token_id else -100 for token in labels]
        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(labels, dtype=torch.long),
            "sample_id": row["sample_id"],
            "variant": row["variant"],
            "label": row["label"],
        }


def collate(batch: list[dict]) -> dict:
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "sample_id": [item["sample_id"] for item in batch],
        "variant": [item["variant"] for item in batch],
        "label": [item["label"] for item in batch],
    }


def load_split(eval_dir: Path, variants: list[str], split: str) -> pd.DataFrame:
    df = pd.read_csv(eval_dir / "eval_manifest.tsv", sep="\t")
    df = df[(df["variant"].isin(variants)) & (df["source_split"] == split)].copy()
    return df.sort_values(["page", "row_index"]).reset_index(drop=True)


def configure_model(model: VisionEncoderDecoderModel, processor: TrOCRProcessor) -> None:
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.generation_config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    model.generation_config.eos_token_id = processor.tokenizer.sep_token_id
    model.generation_config.max_length = 48
    model.generation_config.early_stopping = False
    model.generation_config.no_repeat_ngram_size = 0
    model.generation_config.length_penalty = 1.0
    model.generation_config.num_beams = 1


def evaluate_loss(model, loader, device: str) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                pixel_values=batch["pixel_values"].to(device),
                labels=batch["labels"].to(device),
            )
            total += float(outputs.loss.item())
            count += 1
    model.train()
    return total / max(1, count)


def generate_predictions(model, processor, loader, device: str, max_new_tokens: int = 48) -> list[dict]:
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            generated = model.generate(
                batch["pixel_values"].to(device),
                max_new_tokens=max_new_tokens,
                num_beams=1,
            )
            texts = processor.batch_decode(generated, skip_special_tokens=True)
            for sample_id, variant, text in zip(batch["sample_id"], batch["variant"], texts):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "variant": variant,
                        "prediction": text.strip().replace(" ", ""),
                    }
                )
    model.train()
    return rows


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = TrOCRProcessor.from_pretrained(args.model)
    model = VisionEncoderDecoderModel.from_pretrained(args.model)
    configure_model(model, processor)
    model.to(device)

    train_variants = args.train_variants or [args.variant]
    train_rows = load_split(args.eval_dir, train_variants, "train")
    val_rows = load_split(args.eval_dir, [args.variant], "val")
    test_rows = load_split(args.eval_dir, [args.variant], "test")
    if args.eval_limit:
        train_rows = train_rows.head(args.eval_limit)
        val_rows = val_rows.head(max(1, args.eval_limit // 4))
        test_rows = test_rows.head(max(1, args.eval_limit // 4))

    train_ds = WupinDataset(args.eval_dir, train_rows, processor, args.max_label_length)
    val_ds = WupinDataset(args.eval_dir, val_rows, processor, args.max_label_length)
    test_ds = WupinDataset(args.eval_dir, test_rows, processor, args.max_label_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history = []
    best_val = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        steps = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                pixel_values=batch["pixel_values"].to(device),
                labels=batch["labels"].to(device),
            )
            outputs.loss.backward()
            optimizer.step()
            total += float(outputs.loss.item())
            steps += 1
        train_loss = total / max(1, steps)
        val_loss = evaluate_loss(model, val_loader, device)
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            if args.save_each_best:
                model.save_pretrained(args.out_dir / "best")
                processor.save_pretrained(args.out_dir / "best")
            best_marker = "*"
        else:
            best_marker = ""
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "best": best_marker})
        print(f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}{best_marker}")
        pd.DataFrame(history).to_csv(args.out_dir / "history.tsv", sep="\t", index=False)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.save_pretrained(args.out_dir)
    processor.save_pretrained(args.out_dir)
    pd.DataFrame(history).to_csv(args.out_dir / "history.tsv", sep="\t", index=False)

    pred_rows = generate_predictions(model, processor, test_loader, device)
    pred_path = args.out_dir / f"predictions_{args.variant}.tsv"
    with pred_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "variant", "prediction"], delimiter="\t")
        writer.writeheader()
        writer.writerows(pred_rows)
    print(f"wrote {pred_path}")


if __name__ == "__main__":
    main()
