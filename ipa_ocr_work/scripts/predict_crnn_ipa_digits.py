"""Run a trained CTC OCR model on an eval manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from train_crnn_ipa_digits import (
    OcrDataset,
    build_model,
    collate,
    decode,
    load_rows,
    normalize_label,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict IPA+digit labels with a trained CTC model.")
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variant", default="original_export")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--max-width", type=int, default=640)
    return parser.parse_args()


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    charset = checkpoint["charset"]
    char_to_id = {ch: idx for idx, ch in enumerate(charset)}

    rows = []
    for split in args.splits:
        rows.extend(load_rows(args.eval_dir, [args.variant], split))
    dataset = OcrDataset(args.eval_dir, rows, char_to_id, args.height, args.max_width)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    backbone = checkpoint.get("args", {}).get("backbone", "crnn")
    max_width = int(checkpoint.get("args", {}).get("max_width", args.max_width))
    model = build_model(backbone, len(charset), max_width).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "variant", "label", "prediction", "edit_distance", "cer"],
            delimiter="\t",
        )
        writer.writeheader()
        with torch.no_grad():
            for batch in loader:
                images = batch["images"].to(device)
                widths = batch["widths"].to(device)
                logits = model(images)
                input_lens = model.output_lengths(widths)
                preds = decode(logits, input_lens, charset)
                for sample_id, variant, label, pred in zip(batch["sample_id"], batch["variant"], batch["label"], preds):
                    label = normalize_label(label)
                    ed = edit_distance(pred, label)
                    writer.writerow(
                        {
                            "sample_id": sample_id,
                            "variant": variant,
                            "label": label,
                            "prediction": pred,
                            "edit_distance": ed,
                            "cer": ed / max(1, len(label)),
                        }
                    )
    print(f"rows: {len(rows)}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
