"""Run a pretrained TrOCR model on the enhancement A/B eval set."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_DIR = PROJECT_ROOT / "ipa_ocr_work" / "eval" / "enhancement_ab"
DEFAULT_MODEL = "microsoft/trocr-small-printed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pretrained TrOCR baseline.")
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--variants", nargs="+", default=["original", "superres_gray", "superres_otsu"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def main() -> None:
    args = parse_args()
    manifest_path = args.eval_dir / "eval_manifest.tsv"
    df = pd.read_csv(manifest_path, sep="\t")
    df = df[df["variant"].isin(args.variants)].copy()
    if args.limit:
        df = df.groupby("variant", group_keys=False).head(args.limit)
    df = df.sort_values(["variant", "sample_id"]).reset_index(drop=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = TrOCRProcessor.from_pretrained(args.model)
    model = VisionEncoderDecoderModel.from_pretrained(args.model).to(device)
    model.eval()

    predictions = []
    for start in range(0, len(df), args.batch_size):
        batch = df.iloc[start : start + args.batch_size]
        images = [load_image(args.eval_dir / image) for image in batch["image"]]
        pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            generated_ids = model.generate(pixel_values, max_new_tokens=args.max_new_tokens)
        texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        for (_, row), text in zip(batch.iterrows(), texts):
            predictions.append(
                {
                    "sample_id": row["sample_id"],
                    "variant": row["variant"],
                    "prediction": text.strip().replace(" ", ""),
                }
            )
        print(f"processed {min(start + args.batch_size, len(df))}/{len(df)}")

    out = args.out or (args.eval_dir / "predictions_trocr_small_printed.tsv")
    pd.DataFrame(predictions).to_csv(out, sep="\t", index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
