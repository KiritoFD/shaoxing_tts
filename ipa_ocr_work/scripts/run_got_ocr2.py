"""Run GOT-OCR 2.0 on selected images.

GOT-OCR 2.0 is a generative end-to-end OCR model. This runner keeps the first
experiments tiny: one page or one crop at a time, with TSV output that can be
adapted/scored by the existing Wu-pinyin scripts.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_DIR = PROJECT_ROOT / "ipa_ocr_work" / "eval" / "quick_page129_thresholds"
DEFAULT_MODEL = "ucaslcl/GOT-OCR2_0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GOT-OCR 2.0.")
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--variants", nargs="+", default=["original"])
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--image-file", type=Path, default=None, help="Run a single image instead of an eval manifest.")
    parser.add_argument("--ocr-type", choices=("ocr", "format"), default="ocr")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def load_eval_images(eval_dir: Path, variants: list[str], limit: int) -> pd.DataFrame:
    df = pd.read_csv(eval_dir / "eval_manifest.tsv", sep="\t", keep_default_na=False)
    df = df[df["variant"].isin(variants)].copy()
    if limit:
        df = df.groupby("variant", group_keys=False).head(limit)
    return df.sort_values(["variant", "sample_id"]).reset_index(drop=True)


def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        use_safetensors=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return tokenizer, model


def main() -> None:
    args = parse_args()
    tokenizer, model = load_model(args.model)

    if args.image_file:
        rows = pd.DataFrame(
            [
                {
                    "sample_id": args.image_file.stem,
                    "variant": "single",
                    "image": str(args.image_file),
                    "label": "",
                }
            ]
        )
        root = Path(".")
    else:
        rows = load_eval_images(args.eval_dir, args.variants, args.limit)
        root = args.eval_dir

    predictions = []
    for i, row in rows.iterrows():
        image_path = Path(row["image"])
        if not image_path.is_absolute():
            image_path = root / image_path
        text = model.chat(tokenizer, str(image_path), ocr_type=args.ocr_type)
        predictions.append(
            {
                "sample_id": row["sample_id"],
                "variant": row["variant"],
                "prediction": str(text).strip().replace("\n", " "),
            }
        )
        print(f"{i + 1}/{len(rows)} {row['sample_id']} {row['variant']}: {predictions[-1]['prediction']}")

    out = args.out or (args.eval_dir / "predictions_got_ocr2.tsv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "variant", "prediction"], delimiter="\t")
        writer.writeheader()
        writer.writerows(predictions)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
