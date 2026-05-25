"""Generate synthetic TrOCR rows by mosaicing real syllable crops.

Unlike font-rendered synthesis, this keeps glyph shapes from the PDF-derived
syllable crops. It recombines legal observed syllables into new row images and
uses concatenated Wu-pinyin/IPA labels from the same syllable records.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REAL_ROW_DIR = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_pdf136_clean" / "trocr_direct_span_v1_matched_ok_conf80"
DEFAULT_SYLLABLE_DIR = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_pdf136_clean" / "syllable_ocr_core_strict"
DEFAULT_OUT = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_pdf136_clean" / "trocr_syllable_mosaic_v1_smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic row crops from real syllable crops.")
    parser.add_argument("--real-row-dir", type=Path, default=DEFAULT_REAL_ROW_DIR)
    parser.add_argument("--syllable-dir", type=Path, default=DEFAULT_SYLLABLE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--synthetic-count", type=int, default=2000)
    parser.add_argument("--max-syllables", type=int, default=6)
    parser.add_argument("--min-syllable-count", type=int, default=1)
    parser.add_argument("--syllable-cleaning-flags", default="ok")
    parser.add_argument("--max-syllable-width", type=int, default=360)
    parser.add_argument("--max-width-per-label-char", type=float, default=80.0)
    parser.add_argument("--include-real-train", action="store_true", default=True)
    parser.add_argument("--no-real-train", dest="include_real_train", action="store_false")
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--variant", default="syllable_mosaic_v1")
    parser.add_argument("--qa-count", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_manifest(root: Path) -> pd.DataFrame:
    return pd.read_csv(root / "eval_manifest.tsv", sep="\t", keep_default_na=False)


def trim_content(img: Image.Image, pad: int = 4) -> Image.Image:
    gray = img.convert("L")
    arr = np.asarray(gray)
    threshold = min(235, max(150, int(np.percentile(arr, 42))))
    mask = arr < threshold
    if not mask.any():
        return img.convert("L")
    ys, xs = np.where(mask)
    left = max(0, int(xs.min()) - pad)
    right = min(img.width, int(xs.max()) + pad + 1)
    top = max(0, int(ys.min()) - pad)
    bottom = min(img.height, int(ys.max()) + pad + 1)
    return gray.crop((left, top, right, bottom))


def real_profiles(real_row_dir: Path, rows: pd.DataFrame, rng: random.Random, limit: int = 1000) -> list[dict[str, float]]:
    train = rows[rows["source_split"].eq("train")]
    if len(train) > limit:
        train = train.sample(limit, random_state=rng.randrange(2**32))
    profiles = []
    for _, row in train.iterrows():
        path = real_row_dir / str(row["image"])
        if not path.exists():
            continue
        img = Image.open(path).convert("L")
        arr = np.asarray(img)
        profiles.append(
            {
                "width": float(img.width),
                "height": float(img.height),
                "bg": float(np.percentile(arr, 88)),
                "ink": float(np.percentile(arr, 18)),
            }
        )
    if not profiles:
        raise RuntimeError("No real row profiles found.")
    return profiles


def build_syllable_pool(args: argparse.Namespace, df: pd.DataFrame) -> pd.DataFrame:
    syllable_dir = args.syllable_dir
    train = df[df["source_split"].eq("train")].copy()
    if "quality" in train:
        train = train[train["quality"].eq("matched")].copy()
    if "ipa_conversion_status" in train:
        train = train[train["ipa_conversion_status"].eq("ok")].copy()
    if args.syllable_cleaning_flags:
        allowed_flags = {flag.strip() for flag in args.syllable_cleaning_flags.split(",") if flag.strip()}
        train = train[train["cleaning_flags"].isin(allowed_flags)].copy()
    keep_rows = []
    for _, row in train.iterrows():
        path = syllable_dir / str(row["image"])
        if not path.exists():
            continue
        with Image.open(path) as img:
            width = img.width
        label_len = max(1, len(str(row["label"])))
        if width > args.max_syllable_width:
            continue
        if width / label_len > args.max_width_per_label_char:
            continue
        keep_rows.append(row)
    train = pd.DataFrame(keep_rows)
    train["wupin"] = train["wupin_base"].astype(str) + train["selected_tone"].astype(str)
    train = train[train["label"].astype(str).ne("") & train["wupin"].astype(str).ne("")]
    return train.reset_index(drop=True)


def resize_to_height(img: Image.Image, height: int) -> Image.Image:
    scale = height / max(1, img.height)
    return img.resize((max(6, int(round(img.width * scale))), height), Image.Resampling.BICUBIC)


def composite_row(syllable_dir: Path, parts: pd.DataFrame, profile: dict[str, float], rng: random.Random) -> Image.Image:
    target_h = int(max(150, min(260, round(profile["height"] + rng.gauss(0, 5)))))
    content_h = rng.randint(int(target_h * 0.42), int(target_h * 0.66))
    bg = int(max(226, min(255, profile["bg"] + rng.gauss(0, 4))))
    tiles = []
    for _, row in parts.iterrows():
        tile = trim_content(Image.open(syllable_dir / str(row["image"])).convert("L"), pad=rng.randint(2, 8))
        tile = resize_to_height(tile, max(24, content_h + rng.randint(-8, 8)))
        tiles.append(tile)
    gaps = [rng.randint(-10, 10) for _ in range(max(0, len(tiles) - 1))]
    width = sum(tile.width for tile in tiles) + sum(gaps) + rng.randint(12, 42)
    width = int(max(90, min(1050, width)))
    canvas = Image.new("L", (width, target_h), bg)
    x = rng.randint(4, 18)
    baseline = rng.randint(int(target_h * 0.50), int(target_h * 0.66))
    for i, tile in enumerate(tiles):
        y = max(0, min(target_h - tile.height, baseline - int(tile.height * rng.uniform(0.68, 0.88)) + rng.randint(-5, 6)))
        canvas.paste(tile, (x, y))
        if i < len(gaps):
            x += tile.width + gaps[i]
    return degrade(canvas, rng)


def degrade(img: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() < 0.55:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.15, 0.85)))
    arr = np.asarray(img, dtype=np.float32)
    arr = (arr - 235.0) * rng.uniform(0.82, 1.22) + 235.0 + rng.uniform(-5, 8)
    arr += np.random.default_rng(rng.randrange(2**32)).normal(0, rng.uniform(0.8, 4.0), arr.shape)
    if rng.random() < 0.20:
        arr += np.linspace(rng.uniform(-4, 4), rng.uniform(-4, 4), arr.shape[1], dtype=np.float32)[None, :]
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, "L")
    if rng.random() < 0.14:
        threshold = rng.randint(145, 205)
        img = img.point(lambda p: 0 if p < threshold else 255).filter(ImageFilter.GaussianBlur(radius=0.25))
    if rng.random() < 0.18:
        img = ImageOps.expand(img, border=(rng.randint(0, 4), 0, rng.randint(0, 8), 0), fill=rng.randint(238, 255))
    if rng.random() < 0.16:
        img = ImageChops.offset(img, rng.randint(-3, 3), 0)
    return img.convert("RGB")


def relative_image_path(out_dir: Path, image_path: Path) -> str:
    try:
        return image_path.resolve().relative_to(out_dir.resolve(), walk_up=True).as_posix()
    except TypeError:
        return image_path.resolve().as_posix()


def write_manifest(rows: list[dict[str, object]], path: Path) -> None:
    fields = sorted({str(key) for row in rows for key in row})
    preferred = ["sample_id", "variant", "image", "label", "wupin", "source_split", "synthetic", "quality", "cleaning_flags", "syllable_count"]
    fields = [field for field in preferred if field in fields] + [field for field in fields if field not in preferred]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def make_contact_sheet(out_dir: Path, rows: list[dict[str, object]], count: int) -> str:
    sample = [row for row in rows if row.get("synthetic") == "1"][:count]
    if not sample:
        return ""
    cols, tile_w, tile_h = 4, 360, 92
    sheet = Image.new("RGB", (cols * tile_w, ((len(sample) + cols - 1) // cols) * tile_h), "white")
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(sample):
        img = Image.open(out_dir / str(row["image"])).convert("RGB")
        scale = min((tile_w - 12) / max(1, img.width), 58 / max(1, img.height))
        view = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.Resampling.BICUBIC)
        x = (i % cols) * tile_w + 6
        y = (i // cols) * tile_h + 24
        sheet.paste(view, (x, y))
        draw.rectangle((x, y, x + view.width, y + view.height), outline=(220, 30, 30), width=1)
        draw.text((x, (i // cols) * tile_h + 5), f"{row['sample_id']} n={row.get('syllable_count')}", fill=(0, 0, 0))
    path = out_dir / "qa_contactsheet.png"
    sheet.save(path)
    return str(path)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    if args.out_dir.exists() and args.overwrite:
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    real_rows = read_manifest(args.real_row_dir)
    syllable_rows = build_syllable_pool(args, read_manifest(args.syllable_dir))
    profiles = real_profiles(args.real_row_dir, real_rows, rng)
    label_counts = syllable_rows["label"].value_counts()
    weights = syllable_rows["label"].map(lambda label: float(label_counts[label]) ** -0.35).astype(float).to_list()

    out_rows: list[dict[str, object]] = []
    for idx in range(args.synthetic_count):
        count = rng.randint(args.min_syllable_count, args.max_syllables)
        parts = syllable_rows.sample(count, replace=True, weights=weights, random_state=rng.randrange(2**32))
        sample_id = f"synth_mosaic_{idx:06d}"
        image_name = f"{sample_id}.png"
        composite_row(args.syllable_dir, parts, rng.choice(profiles), rng).save(image_dir / image_name)
        out_rows.append(
            {
                "sample_id": sample_id,
                "variant": args.variant,
                "image": f"images/{image_name}",
                "label": "".join(parts["label"].astype(str).tolist()),
                "wupin": "".join(parts["wupin"].astype(str).tolist()),
                "source_split": "train",
                "synthetic": "1",
                "quality": "synthetic_mosaic",
                "cleaning_flags": "synthetic",
                "syllable_count": count,
            }
        )

    real_keep = real_rows[real_rows["source_split"].isin(["val", "test"])].copy()
    if args.include_real_train:
        real_keep = pd.concat([real_rows[real_rows["source_split"].eq("train")], real_keep], ignore_index=True)
    for _, row in real_keep.iterrows():
        copied = row.to_dict()
        copied["image"] = relative_image_path(args.out_dir, args.real_row_dir / str(row["image"]))
        copied["synthetic"] = "0"
        out_rows.append(copied)

    write_manifest(out_rows, args.out_dir / "eval_manifest.tsv")
    qa = make_contact_sheet(args.out_dir, out_rows, args.qa_count)
    summary = {
        "real_row_dir": str(args.real_row_dir),
        "syllable_dir": str(args.syllable_dir),
        "synthetic_count": args.synthetic_count,
        "real_rows": int(len(real_keep)),
        "rows": len(out_rows),
        "split_counts": pd.DataFrame(out_rows)["source_split"].value_counts().to_dict(),
        "syllable_pool": int(len(syllable_rows)),
        "unique_syllable_labels": int(syllable_rows["label"].nunique()),
        "qa_contactsheet": qa,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
