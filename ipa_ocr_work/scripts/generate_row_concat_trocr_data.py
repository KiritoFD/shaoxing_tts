"""Generate TrOCR training augmentation by concatenating clean real row crops.

This is deliberately conservative: it only recombines rows that already have
trusted round-trip-stable labels.  Validation and test rows are copied as-is;
only the training split receives synthetic concatenations.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import itertools
import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = (
    PROJECT_ROOT
    / "ipa_ocr_work"
    / "dataset"
    / "shaoxing_pdf136_clean"
    / "trocr_direct_span_all_clean_conf70_nobracket_roundtrip_exact"
)
DEFAULT_OUT = (
    PROJECT_ROOT
    / "ipa_ocr_work"
    / "dataset"
    / "shaoxing_pdf136_clean"
    / "trocr_direct_span_all_clean_conf70_nobracket_roundtrip_exact_concat_v1"
)

WORKER_RECORDS: list[dict[str, object]] = []
WORKER_CUM_WEIGHTS: list[float] = []
WORKER_IMAGE_CACHE: dict[str, Image.Image] = {}
WORKER_ARGS: argparse.Namespace | None = None
WORKER_OUT_DIR: Path | None = None
WORKER_SEED = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Concatenate clean TrOCR row crops for augmentation.")
    parser.add_argument("--src-dir", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--synthetic-count", type=int, default=6000)
    parser.add_argument("--min-parts", type=int, default=2)
    parser.add_argument("--max-parts", type=int, default=4)
    parser.add_argument("--max-label-length", type=int, default=88)
    parser.add_argument("--max-width", type=int, default=1150)
    parser.add_argument("--target-height-min", type=int, default=170)
    parser.add_argument("--target-height-max", type=int, default=240)
    parser.add_argument("--variant", default="row_concat_v1")
    parser.add_argument("--base-variant", default="direct_phonetic_span_v1")
    parser.add_argument("--include-real", action="store_true", default=True)
    parser.add_argument("--no-real", dest="include_real", action="store_false")
    parser.add_argument("--qa-count", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--flush-every", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_manifest(src_dir: Path) -> pd.DataFrame:
    return pd.read_csv(src_dir / "eval_manifest.tsv", sep="\t", keep_default_na=False)


def resolve_image(src_dir: Path, image: object) -> Path:
    path = Path(str(image))
    return path if path.is_absolute() else src_dir / path


def trim_white(image: Image.Image, pad: int = 3) -> Image.Image:
    gray = image.convert("L")
    arr = np.asarray(gray)
    threshold = min(248, max(180, int(np.percentile(arr, 40)) + 35))
    mask = arr < threshold
    if not mask.any():
        return image.convert("RGB")
    ys, xs = np.where(mask)
    left = max(0, int(xs.min()) - pad)
    right = min(image.width, int(xs.max()) + pad + 1)
    top = max(0, int(ys.min()) - pad)
    bottom = min(image.height, int(ys.max()) + pad + 1)
    return image.crop((left, top, right, bottom)).convert("RGB")


def resize_height(image: Image.Image, height: int) -> Image.Image:
    scale = height / max(1, image.height)
    return image.resize((max(6, int(round(image.width * scale))), height), Image.Resampling.BICUBIC)


def degrade(image: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() < 0.55:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.05, 0.45)))
    if rng.random() < 0.65:
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.88, 1.16))
    if rng.random() < 0.65:
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.94, 1.05))
    arr = np.asarray(image.convert("L"), dtype=np.float32)
    arr += np.random.default_rng(rng.randrange(2**32)).normal(0, rng.uniform(0.0, 2.5), arr.shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    out = Image.fromarray(arr, "L").convert("RGB")
    if rng.random() < 0.25:
        out = ImageOps.expand(out, border=(rng.randint(0, 8), rng.randint(0, 3), rng.randint(0, 12), rng.randint(0, 3)), fill="white")
    return out


def load_image_cache(src_dir: Path, rows: pd.DataFrame) -> dict[str, Image.Image]:
    return load_image_cache_records(src_dir, rows.to_dict("records"))


def load_image_cache_records(src_dir: Path, rows: list[dict[str, object]]) -> dict[str, Image.Image]:
    cache: dict[str, Image.Image] = {}
    for row in rows:
        image = str(row["image"])
        if image in cache:
            continue
        path = resolve_image(src_dir, image)
        if not path.exists():
            continue
        cache[image] = trim_white(Image.open(path).convert("RGB"), pad=5)
    return cache


def concat_images(
    image_cache: dict[str, Image.Image],
    rows: list[dict[str, object]],
    rng: random.Random,
    args: argparse.Namespace,
) -> Image.Image | None:
    target_h = rng.randint(args.target_height_min, args.target_height_max)
    tiles = []
    for row in rows:
        cached = image_cache.get(str(row["image"]))
        if cached is None:
            return None
        tile = cached.copy()
        if rng.random() < 0.35:
            tile = ImageOps.expand(
                tile,
                border=(rng.randint(0, 3), rng.randint(0, 2), rng.randint(0, 4), rng.randint(0, 2)),
                fill="white",
            )
        tile = resize_height(tile, rng.randint(max(28, int(target_h * 0.40)), max(30, int(target_h * 0.62))))
        tiles.append(tile)
    gaps = [rng.randint(-4, 14) for _ in range(len(tiles) - 1)]
    left_pad = rng.randint(4, 22)
    right_pad = rng.randint(10, 42)
    width = left_pad + right_pad + sum(tile.width for tile in tiles) + sum(gaps)
    if width > args.max_width:
        return None
    canvas = Image.new("RGB", (max(48, width), target_h), "white")
    x = left_pad
    baseline = rng.randint(int(target_h * 0.54), int(target_h * 0.70))
    for i, tile in enumerate(tiles):
        y = baseline - int(tile.height * rng.uniform(0.70, 0.88)) + rng.randint(-5, 5)
        y = max(0, min(target_h - tile.height, y))
        canvas.paste(tile, (x, y))
        x += tile.width + (gaps[i] if i < len(gaps) else 0)
    return degrade(canvas, rng)


def copy_real_rows(src_dir: Path, out_dir: Path, rows: pd.DataFrame, base_variant: str) -> list[dict[str, object]]:
    out_rows = []
    for row in rows.to_dict("records"):
        image = str(row["image"])
        src = resolve_image(src_dir, image)
        dst = out_dir / image
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        row = dict(row)
        row["variant"] = base_variant
        row["synthetic"] = "0"
        out_rows.append(row)
    return out_rows


def make_contact_sheet(out_dir: Path, rows: list[dict[str, object]], limit: int) -> str:
    sample = [row for row in rows if str(row.get("synthetic", "")) == "1"][:limit]
    if not sample:
        return ""
    cols, tile_w, tile_h = 3, 420, 110
    sheet = Image.new("RGB", (cols * tile_w, ((len(sample) + cols - 1) // cols) * tile_h), "white")
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(sample):
        img = Image.open(out_dir / str(row["image"])).convert("RGB")
        scale = min((tile_w - 12) / max(1, img.width), 66 / max(1, img.height))
        view = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.Resampling.BICUBIC)
        x = (i % cols) * tile_w + 6
        y = (i // cols) * tile_h + 28
        sheet.paste(view, (x, y))
        draw.rectangle((x, y, x + view.width, y + view.height), outline=(210, 45, 35), width=1)
        draw.text((x, (i // cols) * tile_h + 7), f"{row['sample_id']} parts={row.get('concat_parts')} len={len(str(row.get('label','')))}", fill=(0, 0, 0))
    path = out_dir / "qa_concat_contactsheet.png"
    sheet.save(path)
    return str(path)


def write_manifest(rows: list[dict[str, object]], path: Path) -> None:
    preferred = [
        "sample_id",
        "variant",
        "image",
        "label",
        "wupin",
        "source_split",
        "synthetic",
        "quality",
        "cleaning_flags",
        "concat_parts",
        "concat_source_ids",
    ]
    fields = sorted({key for row in rows for key in row})
    fields = [field for field in preferred if field in fields] + [field for field in fields if field not in preferred]
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def write_summary(args: argparse.Namespace, out_dir: Path, out_rows: list[dict[str, object]], attempts: int, contact_sheet: str = "") -> None:
    out_df = pd.DataFrame(out_rows)
    summary = {
        "src_dir": str(args.src_dir),
        "rows": int(len(out_df)),
        "synthetic_rows": int((out_df.get("synthetic", "") == "1").sum()),
        "real_rows": int((out_df.get("synthetic", "") == "0").sum()),
        "split_counts": out_df["source_split"].value_counts().to_dict(),
        "variant_counts": out_df["variant"].value_counts().to_dict(),
        "synthetic_count_requested": args.synthetic_count,
        "attempts": attempts,
        "max_label_length": args.max_label_length,
        "max_width": args.max_width,
        "contact_sheet": contact_sheet,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def sample_parts(
    records: list[dict[str, object]],
    cum_weights: list[float],
    n: int,
    rng: random.Random,
) -> list[dict[str, object]]:
    indices = rng.choices(range(len(records)), cum_weights=cum_weights, k=n)
    return [records[index] for index in indices]


def init_concat_worker(
    src_dir: str,
    out_dir: str,
    records: list[dict[str, object]],
    cum_weights: list[float],
    args_dict: dict[str, object],
    seed: int,
) -> None:
    global WORKER_RECORDS, WORKER_CUM_WEIGHTS, WORKER_IMAGE_CACHE, WORKER_ARGS, WORKER_OUT_DIR, WORKER_SEED
    WORKER_RECORDS = records
    WORKER_CUM_WEIGHTS = cum_weights
    WORKER_ARGS = argparse.Namespace(**args_dict)
    WORKER_ARGS.src_dir = Path(str(WORKER_ARGS.src_dir))
    WORKER_ARGS.out_dir = Path(str(WORKER_ARGS.out_dir))
    WORKER_OUT_DIR = Path(out_dir)
    WORKER_SEED = seed
    WORKER_IMAGE_CACHE = load_image_cache_records(Path(src_dir), records)


def make_synthetic_row_worker(index: int) -> tuple[dict[str, object], int]:
    if WORKER_ARGS is None or WORKER_OUT_DIR is None:
        raise RuntimeError("worker not initialized")
    rng = random.Random(WORKER_SEED + index * 1_000_003 + os.getpid())
    attempts = 0
    while attempts < 1000:
        attempts += 1
        n = rng.randint(WORKER_ARGS.min_parts, WORKER_ARGS.max_parts)
        parts = sample_parts(WORKER_RECORDS, WORKER_CUM_WEIGHTS, n, rng)
        label = "".join(str(part["label"]) for part in parts)
        wupin = "".join(str(part["wupin"]) for part in parts)
        if len(label) > WORKER_ARGS.max_label_length:
            continue
        image = concat_images(WORKER_IMAGE_CACHE, parts, rng, WORKER_ARGS)
        if image is None:
            continue
        sample_id = f"concat_{index:06d}"
        image_rel = f"images/{sample_id}.png"
        image.save(WORKER_OUT_DIR / image_rel)
        return (
            {
                "sample_id": sample_id,
                "variant": WORKER_ARGS.variant,
                "image": image_rel,
                "label": label,
                "ipa": label,
                "wupin": wupin,
                "source_split": "train",
                "synthetic": "1",
                "quality": "row_concat",
                "cleaning_flags": "synthetic_roundtrip_exact_concat",
                "ipa_conversion_status": "ok",
                "concat_parts": n,
                "concat_source_ids": ";".join(str(part["sample_id"]) for part in parts),
            },
            attempts,
        )
    raise RuntimeError(f"failed to make synthetic row index={index}")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    if args.out_dir.exists() and args.overwrite:
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_manifest(args.src_dir)
    train = manifest[manifest["source_split"].eq("train")].copy().reset_index(drop=True)
    usable = train[train["label"].astype(str).map(len).between(1, args.max_label_length)].copy()
    if usable.empty:
        raise SystemExit("no usable train rows")
    label_counts = usable["label"].value_counts()
    weights = usable["label"].map(lambda label: float(label_counts[label]) ** -0.35).to_list()
    cum_weights = list(itertools.accumulate(float(weight) for weight in weights))
    usable_records = usable.to_dict("records")
    out_rows: list[dict[str, object]] = []
    if args.include_real:
        out_rows.extend(copy_real_rows(args.src_dir, args.out_dir, manifest, args.base_variant))
        write_manifest(out_rows, args.out_dir / "eval_manifest.tsv")

    made = 0
    attempts = 0
    start = time.time()
    if args.workers > 1:
        worker_args = vars(args).copy()
        worker_args["src_dir"] = str(args.src_dir)
        worker_args["out_dir"] = str(args.out_dir)
        print(f"starting {args.workers} workers for {args.synthetic_count} synthetic rows", flush=True)
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=init_concat_worker,
            initargs=(str(args.src_dir), str(args.out_dir), usable_records, cum_weights, worker_args, args.seed),
        ) as executor:
            futures = [executor.submit(make_synthetic_row_worker, index) for index in range(args.synthetic_count)]
            for future in as_completed(futures):
                row, row_attempts = future.result()
                out_rows.append(row)
                attempts += row_attempts
                made += 1
                if args.progress_every > 0 and made % args.progress_every == 0:
                    elapsed = max(0.001, time.time() - start)
                    rate = made / elapsed
                    print(f"made={made}/{args.synthetic_count} attempts={attempts} rate={rate:.1f}/s", flush=True)
                if args.flush_every > 0 and made % args.flush_every == 0:
                    write_manifest(out_rows, args.out_dir / "eval_manifest.tsv")
                    write_summary(args, args.out_dir, out_rows, attempts)
    else:
        image_cache = load_image_cache(args.src_dir, usable)
        print(f"loaded image cache: {len(image_cache)}/{len(usable_records)} train crops", flush=True)
        while made < args.synthetic_count and attempts < args.synthetic_count * 80:
            attempts += 1
            n = rng.randint(args.min_parts, args.max_parts)
            parts = sample_parts(usable_records, cum_weights, n, rng)
            label = "".join(str(part["label"]) for part in parts)
            wupin = "".join(str(part["wupin"]) for part in parts)
            if len(label) > args.max_label_length:
                continue
            image = concat_images(image_cache, parts, rng, args)
            if image is None:
                continue
            sample_id = f"concat_{made:06d}"
            image_rel = f"images/{sample_id}.png"
            image.save(args.out_dir / image_rel)
            out_rows.append(
                {
                    "sample_id": sample_id,
                    "variant": args.variant,
                    "image": image_rel,
                    "label": label,
                    "ipa": label,
                    "wupin": wupin,
                    "source_split": "train",
                    "synthetic": "1",
                    "quality": "row_concat",
                    "cleaning_flags": "synthetic_roundtrip_exact_concat",
                    "ipa_conversion_status": "ok",
                    "concat_parts": n,
                    "concat_source_ids": ";".join(str(part["sample_id"]) for part in parts),
                }
            )
            made += 1
            if args.progress_every > 0 and made % args.progress_every == 0:
                elapsed = max(0.001, time.time() - start)
                rate = made / elapsed
                print(f"made={made}/{args.synthetic_count} attempts={attempts} rate={rate:.1f}/s", flush=True)
            if args.flush_every > 0 and made % args.flush_every == 0:
                write_manifest(out_rows, args.out_dir / "eval_manifest.tsv")
                write_summary(args, args.out_dir, out_rows, attempts)

    out_rows.sort(key=lambda row: (str(row.get("source_split")) != "train", str(row.get("sample_id"))))
    write_manifest(out_rows, args.out_dir / "eval_manifest.tsv")
    contact_sheet = make_contact_sheet(args.out_dir, out_rows, args.qa_count)
    write_summary(args, args.out_dir, out_rows, attempts, contact_sheet=contact_sheet)
    summary = json.loads((args.out_dir / "summary.json").read_text(encoding="utf-8"))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {args.out_dir / 'eval_manifest.tsv'}")


if __name__ == "__main__":
    main()
