"""Generate synthetic IPA OCR data and mixed manifests.

The synthetic training rows use rendered IPA+digit labels. Validation and test
rows are copied from the real clean manifest so reported metrics stay real.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REAL_ROW = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_pdf136_clean" / "ocr_selected_all"
DEFAULT_REAL_SYLLABLE = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_pdf136_clean" / "syllable_ocr_all"
DEFAULT_OUT = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_synthetic_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic Shaoxing IPA OCR manifests.")
    parser.add_argument("--real-row-dir", type=Path, default=DEFAULT_REAL_ROW)
    parser.add_argument("--real-syllable-dir", type=Path, default=DEFAULT_REAL_SYLLABLE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--row-count", type=int, default=50000)
    parser.add_argument("--syllable-count", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--font", type=Path, default=None)
    parser.add_argument("--max-row-syllables", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_manifest(root: Path) -> pd.DataFrame:
    return pd.read_csv(root / "eval_manifest.tsv", sep="\t", keep_default_na=False)


def normalize_label(text: object) -> str:
    return unicodedata.normalize("NFC", str(text)).strip().replace(" ", "")


def candidate_fonts() -> list[Path]:
    roots = [
        Path("C:/Windows/Fonts"),
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
        Path.home() / ".fonts",
        Path.home() / ".local/share/fonts",
    ]
    names = [
        "CharisSIL",
        "Doulos",
        "NotoSerif",
        "NotoSans",
        "DejaVuSerif",
        "DejaVuSans",
        "Times",
        "Arial",
        "Cambria",
        "NotoSansSC",
        "NotoSerifSC",
    ]
    out: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for suffix in ("*.ttf", "*.otf", "*.ttc"):
            for path in root.rglob(suffix):
                if any(name.lower() in path.name.lower() for name in names):
                    out.append(path)
    return out


def font_support_score(path: Path, chars: set[str]) -> int:
    try:
        font = ImageFont.truetype(str(path), 42)
    except Exception:
        return -1
    score = 0
    for ch in chars:
        try:
            mask = font.getmask(ch)
            bbox = mask.getbbox()
            if bbox and (bbox[2] - bbox[0]) > 0 and (bbox[3] - bbox[1]) > 0:
                score += 1
        except Exception:
            pass
    return score


def choose_font(path: Path | None, labels: list[str]) -> Path:
    chars = {ch for label in labels for ch in label}
    if path:
        return path
    scored = [(font_support_score(font, chars), font) for font in candidate_fonts()]
    scored = [item for item in scored if item[0] >= 0]
    if not scored:
        raise RuntimeError("No usable font found. Install Noto/DejaVu/Charis or pass --font.")
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def split_label_units(label: str) -> list[tuple[str, str]]:
    units: list[tuple[str, str]] = []
    base = []
    tone = []
    for ch in label:
        if ch.isdigit():
            tone.append(ch)
        else:
            if tone:
                units.append(("".join(base), "".join(tone)))
                base = [ch]
                tone = []
            else:
                base.append(ch)
    if base or tone:
        units.append(("".join(base), "".join(tone)))
    return [(b, t) for b, t in units if b or t]


def render_label(label: str, font_path: Path, rng: random.Random, mode: str) -> Image.Image:
    label = normalize_label(label)
    base_size = rng.randint(34, 48) if mode == "row" else rng.randint(42, 58)
    tone_size = max(15, int(base_size * rng.uniform(0.42, 0.56)))
    font = ImageFont.truetype(str(font_path), base_size)
    tone_font = ImageFont.truetype(str(font_path), tone_size)
    units = split_label_units(label)

    widths = []
    max_above = 0
    max_below = 0
    for base, tone in units:
        base_bbox = font.getbbox(base or " ")
        tone_bbox = tone_font.getbbox(tone or " ")
        base_w = max(1, base_bbox[2] - base_bbox[0])
        tone_w = max(0, tone_bbox[2] - tone_bbox[0]) if tone else 0
        widths.append(base_w + max(0, int(tone_w * 0.45)))
        max_above = max(max_above, tone_size)
        max_below = max(max_below, base_bbox[3] - base_bbox[1])

    pad_x = rng.randint(8, 26)
    pad_y = rng.randint(6, 20)
    width = min(1400, max(32, sum(widths) + 2 * pad_x + rng.randint(0, 18)))
    height = max(48, max_above + max_below + 2 * pad_y + rng.randint(0, 12))
    img = Image.new("L", (width, height), color=rng.randint(238, 255))
    draw = ImageDraw.Draw(img)
    x = pad_x
    baseline_y = pad_y + max_above + rng.randint(-2, 3)
    ink = rng.randint(0, 45)
    for base, tone in units:
        draw.text((x, baseline_y), base, font=font, fill=ink, anchor="ls")
        base_bbox = font.getbbox(base or " ")
        base_w = max(1, base_bbox[2] - base_bbox[0])
        if tone:
            tone_x = x + max(1, int(base_w * rng.uniform(0.62, 0.92)))
            tone_y = baseline_y - int(base_size * rng.uniform(0.62, 0.86))
            draw.text((tone_x, tone_y), tone, font=tone_font, fill=ink)
        x += base_w + rng.randint(0, 4)

    return degrade(img, rng)


def degrade(img: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() < 0.75:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.15, 1.05)))
    arr = np.asarray(img, dtype=np.float32)
    contrast = rng.uniform(0.55, 1.35)
    brightness = rng.uniform(-8, 14)
    arr = (arr - 220.0) * contrast + 220.0 + brightness
    if rng.random() < 0.75:
        arr += rng.normalvariate(0, rng.uniform(1.0, 5.0))
        noise = np.random.default_rng(rng.randrange(2**32)).normal(0, rng.uniform(1.5, 7.0), arr.shape)
        arr += noise
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L")
    if rng.random() < 0.25:
        threshold = rng.randint(150, 215)
        img = img.point(lambda p: 0 if p < threshold else 255)
        if rng.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.15, 0.55)))
    if rng.random() < 0.35:
        scale_x = rng.uniform(0.86, 1.16)
        scale_y = rng.uniform(0.94, 1.08)
        img = img.resize((max(16, int(img.width * scale_x)), max(32, int(img.height * scale_y))), Image.Resampling.BICUBIC)
    return img


def real_image_path(target_root: Path, real_root: Path, image: str) -> str:
    src = (real_root / image).resolve()
    try:
        return src.relative_to(target_root.resolve(), walk_up=True).as_posix()
    except TypeError:
        return src.as_posix()


def sample_row_labels(real_rows: pd.DataFrame, syllable_labels: list[str], rng: random.Random, n: int, max_syllables: int) -> list[str]:
    train_labels = [normalize_label(x) for x in real_rows[real_rows["source_split"].eq("train")]["label"] if normalize_label(x)]
    labels = []
    for _ in range(n):
        if rng.random() < 0.65 and train_labels:
            labels.append(rng.choice(train_labels))
        else:
            count = rng.randint(1, max_syllables)
            labels.append("".join(rng.choice(syllable_labels) for _ in range(count)))
    return labels


def write_manifest(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    preferred = ["sample_id", "variant", "image", "label", "source_split", "pdf_page", "page", "row_index", "hanzi", "wupin", "synthetic"]
    fields = [f for f in preferred if f in fields] + [f for f in fields if f not in preferred]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def build_row_dataset(real_root: Path, real_rows: pd.DataFrame, syllable_labels: list[str], out_root: Path, font_path: Path, rng: random.Random, count: int, max_syllables: int) -> dict:
    root = out_root / "ocr_mixed"
    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    labels = sample_row_labels(real_rows, syllable_labels, rng, count, max_syllables)
    for idx, label in enumerate(labels):
        name = f"synth_row_{idx:06d}.png"
        render_label(label, font_path, rng, "row").save(image_dir / name)
        rows.append(
            {
                "sample_id": f"synth_row_{idx:06d}",
                "variant": "original_export",
                "image": f"images/{name}",
                "label": label,
                "source_split": "train",
                "synthetic": "1",
            }
        )
    for _, row in real_rows.iterrows():
        copied = row.to_dict()
        copied["image"] = real_image_path(root, real_root, row["image"])
        copied["synthetic"] = "0"
        rows.append(copied)
    write_manifest(rows, root / "eval_manifest.tsv")
    return {"rows": len(rows), "synthetic_train": count, "root": str(root)}


def build_syllable_dataset(real_root: Path, real_rows: pd.DataFrame, out_root: Path, font_path: Path, rng: random.Random, count: int) -> dict:
    root = out_root / "syllable_mixed"
    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    train_labels = [normalize_label(x) for x in real_rows[real_rows["source_split"].eq("train")]["label"] if normalize_label(x)]
    all_labels = [normalize_label(x) for x in real_rows["label"] if normalize_label(x)]
    rows: list[dict[str, str]] = []
    for idx in range(count):
        label = rng.choice(train_labels or all_labels)
        name = f"synth_syll_{idx:06d}.png"
        render_label(label, font_path, rng, "syllable").save(image_dir / name)
        rows.append(
            {
                "sample_id": f"synth_syll_{idx:06d}",
                "variant": "syllable_crop",
                "image": f"images/{name}",
                "label": label,
                "source_split": "train",
                "synthetic": "1",
            }
        )
    for _, row in real_rows.iterrows():
        copied = row.to_dict()
        copied["image"] = real_image_path(root, real_root, row["image"])
        copied["synthetic"] = "0"
        rows.append(copied)
    write_manifest(rows, root / "eval_manifest.tsv")
    return {"rows": len(rows), "synthetic_train": count, "root": str(root)}


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    if args.out_dir.exists() and args.overwrite:
        import shutil

        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    row_df = read_manifest(args.real_row_dir)
    syll_df = read_manifest(args.real_syllable_dir)
    syllable_labels = sorted({normalize_label(x) for x in syll_df["label"] if normalize_label(x)})
    font = choose_font(args.font, list(row_df["label"]) + syllable_labels)
    row_summary = build_row_dataset(args.real_row_dir, row_df, syllable_labels, args.out_dir, font, rng, args.row_count, args.max_row_syllables)
    syll_summary = build_syllable_dataset(args.real_syllable_dir, syll_df, args.out_dir, font, rng, args.syllable_count)
    summary = {
        "font": str(font),
        "seed": args.seed,
        "row": row_summary,
        "syllable": syll_summary,
        "real_row_dir": str(args.real_row_dir),
        "real_syllable_dir": str(args.real_syllable_dir),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
