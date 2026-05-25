"""Generate rule-valid synthetic TrOCR row data.

This generator samples legal Wu-pinyin syllables from the trusted training
manifest, converts them to IPA labels with the project mapping, and renders
document-like row crops. Validation/test rows are copied from real data so model
selection remains anchored to real OCR performance.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from wupin_ipa_convert import DEFAULT_MAP, load_mapping, split_syllables, wupin_to_ipa  # noqa: E402


DEFAULT_REAL_DIR = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_pdf136_clean" / "trocr_direct_span_v1_matched_ok_conf80"
DEFAULT_OUT = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_pdf136_clean" / "trocr_rule_synth_v2_smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate rule-valid synthetic TrOCR rows.")
    parser.add_argument("--real-dir", type=Path, default=DEFAULT_REAL_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--font", type=Path, default=None)
    parser.add_argument("--synthetic-count", type=int, default=2000)
    parser.add_argument("--real-train-limit", type=int, default=0, help="0 keeps all real train rows.")
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--variant", default="rule_synth_v2")
    parser.add_argument("--real-row-prob", type=float, default=0.45)
    parser.add_argument("--max-syllables", type=int, default=6)
    parser.add_argument("--qa-count", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_manifest(root: Path) -> pd.DataFrame:
    return pd.read_csv(root / "eval_manifest.tsv", sep="\t", keep_default_na=False)


def normalize(text: object) -> str:
    return unicodedata.normalize("NFC", str(text)).strip().replace(" ", "")


def candidate_fonts() -> list[Path]:
    explicit = [
        Path("C:/Windows/Fonts/simsun.ttc"),
        Path("C:/Windows/Fonts/simsunb.ttf"),
        Path("C:/Windows/Fonts/NotoSerifSC-VF.ttf"),
        Path("C:/Windows/Fonts/NotoSansSC-VF.ttf"),
        Path("C:/Windows/Fonts/Noto Sans SC (TrueType).otf"),
        Path("C:/Windows/Fonts/mingliub.ttc"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/cambria.ttc"),
        Path("C:/Windows/Fonts/times.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    roots = [Path("C:/Windows/Fonts"), Path("/usr/share/fonts"), Path("/usr/local/share/fonts")]
    out = [path for path in explicit if path.exists()]
    names = ("simsun", "noto", "ming", "song", "serif", "times", "cambria", "arial")
    for root in roots:
        if not root.exists():
            continue
        for suffix in ("*.ttf", "*.otf", "*.ttc"):
            for path in root.rglob(suffix):
                if any(name in path.name.lower() for name in names) and path not in out:
                    out.append(path)
    return out


def rendered_ink_score(font_path: Path, chars: set[str]) -> float:
    try:
        font = ImageFont.truetype(str(font_path), 72)
    except Exception:
        return -1.0
    scores = []
    for ch in chars:
        img = Image.new("L", (110, 110), 255)
        ImageDraw.Draw(img).text((10, 82), ch, fill=0, font=font, anchor="ls")
        arr = np.asarray(img)
        ink = int((arr < 230).sum())
        scores.append(ink)
    if not scores:
        return -1.0
    # Missing-glyph boxes tend to have near-identical ink counts. Penalize very
    # low diversity while still preferring fonts that draw all target chars.
    return float(sum(scores)) + 10.0 * float(np.std(scores))


def choose_font(path: Path | None, labels: list[str]) -> Path:
    if path:
        return path
    chars = {ch for label in labels for ch in normalize(label)}
    scored = [(rendered_ink_score(font, chars), font) for font in candidate_fonts()]
    scored = [item for item in scored if item[0] > 0]
    if not scored:
        raise RuntimeError("No usable font found; pass --font.")
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def sample_real_profiles(real_dir: Path, df: pd.DataFrame, rng: random.Random, limit: int = 1200) -> list[dict[str, float]]:
    rows = df[df["source_split"].eq("train")].copy()
    if len(rows) > limit:
        rows = rows.sample(limit, random_state=rng.randrange(2**32))
    profiles = []
    for _, row in rows.iterrows():
        path = real_dir / str(row["image"])
        if not path.exists():
            continue
        img = Image.open(path).convert("L")
        arr = np.asarray(img)
        ink = arr[arr < np.percentile(arr, 42)]
        bg = arr[arr > np.percentile(arr, 72)]
        profiles.append(
            {
                "width": float(img.width),
                "height": float(img.height),
                "ink": float(np.median(ink)) if len(ink) else 55.0,
                "bg": float(np.median(bg)) if len(bg) else 248.0,
                "contrast": float(np.std(arr)),
            }
        )
    if not profiles:
        raise RuntimeError(f"No real image profiles found under {real_dir}")
    return profiles


def build_syllable_pool(train_rows: pd.DataFrame) -> tuple[list[str], list[int], list[str]]:
    syll_counter: Counter[str] = Counter()
    length_counter: Counter[int] = Counter()
    row_wupins = []
    for wupin in train_rows["wupin"]:
        wupin = normalize(wupin).lower()
        syllables, remainder = split_syllables(wupin)
        if remainder or not syllables:
            continue
        row_wupins.append("".join(base + tone for base, tone in syllables))
        length_counter[len(syllables)] += 1
        for base, tone in syllables:
            syll_counter[base + tone] += 1
    syllables = list(syll_counter.keys())
    weights = [syll_counter[s] for s in syllables]
    lengths = []
    for length, count in length_counter.items():
        lengths.extend([length] * count)
    return syllables, weights, row_wupins


def synthesize_wupin(syllables: list[str], weights: list[int], row_wupins: list[str], rng: random.Random, real_row_prob: float, max_syllables: int) -> str:
    if row_wupins and rng.random() < real_row_prob:
        return rng.choice(row_wupins)
    count = min(max_syllables, max(1, int(round(rng.triangular(1, max_syllables, 2)))))
    return "".join(rng.choices(syllables, weights=weights, k=count))


def split_ipa_units(label: str) -> list[tuple[str, str]]:
    units: list[tuple[str, str]] = []
    base: list[str] = []
    tone: list[str] = []
    for ch in normalize(label):
        if ch.isdigit():
            tone.append(ch)
            continue
        if tone:
            units.append(("".join(base), "".join(tone)))
            base = [ch]
            tone = []
        else:
            base.append(ch)
    if base or tone:
        units.append(("".join(base), "".join(tone)))
    return [(base, tone) for base, tone in units if base or tone]


def render_label(label: str, font_path: Path, profile: dict[str, float], rng: random.Random) -> Image.Image:
    target_h = int(max(140, min(260, round(profile["height"] + rng.gauss(0, 5)))))
    bg = int(max(224, min(255, profile["bg"] + rng.gauss(0, 4))))
    ink = int(max(8, min(95, profile["ink"] + rng.gauss(0, 12))))
    base_size = rng.randint(max(58, int(target_h * 0.36)), max(74, int(target_h * 0.52)))
    tone_size = max(18, int(base_size * rng.uniform(0.38, 0.52)))
    font = ImageFont.truetype(str(font_path), base_size)
    tone_font = ImageFont.truetype(str(font_path), tone_size)
    units = split_ipa_units(label)
    pad_x = rng.randint(10, 28)
    max_above = int(base_size * rng.uniform(0.48, 0.62))
    baseline_y = rng.randint(int(target_h * 0.50), int(target_h * 0.68))
    widths: list[int] = []
    for base, tone in units:
        bbox = font.getbbox(base or " ")
        base_w = max(1, bbox[2] - bbox[0])
        tone_w = tone_font.getlength(tone) if tone else 0
        widths.append(int(base_w + max(0, tone_w * rng.uniform(0.12, 0.38)) + rng.randint(0, 4)))
    natural_w = max(48, sum(widths) + 2 * pad_x)
    sampled_w = int(max(160, min(980, round(profile["width"] + rng.gauss(0, 36)))))
    target_w = int(max(natural_w + rng.randint(0, 24), min(1050, sampled_w if rng.random() < 0.55 else natural_w + rng.randint(8, 80))))
    img = Image.new("L", (target_w, target_h), bg)
    draw = ImageDraw.Draw(img)
    x = pad_x + rng.randint(-3, 5)
    for base, tone in units:
        draw.text((x, baseline_y), base, font=font, fill=ink, anchor="ls")
        bbox = font.getbbox(base or " ")
        base_w = max(1, bbox[2] - bbox[0])
        if tone:
            tone_x = x + max(1, int(base_w * rng.uniform(0.56, 0.88)))
            tone_y = baseline_y - max_above + rng.randint(-4, 5)
            draw.text((tone_x, tone_y), tone, font=tone_font, fill=ink)
        x += widths.pop(0)
    return degrade(img, rng)


def degrade(img: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() < 0.35:
        img = ImageOps.expand(img, border=(rng.randint(0, 6), 0, rng.randint(0, 10), 0), fill=rng.randint(238, 255))
    if rng.random() < 0.70:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.35, 1.25)))
    arr = np.asarray(img, dtype=np.float32)
    arr = (arr - 235.0) * rng.uniform(0.72, 1.25) + 235.0 + rng.uniform(-6, 8)
    noise = np.random.default_rng(rng.randrange(2**32)).normal(0, rng.uniform(1.2, 5.5), arr.shape)
    arr += noise
    if rng.random() < 0.30:
        cols = np.linspace(rng.uniform(-5, 5), rng.uniform(-5, 5), arr.shape[1], dtype=np.float32)
        arr += cols[None, :]
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, "L")
    if rng.random() < 0.18:
        threshold = rng.randint(150, 210)
        img = img.point(lambda p: 0 if p < threshold else 255)
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.15, 0.45)))
    if rng.random() < 0.30:
        scale_x = rng.uniform(0.90, 1.12)
        scale_y = rng.uniform(0.96, 1.06)
        img = img.resize((max(32, int(img.width * scale_x)), max(96, int(img.height * scale_y))), Image.Resampling.BICUBIC)
    if rng.random() < 0.20:
        shift = rng.randint(-5, 5)
        img = ImageChops.offset(img, shift, 0)
    return img.convert("RGB")


def relative_image_path(out_dir: Path, image_path: Path) -> str:
    try:
        return image_path.resolve().relative_to(out_dir.resolve(), walk_up=True).as_posix()
    except TypeError:
        return image_path.resolve().as_posix()


def write_manifest(rows: list[dict[str, object]], path: Path) -> None:
    fields = sorted({str(key) for row in rows for key in row})
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
        "page",
        "row_index",
    ]
    fields = [field for field in preferred if field in fields] + [field for field in fields if field not in preferred]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def make_contact_sheet(out_dir: Path, rows: list[dict[str, object]], count: int) -> str:
    if count <= 0:
        return ""
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
        draw.text((x, (i // cols) * tile_h + 5), str(row["sample_id"]), fill=(0, 0, 0))
    qa_path = out_dir / "qa_contactsheet.png"
    sheet.save(qa_path)
    return str(qa_path)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    if args.out_dir.exists() and args.overwrite:
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    mapping = load_mapping(args.mapping)
    real = read_manifest(args.real_dir)
    train_real = real[real["source_split"].eq("train")].copy()
    val_test_real = real[real["source_split"].isin(["val", "test"])].copy()
    if args.real_train_limit > 0:
        train_real = train_real.sample(min(args.real_train_limit, len(train_real)), random_state=args.seed)
    syllables, weights, row_wupins = build_syllable_pool(train_real)
    if not syllables:
        raise RuntimeError("No legal Wu-pinyin syllables found in real training manifest.")
    font = choose_font(args.font, list(real["label"]))
    profiles = sample_real_profiles(args.real_dir, train_real, rng)

    out_rows: list[dict[str, object]] = []
    generated = 0
    attempts = 0
    while generated < args.synthetic_count and attempts < args.synthetic_count * 10:
        attempts += 1
        wupin = synthesize_wupin(syllables, weights, row_wupins, rng, args.real_row_prob, args.max_syllables)
        label, errors = wupin_to_ipa(wupin, mapping, "digits")
        if errors or not label:
            continue
        sample_id = f"synth_rule_{generated:06d}"
        image_name = f"{sample_id}.png"
        render_label(label, font, rng.choice(profiles), rng).save(image_dir / image_name)
        syllable_count = len(split_syllables(wupin)[0])
        out_rows.append(
            {
                "sample_id": sample_id,
                "variant": args.variant,
                "image": f"images/{image_name}",
                "label": label,
                "wupin": wupin,
                "source_split": "train",
                "synthetic": "1",
                "quality": "synthetic_rule",
                "cleaning_flags": "synthetic",
                "syllable_count": syllable_count,
            }
        )
        generated += 1

    for _, row in pd.concat([train_real, val_test_real], ignore_index=True).iterrows():
        copied = row.to_dict()
        copied["image"] = relative_image_path(args.out_dir, args.real_dir / str(row["image"]))
        copied["synthetic"] = "0"
        out_rows.append(copied)

    write_manifest(out_rows, args.out_dir / "eval_manifest.tsv")
    qa_path = make_contact_sheet(args.out_dir, out_rows, args.qa_count)
    summary = {
        "real_dir": str(args.real_dir),
        "font": str(font),
        "seed": args.seed,
        "synthetic_count": generated,
        "real_train": int(len(train_real)),
        "real_val_test": int(len(val_test_real)),
        "rows": len(out_rows),
        "split_counts": pd.DataFrame(out_rows)["source_split"].value_counts().to_dict(),
        "unique_syllables": len(syllables),
        "unique_real_rows": len(row_wupins),
        "qa_contactsheet": qa_path,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
