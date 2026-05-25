"""Apply Real-ESRGAN to Shaoxing OCR crop manifests.

The script is deliberately manifest-in/manifest-out so enhancement experiments
stay separate from the trusted training manifests.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
import torchvision.transforms.functional as tv_functional

# basicsr still imports this removed torchvision module name on newer builds.
sys.modules.setdefault("torchvision.transforms.functional_tensor", tv_functional)

from basicsr.archs.rrdbnet_arch import RRDBNet  # noqa: E402
from basicsr.utils.download_util import load_file_from_url  # noqa: E402
from realesrgan import RealESRGANer  # noqa: E402
from realesrgan.archs.srvgg_arch import SRVGGNetCompact  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "ipa_ocr_work"
    / "dataset"
    / "shaoxing_pdf136_clean"
    / "ocr_selected_phonetic_reliable"
    / "eval_manifest.tsv"
)
DEFAULT_OUT = PROJECT_ROOT / "ipa_ocr_work" / "eval" / "realesrgan_phonetic_reliable_smoke"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "ipa_ocr_work" / "models" / "realesrgan"

MODEL_URLS = {
    "RealESRGAN_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "realesr-general-x4v3": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enhance OCR crops with Real-ESRGAN.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_URLS),
        default="realesr-general-x4v3",
        help="Use x4plus for strongest RRDB restoration, general-x4v3 for faster denoise/SR.",
    )
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--qualities", nargs="+", default=["matched", "weak_match"])
    parser.add_argument("--max-samples", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--tile", type=int, default=192)
    parser.add_argument("--tile-pad", type=int, default=12)
    parser.add_argument("--outscale", type=float, default=2.0)
    parser.add_argument("--denoise-strength", type=float, default=0.35)
    parser.add_argument("--suffix", default=None)
    parser.add_argument("--include-original", action="store_true")
    parser.add_argument("--contact-sheet", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def resolve_image(manifest: Path, value: str) -> Path:
    image = Path(value)
    if not image.is_absolute():
        image = (manifest.parent / image).resolve()
    return image


def select_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    rows = read_manifest(args.manifest)
    allowed_splits = set(args.splits)
    allowed_qualities = set(args.qualities)
    selected = [
        row
        for row in rows
        if (row.get("source_split") or row.get("split") or "") in allowed_splits
        and row.get("quality", "") in allowed_qualities
        and row.get("label", "")
        and resolve_image(args.manifest, row.get("image", "")).exists()
    ]
    selected.sort(key=lambda r: (r.get("source_split", ""), int(r.get("pdf_page") or 0), int(r.get("row_index") or 0)))
    if args.max_samples and len(selected) > args.max_samples:
        rng = random.Random(args.seed)
        selected = sorted(
            rng.sample(selected, args.max_samples),
            key=lambda r: (r.get("source_split", ""), int(r.get("pdf_page") or 0), int(r.get("row_index") or 0)),
        )
    return selected


def build_model(model_name: str) -> tuple[torch.nn.Module, int, str]:
    if model_name == "RealESRGAN_x4plus":
        return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4), 4, MODEL_URLS[model_name]
    return (
        SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=32,
            upscale=4,
            act_type="prelu",
        ),
        4,
        MODEL_URLS[model_name],
    )


def model_path(args: argparse.Namespace) -> Path:
    args.model_dir.mkdir(parents=True, exist_ok=True)
    model, _scale, url = build_model(args.model)
    del model
    target = args.model_dir / f"{args.model}.pth"
    if target.exists():
        return target
    downloaded = Path(load_file_from_url(url=url, model_dir=str(args.model_dir), progress=True, file_name=target.name))
    return downloaded


def make_upsampler(args: argparse.Namespace) -> RealESRGANer:
    model, scale, _url = build_model(args.model)
    dni_weight = None
    model_file = model_path(args)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    return RealESRGANer(
        scale=scale,
        model_path=str(model_file),
        dni_weight=dni_weight,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=0,
        half=device.type == "cuda",
        device=device,
    )


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv.imwrite(str(path), image):
        raise OSError(f"failed to write {path}")


def add_white_border(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    if img.ndim == 2:
        canvas = np.full((target_h, target_w), 255, dtype=img.dtype)
        canvas[: img.shape[0], : img.shape[1]] = img
        return canvas
    canvas = np.full((target_h, target_w, img.shape[2]), 255, dtype=img.dtype)
    canvas[: img.shape[0], : img.shape[1], :] = img
    return canvas


def write_contact_sheet(out_path: Path, pairs: list[tuple[str, Path, Path]], max_rows: int = 24) -> None:
    rows = []
    for label, original_path, enhanced_path in pairs[:max_rows]:
        original = cv.imread(str(original_path), cv.IMREAD_GRAYSCALE)
        enhanced = cv.imread(str(enhanced_path), cv.IMREAD_GRAYSCALE)
        if original is None or enhanced is None:
            continue
        scale = min(1.0, 360 / max(enhanced.shape[1], 1))
        if scale < 1.0:
            enhanced = cv.resize(enhanced, (round(enhanced.shape[1] * scale), round(enhanced.shape[0] * scale)))
        original = cv.resize(original, (enhanced.shape[1], enhanced.shape[0]), interpolation=cv.INTER_NEAREST)
        h = max(original.shape[0], enhanced.shape[0]) + 28
        w = original.shape[1] + enhanced.shape[1] + 18
        row_img = np.full((h, w), 255, dtype=np.uint8)
        row_img[22 : 22 + original.shape[0], : original.shape[1]] = original
        row_img[22 : 22 + enhanced.shape[0], original.shape[1] + 18 : original.shape[1] + 18 + enhanced.shape[1]] = enhanced
        cv.putText(row_img, label[:90], (4, 15), cv.FONT_HERSHEY_SIMPLEX, 0.42, 0, 1, cv.LINE_AA)
        rows.append(row_img)
    if not rows:
        return
    max_w = max(row.shape[1] for row in rows)
    sheet = np.vstack([add_white_border(row, row.shape[0], max_w) for row in rows])
    write_png(out_path, sheet)


def main() -> None:
    args = parse_args()
    suffix = args.suffix or f"{args.model}_x{args.outscale:g}_dn{args.denoise_strength:g}"
    out_dir = args.out_dir.resolve()
    enhanced_dir = out_dir / "images" / suffix
    original_dir = out_dir / "images" / "original"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = select_rows(args)
    upsampler = make_upsampler(args)
    eval_rows: list[dict[str, str]] = []
    contact_pairs: list[tuple[str, Path, Path]] = []

    for idx, row in enumerate(rows, 1):
        src = resolve_image(args.manifest, row["image"])
        img = cv.imread(str(src), cv.IMREAD_COLOR)
        if img is None:
            continue

        sample_id = row["sample_id"]
        if args.include_original:
            original_rel = Path("images") / "original" / f"{sample_id}.png"
            original_path = out_dir / original_rel
            if args.overwrite or not original_path.exists():
                write_png(original_path, img)
            original_row = dict(row)
            original_row["variant"] = "original"
            original_row["image"] = original_rel.as_posix()
            eval_rows.append(original_row)
        else:
            original_path = src

        enhanced_rel = Path("images") / suffix / f"{sample_id}.png"
        enhanced_path = out_dir / enhanced_rel
        if args.overwrite or not enhanced_path.exists():
            enhanced, _mode = upsampler.enhance(img, outscale=args.outscale, alpha_upsampler=None)
            if args.model == "realesr-general-x4v3" and args.denoise_strength < 1:
                # Blend back a little original structure; full blind restoration can over-hallucinate tiny tone digits.
                resized = cv.resize(img, (enhanced.shape[1], enhanced.shape[0]), interpolation=cv.INTER_CUBIC)
                enhanced = cv.addWeighted(enhanced, args.denoise_strength, resized, 1 - args.denoise_strength, 0)
            write_png(enhanced_path, enhanced)

        enhanced_row = dict(row)
        enhanced_row["variant"] = suffix
        enhanced_row["image"] = enhanced_rel.as_posix()
        eval_rows.append(enhanced_row)
        contact_pairs.append((f"{sample_id}  {row.get('wupin','')}", original_path, enhanced_path))
        print(f"[{idx}/{len(rows)}] {sample_id} -> {enhanced_path}")

    fieldnames = list(read_manifest(args.manifest)[0].keys())
    if "variant" not in fieldnames:
        fieldnames.insert(1, "variant")
    manifest_out = out_dir / "eval_manifest.tsv"
    with manifest_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(eval_rows)

    if args.contact_sheet:
        write_contact_sheet(out_dir / f"contact_sheet_{suffix}.png", contact_pairs)

    print(f"selected_rows\t{len(rows)}")
    print(f"eval_rows\t{len(eval_rows)}")
    print(f"manifest\t{manifest_out}")


if __name__ == "__main__":
    main()
