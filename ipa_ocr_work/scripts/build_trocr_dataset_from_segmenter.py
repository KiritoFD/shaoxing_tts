"""Build a TrOCR fine-tuning dataset from segmenter-scored phonetic spans."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from train_phonetic_segmenter import CandidateDataset, FEATURE_COLUMNS, SpanQualityNet
from wupin_ipa_convert import DEFAULT_MAP, canonicalize_wupin_base, load_mapping, wupin_to_ipa


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATES = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_pdf136_clean" / "phonetic_segmenter_candidates"
DEFAULT_STRUCTURED = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_structured_tone_labels" / "structured_tone_syllables.tsv"
DEFAULT_MODEL = PROJECT_ROOT / "ipa_ocr_work" / "models" / "phonetic_segmenter" / "best.pt"
DEFAULT_OUT = PROJECT_ROOT / "ipa_ocr_work" / "dataset" / "shaoxing_pdf136_clean" / "trocr_segmenter_clean"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TrOCR dataset from segmenter-selected spans.")
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--structured-syllables", type=Path, default=DEFAULT_STRUCTURED)
    parser.add_argument("--segmenter", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--variant", default="segmenter_phonetic_span")
    parser.add_argument("--min-prob-train", type=float, default=0.88)
    parser.add_argument("--min-prob-eval", type=float, default=0.70)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", keep_default_na=False)


def split_for_row(row: pd.Series) -> str:
    split = str(row.get("split", ""))
    return "train" if split == "review" else split


def row_labels(structured: pd.DataFrame, mapping: dict) -> dict[str, dict[str, str]]:
    labels: dict[str, dict[str, str]] = {}
    for (page, row_index), group in structured.groupby(["page", "row_index"], sort=True):
        group = group.sort_values("syllable_index")
        wupin_parts = []
        ipa_parts = []
        for _, row in group.iterrows():
            base = canonicalize_wupin_base(row.get("wupin_base", ""))
            tone = str(row.get("selected_tone", ""))
            wupin_parts.append(f"{base}{tone}")
            ipa_parts.append(str(row.get("ipa_selected", "")))
        wupin = "".join(wupin_parts)
        ipa_from_wupin, errors = wupin_to_ipa(wupin, mapping, "digits")
        label = ipa_from_wupin if not errors else "".join(ipa_parts)
        row0 = group.iloc[0]
        labels[f"p{int(page):03d}_{int(row_index):04d}"] = {
            "wupin": wupin,
            "label": label,
            "label_status": "ok" if not errors else ";".join(errors),
            "source_split": split_for_row(row0),
        }
    return labels


def collate_candidates(batch: list[dict]) -> dict:
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "features": torch.stack([item["features"] for item in batch]),
        "sample_id": [item["sample_id"] for item in batch],
    }


def score_candidates(args: argparse.Namespace, rows: list[dict[str, str]]) -> list[float]:
    checkpoint = torch.load(args.segmenter, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata", {})
    height = int(metadata.get("height", 96))
    width = int(metadata.get("width", 384))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SpanQualityNet(feature_dim=len(FEATURE_COLUMNS)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dataset = CandidateDataset(args.candidate_dir, rows, height, width, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_candidates)
    probs: list[float] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(device), batch["features"].to(device))
            probs.extend(torch.sigmoid(logits).cpu().tolist())
    return probs


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and args.overwrite:
        shutil.rmtree(args.out_dir)
    image_dir = args.out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    mapping = load_mapping(args.mapping)
    structured = read_tsv(args.structured_syllables)
    labels = row_labels(structured, mapping)
    candidates_df = read_tsv(args.candidate_dir / "candidate_manifest.tsv")
    candidate_rows = candidates_df.to_dict("records")
    probs = score_candidates(args, candidate_rows)
    candidates_df["segmenter_prob"] = probs

    selected_rows = []
    for row_id, group in candidates_df.groupby("row_id", sort=True):
        if row_id not in labels:
            continue
        best = group.sort_values(["segmenter_prob", "teacher_score"], ascending=False).iloc[0]
        label_info = labels[row_id]
        split = str(best.get("source_split", label_info["source_split"]))
        min_prob = args.min_prob_train if split == "train" else args.min_prob_eval
        if float(best["segmenter_prob"]) < min_prob:
            continue
        source_image = args.candidate_dir / str(best["image"])
        if not source_image.exists():
            continue
        sample_id = row_id
        target_image = image_dir / f"{sample_id}.png"
        shutil.copy2(source_image, target_image)
        selected_rows.append(
            {
                "sample_id": sample_id,
                "variant": args.variant,
                "image": f"images/{sample_id}.png",
                "label": label_info["label"],
                "wupin": label_info["wupin"],
                "label_status": label_info["label_status"],
                "source_split": split,
                "page": int(best["page"]),
                "source_page": best.get("source_page", best["page"]),
                "pdf_page": best.get("pdf_page", ""),
                "row_index": int(best["row_index"]),
                "syllable_count": int(best["syllable_count"]),
                "segmenter_prob": float(best["segmenter_prob"]),
                "teacher_score": float(best["teacher_score"]),
                "span_x0": int(best["span_x0"]),
                "span_x1": int(best["span_x1"]),
                "row_width": int(best["row_width"]),
                "quality": best.get("quality", ""),
                "cleaning_flags": best.get("cleaning_flags", ""),
                "candidate_kind": best.get("candidate_kind", ""),
            }
        )

    out = pd.DataFrame(selected_rows)
    out = out.sort_values(["source_split", "page", "row_index"]).reset_index(drop=True)
    out.to_csv(args.out_dir / "eval_manifest.tsv", sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)
    candidates_df.to_csv(args.out_dir / "candidate_scores.tsv", sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)

    summary = {
        "candidate_dir": str(args.candidate_dir),
        "segmenter": str(args.segmenter),
        "variant": args.variant,
        "min_prob_train": args.min_prob_train,
        "min_prob_eval": args.min_prob_eval,
        "rows": int(len(out)),
        "split_counts": out["source_split"].value_counts().to_dict() if len(out) else {},
        "prob_summary": out["segmenter_prob"].describe().to_dict() if len(out) else {},
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {args.out_dir / 'eval_manifest.tsv'}")


if __name__ == "__main__":
    main()
