"""Run local GLM-OCR inference on one page image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = PROJECT_ROOT / "ipa_ocr_work" / "models" / "modelscope_cache" / "ZhipuAI" / "GLM-OCR"
DEFAULT_IMAGE = PROJECT_ROOT / "ipa_ocr_work" / "eval" / "vlm_pages" / "post136_cropped_180dpi" / "images" / "pdf137_p259.png"
DEFAULT_OUT = PROJECT_ROOT / "ipa_ocr_work" / "eval" / "glm_ocr" / "pdf137_p259_text_recognition.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GLM-OCR on one image.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--prompt", default="Text Recognition:")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": str(args.image)},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        str(args.model),
        torch_dtype="auto",
        device_map="auto",
        local_files_only=True,
    )
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    inputs.pop("token_type_ids", None)
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    output_text = processor.decode(generated_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=False)
    args.out.write_text(output_text, encoding="utf-8")
    meta = {
        "model": str(args.model),
        "image": str(args.image),
        "out": str(args.out),
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "output_chars": len(output_text),
    }
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(output_text[:2000])


if __name__ == "__main__":
    main()
