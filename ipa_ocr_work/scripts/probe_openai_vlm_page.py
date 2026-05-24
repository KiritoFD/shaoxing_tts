"""Probe an OpenAI-compatible VLM endpoint with one local page image.

This script is intentionally small and explicit:
- reads a local image
- embeds it as a Base64 Data URI in the request body
- sends a standard /chat/completions request
- saves the exact raw response and assistant content

It does not score or postprocess OCR output.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE = PROJECT_ROOT / "ipa_ocr_work" / "eval" / "vlm_pages" / "test13_cropped_140dpi" / "images" / "pdf007_p129.png"
DEFAULT_OUT = PROJECT_ROOT / "ipa_ocr_work" / "eval" / "vlm_pages" / "openai_vlm_probe"
DEFAULT_BASE_URL = "https://new-api.abrdns.com/v1"
DEFAULT_MODEL = "qwen3.6-plus-2026-04-02"


DEFAULT_PROMPT = """你是 OCR 引擎。识别这张绍兴方言词典页。
每一行识别出来音标，用 JSON 写出来。
输出必须是一个 JSON 对象，不要 Markdown，不要解释，不要输出思考过程。
JSON schema:
{"rows":[{"row_index":0,"headword":"词头","ipa":"音标","confidence":0.0,"notes":""}]}

规则：
1. 按页面从上到下排序。
2. ipa 只写词头后面的音标，不写中文释义。
3. 数字规则：有右下角数字就看右下角；没有右下角数字就看右上角。
4. 声调数字写成普通数字，例如 11、33、231、52，不要写上标或下标。
5. 只根据图片内容识别，不要查资料，不要概括页面，不要说明声调系统。
6. 看不清的词条也保留，ipa 写 [?]，confidence 降低，notes 简短说明。"""


def safe_print(text: object, limit: int | None = None) -> None:
    value = str(text)
    if limit is not None:
        value = value[:limit]
    try:
        print(value)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(value.encode("utf-8", errors="replace") + b"\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe one page image through an OpenAI-compatible VLM endpoint.")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=os.getenv("OPENAI_VLM_MODEL", DEFAULT_MODEL))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--response-format-json", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=180)
    return parser.parse_args()


def make_data_uri(path: Path) -> tuple[str, dict[str, object]]:
    image_bytes = path.read_bytes()
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    b64 = base64.b64encode(image_bytes).decode("ascii")
    data_uri = f"data:{mime};base64,{b64}"
    meta = {
        "image": str(path),
        "mime": mime,
        "image_bytes": len(image_bytes),
        "sha256": hashlib.sha256(image_bytes).hexdigest(),
        "data_uri_chars": len(data_uri),
        "data_uri_prefix": data_uri[:48],
    }
    return data_uri, meta


def main() -> None:
    args = parse_args()
    api_key = args.api_key or os.getenv(args.api_key_env, "")
    if not api_key:
        raise SystemExit(f"Missing API key. Set {args.api_key_env} or pass --api-key.")
    if not args.image.exists():
        raise SystemExit(f"Image not found: {args.image}")
    if args.prompt_file is not None:
        args.prompt = args.prompt_file.read_text(encoding="utf-8")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data_uri, image_meta = make_data_uri(args.image)
    body: dict[str, object] = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": args.prompt},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    if args.response_format_json:
        body["response_format"] = {"type": "json_object"}

    request_debug = {
        "endpoint": args.base_url.rstrip("/") + "/chat/completions",
        "model": args.model,
        "image_meta": image_meta,
        "message_content_types": ["text", "image_url"],
        "prompt_chars": len(args.prompt),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "response_format_json": args.response_format_json,
    }
    (args.out_dir / "request_debug.json").write_text(
        json.dumps(request_debug, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    safe_print(json.dumps(request_debug, ensure_ascii=False, indent=2))
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    safe_print(f"payload_bytes={len(payload)}")

    request = Request(
        args.base_url.rstrip("/") + "/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=args.timeout_seconds) as response:
            raw = response.read().decode("utf-8", errors="replace")
            status = f"HTTP {response.status}"
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        status = f"HTTP_ERROR {exc.code}"
    except URLError as exc:
        raw = repr(exc)
        status = "URL_ERROR"
    except Exception as exc:
        raw = repr(exc)
        status = "EXCEPTION " + type(exc).__name__

    (args.out_dir / "raw_response.json_or_error.txt").write_text(raw, encoding="utf-8")
    safe_print(status)
    safe_print(f"wrote {args.out_dir / 'raw_response.json_or_error.txt'}")
    safe_print(raw, 12000)

    try:
        parsed = json.loads(raw)
        content = parsed["choices"][0]["message"]["content"]
    except Exception:
        return
    (args.out_dir / "assistant_content.txt").write_text(str(content), encoding="utf-8")
    safe_print("\n--- assistant content ---")
    safe_print(content, 12000)


if __name__ == "__main__":
    main()
