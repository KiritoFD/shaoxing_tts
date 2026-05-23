"""Print dependency and CUDA status for OCR training hosts."""

from __future__ import annotations

import json


def main() -> None:
    info: dict[str, object] = {}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    except Exception as exc:  # pragma: no cover - diagnostic script
        info["torch_error"] = repr(exc)

    for name, module in [("pandas", "pandas"), ("PIL", "PIL"), ("cv2", "cv2")]:
        try:
            imported = __import__(module)
            info[name] = getattr(imported, "__version__", "ok")
        except Exception as exc:  # pragma: no cover - diagnostic script
            info[f"{name}_error"] = repr(exc)

    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
