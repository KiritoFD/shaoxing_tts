"""Run PaddleX's config engine from an installed PaddleX package."""

from __future__ import annotations

from paddlex.engine import Engine


def main() -> None:
    Engine().run()


if __name__ == "__main__":
    main()
