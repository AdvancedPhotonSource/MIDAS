"""Allow ``python -m midas_transforms <stage> [args]``."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
