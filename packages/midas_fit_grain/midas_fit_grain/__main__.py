"""Allow `python -m midas_fit_grain` invocation."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
