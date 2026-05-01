"""Entry point so ``python -m midas_hkls ...`` works alongside the
``midas-hkls`` console script."""
from .cli import main
import sys

if __name__ == "__main__":
    sys.exit(main())
