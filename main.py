from __future__ import annotations

import importlib
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parent
    src_dir = project_root / "src"

    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    if len(sys.argv) == 1:
        sys.argv.append("--gui")

    application_main = importlib.import_module("ezw_compression.app")
    application_main.main()


if __name__ == "__main__":
    main()
