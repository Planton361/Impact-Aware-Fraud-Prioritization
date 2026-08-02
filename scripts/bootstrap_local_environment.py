"""Fresh-clone stdlib wrapper for the shared local setup workflow."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fraud_detection.setup.environment import bootstrap_main  # noqa: E402


def main(root: Path | None = None) -> int:
    return bootstrap_main(REPOSITORY_ROOT if root is None else root)


if __name__ == "__main__":
    raise SystemExit(main())
