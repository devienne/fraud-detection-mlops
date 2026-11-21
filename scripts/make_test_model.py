"""Write a tiny TEST model (production artifact format) into a folder - for CI only.

CI has no access to the real model (a DVC output), but the Docker image needs model files to build
and start. This creates synthetic stand-ins so the image can be built and smoke-tested.

Usage:  python scripts/make_test_model.py <output_folder>
The output folder is required on purpose, so the real models/ folder is never overwritten by accident.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `tests` (project root) and `fraud_detection` (src/) importable without installing the package,
# exactly as on a fresh CI machine.
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tests.model_fixture import build_test_model  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    folder = build_test_model(Path(sys.argv[1]))
    print(f"Test model written to {folder}: {sorted(p.name for p in folder.iterdir())}")
