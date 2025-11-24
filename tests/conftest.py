import sys
from pathlib import Path


# Ensure the src package root is on PYTHONPATH for local tests without installation
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))
