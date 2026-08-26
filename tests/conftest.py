import sys
from pathlib import Path

# tests/ has no __init__.py, so pytest's default import mode prepends tests/
# itself to sys.path, not the project root above it -- without this, `from
# Metrics_Functions import ...` / `from Features import ...` in the test
# files below would fail with ModuleNotFoundError even though those modules
# are right there at the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
