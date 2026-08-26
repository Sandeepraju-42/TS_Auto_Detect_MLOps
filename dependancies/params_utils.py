"""
Shared params.yaml loader. Every pipeline-stage script imports load_params()
from here rather than reading YAML itself -- one place to change if the
file's location or format ever moves.
"""

from pathlib import Path
import yaml

_PARAMS_PATH = Path(__file__).resolve().parent / "params.yaml"

print(_PARAMS_PATH)
def load_params() -> dict:
    with open(_PARAMS_PATH) as f:
        return yaml.safe_load(f)
