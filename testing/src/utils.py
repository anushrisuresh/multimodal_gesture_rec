import yaml
from typing import Any, Dict

LABELS = ["power", "forward", "backward", "up", "down"]

def load_config(path: str) -> Dict[str, Any]:
    """
    Load YAML configuration from `path` into dict.
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)
