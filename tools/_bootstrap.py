import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
LG_ROOT = Path(os.environ.get("LIGHTGLUE_ROOT", ROOT / "third_party" / "LightGlue")).resolve()
TORCH_HOME = Path(os.environ.get("TORCH_HOME", ROOT / ".cache" / "torch")).resolve()


def _prepend_once(path: Path) -> None:
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)


TORCH_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TORCH_HOME", str(TORCH_HOME))
os.environ.setdefault("XDG_CACHE_HOME", str(TORCH_HOME.parent))

_prepend_once(ROOT)
_prepend_once(LG_ROOT)
