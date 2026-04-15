import sys
from pathlib import Path


def _ensure_marlin_on_path():
    repo_root = Path(__file__).resolve().parent.parent
    marlin_src = repo_root / "third_party" / "MARLIN" / "src"
    if not marlin_src.exists():
        raise FileNotFoundError(
            f"MARLIN source directory not found: {marlin_src}. Clone the repo into third_party/MARLIN first."
        )
    marlin_src_str = str(marlin_src)
    if marlin_src_str not in sys.path:
        sys.path.insert(0, marlin_src_str)


def load_marlin(model_name, checkpoint_path=None, from_online=False):
    _ensure_marlin_on_path()

    try:
        from marlin_pytorch import Marlin
    except ImportError as exc:
        raise ImportError(
            "Failed to import MARLIN. Install its runtime dependencies from third_party/MARLIN/requirements.lib.txt."
        ) from exc

    if checkpoint_path:
        return Marlin.from_file(model_name, checkpoint_path)
    if from_online:
        return Marlin.from_online(model_name)
    raise ValueError(
        "MARLIN requires either a local --marlin_checkpoint or enabling --marlin_from_online at runtime."
    )
