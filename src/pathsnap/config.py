"""Project-wide configuration.

Single source of truth for where the dataset lives, where outputs are written and
how a run is seeded. Nothing in this module reads the dataset itself.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: The three dataset partitions. ``eval`` is injected by the evaluators at grading time.
PARTITIONS: tuple[str, ...] = ("train", "student_test", "eval")

#: Default random seed, applied to every library used in this project.
DEFAULT_SEED: int = 42

#: A time gap larger than this is treated as a trajectory break by validators and the decoder.
MAX_DT_GAP_S: float = 60.0


def _holds_partitions(candidate: Path) -> bool:
    """True when ``candidate`` directly contains a supplied partition folder."""
    return any((candidate / name).is_dir() for name in ("train", "student_test"))


def data_root() -> Path:
    """Directory containing the partition folders ``train/`` and ``student_test/``.

    Resolution order:
      1. ``PATHSNAP_DATA_ROOT`` environment variable, if set;
      2. ``data/raw/`` when it already holds the partition folders;
      3. ``data/raw/dataset/`` (the layout used when the supplied archive is extracted there);
      4. ``data/raw/`` as a fallback so the error message points at the right place.

    A candidate counts only if it contains a partition folder, so an empty
    ``data/raw/`` does not shadow a populated ``data/raw/dataset/``.
    """
    env = os.environ.get("PATHSNAP_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    fallback = PROJECT_ROOT / "data" / "raw"
    for candidate in (fallback, fallback / "dataset"):
        if _holds_partitions(candidate):
            return candidate.resolve()
    return fallback.resolve()


def configs_dir() -> Path:
    return PROJECT_ROOT / "configs"


def results_dir() -> Path:
    return PROJECT_ROOT / "results"


def maps_dir() -> Path:
    return PROJECT_ROOT / "maps"


def artifacts_dir() -> Path:
    return PROJECT_ROOT / "artifacts"


def processed_dir() -> Path:
    return PROJECT_ROOT / "data" / "processed"


def set_global_seed(seed: int = DEFAULT_SEED) -> None:
    """Seed every random number generator this project uses."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
