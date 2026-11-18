"""Loading and describing the saved inference artifacts.

Layout:
  artifacts/models/final/     tuned model chosen by scripts/hpo.py (preferred)
      model.joblib, feature_spec.json, selection.json, decoder_config.json, manifest.json
  artifacts/models/selected/  Member 2's untuned selection (fallback if final/ is missing)
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path

from pathsnap.config import PROJECT_ROOT, artifacts_dir


def model_dir() -> Path:
    final = artifacts_dir() / "models" / "final"
    return final if (final / "model.joblib").exists() else artifacts_dir() / "models" / "selected"


def file_sha(path: Path, n: int = 8) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:n]


def load_artifacts(directory: str | Path | None = None) -> dict:
    """Model + feature order + decoder settings + a version string. Load once, reuse."""
    from pathsnap.hmm.viterbi import DecoderConfig
    from pathsnap.models.predict import load_model_artifacts

    d = Path(directory) if directory else model_dir()
    arts = load_model_artifacts(d)
    dec = d / "decoder_config.json"
    arts["decoder"] = DecoderConfig.load(dec if dec.exists() else None)
    sel = arts.get("selection", {})
    variant = sel.get("variant", sel.get("selected_model", "model")).lower().replace(" ", "-").replace("(", "").replace(")", "")
    arts["model_version"] = f"pathsnap-{variant}-{file_sha(d / 'model.joblib')}"
    arts["model_dir"] = str(d)
    return arts


def library_versions() -> dict:
    import lightgbm
    import numpy
    import pandas
    import sklearn

    return {"python": platform.python_version(), "numpy": numpy.__version__, "pandas": pandas.__version__,
            "scikit-learn": sklearn.__version__, "lightgbm": lightgbm.__version__}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def write_manifest(directory: str | Path | None = None) -> Path:
    d = Path(directory) if directory else model_dir()
    files = {p.name: file_sha(p, 16) for p in sorted(d.iterdir()) if p.is_file() and p.name != "manifest.json"}
    out = d / "manifest.json"
    out.write_text(json.dumps({"files_sha256_16": files, "library_versions": library_versions(),
                               "git_commit": git_commit()}, indent=2) + "\n")
    return out
