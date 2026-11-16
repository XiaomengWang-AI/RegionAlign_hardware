"""Paths, offline settings and output containment of the RegionAlign package.

This module owns every path the package uses; nothing is delegated to code
outside ``regionalign`` and no library path is ever added to ``sys.path``.
``PROJECT`` is the directory that contains the package, so recorded source
paths read ``regionalign/...`` and commands run as ``python -m regionalign...``
from there.  New outputs are accepted only below ``regionalign/results``.

The one external input is the Hugging Face cache with the model and dataset
snapshots.  ``runtime.json`` beside this file names it with its single key
``hf_home`` (a relative path resolves against the package directory); the
environment variable ``REGIONALIGN_HF_HOME`` replaces that key.  No models are
downloaded.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

PACKAGE = Path(__file__).resolve().parent
PROJECT = PACKAGE.parent
CONFIG_PATH = PACKAGE / "runtime.json"
CACHE = PACKAGE / ".cache"


def _load_config(path):
    """Read the package's own ``runtime.json``; refuse a link that leaves the package."""
    if not path.exists():
        return {}
    if path.resolve().parent != PACKAGE:
        raise ValueError(f"{path} must be a regular file inside the package")
    with path.open() as handle:
        data = json.load(handle)
    if (not isinstance(data, dict) or set(data) - {"hf_home"}
            or not isinstance(data.get("hf_home", ""), str)):
        raise ValueError(f"{path} must contain a JSON object with at most the string key hf_home")
    return data


def _path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PACKAGE / path


HUB = _path(os.environ.get("REGIONALIGN_HF_HOME") or _load_config(CONFIG_PATH).get("hf_home")
            or CACHE / "huggingface")


def __getattr__(name):
    """``RESULTS_ROOT`` follows ``PROJECT`` unless a caller or test sets it explicitly."""
    if name == "RESULTS_ROOT":
        return PROJECT / "regionalign" / "results"
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def results_root():
    """Return the result root, following PROJECT patches used by CPU tests."""
    return getattr(sys.modules[__name__], "RESULTS_ROOT")


def setup_environment():
    """Run offline against the configured cache, with other caches kept inside the package."""
    settings = {
        "HF_HOME": HUB, "HF_HUB_CACHE": HUB / "hub", "HF_DATASETS_CACHE": HUB / "datasets",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false", "XDG_CACHE_HOME": CACHE,
        "TORCH_HOME": CACHE / "torch", "MPLCONFIGDIR": CACHE / "matplotlib",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    for key, value in settings.items():
        os.environ[key] = str(value)


def output_path(path):
    """Resolve an output path and require it to lie strictly below the result root."""
    target = Path(path).expanduser()
    if not target.is_absolute():
        target = PROJECT / target
    target = target.resolve()
    root = results_root().resolve()
    if target == root or root not in target.parents:
        raise ValueError(f"New outputs must be inside {root}: {target}")
    return target


def fresh_output_dir(group, label, requested=None):
    """Allocate a new result directory and refuse existing destinations."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    target = output_path(requested) if requested else output_path(
        results_root() / group / f"{stamp}_{label}_{uuid4().hex[:8]}")
    target.mkdir(parents=True, exist_ok=False)
    return target
