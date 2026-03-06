"""HF Hub cache filesystem probing -- no torch/open_clip imports.

Replicates the model_id resolution and weight-file discovery from
grape.model without pulling in heavy dependencies.  Only needs os
and pathlib.  This lets the CLI skip the ~1.5s torch import on
warm-cache runs.
"""

import os
from pathlib import Path

WEIGHT_FILENAMES = (
    "open_clip_model.safetensors",
    "open_clip_pytorch_model.safetensors",
    "open_clip_pytorch_model.bin",
)


def hf_cache_root() -> Path:
    """Return the huggingface hub cache root directory."""
    hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hub_cache:
        return Path(hub_cache)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def cached_file_from_repo(repo_id: str, filename: str) -> str | None:
    """Return cached file path for ``repo_id/filename`` if present."""
    repo_dir = hf_cache_root() / f"models--{repo_id.replace('/', '--')}"
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return None

    ref_main = repo_dir / "refs" / "main"
    if ref_main.is_file():
        commit = ref_main.read_text(encoding="utf-8").strip()
        if commit:
            candidate = snapshots_dir / commit / filename
            if candidate.is_file():
                return str(candidate)

    newest: Path | None = None
    newest_mtime = -1
    for snapshot in snapshots_dir.iterdir():
        candidate = snapshot / filename
        if not candidate.is_file():
            continue
        mtime = candidate.stat().st_mtime_ns
        if mtime > newest_mtime:
            newest = candidate
            newest_mtime = mtime
    if newest is None:
        return None
    return str(newest)


def resolve_model_id(hf_hub: str) -> str:
    """Resolve model_id from an hf_hub repo string by probing the cache.

    Returns ``hf_hub@commit`` if cached weights are found, or just
    ``hf_hub`` as a fallback.
    """
    for filename in WEIGHT_FILENAMES:
        cached = cached_file_from_repo(hf_hub, filename)
        if cached:
            commit = Path(cached).parent.name
            return f"{hf_hub}@{commit}"
    return hf_hub


def find_cached_weight(hf_hub: str) -> str | None:
    """Return the local path to cached model weights, or None."""
    for filename in WEIGHT_FILENAMES:
        cached = cached_file_from_repo(hf_hub, filename)
        if cached is not None:
            return cached
    return None
