from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image
from tqdm import tqdm

# CLIPModel is under TYPE_CHECKING so that importing search.py does
# not pull in torch/open_clip (~2 s).  The cache-hit path never needs them.
if TYPE_CHECKING:
    from grape.cache import EmbeddingCache
    from grape.model import CLIPModel


def _stat_key_from_stat(st: os.stat_result) -> str:
    """Build cache invalidation key from an existing stat result."""
    return json.dumps([
        st.st_size, st.st_mtime_ns,
        st.st_ino, st.st_dev, st.st_ctime_ns,
    ])


def _is_image(
    path: Path,
    cache: EmbeddingCache | None = None,
    *,
    path_key: str | None = None,
    file_stat: str | None = None,
) -> bool:
    """Check whether a file is a recognized image format.

    When a cache is provided, consults and updates it to avoid
    re-opening files already known not to be images.
    """
    if cache is not None and cache.is_not_image(
        path, path_key=path_key, file_stat=file_stat
    ):
        return False
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except OSError as e:
        # PIL raises OSError with errno=None for unrecognized formats.
        # Real filesystem errors (EACCES, ENOENT, etc.) have an errno set.
        if e.errno is not None:
            raise
        if cache is not None:
            cache.put_not_image(path, path_key=path_key, file_stat=file_stat)
        return False


def find_images(
    directory: str,
    recursive: bool = False,
    cache: EmbeddingCache | None = None,
) -> list[Path]:
    """Return sorted image paths under *directory*.

    Uses ``PIL.Image.verify`` for content-based detection (not file
    extensions).  Pass a *cache* to skip re-checking known non-images.
    """
    # Canonicalize the root once so parent-directory symlinks are resolved
    # without paying realpath cost repeatedly per discovered file.
    root = Path(os.path.realpath(directory))
    if not root.is_dir():
        return []
    found: list[Path] = []
    stack = [root]
    while stack:
        current = stack.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                # Match Path.is_file()/is_dir() behavior (follows symlinks).
                if recursive and entry.is_dir():
                    stack.append(Path(entry.path))
                    continue
                if not entry.is_file():
                    continue
                path = Path(entry.path)
                stat_key = _stat_key_from_stat(entry.stat())
                if _is_image(
                    path,
                    cache,
                    path_key=entry.path,
                    file_stat=stat_key,
                ):
                    found.append(path)
    return sorted(found)


def _build_result(
    path: Path,
    keywords: list[str],
    sims: np.ndarray,
) -> dict:
    return {
        "path": path,
        "scores": {kw: float(s) for kw, s in zip(keywords, sims)},
        "score": float(sims.mean()),
        "min_score": float(sims.min()),
    }


def _get_embedding(
    model: CLIPModel,
    path: Path,
    cache: EmbeddingCache | None,
) -> np.ndarray:
    """Encode an image, using the cache when available."""
    if cache is not None:
        cached = cache.get(path, model.model_id())
        if cached is not None:
            return cached
    emb = model.encode_image(str(path))
    if cache is not None:
        cache.put(path, model.model_id(), emb)
    return emb


def score_image(
    model: CLIPModel,
    image_path: Path,
    keywords: list[str],
    prompt_template: str = "a photo of {}",
    cache: EmbeddingCache | None = None,
) -> dict:
    """Score a single image against all keywords."""
    prompts = [prompt_template.format(kw) for kw in keywords]
    text_emb = model.encode_texts(prompts)
    img_emb = _get_embedding(model, image_path, cache)
    sims = (img_emb @ text_emb.T)[0]
    return _build_result(image_path, keywords, sims)


def score_images(
    model: CLIPModel,
    image_paths: list[Path],
    keywords: list[str],
    prompt_template: str = "a photo of {}",
    quiet: bool = False,
    cache: EmbeddingCache | None = None,
) -> list[dict]:
    """Score each image against all keywords.

    Returns a list of dicts with:
      - path: image file path
      - scores: {keyword: cosine_similarity}
      - score: arithmetic mean of cosine similarities (primary ranking)
      - min_score: minimum similarity across keywords
    """
    prompts = [prompt_template.format(kw) for kw in keywords]
    text_emb = model.encode_texts(prompts)

    results = []
    for path in tqdm(image_paths, desc="Scoring", file=sys.stderr,
                     disable=quiet):
        try:
            img_emb = _get_embedding(model, path, cache)
        except OSError as e:
            if e.errno is not None:
                raise
            tqdm.write(f"  skipping {path}: {e}", file=sys.stderr)
            continue
        sims = (img_emb @ text_emb.T)[0]
        results.append(_build_result(path, keywords, sims))

    return results
