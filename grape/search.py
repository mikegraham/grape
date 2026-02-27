from __future__ import annotations

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


def _is_image(path: Path, cache: EmbeddingCache | None = None) -> bool:
    """Check whether a file is a recognized image format.

    When a cache is provided, consults and updates it to avoid
    re-opening files already known not to be images.
    """
    if cache is not None and cache.is_not_image(path):
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
            cache.put_not_image(path)
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
    root = Path(directory)
    if not root.is_dir():
        return []
    pattern = "**/*" if recursive else "*"
    return sorted(
        f for f in root.glob(pattern)
        if f.is_file() and _is_image(f, cache)
    )


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
