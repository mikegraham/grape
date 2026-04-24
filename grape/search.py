from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from tqdm import tqdm

# CLIPModel is under TYPE_CHECKING so that importing search.py does
# not pull in torch/open_clip (~2 s).  The cache-hit path never needs them.
if TYPE_CHECKING:
    from grape.cache import EmbeddingCache
    from grape.model import CLIPModel


class ImageRecord(NamedTuple):
    path: Path
    path_key: str
    file_stat: str


@dataclass
class ScoredImage:
    path: Path
    scores: dict[str, float] = field(default_factory=dict)
    like_scores: list[tuple[str, float]] = field(default_factory=list)
    score: float = 0.0


def _stat_key_from_stat(st: os.stat_result) -> str:
    """Build cache invalidation key from an existing stat result."""
    # Keep exact spacing compatible with json.dumps(list) to preserve
    # cache-key stability while avoiding JSON encoder overhead.
    return (
        f"[{st.st_size}, {st.st_mtime_ns},"
        f" {st.st_ino}, {st.st_dev}, {st.st_ctime_ns}]"
    )


def is_image(
    path: Path,
    cache: EmbeddingCache | None = None,
    *,
    path_key: str | None = None,
    file_stat: str | None = None,
) -> bool:
    """Check whether a file is a recognized, loadable image.

    When a cache is provided, consults and updates it to avoid
    re-opening files already known not to be images.

    Uses ``im.load()`` instead of ``im.verify()`` because verify only
    checks headers -- truncated images and some video containers pass
    verify but fail when actual pixel data is decoded later.
    See https://github.com/python-pillow/Pillow/issues/3012
    """
    if cache is not None and cache.is_not_image(
        path, path_key=path_key, file_stat=file_stat
    ):
        return False
    try:
        with Image.open(path) as im:
            # load() decodes pixel data, catching truncated files and
            # formats that verify() lets through (e.g. some .mp4 files).
            im.load()
        return True
    except SyntaxError:
        # PIL raises SyntaxError for some corrupt/unrecognized formats.
        if cache is not None:
            cache.put_not_image(path, path_key=path_key, file_stat=file_stat)
        return False
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
    """Return sorted image paths under *directory*."""
    return sorted(iter_images(directory, recursive=recursive, cache=cache))


def iter_image_records(
    directory: str,
    recursive: bool = False,
    cache: EmbeddingCache | None = None,
    *,
    image_hits: set[tuple[str, str]] | None = None,
    not_image_hits: set[tuple[str, str]] | None = None,
) -> Iterator[ImageRecord]:
    """Yield image records under *directory* with cache key metadata."""
    # Canonicalize the root once so parent-directory symlinks are resolved
    # without paying realpath cost repeatedly per discovered file.
    root = Path(os.path.realpath(directory))
    if not root.is_dir():
        return
    if image_hits is None and cache is not None:
        image_hits = cache.image_hit_index()
    if not_image_hits is None and cache is not None:
        not_image_hits = cache.not_image_index()
    # Track visited real directory paths to avoid infinite loops from
    # symlink cycles (e.g. a -> b -> a).
    seen_dirs: set[str] = {str(root)}
    stack = [root]
    while stack:
        current = stack.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                # Check file first to avoid calling is_dir() on every file.
                # On large flat trees this removes a costly extra syscall.
                if not entry.is_file():
                    if recursive and entry.is_dir():
                        real = os.path.realpath(entry.path)
                        if real not in seen_dirs:
                            seen_dirs.add(real)
                            stack.append(Path(entry.path))
                    continue
                path = Path(entry.path)
                stat_key = _stat_key_from_stat(entry.stat())
                cache_key = (entry.path, stat_key)
                # Check not-image before image-hit: a file that was
                # once embedded but later found to be broken (truncated,
                # video container, etc.) must stay excluded.
                if not_image_hits is not None and cache_key in not_image_hits:
                    continue
                if image_hits is not None and cache_key in image_hits:
                    yield ImageRecord(
                        path=path,
                        path_key=entry.path,
                        file_stat=stat_key,
                    )
                    continue
                if is_image(
                    path,
                    cache,
                    path_key=entry.path,
                    file_stat=stat_key,
                ):
                    yield ImageRecord(
                        path=path,
                        path_key=entry.path,
                        file_stat=stat_key,
                    )


def iter_images(
    directory: str,
    recursive: bool = False,
    cache: EmbeddingCache | None = None,
    *,
    image_hits: set[tuple[str, str]] | None = None,
    not_image_hits: set[tuple[str, str]] | None = None,
) -> Iterator[Path]:
    """Yield image paths under *directory* (path-only wrapper)."""
    for record in iter_image_records(
        directory,
        recursive=recursive,
        cache=cache,
        image_hits=image_hits,
        not_image_hits=not_image_hits,
    ):
        yield record.path


def _build_result(
    path: Path,
    keywords: list[str],
    sims: NDArray[np.float32],
) -> ScoredImage:
    return ScoredImage(
        path=path,
        scores={kw: float(s) for kw, s in zip(keywords, sims)},
        score=float(sims.mean()),
    )


def _get_embedding(
    model: CLIPModel,
    path: Path,
    cache: EmbeddingCache | None,
) -> NDArray[np.float32]:
    """Encode an image, using the cache when available."""
    if cache is not None:
        cached = cache.get(path, model.model_id())
        if cached is not None:
            return cached
    emb = model.encode_image(str(path))
    if cache is not None:
        cache.put(path, model.model_id(), emb)
    return emb


def _encode_keyword_embeddings(
    model: CLIPModel,
    keywords: list[str],
    prompt_template: str,
    prompt_templates: list[str] | None,
) -> NDArray[np.float32]:
    """Encode keyword prompts, optionally with prompt ensembling."""
    if prompt_templates is None:
        prompts = [prompt_template.format(kw) for kw in keywords]
        return model.encode_texts(prompts)

    if not prompt_templates:
        raise ValueError("prompt_templates must not be empty")

    prompts = [
        template.format(keyword)
        for keyword in keywords
        for template in prompt_templates
    ]
    text_emb = model.encode_texts(prompts)
    num_keywords = len(keywords)
    num_templates = len(prompt_templates)
    reshaped = text_emb.reshape(num_keywords, num_templates, -1)
    merged = reshaped.mean(axis=1)
    norms = np.linalg.norm(merged, axis=1, keepdims=True)
    merged = merged / np.clip(norms, 1e-12, None)
    return cast(NDArray[np.float32], merged.astype(np.float32))


def encode_keywords(
    model: CLIPModel,
    keywords: list[str],
    prompt_template: str = "a photo of {}",
    prompt_templates: list[str] | None = None,
) -> NDArray[np.float32]:
    """Encode keyword prompts once for repeated image scoring."""
    return _encode_keyword_embeddings(
        model,
        keywords,
        prompt_template=prompt_template,
        prompt_templates=prompt_templates,
    )


def score_image_with_text_embeddings(
    model: CLIPModel,
    image_path: Path,
    keywords: list[str],
    text_emb: NDArray[np.float32],
    cache: EmbeddingCache | None = None,
) -> ScoredImage:
    """Score a single image using precomputed keyword embeddings."""
    img_emb = _get_embedding(model, image_path, cache)
    sims = (img_emb @ text_emb.T)[0]
    return _build_result(image_path, keywords, sims)


def score_image(
    model: CLIPModel,
    image_path: Path,
    keywords: list[str],
    prompt_template: str = "a photo of {}",
    prompt_templates: list[str] | None = None,
    cache: EmbeddingCache | None = None,
) -> ScoredImage:
    """Score a single image against all keywords."""
    text_emb = encode_keywords(
        model,
        keywords,
        prompt_template=prompt_template,
        prompt_templates=prompt_templates,
    )
    return score_image_with_text_embeddings(
        model,
        image_path,
        keywords,
        text_emb,
        cache=cache,
    )


def score_images(
    model: CLIPModel,
    image_paths: list[Path],
    keywords: list[str],
    prompt_template: str = "a photo of {}",
    prompt_templates: list[str] | None = None,
    quiet: bool = False,
    cache: EmbeddingCache | None = None,
) -> list[ScoredImage]:
    """Score each image against all keywords."""
    text_emb = encode_keywords(
        model,
        keywords,
        prompt_template=prompt_template,
        prompt_templates=prompt_templates,
    )

    results = []
    for path in tqdm(image_paths, desc="Scoring", file=sys.stderr,
                     disable=quiet):
        try:
            results.append(
                score_image_with_text_embeddings(
                    model,
                    path,
                    keywords,
                    text_emb,
                    cache=cache,
                )
            )
        except OSError as e:
            if e.errno is not None:
                raise
            tqdm.write(f"  skipping {path}: {e}", file=sys.stderr)
            continue

    return results
