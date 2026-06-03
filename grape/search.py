from __future__ import annotations

import logging
import os
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from tqdm import tqdm

from grape.cache import stat_key_from_stat

# CLIPModel is under TYPE_CHECKING so that importing search.py does
# not pull in torch/open_clip (~2 s).  The cache-hit path never needs them.
if TYPE_CHECKING:
    from grape.cache import EmbeddingCache
    from grape.cli import _LazyModel
    from grape.model import CLIPModel

log = logging.getLogger("grape")


class ImageRecord(NamedTuple):
    # ``path`` is the display string -- preserves the caller's input style,
    # including a leading ``./`` the way ``find``/``rg``/``du`` do.  Wrapping
    # it in ``pathlib.Path`` would normalize the prefix away.  ``path_key`` is
    # the realpath used as the SQLite cache key.
    path: str
    path_key: str
    file_stat: str

    @classmethod
    def from_display_path(cls, display: str) -> "ImageRecord":
        """Build a record from a user-facing path. Resolves realpath +
        stats the file once, so callers can't drift the cache key by
        recomputing it inconsistently elsewhere."""
        path_key = os.path.realpath(display)
        return cls(
            path=display,
            path_key=path_key,
            file_stat=stat_key_from_stat(os.stat(path_key)),
        )


@dataclass
class ScoredImage:
    # Display string (see ImageRecord). Construct a Path locally where
    # pathlib semantics are needed.
    path: str
    scores: dict[str, float] = field(default_factory=dict)
    like_scores: list[tuple[str, float]] = field(default_factory=list)
    score: float = 0.0


def is_image(
    path: str,
    cache: EmbeddingCache | None = None,
    *,
    path_key: str | None = None,
    file_stat: str | None = None,
) -> bool:
    """Check whether a file is a recognized, loadable image.

    When a cache is provided, consults and updates it to avoid
    re-opening files already known not to be images.

    Uses ``im.load()`` to catch truncated images and video containers
    that pass header-only checks (e.g. ``im.verify()``). Only called
    for files grape has never seen; known paths skip this via image_paths.
    """
    if cache is not None and cache.is_not_image(
        path, path_key=path_key, file_stat=file_stat
    ):
        return False
    try:
        with Image.open(path) as im:
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
) -> list[str]:
    """Return sorted display paths for images under *directory*."""
    return sorted(iter_images(directory, recursive=recursive, cache=cache))


def iter_image_records(
    directory: str,
    recursive: bool = False,
    cache: EmbeddingCache | None = None,
    *,
    image_hits: set[tuple[str, str]] | None = None,
    image_paths: set[str] | None = None,
    not_image_hits: set[tuple[str, str]] | None = None,
) -> Iterator[ImageRecord]:
    """Yield image records under *directory* with cache key metadata.

    Display paths (``record.path``) preserve the caller's input style:
    a relative *directory* yields relative paths, an absolute one yields
    absolute paths.  Cache keys (``record.path_key``) always use the
    realpath so symlink aliases, ``./`` prefixes, and the like don't
    fragment the cache between scan modes.
    """
    real_root = os.path.realpath(directory)
    if not os.path.isdir(real_root):
        return
    if image_hits is None and cache is not None:
        image_hits = cache.image_hit_index()
    if image_paths is None and image_hits is not None:
        image_paths = {p for p, _ in image_hits}
    if not_image_hits is None and cache is not None:
        not_image_hits = cache.not_image_index()
    # Track visited real directory paths to avoid infinite loops from
    # symlink cycles (e.g. a -> b -> a).
    seen_dirs: set[str] = {real_root}
    # Each stack entry pairs a display directory (what the user sees)
    # with the realpath we actually scandir.  Scanning the realpath
    # avoids surprises if intermediate dirs are symlinks.
    stack: list[tuple[str, str]] = [(directory, real_root)]
    while stack:
        display_dir, real_dir = stack.pop()
        log.debug("scanning dir %s", display_dir)
        with os.scandir(real_dir) as entries:
            for entry in entries:
                # Check file first to avoid calling is_dir() on every file.
                # On large flat trees this removes a costly extra syscall.
                if not entry.is_file():
                    if recursive and entry.is_dir():
                        real_child = os.path.realpath(entry.path)
                        if real_child not in seen_dirs:
                            seen_dirs.add(real_child)
                            stack.append(
                                (
                                    os.path.join(display_dir, entry.name),
                                    real_child,
                                )
                            )
                    continue
                # Raw string concatenation, not Path: pathlib normalizes
                # away "./" the way Unix tools do not.
                display_path = os.path.join(display_dir, entry.name)
                # entry.path has real_dir as parent, so it is the realpath
                # of the file unless the leaf itself is a symlink.
                if entry.is_symlink():
                    real_path = os.path.realpath(entry.path)
                else:
                    real_path = entry.path
                stat_key = stat_key_from_stat(entry.stat())
                cache_key = (real_path, stat_key)
                # Check not-image before image-hit: a file that was
                # once embedded but later found to be broken (truncated,
                # video container, etc.) must stay excluded.
                if not_image_hits is not None and cache_key in not_image_hits:
                    continue
                if image_hits is not None and cache_key in image_hits:
                    yield ImageRecord(
                        path=display_path,
                        path_key=real_path,
                        file_stat=stat_key,
                    )
                    continue
                # image_paths covers all models: whether a file is an image
                # doesn't depend on which model encoded it, so a hit under
                # any model (or with a stale stat) skips the format check,
                # which requires opening the file to read its header.
                if image_paths is not None and real_path in image_paths:
                    yield ImageRecord(
                        path=display_path,
                        path_key=real_path,
                        file_stat=stat_key,
                    )
                    continue
                if is_image(
                    display_path,
                    cache,
                    path_key=real_path,
                    file_stat=stat_key,
                ):
                    yield ImageRecord(
                        path=display_path,
                        path_key=real_path,
                        file_stat=stat_key,
                    )


def iter_images(
    directory: str,
    recursive: bool = False,
    cache: EmbeddingCache | None = None,
    *,
    image_hits: set[tuple[str, str]] | None = None,
    not_image_hits: set[tuple[str, str]] | None = None,
) -> Iterator[str]:
    """Yield display image paths under *directory* (path-only wrapper)."""
    for record in iter_image_records(
        directory,
        recursive=recursive,
        cache=cache,
        image_hits=image_hits,
        not_image_hits=not_image_hits,
    ):
        yield record.path


def _build_result(
    path: str,
    keywords: list[str],
    sims: NDArray[np.float32],
) -> ScoredImage:
    return ScoredImage(
        path=os.fspath(path),
        scores={kw: float(s) for kw, s in zip(keywords, sims)},
        score=float(sims.mean()),
    )


def _get_embedding(
    model: CLIPModel | _LazyModel,
    path: str,
    cache: EmbeddingCache | None,
    *,
    path_key: str | None = None,
    file_stat: str | None = None,
) -> NDArray[np.float32]:
    """Encode an image, using the cache when available.

    When the caller already has the path's realpath and stat-token (e.g.
    from an ``ImageRecord``), pass them through ``path_key`` / ``file_stat``
    to avoid a redundant realpath+stat inside the cache and to prevent
    a TOCTOU race where the file is touched between scan and encode.
    """
    if cache is not None:
        cached = cache.get(
            path, model.model_id(),
            path_key=path_key, file_stat=file_stat,
        )
        if cached is not None:
            return cached
    emb = model.encode_image(os.fspath(path))
    if cache is not None:
        cache.put(
            path, model.model_id(), emb,
            path_key=path_key, file_stat=file_stat,
        )
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
    image_path: str,
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
    image_path: str,
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
    image_paths: Sequence[str | Path],
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
