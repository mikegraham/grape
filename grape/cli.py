from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import shlex
import sys
import tempfile
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, closing, nullcontext
from dataclasses import dataclass
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image, UnidentifiedImageError

from grape.search import (
    ImageRecord,
    ScoredImage,
    is_image,
    iter_image_records,
)

if TYPE_CHECKING:
    import jinja2
    import numpy as np
    from numpy.typing import NDArray

    from grape.cache import EmbeddingCache
    from grape.model import CLIPModel

log = logging.getLogger("grape")

# EVA-CLIP: Improved Training Techniques for CLIP at Scale
# (Sun et al., 2023) https://arxiv.org/abs/2303.15389
DEFAULT_MODEL = "EVA02-L-14/merged2b_s4b_b131k"

# Prompt ensembling: average embeddings across multiple prompt templates
# per keyword, then L2-normalize. Improves zero-shot accuracy over a
# single template. See CLIP (Radford et al., 2021) Section 3.1.4 and
# Appendix A. https://arxiv.org/abs/2103.00020
DEFAULT_PROMPT_ENSEMBLE = [
    "a photo of a {}",
    "a photo of the {}",
    "a photo of including {}",
]

_HTML_TEMPLATE_TEXT = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>grape results</title>
  <style>
    /* proximity, not mandatory: small wheel scrolls don't snap. */
    html { scroll-snap-type: y proximity; }
    body { margin: 12px; color-scheme: light dark; font-family: sans-serif; }
    .meta { font-size: 12px; line-height: 1.3; margin: 0; }
    /* Space/PgDn lands on the start of each result. */
    .path-line {
      display: flex; align-items: baseline; gap: 8px;
      scroll-snap-align: start;
    }
    /* flex: 0 1 auto so the path uses its natural width when short
       (resolution sits right next to it) and shrinks with ellipsis
       only when there isn't room. */
    .path {
      flex: 0 1 auto; min-width: 0;
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .res { flex: 0 0 auto; opacity: 0.6; font-size: 11px; }
    .scoreline {
      opacity: 0.6; font-size: 11px; margin-bottom: 4px;
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    /* max-height: 100vh: never taller than one viewport. */
    img { display: block; max-width: 100%; max-height: 100vh; margin-bottom: 16px; }
  </style>
</head>
<body>
  <p class="meta">{{ count }} image(s) for {{ keywords }}</p>
  {% for row in rows %}
  <p class="meta path-line">
    <span class="path" title="{{ row.path_text }}">{{ row.path_text }}</span>
    {% if row.resolution %}<span class="res">{{ row.resolution }}</span>{% endif %}
  </p>
  <p class="meta scoreline" title="{{ row.score_line }}">{{ row.score_line }}</p>
  <img src="{{ row.image_src }}" alt="{{ row.path_text }}">
  {% endfor %}
</body>
</html>
""".strip()



def _get_webview() -> Any:
    """Lazily import pywebview for --view mode only."""
    import webview

    return webview


# ---------------------------------------------------------------------------
# Pipeline building blocks
#
# Each function below is a node in the in-process task graph wired up by
# _run_pipeline(), which runs the independent roots concurrently on a
# stdlib ThreadPoolExecutor:
#
#   _load_model (instant) ------+-- _encode_keywords ------+
#                                \-- _encode_like_images --+ |
#                                                    _combine +
#   _resolve_and_index_cache --+                             |
#   _scan_files ------+--------+-- _prepare                  |
#                                   \-- _score_all ----------+
#
# After the graph completes, _filter_and_sort and _emit run on the main
# thread (pywebview requires it, stdout is cleaner without interleaving).
#
# We used to use dask.delayed here, but dask itself takes ~99ms to import
# -- a tax the warm-cache path can't afford. concurrent.futures has zero
# extra import cost (stdlib) and the graph is small enough that explicit
# Future chaining is easy to read.
#
# Performance notes:
# - _load_model creates a _LazyModel proxy; the expensive torch/open_clip
#   import and weight loading only happen if a downstream node actually
#   accesses the model (i.e. cache miss).  When everything is cached,
#   no import happens and _scan_files runs uncontested by the GIL.
# - _resolve_and_index_cache caches model_id in SQLite so it can skip
#   the open_clip import on warm-cache runs.  An assert in
#   _encode_keywords verifies the cached model_id when the model loads.
# ---------------------------------------------------------------------------


class _LazyModel:
    """Deferred model loader: imports torch/open_clip only on first use.

    When all embeddings (text + image) are cached, the model is never
    accessed and no heavy import happens -- keeping scan_files free
    from GIL contention.  Thread-safe for concurrent worker threads.
    """

    def __init__(
        self, model_name: str, pretrained: str, quiet: bool,
    ) -> None:
        self._model_name = model_name
        self._pretrained = pretrained
        self._quiet = quiet
        self._model: CLIPModel | None = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> CLIPModel:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    import grape.model
                    grape.model.preload_weights(
                        self._model_name, self._pretrained,
                    )
                    self._model = grape.model.CLIPModel(
                        model_name=self._model_name,
                        pretrained=self._pretrained,
                        quiet=self._quiet,
                    )
        return self._model

    def encode_image(self, image_path: str) -> NDArray[np.float32]:
        return self._ensure_loaded().encode_image(image_path)

    def encode_texts(self, texts: list[str]) -> NDArray[np.float32]:
        return self._ensure_loaded().encode_texts(texts)

    def model_id(self) -> str:
        return self._ensure_loaded().model_id()


def _load_model(
    model_name: str, pretrained: str, quiet: bool,
) -> _LazyModel:
    """Return a lazy model proxy -- no import until first method call.

    When everything is cached, no downstream node touches the model
    and torch/open_clip are never imported.  scan_files runs without
    GIL contention.
    """
    return _LazyModel(model_name, pretrained, quiet)


def _encode_keywords(
    model: _LazyModel,
    score_keywords: list[str],
    prompt_templates: list[str],
    cache_context: tuple[str | None, dict[tuple[str, str], NDArray[np.float32]] | None],
    cache: EmbeddingCache | None,
) -> NDArray[np.float32]:
    """Encode keyword prompts into text embeddings.

    Caches at the prompt level (the actual text sent to the model), not
    at the keyword level.  This means "a photo of a dog" is cached once
    and reused regardless of which template set generated it.

    With ensembling, each keyword produces N prompts.  We cache each
    prompt individually, then average + renormalize per keyword.
    """
    import numpy as np

    model_id = cache_context[0] if cache_context else None

    # Build all prompts: one per (keyword, template) pair.
    all_prompts = [
        template.format(kw)
        for kw in score_keywords
        for template in prompt_templates
    ]

    # Check cache for each prompt.
    cached_prompts: dict[str, NDArray[np.float32]] = {}
    if cache is not None and model_id is not None:
        cached_prompts = cache.get_text_embeddings(model_id, all_prompts)

    uncached_prompts = [p for p in all_prompts if p not in cached_prompts]
    log.debug(
        "text prompts: %d total, %d cached, %d to encode",
        len(all_prompts), len(cached_prompts), len(uncached_prompts),
    )

    if uncached_prompts:
        # Encode only uncached prompts through the model.
        fresh = model.encode_texts(uncached_prompts)

        # Now that the model is loaded, verify our cached model_id
        # is still correct (could be stale if HF cache was updated).
        real_model_id = model.model_id()
        if model_id is not None and real_model_id != model_id:
            if real_model_id.startswith(model_id + "@"):
                # Format migration: bare hf_hub path gained a @commit suffix.
                # Rename all existing cache entries so they don't need
                # to be recomputed.
                log.debug("migrating model_id %r -> %r", model_id, real_model_id)
                if cache is not None:
                    cache.rename_model_id(model_id, real_model_id)
            else:
                raise AssertionError(
                    f"model_id mismatch: cached {model_id!r},"
                    f" resolved {real_model_id!r}"
                )
        model_id = real_model_id

        new_pairs: list[tuple[str, NDArray[np.float32]]] = []
        for i, prompt in enumerate(uncached_prompts):
            emb = fresh[i : i + 1]
            cached_prompts[prompt] = emb
            new_pairs.append((prompt, emb))
        if cache is not None and model_id is not None:
            cache.put_text_embeddings(model_id, new_pairs)

    # Reassemble per-keyword embeddings: average across templates,
    # then L2-normalize (same logic as _encode_keyword_embeddings).
    n_templates = len(prompt_templates)
    keyword_embs: list[NDArray[np.float32]] = []
    for kw in score_keywords:
        prompts = [t.format(kw) for t in prompt_templates]
        embs = np.vstack([cached_prompts[p] for p in prompts])
        if n_templates == 1:
            keyword_embs.append(embs)
        else:
            merged = embs.mean(axis=0, keepdims=True)
            norm = np.linalg.norm(merged)
            if norm == 0:
                raise ValueError(f"prompt embeddings for {kw!r} average to zero")
            merged = merged / norm
            keyword_embs.append(merged.astype(np.float32))

    return np.vstack(keyword_embs)



def _encode_like_images(
    model: _LazyModel,
    like_paths: list[str],
    cache_context: tuple[str | None, dict[tuple[str, str], NDArray[np.float32]] | None],
    cache: EmbeddingCache | None,
) -> NDArray[np.float32]:
    """Encode --like reference images into query embeddings.

    Uses cached embeddings when available so that --like self-matches
    produce identical bytes (and therefore exactly 1.0 similarity).
    Falls back to model.encode_image() on cache miss.
    """
    import numpy as np
    model_id = cache_context[0] if cache_context else None
    embeddings = []
    for p in like_paths:
        cached = None
        if cache is not None and model_id is not None:
            cached = cache.get(p, model_id)
        if cached is not None:
            embeddings.append(cached)
        else:
            embeddings.append(model.encode_image(p))
    return np.vstack(embeddings)


def _combine_query_embeddings(
    text_emb: NDArray[np.float32] | None,
    like_emb: NDArray[np.float32] | None,
) -> NDArray[np.float32]:
    """Stack text and --like image query embeddings into one matrix."""
    import numpy as np
    parts = [e for e in (text_emb, like_emb) if e is not None]
    return np.vstack(parts)


def _resolve_and_index_cache(
    model_name: str,
    pretrained: str,
    cache: "EmbeddingCache | None",
) -> tuple[str | None, dict[tuple[str, str], NDArray[np.float32]] | None]:
    """Resolve model_id and materialize the cache index.

    Caches model_id in SQLite so subsequent runs skip the torch import.
    On first use, imports grape.model to resolve the id.
    """
    if cache is None:
        return None, None

    model_id = cache.get_model_id(model_name, pretrained)
    if model_id is None:
        # First run with this model -- must import to resolve.
        # _LazyModel is already loading in a background thread, so
        # grape.model may or may not be imported yet.
        import grape.model
        model_id = grape.model.resolve_model_id(model_name, pretrained)
        cache.put_model_id(model_name, pretrained, model_id)
        log.debug("model_id resolved fresh: %s", model_id)
    else:
        log.debug("model_id from cache: %s", model_id)

    cached_index = cache.embedding_index_for_model(model_id)
    log.debug("cache index: %d image embeddings", len(cached_index))
    return model_id, cached_index


def _expand_stdin_paths(path_args: list[str]) -> list[str]:
    """Replace any '-' arguments with newline-delimited lines from stdin."""
    if "-" not in path_args:
        return path_args
    expanded: list[str] = []
    for p in path_args:
        if p == "-":
            for raw in sys.stdin:
                line = raw.rstrip("\n\r")
                # Skip NUL bytes -- a path with one would crash os.stat
                # with embedded-null ValueError. Most likely a user
                # piped find -print0 without xargs -0.
                if line and "\0" not in line:
                    expanded.append(line)
        else:
            expanded.append(p)
    return expanded


def _scan_files(
    path_args: list[str],
    recursive: bool,
    cache: "EmbeddingCache | None",
) -> tuple[list[ImageRecord], ScanReport]:
    """Discover image files from CLI paths. Independent of model loading."""
    if cache is not None:
        image_hits = cache.image_hit_index()
        image_paths = {p for p, _ in image_hits}
        not_image_hits = cache.not_image_index()
    else:
        image_hits = None
        image_paths = None
        not_image_hits = None
    items: list[ImageRecord] = []
    error_message: str | None = None

    for p in path_args:
        # Use os.path here, not pathlib: pathlib normalizes "./" away,
        # but `find`/`rg`/`du` (and now grape) preserve the caller's
        # input style verbatim.
        if os.path.isfile(p):
            record = ImageRecord.from_display_path(p)
            cache_key = (record.path_key, record.file_stat)
            if not_image_hits is not None and cache_key in not_image_hits:
                continue
            if (
                (image_hits is None or cache_key not in image_hits)
                and (image_paths is None or record.path_key not in image_paths)
                and not is_image(
                    record.path, cache,
                    path_key=record.path_key, file_stat=record.file_stat,
                )
            ):
                continue
            items.append(record)
            continue
        if os.path.isdir(p):
            if not recursive:
                print(f"grape: {p}: Is a directory", file=sys.stderr)
                continue
            log.debug("scanning %s ...", p)
            items.extend(iter_image_records(
                p,
                recursive=True,
                cache=cache,
                image_hits=image_hits,
                image_paths=image_paths,
                not_image_hits=not_image_hits,
            ))
            continue
        error_message = f"grape: {p}: No such file or directory"
        break

    log.debug("scan_files: %d images found", len(items))
    return items, ScanReport(
        image_count=len(items), error_message=error_message,
    )


def _prepare_cached_embeddings(
    scan_result: tuple[list[ImageRecord], ScanReport],
    cache_context: tuple[str | None, dict[tuple[str, str], NDArray[np.float32]] | None],
) -> tuple[
    NDArray[np.float32] | None, list[ImageRecord], list[ImageRecord], ScanReport,
]:
    """Split scanned images into cached/uncached and vstack cached vectors.

    Runs as soon as scanning and cache indexing finish -- does not wait for
    model loading or text encoding, so the ~23ms vstack overlaps with those.
    Returns (image_emb_matrix | None, cached_items, uncached_items, scan_done).
    """
    import numpy as np

    _model_id, cached_index = cache_context
    items, scan_done = scan_result

    cached_items: list[ImageRecord] = []
    cached_vectors: list[NDArray[np.float32]] = []
    uncached_items: list[ImageRecord] = []

    if cached_index is not None:
        for item in items:
            emb = cached_index.get((item.path_key, item.file_stat))
            if emb is not None:
                cached_items.append(item)
                cached_vectors.append(emb)
            else:
                uncached_items.append(item)
    else:
        uncached_items = items

    log.debug(
        "prepare: %d cached, %d uncached images",
        len(cached_items), len(uncached_items),
    )
    image_emb = np.vstack(cached_vectors) if cached_vectors else None
    return image_emb, cached_items, uncached_items, scan_done


def _score_all(
    prepared: tuple[
        NDArray[np.float32] | None, list[ImageRecord], list[ImageRecord], ScanReport,
    ],
    model: _LazyModel,
    score_keywords: list[str],
    like_paths: list[str],
    text_emb: NDArray[np.float32],
    cache: EmbeddingCache | None,
    quiet: bool,
    verbose: bool,
) -> tuple[list[ScoredImage], ScanReport]:
    """Score all scanned images against text embeddings.

    ``score_keywords`` are text keyword labels (include + exclude).
    ``like_paths`` are --like image paths.  Keeping them separate avoids
    score-dict key collisions when basenames repeat or match a keyword.
    """
    from tqdm import tqdm

    from grape.search import _get_embedding

    image_emb, cached_items, uncached_items, scan_done = prepared
    n_text = len(score_keywords)

    def _make_result(path: str, sims: NDArray[np.float32]) -> ScoredImage:
        return ScoredImage(
            path=path,
            scores={
                kw: float(s)
                for kw, s in zip(score_keywords, sims[:n_text])
            },
            like_scores=[
                (lp, float(s))
                for lp, s in zip(like_paths, sims[n_text:])
            ],
            score=float(sims.mean()),
        )

    results: list[ScoredImage] = []

    # Fast path: score cached items via a single matrix multiply.
    if image_emb is not None:
        sims_matrix = image_emb @ text_emb.T
        for idx, item in enumerate(cached_items):
            results.append(_make_result(item.path, sims_matrix[idx]))

    # Slow path: encode uncached images through the model one at a time.
    if uncached_items:
        log.debug("score_all: encoding %d uncached images", len(uncached_items))
    for item in tqdm(
        uncached_items, desc="Encoding", file=sys.stderr, disable=quiet,
    ):
        if verbose:
            # tqdm.write avoids breaking the progress bar.
            tqdm.write(item.path, file=sys.stderr)
        try:
            img_emb = _get_embedding(model, item.path, cache)
            sims = (img_emb @ text_emb.T)[0]
            results.append(_make_result(item.path, sims))
        except SyntaxError as e:
            if not quiet:
                print(f"  skipping {item.path}: {e}", file=sys.stderr)
            if cache is not None:
                cache.put_not_image(
                    item.path,
                    path_key=item.path_key,
                    file_stat=item.file_stat,
                )
        except OSError as e:
            if e.errno is not None:
                raise
            if not quiet:
                print(f"  skipping {item.path}: {e}", file=sys.stderr)
            # Record as not-image so future scans skip this file.
            if cache is not None:
                cache.put_not_image(
                    item.path,
                    path_key=item.path_key,
                    file_stat=item.file_stat,
                )

    return results, scan_done


def _filter_and_sort(
    score_result: tuple[list[ScoredImage], ScanReport],
    keywords: list[str],
    exclude_keywords: list[str],
    like_names: list[str],
    threshold: float | None,
    top: int | None,
    quiet: bool,
) -> list[ScoredImage]:
    """Apply excludes, sort, threshold, top-N, and validate scan status.

    Runs on the main thread after the executor finishes.
    """
    results, scan_done = score_result

    if scan_done.error_message:
        print(scan_done.error_message, file=sys.stderr)
        sys.exit(1)
    if scan_done.image_count == 0:
        print("grape: no images found", file=sys.stderr)
        sys.exit(1)

    if not quiet:
        n = len(results)
        query_text = _format_query_summary(
            keywords, exclude_keywords, like_names,
        )
        print(
            f"{n} image{'s' * (n != 1)}, {query_text}",
            file=sys.stderr,
        )

    _apply_excluded_keywords(
        results, keywords, exclude_keywords,
    )

    results.sort(key=lambda r: r.score, reverse=True)

    if threshold is not None:
        results = [r for r in results if r.score >= threshold]
    if top is not None:
        results = results[:top]
    return results


def _emit(
    results: list[ScoredImage],
    keywords: list[str],
    exclude_keywords: list[str],
    like_names: list[str],
    scores: bool,
    verbose: bool,
    print0: bool,
    view: bool,
    quiet: bool,
) -> int:
    """Format and output results on the main thread.

    Runs after the pipeline finishes, so pywebview
    and stdout output happen on the main thread where they belong.
    """
    if not results:
        print("grape: no images above threshold", file=sys.stderr)
        return 0

    show_scores = scores or verbose
    if view:
        display_keywords = (
            keywords
            + [f"like:{name}" for name in like_names]
            + [f"not:{kw}" for kw in exclude_keywords]
        )
        html_doc = _format_html(results, display_keywords)
        _show_in_webview(html_doc)
        return len(results)
    if show_scores:
        print(_format_results(results, verbose=verbose))
        return len(results)
    if print0:
        for r in results:
            sys.stdout.write(f"{r.path}\0")
        sys.stdout.flush()
        return len(results)
    for r in results:
        print(shlex.quote(r.path))
    return len(results)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ScanReport:
    image_count: int
    error_message: str | None = None


# ---------------------------------------------------------------------------
# Pure helpers (not delayed -- used inside delayed tasks or at parse time)
# ---------------------------------------------------------------------------

def parse_keywords(raw: str, separator: str = ",") -> list[str]:
    """Split a keyword string on a separator. Strips whitespace from each.

    If *separator* is empty, the entire string is treated as a single keyword.
    """
    if not separator:
        stripped = raw.strip()
        return [stripped] if stripped else []
    return [k.strip() for k in raw.split(separator) if k.strip()]


def parse_prompt_templates(raw: str) -> list[str]:
    """Split comma-separated prompt templates."""
    return [t.strip() for t in raw.split(",") if t.strip()]


def _format_results(results: list[ScoredImage], verbose: bool) -> str:
    """Format scored results for display."""
    lines = []
    for r in results:
        lines.append(f"{r.score:.3f}  {shlex.quote(r.path)}")
        if verbose:
            parts = [f"  {kw}: {s:.3f}" for kw, s in r.scores.items()]
            for lp, s in r.like_scores:
                parts.append(f"  like:{Path(lp).name}: {s:.3f}")
            lines.append("".join(parts))
    return "\n".join(lines)


def _apply_excluded_keywords(
    results: list[ScoredImage],
    include_keywords: list[str],
    exclude_keywords: list[str],
) -> None:
    """Adjust result scores using include-vs-exclude keyword means.

    ``include_keywords`` are the text keywords to keep (not --like).
    Like scores contribute to the include mean via ``result.like_scores``.
    """
    for result in results:
        raw_scores = result.scores
        include_values = [raw_scores[kw] for kw in include_keywords]
        include_values += [s for _, s in result.like_scores]

        include_mean = (
            sum(include_values) / len(include_values)
            if include_values else 0.0
        )

        if exclude_keywords:
            exclude_components = [raw_scores[kw] for kw in exclude_keywords]
            exclude_mean = sum(exclude_components) / len(exclude_components)
        else:
            exclude_components = []
            exclude_mean = 0.0
        result.score = float(include_mean - exclude_mean)

        # Keep verbose output readable by labeling excluded keywords.
        labeled_scores: dict[str, float] = {}
        for kw in include_keywords:
            labeled_scores[kw] = raw_scores[kw]
        for kw, component in zip(exclude_keywords, exclude_components):
            labeled_scores[f"not:{kw}"] = component
        result.scores = labeled_scores


_html_template_cache: jinja2.Template | None = None


def _read_resolution(path: str) -> str | None:
    """Return ``"WIDTHxHEIGHT"`` or ``None`` if the image can't be read."""
    try:
        # Suppress DecompressionBombWarning: we're reading the header only,
        # not decompressing pixels, so the DOS-attack guard is a false positive.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(path) as im:
                w, h = im.size
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError):
        return None
    return f"{w}x{h}"


def _format_html(
    results: list[ScoredImage],
    keywords: list[str],
) -> str:
    """Format results as simple HTML with file-backed <img> tags."""
    global _html_template_cache  # noqa: PLW0603
    if _html_template_cache is None:
        from jinja2 import Template
        _html_template_cache = Template(_HTML_TEMPLATE_TEXT, autoescape=True)
    template = _html_template_cache
    rows: list[dict[str, str]] = []
    for r in results:
        # Path.as_uri() requires absolute; .absolute() is a no-op when already so.
        src_uri = Path(r.path).absolute().as_uri()
        parts: list[str] = [f"score: {r.score:.3f}"]
        for kw, score in r.scores.items():
            parts.append(f"{kw}: {score:.3f}")
        for lp, score in r.like_scores:
            parts.append(f"like:{Path(lp).name}: {score:.3f}")
        score_line = " \N{MIDDLE DOT} ".join(parts)
        rows.append({
            "image_src": src_uri,
            "path_text": r.path,
            "resolution": _read_resolution(r.path) or "",
            "score_line": score_line,
        })
    result: str = template.render(
        count=len(results),
        keywords=", ".join(keywords),
        rows=rows,
    )
    return result


def _show_in_webview(html_doc: str) -> None:
    """Display HTML in a native webview window."""
    webview = _get_webview()
    with tempfile.TemporaryDirectory(prefix="grape-view-") as tmpdir:
        html_path = Path(tmpdir) / "index.html"
        html_path.write_text(html_doc, encoding="utf-8")
        # debug=True is the only way pywebview 6.x enables the right-click
        # context menu (Copy Image, Save As) -- all backends hardcode the
        # coupling. OPEN_DEVTOOLS_IN_DEBUG=False keeps the inspector hidden.
        webview.settings["OPEN_DEVTOOLS_IN_DEBUG"] = False
        webview.create_window(
            "grape results",
            url=html_path.as_uri(),
            width=1280,
            height=900,
            maximized=True,
            resizable=True,
            text_select=True,
            zoomable=True,
            min_size=(480, 320),
        )
        webview.start(debug=True)


def _format_query_summary(
    keywords: list[str],
    exclude_keywords: list[str],
    like_names: list[str] | None = None,
) -> str:
    """Build status text shown before scoring."""
    query_parts: list[str] = []
    if keywords:
        query_parts.append(", ".join(keywords))
    if like_names:
        query_parts.append(f"like: {', '.join(like_names)}")
    if exclude_keywords:
        query_parts.append(f"excluding: {', '.join(exclude_keywords)}")
    if query_parts:
        return "; ".join(query_parts)
    return "(no keywords)"





# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Construct and return the CLI parser."""
    parser = argparse.ArgumentParser(
        prog="grape",
        description=(
            "Find images matching keywords using CLIP.\n"
            "\n"
            "Each image's score is:\n"
            "    mean(similarity to --keywords and --like)"
            " - mean(similarity to --exclude)"
        ),
        epilog="examples:\n"
               "  grape --keywords sunset photo.jpg\n"
               "  grape -R --keywords 'cat,dog' ~/Pictures\n"
               "  grape -R --like ref.jpg ~/Pictures\n"
               "  grape -R --keywords dog --exclude cat ~/Pictures\n"
               "  grape -R -n 20 --ensemble-prompts '{}'"
               " --keywords 'beautiful photo' --exclude 'ugly photo'"
               " --view ~/Pictures\n"
               "  find ~/Pictures -mtime -7 | grape --keywords selfie -\n"
               "  grape -R -print0 -n 10 --keywords cat ~/Pictures"
               " | xargs -0 cp -t ~/cats/\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # -- Positional ---------------------------------------------------------
    parser.add_argument(
        "path",
        nargs="+",
        metavar="PATH",
        help="image file; directory (use -R to recurse); '-' to read paths from stdin",
    )

    # -- Query: what to search for ------------------------------------------
    parser.add_argument(
        "-k", "--keywords",
        default=None,
        metavar="KEYWORDS",
        help="comma-separated keywords to match",
    )
    parser.add_argument(
        "-x", "--exclude",
        dest="exclude",
        default=None,
        metavar="KEYWORDS",
        help="comma-separated keywords to penalize",
    )
    parser.add_argument(
        "--like",
        action="append",
        default=[],
        metavar="IMAGE",
        help="reference image for similarity search (repeatable)",
    )

    # -- Input: where to search ---------------------------------------------
    parser.add_argument(
        "-R", "--dereference-recursive",
        action="store_true",
        dest="recursive",
        default=False,
        help="search directories recursively, following symlinks",
    )

    # -- Filtering -----------------------------------------------------------
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=None,
        metavar="SCORE",
        help="only show results with score >= SCORE",
    )
    parser.add_argument(
        "-n", "--top",
        type=int,
        default=None,
        metavar="N",
        help="show only top N results",
    )

    # -- Output format -------------------------------------------------------
    parser.add_argument(
        "-s", "--scores",
        action="store_true",
        help="show scores alongside paths",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="show per-keyword score breakdown (implies -s);"
             " print each newly-encoded file's path to stderr",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="suppress progress messages on stderr",
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "-print0",
        action="store_true",
        help="NUL-separated, unquoted paths",
    )
    _has_view_deps = (
        importlib.util.find_spec("webview") is not None
        and importlib.util.find_spec("jinja2") is not None
    )
    if _has_view_deps:
        output_group.add_argument(
            "--view",
            action="store_true",
            help="open results in a graphical window"
                 " (HTML grid with thumbnails)",
        )

    # -- Configuration -------------------------------------------------------
    from platformdirs import user_cache_dir
    default_cache = os.path.join(user_cache_dir("grape"), "embeddings.db")
    parser.add_argument(
        "--cache",
        metavar="PATH",
        default=default_cache,
        help=f"embedding cache file (default: {default_cache})",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        default=False,
        help="disable caching",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        metavar="NAME/PRETRAINED",
        help=f"CLIP model (default: {DEFAULT_MODEL});"
             " valid values at https://github.com/mlfoundations/open_clip",
    )
    parser.add_argument(
        "--ensemble-prompts",
        nargs="?",
        const=",".join(DEFAULT_PROMPT_ENSEMBLE),
        default=",".join(DEFAULT_PROMPT_ENSEMBLE),
        metavar="TEMPLATES",
        help="comma-separated prompt templates ('{}' is replaced with each"
             " keyword); the keyword's final embedding is the normalized mean"
             " across templates. Pass a single template (e.g. '{}') to"
             " disable ensembling. Default:"
             f" {','.join(DEFAULT_PROMPT_ENSEMBLE)}",
    )
    parser.add_argument(
        "--keyword-separator",
        default=",",
        metavar="SEP",
        help=argparse.SUPPRESS,
    )

    # -- Meta ----------------------------------------------------------------
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_pkg_version('grape-image-search')}",
    )
    return parser


def _validate_model_arg(parser: argparse.ArgumentParser, model: str) -> tuple[str, str]:
    """Validate model format and split into model/pretrained components."""
    if "/" not in model:
        parser.error(
            f"invalid model '{model}':"
            " expected format model_name/pretrained"
            " (e.g. ViT-B-16/laion2b_s34b_b88k)"
        )
    model_name, pretrained = model.split("/", 1)
    return model_name, pretrained


def _parse_prompt_templates(
    parser: argparse.ArgumentParser,
    raw_templates: str,
) -> list[str]:
    """Parse and validate prompt templates."""
    templates = parse_prompt_templates(raw_templates)
    invalid_templates = [t for t in templates if "{}" not in t]
    if invalid_templates:
        parser.error(
            "--ensemble-prompts templates must include '{}' placeholder"
        )
    return templates


# ---------------------------------------------------------------------------
# Pipeline assembly
# ---------------------------------------------------------------------------

def _run_pipeline(
    *,
    score_keywords: list[str],
    keywords: list[str],
    exclude_keywords: list[str],
    like_paths: list[str],
    model_name: str,
    pretrained: str,
    prompt_templates: list[str],
    cache: "EmbeddingCache | None",
    quiet: bool,
    path_args: list[str],
    recursive: bool,
    threshold: float | None,
    top: int | None,
    scores: bool,
    verbose: bool,
    print0: bool,
    view: bool,
) -> None:
    """Run the full CLI pipeline on a small ThreadPoolExecutor."""
    like_names = [Path(p).name for p in like_paths]

    # Each dependent task is wrapped in a closure that calls .result()
    # on its inputs -- the executor schedules Futures, workers blocked
    # on .result() don't consume CPU. max_workers=4 matches our truly
    # independent roots (model+resolve+scan, plus one for score_all);
    # higher counts left idle threads competing during the encode loop
    # and added ~9% to cold-large wall time.
    with ThreadPoolExecutor(max_workers=4, thread_name_prefix="grape") as ex:
        # Three independent roots, started in parallel.
        f_model = ex.submit(_load_model, model_name, pretrained, quiet)
        f_cache_ctx = ex.submit(
            _resolve_and_index_cache, model_name, pretrained, cache,
        )
        f_scan = ex.submit(_scan_files, path_args, recursive, cache)

        # Text keyword embeddings (None when no text keywords).
        f_text_emb = ex.submit(
            lambda: _encode_keywords(
                f_model.result(), score_keywords, prompt_templates,
                f_cache_ctx.result(), cache,
            ),
        ) if score_keywords else None
        # --like image embeddings (None when no --like). Depends on
        # cache_context so cached --like embeddings match exactly.
        f_like_emb = ex.submit(
            lambda: _encode_like_images(
                f_model.result(), like_paths, f_cache_ctx.result(), cache,
            ),
        ) if like_paths else None
        # Combined query matrix: [text keywords..., like embeddings...].
        f_query = ex.submit(
            lambda: _combine_query_embeddings(
                f_text_emb.result() if f_text_emb is not None else None,
                f_like_emb.result() if f_like_emb is not None else None,
            ),
        )
        # vstack of cached embeddings; overlaps with model load + encoding.
        f_prepared = ex.submit(
            lambda: _prepare_cached_embeddings(
                f_scan.result(), f_cache_ctx.result(),
            ),
        )
        # Convergence: scoring needs prepared + query + model.
        f_score = ex.submit(
            lambda: _score_all(
                f_prepared.result(), f_model.result(),
                score_keywords, like_paths, f_query.result(),
                cache, quiet, verbose,
            ),
        )
        score_result_value = f_score.result()

    # Post-processing and output run on the main thread (pywebview
    # requires it, and stdout is cleaner without thread interleaving).
    results = _filter_and_sort(
        score_result_value, keywords, exclude_keywords, like_names,
        threshold, top, quiet,
    )
    _emit(
        results, keywords, exclude_keywords, like_names,
        scores, verbose, print0, view, quiet,
    )


def main() -> None:
    # Hard-exit on Ctrl-C: torch threads otherwise wedge the shutdown.
    import signal
    signal.signal(signal.SIGINT, lambda *_: os._exit(128 + signal.SIGINT))

    parser = _build_parser()

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(
            level=logging.WARNING, format="%(name)s: %(message)s",
            stream=sys.stderr,
        )
        logging.getLogger("grape").setLevel(logging.DEBUG)
    elif not args.quiet:
        logging.basicConfig(
            level=logging.INFO, format="%(name)s: %(message)s",
            stream=sys.stderr,
        )
    sep = args.keyword_separator
    keywords = parse_keywords(args.keywords, sep) if args.keywords else []
    exclude_keywords: list[str] = []
    if args.exclude:
        exclude_keywords = parse_keywords(args.exclude, sep)
    prompt_templates = _parse_prompt_templates(parser, args.ensemble_prompts)
    model_name, pretrained = _validate_model_arg(parser, args.model)
    score_keywords = keywords + exclude_keywords
    like_paths = args.like

    if not keywords and not like_paths:
        parser.error("at least one of -k/--keywords or --like is required")

    # Open cache before scanning so find_images can skip files
    # already known not to be images.
    cache_cm: AbstractContextManager[EmbeddingCache | None]
    if not args.no_cache:
        import sqlite3

        from grape.cache import EmbeddingCache

        # Default cache: auto-mkdir, fail soft. User --cache: as-is, fail hard.
        is_default_cache = args.cache == parser.get_default("cache")
        try:
            if is_default_cache:
                os.makedirs(os.path.dirname(args.cache), exist_ok=True)
            cache_cm = closing(EmbeddingCache(args.cache))
        except sqlite3.OperationalError as e:
            if is_default_cache:
                print(f"grape: cache disabled: {args.cache}: {e}", file=sys.stderr)
                cache_cm = nullcontext()
            else:
                print(f"grape: cannot open cache {args.cache}: {e}", file=sys.stderr)
                sys.exit(1)
        except sqlite3.DatabaseError as e:
            if is_default_cache:
                print(f"grape: cache disabled: {args.cache}: {e}", file=sys.stderr)
                cache_cm = nullcontext()
            else:
                print(
                    f"grape: cache corrupted: {args.cache}\n"
                    f"grape: delete it and retry",
                    file=sys.stderr,
                )
                sys.exit(1)
        except OSError as e:
            # makedirs of the default cache parent failed.
            print(f"grape: cache disabled: {args.cache}: {e}", file=sys.stderr)
            cache_cm = nullcontext()
    else:
        cache_cm = nullcontext()

    with cache_cm as cache:
        _run_pipeline(
            score_keywords=score_keywords,
            keywords=keywords,
            exclude_keywords=exclude_keywords,
            like_paths=like_paths,
            model_name=model_name,
            pretrained=pretrained,
            prompt_templates=prompt_templates,
            cache=cache,
            quiet=args.quiet,
            path_args=_expand_stdin_paths(args.path),
            recursive=args.recursive,
            threshold=args.threshold,
            top=args.top,
            scores=args.scores,
            verbose=args.verbose,
            print0=args.print0,
            view=getattr(args, "view", False),
        )


