import argparse
import importlib.util
import logging
import os
import shlex
import sys
import tempfile
import threading
from contextlib import closing, nullcontext
from dataclasses import dataclass
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import dask
from dask.threaded import get as _dask_threaded_get

from grape.search import (
    ScoredImage,
    iter_image_records,
)

if TYPE_CHECKING:
    from grape.cache import EmbeddingCache
    from grape.model import CLIPModel

log = logging.getLogger("grape")

DEFAULT_MODEL = "ViT-B-16/laion2b_s34b_b88k"

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
    body {
      margin: 12px;
    }
    .meta {
      font: 12px/1.3 sans-serif;
      word-break: break-all;
    }
    .path {
      margin: 0;
    }
    .scoreline {
      margin: 0 0 4px 0;
      color: #555;
      font-size: 11px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    img {
      display: block;
      max-width: 100%;
      width: auto;
      height: auto;
      margin-bottom: 10px;
    }
  </style>
</head>
<body>
  <p>{{ count }} image(s) for {{ keywords }}</p>
  {% for row in rows %}
  <p class="meta path">{{ row.path_text }}</p>
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
# Dask delayed building blocks
#
# Each @dask.delayed function is a node in the task graph, wired together
# in _run_pipeline(). A single dask.compute() call runs four branches
# concurrently on a thread pool:
#
#   _load_model (instant) ------+-- _encode_keywords ------+
#                                \-- _encode_like_images --+ |
#                                                    _combine +
#   _resolve_and_index_cache --+                             |
#   _scan_files ------+--------+-- _prepare                  |
#                                   \-- _score_all ----------+
#
# After dask.compute(), _filter_and_sort and _emit run on the main
# thread (pywebview requires it, stdout is cleaner without interleaving).
#
# Performance notes:
# - The scheduler is passed as a function (dask.threaded.get) not a string
#   ("threads") to avoid a ~0.2s lazy import of dask.distributed that
#   happens inside dask's get_scheduler() lookup.
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
    from GIL contention.  Thread-safe for concurrent dask tasks.
    """

    def __init__(
        self, model_name: str, pretrained: str, quiet: bool,
    ) -> None:
        self._model_name = model_name
        self._pretrained = pretrained
        self._quiet = quiet
        self._model: Any = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> Any:
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

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ensure_loaded(), name)


@dask.delayed
def _load_model(
    model_name: str,
    pretrained: str,
    quiet: bool,
) -> Any:
    """Return a lazy model proxy -- no import until first access.

    When everything is cached, no downstream node touches the model
    and torch/open_clip are never imported.  scan_files runs without
    GIL contention.
    """
    return _LazyModel(model_name, pretrained, quiet)


@dask.delayed
def _encode_keywords(
    model: "CLIPModel",
    score_keywords: list[str],
    prompt_templates: list[str],
    cache_context: tuple[str | None, dict[tuple[str, str], Any] | None],
    cache: "EmbeddingCache | None",
) -> Any:
    """Encode keyword prompts into text embeddings.

    Caches at the prompt level (the actual text sent to the model), not
    at the keyword level.  This means "a photo of a dog" is cached once
    and reused regardless of which template set generated it.

    With ensembling, each keyword produces N prompts.  We cache each
    prompt individually, then average + renormalize per keyword.
    """
    from typing import cast

    import numpy as np

    model_id = cache_context[0] if cache_context else None

    # Build all prompts: one per (keyword, template) pair.
    all_prompts = [
        template.format(kw)
        for kw in score_keywords
        for template in prompt_templates
    ]

    # Check cache for each prompt.
    cached_prompts: dict[str, Any] = {}
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
        assert model_id is None or real_model_id == model_id, (
            f"model_id mismatch: cached {model_id!r},"
            f" resolved {real_model_id!r}"
        )
        model_id = real_model_id

        new_pairs: list[tuple[str, Any]] = []
        for i, prompt in enumerate(uncached_prompts):
            emb = fresh[i : i + 1]
            cached_prompts[prompt] = emb
            new_pairs.append((prompt, emb))
        if cache is not None and model_id is not None:
            cache.put_text_embeddings(model_id, new_pairs)

    # Reassemble per-keyword embeddings: average across templates,
    # then L2-normalize (same logic as _encode_keyword_embeddings).
    n_templates = len(prompt_templates)
    keyword_embs: list[Any] = []
    for kw in score_keywords:
        prompts = [t.format(kw) for t in prompt_templates]
        embs = np.vstack([cached_prompts[p] for p in prompts])
        if n_templates == 1:
            keyword_embs.append(embs)
        else:
            merged = embs.mean(axis=0, keepdims=True)
            norm = np.linalg.norm(merged)
            merged = merged / max(norm, 1e-12)
            keyword_embs.append(cast(np.ndarray, merged.astype(np.float32)))

    return np.vstack(keyword_embs)



@dask.delayed
def _encode_like_images(
    model: "CLIPModel",
    like_paths: list[str],
    cache_context: tuple[str | None, dict[tuple[str, str], Any] | None],
    cache: "EmbeddingCache | None",
) -> Any:
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
            cached = cache.get(Path(p), model_id)
        if cached is not None:
            embeddings.append(cached)
        else:
            embeddings.append(model.encode_image(p))
    return np.vstack(embeddings)


@dask.delayed
def _combine_query_embeddings(
    text_emb: Any,
    like_emb: Any,
) -> Any:
    """Stack text and --like image query embeddings into one matrix."""
    import numpy as np
    parts = [e for e in (text_emb, like_emb) if e is not None]
    return np.vstack(parts)


@dask.delayed
def _resolve_and_index_cache(
    model_name: str,
    pretrained: str,
    cache: "EmbeddingCache | None",
) -> tuple[str | None, dict[tuple[str, str], Any] | None]:
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


@dask.delayed
def _scan_files(
    path_args: list[str],
    recursive: bool,
    cache: "EmbeddingCache | None",
) -> "tuple[list[_ScannedImage], _ScanDone]":
    """Discover image files from CLI paths. Independent of model loading."""
    if cache is not None:
        image_hits = cache.image_hit_index()
        not_image_hits = cache.not_image_index()
    else:
        image_hits = None
        not_image_hits = None
    items: list[_ScannedImage] = []
    error_message: str | None = None

    for p in path_args:
        target = Path(p)
        if target.is_file():
            path_key = os.path.realpath(target)
            st = os.stat(path_key)
            file_stat = (
                f"[{st.st_size}, {st.st_mtime_ns},"
                f" {st.st_ino}, {st.st_dev}, {st.st_ctime_ns}]"
            )
            items.append(_ScannedImage(
                path=target, path_key=path_key, file_stat=file_stat,
            ))
            continue
        if target.is_dir():
            if not recursive:
                print(f"grape: {p}: Is a directory", file=sys.stderr)
                continue
            for record in iter_image_records(
                str(target),
                recursive=True,
                cache=cache,
                image_hits=image_hits,
                not_image_hits=not_image_hits,
            ):
                items.append(_ScannedImage(
                    path=record.path,
                    path_key=record.path_key,
                    file_stat=record.file_stat,
                ))
            continue
        error_message = f"grape: {p}: No such file or directory"
        break

    log.debug("scan_files: %d images found", len(items))
    return items, _ScanDone(
        image_count=len(items), error_message=error_message,
    )


@dask.delayed
def _prepare_cached_embeddings(
    scan_result: "tuple[list[_ScannedImage], _ScanDone]",
    cache_context: tuple[str | None, dict[tuple[str, str], Any] | None],
) -> "tuple[Any, list[_ScannedImage], list[_ScannedImage], _ScanDone]":
    """Split scanned images into cached/uncached and vstack cached vectors.

    Runs as soon as scanning and cache indexing finish -- does not wait for
    model loading or text encoding, so the ~23ms vstack overlaps with those.
    Returns (image_emb_matrix | None, cached_items, uncached_items, scan_done).
    """
    import numpy as np

    _model_id, cached_index = cache_context
    items, scan_done = scan_result

    cached_items: list[_ScannedImage] = []
    cached_vectors: list[Any] = []
    uncached_items: list[_ScannedImage] = []

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


@dask.delayed
def _score_all(
    prepared: "tuple[Any, list[_ScannedImage], list[_ScannedImage], _ScanDone]",
    model: "CLIPModel",
    score_keywords: list[str],
    like_paths: list[str],
    text_emb: Any,
    cache: "EmbeddingCache | None",
    quiet: bool,
) -> "tuple[list[ScoredImage], _ScanDone]":
    """Score all scanned images against text embeddings.

    ``score_keywords`` are text keyword labels (include + exclude).
    ``like_paths`` are --like image paths.  Keeping them separate avoids
    score-dict key collisions when basenames repeat or match a keyword.
    """
    from grape.search import _get_embedding

    image_emb, cached_items, uncached_items, scan_done = prepared
    n_text = len(score_keywords)

    def _make_result(path: Path, sims: Any) -> ScoredImage:
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
    for item in uncached_items:
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
    score_result: "tuple[list[ScoredImage], _ScanDone]",
    keywords: list[str],
    exclude_keywords: list[str],
    like_names: list[str],
    threshold: float | None,
    top: int | None,
    quiet: bool,
) -> list[ScoredImage]:
    """Apply excludes, sort, threshold, top-N, and validate scan status.

    Not a dask node -- runs on the main thread after dask.compute().
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

    if exclude_keywords:
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

    Not a dask node -- runs after dask.compute() returns, so pywebview
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
        print(shlex.quote(str(r.path)))
    return len(results)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class _ScanDone:
    image_count: int
    error_message: str | None = None


@dataclass
class _ScannedImage:
    path: Path
    path_key: str
    file_stat: str


# ---------------------------------------------------------------------------
# Pure helpers (not delayed -- used inside delayed tasks or at parse time)
# ---------------------------------------------------------------------------

def parse_keywords(raw: str) -> list[str]:
    """Split a keyword string on commas. Strips whitespace from each keyword."""
    return [k.strip() for k in raw.split(",") if k.strip()]


def parse_prompt_templates(raw: str) -> list[str]:
    """Split comma-separated prompt templates."""
    return [t.strip() for t in raw.split(",") if t.strip()]


def _format_results(results: list[ScoredImage], verbose: bool) -> str:
    """Format scored results for display."""
    lines = []
    for r in results:
        lines.append(f"{r.score:.3f}  {shlex.quote(str(r.path))}")
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


_html_template_cache: Any = None


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
        src_path = r.path
        if not src_path.is_absolute():
            src_path = src_path.absolute()
        all_scores = list(r.scores.items()) + [
            (f"like:{Path(lp).name}", s) for lp, s in r.like_scores
        ]
        breakdown = " \N{MIDDLE DOT} ".join(
            f"{kw}: {score:.3f}"
            for kw, score in all_scores
        )
        score_line = f"score: {r.score:.3f}"
        if breakdown:
            score_line = f"{score_line} \N{MIDDLE DOT} {breakdown}"
        rows.append(
            {
                "image_src": src_path.as_uri(),
                "path_text": str(r.path),
                "score_line": score_line,
            }
        )
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
    default_prompt_templates = ", ".join(DEFAULT_PROMPT_ENSEMBLE)
    parser = argparse.ArgumentParser(
        prog="grape",
        description="Find images matching keywords using CLIP.",
        epilog="examples:\n"
               "  grape -k sunset photo.jpg\n"
               "  grape -k 'cat,dog' *.jpg\n"
               "  grape -R -k 'golden retriever' ~/Pictures\n"
               "  grape --like ref.jpg -R ~/Pictures\n"
               "  grape -k dog --like ref.jpg -R ~/Pictures\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_pkg_version('grape')}",
    )

    # -- Positional ---------------------------------------------------------
    parser.add_argument(
        "path",
        nargs="+",
        metavar="PATH",
        help="image file(s) or directory (with -R)",
    )

    # -- Query: what to search for ------------------------------------------
    parser.add_argument(
        "-k", "--keywords",
        default=None,
        metavar="KEYWORDS",
        help="comma-separated keywords to match"
             " (e.g. 'cat,dog' or 'golden retriever,sunset')",
    )
    parser.add_argument(
        "-x", "--exclude",
        dest="exclude",
        default=None,
        metavar="KEYWORDS",
        help="comma-separated anti-match keywords to penalize"
             " (score = include_mean - exclude_mean)",
    )
    parser.add_argument(
        "--like",
        action="append",
        default=[],
        metavar="IMAGE",
        help="reference image to find similar images"
             " (repeatable; combined with keyword queries)",
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
        help="only show results >= SCORE"
             " (cosine similarity, not a probability;"
             " even strong matches rarely exceed 0.35;"
             " scores are not comparable across models;"
             " use -s to see scores and pick a threshold)",
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
        help="show per-keyword score breakdown"
             " (implies -s)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="suppress progress and status messages"
             " on stderr",
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "-print0",
        action="store_true",
        help="print matching paths separated by NUL"
             " bytes (raw paths, no shell quoting)",
    )
    _has_view_deps = (
        importlib.util.find_spec("webview") is not None
        and importlib.util.find_spec("jinja2") is not None
    )
    if _has_view_deps:
        output_group.add_argument(
            "--view",
            action="store_true",
            help="open results in a native webview window"
                 " using simple HTML with <img> tags",
        )

    # -- Configuration -------------------------------------------------------
    parser.add_argument(
        "--cache",
        metavar="PATH",
        default=None,
        help="cache file for embeddings"
             " (created if it doesn't exist)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="open_clip model/pretrained tag"
             f" (default: {DEFAULT_MODEL});"
             " see https://github.com/mlfoundations/open_clip"
             " for available models",
    )
    parser.add_argument(
        "--ensemble-prompts",
        nargs="?",
        const=",".join(DEFAULT_PROMPT_ENSEMBLE),
        default=",".join(DEFAULT_PROMPT_ENSEMBLE),
        metavar="TEMPLATES",
        help="comma-separated prompt templates for keyword ensembling"
             f" (default: {default_prompt_templates})",
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
    """Build and execute the dask task graph for the full CLI pipeline."""
    # --- Build the task graph ---
    # Three independent roots run concurrently:
    #   1. _load_model (instant -- creates lazy proxy)
    #   2. _resolve_and_index_cache (fast SQLite lookup on warm cache)
    #   3. _scan_files (IO-bound directory walk)
    # When the cache is warm, torch/open_clip are never imported and
    # _scan_files runs uncontested by the GIL.

    # Root 1: lazy model proxy (no import, no weight loading)
    model = _load_model(model_name, pretrained, quiet)

    # Root 2: model_id resolution + cache index (SQLite only when warm)
    cache_context = _resolve_and_index_cache(
        model_name, pretrained, cache,
    )

    # Text keyword embeddings (None when no text keywords).
    text_emb = (
        _encode_keywords(
            model, score_keywords, prompt_templates, cache_context, cache,
        )
        if score_keywords else None
    )
    # --like image embeddings (None when no --like).
    # Depends on cache_context so we can reuse cached embeddings,
    # ensuring --like self-matches produce exactly 1.0 similarity.
    like_emb = (
        _encode_like_images(model, like_paths, cache_context, cache)
        if like_paths else None
    )
    # Combined query matrix: [text keywords..., like embeddings...].
    query_emb = _combine_query_embeddings(text_emb, like_emb)
    like_names = [Path(p).name for p in like_paths]

    # Branch 3: file scanning (IO-bound, fully independent root)
    scan_result = _scan_files(path_args, recursive, cache)

    # Runs as soon as scan + cache index are ready (no model/text dependency).
    # The vstack of cached embeddings (~23ms at 10k images) overlaps with
    # model loading and text encoding.
    prepared = _prepare_cached_embeddings(scan_result, cache_context)

    # Convergence: scoring needs prepared embeddings + query embeddings + model
    score_result = _score_all(
        prepared, model, score_keywords, like_paths, query_emb,
        cache, quiet,
    )

    # Single compute call -- dask resolves the whole graph.
    # Pass the get function directly to avoid dask.distributed import (~0.2s).
    (score_result_value,) = dask.compute(
        score_result, scheduler=_dask_threaded_get,
    )

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
    parser = _build_parser()

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG, format="%(name)s: %(message)s",
            stream=sys.stderr,
        )
    elif not args.quiet:
        logging.basicConfig(
            level=logging.INFO, format="%(name)s: %(message)s",
            stream=sys.stderr,
        )
    keywords = parse_keywords(args.keywords) if args.keywords else []
    exclude_keywords: list[str] = []
    if args.exclude:
        exclude_keywords = parse_keywords(args.exclude)
    prompt_templates = _parse_prompt_templates(parser, args.ensemble_prompts)
    model_name, pretrained = _validate_model_arg(parser, args.model)
    score_keywords = keywords + exclude_keywords
    like_paths = args.like

    if not keywords and not like_paths:
        parser.error("at least one of -k/--keywords or --like is required")

    # Open cache (if requested) before scanning so find_images can
    # skip files already known not to be images.
    cache_cm: Any
    if args.cache:
        from grape.cache import EmbeddingCache

        cache_cm = closing(EmbeddingCache(args.cache))
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
            path_args=args.path,
            recursive=args.recursive,
            threshold=args.threshold,
            top=args.top,
            scores=args.scores,
            verbose=args.verbose,
            print0=args.print0,
            view=getattr(args, "view", False),
        )


