import argparse
import os
import shlex
import sys
import tempfile
from contextlib import closing, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import dask
from dask.threaded import get as _dask_threaded_get

from grape.search import (
    ScoredImage,
    encode_keywords,
    iter_image_records,
    score_image_with_text_embeddings,
)

if TYPE_CHECKING:
    from grape.cache import EmbeddingCache
    from grape.model import CLIPModel

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
# in _run_pipeline(). A single dask.compute() call runs three branches
# concurrently on a thread pool:
#
#   _import_model_module --+-- _load_model -- _encode_keywords --+
#                          \-- _resolve_and_index_cache --+      |
#   _scan_files --------------------------------------+-- _prepare
#       \-- _score_all -- _filter_and_sort -- _emit
#
# Performance notes:
# - The scheduler is passed as a function (dask.threaded.get) not a string
#   ("threads") to avoid a ~0.2s lazy import of dask.distributed that
#   happens inside dask's get_scheduler() lookup.
# - _import_model_module eagerly imports open_clip (~1s) so that cost is
#   paid concurrently with file scanning, rather than inside _load_model.
# - _resolve_and_index_cache depends on _import_model_module (not a root
#   node) because open_clip's import uses global state that is not
#   thread-safe -- it must finish before other threads call into it.
# ---------------------------------------------------------------------------

@dask.delayed
def _import_model_module(model_name: str, pretrained: str) -> Any:
    """Import grape.model and open_clip (pulls in torch -- slow).

    Eagerly triggers the open_clip import so that the ~1s cost is paid
    here (concurrent with file scanning) rather than inside _load_model.

    Also kicks off a background thread to pre-read the safetensors
    weights from disk (see preload_weights docstring in model.py).
    By the time _load_model runs, the state dict is already in memory
    and we can use the fast meta-device path (~30ms vs ~800ms).
    """
    import grape.model
    grape.model.preload_weights(model_name, pretrained)
    return grape.model


@dask.delayed
def _load_model(
    model_module: Any,
    model_name: str,
    pretrained: str,
    quiet: bool,
) -> Any:
    """Load CLIP model weights."""
    return model_module.CLIPModel(
        model_name=model_name,
        pretrained=pretrained,
        quiet=quiet,
    )


@dask.delayed
def _encode_keywords(
    model: "CLIPModel",
    score_keywords: list[str],
    prompt_templates: list[str],
) -> Any:
    """Encode keyword prompts into text embeddings."""
    return encode_keywords(
        model,
        score_keywords,
        prompt_templates=prompt_templates,
    )



@dask.delayed
def _resolve_and_index_cache(
    model_module: Any,
    model_name: str,
    pretrained: str,
    cache: "EmbeddingCache | None",
) -> tuple[str | None, dict[tuple[str, str], Any] | None]:
    """Resolve model_id and materialize the cache index.

    Depends on model_module to ensure open_clip is already imported --
    open_clip's import machinery is not thread-safe, so we must not
    race with _load_model which also uses it.  Once open_clip is
    imported, resolve_model_id only reads config + probes the HF
    filesystem cache (no weight loading), so this still runs
    concurrently with _load_model.
    """
    if cache is None:
        return None, None
    model_id = model_module.resolve_model_id(model_name, pretrained)
    cached_index = cache.embedding_index_for_model(model_id)
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
                cache=None,
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

    image_emb = np.vstack(cached_vectors) if cached_vectors else None
    return image_emb, cached_items, uncached_items, scan_done


@dask.delayed
def _score_all(
    prepared: "tuple[Any, list[_ScannedImage], list[_ScannedImage], _ScanDone]",
    model: "CLIPModel",
    score_keywords: list[str],
    text_emb: Any,
    cache: "EmbeddingCache | None",
    quiet: bool,
) -> "tuple[list[ScoredImage], _ScanDone]":
    """Score all scanned images against text embeddings."""
    image_emb, cached_items, uncached_items, scan_done = prepared

    results: list[ScoredImage] = []

    # Fast path: score cached items via a single matrix multiply.
    if image_emb is not None:
        sims_matrix = image_emb @ text_emb.T
        means = sims_matrix.mean(axis=1)
        for idx, item in enumerate(cached_items):
            sims = sims_matrix[idx]
            results.append(ScoredImage(
                path=item.path,
                scores={kw: float(s) for kw, s in zip(score_keywords, sims)},
                score=float(means[idx]),
            ))

    # Slow path: encode uncached images through the model one at a time.
    for item in uncached_items:
        try:
            results.append(score_image_with_text_embeddings(
                model, item.path, score_keywords, text_emb, cache=cache,
            ))
        except OSError as e:
            if e.errno is not None:
                raise
            if not quiet:
                print(f"  skipping {item.path}: {e}", file=sys.stderr)

    return results, scan_done


@dask.delayed
def _filter_and_sort(
    score_result: "tuple[list[ScoredImage], _ScanDone]",
    keywords: list[str],
    exclude_keywords: list[str],
    threshold: float | None,
    top: int | None,
    quiet: bool,
) -> list[ScoredImage]:
    """Apply excludes, sort, threshold, top-N, and validate scan status."""
    results, scan_done = score_result

    if scan_done.error_message:
        print(scan_done.error_message, file=sys.stderr)
        sys.exit(1)
    if scan_done.image_count == 0:
        print("grape: no images found", file=sys.stderr)
        sys.exit(1)

    if not quiet:
        n = len(results)
        query_text = _format_query_summary(keywords, exclude_keywords)
        print(
            f"{n} image{'s' * (n != 1)}, {query_text}",
            file=sys.stderr,
        )

    if exclude_keywords:
        _apply_excluded_keywords(results, keywords, exclude_keywords)

    results.sort(key=lambda r: r.score, reverse=True)

    if threshold is not None:
        results = [r for r in results if r.score >= threshold]
    if top is not None:
        results = results[:top]
    return results


@dask.delayed
def _emit(
    results: list[ScoredImage],
    keywords: list[str],
    exclude_keywords: list[str],
    scores: bool,
    verbose: bool,
    print0: bool,
    view: bool,
    quiet: bool,
) -> int:
    """Format and output results. Returns count of emitted rows."""
    if not results:
        # Always print -- this is a result status, not a progress message.
        print("grape: no images above threshold", file=sys.stderr)
        return 0

    show_scores = scores or verbose
    if view:
        display_keywords = keywords + [f"not:{kw}" for kw in exclude_keywords]
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
            lines.append("".join(parts))
    return "\n".join(lines)


def _apply_excluded_keywords(
    results: list[ScoredImage],
    include_keywords: list[str],
    exclude_keywords: list[str],
) -> None:
    """Adjust result scores using include-vs-exclude keyword means."""
    for result in results:
        raw_scores = result.scores
        if include_keywords:
            include_mean = sum(raw_scores[kw] for kw in include_keywords) / len(
                include_keywords
            )
        else:
            include_mean = 0.0

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
        breakdown = " \N{MIDDLE DOT} ".join(
            f"{kw}: {score:.3f}"
            for kw, score in r.scores.items()
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
) -> str:
    """Build status text shown before scoring."""
    query_parts: list[str] = []
    if keywords:
        query_parts.append(", ".join(keywords))
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
               "  grape sunset photo.jpg\n"
               "  grape 'cat,dog' *.jpg\n"
               "  grape -r 'golden retriever,tennis ball' ~/Pictures\n"
               "  grape -v -t 0.25 sunset photo.jpg\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "keywords",
        metavar="KEYWORDS",
        help="comma-separated keywords"
             " (e.g. 'cat,dog' or 'golden retriever,sunset')",
    )
    parser.add_argument(
        "path",
        nargs="+",
        metavar="PATH",
        help="image file(s) or directory (with -r)",
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        default=False,
        help="search directories recursively",
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=None,
        metavar="SCORE",
        help="only show results >= SCORE"
             " (0.0-1.0, higher is stricter)",
    )
    parser.add_argument(
        "-n", "--top",
        type=int,
        default=None,
        metavar="N",
        help="show only top N results",
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
        "-s", "--scores",
        action="store_true",
        help="show scores alongside paths",
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
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="show per-keyword score breakdown"
             " (implies -s)",
    )
    output_group = parser.add_mutually_exclusive_group()
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="suppress progress and status messages"
             " on stderr",
    )
    output_group.add_argument(
        "-print0",
        action="store_true",
        help="print matching paths separated by NUL"
             " bytes (raw paths, no shell quoting)",
    )
    output_group.add_argument(
        "--view",
        action="store_true",
        help="open results in a native webview window"
             " using simple HTML with <img> tags",
    )
    parser.add_argument(
        "--cache",
        metavar="PATH",
        default=None,
        help="cache file for embeddings"
             " (created if it doesn't exist)",
    )
    parser.add_argument(
        "--model",
        default="ViT-B-16/laion2b_s34b_b88k",
        help="open_clip model/pretrained tag"
             " (default: ViT-B-16/laion2b_s34b_b88k)",
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
    #   1. model import -> weight loading -> text encoding
    #   2. model import -> model_id resolution + cache index
    #   3. file scanning (builds scan indexes, then walks paths)
    # All converge at _score_all.

    # Branch 1: model module import -> weight loading -> text encoding
    model_module = _import_model_module(model_name, pretrained)
    model = _load_model(model_module, model_name, pretrained, quiet)
    text_emb = _encode_keywords(model, score_keywords, prompt_templates)

    # Branch 2: model module import -> model_id resolution + cache index
    cache_context = _resolve_and_index_cache(
        model_module, model_name, pretrained, cache,
    )

    # Branch 3: file scanning (IO-bound, fully independent root)
    scan_result = _scan_files(path_args, recursive, cache)

    # Runs as soon as scan + cache index are ready (no model/text dependency).
    # The vstack of cached embeddings (~23ms at 10k images) overlaps with
    # model loading and text encoding.
    prepared = _prepare_cached_embeddings(scan_result, cache_context)

    # Convergence: scoring needs prepared embeddings + text embeddings + model
    score_result = _score_all(
        prepared, model, score_keywords, text_emb,
        cache, quiet,
    )

    # Depends on score_result
    filtered = _filter_and_sort(
        score_result, keywords, exclude_keywords,
        threshold, top, quiet,
    )

    # Depends on filtered
    emitted_count = _emit(
        filtered, keywords, exclude_keywords,
        scores, verbose, print0, view, quiet,
    )

    # Single compute call -- dask resolves the whole graph.
    # Pass the get function directly to avoid dask.distributed import (~0.2s).
    dask.compute(emitted_count, scheduler=_dask_threaded_get)


def main() -> None:
    parser = _build_parser()

    args = parser.parse_args()
    keywords = parse_keywords(args.keywords)
    exclude_keywords: list[str] = []
    if args.exclude:
        exclude_keywords = parse_keywords(args.exclude)
    prompt_templates = _parse_prompt_templates(parser, args.ensemble_prompts)
    model_name, pretrained = _validate_model_arg(parser, args.model)
    score_keywords = keywords + exclude_keywords

    # Open cache (if requested) before scanning so find_images can
    # skip files already known not to be images.
    cache_cm: Any
    if args.cache:
        from grape.cache import EmbeddingCache

        cache_cm = closing(EmbeddingCache(args.cache))
    else:
        cache_cm = nullcontext()

    if not score_keywords:
        parser.error("at least one keyword is required")

    with cache_cm as cache:
        _run_pipeline(
            score_keywords=score_keywords,
            keywords=keywords,
            exclude_keywords=exclude_keywords,
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
            view=args.view,
        )


