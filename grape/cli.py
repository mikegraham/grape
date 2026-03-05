import argparse
import queue
import shlex
import sys
import tempfile
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, nullcontext
from pathlib import Path

DEFAULT_PROMPT_ENSEMBLE = [
    "a photo of a {}",
    "a photo of the {}",
    "a close-up photo of a {}",
    "a photo of one {}",
    "a blurry photo of a {}",
]

_HTML_TEMPLATE_TEXT = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>grape results</title>
  <style>
    img {
      display: block;
      max-width: 100%;
      width: auto;
      height: auto;
    }
  </style>
</head>
<body>
  <p>{{ count }} image(s) for {{ keywords }}</p>
  {% for row in rows %}
  <p>{{ row.score }} {{ row.path_text }}</p>
  <img src="{{ row.image_src }}" alt="{{ row.path_text }}">
  {% if row.breakdown %}
  <p>{{ row.breakdown }}</p>
  {% endif %}
  {% endfor %}
</body>
</html>
""".strip()
_html_template = None


def _get_html_template():
    """Lazily compile the HTML template."""
    global _html_template
    if _html_template is None:
        from jinja2 import Template

        _html_template = Template(_HTML_TEMPLATE_TEXT, autoescape=True)
    return _html_template


def _get_webview():
    """Lazily import pywebview for --view mode only."""
    import webview

    return webview


def _preload_view_modules() -> None:
    """Warm view-only imports so they are off the tail latency path."""
    _get_html_template()
    _get_webview()


def _load_model(
    model_name: str,
    pretrained: str,
    quiet: bool,
):
    """Load and return a CLIPModel instance."""
    from grape.model import CLIPModel

    return CLIPModel(
        model_name=model_name,
        pretrained=pretrained,
        quiet=quiet,
    )


def find_images(*args, **kwargs):
    """Lazily import and call grape.search.find_images."""
    from grape.search import find_images as _find_images

    return _find_images(*args, **kwargs)


def score_image(*args, **kwargs):
    """Lazily import and call grape.search.score_image."""
    from grape.search import score_image as _score_image

    return _score_image(*args, **kwargs)


def score_images(*args, **kwargs):
    """Lazily import and call grape.search.score_images."""
    from grape.search import score_images as _score_images

    return _score_images(*args, **kwargs)


def iter_images(*args, **kwargs):
    """Lazily import and call grape.search.iter_images."""
    from grape.search import iter_images as _iter_images

    return _iter_images(*args, **kwargs)


def iter_image_records(*args, **kwargs):
    """Lazily import and call grape.search.iter_image_records."""
    from grape.search import iter_image_records as _iter_image_records

    return _iter_image_records(*args, **kwargs)


def encode_keywords(*args, **kwargs):
    """Lazily import and call grape.search.encode_keywords."""
    from grape.search import encode_keywords as _encode_keywords

    return _encode_keywords(*args, **kwargs)


def score_image_with_text_embeddings(*args, **kwargs):
    """Lazily import and call grape.search.score_image_with_text_embeddings."""
    from grape.search import score_image_with_text_embeddings as _score

    return _score(*args, **kwargs)


def parse_keywords(raw: str) -> list[str]:
    """Split a keyword string on commas. Strips whitespace from each keyword."""
    return [k.strip() for k in raw.split(",") if k.strip()]


def parse_prompt_templates(raw: str) -> list[str]:
    """Split comma-separated prompt templates."""
    return [t.strip() for t in raw.split(",") if t.strip()]


def _format_results(results: list[dict], verbose: bool) -> str:
    """Format scored results for display."""
    lines = []
    for r in results:
        lines.append(f"{r['score']:.3f}  {shlex.quote(str(r['path']))}")
        if verbose:
            parts = [f"  {kw}: {s:.3f}" for kw, s in r["scores"].items()]
            lines.append("".join(parts))
    return "\n".join(lines)


def _apply_excluded_keywords(
    results: list[dict],
    include_keywords: list[str],
    exclude_keywords: list[str],
) -> None:
    """Adjust result scores using include-vs-exclude keyword means."""
    for result in results:
        raw_scores: dict[str, float] = result["scores"]
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
        result["score"] = float(include_mean - exclude_mean)
        result["include_score"] = float(include_mean)
        result["exclude_score"] = float(exclude_mean)

        # Keep verbose output readable by labeling excluded keywords.
        labeled_scores: dict[str, float] = {}
        for kw in include_keywords:
            labeled_scores[kw] = raw_scores[kw]
        for kw, component in zip(exclude_keywords, exclude_components):
            labeled_scores[f"not:{kw}"] = component
        result["scores"] = labeled_scores


def _format_html(
    results: list[dict],
    keywords: list[str],
    verbose: bool,
) -> str:
    """Format results as simple HTML with file-backed <img> tags."""
    rows: list[dict[str, str]] = []
    for r in results:
        image_path = Path(r["path"])
        abs_path = image_path.resolve()
        breakdown = ""
        if verbose:
            breakdown = " ".join(
                f"{kw}: {score:.3f}"
                for kw, score in r["scores"].items()
            )
        rows.append(
            {
                "image_src": abs_path.as_uri(),
                "path_text": str(abs_path),
                "score": f"{r['score']:.3f}",
                "breakdown": breakdown,
            }
        )
    template = _get_html_template()
    return template.render(
        count=len(results),
        keywords=", ".join(keywords),
        rows=rows,
    )


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
        "-x", "--exclude", "--anti",
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
    output_group.add_argument(
        "-c", "--count",
        action="store_true",
        help="print only the count of matching images"
             " (like grep -c)",
    )
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
    return model.split("/", 1)


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


def _collect_images(
    path_args: list[str],
    recursive: bool,
    cache,
) -> list[Path]:
    """Collect image paths from CLI path arguments."""
    images: list[Path] = []
    for p in path_args:
        target = Path(p)
        if target.is_file():
            images.append(target)
            continue
        if target.is_dir():
            if not recursive:
                print(f"grape: {p}: Is a directory", file=sys.stderr)
                continue
            images.extend(find_images(str(target), recursive=True, cache=cache))
            continue
        print(f"grape: {p}: No such file or directory", file=sys.stderr)
        sys.exit(1)
    return images


@dataclass
class _ScanDone:
    image_count: int
    error_message: str | None = None


@dataclass
class _ScannedImage:
    path: Path
    path_key: str | None = None
    file_stat: str | None = None


_SCORE_BATCH_SIZE = 512
_SCAN_QUEUE_BATCH_SIZE = _SCORE_BATCH_SIZE


def _build_scan_indexes(cache) -> tuple[set[tuple[str, str]] | None, set[tuple[str, str]] | None]:
    """Fetch cache indexes once for fast, DB-free scan membership checks."""
    if cache is None:
        return None, None
    return cache.image_hit_index(), cache.not_image_index()


def _scan_paths_worker(
    path_args: list[str],
    recursive: bool,
    image_hits: set[tuple[str, str]] | None,
    not_image_hits: set[tuple[str, str]] | None,
    out_queue: queue.Queue,
) -> None:
    """Scan input paths and stream image batches into *out_queue*."""
    image_count = 0
    error_message: str | None = None
    batch: list[_ScannedImage] = []

    def _flush_batch() -> None:
        nonlocal batch
        if not batch:
            return
        out_queue.put(batch)
        batch = []

    try:
        for p in path_args:
            target = Path(p)
            if target.is_file():
                batch.append(_ScannedImage(path=target))
                image_count += 1
                if len(batch) >= _SCAN_QUEUE_BATCH_SIZE:
                    _flush_batch()
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
                    batch.append(
                        _ScannedImage(
                            path=record.path,
                            path_key=record.path_key,
                            file_stat=record.file_stat,
                        )
                    )
                    image_count += 1
                    if len(batch) >= _SCAN_QUEUE_BATCH_SIZE:
                        _flush_batch()
                continue
            error_message = f"grape: {p}: No such file or directory"
            break
    finally:
        _flush_batch()
        out_queue.put(_ScanDone(image_count=image_count, error_message=error_message))


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
    return "; ".join(query_parts) if query_parts else "(no keywords)"


def _build_scored_result(
    path: Path,
    keywords: list[str],
    sims,
) -> dict:
    """Build a score dict from similarity values."""
    return {
        "path": path,
        "scores": {kw: float(s) for kw, s in zip(keywords, sims)},
        "score": float(sims.mean()),
        "min_score": float(sims.min()),
    }


def _score_batch(
    *,
    batch: list[_ScannedImage],
    model,
    score_keywords: list[str],
    text_emb,
    cache,
    model_id: str | None,
    quiet: bool,
) -> list[dict]:
    """Score a batch of scanned images with batched cache lookups."""
    if not batch:
        return []

    cached_vectors: dict[str, object] = {}
    if cache is not None and model_id is not None:
        path_stats = {
            item.path_key: item.file_stat
            for item in batch
            if item.path_key is not None and item.file_stat is not None
        }
        if path_stats:
            cached_vectors = cache.get_many_for_paths(model_id, path_stats)

    cached_items: list[_ScannedImage] = []
    uncached_items: list[_ScannedImage] = []
    for item in batch:
        if item.path_key is not None and item.path_key in cached_vectors:
            cached_items.append(item)
        else:
            uncached_items.append(item)

    results: list[dict] = []
    if cached_items:
        import numpy as np

        image_emb = np.vstack([cached_vectors[item.path_key] for item in cached_items])
        sims_matrix = image_emb @ text_emb.T
        for item, sims in zip(cached_items, sims_matrix):
            results.append(_build_scored_result(item.path, score_keywords, sims))

    for item in uncached_items:
        try:
            results.append(
                score_image_with_text_embeddings(
                    model,
                    item.path,
                    score_keywords,
                    text_emb,
                    cache=cache,
                )
            )
        except OSError as e:
            if e.errno is not None:
                raise
            if not quiet:
                print(f"  skipping {item.path}: {e}", file=sys.stderr)
    return results


def _score_or_stub_results(
    images: list[Path] | None,
    score_keywords: list[str],
    model_future,
    cache,
    prompt_templates: list[str],
    quiet: bool,
    path_args: list[str],
    recursive: bool,
    image_hits: set[tuple[str, str]] | None,
    not_image_hits: set[tuple[str, str]] | None,
    executor: ThreadPoolExecutor,
) -> list[dict]:
    """Score images when keywords are present; otherwise emit neutral rows."""
    if not score_keywords:
        assert images is not None
        return [
            {
                "path": path,
                "scores": {},
                "score": 0.0,
                "min_score": 0.0,
            }
            for path in images
        ]

    path_queue: queue.Queue = queue.Queue()
    scan_future = executor.submit(
        _scan_paths_worker,
        path_args,
        recursive,
        image_hits,
        not_image_hits,
        path_queue,
    )

    model = model_future.result()
    text_emb = encode_keywords(
        model,
        score_keywords,
        prompt_templates=prompt_templates,
    )
    model_id = model.model_id() if cache is not None else None
    results: list[dict] = []
    image_count = 0
    scan_done: _ScanDone | None = None
    while scan_done is None:
        item = path_queue.get()
        if isinstance(item, _ScanDone):
            scan_done = item
            continue

        assert isinstance(item, list)
        image_count += len(item)
        results.extend(
            _score_batch(
                batch=item,
                model=model,
                score_keywords=score_keywords,
                text_emb=text_emb,
                cache=cache,
                model_id=model_id,
                quiet=quiet,
            )
        )

    scan_future.result()
    if scan_done.error_message:
        print(scan_done.error_message, file=sys.stderr)
        sys.exit(1)
    if image_count == 0:
        print("grape: no images found", file=sys.stderr)
        sys.exit(1)
    return results


def _emit_results(
    args: argparse.Namespace,
    results: list[dict],
    keywords: list[str],
    exclude_keywords: list[str],
    view_prep_future,
) -> None:
    """Print or render results based on output mode flags."""
    show_scores = args.scores or args.verbose
    if args.view:
        if view_prep_future is not None:
            view_prep_future.result()
        display_keywords = keywords + [f"not:{kw}" for kw in exclude_keywords]
        html_doc = _format_html(results, display_keywords, verbose=args.verbose)
        _show_in_webview(html_doc)
        return
    if args.count:
        print(len(results))
        return
    if show_scores:
        print(_format_results(results, verbose=args.verbose))
        return
    if args.print0:
        for r in results:
            sys.stdout.write(f"{r['path']}\0")
        sys.stdout.flush()
        return
    for r in results:
        print(shlex.quote(str(r["path"])))


def main():
    parser = _build_parser()

    args = parser.parse_args()
    keywords = parse_keywords(args.keywords)
    exclude_keywords = parse_keywords(args.exclude) if args.exclude else []
    prompt_templates = _parse_prompt_templates(parser, args.ensemble_prompts)
    model_name, pretrained = _validate_model_arg(parser, args.model)
    score_keywords = keywords + exclude_keywords

    # Open cache (if requested) before scanning so find_images can
    # skip files already known not to be images.
    if args.cache:
        from grape.cache import EmbeddingCache

        cache_cm = closing(EmbeddingCache(args.cache))
    else:
        cache_cm = nullcontext()

    if score_keywords and args.view:
        max_workers = 3
    elif score_keywords:
        max_workers = 2
    else:
        max_workers = 1
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Start model loading so it overlaps with image scanning below.
        model_future = (
            executor.submit(
                _load_model,
                model_name=model_name,
                pretrained=pretrained,
                quiet=args.quiet,
            )
            if score_keywords
            else None
        )
        if args.view:
            view_prep_future = executor.submit(_preload_view_modules)
        else:
            view_prep_future = None

        with cache_cm as cache:
            image_hits, not_image_hits = _build_scan_indexes(cache)
            if score_keywords:
                images = None
            else:
                images = _collect_images(args.path, args.recursive, cache)
                if not images:
                    print("grape: no images found", file=sys.stderr)
                    sys.exit(1)

            results = _score_or_stub_results(
                images=images,
                score_keywords=score_keywords,
                model_future=model_future,
                cache=cache,
                prompt_templates=prompt_templates,
                quiet=args.quiet,
                path_args=args.path,
                recursive=args.recursive,
                image_hits=image_hits,
                not_image_hits=not_image_hits,
                executor=executor,
            )
            if not args.quiet:
                n = len(results)
                query_text = _format_query_summary(keywords, exclude_keywords)
                print(
                    f"{n} image{'s' * (n != 1)}, {query_text}",
                    file=sys.stderr,
                )

    if exclude_keywords:
        _apply_excluded_keywords(results, keywords, exclude_keywords)

    # Sort by score descending (best matches first)
    results.sort(key=lambda r: r["score"], reverse=True)

    # Filter
    if args.threshold is not None:
        results = [
            r for r in results if r["score"] >= args.threshold
        ]

    if args.top is not None:
        results = results[: args.top]

    if not results:
        if args.count:
            print(0)
        else:
            print(
                "grape: no images above threshold",
                file=sys.stderr,
            )
        sys.exit(0)

    _emit_results(
        args=args,
        results=results,
        keywords=keywords,
        exclude_keywords=exclude_keywords,
        view_prep_future=view_prep_future,
    )
