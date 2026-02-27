import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, nullcontext
from pathlib import Path

from grape.search import find_images, score_image, score_images


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


def parse_keywords(raw: str) -> list[str]:
    """Split a keyword string on commas. Strips whitespace from each keyword."""
    return [k.strip() for k in raw.split(",") if k.strip()]


def _format_results(results: list[dict], verbose: bool) -> str:
    """Format scored results for display."""
    lines = []
    for r in results:
        lines.append(f"{r['score']:.3f}  {r['path']}")
        if verbose:
            parts = [f"  {kw}: {s:.3f}" for kw, s in r["scores"].items()]
            lines.append("".join(parts))
    return "\n".join(lines)


def main():
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
    parser.add_argument(
        "--cache",
        metavar="PATH",
        default=None,
        help="cache file for embeddings"
             " (created if it doesn't exist)",
    )
    parser.add_argument(
        "--model",
        default="ViT-B-32/laion2b_s34b_b79k",
        help="open_clip model/pretrained tag"
             " (default: ViT-B-32/laion2b_s34b_b79k)",
    )

    args = parser.parse_args()
    keywords = parse_keywords(args.keywords)

    if not keywords:
        parser.error("no keywords provided")

    if "/" not in args.model:
        parser.error(
            f"invalid model '{args.model}':"
            " expected format model_name/pretrained"
            " (e.g. ViT-B-32/laion2b_s34b_b79k)"
        )
    model_name, pretrained = args.model.split("/", 1)

    # Open cache (if requested) before scanning so find_images can
    # skip files already known not to be images.
    if args.cache:
        from grape.cache import EmbeddingCache

        cache_cm = closing(EmbeddingCache(args.cache))
    else:
        cache_cm = nullcontext()

    with ThreadPoolExecutor(max_workers=1) as executor:
        # Start model loading so it overlaps with image scanning below.
        model_future = executor.submit(
            _load_model,
            model_name=model_name,
            pretrained=pretrained,
            quiet=args.quiet,
        )

        with cache_cm as cache:
            # Collect image paths from all positional PATH args
            images: list[Path] = []
            for p in args.path:
                target = Path(p)
                if target.is_file():
                    images.append(target)
                elif target.is_dir():
                    if not args.recursive:
                        print(
                            f"grape: {p}: Is a directory",
                            file=sys.stderr,
                        )
                        continue
                    images.extend(
                        find_images(str(target), recursive=True, cache=cache)
                    )
                else:
                    print(
                        f"grape: {p}: No such file or directory",
                        file=sys.stderr,
                    )
                    sys.exit(1)

            if not images:
                print("grape: no images found", file=sys.stderr)
                sys.exit(1)

            if not args.quiet:
                n = len(images)
                kw_str = ", ".join(keywords)
                print(f"{n} image{'s' * (n != 1)}, {kw_str}", file=sys.stderr)

            model = model_future.result()

            if len(images) == 1:
                results = [
                    score_image(model, images[0], keywords, cache=cache)
                ]
            else:
                results = score_images(
                    model, images, keywords,
                    quiet=args.quiet, cache=cache,
                )

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

    # Output
    show_scores = args.scores or args.verbose
    if args.count:
        print(len(results))
    elif show_scores:
        print(_format_results(results, verbose=args.verbose))
    else:
        for r in results:
            print(r["path"])
