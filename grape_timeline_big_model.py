"""Timeline profiler for the dask-based grape pipeline.

Prints a color-coded Gantt chart of all delayed nodes, grouped by thread.
Usage: .venv/bin/python grape_timeline_big_model.py tests/fixtures grape.db
"""
import argparse
import sys
import threading
import time
from functools import wraps

process_start = time.perf_counter()

import dask
import grape.cli as cli
from grape.cache import EmbeddingCache

import_done = time.perf_counter()

# ANSI color palette for the Gantt bars.
COLORS = [
    "\033[41m",   # red bg
    "\033[42m",   # green bg
    "\033[43m",   # yellow bg
    "\033[44m",   # blue bg
    "\033[45m",   # magenta bg
    "\033[46m",   # cyan bg
    "\033[47m",   # white bg (black text)
    "\033[100m",  # bright black bg
    "\033[101m",  # bright red bg
    "\033[103m",  # bright yellow bg
    "\033[104m",  # bright blue bg
]
RESET = "\033[0m"
BLACK_TEXT = "\033[30m"


def run_case(image_dir: str, cache_db: str, name: str) -> None:
    # (timestamp, label, start_or_end, thread_name)
    events: list[tuple[float, str, str, str]] = []

    def mark(label: str, phase: str) -> None:
        events.append((
            time.perf_counter(), label, phase,
            threading.current_thread().name,
        ))

    def wrap_delayed(label: str, delayed_fn):
        orig_func = delayed_fn.__wrapped__

        @dask.delayed
        @wraps(orig_func)
        def instrumented(*args, **kwargs):
            mark(label, "start")
            try:
                return orig_func(*args, **kwargs)
            finally:
                mark(label, "end")

        return instrumented

    delayed_nodes = [
        "_load_model", "_encode_keywords",
        "_encode_like_images", "_combine_query_embeddings",
        "_resolve_and_index_cache", "_scan_files",
        "_prepare_cached_embeddings", "_score_all",
    ]

    for fn_name in delayed_nodes:
        nice_name = fn_name.lstrip("_")
        setattr(cli, fn_name, wrap_delayed(
            nice_name, getattr(cli, fn_name),
        ))

    cli._show_in_webview = lambda html_doc: None

    argv = [
        "grape", "-q", "--view", "--cache", cache_db,
        "--model", "ViT-L-14/laion2b_s32b_b82k",
        "-k", "dog,puppy,pet,cat", "-R", image_dir,
    ]

    mark("main", "start")
    code = 0
    try:
        old_argv = sys.argv
        sys.argv = argv
        cli.main()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    finally:
        sys.argv = old_argv
        mark("main", "end")

    _print_timeline(events, process_start, import_done, name, code)


def _print_timeline(events, t0, t_import, name, code):
    """Print a text-based color-coded Gantt chart."""
    # Build spans: {label: (start, end, thread)}
    starts: dict[str, tuple[float, str]] = {}
    spans: list[tuple[str, float, float, str]] = []

    for ts, label, phase, thread in events:
        if phase == "start":
            starts[label] = (ts, thread)
        elif label in starts:
            s, th = starts.pop(label)
            spans.append((label, s - t0, ts - t0, th))

    if not spans:
        print("No events recorded.")
        return

    total = max(end for _, _, end, _ in spans)
    width = 72  # chart columns

    # Assign colors to labels
    label_colors = {}
    ci = 0
    for label, _, _, _ in sorted(spans, key=lambda s: s[1]):
        if label not in label_colors:
            label_colors[label] = COLORS[ci % len(COLORS)]
            ci += 1

    # Group by thread
    threads: dict[str, list[tuple[str, float, float]]] = {}
    for label, start, end, thread in spans:
        threads.setdefault(thread, []).append((label, start, end))

    # Print header
    print(f"\n{'=' * 40}")
    print(f"  {name}  (exit={code}, total={total:.3f}s)")
    print(f"{'=' * 40}")
    print(f"  import: {t_import - t0:.3f}s")
    print()

    # Time axis
    max_label_w = 24
    axis_marks = 5
    header = " " * (max_label_w + 3)
    for i in range(axis_marks + 1):
        t = total * i / axis_marks
        header += f"{t:.1f}s".ljust(width // axis_marks)
    print(header)
    print(" " * (max_label_w + 3) + "-" * width)

    # Print each span as a bar
    for thread_name in sorted(threads):
        thread_spans = sorted(threads[thread_name], key=lambda s: s[1])
        print(f"  [{thread_name}]")
        for label, start, end, in thread_spans:
            color = label_colors[label]
            duration = end - start
            short_label = label[:max_label_w].ljust(max_label_w)

            # Build the bar
            bar = [" "] * width
            col_start = int(start / total * width)
            col_end = max(col_start + 1, int(end / total * width))
            col_end = min(col_end, width)

            # Fill bar chars with the label text inside
            bar_text = f" {label} {duration:.3f}s "
            for i in range(col_start, col_end):
                offset = i - col_start
                if offset < len(bar_text):
                    bar[i] = bar_text[offset]
                else:
                    bar[i] = " "

            # Render with color
            line = list(" " * width)
            prefix = "".join(bar[:col_start])
            colored = color + BLACK_TEXT + "".join(bar[col_start:col_end]) + RESET
            suffix = "".join(bar[col_end:])
            print(f"  {short_label} |{prefix}{colored}{suffix}|")

    # Legend
    print()
    print("  Legend:")
    for label in sorted(label_colors, key=lambda l: next(
        s[1] for s in spans if s[0] == l
    )):
        color = label_colors[label]
        dur = next(s[2] - s[1] for s in spans if s[0] == label)
        print(f"    {color}{BLACK_TEXT} {label} {RESET} {dur:.3f}s")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir")
    parser.add_argument("cache_db")
    parser.add_argument("--name", default="grape pipeline")
    args = parser.parse_args()
    run_case(args.image_dir, args.cache_db, args.name)
