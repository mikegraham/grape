import argparse
import sys
import threading
import time
from types import SimpleNamespace

process_start = time.perf_counter()

import concurrent.futures as cf

import grape.cli as cli
from grape.cache import EmbeddingCache

import_done = time.perf_counter()


def run_case(image_dir: str, cache_db: str, name: str) -> None:
    events: list[tuple[float, str, str]] = []
    score_stats = SimpleNamespace(count=0, total=0.0)

    def mark(label: str) -> None:
        events.append((time.perf_counter(), label, threading.current_thread().name))

    def wrap(label: str, fn):
        def inner(*args, **kwargs):
            mark(f"{label}:start")
            try:
                return fn(*args, **kwargs)
            finally:
                mark(f"{label}:end")

        return inner

    # Capture task submission timing from main()'s executor usage.
    class InstrumentedExecutor(cf.ThreadPoolExecutor):
        def submit(self, fn, *args, **kwargs):
            fn_name = getattr(fn, "__name__", repr(fn))
            mark(f"executor.submit:{fn_name}")
            return super().submit(fn, *args, **kwargs)

    orig_executor = cli.ThreadPoolExecutor
    cli.ThreadPoolExecutor = InstrumentedExecutor

    # Wrap key stage functions used by production flow.
    originals = {
        "_build_scan_indexes": cli._build_scan_indexes,
        "_load_model": cli._load_model,
        "_preload_view_modules": cli._preload_view_modules,
        "_scan_paths_worker": cli._scan_paths_worker,
        "encode_keywords": cli.encode_keywords,
        "_score_or_stub_results": cli._score_or_stub_results,
        "_score_batch": cli._score_batch,
        "_format_html": cli._format_html,
        "_show_in_webview": cli._show_in_webview,
    }

    cli._build_scan_indexes = wrap("build_scan_indexes", cli._build_scan_indexes)
    cli._load_model = wrap("load_model", cli._load_model)
    cli._preload_view_modules = wrap("preload_view_modules", cli._preload_view_modules)
    cli._scan_paths_worker = wrap("scan_paths_worker", cli._scan_paths_worker)
    cli.encode_keywords = wrap("encode_keywords", cli.encode_keywords)
    cli._score_or_stub_results = wrap("score_or_stub_results", cli._score_or_stub_results)

    orig_score_batch = cli._score_batch

    def instrumented_score_batch(*args, **kwargs):
        if score_stats.count == 0:
            mark("score_batch:first:start")
        s = time.perf_counter()
        out = orig_score_batch(*args, **kwargs)
        e = time.perf_counter()
        score_stats.total += e - s
        score_stats.count += 1
        if score_stats.count == 1:
            mark("score_batch:first:end")
        return out

    cli._score_batch = instrumented_score_batch
    cli._format_html = wrap("format_html", cli._format_html)

    # Prevent opening a real window while preserving production flow up to render.
    def fake_show_in_webview(html_doc: str):
        mark("show_in_webview:start")
        _ = len(html_doc)
        mark("show_in_webview:end")

    cli._show_in_webview = fake_show_in_webview

    orig_embed_index = EmbeddingCache.embedding_index_for_model

    def instrumented_embed_index(self, model_id):
        mark("embedding_index_for_model:start")
        try:
            return orig_embed_index(self, model_id)
        finally:
            mark("embedding_index_for_model:end")

    EmbeddingCache.embedding_index_for_model = instrumented_embed_index

    argv = [
        "grape",
        "-q",
        "--view",
        "--cache",
        cache_db,
        "-r",
        "dog",
        image_dir,
    ]

    mark("main:start")
    code = 0
    try:
        old_argv = sys.argv
        sys.argv = argv
        cli.main()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    finally:
        sys.argv = old_argv
        mark("main:end")

        # Restore patched symbols.
        cli.ThreadPoolExecutor = orig_executor
        for k, v in originals.items():
            setattr(cli, k, v)
        EmbeddingCache.embedding_index_for_model = orig_embed_index

    # Print timeline indexed from process start (true t=0 for run).
    print(f"\n=== {name} ===")
    print(f"exit_code={code}")
    print(f"t=0 process_start")
    print(f"t={import_done - process_start:8.3f} import_grape_cli_done")

    for ts, label, thread in sorted(events, key=lambda x: x[0]):
        print(f"t={ts - process_start:8.3f} {label} [{thread}]")

    print(f"score_batch_calls={score_stats.count}")
    print(f"score_batch_total_s={score_stats.total:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir")
    parser.add_argument("cache_db")
    parser.add_argument("--name", default="case")
    args = parser.parse_args()
    run_case(args.image_dir, args.cache_db, args.name)
