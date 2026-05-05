#!/usr/bin/env python3
"""Per-stage timing + concurrency timeline for grape's dask pipeline.

Wraps each @dask.delayed node in grape/cli.py to record (thread, t_start,
t_end). Runs one full grape invocation in-process and prints a Gantt-style
timeline showing which stages overlap.

Usage:
    .venv/bin/python scripts/timeline.py warm   # warm cache (default keywords)
    .venv/bin/python scripts/timeline.py cold   # empty cache
    .venv/bin/python scripts/timeline.py warm --keywords cat,dog,beach
    .venv/bin/python scripts/timeline.py warm --imgs /path/to/dir
    .venv/bin/python scripts/timeline.py warm --width 100   # chart width

The interesting questions this answers:
  - Does _load_model overlap _scan_files / _resolve_and_index_cache?
  - Where does the warm-cache critical path live (model? cache lookup? scan?)?
  - On cold runs, is encoding parallelized with text-keyword encoding?
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "tests" / "fixtures"

@dataclass
class Event:
    key: str
    thread: str
    t_start: float
    t_end: float = 0.0


@dataclass
class Trace:
    events: list = field(default_factory=list)
    process_start: float = 0.0


def _profiler_to_events(prof, t_origin: float) -> list[Event]:
    """Convert a dask.diagnostics.Profiler's results into Event records.

    Profiler.results entries are TaskData(key, task, start_time, end_time,
    worker_id) -- start/end are real wall-clock seconds on the worker
    thread. Subtract t_origin to make times relative.
    """
    events = []
    for r in prof.results:
        events.append(Event(
            key=str(r.key),
            thread=str(r.worker_id),
            t_start=r.start_time - t_origin,
            t_end=r.end_time - t_origin,
        ))
    return events


def _shorten_key(k: str) -> str:
    """Strip the dask task-key UUID suffix so '_load_model-<uuid>' -> '_load_model'."""
    import re
    # Dask uses "func-<uuid4>" keys. Strip everything after the first '-'
    # if what follows starts with hex characters (UUID).
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)-[0-9a-f]", k)
    return m.group(1) if m else k


def _render_gantt(trace: Trace, width: int) -> str:
    """ASCII Gantt chart, only including grape.cli's named delayed nodes."""
    events = list(trace.events)
    if not events:
        return "(no events captured)"
    t0 = min(ev.t_start for ev in events)
    t1 = max(ev.t_end for ev in events)
    span = max(t1 - t0, 1e-6)
    short = {ev.key: _shorten_key(ev.key) for ev in events}
    label_w = max(len(s) for s in short.values())
    thread_w = max(len(ev.thread) for ev in events)
    lines = [
        f"{'stage':<{label_w}}  {'thread':<{thread_w}}"
        f"  {'start':>8s} {'dur':>8s}  timeline"
    ]
    lines.append(
        f"{'':-<{label_w}}  {'':-<{thread_w}}  {'':->8s} {'':->8s}  {'':->{width}s}"
    )
    for ev in sorted(events, key=lambda e: e.t_start):
        rel_start = ev.t_start - t0
        dur = ev.t_end - ev.t_start
        bar_start = int(width * rel_start / span)
        bar_len = max(int(width * dur / span), 1)
        bar = " " * bar_start + "#" * bar_len
        bar = bar[:width].ljust(width)
        lines.append(
            f"{short[ev.key]:<{label_w}}  {ev.thread:<{thread_w}}"
            f"  {rel_start * 1000:>7.1f}ms {dur * 1000:>7.1f}ms  {bar}"
        )
    lines.append("")
    lines.append(f"total span: {span * 1000:.1f} ms")
    return "\n".join(lines)


def _summarize(trace: Trace) -> str:
    events = list(trace.events)
    if not events:
        return ""
    totals: dict[str, float] = {}
    for ev in events:
        label = _shorten_key(ev.key)
        totals[label] = totals.get(label, 0.0) + (ev.t_end - ev.t_start)
    out = ["", "stage totals (sum of wall time across calls):"]
    for label, total in sorted(totals.items(), key=lambda kv: -kv[1]):
        out.append(f"  {label:<32s} {total * 1000:>8.2f} ms")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("cold", "warm"),
        help="cold = empty cache; warm = pre-populated",
    )
    parser.add_argument(
        "--imgs", default=str(FIXTURES),
        help="image directory to scan (default: test fixtures)",
    )
    parser.add_argument(
        "--keywords", default="cat",
        help="comma-separated keywords (default: cat)",
    )
    parser.add_argument(
        "--model", default="ViT-B-32/laion2b_s34b_b79k",
        help="grape --model arg",
    )
    parser.add_argument(
        "--width", type=int, default=80, help="ASCII chart width",
    )
    args = parser.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="grape-timeline-"))
    cache_db = workdir / "cache.db"

    try:
        if args.mode == "warm":
            # Pre-populate the cache via a real subprocess so this run sees hits.
            print(f"populating cache via subprocess (warm setup)...", file=sys.stderr)
            subprocess.run(
                [
                    sys.executable, "-m", "grape",
                    "--model", args.model, "-q",
                    "--cache", str(cache_db),
                    "-R", "-k", args.keywords, args.imgs,
                ],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            print("done.", file=sys.stderr)

        from dask.diagnostics import Profiler

        sys.argv = [
            "grape", "--model", args.model, "-q",
            "--cache", str(cache_db),
            "-R", "-k", args.keywords, args.imgs,
        ]
        t_origin = time.perf_counter()
        from grape.cli import main as grape_main
        prof = Profiler()
        try:
            with prof:
                grape_main()
        except SystemExit:
            pass

        trace = Trace(events=_profiler_to_events(prof, t_origin))
        print(_render_gantt(trace, args.width))
        print(_summarize(trace))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
