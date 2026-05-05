#!/usr/bin/env python3
"""End-to-end benchmark suite for grape.

Runs each scenario `--repeats` times and reports min / median wall-clock.
Min is the most noise-resistant statistic on a busy machine; median is
shown for context.

Usage:
    .venv/bin/python scripts/benchmark.py
    .venv/bin/python scripts/benchmark.py --repeats 5 --filter warm
    .venv/bin/python scripts/benchmark.py --case animated_gif
    .venv/bin/python scripts/benchmark.py --keep-temp   # leave caches/dirs

Each case is a fresh `python -m grape ...` subprocess, so it includes
interpreter startup, import time, and shutdown -- the same costs a
real user pays.
"""

from __future__ import annotations

import argparse
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "tests" / "fixtures"
DEFAULT_MODEL = "ViT-B-32/laion2b_s34b_b79k"
LARGE_MULTIPLIER = 10  # 14 fixtures * 10 = 140 images


def _grape(args: list[str], cache: Path | None) -> list[str]:
    """Build the argv for a `python -m grape` subprocess."""
    cmd = [sys.executable, "-m", "grape", "--model", DEFAULT_MODEL, "-q"]
    if cache is None:
        cmd.append("--no-cache")
    else:
        cmd.extend(["--cache", str(cache)])
    cmd.extend(args)
    return cmd


def _run(argv: list[str]) -> float:
    """Run a subprocess once; return wall-clock seconds."""
    t0 = time.perf_counter()
    subprocess.run(
        argv, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return time.perf_counter() - t0


def _build_large_dir(parent: Path) -> Path:
    """Duplicate fixtures into a larger corpus to exercise scan + batch paths."""
    out = parent / "large"
    out.mkdir(exist_ok=True)
    sources = sorted(FIXTURES.glob("*.jpg")) + sorted(FIXTURES.glob("*.png"))
    for i in range(LARGE_MULTIPLIER):
        for src in sources:
            shutil.copy(src, out / f"{i}_{src.name}")
    return out


def _build_animated_gif(parent: Path, n_frames: int = 8) -> Path:
    """Build a multi-frame GIF from a few real fixture images."""
    from PIL import Image

    names = [
        "cat", "dog", "beach", "forest",
        "food", "city", "blacksmith", "lenna",
    ]
    frames = [
        Image.open(FIXTURES / f"{n}.jpg" if n != "lenna" else FIXTURES / "lenna.png")
        .convert("RGB").resize((224, 224))
        for n in names[:n_frames]
    ]
    out = parent / "animated.gif"
    frames[0].save(
        out, format="GIF", save_all=True, append_images=frames[1:],
        duration=200, loop=0,
    )
    return out


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

def case_help(_workdir: Path) -> tuple[str, list[list[str]]]:
    """CLI startup with no work (--help). Pure interpreter + import cost."""
    return ("help (startup baseline)", [
        [sys.executable, "-m", "grape", "--help"],
    ])


def case_warm_small(workdir: Path) -> tuple[str, list[list[str]]]:
    """Warm cache, ~14 fixture images, 1 keyword. The hot path."""
    cache = workdir / "warm-small.db"
    # Pre-populate the cache before timing.
    subprocess.run(
        _grape(["-R", "-k", "cat", str(FIXTURES)], cache),
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return ("warm small (14 hits, 1 kw)",
            [_grape(["-R", "-k", "cat", str(FIXTURES)], cache)])


def case_warm_large(workdir: Path) -> tuple[str, list[list[str]]]:
    """Warm cache, ~140 images, 1 keyword. Tests cache lookup at scale."""
    large = _build_large_dir(workdir)
    cache = workdir / "warm-large.db"
    subprocess.run(
        _grape(["-R", "-k", "cat", str(large)], cache),
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return ("warm large (140 hits, 1 kw)",
            [_grape(["-R", "-k", "cat", str(large)], cache)])


def case_warm_8kw(workdir: Path) -> tuple[str, list[list[str]]]:
    """Warm image cache + warm text cache, 8 keywords."""
    large = _build_large_dir(workdir)
    cache = workdir / "warm-8kw.db"
    keywords = "cat,dog,beach,city,forest,food,car,plane"
    # Pre-populate both image and text caches.
    subprocess.run(
        _grape(["-R", "-k", keywords, str(large)], cache),
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return ("warm large + 8 kw (text cache hot)",
            [_grape(["-R", "-k", keywords, str(large)], cache)])


def case_cold_small(workdir: Path) -> tuple[str, list[list[str]]]:
    """Cold cache, ~14 images. Each repeat starts with an empty cache,
    so this measures end-to-end first-run encode time."""
    cmds = []
    for i in range(20):  # generate enough fresh-cache cmds for any --repeats
        cache = workdir / f"cold-small-{i}.db"
        cmds.append(_grape(["-R", "-k", "cat", str(FIXTURES)], cache))
    return ("cold small (14 imgs encoded)", cmds)


def case_cold_large(workdir: Path) -> tuple[str, list[list[str]]]:
    """Cold cache, ~140 images. Heavy: full model load + 140 image encodes."""
    large = _build_large_dir(workdir)
    cmds = []
    for i in range(20):
        cache = workdir / f"cold-large-{i}.db"
        cmds.append(_grape(["-R", "-k", "cat", str(large)], cache))
    return ("cold large (140 imgs encoded)", cmds)


def case_animated_gif(workdir: Path) -> tuple[str, list[list[str]]]:
    """Score a single animated GIF (multi-frame K=8 mean pool, no cache)."""
    gif = _build_animated_gif(workdir)
    return ("animated gif (8 frames, no cache)",
            [_grape(["-k", "cat", str(gif)], None)])


CASES = {
    "help": case_help,
    "warm_small": case_warm_small,
    "warm_large": case_warm_large,
    "warm_8kw": case_warm_8kw,
    "cold_small": case_cold_small,
    "cold_large": case_cold_large,
    "animated_gif": case_animated_gif,
}


def run(case_name: str, repeats: int, workdir: Path) -> dict:
    label, commands = CASES[case_name](workdir)
    times = []
    for i in range(repeats):
        cmd = commands[min(i, len(commands) - 1)]
        times.append(_run(cmd))
    return {
        "name": label,
        "min": min(times),
        "median": statistics.median(times),
        "max": max(times),
        "n": len(times),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repeats", type=int, default=3,
        help="how many times to run each case (default 3)",
    )
    parser.add_argument(
        "--filter", default="",
        help="substring filter on case names (e.g. 'warm')",
    )
    parser.add_argument(
        "--case", action="append", default=[],
        help="run only this case (repeatable)",
    )
    parser.add_argument(
        "--keep-temp", action="store_true",
        help="don't delete the temp workdir on exit",
    )
    args = parser.parse_args()

    selected = args.case or [
        n for n in CASES if args.filter in n
    ]
    if not selected:
        print(f"no cases matched (available: {', '.join(CASES)})")
        sys.exit(2)

    if "HUGGINGFACE_HUB_CACHE" not in os.environ:
        print(
            "warning: HUGGINGFACE_HUB_CACHE not set;"
            " first run may download model weights",
            file=sys.stderr,
        )

    workdir = Path(tempfile.mkdtemp(prefix="grape-bench-"))
    print(f"workdir: {workdir}")
    print(f"repeats per case: {args.repeats}")
    print()
    print(f"{'case':<40} {'min':>8} {'median':>8} {'max':>8}")
    print("-" * 70)
    try:
        for case_name in selected:
            r = run(case_name, args.repeats, workdir)
            print(
                f"{r['name']:<40}"
                f" {r['min']:>7.3f}s {r['median']:>7.3f}s {r['max']:>7.3f}s"
            )
    finally:
        if not args.keep_temp:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
