#!/usr/bin/env python3
"""Regenerate the documentation samples from the test fixtures.

Usage:
    .venv/bin/python docs/generate_screenshots.py

Produces:
    docs/screenshot_view.png       - browser render of --view HTML output
    docs/sample_scores.txt         - text output with -s (scores)
    docs/sample_verbose.txt        - text output with -v (per-keyword)
    docs/sample_inputs.sha256      - hash of the inputs that produced the above

A test (tests/test_docs.py) guards against drift by comparing the saved
hash to the current inputs.

Requirements: playwright (with firefox installed).
"""

import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "tests" / "fixtures"
DOCS = REPO / "docs"
MODEL = "ViT-B-32/laion2b_s34b_b79k"

# Repo-relative fixture path here (subprocess runs with cwd=REPO) so the
# hash is reproducible across machines.
SAMPLE_SCORES_ARGS = [
    "--scores", "--keywords", "sunset", "--top", "5", "-R", "tests/fixtures",
]
SAMPLE_VERBOSE_ARGS = [
    "-v", "--keywords", "sunset,beach", "--top", "3", "-R", "tests/fixtures",
]


def screenshot_inputs_hash() -> str:
    """Hash of everything the rendered samples depend on.

    If this changes, the saved sample files are stale and must be
    regenerated with this script.
    """
    h = hashlib.sha256()

    for p in sorted(FIXTURES.rglob("*")):
        if p.is_file() and not p.name.startswith("."):
            h.update(str(p.relative_to(FIXTURES)).encode())
            h.update(b"\0")
            h.update(p.read_bytes())
            h.update(b"\0")

    from grape.cli import _HTML_TEMPLATE_TEXT
    h.update(b"template\0")
    h.update(_HTML_TEMPLATE_TEXT.encode())

    h.update(b"generator\0")
    h.update(Path(__file__).read_bytes())

    h.update(b"model\0")
    h.update(MODEL.encode())

    for args in (SAMPLE_SCORES_ARGS, SAMPLE_VERBOSE_ARGS):
        h.update(b"args\0")
        h.update("\0".join(args).encode())

    return h.hexdigest()


def _run_grape(args: list[str]) -> str:
    result = subprocess.run(
        [sys.executable, "-m", "grape", "--no-cache", "--model", MODEL, *args],
        capture_output=True, text=True, check=True, cwd=REPO,
    )
    return result.stdout


def generate_view_screenshot() -> None:
    """Run grape, render --view HTML in headless Firefox, screenshot it."""
    from grape.cli import _format_html
    from grape.search import ScoredImage

    result = subprocess.run(
        [
            sys.executable, "-m", "grape",
            "--no-cache", "--model", MODEL,
            "--scores", "--keywords", "sunset",
            "--top", "5",
            "-R", "tests/fixtures",
        ],
        capture_output=True, text=True, check=True, cwd=REPO,
    )

    results = []
    for line in result.stdout.strip().splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2:
            continue
        score_str, path_str = parts
        path = path_str.strip()
        results.append(ScoredImage(
            path=path,
            scores={"sunset": float(score_str)},
            score=float(score_str),
        ))

    html = _format_html(results, ["sunset"])

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        page = browser.new_page(viewport={"width": 800, "height": 1000})
        with tempfile.NamedTemporaryFile(
            suffix=".html", prefix="grape-", delete=False,
        ) as f:
            f.write(html.encode())
            html_path = Path(f.name)
        page.goto(html_path.as_uri())
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(500)
        out = DOCS / "screenshot_view.png"
        page.screenshot(path=str(out), full_page=True)
        browser.close()
        html_path.unlink()
    print(f"wrote {out}")


def generate_text_samples() -> None:
    (DOCS / "sample_scores.txt").write_text(_run_grape(SAMPLE_SCORES_ARGS))
    print(f"wrote {DOCS / 'sample_scores.txt'}")
    (DOCS / "sample_verbose.txt").write_text(_run_grape(SAMPLE_VERBOSE_ARGS))
    print(f"wrote {DOCS / 'sample_verbose.txt'}")


def main() -> None:
    generate_view_screenshot()
    generate_text_samples()
    hash_file = DOCS / "sample_inputs.sha256"
    hash_file.write_text(screenshot_inputs_hash() + "\n")
    print(f"wrote {hash_file}")


if __name__ == "__main__":
    main()
