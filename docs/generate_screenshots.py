#!/usr/bin/env python3
"""Regenerate README screenshot from the test fixtures.

Usage:
    .venv/bin/python docs/generate_screenshots.py

Produces:
    docs/screenshot_view.png  -- browser render of --view HTML output

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


def screenshot_inputs_hash() -> str:
    """Hash of everything the rendered screenshot depends on.

    If this changes, docs/screenshot_view.png is stale and must be
    regenerated with this script.  A test (tests/test_docs.py) guards
    against drift by comparing this hash to docs/screenshot_view.inputs.sha256.
    """
    h = hashlib.sha256()

    # All fixture files, in sorted order.
    for p in sorted(FIXTURES.rglob("*")):
        if p.is_file() and not p.name.startswith("."):
            h.update(str(p.relative_to(FIXTURES)).encode())
            h.update(b"\0")
            h.update(p.read_bytes())
            h.update(b"\0")

    # The HTML template used to render --view.
    from grape.cli import _HTML_TEMPLATE_TEXT
    h.update(b"template\0")
    h.update(_HTML_TEMPLATE_TEXT.encode())

    # The generator script itself (this file).
    h.update(b"generator\0")
    h.update(Path(__file__).read_bytes())

    # The model used to score -- embedding changes would change results.
    h.update(b"model\0")
    h.update(MODEL.encode())

    return h.hexdigest()


def generate_view_screenshot() -> None:
    """Run grape, render --view HTML in headless Firefox, screenshot it."""
    from grape.cli import _format_html
    from grape.search import ScoredImage

    # Run grape to get real scores against fixtures.
    result = subprocess.run(
        [
            sys.executable, "-m", "grape",
            "--no-cache", "--model", MODEL,
            "--scores", "--keywords", "sunset",
            "--top", "5",
            "-R", str(FIXTURES),
        ],
        capture_output=True, text=True,
    )

    # Parse output into ScoredImage objects.
    results = []
    for line in result.stdout.strip().splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2:
            continue
        score_str, path_str = parts
        path = Path(path_str.strip())
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
    hash_file = DOCS / "screenshot_view.inputs.sha256"
    hash_file.write_text(screenshot_inputs_hash() + "\n")
    print(f"wrote {out}")
    print(f"wrote {hash_file}")


if __name__ == "__main__":
    generate_view_screenshot()
