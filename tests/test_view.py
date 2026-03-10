"""Browser-level smoke tests for --view HTML output.

Renders the HTML in a headless browser and checks that images load,
appear in the right order, and are visually present in a screenshot.
Requires playwright and a browser (firefox).
"""

import numpy as np
import pytest
from PIL import Image

from grape.cli import _format_html
from grape.search import ScoredImage


@pytest.fixture(scope="module")
def _browser():
    """Launch Firefox once for the whole module."""
    from playwright.sync_api import sync_playwright
    pw = sync_playwright().start()
    browser = pw.firefox.launch(headless=True)
    yield browser
    browser.close()
    pw.stop()


@pytest.fixture()
def browser_page(_browser):
    page = _browser.new_page(viewport={"width": 1280, "height": 2000})
    yield page
    page.close()


def _render(page, html: str) -> None:
    """Write HTML to a temp file and navigate to it."""
    import tempfile
    from pathlib import Path

    # Write to a persistent temp file (cleaned up by OS).
    f = tempfile.NamedTemporaryFile(
        suffix=".html", prefix="grape-test-", delete=False,
    )
    f.write(html.encode())
    f.close()
    page.goto(Path(f.name).as_uri())
    page.wait_for_load_state("networkidle")


def _make_test_image(path, color):
    """Create a small solid-color JPEG for testing."""
    Image.new("RGB", (80, 80), color).save(path, format="JPEG")


def test_images_load_and_ordered_by_score(tmp_path, browser_page):
    """All <img> elements load successfully, best score on top."""
    red = tmp_path / "red.jpg"
    blue = tmp_path / "blue.jpg"
    _make_test_image(red, (255, 0, 0))
    _make_test_image(blue, (0, 0, 255))

    results = [
        ScoredImage(path=red, scores={"fire": 0.80}, score=0.80),
        ScoredImage(path=blue, scores={"fire": 0.30}, score=0.30),
    ]
    html = _format_html(results, ["fire"])
    _render(browser_page, html)

    imgs = browser_page.query_selector_all("img")
    assert len(imgs) == len(results), "one <img> per result"

    prev_y = -1
    for i, img in enumerate(imgs):
        info = img.evaluate(
            "el => ({w: el.naturalWidth, h: el.naturalHeight, complete: el.complete})",
        )
        assert info["complete"], f"img[{i}] not loaded"
        assert info["w"] > 0 and info["h"] > 0, f"img[{i}] has zero dimensions"

        box = img.bounding_box()
        assert box is not None, f"img[{i}] not visible"
        assert box["y"] > prev_y, "images should appear in top-to-bottom order"
        prev_y = box["y"]


def test_screenshot_contains_result_images(tmp_path, browser_page):
    """The rendered page screenshot contains the source images.

    Uses average color of each source image and checks that the
    corresponding color appears in the screenshot region for that image.
    """
    red = tmp_path / "red.jpg"
    green = tmp_path / "green.jpg"
    _make_test_image(red, (255, 0, 0))
    _make_test_image(green, (0, 180, 0))

    results = [
        ScoredImage(path=red, scores={"color": 0.90}, score=0.90),
        ScoredImage(path=green, scores={"color": 0.50}, score=0.50),
    ]
    html = _format_html(results, ["color"])
    _render(browser_page, html)

    imgs = browser_page.query_selector_all("img")
    assert len(imgs) == 2

    # Check that each <img> region in the screenshot has the expected color.
    expected_colors = [(255, 0, 0), (0, 180, 0)]
    for img_el, (er, eg, eb) in zip(imgs, expected_colors):
        screenshot_bytes = img_el.screenshot(type="png")
        screenshot = Image.open(__import__("io").BytesIO(screenshot_bytes))
        pixels = np.array(screenshot)
        # Average color of the screenshot region (ignore alpha if present).
        avg = pixels[..., :3].mean(axis=(0, 1))
        # The dominant channel should match. Use loose threshold because
        # JPEG compression shifts colors slightly.
        assert avg[0] == pytest.approx(er, abs=40), f"red channel: {avg}"
        assert avg[1] == pytest.approx(eg, abs=40), f"green channel: {avg}"
        assert avg[2] == pytest.approx(eb, abs=40), f"blue channel: {avg}"
