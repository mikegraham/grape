"""Browser-level smoke tests for --view HTML output.

Renders the HTML in a headless browser and checks that images load,
appear in the right order, and are visually present in a screenshot.
Requires playwright and a browser (firefox).
"""

import io

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


def _render(page, html: str, tmp_path) -> None:
    """Write HTML under tmp_path (auto-cleaned by pytest) and navigate."""
    html_path = tmp_path / "rendered.html"
    html_path.write_bytes(html.encode())
    page.goto(html_path.as_uri())
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
        ScoredImage(path=str(red), scores={"fire": 0.80}, score=0.80),
        ScoredImage(path=str(blue), scores={"fire": 0.30}, score=0.30),
    ]
    html = _format_html(results, ["fire"])
    _render(browser_page, html, tmp_path)

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

    Verifies that the higher-score image actually renders ABOVE the
    lower-score image (catches order regressions, not just "an image
    rendered somewhere"). Each <img> is sized to natural 80x80 so a
    direct screenshot captures pure colour pixels.
    """
    red = tmp_path / "red.jpg"
    green = tmp_path / "green.jpg"
    _make_test_image(red, (255, 0, 0))
    _make_test_image(green, (0, 180, 0))

    results = [
        ScoredImage(path=str(red), scores={"color": 0.90}, score=0.90),
        ScoredImage(path=str(green), scores={"color": 0.50}, score=0.50),
    ]
    html = _format_html(results, ["color"])
    _render(browser_page, html, tmp_path)

    imgs = browser_page.query_selector_all("img")
    assert len(imgs) == 2

    expected = [
        ((255, 0, 0), "red"),
        ((0, 180, 0), "green"),
    ]
    last_y = -1.0
    for img_el, ((er, eg, eb), name) in zip(imgs, expected):
        bbox = img_el.bounding_box()
        assert bbox is not None
        assert bbox["y"] > last_y, (
            f"{name} should render below previous image (y={bbox['y']})"
        )
        last_y = bbox["y"]
        screenshot_bytes = img_el.screenshot(type="png")
        pixels = np.array(Image.open(io.BytesIO(screenshot_bytes)))
        avg = pixels[..., :3].mean(axis=(0, 1))
        assert avg[0] == pytest.approx(er, abs=40), f"{name} red avg: {avg}"
        assert avg[1] == pytest.approx(eg, abs=40), f"{name} green avg: {avg}"
        assert avg[2] == pytest.approx(eb, abs=40), f"{name} blue avg: {avg}"


def test_tall_image_capped_at_viewport(tmp_path, browser_page):
    """A source image taller than the viewport must scale to <= 100vh.

    Without ``max-height: 100vh`` on <img>, a 4000px-tall source would
    render at full natural height and you'd have to scroll to see it.
    """
    tall = tmp_path / "tall.jpg"
    Image.new("RGB", (400, 4000), (200, 100, 50)).save(tall, format="JPEG")
    results = [ScoredImage(path=str(tall), scores={"x": 0.5}, score=0.5)]
    html = _format_html(results, ["x"])
    _render(browser_page, html, tmp_path)

    img = browser_page.query_selector("img")
    assert img is not None
    bbox = img.bounding_box()
    viewport_h = browser_page.viewport_size["height"]
    assert bbox is not None
    # Subpixel rounding tolerance.
    assert bbox["height"] <= viewport_h + 1, (
        f"tall image rendered at {bbox['height']}px,"
        f" exceeds viewport {viewport_h}px"
    )


def test_small_image_not_upscaled(tmp_path, browser_page):
    """A small source image keeps its natural size (no blurry upscale)."""
    small = tmp_path / "small.jpg"
    Image.new("RGB", (80, 80), (50, 100, 200)).save(small, format="JPEG")
    results = [ScoredImage(path=str(small), scores={"x": 0.5}, score=0.5)]
    html = _format_html(results, ["x"])
    _render(browser_page, html, tmp_path)

    img = browser_page.query_selector("img")
    assert img is not None
    bbox = img.bounding_box()
    assert bbox is not None
    assert bbox["width"] <= 81, f"width {bbox['width']} > natural 80"
    assert bbox["height"] <= 81, f"height {bbox['height']} > natural 80"


def test_multiple_small_images_share_viewport(tmp_path, browser_page):
    """Several small images stack within one viewport height -- no
    forced one-image-per-viewport layout."""
    paths = []
    for i in range(4):
        p = tmp_path / f"small{i}.jpg"
        Image.new("RGB", (80, 80), (i * 60, 100, 200 - i * 60)).save(
            p, format="JPEG",
        )
        paths.append(p)
    results = [
        ScoredImage(path=str(p), scores={"x": 0.9 - i * 0.1}, score=0.9 - i * 0.1)
        for i, p in enumerate(paths)
    ]
    html = _format_html(results, ["x"])
    _render(browser_page, html, tmp_path)

    imgs = browser_page.query_selector_all("img")
    viewport_h = browser_page.viewport_size["height"]
    last_y = imgs[-1].bounding_box()["y"]
    assert last_y < viewport_h, (
        f"4 small images should fit in one {viewport_h}px viewport;"
        f" last image y={last_y}"
    )


def test_resolution_shown_next_to_filename(tmp_path, browser_page):
    """Image dimensions render right after the filename on the same line."""
    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (321, 234), (50, 100, 200)).save(img_path, format="JPEG")
    results = [ScoredImage(path=str(img_path), scores={"x": 0.5}, score=0.5)]
    html = _format_html(results, ["x"])
    _render(browser_page, html, tmp_path)

    res_el = browser_page.query_selector(".res")
    assert res_el is not None, "resolution element missing"
    assert res_el.text_content().strip() == "321x234"
    # Resolution should be visually adjacent to the path, not on a new line.
    path_box = browser_page.query_selector(".path").bounding_box()
    res_box = res_el.bounding_box()
    assert abs(path_box["y"] - res_box["y"]) < 4, (
        f"resolution not on same line as path: {path_box} vs {res_box}"
    )


def test_long_path_stays_single_line(tmp_path, browser_page):
    """Long paths render on one line with ellipsis, not wrapped."""
    img_path = tmp_path / ("a" * 200 + ".jpg")
    Image.new("RGB", (100, 100)).save(img_path, format="JPEG")
    results = [ScoredImage(path=str(img_path), scores={"x": 0.5}, score=0.5)]
    html = _format_html(results, ["x"])
    _render(browser_page, html, tmp_path)

    path_el = browser_page.query_selector(".path")
    assert path_el is not None
    info = path_el.evaluate(
        "el => ({"
        " offset: el.offsetHeight,"
        " line: parseFloat(getComputedStyle(el).lineHeight),"
        " scroll: el.scrollWidth,"
        " client: el.clientWidth,"
        "})"
    )
    assert info["offset"] <= info["line"] + 2, (
        f"path wrapped: offsetHeight={info['offset']} lineHeight={info['line']}"
    )
    assert info["scroll"] > info["client"], (
        "expected horizontal overflow so ellipsis activates"
    )
