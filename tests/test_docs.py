"""Guard against docs/screenshot_view.png drifting out of date.

The screenshot is a static render of grape's --view HTML over the test
fixtures.  When fixtures, the HTML template, the screenshot generator,
or the model change, the image becomes stale.  We hash the inputs and
store the digest next to the PNG; this test compares the committed
digest to the current inputs.

If this fails, run `python docs/generate_screenshots.py` and commit the
updated PNG + .inputs.sha256 sidecar.
"""

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_screenshot_view_is_current() -> None:
    import importlib.util
    script = REPO / "docs" / "generate_screenshots.py"
    sidecar = REPO / "docs" / "screenshot_view.inputs.sha256"

    spec = importlib.util.spec_from_file_location("generate_screenshots", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    actual = mod.screenshot_inputs_hash()
    expected = sidecar.read_text().strip()
    assert actual == expected, (
        f"docs/screenshot_view.png inputs have drifted.\n"
        f"  expected: {expected}\n"
        f"  actual:   {actual}\n"
        f"Regenerate: python docs/generate_screenshots.py"
    )
