"""Guard against docs/ sample outputs drifting out of date.

The sample files in docs/ (screenshot, text snapshots) are produced by
docs/generate_screenshots.py over the test fixtures.  When fixtures, the
HTML template, the generator, the model, or the CLI args change, the
saved files become stale.  We hash the inputs and store the digest next
to the samples; the fast test compares the committed digest to the
current inputs.  A slow test additionally runs grape end-to-end and
diffs stdout against the committed text snapshots.

If either fails, run `python docs/generate_screenshots.py` and commit
the regenerated samples (plus the .inputs.sha256 sidecar).
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "generate_screenshots", DOCS / "generate_screenshots.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_sample_inputs_hash_is_current() -> None:
    mod = _load_generator()
    actual = mod.screenshot_inputs_hash()
    expected = (DOCS / "sample_inputs.sha256").read_text().strip()
    assert actual == expected, (
        f"docs/ samples have drifted.\n"
        f"  expected: {expected}\n"
        f"  actual:   {actual}\n"
        f"Regenerate: python docs/generate_screenshots.py"
    )


@pytest.mark.slow
@pytest.mark.parametrize("snapshot,args_attr", [
    ("sample_scores.txt", "SAMPLE_SCORES_ARGS"),
    ("sample_verbose.txt", "SAMPLE_VERBOSE_ARGS"),
])
def test_text_snapshot_matches_grape_output(snapshot: str, args_attr: str) -> None:
    """Run grape end-to-end and compare stdout to the saved snapshot."""
    mod = _load_generator()
    args = getattr(mod, args_attr)
    result = subprocess.run(
        [sys.executable, "-m", "grape", "--no-cache", "--model", mod.MODEL, *args],
        capture_output=True, text=True, check=True,
    )
    actual = result.stdout.replace(str(mod.FIXTURES), "tests/fixtures")
    expected = (DOCS / snapshot).read_text()
    assert actual == expected, (
        f"docs/{snapshot} is out of date.\n"
        f"Regenerate: python docs/generate_screenshots.py"
    )
