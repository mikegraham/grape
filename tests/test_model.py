"""Integration tests that load the CLIP model.

These are slow on first run (model download ~350MB).
"""

import re
import shutil
from pathlib import Path

import numpy as np
import pytest

from grape.search import find_images, score_image, score_images

pytestmark = pytest.mark.slow


# --- embedding properties ---

def test_encode_image_normalized(clip_model, fixtures_dir):
    emb = clip_model.encode_image(str(fixtures_dir / "dog.jpg"))
    norm = np.linalg.norm(emb, axis=-1)
    np.testing.assert_allclose(norm, 1.0, atol=1e-5)


def test_encode_texts_normalized(clip_model):
    emb = clip_model.encode_texts(["hello"])
    norm = np.linalg.norm(emb, axis=-1)
    np.testing.assert_allclose(norm, 1.0, atol=1e-5)


def test_different_images_produce_different_embeddings(clip_model, fixtures_dir):
    dog_emb = clip_model.encode_image(str(fixtures_dir / "dog.jpg"))
    beach_emb = clip_model.encode_image(str(fixtures_dir / "beach.jpg"))
    sim = float((dog_emb @ beach_emb.T)[0, 0])
    assert sim < 0.95


# --- semantic matching ---

# (query, image that should rank FIRST) -- things a human would get right.
POSITIVE_MATCHES = [
    ("dog", "dog.jpg"),
    ("cat", "cat.jpg"),
    ("beach", "beach.jpg"),
    ("raspberries", "food.jpg"),
    ("city", "city.jpg"),
    ("forest", "forest.jpg"),
    ("woman wearing hat", "lenna.png"),
    ("teapot", "teapot.png"),
    ("handwritten digits", "mnist.png"),
    ("omelette", "omelette.jpg"),
    ("woodworking", "dovetail.jpg"),
    ("blacksmith", "blacksmith.jpg"),
    ("restaurant", "toms_diner.jpg"),
]

# (query, image that should NOT rank first) -- things a human would never confuse.
NEGATIVE_MATCHES = [
    ("dog", "teapot.png"),
    ("beach", "mnist.png"),
    ("teapot", "forest.jpg"),
    ("handwritten digits", "beach.jpg"),
    ("cat", "toms_diner.jpg"),
    ("restaurant", "cat.jpg"),
    ("forest", "omelette.jpg"),
]


def test_semantic_ranking(clip_model, fixtures_dir):
    """The model should rank the right image first for most queries."""
    images = find_images(str(fixtures_dir), recursive=False)

    correct = 0
    details = []
    checks = []

    for keyword, expected_name in POSITIVE_MATCHES:
        results = score_images(clip_model, images, [keyword], quiet=True)
        results.sort(key=lambda r: r.score, reverse=True)
        top_name = results[0].path.name
        ok = top_name == expected_name
        correct += ok
        checks.append(ok)
        details.append(f"  {'ok' if ok else 'MISS':4s}  +{keyword:25s} -> {top_name}"
                        + (f" (expected {expected_name})" if not ok else ""))

    for keyword, wrong_name in NEGATIVE_MATCHES:
        results = score_images(clip_model, images, [keyword], quiet=True)
        results.sort(key=lambda r: r.score, reverse=True)
        top_name = results[0].path.name
        ok = top_name != wrong_name
        correct += ok
        checks.append(ok)
        details.append(f"  {'ok' if ok else 'MISS':4s}  -{keyword:25s} != {wrong_name}"
                        + (f" (but got {top_name})" if not ok else ""))

    summary = "\n".join(details)
    threshold = len(checks) * 3 // 4  # 75%
    assert correct >= threshold, (
        f"Only {correct}/{len(checks)} correct"
        f" (need {threshold}):\n{summary}"
    )


# --- score_image / score_images ---

def test_score_image_mean_matches_keyword_average(clip_model, fixtures_dir):
    result = score_image(clip_model, fixtures_dir / "dog.jpg", ["dog", "cat", "car"])
    scores = list(result.scores.values())
    np.testing.assert_allclose(result.score, np.mean(scores), atol=1e-7)


def test_score_images_skips_unreadable(clip_model, tmp_path):
    """Corrupt files should be skipped, not crash the batch."""
    good = tmp_path / "good.jpg"
    bad = tmp_path / "corrupt.jpg"

    src = Path(__file__).parent / "fixtures" / "dog.jpg"
    shutil.copy(src, good)
    bad.write_bytes(b"not a real image at all")

    results = score_images(clip_model, [good, bad], ["dog"], quiet=True)
    assert len(results) == 1
    assert results[0].path == good


# --- model metadata ---

def test_model_id_format(clip_model):
    """model_id should be '{repo}@{40-char-hex}'."""
    mid = clip_model.model_id()
    assert re.fullmatch(r".+@[0-9a-f]{40}", mid), f"unexpected model_id: {mid}"


def test_embed_dim(clip_model):
    assert clip_model.embed_dim() == 512
