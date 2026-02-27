"""Unit tests for the embedding cache (no model needed)."""

import time

import numpy as np
import pytest

from grape.cache import EmbeddingCache

EMBED_DIM = 512


def _rand_embedding(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal((1, EMBED_DIM)).astype(np.float32)
    emb /= np.linalg.norm(emb)
    return emb


@pytest.fixture
def cache(tmp_path):
    db = tmp_path / "test.db"
    c = EmbeddingCache(db)
    yield c
    c.close()


@pytest.fixture
def img(tmp_path):
    p = tmp_path / "image.jpg"
    p.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
    return p


def test_cache_miss_returns_none(cache, img):
    assert cache.get(img, "model-a") is None


def test_cache_roundtrip(cache, img):
    emb = _rand_embedding(1)
    cache.put(img, "model-a", emb)
    got = cache.get(img, "model-a")
    assert got is not None
    np.testing.assert_array_equal(got, emb)


def test_cache_invalidated_by_mtime(cache, img):
    emb = _rand_embedding(2)
    cache.put(img, "model-a", emb)

    # Touch the file so mtime_ns changes
    time.sleep(0.05)
    img.write_bytes(img.read_bytes())

    assert cache.get(img, "model-a") is None


def test_cache_invalidated_by_size(cache, img):
    emb = _rand_embedding(3)
    cache.put(img, "model-a", emb)

    # Rewrite with different size
    img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 200)

    assert cache.get(img, "model-a") is None


def test_cache_per_model(cache, img):
    emb_a = _rand_embedding(10)
    emb_b = _rand_embedding(20)
    cache.put(img, "model-a", emb_a)
    cache.put(img, "model-b", emb_b)

    got_a = cache.get(img, "model-a")
    got_b = cache.get(img, "model-b")
    assert got_a is not None
    assert got_b is not None
    np.testing.assert_array_equal(got_a, emb_a)
    np.testing.assert_array_equal(got_b, emb_b)


# --- not-image tracking ---

def test_not_image_roundtrip(cache, img):
    assert not cache.is_not_image(img)
    cache.put_not_image(img)
    assert cache.is_not_image(img)


def test_not_image_invalidated_on_change(cache, img):
    cache.put_not_image(img)
    time.sleep(0.05)
    img.write_bytes(img.read_bytes())
    assert not cache.is_not_image(img)
