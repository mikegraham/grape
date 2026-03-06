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


def test_get_many_for_paths_returns_matching_embeddings(cache, img):
    emb = _rand_embedding(21)
    cache.put(img, "model-a", emb)
    row = cache._conn.execute(
        "SELECT path, file_stat FROM embeddings WHERE model = ?",
        ("model-a",),
    ).fetchone()
    assert row is not None
    path, file_stat = row
    got = cache.get_many_for_paths("model-a", {path: file_stat})
    assert path in got
    np.testing.assert_array_equal(got[path].reshape(1, -1), emb)


def test_get_many_for_paths_filters_stale_stats(cache, img):
    emb = _rand_embedding(22)
    cache.put(img, "model-a", emb)
    row = cache._conn.execute(
        "SELECT path FROM embeddings WHERE model = ?",
        ("model-a",),
    ).fetchone()
    assert row is not None
    path = row[0]
    got = cache.get_many_for_paths("model-a", {path: "bad-stat"})
    assert got == {}


def test_embedding_index_for_model_returns_all_rows(cache, img, tmp_path):
    img2 = tmp_path / "image2.jpg"
    img2.write_bytes(b"\xff\xd8\xff\xe0" + b"\x11" * 80)
    emb1 = _rand_embedding(23)
    emb2 = _rand_embedding(24)
    emb_other = _rand_embedding(25)
    cache.put(img, "model-a", emb1)
    cache.put(img2, "model-a", emb2)
    cache.put(img, "model-b", emb_other)

    index = cache.embedding_index_for_model("model-a")
    rows = cache._conn.execute(
        "SELECT path, file_stat FROM embeddings WHERE model = ?",
        ("model-a",),
    ).fetchall()
    expected_by_path = {
        str(img.resolve()): emb1,
        str(img2.resolve()): emb2,
    }
    assert len(index) == 2
    for path, file_stat in rows:
        key = (path, file_stat)
        assert key in index
        np.testing.assert_array_equal(
            index[key].reshape(1, -1),
            expected_by_path[path],
        )


# --- image-hit tracking ---

def test_has_any_embedding_roundtrip(cache, img):
    assert not cache.has_any_embedding(img)
    cache.put(img, "model-a", _rand_embedding(30))
    assert cache.has_any_embedding(img)


def test_has_any_embedding_invalidated_on_change(cache, img):
    cache.put(img, "model-a", _rand_embedding(31))
    time.sleep(0.05)
    img.write_bytes(img.read_bytes())
    assert not cache.has_any_embedding(img)


def test_image_hit_index_contains_cached_file(cache, img):
    cache.put(img, "model-a", _rand_embedding(32))
    stat = cache._conn.execute(
        "SELECT file_stat FROM embeddings WHERE path = ?",
        (str(img.resolve()),),
    ).fetchone()[0]
    assert (str(img.resolve()), stat) in cache.image_hit_index()


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


def test_not_image_index_contains_cached_non_image(cache, img):
    cache.put_not_image(img)
    stat = cache._conn.execute(
        "SELECT file_stat FROM not_images WHERE path = ?",
        (str(img.resolve()),),
    ).fetchone()[0]
    assert (str(img.resolve()), stat) in cache.not_image_index()


# --- text embedding cache ---

def test_text_embedding_roundtrip(cache):
    emb_cat = _rand_embedding(40)
    emb_dog = _rand_embedding(41)

    cache.put_text_embeddings("model-a", [
        ("a photo of a cat", emb_cat),
        ("a photo of a dog", emb_dog),
    ])
    got = cache.get_text_embeddings(
        "model-a", ["a photo of a cat", "a photo of a dog"],
    )
    assert len(got) == 2
    np.testing.assert_array_equal(got["a photo of a cat"], emb_cat)
    np.testing.assert_array_equal(got["a photo of a dog"], emb_dog)


def test_text_embedding_miss_returns_empty(cache):
    got = cache.get_text_embeddings("model-a", ["a photo of a cat"])
    assert got == {}


def test_text_embedding_per_model(cache):
    emb_a = _rand_embedding(42)
    emb_b = _rand_embedding(43)

    cache.put_text_embeddings("model-a", [("a photo of a cat", emb_a)])
    cache.put_text_embeddings("model-b", [("a photo of a cat", emb_b)])

    got_a = cache.get_text_embeddings("model-a", ["a photo of a cat"])
    got_b = cache.get_text_embeddings("model-b", ["a photo of a cat"])
    np.testing.assert_array_equal(got_a["a photo of a cat"], emb_a)
    np.testing.assert_array_equal(got_b["a photo of a cat"], emb_b)


def test_text_embedding_same_keyword_different_templates(cache):
    """Same keyword formatted with different templates produces distinct entries."""
    emb1 = _rand_embedding(44)
    emb2 = _rand_embedding(45)

    cache.put_text_embeddings("model-a", [("a photo of a cat", emb1)])
    cache.put_text_embeddings("model-a", [("a cat in the wild", emb2)])

    got = cache.get_text_embeddings(
        "model-a", ["a photo of a cat", "a cat in the wild"],
    )
    np.testing.assert_array_equal(got["a photo of a cat"], emb1)
    np.testing.assert_array_equal(got["a cat in the wild"], emb2)
