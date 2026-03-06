"""Tests for image discovery (no model needed)."""

import numpy as np
import pytest
from PIL import Image

from grape.cache import EmbeddingCache
from grape.search import find_images, iter_image_records, score_image


def test_finds_images_not_text(fixtures_dir):
    images = find_images(str(fixtures_dir), recursive=False)
    names = {p.name for p in images}
    assert "dog.jpg" in names
    assert "cat.jpg" in names
    assert "readme.txt" not in names


def test_recursive_finds_subdirs(fixtures_dir):
    images = find_images(str(fixtures_dir), recursive=True)
    names = {p.name for p in images}
    assert "sunset.jpg" in names


def test_non_recursive_skips_subdirs(fixtures_dir):
    images = find_images(str(fixtures_dir), recursive=False)
    names = {p.name for p in images}
    assert "sunset.jpg" not in names


def test_nonexistent_dir():
    images = find_images("/tmp/nonexistent_dir_grape_test")
    assert images == []


def test_returns_sorted(fixtures_dir):
    images = find_images(str(fixtures_dir), recursive=False)
    names = [p.name for p in images]
    assert names == sorted(names)


def test_detects_by_content_not_extension(tmp_path):
    """Files are identified as images by content, not extension."""
    # Real image, wrong extension -- should be found
    Image.new("RGB", (1, 1)).save(tmp_path / "photo.dat", format="JPEG")
    # Text file with image extension -- should be skipped
    (tmp_path / "fake.jpg").write_bytes(b"not an image at all")
    # Plain text -- should be skipped
    (tmp_path / "notes.txt").write_bytes(b"hello world")

    images = find_images(str(tmp_path))
    names = {p.name for p in images}
    assert "photo.dat" in names
    assert "fake.jpg" not in names
    assert "notes.txt" not in names


def test_iter_image_records_includes_cache_keys(tmp_path):
    image_path = tmp_path / "real.jpg"
    Image.new("RGB", (1, 1)).save(image_path, format="JPEG")

    records = list(iter_image_records(str(tmp_path)))
    assert len(records) == 1
    record = records[0]
    assert record.path == image_path
    assert record.path_key == str(image_path)
    assert isinstance(record.file_stat, str)


# --- find_images with cache ---

def test_find_images_caches_non_images(tmp_path):
    """Second scan of a mixed directory skips PIL for known non-images."""
    Image.new("RGB", (1, 1)).save(tmp_path / "real.jpg", format="JPEG")
    (tmp_path / "data.bin").write_bytes(b"not an image")

    cache = EmbeddingCache(tmp_path / "test.db")
    images = find_images(str(tmp_path), cache=cache)
    assert {p.name for p in images} == {"real.jpg"}
    # The non-image is now recorded
    assert cache.is_not_image(tmp_path / "data.bin")
    # Second call still returns the same results
    assert find_images(str(tmp_path), cache=cache) == images
    cache.close()


def test_find_images_cache_invalidates_on_change(tmp_path):
    """If a non-image file is replaced with a real image, cache adapts."""
    bad = tmp_path / "file.dat"
    bad.write_bytes(b"not an image")

    cache = EmbeddingCache(tmp_path / "test.db")
    assert find_images(str(tmp_path), cache=cache) == []
    assert cache.is_not_image(bad)

    # Replace the file with a real image
    Image.new("RGB", (1, 1)).save(bad, format="PNG")
    # Cache entry is stale (stat changed), so PIL re-checks
    images = find_images(str(tmp_path), cache=cache)
    assert len(images) == 1
    cache.close()


def test_find_images_uses_embedding_cache_for_known_images(tmp_path, monkeypatch):
    """Known cached images skip PIL verification on re-scan."""
    scan_dir = tmp_path / "images"
    scan_dir.mkdir()
    image_path = scan_dir / "real.jpg"
    Image.new("RGB", (1, 1)).save(image_path, format="JPEG")

    cache = EmbeddingCache(tmp_path / "test.db")
    cache.put(image_path, "model-a", np.ones((1, 4), dtype=np.float32))

    def _fail_open(*_args, **_kwargs):
        pytest.fail("Image.open should not be called for cached image hits")

    monkeypatch.setattr("grape.search.Image.open", _fail_open)
    images = find_images(str(scan_dir), cache=cache)
    assert [p.name for p in images] == ["real.jpg"]
    cache.close()


def test_find_images_real_filesystem_error_propagates(tmp_path):
    """OSError with an errno (e.g. permission denied) is not swallowed."""
    img = tmp_path / "locked.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
    img.chmod(0o000)
    try:
        with pytest.raises(PermissionError):
            find_images(str(tmp_path))
    finally:
        img.chmod(0o644)


def test_recursive_symlink_cycle_terminates(tmp_path):
    """Symlink cycles must not cause infinite traversal."""
    sub_a = tmp_path / "a"
    sub_b = tmp_path / "a" / "b"
    sub_a.mkdir()
    sub_b.mkdir()
    # Create a cycle: a/b/loop -> a
    (sub_b / "loop").symlink_to(sub_a)
    Image.new("RGB", (1, 1)).save(sub_a / "img.jpg", format="JPEG")

    images = find_images(str(tmp_path), recursive=True)
    # The image should appear exactly once, not infinitely.
    names = [p.name for p in images]
    assert names.count("img.jpg") == 1


def test_truncated_image_rejected(fixtures_dir):
    """Truncated JPEG (valid header, incomplete body) is not an image.

    Pillow's verify() only checks headers and lets these through.
    We use load() to catch them.  Regression test for files that pass
    the scan but fail encode_image, triggering an unnecessary model load.
    """
    images = find_images(str(fixtures_dir), recursive=False)
    names = {p.name for p in images}
    assert "truncated.jpg" not in names


def test_mp4_rejected(fixtures_dir):
    """Video files are not images."""
    images = find_images(str(fixtures_dir), recursive=False)
    names = {p.name for p in images}
    assert "not_an_image.mp4" not in names


def test_not_image_takes_priority_over_image_hit(tmp_path):
    """A file in both image_hits and not_image_hits is excluded.

    This happens when a file was once embedded successfully but later
    found to be broken (e.g. truncated).  not_images must win.
    """
    scan_dir = tmp_path / "images"
    scan_dir.mkdir()
    bad = scan_dir / "broken.jpg"
    bad.write_bytes(b"not a real image")

    cache = EmbeddingCache(tmp_path / "test.db")
    # Simulate: file has a stale embedding from a previous run.
    cache.put(bad, "old-model", np.ones((1, 4), dtype=np.float32))
    # And it was also recorded as not-image (e.g. by _score_all).
    cache.put_not_image(bad)

    records = list(iter_image_records(str(scan_dir), cache=cache))
    assert len(records) == 0, "not_image should take priority over image_hit"
    cache.close()


def test_score_image_prompt_ensemble_averages_templates(tmp_path):
    """Prompt ensembling should average text embeddings per keyword."""

    class DummyModel:
        def encode_texts(self, texts):
            table = {
                "a photo of a dog": [1.0, 0.0],
                "a close-up photo of a dog": [0.0, 1.0],
            }
            return np.array([table[t] for t in texts], dtype=np.float32)

        def encode_image(self, _):
            return np.array([[1.0, 0.0]], dtype=np.float32)

    image_path = tmp_path / "x.jpg"
    image_path.write_bytes(b"unused")
    result = score_image(
        DummyModel(),
        image_path,
        ["dog"],
        prompt_templates=["a photo of a {}", "a close-up photo of a {}"],
    )
    assert result.score == pytest.approx(2 ** -0.5, rel=1e-5)
