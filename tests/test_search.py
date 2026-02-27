"""Tests for image discovery (no model needed)."""

import pytest
from PIL import Image

from grape.cache import EmbeddingCache
from grape.search import find_images


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
    # Real image, wrong extension — should be found
    Image.new("RGB", (1, 1)).save(tmp_path / "photo.dat", format="JPEG")
    # Text file with image extension — should be skipped
    (tmp_path / "fake.jpg").write_bytes(b"not an image at all")
    # Plain text — should be skipped
    (tmp_path / "notes.txt").write_bytes(b"hello world")

    images = find_images(str(tmp_path))
    names = {p.name for p in images}
    assert "photo.dat" in names
    assert "fake.jpg" not in names
    assert "notes.txt" not in names


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
