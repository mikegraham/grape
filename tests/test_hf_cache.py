"""Tests for HF Hub cache filesystem probing (no torch, no network)."""

import os
from pathlib import Path

import pytest

from grape import hf_cache


def _make_snapshot(
    root: Path,
    repo_id: str,
    commit: str,
    filename: str,
    *,
    write_ref: bool = True,
) -> Path:
    """Create ``models--org--repo/snapshots/<commit>/<filename>``."""
    repo_dir = root / f"models--{repo_id.replace('/', '--')}"
    snapshot = repo_dir / "snapshots" / commit
    snapshot.mkdir(parents=True, exist_ok=True)
    weight = snapshot / filename
    weight.write_bytes(b"weights")
    if write_ref:
        ref = repo_dir / "refs" / "main"
        ref.parent.mkdir(parents=True, exist_ok=True)
        ref.write_text(commit, encoding="utf-8")
    return weight


@pytest.fixture
def hub(tmp_path, monkeypatch):
    root = tmp_path / "hub"
    root.mkdir()
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(root))
    monkeypatch.delenv("HF_HOME", raising=False)
    return root


# --- hf_cache_root env resolution ---

def test_root_prefers_hub_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "explicit"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "ignored"))
    assert hf_cache.hf_cache_root() == tmp_path / "explicit"


def test_root_falls_back_to_hf_home(tmp_path, monkeypatch):
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hfhome"))
    assert hf_cache.hf_cache_root() == tmp_path / "hfhome" / "hub"


def test_root_default_location(monkeypatch):
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    assert hf_cache.hf_cache_root() == (
        Path.home() / ".cache" / "huggingface" / "hub"
    )


# --- cached_file_from_repo ---

def test_cached_file_uses_refs_main(hub):
    _make_snapshot(hub, "org/model", "abc123", "open_clip_model.safetensors")
    got = hf_cache.cached_file_from_repo(
        "org/model", "open_clip_model.safetensors"
    )
    assert got is not None
    assert got.endswith("snapshots/abc123/open_clip_model.safetensors")


def test_cached_file_missing_repo_returns_none(hub):
    assert hf_cache.cached_file_from_repo(
        "org/absent", "open_clip_model.safetensors"
    ) is None


def test_cached_file_newest_snapshot_when_no_ref(hub):
    older = _make_snapshot(
        hub, "org/model", "old", "open_clip_pytorch_model.bin",
        write_ref=False,
    )
    newer = _make_snapshot(
        hub, "org/model", "new", "open_clip_pytorch_model.bin",
        write_ref=False,
    )
    os.utime(older, ns=(1_000_000_000, 1_000_000_000))
    os.utime(newer, ns=(2_000_000_000, 2_000_000_000))
    got = hf_cache.cached_file_from_repo(
        "org/model", "open_clip_pytorch_model.bin"
    )
    assert got is not None
    assert "snapshots/new/" in got


# --- resolve_model_id / find_cached_weight ---

def test_resolve_model_id_appends_commit(hub):
    _make_snapshot(hub, "org/model", "deadbeef", "open_clip_model.safetensors")
    assert hf_cache.resolve_model_id("org/model") == "org/model@deadbeef"


def test_resolve_model_id_fallback_when_uncached(hub):
    assert hf_cache.resolve_model_id("org/uncached") == "org/uncached"


def test_find_cached_weight_hit_and_miss(hub):
    assert hf_cache.find_cached_weight("org/model") is None
    _make_snapshot(
        hub, "org/model", "c0ffee", "open_clip_pytorch_model.safetensors"
    )
    got = hf_cache.find_cached_weight("org/model")
    assert got is not None
    assert got.endswith("open_clip_pytorch_model.safetensors")
