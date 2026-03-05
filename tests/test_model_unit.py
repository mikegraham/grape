"""Fast unit tests for model-loading helpers."""

import os

import grape.model as model_mod
from grape.model import (
    _has_cached_weights,
    _temporary_env,
    _temporary_hf_hub_offline,
)


def test_has_cached_weights_false_without_hf_repo(monkeypatch):
    monkeypatch.setattr(
        "grape.model.open_clip.get_pretrained_cfg",
        lambda *_args, **_kwargs: {},
    )
    assert _has_cached_weights("ViT-B-16", "laion2b_s34b_b88k") is False


def test_has_cached_weights_true_when_any_file_cached(monkeypatch):
    monkeypatch.setattr(
        "grape.model.open_clip.get_pretrained_cfg",
        lambda *_args, **_kwargs: {"hf_hub": "repo/id"},
    )

    def _fake_try_to_load_from_cache(repo_id, filename):
        if repo_id == "repo/id" and filename == "open_clip_pytorch_model.bin":
            return "/tmp/cached-model.bin"
        return None

    monkeypatch.setattr(
        "grape.model.try_to_load_from_cache",
        _fake_try_to_load_from_cache,
    )
    assert _has_cached_weights("ViT-B-16", "laion2b_s34b_b88k") is True


def test_temporary_env_sets_and_restores_missing_var(monkeypatch):
    monkeypatch.delenv("GRAPE_TEST_ENV", raising=False)
    with _temporary_env("GRAPE_TEST_ENV", "new"):
        assert os.environ["GRAPE_TEST_ENV"] == "new"
    assert "GRAPE_TEST_ENV" not in os.environ


def test_temporary_env_restores_existing_var(monkeypatch):
    monkeypatch.setenv("GRAPE_TEST_ENV", "old")
    with _temporary_env("GRAPE_TEST_ENV", "new"):
        assert os.environ["GRAPE_TEST_ENV"] == "new"
    assert os.environ["GRAPE_TEST_ENV"] == "old"


def test_temporary_hf_hub_offline_sets_and_restores(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setattr("grape.model.hf_constants.HF_HUB_OFFLINE", False)
    with _temporary_hf_hub_offline():
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert bool(model_mod.hf_constants.HF_HUB_OFFLINE)
    assert os.environ["HF_HUB_OFFLINE"] == "0"
    assert bool(model_mod.hf_constants.HF_HUB_OFFLINE) is False
