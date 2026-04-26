"""Fast unit tests for model-loading helpers."""

import os
from types import SimpleNamespace

from grape.model import (
    _has_cached_weights,
    _suppress_open_clip_no_weights_warning,
    _temporary_env,
    _temporary_hf_hub_offline,
)


def test_has_cached_weights_false_without_hf_repo(monkeypatch):
    monkeypatch.setattr(
        "grape.model._import_open_clip",
        lambda **_kwargs: SimpleNamespace(
            get_pretrained_cfg=lambda *_args, **_kw: {},
        ),
    )
    assert _has_cached_weights("ViT-B-16", "laion2b_s34b_b88k") is False


def test_has_cached_weights_true_when_any_file_cached(monkeypatch):
    monkeypatch.setattr(
        "grape.model._import_open_clip",
        lambda **_kwargs: SimpleNamespace(
            get_pretrained_cfg=lambda *_args, **_kw: {"hf_hub": "repo/id"},
        ),
    )
    monkeypatch.setattr(
        "grape.model._cached_file_from_repo",
        lambda repo_id, filename: (
            "/tmp/cached-model.bin"
            if repo_id == "repo/id" and filename == "open_clip_pytorch_model.bin"
            else None
        ),
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
    with _temporary_hf_hub_offline():
        assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["HF_HUB_OFFLINE"] == "0"


# --- _suppress_open_clip_no_weights_warning ---

def test_suppress_filter_drops_target_message(caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger=""):
        with _suppress_open_clip_no_weights_warning():
            logging.warning("No pretrained weights loaded for model 'X'.")
    assert not any(
        "No pretrained weights loaded" in r.getMessage() for r in caplog.records
    )


def test_suppress_filter_keeps_other_root_warnings(caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger=""):
        with _suppress_open_clip_no_weights_warning():
            logging.warning("a different warning")
    assert any("a different warning" in r.getMessage() for r in caplog.records)


def test_suppress_filter_removed_after_context_exits():
    import logging
    root = logging.getLogger()
    before = list(root.filters)
    with _suppress_open_clip_no_weights_warning():
        assert len(root.filters) == len(before) + 1
    assert root.filters == before
