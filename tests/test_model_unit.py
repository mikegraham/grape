"""Fast unit tests for model-loading helpers."""

import os
import time
from types import SimpleNamespace

import pytest

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


def test_no_parameter_init_stubs_nn_init_but_not_tensor_methods():
    """nn.init is stubbed; Tensor methods that build real buffers are not.

    Module code builds the causal attention mask with
    ``empty(77, 77).fill_(-inf).triu_(1)``.  Stubbing ``fill_`` would
    silently yield a garbage mask, so the context manager must leave
    Tensor methods alone.
    """
    import torch
    import torch.nn.init as init

    from grape.model import _no_parameter_init

    with _no_parameter_init():
        param = torch.zeros(4)
        assert init.normal_(param).equal(torch.zeros(4))
        assert torch.empty(3).fill_(2.5).equal(torch.full((3,), 2.5))
        assert torch.empty(3).zero_().equal(torch.zeros(3))


def test_no_parameter_init_restores_on_exception():
    import torch
    import torch.nn.init as init

    from grape.model import _no_parameter_init

    before = init.normal_
    with pytest.raises(RuntimeError):
        with _no_parameter_init():
            raise RuntimeError("boom")
    assert init.normal_ is before
    assert not init.normal_(torch.zeros(64)).equal(torch.zeros(64))


def test_no_parameter_init_serializes_overlapping_builds():
    """Overlapping builds must never be inside the patch at the same time.

    It mutates a global module: if a second thread entered while the
    first held the patch, it would save the *stub* as its original and
    restore that, leaving torch.nn.init no-oped process-wide.
    """
    import threading

    import torch
    import torch.nn.init as init

    from grape.model import _no_parameter_init

    real = init.normal_
    state = {"inside": 0, "peak": 0}
    guard = threading.Lock()

    def worker():
        with _no_parameter_init():
            with guard:
                state["inside"] += 1
                state["peak"] = max(state["peak"], state["inside"])
            time.sleep(0.05)
            with guard:
                state["inside"] -= 1

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert not any(t.is_alive() for t in threads), "deadlock"

    assert state["peak"] == 1, "two builds were inside the patch at once"
    assert init.normal_ is real
    assert not init.normal_(torch.zeros(32)).equal(torch.zeros(32))


def test_no_parameter_init_nests():
    import torch.nn.init as init

    from grape.model import _no_parameter_init

    real = init.normal_
    with _no_parameter_init():
        with _no_parameter_init():
            pass
    assert init.normal_ is real


def test_preloaded_tokenizer_not_reused_across_models(monkeypatch):
    """A preload that overran its timeout must not be handed to another model.

    The thread stays tracked after a timeout, so without keying on the
    model name the next (different) model receives this tokenizer and
    embeds with the wrong vocab -- silently, since tokenizers are
    duck-typed and nothing validates the pairing.
    """
    import threading

    import grape.model as gm

    slow = threading.Event()

    def fake_load(_open_clip, model_name):
        if model_name == "ModelA":
            slow.wait(timeout=5)
        return f"tokenizer-for-{model_name}"

    monkeypatch.setattr(gm, "_load_tokenizer", fake_load)
    monkeypatch.setattr(gm, "_tokenizer_thread", None)
    monkeypatch.setattr(gm, "_preloaded_tokenizer", None)

    gm._preload_tokenizer(None, "ModelA")
    gm._tokenizer_thread.join(timeout=0.05)
    assert gm._tokenizer_thread.is_alive(), "expected the A load to overrun"
    slow.set()
    gm._tokenizer_thread.join(timeout=5)

    gm._preload_tokenizer(None, "ModelB")
    assert gm._take_preloaded_tokenizer("ModelB") is None

    monkeypatch.setattr(gm, "_tokenizer_thread", None)
    monkeypatch.setattr(gm, "_preloaded_tokenizer", None)


def test_preloaded_tokenizer_returned_for_matching_model(monkeypatch):
    import grape.model as gm

    monkeypatch.setattr(gm, "_load_tokenizer", lambda _oc, name: f"tok-{name}")
    monkeypatch.setattr(gm, "_tokenizer_thread", None)
    monkeypatch.setattr(gm, "_preloaded_tokenizer", None)
    gm._preload_tokenizer(None, "ModelA")
    assert gm._take_preloaded_tokenizer("ModelA") == "tok-ModelA"


def _tiny_hf_tokenizer(snapshot):
    """A tokenizer.json + tokenizer_config.json shaped like SigLIP's."""
    import json

    from tokenizers import Tokenizer, models, pre_tokenizers, processors

    words = "a photo of the dog cat".split()
    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2}
    vocab.update({w: i + 3 for i, w in enumerate(words)})
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.TemplateProcessing(
        single="$A <eos>", special_tokens=[("<eos>", 1)],
    )
    tok.save(str(snapshot / "tokenizer.json"))
    (snapshot / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast",
        "pad_token": "<pad>", "eos_token": "<eos>", "unk_token": "<unk>",
    }))


def test_fast_hf_tokenizer_matches_open_clip_hf_tokenizer(tmp_path):
    """Same ids as open_clip's HFTokenizer: cleaning, <eos>, padding, truncation."""
    import torch

    import grape.model as gm

    _tiny_hf_tokenizer(tmp_path)
    texts = ["a photo of a dog", "A  Photo of THE cat!", "", "dog " * 20]
    open_clip = gm._import_open_clip(use_transformers=True)
    ref = open_clip.tokenizer.HFTokenizer(
        str(tmp_path), context_length=8, clean="canonicalize",
    )
    fast = gm._FastHFTokenizer(tmp_path, 8, "canonicalize")
    assert torch.equal(fast(texts), ref(texts))


def test_model_needs_transformers_only_without_fast_tokenizer(monkeypatch):
    import grape.model as gm

    cached = {"tokenizer.json", "tokenizer_config.json"}
    monkeypatch.setattr(
        gm, "_cached_file_from_repo",
        lambda repo, fn: f"/snap/{fn}" if fn in cached else None,
    )

    def needs(text_cfg):
        monkeypatch.setattr(
            gm, "_builtin_model_config", lambda _name: {"text_cfg": text_cfg},
        )
        return gm._model_needs_transformers("m")

    siglip = {"hf_tokenizer_name": "org/r", "tokenizer_kwargs": {"clean": "x"}}
    assert needs({}) is False
    assert needs(siglip) is False
    assert needs({**siglip, "hf_model_name": "roberta"}) is True
    assert needs({**siglip, "tokenizer_kwargs": {"strip_sep_token": True}}) is True
    cached.discard("tokenizer.json")
    assert needs(siglip) is True
