"""Fast unit tests for model-loading helpers."""

import errno
import os
import struct
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


def test_suppress_filter_drops_open_clip_4_factory_logger_message(caplog):
    """open_clip 4.x logs the warning on its own module logger."""
    import logging
    with caplog.at_level(logging.WARNING, logger=""):
        with _suppress_open_clip_no_weights_warning():
            logging.getLogger("open_clip.factory").warning(
                "No pretrained weights loaded for model 'X'."
                " Model initialized randomly.",
            )
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
    loggers = [logging.getLogger(), logging.getLogger("open_clip.factory")]
    before = [list(lg.filters) for lg in loggers]
    with _suppress_open_clip_no_weights_warning():
        for lg, b in zip(loggers, before):
            assert len(lg.filters) == len(b) + 1
    assert [lg.filters for lg in loggers] == before


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


# --- _load_frames: decode failures become errno-less OSError (issue #15) ---

class _FakeImage:
    """Context-manager stand-in for a Pillow image whose decode fails."""

    def __init__(self, exc):
        self._exc = exc

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    @property
    def n_frames(self):
        raise self._exc


@pytest.mark.parametrize("exc", [
    IndexError("index out of range"),
    struct.error("unpack requires a buffer of 2 bytes"),
    ValueError("tile cannot extend outside image"),
    EOFError("no more images in GIF file"),
])
def test_load_frames_wraps_pillow_decode_errors_as_oserror(monkeypatch, exc):
    from grape.model import _load_frames

    monkeypatch.setattr("grape.model.Image.open", lambda _p: _FakeImage(exc))
    with pytest.raises(OSError) as info:
        list(_load_frames("x.gif"))
    assert not isinstance(info.value, (IndexError, ValueError, EOFError, struct.error))
    assert info.value.errno is None
    assert "x.gif" in str(info.value)
    assert str(exc) in str(info.value)
    assert info.value.__cause__ is exc


def test_load_frames_wraps_struct_error_from_seek(monkeypatch):
    from grape.model import _load_frames

    class _SeekFails(_FakeImage):
        n_frames = 6

        def seek(self, idx):
            raise struct.error("unpack_from requires a buffer of at least 2 bytes")

    monkeypatch.setattr("grape.model.Image.open", lambda _p: _SeekFails(None))
    with pytest.raises(OSError) as info:
        list(_load_frames("x.gif"))
    assert info.value.errno is None
    assert "x.gif" in str(info.value)


def test_load_frames_propagates_filesystem_errors_unchanged(tmp_path):
    from grape.model import _load_frames

    missing = tmp_path / "missing.gif"
    with pytest.raises(FileNotFoundError) as info:
        list(_load_frames(str(missing)))
    assert info.value.errno == errno.ENOENT
    assert "cannot decode" not in str(info.value)


def test_load_frames_samples_static_and_animated(tmp_path):
    from PIL import Image

    from grape.model import MAX_ANIMATION_FRAMES, _load_frames

    static = tmp_path / "static.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(static, format="PNG")
    frames = list(_load_frames(str(static)))
    assert len(frames) == 1 and frames[0].mode == "RGB"

    n = MAX_ANIMATION_FRAMES + 4
    colors = [(int(255 * i / (n - 1)), 0, 0) for i in range(n)]
    gif = tmp_path / "anim.gif"
    imgs = [Image.new("RGB", (8, 8), c) for c in colors]
    imgs[0].save(gif, format="GIF", save_all=True, append_images=imgs[1:])
    frames = list(_load_frames(str(gif)))
    assert len(frames) == MAX_ANIMATION_FRAMES
    assert all(f.mode == "RGB" for f in frames)
    # First and last sampled frames are the first and last real frames.
    assert frames[0].getpixel((0, 0))[0] < 16
    assert frames[-1].getpixel((0, 0))[0] > 240


def test_load_frames_never_leaks_pillow_errors_for_truncated_gif(tmp_path):
    """Every truncation of an animated GIF either
    decodes or fails as an errno-less OSError -- never as Pillow's raw
    IndexError / struct.error / ValueError / EOFError (issue #15)."""
    from PIL import Image

    from grape.model import _load_frames

    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (0, 255, 255), (255, 0, 255),
    ]
    imgs = [Image.new("RGB", (64, 64), c) for c in colors]
    full = tmp_path / "full.gif"
    imgs[0].save(
        full, format="GIF", save_all=True, append_images=imgs[1:],
        duration=100, loop=0,
    )
    data = full.read_bytes()

    trunc = tmp_path / "trunc.gif"
    passed = raised = 0
    for n in range(int(len(data) * 0.6), len(data) + 1):
        trunc.write_bytes(data[:n])
        try:
            frames = list(_load_frames(str(trunc)))
        except OSError as e:
            assert e.errno is None, (n, e)
            assert not isinstance(
                e, (IndexError, ValueError, EOFError, struct.error)
            ), (n, e)
            raised += 1
        else:
            assert frames, n
            passed += 1
    assert passed > 0, "no truncation decoded"
    if raised == 0:
        pytest.skip("no truncation reproduced a decode error on this Pillow")


@pytest.mark.parametrize("cli, expected", [
    (True, ["stubbed", "plain"]),
    (False, ["plain"]),  # library use: never stubbed
])
def test_import_open_clip_falls_back_when_stubs_break_import(
    monkeypatch, cli, expected,
):
    """open_clip 4.x imports names the coca_model stub lacks; the stubbed
    import must fall back to a plain one instead of raising ImportError.
    Only the CLI installs the stubs."""
    import sys

    import grape.cli
    import grape.model as gm

    real = SimpleNamespace(__name__="open_clip")
    calls = []

    def fake_import(name):
        assert name == "open_clip"
        coca = sys.modules.get("open_clip.coca_model")
        stubbed = coca is not None and getattr(coca, "__file__", None) is None
        calls.append("stubbed" if stubbed else "plain")
        if stubbed:
            sys.modules["open_clip.mammut_model"] = SimpleNamespace()  # half-import
            raise ImportError("cannot import name 'MultimodalCfg'")
        return real

    # Isolate sys.modules and the module-level import state.
    monkeypatch.setattr(sys, "modules", dict(sys.modules))
    for name in list(sys.modules):
        if name == "open_clip" or name.startswith("open_clip."):
            del sys.modules[name]
    monkeypatch.setattr(gm, "_open_clip_module", None)
    monkeypatch.setattr(gm, "_open_clip_fast_path", False)
    monkeypatch.setattr(gm.importlib, "import_module", fake_import)
    monkeypatch.setattr(gm, "_model_needs_timm", lambda _name: False)
    if cli:
        grape.cli._stub_imports_HACK()

    assert gm._import_open_clip(use_transformers=False, model_name="X") is real
    assert calls == expected
    assert "open_clip.mammut_model" not in sys.modules
    assert gm._open_clip_fast_path is False


def test_load_frames_rejects_truncated_jpeg(fixtures_dir):
    """The scan lists a truncated JPEG like any file; decoding it must
    fail the way _score_all skips: an OSError with no errno."""
    from grape.model import _load_frames

    with pytest.raises(OSError) as info:
        list(_load_frames(str(fixtures_dir / "truncated.jpg")))
    assert info.value.errno is None


def test_preloaded_weights_go_only_to_the_model_they_were_read_for(monkeypatch):
    """Two checkpoints of one architecture both pass load_state_dict(strict=
    True), so a preload for another model must never be handed over."""
    import threading

    import grape.model as gm

    def preloaded(model_name, pretrained):
        done = threading.Thread(target=lambda: None)
        done.start()
        done.join()
        monkeypatch.setattr(gm, "_preload_thread", done)
        monkeypatch.setattr(
            gm, "_preloaded_state_dict", ((model_name, pretrained), {"w": 1}),
        )

    preloaded("ViT-X", "laion")
    assert gm._take_preloaded_state_dict("ViT-X", "openai") is None
    assert gm._preloaded_state_dict is None, "a mismatched preload is dropped"

    preloaded("ViT-X", "laion")
    assert gm._take_preloaded_state_dict("ViT-X", "laion") == {"w": 1}


def test_load_frames_decodes_heic(tmp_path):
    """iPhone photos: HEIC decodes through grape's path (pillow-heif)."""
    from PIL import Image

    from grape.model import _load_frames

    path = tmp_path / "photo.heic"
    Image.new("RGB", (64, 48), (200, 80, 40)).save(path, format="HEIF", quality=95)

    frames = list(_load_frames(str(path)))

    assert len(frames) == 1
    assert frames[0].mode == "RGB" and frames[0].size == (64, 48)
    pixel = frames[0].getpixel((32, 24))
    assert all(abs(a - b) <= 8 for a, b in zip(pixel, (200, 80, 40)))


# --- formats and color modes ---------------------------------------------

_COLOR = (200, 80, 40)


def _format_cases():
    """(filename, Pillow mode to save from, save kwargs, exact_color)."""
    from PIL import features

    cases = [
        ("baseline.jpg", "RGB", {"quality": 95}, False),
        ("progressive.jpg", "RGB", {"quality": 95, "progressive": True}, False),
        ("gray.jpg", "L", {"quality": 95}, False),
        ("cmyk.jpg", "CMYK", {"quality": 95}, False),
        ("rgb.png", "RGB", {}, True),
        ("rgba.png", "RGBA", {}, True),
        ("palette.png", "P", {}, False),
        ("gray.png", "L", {}, False),
        ("gray16.png", "I;16", {}, False),
        ("static.gif", "P", {}, False),
        ("lossy.webp", "RGB", {"quality": 90}, False),
        ("lossless.webp", "RGB", {"lossless": True}, True),
        ("alpha.webp", "RGBA", {"lossless": True}, True),
        ("rgb.tif", "RGB", {}, True),
        ("rgb.bmp", "RGB", {}, True),
        ("rgb.jp2", "RGB", {}, False),
    ]
    if features.check("avif"):
        cases.append(("rgb.avif", "RGB", {"quality": 90}, False))
    return cases


def _write_format_case(path, mode, kwargs):
    from PIL import Image

    img = Image.new("RGB", (64, 48), _COLOR)
    img = img.convert("L").convert("I;16") if mode == "I;16" else img.convert(mode)
    img.save(path, **kwargs)


@pytest.mark.parametrize(
    "name,mode,kwargs,exact", _format_cases(), ids=[c[0] for c in _format_cases()],
)
def test_load_frames_handles_format(tmp_path, name, mode, kwargs, exact):
    """Every format/mode grape may meet decodes to one RGB frame at full
    size; formats that keep color exactly return the same color."""
    from grape.model import _load_frames

    path = tmp_path / name
    _write_format_case(path, mode, kwargs)

    frames = list(_load_frames(str(path)))

    assert len(frames) == 1
    assert frames[0].mode == "RGB"
    assert frames[0].size == (64, 48)
    pixel = frames[0].getpixel((32, 24))
    if exact:
        assert pixel == _COLOR
    elif mode == "RGB":  # lossy: close to the original color
        assert all(abs(a - b) <= 12 for a, b in zip(pixel, _COLOR)), pixel


@pytest.mark.parametrize("name,kwargs", [
    ("anim.gif", {"duration": 100, "loop": 0}),
    ("anim.webp", {"duration": 100, "loop": 0, "lossless": True}),
    ("anim.png", {"duration": 100, "loop": 0}),
])
def test_load_frames_handles_animated_format(tmp_path, name, kwargs):
    """Animated GIF/WEBP/APNG: one independent RGB frame per sampled frame
    (the frames must not alias the reused seek() object)."""
    from PIL import Image

    from grape.model import _load_frames

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    imgs = [Image.new("RGB", (32, 32), c) for c in colors]
    path = tmp_path / name
    imgs[0].save(path, save_all=True, append_images=imgs[1:], **kwargs)

    frames = list(_load_frames(str(path)))

    assert len(frames) == 3
    assert all(f.mode == "RGB" for f in frames)
    got = [f.getpixel((16, 16)) for f in frames]
    for g, want in zip(got, colors):
        assert all(abs(a - b) <= 8 for a, b in zip(g, want)), got


def test_load_frames_skips_copy_for_single_frame_rgb(tmp_path, monkeypatch):
    """An RGB still is yielded as-is: convert() would copy it (~40 ms for
    a 12 MP photo) for an identical result."""
    from PIL import Image

    from grape.model import _load_frames

    path = tmp_path / "rgb.jpg"
    Image.new("RGB", (16, 16), _COLOR).save(path)
    calls = []
    monkeypatch.setattr(
        Image.Image, "convert",
        lambda self, *a, **k: calls.append(a) or Image.Image.copy(self),
    )
    frames = list(_load_frames(str(path)))
    assert len(frames) == 1 and frames[0].mode == "RGB"
    assert calls == []


def test_load_frames_drafts_large_jpegs_to_twice_the_size(tmp_path):
    """With a size, a JPEG decodes scaled down but keeps >= 2x that size;
    other formats decode at full size."""
    from PIL import Image

    from grape.model import _load_frames

    jpeg = tmp_path / "big.jpg"
    Image.new("RGB", (2000, 1500), (90, 120, 150)).save(jpeg, format="JPEG")
    # 1/2 scale keeps both sides >= 512; 1/4 (500x375) would not.
    assert next(_load_frames(str(jpeg), 256)).size == (1000, 750)
    assert next(_load_frames(str(jpeg))).size == (2000, 1500)

    png = tmp_path / "big.png"
    Image.new("RGB", (2000, 1500), (90, 120, 150)).save(png, format="PNG")
    assert next(_load_frames(str(png), 256)).size == (2000, 1500)


@pytest.mark.parametrize("cpus, threads", [(8, 6), (6, 4), (4, 3), (1, 1)])
def test_cpu_threads_leave_a_quarter_free(monkeypatch, cpus, threads):
    from grape.model import _cpu_threads

    monkeypatch.setattr(
        os, "sched_getaffinity", lambda _pid: set(range(cpus)), raising=False,
    )
    assert _cpu_threads() == threads
