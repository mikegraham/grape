"""CLIP model wrapper for grape image search.

Public API:
    CLIPModel          -- encode images and text to embeddings
    resolve_model_id   -- stable model identifier without loading weights
    preload_weights    -- kick off background weight read for fast startup

Uses OpenCLIP (Cherti et al., 2023) for model loading and inference.
https://arxiv.org/abs/2212.07143
https://github.com/mlfoundations/open_clip

This module contains some crazy hacks to reduce latency.
"""

# The bottom half of this file is startup hacks. They are ugly but save ~2s
# of import/init time on every invocation.

from __future__ import annotations

import importlib
import logging
import os
import struct
import sys
import threading
import types
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

from grape.hf_cache import WEIGHT_FILENAMES as _WEIGHT_FILENAMES
from grape.hf_cache import cached_file_from_repo as _cached_file_from_repo
from grape.search import ENCODE_BATCH_SIZE

# Cap on uniformly-sampled frames per animated GIF/WEBP/APNG. K=8 is
# the ViCLIP evaluation default (arXiv:2307.06942) and falls in the
# K=4..16 range used across CLIP-on-video work.
MAX_ANIMATION_FRAMES = 8

log = logging.getLogger("grape")

def pick_device() -> str:
    """Torch device to run on: $GRAPE_DEVICE, else Intel XPU, CUDA, CPU."""
    requested = os.environ.get("GRAPE_DEVICE")
    if requested:
        log.debug("device: %s (from $GRAPE_DEVICE)", requested)
        return requested
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        log.debug("device: xpu (%s)", torch.xpu.get_device_name(0))
        return "xpu"
    if torch.cuda.is_available():
        # ROCm builds report AMD GPUs here too.
        log.debug("device: cuda (%s)", torch.cuda.get_device_name(0))
        return "cuda"
    log.debug("device: cpu (torch %s sees no xpu or cuda)", torch.__version__)
    return "cpu"


def _cpu_threads() -> int:
    """Threads for torch on the CPU: three quarters of the usable CPUs.

    Every CPU (torch's default) was the slowest setting for a full run on
    both a hybrid 8-core laptop and a 24-core server: grape's image-decode
    threads need CPU at the same time.
    """
    cpus: int | None
    getaffinity = getattr(os, "sched_getaffinity", None)
    if getaffinity is not None:
        cpus = len(getaffinity(0))  # respects taskset and containers
    else:
        cpus = os.cpu_count()
    if cpus is None:  # the OS can't say
        return 1
    return max(1, cpus * 3 // 4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class CLIPModel:
    """Wrapper around an open_clip model for encoding images and text."""

    def __init__(
        self,
        model_name: str,
        pretrained: str,
        quiet: bool = False,
    ) -> None:
        self._model_name = model_name
        self._pretrained = pretrained
        self._model_id: str | None = None
        # Weights load on CPU (the fast path assigns a CPU state dict);
        # the model moves to self.device once built.
        self.device = "cpu"
        self._decode_pool: ThreadPoolExecutor | None = None
        open_clip = _import_open_clip(
            use_transformers=_model_needs_transformers(model_name),
            model_name=model_name,
        )
        if not quiet:
            print("Loading model...", end=" ", flush=True, file=sys.stderr)
        try:
            self._init_model(open_clip, model_name, pretrained)
        except (ImportError, RuntimeError) as exc:
            if not _requires_transformers(exc):
                raise
            open_clip = _import_open_clip(use_transformers=True)
            self._init_model(open_clip, model_name, pretrained)
        size = self.model.visual.preprocess_cfg["size"]
        self._image_size = max(size) if isinstance(size, (tuple, list)) else size
        self.device = pick_device()
        if self.device == "cpu":
            torch.set_num_threads(_cpu_threads())
        # On a GPU, store the weights in fp16: autocast converts them to
        # fp16 for every matmul anyway, so this halves GPU memory, the copy
        # and per-batch traffic (~6% faster encoding) for the same scores
        # within ~0.0005. The CPU has no fast fp16 and stays fp32.
        dtype = torch.float32 if self.device == "cpu" else torch.float16
        self.model.to(self.device, dtype=dtype).eval()
        log.debug(
            "model: %s/%s on %s, %s", model_name, pretrained, self.device,
            "fp32" if self.device == "cpu" else "fp16",
        )
        if not quiet:
            print("done.", file=sys.stderr)

    def _autocast(self) -> Any:
        """fp16 autocast off-CPU: matmuls run on tensor cores (Intel XMX,
        NVIDIA) at ~4x fp32 speed. Scores move by <0.001, well inside the
        noise between models; bf16 drifts ~10x more, so fp16 it is."""
        if self.device == "cpu":
            return nullcontext()
        return torch.autocast(self.device, dtype=torch.float16)

    def _init_model(
        self,
        open_clip: Any,
        model_name: str,
        pretrained: str,
    ) -> None:
        """Initialize model, tokenizer, and preprocess transform.

        Two paths:

        Fast path: if preload_weights() was called earlier and the state
        dict is ready, use _init_model_fast() which skips ~700ms of
        tensor allocation via torch.device('meta').

        Slow path: normal open_clip.create_model_and_transforms().
        """
        state_dict = _take_preloaded_state_dict(model_name, pretrained)
        if state_dict is not None:
            self._init_model_fast(open_clip, model_name, pretrained, state_dict)
            return

        cached = _cached_weight_path(model_name, pretrained)
        pretrained_ref = cached or pretrained
        if cached is not None:
            offline_ctx = _temporary_hf_hub_offline()
        else:
            offline_ctx = nullcontext()
        with offline_ctx:
            self.model, _, self.preprocess = (
                open_clip.create_model_and_transforms(
                    model_name,
                    pretrained=pretrained_ref,
                    device=self.device,
                )
            )
        self.tokenizer = (
            _take_preloaded_tokenizer(model_name)
            or _load_tokenizer(open_clip, model_name)
        )

    def _init_model_fast(
        self,
        open_clip: Any,
        model_name: str,
        pretrained: str,
        state_dict: dict,
    ) -> None:
        """Load model using a pre-read state dict.

        Skips open_clip's download/checkpoint logic by constructing the
        model with load_weights=False and assigning the pre-loaded tensors.
        """
        _init_from_state_dict(
            self, open_clip, model_name, pretrained, state_dict,
        )

    def model_id(self) -> str:
        """Stable identifier: ``{hf_repo_id}@{commit_hash}``.

        Falls back to ``{model_name}/{pretrained}`` when the HuggingFace
        repo or local cache isn't available.
        """
        if self._model_id is not None:
            return self._model_id
        self._model_id = resolve_model_id(
            self._model_name,
            self._pretrained,
        )
        return self._model_id

    def embed_dim(self) -> int:
        """Embedding dimensionality for this model architecture."""
        open_clip = _import_open_clip(
            use_transformers=False, model_name=self._model_name,
        )
        cfg = open_clip.get_model_config(self._model_name)
        dim: int = cfg["embed_dim"]
        return dim

    @torch.no_grad()
    def encode_texts(self, texts: list[str]) -> NDArray[np.float32]:
        """Encode text strings to L2-normalized embeddings. Shape: (n, dim)."""
        tokens = self.tokenizer(texts).to(self.device)
        with self._autocast():
            emb = self.model.encode_text(tokens).float()
        emb = emb / emb.norm(dim=-1, keepdim=True)
        result: NDArray[np.float32] = emb.cpu().numpy().astype(np.float32)
        return result

    def encode_image(self, image_path: str) -> NDArray[np.float32]:
        """Encode an image to an L2-normalized embedding. Shape: (1, dim).

        Animated GIF/WEBP/APNG: sample K = min(n_frames,
        MAX_ANIMATION_FRAMES) frames uniformly, L2-normalize each frame's
        embedding, average, re-normalize (CLIP4Clip's meanP,
        arXiv:2104.08860). Static images are the K=1 case.

        Known bias, left on purpose: averaging keeps what frames share but
        partly cancels what differs, so varied clips drift toward the
        embedding centroid and cluster together (within-gif cosine ~0.5 vs
        ~0.28 static, ViT-B-32). Mean-centering can't restore the lost
        detail and would shift static and text embeddings, invalidating
        the cache. Guarded by test_identical_frames_do_not_change_embedding
        and test_gif_frames_match_their_own_gif. Frame sampling lives in
        ``_load_frames``.

        Decode failures (e.g. a truncated animated file whose later frames
        Pillow cannot read) surface as an errno-less ``OSError`` so callers
        can skip the file like any other unreadable image.
        """
        result = self.encode_images([image_path])[0]
        if isinstance(result, BaseException):
            raise result
        return result

    @torch.no_grad()
    def encode_images(
        self, image_paths: Sequence[str],
    ) -> list[NDArray[np.float32] | OSError | SyntaxError]:
        """Encode many images, batched. Shape of each result: (1, dim).

        Same embeddings as ``encode_image``, but files decode in parallel
        threads and frames run through the model ENCODE_BATCH_SIZE at a
        time. An unreadable file comes back as its ``OSError`` /
        ``SyntaxError`` in place of an embedding, so one bad file doesn't
        sink the batch; anything else raises.
        """
        if self._decode_pool is None:
            self._decode_pool = ThreadPoolExecutor(
                max_workers=os.cpu_count() or 4,
                thread_name_prefix="grape-decode",
            )
        loaded = list(self._decode_pool.map(self._load_tensor, image_paths))
        ok = [t for t in loaded if isinstance(t, torch.Tensor)]
        if not ok:
            return loaded  # type: ignore[return-value]
        frames = torch.cat(ok)
        chunks = []
        with self._autocast():
            for i in range(0, len(frames), ENCODE_BATCH_SIZE):
                batch = frames[i:i + ENCODE_BATCH_SIZE].to(self.device)
                chunks.append(self.model.encode_image(batch).float())
        emb = torch.cat(chunks)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        means = torch.stack([
            seg.mean(dim=0) for seg in emb.split([len(t) for t in ok])
        ])
        means = means / means.norm(dim=-1, keepdim=True)
        rows = iter(means.cpu().numpy().astype(np.float32))
        return [
            next(rows)[None, :] if isinstance(t, torch.Tensor) else t
            for t in loaded
        ]

    def _load_tensor(self, image_path: str) -> torch.Tensor | OSError | SyntaxError:
        """Decode and preprocess one file's frames: (k, 3, H, W)."""
        try:
            return torch.stack([
                self.preprocess(f)
                for f in _load_frames(image_path, self._image_size)
            ])
        except (OSError, SyntaxError) as e:
            # PIL raises SyntaxError for some corrupt/unrecognized formats.
            return e


def _register_heif() -> None:
    from pillow_heif import register_heif_opener
    register_heif_opener()


# HEIC/HEIF (iPhone photos) isn't built into Pillow, and pillow_heif takes
# ~80 ms to import: a background thread, started alongside the weight read
# (preload_weights) or else by the first decode.
_heif_thread = threading.Thread(target=_register_heif, daemon=True)
_heif_lock = threading.Lock()


def start_heif_registration() -> None:
    with _heif_lock:
        if _heif_thread.ident is None:
            _heif_thread.start()


def _load_frames(
    image_path: str, size: int | None = None,
) -> Iterator[Image.Image]:
    """Open an image and yield its uniformly-sampled frames as RGB.

    Given *size*, large JPEGs are read at lower resolution, but never
    below twice *size*.

    Samples K = min(n_frames, MAX_ANIMATION_FRAMES) frame indices; a
    static image is the K=1 case. A generator so the caller preprocesses
    each frame before the next is decoded; collecting all K first would
    hold up to 8 full-size RGB copies of a large animation.

    The scan opens no files, so this is where any file first meets
    Pillow. A truncated animated GIF/WEBP/APNG can make Pillow raise
    ``IndexError``, ``struct.error``, ``ValueError`` or ``EOFError`` from
    ``n_frames``/``seek``/``convert`` here. Those are re-raised as an
    ``OSError`` with ``errno`` unset, the same shape PIL uses for an
    unrecognized file, so ``_score_all`` skips the file and records it as
    not an image. Real filesystem errors (errno set) pass through.
    """
    start_heif_registration()
    _heif_thread.join()
    try:
        with Image.open(image_path) as image:
            if size is not None:
                # Much faster for big photos. Going all the way down to
                # *size* blurs fine detail; twice *size* kept scores within
                # 0.004 in tests.
                image.draft("RGB", (2 * size, 2 * size))  # no-op for non-JPEGs
            n_frames = getattr(image, "n_frames", 1)
            k = min(n_frames, MAX_ANIMATION_FRAMES)
            # max(k-1, 1) keeps the formula well-defined when k == 1.
            divisor = max(k - 1, 1)
            indices = [round(i * (n_frames - 1) / divisor) for i in range(k)]
            for idx in indices:
                image.seek(idx)
                if n_frames == 1 and image.mode == "RGB":
                    # convert() would return a full copy (~40 ms for a
                    # 12 MP photo). Animations still copy: seek() reuses
                    # this object for every frame. load() here, not in the
                    # caller, so decode errors pass through the except below.
                    image.load()
                    yield image
                else:
                    yield image.convert("RGB")
    except (IndexError, ValueError, EOFError, struct.error) as e:
        raise OSError(f"cannot decode image file {image_path!r}: {e}") from e


def get_hf_hub(model_name: str, pretrained: str) -> str:
    """Return the hf_hub repo string from open_clip's pretrained config.

    Returns empty string if the model has no hf_hub config.
    Requires open_clip to be imported.
    """
    try:
        open_clip = _import_open_clip(
            use_transformers=False, model_name=model_name,
        )
    except ImportError as exc:
        if not _requires_transformers(exc):
            raise
        open_clip = _import_open_clip(
            use_transformers=True, model_name=model_name,
        )
    cfg = open_clip.get_pretrained_cfg(model_name, pretrained)
    hf_hub: str = (cfg or {}).get("hf_hub", "")
    return hf_hub.rstrip("/")


def resolve_model_id(
    model_name: str,
    pretrained: str,
) -> str:
    """Resolve a stable model id without loading model weights."""
    hf_hub = get_hf_hub(model_name, pretrained)
    if not hf_hub:
        return f"{model_name}/{pretrained}"
    from grape.hf_cache import resolve_model_id as _resolve_from_hf_hub
    return _resolve_from_hf_hub(hf_hub)


# ---------------------------------------------------------------------------
# Startup hacks
#
# Everything below exists to reduce startup latency. The normal open_clip
# path (import open_clip -> create_model -> load_checkpoint) takes ~3s on
# a cold start.  These hacks bring it to ~1s by:
#
#   1. Import stubs (~1.2s saved): inject fake modules into sys.modules
#      so that `import open_clip` skips heavy dependencies we never use
#      (transformers, CoCa, timm, torch._dynamo/sympy).
#
#   2. HF cache probing (~0.2s saved): read the HF Hub cache directory
#      layout directly instead of going through huggingface_hub APIs.
#
#   3. Background weight preload: start reading weights from disk in a
#      thread while the open_clip import is still happening, so the
#      state dict is warm by the time we need it.
#
#   4. Fast init from state dict: construct the model with
#      load_weights=False, then assign the pre-loaded state dict.
#      Skips open_clip's download/checkpoint logic entirely.
#
# Each hack is brittle in a specific way (documented inline). When an
# assumption breaks, we fall back to the normal open_clip path -- slower
# but correct.
# ---------------------------------------------------------------------------

_open_clip_module = None
_open_clip_fast_path = False
# Tagged with the (model_name, pretrained) it was read for, like the tokenizer.
_preloaded_state_dict: tuple[tuple[str, str], dict] | None = None
_preload_thread: "threading.Thread | None" = None
_tokenizer_thread: "threading.Thread | None" = None
_preloaded_tokenizer: Any | None = None


# -- Hack 1: Import stubs -------------------------------------------------
#
# open_clip eagerly imports modules we never use for plain CLIP inference:
#   - transformers (~0.5s) -- HF text towers, only needed for custom text
#   - open_clip.coca_model (~0.35s) -- CoCa architecture
#   - open_clip.timm_model (~0.25s) -- timm vision backbones
#   - torch._dynamo + sympy (~1.2s) -- pulled in by torchvision.ops;
#     we never call torch.compile so this is pure waste
#
# We inject stub modules into sys.modules before importing open_clip.
# The stubs are permanent for the process lifetime. If a model actually
# needs one of these (e.g. CoCa or a timm backbone), _import_open_clip
# purges them and reimports cleanly.

def _make_stub(name: str, doc: str, **attrs: Any) -> types.ModuleType:
    """Create a stub module with a docstring explaining why it exists."""
    stub = types.ModuleType(name, doc)
    for attr, val in attrs.items():
        setattr(stub, attr, val)
    return stub


def _builtin_model_config(model_name: str) -> dict | None:
    """Read open_clip's built-in JSON config without importing open_clip."""
    oc_init = sys.modules.get("open_clip", None)
    if oc_init is not None and getattr(oc_init, "__file__", None) is not None:
        cfg_dir = Path(oc_init.__file__).parent / "model_configs"  # type: ignore[arg-type]
    else:
        import importlib.util
        spec = importlib.util.find_spec("open_clip")
        if spec is None or spec.origin is None:
            return None
        cfg_dir = Path(spec.origin).parent / "model_configs"
    cfg_file = cfg_dir / f"{model_name}.json"
    if not cfg_file.is_file():
        return None
    import json
    # JSON is UTF-8 by spec; the locale default is cp1252 on Windows.
    cfg: dict = json.loads(cfg_file.read_text(encoding="utf-8"))
    return cfg


def _model_needs_timm(model_name: str) -> bool:
    """Check whether a model architecture requires timm."""
    cfg = _builtin_model_config(model_name)
    if cfg is None:
        return True  # can't tell, assume yes (safe)
    vcfg = cfg.get("vision_cfg", {})
    return isinstance(vcfg, dict) and "timm_model_name" in vcfg


def _model_needs_transformers(model_name: str) -> bool:
    """True when the model's text tower or tokenizer needs transformers."""
    cfg = _builtin_model_config(model_name)
    if cfg is None:
        return True  # unknown model, assume yes (safe)
    text_cfg = cfg.get("text_cfg", {})
    if not text_cfg.get("hf_tokenizer_name"):
        return False
    return _fast_tokenizer_dir(text_cfg) is None


def _fast_tokenizer_dir(text_cfg: dict) -> Path | None:
    """Cached snapshot dir, if _FastHFTokenizer can stand in for HFTokenizer."""
    if text_cfg.get("hf_model_name") or text_cfg.get("tokenizer_mode"):
        return None
    if set(text_cfg.get("tokenizer_kwargs", {})) - {"clean"}:
        return None
    repo = text_cfg["hf_tokenizer_name"]
    config = _cached_file_from_repo(repo, "tokenizer_config.json")
    if config is None or _cached_file_from_repo(repo, "tokenizer.json") is None:
        return None
    return Path(config).parent


class _FastHFTokenizer:
    """open_clip's HFTokenizer on the tokenizers library, without transformers.

    HFTokenizer only cleans the text and has transformers pad and truncate;
    importing transformers for that costs ~2.5s.
    """

    def __init__(self, snapshot: Path, context_length: int, clean: str) -> None:
        import json

        from open_clip.tokenizer import get_clean_fn
        from tokenizers import Tokenizer

        config = json.loads(  # UTF-8 by spec, not the locale's encoding
            (snapshot / "tokenizer_config.json").read_text(encoding="utf-8")
        )
        pad = config["pad_token"]
        pad = pad["content"] if isinstance(pad, dict) else pad
        self._clean = get_clean_fn(clean)
        self._tokenizer = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
        self._tokenizer.enable_truncation(context_length)
        self._tokenizer.enable_padding(
            direction=config.get("padding_side", "right"),
            pad_id=self._tokenizer.token_to_id(pad),
            pad_token=pad,
            length=context_length,
        )

    def __call__(self, texts: list[str]) -> torch.Tensor:
        encodings = self._tokenizer.encode_batch([self._clean(t) for t in texts])
        return torch.tensor([e.ids for e in encodings], dtype=torch.long)


def _install_import_stubs(*, stub_timm: bool = True) -> None:
    """Inject stub modules into sys.modules to skip heavy imports.

    Permanent for the process lifetime. See "Hack 1" comment above.
    """
    if "transformers" not in sys.modules:
        sys.modules["transformers"] = _make_stub(
            "transformers",
            "grape startup stub: blocks heavy transformers import",
        )
    if "open_clip.coca_model" not in sys.modules:
        sys.modules["open_clip.coca_model"] = _make_stub(
            "open_clip.coca_model",
            "grape startup stub: CoCa not needed for plain CLIP",
            CoCa="STUBBED by grape/model.py: CoCa not needed",
        )
    if stub_timm and "open_clip.timm_model" not in sys.modules:
        sys.modules["open_clip.timm_model"] = _make_stub(
            "open_clip.timm_model",
            "grape startup stub: timm backbone not needed for this model",
            TimmModel="STUBBED by grape/model.py: timm not needed",
        )
    # torch._dynamo stub must provide `disable` (used as a no-op decorator
    # by torch.compiler.disable and torch._compile.inner) and `utils` with
    # `is_compile_supported`. Since we never call torch.compile, `disable`
    # just returns the decorated function unchanged.
    #
    # The stub is tagged with _grape_stub=True so remove_dynamo_stubs() can
    # find and purge it when the caller actually wants torch.compile.
    if "torch._dynamo" not in sys.modules:

        def _dynamo_disable_noop(fn=None, recursive=True, **kwargs):
            """No-op: torch.compile is never used."""
            if fn is None:
                return _dynamo_disable_noop
            return fn

        dynamo_stub = _make_stub(
            "torch._dynamo",
            "grape startup stub: torch.compile not used",
            disable=_dynamo_disable_noop,
            _grape_stub=True,
        )
        dynamo_utils_stub = _make_stub(
            "torch._dynamo.utils",
            "grape startup stub: deferred",
            is_compile_supported=lambda: False,
            _grape_stub=True,
        )
        dynamo_stub.utils = dynamo_utils_stub  # type: ignore[attr-defined]
        sys.modules["torch._dynamo"] = dynamo_stub
        sys.modules["torch._dynamo.utils"] = dynamo_utils_stub


def remove_dynamo_stubs() -> None:
    """Remove grape's torch._dynamo stubs from sys.modules.

    Call this before torch.compile to let the real torch._dynamo load.
    The stubs are only installed to skip the ~620ms torchvision.ops import
    cost on startup; once open_clip is loaded they are no longer needed.
    Does nothing if the real module is already loaded.
    """
    for key in list(sys.modules):
        if (key == "torch._dynamo" or key.startswith("torch._dynamo.")) and (
            getattr(sys.modules[key], "_grape_stub", False)
        ):
            del sys.modules[key]


def _import_open_clip(
    *,
    use_transformers: bool,
    model_name: str | None = None,
):
    """Import open_clip, optionally with import stubs installed."""
    global _open_clip_fast_path, _open_clip_module
    if _open_clip_module is not None:
        if use_transformers and _open_clip_fast_path:
            # Previously imported with stubs; purge and reimport cleanly.
            for name in list(sys.modules):
                if name == "open_clip" or name.startswith("open_clip."):
                    del sys.modules[name]
            _open_clip_module = None
        else:
            return _open_clip_module

    if use_transformers:
        # Purge any stubs we installed so the real modules load.
        for prefix in ("transformers", "open_clip.coca_model",
                        "open_clip.timm_model"):
            for key in list(sys.modules):
                mod = sys.modules[key]
                if (key == prefix or key.startswith(prefix + ".")) and (
                    getattr(mod, "__file__", None) is None
                ):
                    del sys.modules[key]
        remove_dynamo_stubs()  # transformers' lazy imports trip on it
        _open_clip_module = importlib.import_module("open_clip")
        _open_clip_fast_path = False
        return _open_clip_module

    # Only stub timm when we know the model doesn't need it.
    stub_timm = model_name is not None and not _model_needs_timm(model_name)
    _install_import_stubs(stub_timm=stub_timm)
    try:
        _open_clip_module = importlib.import_module("open_clip")
    except ImportError as exc:
        # open_clip's own imports outgrew the stubs (4.x's mammut_model
        # pulls names from coca_model at import time). Drop the
        # half-imported package and take the slow, stub-free path.
        log.debug("open_clip import failed with stubs (%s); retrying without", exc)
        for name in list(sys.modules):
            if name == "open_clip" or name.startswith("open_clip."):
                del sys.modules[name]
        return _import_open_clip(use_transformers=True)
    _open_clip_fast_path = True
    return _open_clip_module


def _requires_transformers(exc: Exception) -> bool:
    """Return True when open_clip failed due to missing transformers."""
    text = str(exc).lower()
    if "transformers" not in text:
        return False
    if "install" in text:
        return True
    # Against our stub, HF tokenizers fail with "cannot import name", not "install".
    stub = sys.modules.get("transformers")
    return stub is not None and getattr(stub, "__file__", None) is None


# -- Hack 2: HF cache probing ---------------------------------------------
#
# huggingface_hub's Python API adds ~0.2s of overhead to locate a cached
# file. We probe the cache directory layout directly: the structure is
# models--{org}--{repo}/snapshots/{commit_hash}/{filename}. If the layout
# changes, _cached_file_from_repo returns None and callers fall back to
# normal open_clip resolution.



def _has_cached_weights(model_name: str, pretrained: str) -> bool:
    """Return True if this model has local HF weight files cached."""
    return _cached_weight_path(model_name, pretrained) is not None


def _cached_weight_path(model_name: str, pretrained: str) -> str | None:
    """Return a local cached checkpoint path for this model when available."""
    open_clip = _import_open_clip(
        use_transformers=False, model_name=model_name,
    )
    cfg = open_clip.get_pretrained_cfg(model_name, pretrained) or {}
    hf_hub = str(cfg.get("hf_hub", "")).rstrip("/")
    if not hf_hub:
        return None
    for filename in _WEIGHT_FILENAMES:
        cached = _cached_file_from_repo(hf_hub, filename)
        if cached is not None:
            return cached
    return None


@contextmanager
def _temporary_env(name: str, value: str):
    """Temporarily set an environment variable."""
    had_value = name in os.environ
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if had_value:
            assert previous is not None
            os.environ[name] = previous
        else:
            os.environ.pop(name, None)


def _load_tokenizer(open_clip: Any, model_name: str) -> Any:
    """HF tokenizers probe the Hub even when cached; load from the snapshot dir."""
    cfg = open_clip.get_model_config(model_name) or {}
    text_cfg = cfg.get("text_cfg", {})
    context_length = text_cfg.get(
        "context_length", open_clip.tokenizer.DEFAULT_CONTEXT_LENGTH,
    )
    fast = _fast_tokenizer_dir(text_cfg) if "hf_tokenizer_name" in text_cfg else None
    if fast is not None:
        clean = text_cfg.get("tokenizer_kwargs", {}).get("clean", "whitespace")
        return _FastHFTokenizer(fast, context_length, clean)
    repo = text_cfg.get("hf_tokenizer_name")
    local = _cached_file_from_repo(repo, "tokenizer_config.json") if repo else None
    if local is None:
        return open_clip.get_tokenizer(model_name)  # built-in tokenizer, or first run
    return open_clip.tokenizer.HFTokenizer(
        str(Path(local).parent),
        context_length=context_length,
        tokenizer_mode=text_cfg.get("tokenizer_mode"),
        **text_cfg.get("tokenizer_kwargs", {}),
    )


def _temporary_hf_hub_offline():
    """Temporarily force huggingface_hub into offline mode."""
    return _temporary_env("HF_HUB_OFFLINE", "1")


@contextmanager
def _suppress_open_clip_no_weights_warning():
    """Suppress open_clip's stale "no pretrained weights" warning during fast-init.

    open_clip emits this warning at WARNING level when called with
    ``load_weights=False``: on the root logger in 3.x
    (``logging.warning(...)``), on its ``open_clip.factory`` logger in
    4.x. In our fast-init path the warning is technically correct at the
    moment it fires but stale a few lines later, after we
    ``load_state_dict`` from preloaded weights.

    A logger's filters only see records created on that logger, not ones
    propagated from children, so the filter goes on both loggers. It is
    narrow on three axes -- logger name, level, message substring -- and
    is installed only for the duration of the wrapped call, so it cannot
    accidentally drop unrelated warnings.
    """
    import logging

    names = ("root", "open_clip.factory")

    class _Filter(logging.Filter):
        def filter(self, record):
            return not (
                record.name in names
                and record.levelno == logging.WARNING
                and "No pretrained weights loaded" in record.getMessage()
            )

    f = _Filter()
    loggers = [logging.getLogger(), logging.getLogger("open_clip.factory")]
    for lg in loggers:
        lg.addFilter(f)
    try:
        yield
    finally:
        for lg in loggers:
            lg.removeFilter(f)


# -- Hacks 3 & 4: Background weight preload + fast init -------------------
#
# open_clip's normal path downloads + loads weights synchronously inside
# create_model_and_transforms().  We split this into two cheaper steps:
#
#   1. Reading the safetensors file in a background thread (preload_weights)
#      while the open_clip import is still happening.
#
#   2. Constructing the model on CPU with load_weights=False, then
#      assigning the pre-loaded state dict via load_state_dict(assign=True).
#      This skips open_clip's download/checkpoint logic entirely.
#
# Float16 checkpoints (e.g. EVA02-L-14) are upcast to float32 before
# assignment, since CPU inference expects float32 inputs.
#
# If the safetensors file isn't cached locally, or preload_weights wasn't
# called, CLIPModel falls back to the normal open_clip path.

def preload_weights(model_name: str, pretrained: str) -> None:
    """Start reading model weights from disk in a background thread.

    Call this early (e.g. during imports) so the state dict is warm by
    the time CLIPModel.__init__ runs. Only works when weights are
    locally cached in the HF hub directory.
    """
    global _preload_thread, _preloaded_state_dict
    if _preload_thread is not None:
        return  # already started
    # Side effect: installs import stubs for open_clip (unless the model
    # needs the real transformers).
    needs_tf = _model_needs_transformers(model_name)
    open_clip = _import_open_clip(use_transformers=needs_tf, model_name=model_name)
    cfg = _builtin_model_config(model_name) or {}
    if cfg.get("text_cfg", {}).get("hf_tokenizer_name"):
        _preload_tokenizer(open_clip, model_name)
    path = _cached_weight_path(model_name, pretrained)
    if path is None:
        return

    def _load():
        global _preloaded_state_dict
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(path)
        else:
            state_dict = torch.load(path, map_location="cpu", weights_only=True)
        _preloaded_state_dict = ((model_name, pretrained), state_dict)

    _preload_thread = threading.Thread(target=_load, daemon=True)
    _preload_thread.start()
    start_heif_registration()


def _preload_tokenizer(open_clip: Any, model_name: str) -> None:
    """Import transformers and build the tokenizer while weights are read."""
    global _tokenizer_thread, _preloaded_tokenizer
    if _tokenizer_thread is not None:
        return

    def _load():
        global _preloaded_tokenizer
        # Tagged with the model it belongs to. A load that overran the
        # take() timeout stays tracked, so without the tag a later,
        # different model would be handed this tokenizer and would
        # silently embed with the wrong vocab.
        _preloaded_tokenizer = (model_name, _load_tokenizer(open_clip, model_name))

    _tokenizer_thread = threading.Thread(target=_load, daemon=True)
    _tokenizer_thread.start()


def _take_preloaded_tokenizer(model_name: str) -> Any | None:
    """Consume the preloaded tokenizer, but only if it is for *model_name*."""
    global _tokenizer_thread, _preloaded_tokenizer
    if _tokenizer_thread is None:
        return None
    _tokenizer_thread.join(timeout=30)
    if _tokenizer_thread.is_alive():
        return None
    loaded = _preloaded_tokenizer
    _tokenizer_thread = None
    _preloaded_tokenizer = None
    if loaded is None:
        return None
    loaded_for, tokenizer = loaded
    return tokenizer if loaded_for == model_name else None


def _take_preloaded_state_dict(model_name: str, pretrained: str) -> dict | None:
    """Wait for and consume the preloaded state dict, if it is for this model.

    Clears the preload state so it can't be used twice. Returns None if no
    preload was started, or if it was for another model or checkpoint:
    load_state_dict(strict=True) would accept any checkpoint of the same
    architecture, silently giving the wrong weights.
    """
    global _preload_thread, _preloaded_state_dict
    if _preload_thread is None:
        return None
    _preload_thread.join(timeout=30)
    if _preload_thread.is_alive():
        # Timed out waiting for background weight read; fall back to
        # normal open_clip loading.
        return None
    loaded = _preloaded_state_dict
    _preload_thread = None
    _preloaded_state_dict = None
    if loaded is None:
        return None
    loaded_for, state_dict = loaded
    return state_dict if loaded_for == (model_name, pretrained) else None


_INIT_PATCH_LOCK = threading.RLock()


@contextmanager
def _no_parameter_init():
    """Make ``torch.nn.init.*`` a no-op for the duration of a model build.

    Every parameter is overwritten by the ``strict=True`` load_state_dict
    that follows, so the random values open_clip generates are dead on
    arrival -- and generating them for an L-size model costs ~1.7s.

    Only ``nn.init`` is patched, never Tensor methods like ``fill_`` or
    ``zero_``: module code uses those to construct real buffers (the
    causal attention mask is ``empty(77,77).fill_(-inf).triu_(1)``), and
    stubbing them would silently produce a garbage mask.  Guarded by
    test_no_parameter_init_stubs_nn_init_but_not_tensor_methods.

    Serialized on a lock because it mutates a global module: two
    overlapping builds would otherwise have the second save the *stub* as
    its original and restore that, leaving torch.nn.init permanently
    no-oped for the whole process. Reentrant so a nested build still
    works.
    """
    import torch.nn.init as init
    with _INIT_PATCH_LOCK:
        names = [
            n for n in dir(init) if n.endswith("_") and not n.startswith("_")
        ]
        saved = {n: getattr(init, n) for n in names}

        def _noop(tensor, *args, **kwargs):
            return tensor

        try:
            for n in names:
                setattr(init, n, _noop)
            yield
        finally:
            for n, fn in saved.items():
                setattr(init, n, fn)


def _init_from_state_dict(
    clip: CLIPModel,
    open_clip: Any,
    model_name: str,
    pretrained: str,
    state_dict: dict,
) -> None:
    """Build model on CPU and load a pre-read state dict.

    The pretrained config specifies model-specific normalization
    (mean/std) and resize mode.  create_model(load_weights=False)
    does NOT apply these, so we must pass them explicitly via
    force_preprocess_cfg.  Without this, models that use non-default
    normalization (e.g. mean/std = 0.5 instead of ImageNet defaults)
    produce wrong embeddings.
    """
    from open_clip.factory import merge_preprocess_kwargs
    pt_cfg = open_clip.get_pretrained_cfg(model_name, pretrained) or {}
    force_pp = merge_preprocess_kwargs(
        {},
        mean=pt_cfg.get("mean"),
        std=pt_cfg.get("std"),
        interpolation=pt_cfg.get("interpolation"),
        resize_mode=pt_cfg.get("resize_mode"),
    )
    def _build(device: str) -> Any:
        with _suppress_open_clip_no_weights_warning():
            return open_clip.create_model(
                model_name,
                load_weights=False,
                device=device,
                force_preprocess_cfg=force_pp,
            )

    # Build under meta: every parameter is overwritten by the state dict
    # below, so the uniform_/normal_ init open_clip runs is wasted work
    # (~6s on an L-size model). device="meta" also makes open_clip's
    # closing .to(device) a no-op; without it that call raises.
    with torch.device("meta"):
        model = _build("meta")
    # Some checkpoints store weights in float16 (e.g. EVA02-L-14).
    # Convert to float32 before loading so all parameters match the
    # float32 input tensors produced by the image preprocessor.
    state_dict = {
        k: v.float() if v.is_floating_point() else v
        for k, v in state_dict.items()
    }
    model.load_state_dict(state_dict, assign=True, strict=True)
    # A tensor the checkpoint doesn't supply (e.g. a non-persistent buffer)
    # would still be meta here, and would fail at forward time. Rebuild.
    if any(t.is_meta for t in [*model.parameters(), *model.buffers()]):
        # Rebuild on CPU, skipping parameter init. EVA02-L-14 leaves just
        # two non-persistent buffers on meta (rope.pos_embed, attn_mask,
        # 38k elements total), and regenerating 428M random parameters to
        # recover them cost ~1.7s of a ~6s run.
        with _no_parameter_init():
            model = _build("cpu")
        model.load_state_dict(state_dict, assign=True, strict=True)
    clip.model = model
    from open_clip.transform import PreprocessCfg, image_transform_v2
    pp_cfg = PreprocessCfg(**model.visual.preprocess_cfg)
    clip.preprocess = image_transform_v2(pp_cfg, is_train=False)
    clip.tokenizer = (
        _take_preloaded_tokenizer(model_name)
        or _load_tokenizer(open_clip, model_name)
    )
