"""CLIP model wrapper for grape image search.

Public API:
    CLIPModel          -- encode images and text to embeddings
    resolve_model_id   -- stable model identifier without loading weights
    preload_weights    -- kick off background weight read for fast startup

This module contains some crazy hacks to recude latency.
"""

# The bottom half of this file is startup hacks. They are ugly but save ~2s
# of import/init time on every invocation.

import importlib
import os
import sys
import threading
import types
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

from grape.hf_cache import WEIGHT_FILENAMES as _WEIGHT_FILENAMES
from grape.hf_cache import cached_file_from_repo as _cached_file_from_repo

# Default to offline HF Hub access in this process to avoid unexpected
# network latency during model startup. This only applies when the caller
# did not already set `HF_HUB_OFFLINE` in the environment.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# Default model: ViT-B-16 provides stronger semantic quality than B-32.
# Embedding dimensionality remains 512.
DEFAULT_MODEL = "ViT-B-16"
DEFAULT_PRETRAINED = "laion2b_s34b_b88k"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class CLIPModel:
    """Wrapper around an open_clip model for encoding images and text."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        pretrained: str = DEFAULT_PRETRAINED,
        quiet: bool = False,
    ) -> None:
        self._model_name = model_name
        self._pretrained = pretrained
        self._model_id: str | None = None
        self.device = "cpu"
        open_clip = _import_open_clip(
            use_transformers=False, model_name=model_name,
        )
        if not quiet:
            print("Loading model...", end=" ", flush=True, file=sys.stderr)
        try:
            self._init_model(open_clip, model_name, pretrained)
        except (ImportError, AttributeError) as exc:
            if not _requires_transformers(exc):
                raise
            open_clip = _import_open_clip(use_transformers=True)
            self._init_model(open_clip, model_name, pretrained)
        self.model.eval()
        if not quiet:
            print("done.", file=sys.stderr)

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
        state_dict = _take_preloaded_state_dict()
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
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def _init_model_fast(
        self,
        open_clip: Any,
        model_name: str,
        pretrained: str,
        state_dict: dict,
    ) -> None:
        """Load model via meta device + pre-loaded state dict.

        See _MetaDeviceLoader docstring for the full explanation.
        """
        _MetaDeviceLoader.load_into(
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
    def encode_texts(self, texts: list[str]) -> NDArray[Any]:
        """Encode text strings to L2-normalized embeddings. Shape: (n, dim)."""
        tokens = self.tokenizer(texts).to(self.device)
        emb = self.model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        result: NDArray[Any] = emb.cpu().numpy().astype(np.float32)
        return result

    @torch.no_grad()
    def encode_image(self, image_path: str) -> NDArray[Any]:
        """Encode a single image to an L2-normalized embedding. Shape: (1, dim)."""
        image = Image.open(image_path).convert("RGB")
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        emb = self.model.encode_image(tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        result: NDArray[Any] = emb.cpu().numpy().astype(np.float32)
        return result


def get_hf_hub(model_name: str, pretrained: str) -> str:
    """Return the hf_hub repo string from open_clip's pretrained config.

    Returns empty string if the model has no hf_hub config.
    Requires open_clip to be imported.
    """
    try:
        open_clip = _import_open_clip(
            use_transformers=False, model_name=model_name,
        )
    except (ImportError, AttributeError) as exc:
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
#   3. Meta-device model init (~0.7s saved): construct the nn.Module on
#      torch.device('meta') (no tensor allocation, ~23ms) and then
#      assign pre-loaded safetensors weights via load_state_dict.
#
#   4. Background weight preload: start reading weights from disk in a
#      thread while the open_clip import is still happening, so the
#      state dict is warm by the time we need it.
#
# Each hack is brittle in a specific way (documented inline). When an
# assumption breaks, we fall back to the normal open_clip path -- slower
# but correct.
# ---------------------------------------------------------------------------

_open_clip_module = None
_open_clip_fast_path = False
_preloaded_state_dict: dict | None = None
_preload_thread: "threading.Thread | None" = None


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


def _model_needs_timm(model_name: str) -> bool:
    """Check whether a model architecture requires timm.

    Reads the JSON config from open_clip's installed model_configs directory
    without importing open_clip itself. Returns True (safe default) if the
    config can't be found.
    """
    oc_init = sys.modules.get("open_clip", None)
    if oc_init is not None and getattr(oc_init, "__file__", None) is not None:
        cfg_dir = Path(oc_init.__file__).parent / "model_configs"  # type: ignore[arg-type]
    else:
        import importlib.util
        spec = importlib.util.find_spec("open_clip")
        if spec is None or spec.origin is None:
            return True  # can't tell, assume yes (safe)
        cfg_dir = Path(spec.origin).parent / "model_configs"
    cfg_file = cfg_dir / f"{model_name}.json"
    if not cfg_file.is_file():
        return True  # unknown model, assume yes (safe)
    import json
    cfg = json.loads(cfg_file.read_text())
    vcfg = cfg.get("vision_cfg", {})
    return isinstance(vcfg, dict) and "timm_model_name" in vcfg


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
        )
        dynamo_utils_stub = _make_stub(
            "torch._dynamo.utils",
            "grape startup stub: deferred",
            is_compile_supported=lambda: False,
        )
        dynamo_stub.utils = dynamo_utils_stub  # type: ignore[attr-defined]
        sys.modules["torch._dynamo"] = dynamo_stub
        sys.modules["torch._dynamo.utils"] = dynamo_utils_stub


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
        _open_clip_module = importlib.import_module("open_clip")
        _open_clip_fast_path = False
        return _open_clip_module

    # Only stub timm when we know the model doesn't need it.
    stub_timm = model_name is not None and not _model_needs_timm(model_name)
    _install_import_stubs(stub_timm=stub_timm)
    _open_clip_module = importlib.import_module("open_clip")
    _open_clip_fast_path = True
    return _open_clip_module


def _requires_transformers(exc: Exception) -> bool:
    """Return True when open_clip failed due to missing transformers."""
    text = str(exc).lower()
    return "transformers" in text and "install" in text


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


def _temporary_hf_hub_offline():
    """Temporarily force huggingface_hub into offline mode."""
    return _temporary_env("HF_HUB_OFFLINE", "1")


# -- Hacks 3 & 4: Meta-device init + background weight preload ------------
#
# open_clip.create_model() spends ~700ms allocating random tensors that
# get immediately overwritten by load_checkpoint(). We avoid this by:
#
#   1. Reading the safetensors file in a background thread (preload_weights)
#      while the open_clip import is still happening.
#
#   2. Constructing the model on torch.device('meta') (~23ms, no real
#      tensors allocated) and then assigning the pre-loaded state dict
#      via load_state_dict(assign=True) -- tensors are moved, not copied.
#
# The only fixup: the causal attention mask buffer is created during
# CLIP.__init__ but not saved in checkpoints, so after meta-device
# construction it's still a meta tensor. We rebuild it manually.
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
    # Side effect: installs import stubs for open_clip.
    _import_open_clip(use_transformers=False, model_name=model_name)
    path = _cached_weight_path(model_name, pretrained)
    if path is None:
        return

    def _load():
        global _preloaded_state_dict
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file
            _preloaded_state_dict = load_file(path)
        else:
            _preloaded_state_dict = torch.load(
                path, map_location="cpu", weights_only=True,
            )

    _preload_thread = threading.Thread(target=_load, daemon=True)
    _preload_thread.start()


def _take_preloaded_state_dict() -> dict | None:
    """Wait for and consume the preloaded state dict, if any.

    Returns the state dict and clears the preload state so it can't be
    used twice. Returns None if no preload was started.
    """
    global _preload_thread, _preloaded_state_dict
    if _preload_thread is None:
        return None
    _preload_thread.join()
    sd = _preloaded_state_dict
    _preload_thread = None
    _preloaded_state_dict = None
    return sd


class _MetaDeviceLoader:
    """Fast model init: meta device + pre-loaded state dict.

    Constructs the model on torch.device('meta') (~23ms vs ~700ms on CPU)
    then assigns the pre-loaded tensors as parameters. See "Hacks 3 & 4"
    comment above for the full explanation.
    """

    @staticmethod
    def load_into(
        clip: CLIPModel,
        open_clip: Any,
        model_name: str,
        pretrained: str,
        state_dict: dict,
    ) -> None:
        # The pretrained config specifies model-specific normalization
        # (mean/std) and resize mode.  create_model(load_weights=False)
        # does NOT apply these, so we must pass them explicitly via
        # force_preprocess_cfg.  Without this, models that use non-default
        # normalization (e.g. mean/std = 0.5 instead of ImageNet defaults)
        # produce wrong embeddings.
        from open_clip.factory import merge_preprocess_kwargs
        pt_cfg = open_clip.get_pretrained_cfg(model_name, pretrained) or {}
        force_pp = merge_preprocess_kwargs(
            {},
            mean=pt_cfg.get("mean"),
            std=pt_cfg.get("std"),
            interpolation=pt_cfg.get("interpolation"),
            resize_mode=pt_cfg.get("resize_mode"),
        )
        with torch.device("meta"):
            model = open_clip.create_model(
                model_name,
                load_weights=False,
                device="meta",
                force_preprocess_cfg=force_pp,
            )
        model.load_state_dict(state_dict, assign=True, strict=True)
        # The causal attention mask is a non-persistent buffer created in
        # CLIP.__init__ but absent from checkpoints. Rebuild on CPU.
        if hasattr(model, "attn_mask"):
            ctx_len = model.context_length
            mask = torch.empty(ctx_len, ctx_len)
            mask.fill_(float("-inf"))
            mask.triu_(1)
            model.attn_mask = mask
        clip.model = model
        from open_clip.transform import PreprocessCfg, image_transform_v2
        pp_cfg = PreprocessCfg(**model.visual.preprocess_cfg)
        clip.preprocess = image_transform_v2(pp_cfg, is_train=False)
        clip.tokenizer = open_clip.get_tokenizer(model_name)
