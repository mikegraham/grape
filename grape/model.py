import importlib
import os
import sys
import types
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import constants as hf_constants, try_to_load_from_cache
from numpy.typing import NDArray
from PIL import Image

# Default model: ViT-B-16 provides stronger semantic quality than B-32.
# Embedding dimensionality remains 512.
DEFAULT_MODEL = "ViT-B-16"
DEFAULT_PRETRAINED = "laion2b_s34b_b88k"
_WEIGHT_FILENAMES = (
    "open_clip_model.safetensors",
    "open_clip_pytorch_model.safetensors",
    "open_clip_pytorch_model.bin",
)
_open_clip_module = None
_open_clip_fast_path = False


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


def _has_cached_weights(model_name: str, pretrained: str) -> bool:
    """Return True if this model has local HF weight files cached."""
    open_clip = _import_open_clip(use_transformers=False)
    cfg = open_clip.get_pretrained_cfg(model_name, pretrained) or {}
    hf_hub = str(cfg.get("hf_hub", "")).rstrip("/")
    if not hf_hub:
        return False
    for filename in _WEIGHT_FILENAMES:
        cached = try_to_load_from_cache(hf_hub, filename)
        if cached and isinstance(cached, str):
            return True
    return False


@contextmanager
def _temporary_hf_hub_offline():
    """Temporarily force huggingface_hub into offline mode."""
    previous_offline = hf_constants.HF_HUB_OFFLINE
    with _temporary_env("HF_HUB_OFFLINE", "1"):
        hf_constants.HF_HUB_OFFLINE = True
        try:
            yield
        finally:
            hf_constants.HF_HUB_OFFLINE = previous_offline


def _purge_open_clip_modules() -> None:
    """Drop loaded open_clip modules so we can re-import with new policy."""
    for name in list(sys.modules):
        if name == "open_clip" or name.startswith("open_clip."):
            del sys.modules[name]


@contextmanager
def _temporary_transformers_stub():
    """Temporarily make `import transformers` fail fast during open_clip import."""
    saved_transformers: dict[str, Any] = {}
    for key in list(sys.modules):
        if key == "transformers" or key.startswith("transformers."):
            saved_transformers[key] = sys.modules.pop(key)
    sys.modules["transformers"] = types.ModuleType("transformers")
    try:
        yield
    finally:
        for key in list(sys.modules):
            if key == "transformers" or key.startswith("transformers."):
                del sys.modules[key]
        sys.modules.update(saved_transformers)


def _import_open_clip(*, use_transformers: bool):
    """Import open_clip, optionally skipping transformers-heavy code paths."""
    global _open_clip_fast_path, _open_clip_module
    if _open_clip_module is not None:
        if use_transformers and _open_clip_fast_path:
            _purge_open_clip_modules()
            _open_clip_module = None
        else:
            return _open_clip_module

    if use_transformers:
        _open_clip_module = importlib.import_module("open_clip")
        _open_clip_fast_path = False
        return _open_clip_module

    with _temporary_transformers_stub():
        _open_clip_module = importlib.import_module("open_clip")
    _open_clip_fast_path = True
    return _open_clip_module


def _requires_transformers(exc: Exception) -> bool:
    """Return True when open_clip failed due to missing transformers."""
    text = str(exc).lower()
    return "transformers" in text and "install" in text


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
        open_clip = _import_open_clip(use_transformers=False)
        if not quiet:
            print("Loading model...", end=" ", flush=True, file=sys.stderr)
        offline_ctx = (
            _temporary_hf_hub_offline()
            if _has_cached_weights(model_name, pretrained)
            else nullcontext()
        )
        try:
            with offline_ctx:
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    model_name, pretrained=pretrained, device=self.device
                )
            self.tokenizer = open_clip.get_tokenizer(model_name)
        except Exception as e:
            if not _requires_transformers(e):
                raise
            open_clip = _import_open_clip(use_transformers=True)
            with offline_ctx:
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    model_name, pretrained=pretrained, device=self.device
                )
            self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        if not quiet:
            print("done.", file=sys.stderr)

    def model_id(self) -> str:
        """Stable identifier: ``{hf_repo_id}@{commit_hash}``.

        Falls back to ``{model_name}/{pretrained}`` when the HuggingFace
        repo or local cache isn't available.
        """
        if self._model_id is not None:
            return self._model_id
        open_clip = _import_open_clip(use_transformers=False)
        cfg = open_clip.get_pretrained_cfg(
            self._model_name, self._pretrained,
        )
        hf_hub: str = (cfg or {}).get("hf_hub", "").rstrip("/")
        if not hf_hub:
            self._model_id = (
                f"{self._model_name}/{self._pretrained}"
            )
            return self._model_id
        # Try common weight filenames to locate the cached snapshot.
        for filename in _WEIGHT_FILENAMES:
            cached = try_to_load_from_cache(hf_hub, filename)
            if cached and isinstance(cached, str):
                commit = Path(cached).parent.name
                self._model_id = f"{hf_hub}@{commit}"
                return self._model_id
        self._model_id = hf_hub
        return self._model_id

    def embed_dim(self) -> int:
        """Embedding dimensionality for this model architecture."""
        open_clip = _import_open_clip(use_transformers=False)
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
