"""Encode text without torch, for searches that need only a new keyword.

Importing torch and building the model takes seconds, but a keyword's
embedding needs only the text half. This runs that half in NumPy, reading
its weights straight from the model's own checkpoint in the Hugging Face
cache (memory-mapped: of the 1 GB vocabulary table, only the rows a
keyword uses are read). Same file, same weights as the torch model.

Only SigLIP 2's text tower is supported (the default model's); for other
models, text_encoder returns None and grape uses torch as before.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from numpy.typing import NDArray

from grape import hf_cache
from grape.cpus import cpu_threads
from grape.openclip_files import (
    SentencePieceTokenizer,
    builtin_model_config,
    sentencepiece_tokenizer,
)

# The text towers this runs: SigLIP 2's (pre-norm, bidirectional, last
# token pooled, tanh GELU). A config with anything else (or more) isn't.
_SUPPORTED_TEXT_CFG = {
    "no_causal_mask": True,
    "pool_type": "last",
    "proj_bias": True,
    "act_kwargs": {"approximate": "tanh"},
}
_SIZE_KEYS = {
    "context_length", "vocab_size", "width", "heads", "layers", "mlp_ratio",
    "hf_tokenizer_name", "tokenizer_kwargs", "norm_kwargs",
}


def text_encoder(model_name: str, model_id: str) -> TextEncoder | None:
    """A torch-free encoder for this model's text, or None if there isn't
    one: an unsupported model, or its files aren't cached."""
    cfg = builtin_model_config(model_name)
    if cfg is None:
        return None
    text_cfg = cfg.get("text_cfg", {})
    if not supports(text_cfg):
        return None
    hf_hub, _, commit = model_id.partition("@")
    weights = hf_cache.cached_weight_file(hf_hub, commit) if commit else None
    if weights is None or weights.suffix != ".safetensors":
        return None
    tokenizer = sentencepiece_tokenizer(text_cfg)
    if tokenizer is None:
        return None
    return TextEncoder(str(weights), tokenizer, text_cfg)


def supports(text_cfg: dict) -> bool:
    """Is this the kind of text tower TextEncoder runs?"""
    shape = {k: v for k, v in text_cfg.items() if k not in _SIZE_KEYS}
    return shape == _SUPPORTED_TEXT_CFG


def _open_safetensors(path: str) -> dict[str, NDArray[Any]]:
    """Memory-map a safetensors file: {name: array view}."""
    dtypes = {"F32": np.float32, "F16": np.float16}
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    size = int(raw[:8].view("<u8")[0])
    header = json.loads(bytes(raw[8:8 + size]))
    header.pop("__metadata__", None)
    base = 8 + size
    tensors = {}
    for name, info in header.items():
        if info["dtype"] not in dtypes:
            continue
        start, end = info["data_offsets"]
        data = raw[base + start:base + end].view(dtypes[info["dtype"]])
        tensors[name] = data.reshape(info["shape"])
    return tensors


def _layer_norm(x: NDArray, w: NDArray, b: NDArray, eps: float) -> NDArray:
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    normed: NDArray = (x - mean) / np.sqrt(var + eps) * w + b
    return normed


def _gelu_tanh(x: NDArray) -> NDArray:
    """GELU, tanh approximation, in place to save time on big arrays."""
    inner: NDArray = x * x
    inner *= 0.044715 * 0.7978845608028654
    inner += 0.7978845608028654
    inner *= x
    np.tanh(inner, out=inner)
    inner += 1
    inner *= x
    inner *= 0.5
    return inner


class TextEncoder:
    """SigLIP 2's text tower, in NumPy, from an open_clip checkpoint."""

    def __init__(
        self, weights: str, tokenizer: SentencePieceTokenizer, text_cfg: dict,
    ) -> None:
        self._t = _open_safetensors(weights)
        self._tokenize = tokenizer
        self._heads = text_cfg["heads"]
        self._layers = text_cfg["layers"]
        self._eps = text_cfg.get("norm_kwargs", {}).get("eps", 1e-5)

    def __call__(self, texts: list[str]) -> NDArray[np.float32]:
        """L2-normalized embeddings, shape (len(texts), dim)."""
        from threadpoolctl import threadpool_limits

        # The CLI loads NumPy's BLAS with one thread (see grape.cli).
        with threadpool_limits(limits=cpu_threads(), user_api="blas"):
            return self.encode_ids(self._tokenize(texts))

    def encode_ids(self, ids: NDArray[np.int64]) -> NDArray[np.float32]:
        """Embeddings for token ids, shape (batch, context_length)."""
        t, eps = self._t, self._eps
        x = t["text.token_embedding.weight"][ids].astype(np.float32)
        x += t["text.positional_embedding"]
        batch, length, width = x.shape
        head_dim = width // self._heads
        # A Python float, not np.sqrt: NumPy's float64 would turn every
        # array after it into float64, ~3x slower.
        scale = head_dim ** -0.5
        for i in range(self._layers):
            p = f"text.transformer.resblocks.{i}."
            # Pre-norm transformer block, bidirectional attention. Weights
            # are torch's (out, in), hence the .T (a view, not a copy).
            h = _layer_norm(x, t[p + "ln_1.weight"], t[p + "ln_1.bias"], eps)
            qkv = h @ t[p + "attn.in_proj_weight"].T + t[p + "attn.in_proj_bias"]
            qkv = qkv.reshape(batch, length, 3, self._heads, head_dim)
            q, k, v = np.ascontiguousarray(qkv.transpose(2, 0, 3, 1, 4))
            att = q @ k.transpose(0, 1, 3, 2) * scale
            att = np.exp(att - att.max(-1, keepdims=True))
            att /= att.sum(-1, keepdims=True)
            out = (att @ v).transpose(0, 2, 1, 3).reshape(batch, length, width)
            out = out @ t[p + "attn.out_proj.weight"].T + t[p + "attn.out_proj.bias"]
            x = x + out
            h = _layer_norm(x, t[p + "ln_2.weight"], t[p + "ln_2.bias"], eps)
            h = h @ t[p + "mlp.c_fc.weight"].T + t[p + "mlp.c_fc.bias"]
            h = _gelu_tanh(h)
            x = x + h @ t[p + "mlp.c_proj.weight"].T + t[p + "mlp.c_proj.bias"]
        # SigLIP 2 pools the last token, after the final norm.
        last = x[:, -1]
        w, b = t["text.ln_final.weight"], t["text.ln_final.bias"]
        pooled = _layer_norm(last, w, b, eps)
        proj = t["text.text_projection.weight"].T
        emb = pooled @ proj + t["text.text_projection.bias"]
        norm = np.linalg.norm(emb, axis=-1, keepdims=True)
        result: NDArray[np.float32] = emb / norm
        return result
