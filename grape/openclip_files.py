"""open_clip's model configs and tokenizer files, without importing torch.

Importing open_clip imports torch (seconds). Both grape.model and the
torch-free text encoder (grape.textfast) need to read a model's config
and tokenize text, so that lives here.
"""

from __future__ import annotations

import hashlib
import html
import importlib.util
import json
import logging
import string
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from grape.hf_cache import cached_file_from_repo

log = logging.getLogger("grape")


def builtin_model_config(model_name: str) -> dict | None:
    """open_clip's built-in JSON config for *model_name*, or None."""
    open_clip = sys.modules.get("open_clip")
    if open_clip is not None and getattr(open_clip, "__file__", None) is not None:
        package_dir = Path(open_clip.__file__).parent  # type: ignore[arg-type]
    else:
        spec = importlib.util.find_spec("open_clip")
        if spec is None or spec.origin is None:
            return None
        package_dir = Path(spec.origin).parent
    cfg_file = package_dir / "model_configs" / f"{model_name}.json"
    if not cfg_file.is_file():
        return None
    # JSON is UTF-8 by spec; the locale default is cp1252 on Windows.
    cfg: dict = json.loads(cfg_file.read_text(encoding="utf-8"))
    return cfg


def fast_tokenizer_dir(text_cfg: dict) -> Path | None:
    """Cached snapshot dir, if _FastHFTokenizer can stand in for HFTokenizer."""
    if text_cfg.get("hf_model_name") or text_cfg.get("tokenizer_mode"):
        return None
    if set(text_cfg.get("tokenizer_kwargs", {})) - {"clean"}:
        return None
    repo = text_cfg["hf_tokenizer_name"]
    config = cached_file_from_repo(repo, "tokenizer_config.json")
    if config is None or cached_file_from_repo(repo, "tokenizer.json") is None:
        return None
    return Path(config).parent


# tokenizer.json SHA-256 -> an equivalent SentencePiece model: (repo, file,
# SHA-256 of that file). tokenizers parses a 256k-word tokenizer.json in
# ~2 s, holding the GIL; SentencePiece loads the same vocabulary in ~0.1 s
# without it. Keyed by contents, so a changed tokenizer.json quietly falls
# back to the JSON. test_sentencepiece_tokenizer_matches_json checks each.
SENTENCEPIECE_EQUIVALENTS = {
    # Gemma's tokenizer, shared by every SigLIP 2 model. timm's repos
    # ship only tokenizer.json; Google's ship tokenizer.model too.
    "220c63d496e0c14e63eb656c91e0215e926202e4c74b1f089e09f1920d779b04": (
        "google/siglip2-base-patch16-256",
        "tokenizer.model",
        "61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def sentencepiece_equivalent(tokenizer_json: Path) -> Path | None:
    """A verified SentencePiece model equivalent to *tokenizer_json*, or None.

    Downloaded once (4 MB) into the Hugging Face cache; offline, the
    tokenizer.json is used instead.
    """
    entry = SENTENCEPIECE_EQUIVALENTS.get(sha256(tokenizer_json))
    if entry is None:
        return None
    repo, filename, expected = entry
    path = cached_file_from_repo(repo, filename)
    if path is None:
        from huggingface_hub import hf_hub_download
        try:
            path = hf_hub_download(repo, filename)
        except OSError as exc:
            log.debug("no %s from %s (%s); using tokenizer.json", filename, repo, exc)
            return None
    if sha256(Path(path)) != expected:
        log.debug("%s from %s has changed; using tokenizer.json", filename, repo)
        return None
    return Path(path)


_PUNCTUATION = str.maketrans("", "", string.punctuation)


def canonicalize(text: str) -> str:
    """open_clip's "canonicalize" text cleaning, which SigLIP 2 uses."""
    import ftfy

    text = html.unescape(html.unescape(ftfy.fix_text(text))).strip()
    text = text.replace("_", " ").translate(_PUNCTUATION).lower()
    return " ".join(text.split())


class SentencePieceTokenizer:
    """The token ids open_clip's HFTokenizer gives, from a SentencePiece
    model equivalent to its tokenizer.json (see sentencepiece_equivalent).

    For tokenizers that clean text with "canonicalize", append the EOS
    token and pad on the right, as SigLIP 2's (Gemma's) does.
    """

    def __init__(self, model_file: Path, snapshot: Path, context_length: int) -> None:
        import sentencepiece

        config = json.loads(  # UTF-8 by spec, not the locale's encoding
            (snapshot / "tokenizer_config.json").read_text(encoding="utf-8")
        )
        pad = config["pad_token"]
        pad = pad["content"] if isinstance(pad, dict) else pad
        self._sp = sentencepiece.SentencePieceProcessor(model_file=str(model_file))
        self._length = context_length
        self._pad: int = self._sp.piece_to_id(pad)

    def __call__(self, texts: list[str]) -> NDArray[np.int64]:
        rows = []
        for text in texts:
            ids = self._sp.encode(canonicalize(text))[: self._length - 1]
            ids.append(self._sp.eos_id())
            ids.extend([self._pad] * (self._length - len(ids)))
            rows.append(ids)
        return np.array(rows, dtype=np.int64)


def sentencepiece_tokenizer(text_cfg: dict) -> SentencePieceTokenizer | None:
    """The SentencePiece tokenizer for a model's text config, if it has one
    and its files are cached (or downloadable)."""
    if "hf_tokenizer_name" not in text_cfg:
        return None
    if text_cfg.get("tokenizer_kwargs", {}).get("clean") != "canonicalize":
        return None
    snapshot = fast_tokenizer_dir(text_cfg)
    if snapshot is None:
        return None
    model_file = sentencepiece_equivalent(snapshot / "tokenizer.json")
    if model_file is None:
        return None
    return SentencePieceTokenizer(model_file, snapshot, text_cfg["context_length"])
