import importlib
import os
import sys
import types
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

# Default to offline HF Hub access in this process to avoid unexpected
# network latency during model startup. This only applies when the caller
# did not already set `HF_HUB_OFFLINE` in the environment.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

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
    return _cached_weight_path(model_name, pretrained) is not None


def _cached_weight_path(model_name: str, pretrained: str) -> str | None:
    """Return a local cached checkpoint path for this model when available."""
    open_clip = _import_open_clip(use_transformers=False)
    cfg = open_clip.get_pretrained_cfg(model_name, pretrained) or {}
    hf_hub = str(cfg.get("hf_hub", "")).rstrip("/")
    if not hf_hub:
        return None
    for filename in _WEIGHT_FILENAMES:
        cached = _cached_file_from_repo(hf_hub, filename)
        if cached is not None:
            return cached
    return None


def _temporary_hf_hub_offline():
    """Temporarily force huggingface_hub into offline mode."""
    return _temporary_env("HF_HUB_OFFLINE", "1")


def _hf_cache_root() -> Path:
    """Return the huggingface hub cache root directory."""
    hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hub_cache:
        return Path(hub_cache)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _cached_file_from_repo(repo_id: str, filename: str) -> str | None:
    """Return cached file path for ``repo_id/filename`` if present."""
    # HACK: This probes Hugging Face hub cache internals directly for speed.
    # Breaks if HF cache directory layout changes (models--*/snapshots/refs),
    # in which case we return None and callers fall back to normal tag-based
    # resolution/download through open_clip.
    repo_dir = _hf_cache_root() / f"models--{repo_id.replace('/', '--')}"
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return None

    ref_main = repo_dir / "refs" / "main"
    if ref_main.is_file():
        commit = ref_main.read_text(encoding="utf-8").strip()
        if commit:
            candidate = snapshots_dir / commit / filename
            if candidate.is_file():
                return str(candidate)

    newest: Path | None = None
    newest_mtime = -1
    for snapshot in snapshots_dir.iterdir():
        candidate = snapshot / filename
        if not candidate.is_file():
            continue
        mtime = candidate.stat().st_mtime_ns
        if mtime > newest_mtime:
            newest = candidate
            newest_mtime = mtime
    if newest is None:
        return None
    return str(newest)


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


@contextmanager
def _temporary_import_stubs(*, stub_timm: bool = True):
    """Stub out heavy modules that open_clip imports eagerly but we don't need.

    HACK: We inject stub modules for ``transformers``, ``open_clip.coca_model``,
    and optionally ``open_clip.timm_model`` so that ``import open_clip`` skips
    the CoCa class (~1.7s), the transformers stack, and timm (~0.25s).
    Plain CLIP/ViT models never touch these code paths.
    Breaks when the chosen model genuinely needs CoCa, a HF text tower, or
    a timm vision backbone. Callers detect that failure and retry with a full
    open_clip import.
    """
    saved: dict[str, Any] = {}
    stub_prefixes = ["transformers", "open_clip.coca_model"]
    if stub_timm:
        stub_prefixes.append("open_clip.timm_model")
    for key in list(sys.modules):
        if any(key == p or key.startswith(p + ".") for p in stub_prefixes):
            saved[key] = sys.modules.pop(key)
    sys.modules["transformers"] = _make_stub(
        "transformers",
        "grape startup stub: blocks heavy transformers import",
    )
    sys.modules["open_clip.coca_model"] = _make_stub(
        "open_clip.coca_model",
        "grape startup stub: CoCa model class not needed for plain CLIP",
        CoCa="STUBBED by grape/model.py: CoCa not needed for this model",
    )
    if stub_timm:
        sys.modules["open_clip.timm_model"] = _make_stub(
            "open_clip.timm_model",
            "grape startup stub: timm vision backbone not needed"
            " for this model architecture",
            TimmModel="STUBBED by grape/model.py: timm not needed"
            " for this model",
        )
    try:
        yield
    finally:
        for key in list(sys.modules):
            if any(key == p or key.startswith(p + ".") for p in stub_prefixes):
                del sys.modules[key]
        sys.modules.update(saved)


def _import_open_clip(
    *,
    use_transformers: bool,
    model_name: str | None = None,
):
    """Import open_clip, optionally skipping transformers-heavy code paths."""
    global _open_clip_fast_path, _open_clip_module
    if _open_clip_module is not None:
        if use_transformers and _open_clip_fast_path:
            # HACK: If we previously imported via the fast-path stub and now
            # need real transformers, we must purge/re-import open_clip.
            # If open_clip import side effects change, this could fail and the
            # model load will raise instead of silently using wrong modules.
            # Iterate over list(...) because we mutate sys.modules during loop.
            for name in list(sys.modules):
                if name == "open_clip" or name.startswith("open_clip."):
                    del sys.modules[name]
            _open_clip_module = None
        else:
            return _open_clip_module

    if use_transformers:
        _open_clip_module = importlib.import_module("open_clip")
        _open_clip_fast_path = False
        return _open_clip_module

    # Only stub timm when we know the model doesn't need it.
    stub_timm = model_name is not None and not _model_needs_timm(model_name)
    with _temporary_import_stubs(stub_timm=stub_timm):
        _open_clip_module = importlib.import_module("open_clip")
    _open_clip_fast_path = True
    return _open_clip_module


def _requires_transformers(exc: Exception) -> bool:
    """Return True when open_clip failed due to missing transformers."""
    text = str(exc).lower()
    return "transformers" in text and "install" in text


def resolve_model_id(
    model_name: str,
    pretrained: str,
) -> str:
    """Resolve a stable model id without loading model weights."""
    try:
        open_clip = _import_open_clip(use_transformers=False)
    except (ImportError, AttributeError) as exc:
        if not _requires_transformers(exc):
            raise
        open_clip = _import_open_clip(use_transformers=True)
    cfg = open_clip.get_pretrained_cfg(model_name, pretrained)
    hf_hub: str = (cfg or {}).get("hf_hub", "").rstrip("/")
    if not hf_hub:
        return f"{model_name}/{pretrained}"
    for filename in _WEIGHT_FILENAMES:
        cached = _cached_file_from_repo(hf_hub, filename)
        if cached:
            commit = Path(cached).parent.name
            return f"{hf_hub}@{commit}"
    return hf_hub


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
        cached_weight_path = _cached_weight_path(model_name, pretrained)
        # HACK: Prefer concrete local checkpoint path over pretrained tag to
        # skip hub/tag resolution and avoid unnecessary network/cache logic.
        # If local cache probing fails or path disappears, we fall back to the
        # original tag and let open_clip resolve/download normally.
        pretrained_ref = cached_weight_path or pretrained
        if not quiet:
            print("Loading model...", end=" ", flush=True, file=sys.stderr)
        if cached_weight_path is not None:
            # HACK: When a concrete local weight path is selected, force
            # offline hub behavior so open_clip does not attempt network
            # resolution. This will fail fast if that local file is missing
            # or stale instead of silently fetching a different artifact.
            offline_ctx = _temporary_hf_hub_offline()
        else:
            offline_ctx = nullcontext()
        try:
            with offline_ctx:
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    model_name, pretrained=pretrained_ref, device=self.device
                )
            self.tokenizer = open_clip.get_tokenizer(model_name)
        except (ImportError, AttributeError) as exc:
            if not _requires_transformers(exc):
                raise
            open_clip = _import_open_clip(use_transformers=True)
            with offline_ctx:
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    model_name, pretrained=pretrained_ref, device=self.device
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
        self._model_id = resolve_model_id(
            self._model_name,
            self._pretrained,
        )
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
