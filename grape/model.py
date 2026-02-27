import sys
from pathlib import Path
from typing import Any

import numpy as np
import open_clip
import torch
from huggingface_hub import try_to_load_from_cache
from numpy.typing import NDArray
from PIL import Image

# Default model: ViT-B-32 is the best speed/quality tradeoff on CPU.
# ~150-300ms per image on a modern CPU, 512-dim embeddings.
DEFAULT_MODEL = "ViT-B-32"
DEFAULT_PRETRAINED = "laion2b_s34b_b79k"


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
        if not quiet:
            print("Loading model...", end=" ", flush=True, file=sys.stderr)
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
        for filename in (
            "open_clip_model.safetensors",
            "open_clip_pytorch_model.safetensors",
            "open_clip_pytorch_model.bin",
        ):
            cached = try_to_load_from_cache(hf_hub, filename)
            if cached and isinstance(cached, str):
                commit = Path(cached).parent.name
                self._model_id = f"{hf_hub}@{commit}"
                return self._model_id
        self._model_id = hf_hub
        return self._model_id

    def embed_dim(self) -> int:
        """Embedding dimensionality for this model architecture."""
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
