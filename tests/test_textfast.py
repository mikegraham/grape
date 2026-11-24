"""grape.textfast: SigLIP 2's text tower in NumPy, without torch.

These need no downloads; test_model.py checks a real model.
"""

import subprocess
import sys
import textwrap

import numpy as np
import pytest

from grape import textfast

# A small SigLIP 2-style text tower: the shape keys of a real SigLIP 2
# config, with small sizes.
TINY_TEXT_CFG = {
    "context_length": 8, "vocab_size": 40, "width": 32, "heads": 4,
    "layers": 2, "no_causal_mask": True, "proj_bias": True,
    "pool_type": "last", "norm_kwargs": {"eps": 1e-6},
    "act_kwargs": {"approximate": "tanh"},
}


def _save_tiny_checkpoint(path):
    """Build the tower with open_clip (random weights), save it as an
    open_clip checkpoint would hold it, and return it."""
    import torch
    from open_clip.model import _build_text_tower
    from safetensors.torch import save_file

    tower = _build_text_tower(16, TINY_TEXT_CFG).eval()
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for param in tower.parameters():  # biases start at zero otherwise
            param.copy_(torch.randn(param.shape, generator=generator) * 0.2)
    state = {f"text.{k}": v.contiguous() for k, v in tower.state_dict().items()}
    save_file(state, str(path))
    return tower


def test_matches_torch_on_a_random_tower(tmp_path):
    import torch

    path = tmp_path / "open_clip_model.safetensors"
    tower = _save_tiny_checkpoint(path)
    ids = np.random.default_rng(0).integers(0, 40, size=(3, 8))
    with torch.no_grad():
        want = tower(torch.from_numpy(ids)).numpy()
    want /= np.linalg.norm(want, axis=-1, keepdims=True)
    encoder = textfast.TextEncoder(str(path), tokenizer=None, text_cfg=TINY_TEXT_CFG)
    got = encoder.encode_ids(ids)
    assert got.dtype == np.float32
    assert (got * want).sum(-1).min() > 0.9999


def test_supports_every_siglip2_model_and_nothing_else():
    """Every SigLIP 2 config open_clip ships has the text tower textfast
    runs; CLIP's and the first SigLIP's don't."""
    import open_clip

    from grape.openclip_files import builtin_model_config

    models = open_clip.list_models()
    siglip2 = [m for m in models if "SigLIP2" in m and "naflex" not in m]
    assert siglip2
    for name in siglip2:
        assert textfast.supports(builtin_model_config(name)["text_cfg"]), name
    for name in ("ViT-B-32", "ViT-B-16-SigLIP"):
        assert not textfast.supports(builtin_model_config(name)["text_cfg"]), name


def test_cli_encodes_keywords_without_importing_torch(tmp_path):
    """With a text encoder available, a keyword search never imports torch."""
    _save_tiny_checkpoint(tmp_path / "open_clip_model.safetensors")
    script = textwrap.dedent(f"""
        import sys

        class NoTorch:
            def find_spec(self, name, path=None, target=None):
                if name == "torch" or name.startswith("torch."):
                    raise ImportError("torch imported")

        sys.meta_path.insert(0, NoTorch())
        import numpy as np

        import grape.cli as cli
        from grape import textfast

        def tokenize(texts):
            return np.zeros((len(texts), 8), dtype=np.int64)

        def text_encoder(model_name, model_id):
            path = {str(tmp_path / "open_clip_model.safetensors")!r}
            return textfast.TextEncoder(path, tokenize, {TINY_TEXT_CFG!r})

        textfast.text_encoder = text_encoder

        class Cache:
            def get_model_id(self, model_name, pretrained):
                return "org/tiny@abc"

        model = cli._LazyModel("tiny", "test", quiet=True, cache=Cache())
        emb = model.encode_texts(["a photo of a dog", "sunset"])
        print(emb.shape, model.model_id(), "torch" in sys.modules)
    """)
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True,
    )
    assert out.stdout.split() == ["(2,", "16)", "org/tiny@abc", "False"]


@pytest.mark.parametrize("model_id", ["org/tiny", "org/missing@abc"])
def test_no_encoder_without_a_cached_checkpoint(model_id):
    """No commit in the id, or no such snapshot: torch it is."""
    assert textfast.text_encoder("ViT-B-32-SigLIP2-256", model_id) is None
