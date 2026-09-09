"""Integration tests that load the CLIP model.

These are slow on first run (model download ~350MB).
"""

import os
import re
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from grape.search import find_images, score_image, score_images

pytestmark = pytest.mark.slow


# --- embedding properties ---

def test_encode_image_normalized(clip_model, fixtures_dir):
    emb = clip_model.encode_image(str(fixtures_dir / "dog.jpg"))
    norm = np.linalg.norm(emb, axis=-1)
    np.testing.assert_allclose(norm, 1.0, atol=1e-5)


def test_encode_texts_normalized(clip_model):
    emb = clip_model.encode_texts(["hello"])
    norm = np.linalg.norm(emb, axis=-1)
    np.testing.assert_allclose(norm, 1.0, atol=1e-5)


def test_different_images_produce_different_embeddings(clip_model, fixtures_dir):
    dog_emb = clip_model.encode_image(str(fixtures_dir / "dog.jpg"))
    beach_emb = clip_model.encode_image(str(fixtures_dir / "beach.jpg"))
    sim = float((dog_emb @ beach_emb.T)[0, 0])
    assert sim < 0.95


# --- semantic matching ---

# (query, image that should rank FIRST) -- things a human would get right.
POSITIVE_MATCHES = [
    ("dog", "dog.jpg"),
    ("cat", "cat.jpg"),
    ("beach", "beach.jpg"),
    ("raspberries", "food.jpg"),
    ("city", "city.jpg"),
    ("forest", "forest.jpg"),
    ("woman wearing hat", "lenna.png"),
    ("teapot", "teapot.png"),
    ("handwritten digits", "mnist.png"),
    ("omelette", "omelette.jpg"),
    ("woodworking", "dovetail.jpg"),
    ("blacksmith", "blacksmith.jpg"),
    ("restaurant", "toms_diner.jpg"),
]

# (query, image that should NOT rank first) -- things a human would never confuse.
NEGATIVE_MATCHES = [
    ("dog", "teapot.png"),
    ("beach", "mnist.png"),
    ("teapot", "forest.jpg"),
    ("handwritten digits", "beach.jpg"),
    ("cat", "toms_diner.jpg"),
    ("restaurant", "cat.jpg"),
    ("forest", "omelette.jpg"),
]


def test_semantic_ranking(clip_model, fixtures_dir):
    """The model should rank the right image first for most queries."""
    images = find_images(str(fixtures_dir), recursive=False)
    checks_to_run = POSITIVE_MATCHES + NEGATIVE_MATCHES

    # Batch once: encode images and query texts up front so we avoid
    # repeatedly re-encoding the same images for every keyword.
    image_embs = np.vstack([clip_model.encode_image(str(p)) for p in images])
    query_texts = [query for query, _ in checks_to_run]
    query_embs = clip_model.encode_texts(query_texts)
    similarity = image_embs @ query_embs.T

    correct = 0
    details = []
    checks = []

    for i, (keyword, expected_name) in enumerate(POSITIVE_MATCHES):
        top_idx = int(np.argmax(similarity[:, i]))
        top_name = os.path.basename(images[top_idx])
        ok = top_name == expected_name
        correct += ok
        checks.append(ok)
        details.append(f"  {'ok' if ok else 'MISS':4s}  +{keyword:25s} -> {top_name}"
                        + (f" (expected {expected_name})" if not ok else ""))

    offset = len(POSITIVE_MATCHES)
    for i, (keyword, wrong_name) in enumerate(NEGATIVE_MATCHES, start=offset):
        top_idx = int(np.argmax(similarity[:, i]))
        top_name = os.path.basename(images[top_idx])
        ok = top_name != wrong_name
        correct += ok
        checks.append(ok)
        details.append(f"  {'ok' if ok else 'MISS':4s}  -{keyword:25s} != {wrong_name}"
                        + (f" (but got {top_name})" if not ok else ""))

    summary = "\n".join(details)
    threshold = len(checks) * 3 // 4  # 75%
    assert correct >= threshold, (
        f"Only {correct}/{len(checks)} correct"
        f" (need {threshold}):\n{summary}"
    )


# --- score_image / score_images ---

def test_score_image_mean_matches_keyword_average(clip_model, fixtures_dir):
    result = score_image(clip_model, fixtures_dir / "dog.jpg", ["dog", "cat", "car"])
    scores = list(result.scores.values())
    np.testing.assert_allclose(result.score, np.mean(scores), atol=1e-7)


def test_score_images_skips_unreadable(clip_model, tmp_path):
    """Corrupt files should be skipped, not crash the batch."""
    good = tmp_path / "good.jpg"
    bad = tmp_path / "corrupt.jpg"

    src = Path(__file__).parent / "fixtures" / "dog.jpg"
    shutil.copy(src, good)
    bad.write_bytes(b"not a real image at all")

    results = score_images(clip_model, [good, bad], ["dog"], quiet=True)
    assert len(results) == 1
    assert results[0].path == str(good)


# --- multi-frame (animated GIF/WEBP) embedding ---

def _save_animated_gif(path, frame_colors, size=(64, 64)):
    """Write an animated GIF with one solid-colour frame per entry."""
    frames = [Image.new("RGB", size, c) for c in frame_colors]
    frames[0].save(
        path, format="GIF", save_all=True, append_images=frames[1:],
        duration=100, loop=0,
    )


def test_single_frame_embedding_unchanged_by_multi_frame_path(
    clip_model, tmp_path,
):
    """Static images take the original code path: byte-identical to a
    direct single-frame encode. This is what keeps existing cache rows
    valid across the multi-frame upgrade."""
    img_path = tmp_path / "static.png"
    Image.new("RGB", (64, 64), (180, 60, 60)).save(img_path, format="PNG")
    emb = clip_model.encode_image(str(img_path))
    assert emb.shape == (1, clip_model.embed_dim())
    np.testing.assert_allclose(np.linalg.norm(emb, axis=-1), 1.0, atol=1e-5)


def test_animated_gif_uses_multi_frame_mean(clip_model, tmp_path):
    """An animated GIF whose frames differ visibly should produce a
    different embedding from any single one of its frames -- the mean
    actually combines multi-frame content (CLIP4Clip arXiv:2104.08860)."""
    gif_path = tmp_path / "animated.gif"
    _save_animated_gif(gif_path, [
        (255, 0, 0), (255, 128, 0), (255, 255, 0),
        (0, 255, 0), (0, 255, 255), (0, 0, 255),
        (128, 0, 255), (255, 0, 255),
    ])
    multi_emb = clip_model.encode_image(str(gif_path))
    assert multi_emb.shape == (1, clip_model.embed_dim())
    np.testing.assert_allclose(
        np.linalg.norm(multi_emb, axis=-1), 1.0, atol=1e-5,
    )

    # First-frame-only encoding should differ from the multi-frame mean.
    first_only = tmp_path / "first.png"
    Image.new("RGB", (64, 64), (255, 0, 0)).save(first_only, format="PNG")
    first_emb = clip_model.encode_image(str(first_only))
    sim = float((multi_emb @ first_emb.T)[0, 0])
    assert sim < 0.99, (
        f"multi-frame embedding too close to first-frame only (sim={sim});"
        " mean-pool may not be sampling later frames"
    )


@pytest.mark.parametrize("n_frames", [3, 8, 20])
def test_multi_frame_mean_matches_manual_computation(
    clip_model, tmp_path, n_frames,
):
    """The animated-image embedding equals a manually-computed mean
    over the expected uniformly-sampled frames -- covers N<K, N==K, N>K."""
    from grape.model import MAX_ANIMATION_FRAMES
    colors = [
        (i * 13 % 256, (i * 29) % 256, (i * 47) % 256)
        for i in range(n_frames)
    ]
    gif_path = tmp_path / f"{n_frames}f.gif"
    _save_animated_gif(gif_path, colors)
    multi_emb = clip_model.encode_image(str(gif_path))

    k = min(n_frames, MAX_ANIMATION_FRAMES)
    indices = [round(i * (n_frames - 1) / (k - 1)) for i in range(k)]

    frame_embs = []
    for j, idx in enumerate(indices):
        p = tmp_path / f"f{j}.png"
        Image.new("RGB", (64, 64), colors[idx]).save(p, format="PNG")
        frame_embs.append(clip_model.encode_image(str(p)))
    stacked = np.vstack(frame_embs)
    expected = stacked.mean(axis=0, keepdims=True)
    expected = expected / np.linalg.norm(expected, axis=-1, keepdims=True)
    np.testing.assert_allclose(multi_emb, expected, atol=1e-5)


def _save_gif(path, frames):
    """Write frames (PIL images) as one animated GIF."""
    frames[0].save(
        path, format="GIF", save_all=True, append_images=frames[1:],
        duration=100, loop=0,
    )


def _panning_frames(src, n, size=96):
    """N frames of a single scene: a slow pan + brightness jitter so the
    frames differ (real animation) while still showing the same subject.
    Stands in for a short clip whose frames should match the clip."""
    from PIL import ImageEnhance
    base = src.convert("RGB").resize((size + 4 * n, size + 4 * n))
    frames = []
    for i in range(n):
        crop = base.crop((4 * i, 4 * i, 4 * i + size, 4 * i + size))
        frames.append(ImageEnhance.Brightness(crop).enhance(0.92 + 0.04 * (i % 3)))
    return frames


def test_identical_frames_do_not_change_embedding(clip_model, tmp_path):
    """A GIF whose frames are all identical must encode to the same vector
    as a single-frame encode of that frame. Mean-pooling K copies of one
    embedding is a no-op up to float rounding; if this drifts, the
    animation path is corrupting otherwise-static content."""
    src = Image.new("RGB", (96, 96))
    # A non-trivial image (gradient) so the embedding isn't degenerate.
    px = src.load()
    for y in range(96):
        for x in range(96):
            px[x, y] = (x * 2 % 256, y * 2 % 256, (x + y) % 256)
    many = tmp_path / "identical.gif"
    _save_gif(many, [src] * 8)
    multi_emb = clip_model.encode_image(str(many))

    # Reference: encode exactly frame 0 of the same GIF as a lone image,
    # so the only difference is the averaging arithmetic (pixels match).
    with Image.open(many) as g:
        g.seek(0)
        frame0 = g.convert("RGB")
    frame0_path = tmp_path / "frame0.png"
    frame0.save(frame0_path, format="PNG")
    single_emb = clip_model.encode_image(str(frame0_path))

    np.testing.assert_allclose(multi_emb, single_emb, atol=1e-5)


def test_gif_frames_match_their_own_gif(clip_model, fixtures_dir, tmp_path):
    """The 'frames of GIFs match GIFs' property: a held-out frame of a
    scene must be more similar to a GIF built from that scene than to a
    GIF of an unrelated scene. Guards against animation mean-pooling
    washing content out so far that GIFs mostly just match each other."""
    cat = Image.open(fixtures_dir / "cat.jpg")
    beach = Image.open(fixtures_dir / "beach.jpg")
    cat_gif = tmp_path / "cat.gif"
    beach_gif = tmp_path / "beach.gif"
    _save_gif(cat_gif, _panning_frames(cat, 8))
    _save_gif(beach_gif, _panning_frames(beach, 8))

    query_path = tmp_path / "cat_query.png"
    cat.convert("RGB").resize((96, 96)).save(query_path, format="PNG")
    q = clip_model.encode_image(str(query_path))
    cat_emb = clip_model.encode_image(str(cat_gif))
    beach_emb = clip_model.encode_image(str(beach_gif))

    sim_own = float((q @ cat_emb.T)[0, 0])
    sim_other = float((q @ beach_emb.T)[0, 0])
    assert sim_own > sim_other, (
        f"a cat frame matched the beach GIF ({sim_other:.3f}) over its own "
        f"cat GIF ({sim_own:.3f}); animation pooling is washing out content"
    )


# --- model metadata ---


def test_model_id_format(clip_model):
    """model_id should be '{repo}@{40-char-hex}'."""
    mid = clip_model.model_id()
    assert re.fullmatch(r".+@[0-9a-f]{40}", mid), f"unexpected model_id: {mid}"


def test_embed_dim(clip_model):
    assert clip_model.embed_dim() == 512
