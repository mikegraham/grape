# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.1] - 2026-11-24

### Fixed
- With open_clip 4, loading the model took ~3 s longer than it should: grape's startup shortcut for importing open_clip failed and fell back to a full import.

## [0.6.0] - 2026-11-20

### Added
- Windows support, tested in CI alongside Linux and macOS.
- Python 3.14 support.

### Changed
- Faster first runs: big JPEGs are decoded at reduced size (never below twice the model's input size), 1.2-1.7x faster overall. Scores for JPEGs can move by up to ~0.004; images already in the cache keep their embeddings.
- On a GPU, model weights are kept in fp16: half the GPU memory and slightly faster encoding.
- The first search after a reboot is about twice as fast: grape reads the model's weight file while torch imports.
- `--view` is much faster with big result sets: it no longer shows each image's dimensions (reading them opened every file before the window appeared), and pages with more than 500 results don't scroll-snap.
- On Windows, printed paths are no longer shell-quoted.

### Fixed
- On Windows, the default model failed to load (`UnicodeDecodeError` reading its tokenizer config).
- On Windows, a photo found by scanning a folder and the same photo given directly were cached separately and encoded twice.

## [0.5.0] - 2026-11-19

### Added
- HEIC/HEIF photos (iPhone) are searchable, via the new `pillow-heif` dependency.
- `--view` lets you select images: click an image (or its checkbox) to select it, and when you close the window grape prints the selected paths: shell-quoted and space-separated on one line, ready to paste as arguments, or raw and NUL-terminated with `-print0`. Pipe them to `rm`, `cp`, a file, or anything else. `--view` and `-print0` can now be combined.
- `--cached-only`: score only images that already have a cached embedding and skip the rest, so the model is never loaded to encode images. Can't be combined with `--no-cache`.
- `-v` prints every thread's stack trace on Ctrl-C, so a hung run shows where it is stuck.
- GPU acceleration: grape runs on an Intel GPU (XPU) or NVIDIA/AMD GPU (CUDA/ROCm) when the installed PyTorch supports it, with fp16 matmuls and batched encoding. About 16x faster indexing than CPU for the default model on an Intel Xe3 iGPU. `GRAPE_DEVICE=cpu` (or any torch device name) overrides the choice; `-v` logs it. Needs a PyTorch build for your GPU: plain `pip install` gets the CUDA build.
- `scripts/install_xpu.sh`: sets up a venv with PyTorch's Intel XPU build and an editable grape, pinned so pip can't swap in the CUDA build (several GB of NVIDIA wheels). Can install torch offline from a folder of wheels.

### Changed
- Encoding records new embeddings and non-image files in the cache with one commit per batch of 32 instead of one per file, so a first run spends ~0.1 s instead of ~3 s on cache writes per 2,000 new files.
- Encoding no longer copies each still RGB image before preprocessing (`convert("RGB")` on an image that is already RGB copies it: ~40 ms for a 12 MP photo, about 15% of its preprocessing). Same pixels, same embeddings.
- Default `--ensemble-prompts` is now open_clip's seven simple ImageNet templates plus the bare keyword (was three "a photo of ..." variants). About +1.6 ImageNet zero-shot accuracy on SigLIP 2. Keyword embeddings are re-encoded on first use; image embeddings are unaffected.
- The scan no longer opens files: it walks the directories and `stat`s each file, like `find`. Whether a file is an image is decided when it's decoded for encoding; one that fails is skipped and remembered as not an image, so later scans leave it out. Previously every file without an embedding was fully decoded during the scan, on every run, so `--cached-only` and interrupted runs reopened the same photos again and again. The scan also no longer waits for the cache index before starting.
- `-v` log lines start with the date and time, to the millisecond (`2026-11-19 14:03:27.512 grape: ...`), to make slow phases easy to spot.

### Deprecated
- `-t`/`--threshold`. It's hidden from `--help`, still works, and prints a notice when used.

### Fixed
- Normal (non-`-v`) runs no longer print other libraries' INFO messages, such as open_clip 4.x's "Parsing model identifier ..." and Hugging Face's HTTP request lines. grape's own messages and all warnings still show.
- With open_clip 4.x, loading the model no longer prints a misleading "No pretrained weights loaded ... Model initialized randomly" warning. The weights are loaded right after; grape already hid this message for open_clip 3.x, but 4.x logs it under a different logger.
- Works with open_clip 4.x. Its new `mammut_model` imports names from `coca_model` that grape's startup stub doesn't provide, so every run that loaded open_clip failed with `ImportError: cannot import name 'MultimodalCfg'`. grape now falls back to a normal open_clip import when the stubbed one fails (startup is a few seconds slower on those runs).
- A truncated animated GIF/WEBP/APNG that passes the scan no longer crashes the run with Pillow's `IndexError`/`struct.error`; it is skipped and recorded as not an image like other unreadable files.
- `-print0` now composes with `-s`/`-v` like grep's `-Z`: the path is unquoted and NUL-terminated; the score before it and the `-v` breakdown line after it are unchanged. Previously the flag was silently ignored with `-s`/`-v`.
- A missing or unreadable `--like` image now exits with `grape: <path>: <reason>` instead of a traceback.
- `--like` reference embeddings are written to the cache, so a reference outside the searched library no longer reloads the model on every run.
- A refreshed Hugging Face cache (new snapshot commit for the same model) no longer crashes every run with `model_id mismatch`. The cached id is re-checked against the disk on every run; when the new snapshot holds the same weight file the cache rows are renamed, otherwise the library is re-encoded under the new id.

## [0.4.0] - 2026-09-19

### Changed
- Default model is now `ViT-L-16-SigLIP2-256/webli` (was `EVA02-L-14/merged2b_s4b_b131k`): +2.7 zero-shot ImageNet, better retrieval, same encode speed. **First run re-encodes your library.** Download is ~3.5 GB (was ~900 MB).
- Fully cached searches are ~2.5x faster (about 0.9s -> 0.35s at 20k images) and use ~7x less CPU.
- Runs that load the model (new keywords or images) finish ~4 s sooner with the default model.

### Fixed
- Models with a Hugging Face tokenizer (SigLIP, SigLIP 2) now load offline once cached, and no longer contact the Hub on every run.
- No more empty `Encoding: 0it` progress bar when every image is already cached.

## [0.3.0] - 2026-08-03

### Fixed
- Cached embeddings now survive reboots and remounts on anonymous-device filesystems (ecryptfs, overlay, NFS, btrfs, ZFS...). The cache keyed on the filesystem device id, which the kernel renumbers at mount time, so the entire library re-encoded after every reboot. Existing cache rows are reused as-is; no rebuild needed.

## [0.2.0] - 2026-05-12

### Added
- Animated GIF/WEBP/APNG inputs are scored across multiple frames, not just the first. **Cache rows for animated images from 0.1.0 are first-frame-only and won't auto-update.**
- `--view` shows image dimensions next to the filename.
- `--view` snaps Space/PgDn to each result.
- `-v` / `--verbose` prints each newly-encoded file's path to stderr.

### Changed
- CLI output preserves the user's input path style (relative stays relative including `./`; absolute stays absolute).
- Warm-cache invocations are ~20% faster.
- `find_images`, `iter_image_records`, `ScoredImage.path`, `ImageRecord.path` return `str` instead of `pathlib.Path`.

### Fixed
- `find ... | grape -` and `grape -R` write to the same cache row when symlinks are involved (previously fragmented).
- Stdin paths containing NUL bytes are dropped instead of crashing.
- `--cache PATH` is taken as-is; only the default cache auto-creates its parent. Fixes a crash on `--cache grape.db` (bare filename).

## [0.1.0] - 2026-04-25

Initial release.
